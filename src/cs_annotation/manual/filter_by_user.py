#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union, Optional


JSONObj = Union[Dict[str, Any], List[Any]]


def iter_jsonl_objects(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yields dict items from a JSONL file where each line can be:
      - a dict (single item)
      - a list of dicts (batch of items)
    """
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: JSONObj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON parse error in {path} at line {i}: {e}")

            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for el in obj:
                    if isinstance(el, dict):
                        yield el
            # ignore anything else


def count_response_entries(responses: Any) -> int:
    """Counts total annotation entries inside `responses` (actual dict entries)."""
    if not isinstance(responses, dict):
        return 0
    total = 0
    for entries in responses.values():
        if isinstance(entries, list):
            total += sum(1 for e in entries if isinstance(e, dict))
    return total


def count_user_entries(responses: Any, target_user_id: str) -> int:
    """Counts entries in `responses` where user_id == target_user_id."""
    if not isinstance(responses, dict):
        return 0
    n = 0
    for entries in responses.values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if isinstance(e, dict) and e.get("user_id") == target_user_id:
                n += 1
    return n


def filter_item_to_user(
    item: Dict[str, Any],
    target_user_id: str,
    drop_empty_response_fields: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Returns a COPY of `item` where `responses` only contains entries for `target_user_id`.
    Returns None if the user has no annotations in the item.
    """
    responses = item.get("responses", {})
    if not isinstance(responses, dict):
        return None

    filtered_responses: Dict[str, List[Dict[str, Any]]] = {}
    found_any = False

    for field_name, entries in responses.items():
        if not isinstance(entries, list):
            continue
        kept_entries = [e for e in entries if isinstance(e, dict) and e.get("user_id") == target_user_id]
        if kept_entries:
            found_any = True
            filtered_responses[field_name] = kept_entries
        elif not drop_empty_response_fields:
            filtered_responses[field_name] = []

    if not found_any:
        return None

    new_item = copy.deepcopy(item)
    new_item["responses"] = filtered_responses
    return new_item


def get_sample_id(item: Dict[str, Any]) -> str:
    md = item.get("metadata")
    if isinstance(md, dict):
        sid = md.get("sample_id")
        if isinstance(sid, str) and sid:
            return sid
    return ""


def write_audit_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_id", "item_id", "status", "total_entries", "user_entries", "reason"]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in header) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Filter JSONL items by annotator user_id in responses; optionally emit an audit report."
    )
    ap.add_argument("-i", "--in", dest="in_path", required=True, help="Input JSONL file")
    ap.add_argument("-o", "--out", dest="out_path", required=True, help="Output JSONL file")
    ap.add_argument("-u", "--user-id", dest="user_id", required=True, help="Annotator user_id to keep")

    ap.add_argument(
        "--keep-empty-response-fields",
        action="store_true",
        help="Keep response fields as [] even if the target user has no entry in that field",
    )
    ap.add_argument(
        "--audit-path",
        default="",
        help="Optional path to write TSV audit (lists empty items + items missing this user).",
    )
    ap.add_argument("--stats", action="store_true", help="Print basic stats + status breakdown")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    # Better emptiness stats: based on actual entries, not responses dict len()
    items_with_any_entries = 0
    items_with_zero_entries = 0

    missing_user = 0
    missing_user_by_status: Dict[str, int] = {}
    zero_entries_by_status: Dict[str, int] = {}

    audit_rows: List[Dict[str, Any]] = []

    out_user_entries = 0
    in_total_entries = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for item in iter_jsonl_objects(in_path):
            total += 1
            status = str(item.get("status") or "")
            item_id = str(item.get("id") or "")
            sample_id = get_sample_id(item)

            responses = item.get("responses", {})
            total_entries = count_response_entries(responses)
            user_entries = count_user_entries(responses, args.user_id)

            in_total_entries += total_entries

            if total_entries > 0:
                items_with_any_entries += 1
            else:
                items_with_zero_entries += 1
                zero_entries_by_status[status] = zero_entries_by_status.get(status, 0) + 1
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "item_id": item_id,
                        "status": status,
                        "total_entries": total_entries,
                        "user_entries": user_entries,
                        "reason": "empty_responses",
                    }
                )
                # don't keep this item (no user entries possible anyway)
                continue

            filtered_item = filter_item_to_user(
                item,
                args.user_id,
                drop_empty_response_fields=not args.keep_empty_response_fields,
            )
            if filtered_item is None:
                missing_user += 1
                missing_user_by_status[status] = missing_user_by_status.get(status, 0) + 1
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "item_id": item_id,
                        "status": status,
                        "total_entries": total_entries,
                        "user_entries": user_entries,
                        "reason": "no_user_entries",
                    }
                )
                continue

            kept += 1
            out_user_entries += count_response_entries(filtered_item.get("responses", {}))
            out_f.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")

    if args.audit_path.strip():
        audit_path = Path(args.audit_path)
        write_audit_tsv(audit_path, audit_rows)

    if args.stats:
        print(f"Read items: {total}")
        print(f"Items with >=1 response entry: {items_with_any_entries}")
        print(f"Items with 0 response entries: {items_with_zero_entries}")
        print(f"Input response entries (all users): {in_total_entries}")
        print(f"Kept items for user_id='{args.user_id}': {kept}")
        print(f"Items missing this user_id (had entries, but none from user): {missing_user}")
        print(f"Output response entries (target user only): {out_user_entries}")
        print(f"Wrote output: {out_path}")
        if args.audit_path.strip():
            print(f"Wrote audit TSV: {args.audit_path}")

        if items_with_zero_entries:
            print("Zero-entry items by status:", dict(sorted(zero_entries_by_status.items())))
        if missing_user:
            print("Missing-user items by status:", dict(sorted(missing_user_by_status.items())))
    else:
        print(f"Wrote {kept}/{total} items for user_id='{args.user_id}' to: {out_path}")


if __name__ == "__main__":
    main()

