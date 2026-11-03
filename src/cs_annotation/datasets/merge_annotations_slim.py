#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MappingRow:
    language: str
    user: str
    user_id: str
    jsonl_path: str


def detect_delimiter(sample_line: str) -> str:
    if "\t" in sample_line:
        return "\t"
    if ";" in sample_line:
        return ";"
    if "," in sample_line:
        return ","
    return "\t"


def normalize_header(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def load_mapping_rows(mapping_path: str) -> List[MappingRow]:
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        first_nonempty = ""
        pos = f.tell()
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            if line.strip():
                first_nonempty = line
                break
        f.seek(pos)

        delim = detect_delimiter(first_nonempty) if first_nonempty else "\t"
        reader = csv.reader(f, delimiter=delim)

        rows: List[MappingRow] = []
        header: Optional[List[str]] = None

        for raw in reader:
            if not raw or all(not c.strip() for c in raw):
                continue

            if header is None:
                maybe = [normalize_header(c) for c in raw]
                # header row?
                if any(h in ("sprache", "language") for h in maybe) and any(
                    h in ("annotator", "user", "user_name") for h in maybe
                ):
                    header = maybe
                    continue
                # else fixed order
                header = ["sprache", "annotator", "user_id", "jsonl"]

            rec = {header[i]: raw[i].strip() if i < len(raw) else "" for i in range(len(header))}
            lang = rec.get("sprache") or rec.get("language") or ""
            user = rec.get("annotator") or rec.get("user") or ""
            user_id = rec.get("user_id") or rec.get("userid") or rec.get("user-id") or ""
            jsonl_path = rec.get("jsonl") or rec.get("path") or ""

            if not (lang and user and user_id and jsonl_path):
                if len(raw) >= 4:
                    lang = lang or raw[0].strip()
                    user = user or raw[1].strip()
                    user_id = user_id or raw[2].strip()
                    jsonl_path = jsonl_path or raw[3].strip()

            if not (lang and user and user_id and jsonl_path):
                raise ValueError(
                    f"Invalid mapping row (need 4 fields). Parsed: "
                    f"language={lang!r}, user={user!r}, user_id={user_id!r}, jsonl={jsonl_path!r}"
                )

            rows.append(MappingRow(language=lang, user=user, user_id=user_id, jsonl_path=jsonl_path))

    return rows


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} line {ln}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def _validate_entries_user_ids(
    entries: List[Any],
    expected_user_id: str,
    field: str,
    source_file: str,
    line_hint: Optional[str],
    warn_only: bool,
) -> None:
    bad: List[str] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        uid = e.get("user_id")
        if uid is None:
            continue
        if uid != expected_user_id:
            bad.append(uid)
    if bad:
        msg = (
            f"Found unexpected user_id(s) in field '{field}' in file {source_file} "
            f"(expected {expected_user_id}, got {sorted(set(bad))})."
        )
        if line_hint:
            msg += f" Context: {line_hint}"
        if warn_only:
            print(f"[WARN] {msg}", file=sys.stderr)
        else:
            raise ValueError(msg)


def extract_value_single_user(
    responses: Dict[str, Any],
    field: str,
    expected_user_id: str,
    *,
    validate_user_ids: bool,
    warn_only_user_mismatch: bool,
    source_file: str,
    line_hint: Optional[str] = None,
) -> Any:
    """
    Assumption: per-user JSONL contains only one user's annotations.

    Supported input shapes:
      responses[field] == [{"value": ..., "user_id": "..."}]  (list)
      responses[field] == {"value": ..., "user_id": "..."}    (dict)
      responses[field] == <other> (rare) -> returned as-is

    If validate_user_ids=True: raise (or warn) if any user_id != expected_user_id.
    """
    if not responses or field not in responses:
        return None

    entries = responses.get(field)

    if isinstance(entries, dict):
        uid = entries.get("user_id")
        if validate_user_ids and uid is not None and uid != expected_user_id:
            msg = (
                f"Unexpected user_id in field '{field}' in file {source_file} "
                f"(expected {expected_user_id}, got {uid})."
            )
            if line_hint:
                msg += f" Context: {line_hint}"
            if warn_only_user_mismatch:
                print(f"[WARN] {msg}", file=sys.stderr)
            else:
                raise ValueError(msg)
        return entries.get("value")

    if isinstance(entries, list):
        if validate_user_ids:
            _validate_entries_user_ids(
                entries,
                expected_user_id=expected_user_id,
                field=field,
                source_file=source_file,
                line_hint=line_hint,
                warn_only=warn_only_user_mismatch,
            )

        # Prefer exact match if present; otherwise take first (but with validation this should be fine)
        for e in entries:
            if isinstance(e, dict) and e.get("user_id") == expected_user_id:
                return e.get("value")
        if len(entries) >= 1 and isinstance(entries[0], dict):
            return entries[0].get("value")
        return None

    return entries


def normalize_references(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if value == []:
        return []
    if isinstance(value, list):
        out: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("type") or item.get("category") or item.get("tag")
            start = item.get("start") or item.get("begin") or item.get("from") or item.get("offset_start")
            end = item.get("end") or item.get("to") or item.get("offset_end")
            if label is None and start is None and end is None:
                continue
            out.append({"label": label, "start": start, "end": end})
        return out
    if isinstance(value, dict):
        label = value.get("label") or value.get("type") or value.get("category") or value.get("tag")
        start = value.get("start") or value.get("begin") or value.get("from") or value.get("offset_start")
        end = value.get("end") or value.get("to") or value.get("offset_end")
        if label is None and start is None and end is None:
            return []
        return [{"label": label, "start": start, "end": end}]
    return []


def build_slim_record(
    raw: Dict[str, Any],
    mapping: MappingRow,
    *,
    strict_language_match: bool,
    validate_user_ids: bool,
    warn_only_user_mismatch: bool,
    source_file: str,
) -> Dict[str, Any]:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    responses = raw.get("responses") if isinstance(raw.get("responses"), dict) else {}

    if strict_language_match and metadata.get("language") and metadata.get("language") != mapping.language:
        raise ValueError(
            f"Language mismatch for {source_file}: mapping={mapping.language}, item={metadata.get('language')}"
        )

    # Helpful context if we need to show where something went wrong:
    line_hint = None
    if isinstance(metadata, dict):
        sid = metadata.get("sample_id")
        if sid:
            line_hint = f"sample_id={sid}"

    ref_val = extract_value_single_user(
        responses, "references", mapping.user_id,
        validate_user_ids=validate_user_ids,
        warn_only_user_mismatch=warn_only_user_mismatch,
        source_file=source_file,
        line_hint=line_hint,
    )
    refs = normalize_references(ref_val)

    lang = metadata.get("language") or mapping.language

    slim = {
        "user": mapping.user,
        "user_id": mapping.user_id,
        "task": metadata.get("task"),
        "suite": metadata.get("suite"),
        "subset": metadata.get("subset"),
        "language": lang,
        "sample_id": metadata.get("sample_id"),
        "required_knowledge": extract_value_single_user(
            responses, "required_knowledge", mapping.user_id,
            validate_user_ids=validate_user_ids,
            warn_only_user_mismatch=warn_only_user_mismatch,
            source_file=source_file,
            line_hint=line_hint,
        ),
        "time_sensitive": extract_value_single_user(
            responses, "time_sensitive", mapping.user_id,
            validate_user_ids=validate_user_ids,
            warn_only_user_mismatch=warn_only_user_mismatch,
            source_file=source_file,
            line_hint=line_hint,
        ),
        "references": refs,
        "cultural_sensitivity_label": extract_value_single_user(
            responses, "cultural_sensitivity_label", mapping.user_id,
            validate_user_ids=validate_user_ids,
            warn_only_user_mismatch=warn_only_user_mismatch,
            source_file=source_file,
            line_hint=line_hint,
        ),
        "subject_category": extract_value_single_user(
            responses, "subject_category", mapping.user_id,
            validate_user_ids=validate_user_ids,
            warn_only_user_mismatch=warn_only_user_mismatch,
            source_file=source_file,
            line_hint=line_hint,
        ),
        "explanation": extract_value_single_user(
            responses, "explanation", mapping.user_id,
            validate_user_ids=validate_user_ids,
            warn_only_user_mismatch=warn_only_user_mismatch,
            source_file=source_file,
            line_hint=line_hint,
        ),
    }
    return slim


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> int:
    ensure_parent_dir(out_path)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False))
            out.write("\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge per-user annotation JSONL into a slim all-users JSONL (single-user files)."
    )
    ap.add_argument(
        "--mapping",
        required=True,
        help="Path to mapping TSV/CSV with columns: Sprache; Annotator; User_ID; JSONL (delimiter auto-detected).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    ap.add_argument(
        "--strict-language-match",
        action="store_true",
        help="Fail if item.metadata.language differs from mapping language.",
    )
    ap.add_argument(
        "--no-validate-user-ids",
        action="store_true",
        help="Disable validation that responses[*].user_id matches mapping user_id.",
    )
    ap.add_argument(
        "--warn-only-user-mismatch",
        action="store_true",
        help="If validation finds mismatching user_id(s), warn instead of failing.",
    )
    ap.add_argument(
        "--skip-missing-jsonl",
        action="store_true",
        help="Skip mapping rows whose JSONL path does not exist (otherwise error).",
    )
    args = ap.parse_args()

    mapping_rows = load_mapping_rows(args.mapping)
    validate_user_ids = not args.no_validate_user_ids

    def all_records() -> Iterable[Dict[str, Any]]:
        for m in mapping_rows:
            if not os.path.exists(m.jsonl_path):
                msg = f"JSONL file not found: {m.jsonl_path}"
                if args.skip_missing_jsonl:
                    print(f"[WARN] {msg} (skipping)", file=sys.stderr)
                    continue
                raise FileNotFoundError(msg)

            for raw_item in iter_jsonl(m.jsonl_path):
                yield build_slim_record(
                    raw_item,
                    m,
                    strict_language_match=args.strict_language_match,
                    validate_user_ids=validate_user_ids,
                    warn_only_user_mismatch=args.warn_only_user_mismatch,
                    source_file=m.jsonl_path,
                )

    count = write_jsonl(all_records(), args.output)
    print(f"[OK] Wrote {count} records to: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

