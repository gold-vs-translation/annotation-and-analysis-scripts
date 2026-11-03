#!/usr/bin/env python3
"""
Prepare a merged JSONL for meta-evaluation of 500 human-annotated native items
against already existing GPT-5.2 automatic annotations.

Input:
  - Human annotations JSONL (single annotator per item)
  - Directory with GPT-annotated JSONL files

Output:
  - Eval-ready JSONL with top-level item metadata + culture_ann_ref + culture_ann
  - JSON summary report
  - JSONL report for unmatched human items
  - JSONL report for ambiguous / duplicate auto matches

Primary join key:
  (task, language, sample_id_human) == (task, language, id_auto)

Fallbacks:
  - replace human task by human suite
  - compare against auto task or auto suite
  - optional ID normalization (Unicode folding + whitespace cleanup)

The resulting JSONL is intentionally shaped so it can be consumed by the existing
step3_eval_split.py workflow that was used for GlobalMMLU-Lite.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]


def read_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize_json(row), ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_json(obj), f, ensure_ascii=False, indent=2)


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"", "none", "nan", "null"}:
        return ""
    return s


def norm_basic_label(value: Any) -> str:
    s = norm_text(value).lower()
    mapping = {
        "cultural": "cultural",
        "regional": "regional",
        "dialect": "dialect",
        "none": "none",
        "yes": "Yes",
        "no": "No",
        "cs": "CS",
        "ca": "CA",
    }
    return mapping.get(s, norm_text(value))


def folded(s: Any) -> str:
    x = norm_text(s)
    if not x:
        return ""
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = " ".join(x.split())
    return x.casefold()


Key = Tuple[str, str, str]


def make_key(name: Any, language: Any, item_id: Any, normalize_id: bool = False) -> Key:
    name_norm = norm_text(name)
    lang_norm = norm_text(language).upper()
    id_norm = folded(item_id) if normalize_id else norm_text(item_id)
    return (name_norm, lang_norm, id_norm)


def build_auto_indexes(rows: List[JsonDict]) -> Dict[str, Dict[Key, List[JsonDict]]]:
    idx_exact_task: Dict[Key, List[JsonDict]] = defaultdict(list)
    idx_exact_suite: Dict[Key, List[JsonDict]] = defaultdict(list)
    idx_fold_task: Dict[Key, List[JsonDict]] = defaultdict(list)
    idx_fold_suite: Dict[Key, List[JsonDict]] = defaultdict(list)

    for row in rows:
        task = row.get("task")
        suite = row.get("suite")
        lang = row.get("language")
        item_id = row.get("id")
        idx_exact_task[make_key(task, lang, item_id, normalize_id=False)].append(row)
        idx_exact_suite[make_key(suite, lang, item_id, normalize_id=False)].append(row)
        idx_fold_task[make_key(task, lang, item_id, normalize_id=True)].append(row)
        idx_fold_suite[make_key(suite, lang, item_id, normalize_id=True)].append(row)

    return {
        "exact_task": idx_exact_task,
        "exact_suite": idx_exact_suite,
        "fold_task": idx_fold_task,
        "fold_suite": idx_fold_suite,
    }


def first_non_empty(*values: Any) -> str:
    for v in values:
        x = norm_text(v)
        if x:
            return x
    return ""


def build_human_ref(h: JsonDict) -> JsonDict:
    rk = norm_basic_label(h.get("required_knowledge"))
    ts = norm_basic_label(h.get("time_sensitive"))
    csl = norm_basic_label(h.get("cultural_sensitivity_label"))
    culture = norm_text(h.get("culture"))
    region = norm_text(h.get("region"))
    country = norm_text(h.get("country"))

    ref: JsonDict = {
        # original single-annotator fields
        "required_knowledge": rk,
        "time_sensitive": ts,
        "cultural_sensitivity_label": csl,
        "subject_category": norm_text(h.get("subject_category")),
        "references": h.get("references") or [],
        "explanation": norm_text(h.get("explanation")),
        "culture": culture,
        "region": region,
        "country": country,
        # agreement-style aliases so existing eval scripts can consume the file
        "required_knowledge_agreement": rk,
        "time_sensitive_agreement": ts,
        "culture_agreement": culture,
        "region_agreement": region,
        "country_agreement": country,
        "cultural_sensitivity_label_agreement": csl,
        # diagnostics mimicking the agreed GlobalMMLU-Lite schema
        "required_knowledge_vote_counts": {rk: 1} if rk else {},
        "n_annotators_required_knowledge": 1,
        "required_knowledge_majority_share": 1.0 if rk else None,
        "required_knowledge_num_labels": 1 if rk else 0,
        "required_knowledge_tie_type": "",
        "required_knowledge_max_labels": [rk] if rk else [],
        "required_knowledge_resolved_from_tie": False,
        # provenance
        "human_user": norm_text(h.get("user")),
        "human_user_id": norm_text(h.get("user_id")),
        "human_sample_id": norm_text(h.get("sample_id")),
        "human_task": norm_text(h.get("task")),
        "human_suite": norm_text(h.get("suite")),
        "human_subset": norm_text(h.get("subset")),
        "human_language": norm_text(h.get("language")),
    }
    return ref


MATCH_ORDER = [
    ("exact_task", "task", False),
    ("exact_suite", "suite", False),
    ("fold_task", "task", True),
    ("fold_suite", "suite", True),
]


def find_candidates(h: JsonDict, indexes: Dict[str, Dict[Key, List[JsonDict]]]) -> Tuple[str, List[JsonDict], Key]:
    for idx_name, human_name_field, normalize_id in MATCH_ORDER:
        human_name = h.get(human_name_field)
        key = make_key(human_name, h.get("language"), h.get("sample_id"), normalize_id=normalize_id)
        candidates = indexes[idx_name].get(key, [])
        if candidates:
            return idx_name, candidates, key
    return "", [], make_key(h.get("task"), h.get("language"), h.get("sample_id"), normalize_id=False)


def dataset_task_counts(rows: Iterable[JsonDict], task_field: str) -> Dict[str, int]:
    ctr: Counter[str] = Counter()
    for r in rows:
        ctr[first_non_empty(r.get(task_field), r.get("suite"), "UNKNOWN")] += 1
    return dict(sorted(ctr.items()))


def language_counts(rows: Iterable[JsonDict]) -> Dict[str, int]:
    ctr: Counter[str] = Counter()
    for r in rows:
        ctr[norm_text(r.get("language")) or "UNKNOWN"] += 1
    return dict(sorted(ctr.items()))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge single-annotator human labels with existing GPT annotations.")
    ap.add_argument("--human-in", required=True, help="Path to all_user_annotations_*.jsonl")
    ap.add_argument("--annotated-dir", required=True, help="Directory with existing GPT-annotated *.jsonl files")
    ap.add_argument("--out-merged", required=True, help="Eval-ready merged JSONL")
    ap.add_argument("--out-summary", required=True, help="JSON summary report")
    ap.add_argument("--out-unmatched", required=True, help="JSONL report for unmatched human items")
    ap.add_argument("--out-ambiguous", required=True, help="JSONL report for ambiguous / duplicate auto matches")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    human_path = Path(args.human_in)
    annotated_dir = Path(args.annotated_dir)

    human_rows = read_jsonl(human_path)
    auto_files = sorted(annotated_dir.glob("*.jsonl"))
    if not auto_files:
        raise FileNotFoundError(f"No *.jsonl files found in {annotated_dir}")

    auto_rows: List[JsonDict] = []
    file_row_counts: Dict[str, int] = {}
    for fp in auto_files:
        rows = read_jsonl(fp)
        auto_rows.extend(rows)
        file_row_counts[fp.name] = len(rows)

    indexes = build_auto_indexes(auto_rows)

    merged_rows: List[JsonDict] = []
    unmatched_rows: List[JsonDict] = []
    ambiguous_rows: List[JsonDict] = []
    match_method_counts: Counter[str] = Counter()

    for h in human_rows:
        method, candidates, used_key = find_candidates(h, indexes)

        if not candidates:
            unmatched_rows.append(
                {
                    "reason": "no_match",
                    "join_key": {
                        "task": norm_text(h.get("task")),
                        "suite": norm_text(h.get("suite")),
                        "language": norm_text(h.get("language")),
                        "sample_id": norm_text(h.get("sample_id")),
                    },
                    "human_item": h,
                }
            )
            continue

        if len(candidates) > 1:
            ambiguous_rows.append(
                {
                    "reason": "multiple_auto_matches",
                    "match_method": method,
                    "used_key": list(used_key),
                    "join_key": {
                        "task": norm_text(h.get("task")),
                        "suite": norm_text(h.get("suite")),
                        "language": norm_text(h.get("language")),
                        "sample_id": norm_text(h.get("sample_id")),
                    },
                    "human_item": h,
                    "auto_candidates": [
                        {
                            "task": c.get("task"),
                            "suite": c.get("suite"),
                            "language": c.get("language"),
                            "id": c.get("id"),
                            "subset": c.get("subset"),
                            "split": c.get("split"),
                        }
                        for c in candidates
                    ],
                }
            )
            continue

        auto = candidates[0]
        out = dict(auto)
        out["culture_ann_ref"] = build_human_ref(h)
        # preserve auto annotation under the expected field name
        out["culture_ann"] = sanitize_json(auto.get("culture_ann") or {})
        out["merge_info"] = {
            "match_method": method,
            "human_task": norm_text(h.get("task")),
            "human_suite": norm_text(h.get("suite")),
            "human_language": norm_text(h.get("language")),
            "human_sample_id": norm_text(h.get("sample_id")),
            "auto_task": norm_text(auto.get("task")),
            "auto_suite": norm_text(auto.get("suite")),
            "auto_language": norm_text(auto.get("language")),
            "auto_id": norm_text(auto.get("id")),
        }
        merged_rows.append(sanitize_json(out))
        match_method_counts[method] += 1

    summary = {
        "human_input_path": str(human_path),
        "annotated_dir": str(annotated_dir),
        "auto_files": [str(fp) for fp in auto_files],
        "auto_file_row_counts": file_row_counts,
        "n_human_input": len(human_rows),
        "n_auto_rows_loaded": len(auto_rows),
        "n_merged": len(merged_rows),
        "n_unmatched": len(unmatched_rows),
        "n_ambiguous": len(ambiguous_rows),
        "match_method_counts": dict(match_method_counts),
        "human_task_counts": dataset_task_counts(human_rows, "task"),
        "merged_task_counts": dataset_task_counts(merged_rows, "task"),
        "human_language_counts": language_counts(human_rows),
        "merged_language_counts": language_counts(merged_rows),
    }

    write_jsonl(Path(args.out_merged), merged_rows)
    write_jsonl(Path(args.out_unmatched), unmatched_rows)
    write_jsonl(Path(args.out_ambiguous), ambiguous_rows)
    write_json(Path(args.out_summary), summary)

    print(f"Loaded human rows:     {len(human_rows)}")
    print(f"Loaded auto rows:      {len(auto_rows)} from {len(auto_files)} files")
    print(f"Merged rows:           {len(merged_rows)}")
    print(f"Unmatched human rows:  {len(unmatched_rows)}")
    print(f"Ambiguous matches:     {len(ambiguous_rows)}")
    print(f"Match methods:         {dict(match_method_counts)}")
    print(f"Wrote merged JSONL:    {args.out_merged}")
    print(f"Wrote summary JSON:    {args.out_summary}")
    print(f"Wrote unmatched JSONL: {args.out_unmatched}")
    print(f"Wrote ambiguous JSONL: {args.out_ambiguous}")


if __name__ == "__main__":
    main()


