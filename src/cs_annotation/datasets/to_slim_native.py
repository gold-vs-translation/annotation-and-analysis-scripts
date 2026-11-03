#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert GPT-5.2-annotated JSONL files (culture_ann nested format)
to the slim single-annotator format used by merge_annotations_slim.py.

Usage:
    python gpt_annotated_to_slim.py \
        --input /path/to/text_*-gpt-5.2-annotated.jsonl [more files...] \
        --output /path/to/output_slim.jsonl \
        --user "GPT-5.2" \
        --user-id "1111"

    # Or process a whole directory:
    python gpt_annotated_to_slim.py \
        --input-dir /path/to/annotated/ \
        --pattern "*-gpt-5.2-annotated.jsonl" \
        --output /path/to/output_slim.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path} line {ln}: {e}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                yield obj


def normalize_references(refs: Any) -> List[Dict[str, Any]]:
    """Keep only label, start, end, span from each reference."""
    if not isinstance(refs, list):
        return []
    out = []
    for item in refs:
        if not isinstance(item, dict):
            continue
        out.append({
            "label": item.get("label"),
            "start": item.get("start"),
            "end": item.get("end"),
            "span": item.get("span"),
        })
    return out


def convert_record(
    raw: Dict[str, Any],
    user: str,
    user_id: str,
    source_file: str,
) -> Optional[Dict[str, Any]]:
    """Convert one GPT-annotated record to slim format."""
    culture_ann = raw.get("culture_ann")
    if not isinstance(culture_ann, dict):
        item_id = raw.get("id", "?")
        print(
            f"[WARN] Missing or invalid culture_ann in {source_file} "
            f"(id={item_id}), skipping.",
            file=sys.stderr,
        )
        return None

    return {
        "user": user,
        "user_id": user_id,
        "task": raw.get("task"),
        "suite": raw.get("suite"),
        "subset": raw.get("subset"),
        "language": raw.get("language"),
        "sample_id": raw.get("id"),
        "required_knowledge": culture_ann.get("required_knowledge"),
        "time_sensitive": culture_ann.get("time_sensitive"),
        "references": normalize_references(culture_ann.get("references")),
        "cultural_sensitivity_label": culture_ann.get("cultural_sensitivity_label"),
        "subject_category": culture_ann.get("subject_category"),
        "explanation": culture_ann.get("explanation"),
    }


def collect_input_files(args) -> List[str]:
    files = []
    if args.input:
        files.extend(args.input)
    if args.input_dir:
        pattern = os.path.join(args.input_dir, args.pattern)
        found = sorted(glob.glob(pattern))
        if not found:
            print(f"[WARN] No files matching {pattern}", file=sys.stderr)
        files.extend(found)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert GPT-5.2-annotated JSONL to slim format."
    )
    ap.add_argument(
        "--input", nargs="*", default=[],
        help="One or more input JSONL files.",
    )
    ap.add_argument(
        "--input-dir",
        help="Directory containing annotated JSONL files.",
    )
    ap.add_argument(
        "--pattern", default="*-gpt-5.2-annotated.jsonl",
        help="Glob pattern for --input-dir (default: *-gpt-5.2-annotated.jsonl).",
    )
    ap.add_argument(
        "--output", required=True,
        help="Output JSONL path.",
    )
    ap.add_argument(
        "--user", default="GPT-5.2",
        help="User name for the annotation records (default: GPT-5.2).",
    )
    ap.add_argument(
        "--user-id", default="1111",
        help="User ID for the annotation records (default: 1111).",
    )
    args = ap.parse_args()

    input_files = collect_input_files(args)
    if not input_files:
        print("[ERROR] No input files specified. Use --input or --input-dir.", file=sys.stderr)
        return 1

    # Check all files exist
    for f in input_files:
        if not os.path.exists(f):
            print(f"[ERROR] File not found: {f}", file=sys.stderr)
            return 1

    # Process
    parent = os.path.dirname(os.path.abspath(args.output))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    n_total = 0
    n_written = 0
    n_skipped = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for fpath in input_files:
            n_file = 0
            for raw in iter_jsonl(fpath):
                n_total += 1
                slim = convert_record(raw, args.user, args.user_id, fpath)
                if slim is None:
                    n_skipped += 1
                    continue
                out.write(json.dumps(slim, ensure_ascii=False))
                out.write("\n")
                n_written += 1
                n_file += 1
            print(f"  {os.path.basename(fpath)}: {n_file} records", file=sys.stderr)

    print(
        f"[OK] Wrote {n_written} records to {args.output} "
        f"(total: {n_total}, skipped: {n_skipped})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


