#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert non-annotated JSONL (e.g. Belebele) to slim format with null annotation fields.

Using null (not "none") to clearly distinguish unannotated items from items
annotated as required_knowledge="none" (which means CA).

Usage:
    python unannotated_to_slim_v2.py \
        --input /path/to/text_belebele.jsonl \
        --output /path/to/text_belebele_slim.jsonl \
        --user "Belebele" \
        --user-id "1111"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable


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


def convert_record(raw: Dict[str, Any], user: str, user_id: str) -> Dict[str, Any]:
    return {
        "user": user,
        "user_id": user_id,
        "task": raw.get("task"),
        "suite": raw.get("suite"),
        "subset": raw.get("subset"),
        "language": raw.get("language"),
        "sample_id": raw.get("id"),
        "required_knowledge": None,
        "time_sensitive": None,
        "references": None,
        "cultural_sensitivity_label": None,
        "subject_category": None,
        "explanation": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert non-annotated JSONL to slim format with null annotation fields."
    )
    ap.add_argument("--input", required=True, help="Input JSONL file.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--user", default="Belebele", help="User name (default: Belebele).")
    ap.add_argument("--user-id", default="1111", help="User ID (default: 1111).")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        return 1

    parent = os.path.dirname(os.path.abspath(args.output))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for raw in iter_jsonl(args.input):
            slim = convert_record(raw, args.user, args.user_id)
            out.write(json.dumps(slim, ensure_ascii=False))
            out.write("\n")
            n += 1

    print(f"[OK] Wrote {n} records to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

