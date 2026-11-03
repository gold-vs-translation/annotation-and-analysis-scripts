#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
from typing import Any, Dict, Iterable, List, Optional


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


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


def iter_jsonl_xz(path: str) -> Iterable[Dict[str, Any]]:
    with lzma.open(path, "rt", encoding="utf-8") as f:
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


def read_input(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".xz"):
        return iter_jsonl_xz(path)
    return iter_jsonl(path)


def normalize_references(value: Any) -> List[Dict[str, Any]]:
    """
    Expected: list of dicts with start/end/label; we normalize keys and types.
    """
    if value is None:
        return []
    if isinstance(value, list):
        out: List[Dict[str, Any]] = []
        for r in value:
            if not isinstance(r, dict):
                continue
            label = r.get("label") or r.get("type") or r.get("category") or r.get("tag")
            start = r.get("start") if r.get("start") is not None else r.get("begin")
            end = r.get("end") if r.get("end") is not None else r.get("to")

            # Convert numeric strings if possible
            try:
                if isinstance(start, str) and start.isdigit():
                    start = int(start)
                if isinstance(end, str) and end.isdigit():
                    end = int(end)
            except Exception:
                pass

            if label is None and start is None and end is None:
                continue
            out.append({"label": label, "start": start, "end": end})
        return out

    if isinstance(value, dict):
        label = value.get("label") or value.get("type") or value.get("category") or value.get("tag")
        start = value.get("start") if value.get("start") is not None else value.get("begin")
        end = value.get("end") if value.get("end") is not None else value.get("to")
        return [{"label": label, "start": start, "end": end}]

    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert GlobalMMLU culture annotations to slim unified JSONL.")
    ap.add_argument("--input", required=True, help="Path to GlobalMMLU annotation JSONL (optionally .xz).")
    ap.add_argument("--output", required=True, help="Output path (slim JSONL).")
    ap.add_argument("--user", default="GlobalMMLU", help="Dummy user name to write.")
    ap.add_argument("--user-id", default="1111", help="Dummy user_id to write.")
    ap.add_argument(
        "--missing-as-empty-string",
        action="store_true",
        help="Write missing optional fields as '' instead of null.",
    )
    args = ap.parse_args()

    ensure_parent_dir(args.output)

    missing_val: Any = "" if args.missing_as_empty_string else None

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for obj in read_input(args.input):
            # Required top-level fields (per your example)
            suite = obj.get("suite")
            task = obj.get("task")
            subset = obj.get("subset")
            split = obj.get("split")
            language = obj.get("language", "EN")  # in practice always EN
            sample_id = obj.get("id")  # map id -> sample_id

            ann = obj.get("culture_ann_ref") if isinstance(obj.get("culture_ann_ref"), dict) else {}

            required_knowledge = ann.get("required_knowledge_agreement")
            time_sensitive = ann.get("time_sensitive_agreement")
            cultural_sensitivity_label = ann.get("cultural_sensitivity_label_agreement")
            references = normalize_references(ann.get("reference"))

            slim = {
                "user": args.user,
                "user_id": args.user_id,
                "task": task,
                "suite": suite,
                "subset": subset,
                "split": split,  # keep it; your eval-merge may use it (or ignore it like before)
                "language": language,
                "sample_id": sample_id,
                "required_knowledge": required_knowledge,
                "time_sensitive": time_sensitive,
                "references": references,
                "cultural_sensitivity_label": cultural_sensitivity_label,
                "subject_category": missing_val,
                "explanation": missing_val,
            }

            out.write(json.dumps(slim, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] Wrote {n} slim GlobalMMLU records to: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

