#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge slim per-user annotations with LM Eval Harness concatenated sample logs (JSONL.xz).

Default join:
  (suite, task, language, subset, sample_id)

For crosslingual annotation tasks (e.g. MMLU), use language-/suite-independent join:
  (task, subset, sample_id)

Important:
- output keeps annotation-side identifiers under annotation_* fields
- output overwrites suite/task/language/subset/sample_id with RESULT-side values
  so downstream analyses see the actual evaluated language/suite
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set


SPECIAL_CHAR_MAP = {
    "ß": "ss",
    "æ": "ae",
    "Æ": "ae",
    "œ": "oe",
    "Œ": "oe",
    "ø": "o",
    "Ø": "o",
    "đ": "d",
    "Đ": "d",
    "ł": "l",
    "Ł": "l",
    "ð": "d",
    "Ð": "d",
    "þ": "th",
    "Þ": "th",
}


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


def norm_language(lang: Any, mode: str) -> Optional[str]:
    if lang is None:
        return None
    s = str(lang).strip()
    if mode == "upper":
        return s.upper()
    if mode == "lower":
        return s.lower()
    return s


def fold_ascii(s: Any, *, lowercase: bool = True) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return ""

    for k, v in SPECIAL_CHAR_MAP.items():
        s = s.replace(k, v)

    s_nfkd = unicodedata.normalize("NFKD", s)
    s_no_marks = "".join(ch for ch in s_nfkd if not unicodedata.combining(ch))
    if lowercase:
        s_no_marks = s_no_marks.lower()

    out: List[str] = []
    prev_us = False

    for ch in s_no_marks:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif ch == "/":
            if out and out[-1] == "_":
                out.pop()
            out.append("/")
            prev_us = False
        elif ch in ".-:":
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True

    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    slug = slug.replace("/_", "/").replace("_/", "/")
    return slug


def normalize_task(task: Any) -> Optional[str]:
    if task is None:
        return None
    return str(task).strip()


def is_crosslingual_task(task: Any, crosslingual_tasks: Set[str]) -> bool:
    t = normalize_task(task)
    return t in crosslingual_tasks


def mk_default_key(
    suite: Any,
    task: Any,
    language: Any,
    subset: Any,
    sample_id: Any,
    *,
    language_norm: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    return (
        None if suite is None else str(suite).strip(),
        normalize_task(task),
        norm_language(language, language_norm),
        None if subset is None else str(subset).strip(),
        None if sample_id is None else str(sample_id).strip(),
    )


def mk_default_key_folded(
    suite: Any,
    task: Any,
    language: Any,
    subset: Any,
    sample_id: Any,
    *,
    language_norm: str,
    fold_subset: bool,
    fold_sample_id: bool,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    subset2 = fold_ascii(subset) if fold_subset else (None if subset is None else str(subset).strip())
    sid2 = fold_ascii(sample_id) if fold_sample_id else (None if sample_id is None else str(sample_id).strip())
    return (
        None if suite is None else str(suite).strip(),
        normalize_task(task),
        norm_language(language, language_norm),
        subset2,
        sid2,
    )


def mk_crosslingual_key(
    task: Any,
    subset: Any,
    sample_id: Any,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        normalize_task(task),
        None if subset is None else str(subset).strip(),
        None if sample_id is None else str(sample_id).strip(),
    )


def mk_crosslingual_key_folded(
    task: Any,
    subset: Any,
    sample_id: Any,
    *,
    fold_subset: bool,
    fold_sample_id: bool,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    subset2 = fold_ascii(subset) if fold_subset else (None if subset is None else str(subset).strip())
    sid2 = fold_ascii(sample_id) if fold_sample_id else (None if sample_id is None else str(sample_id).strip())
    return (
        normalize_task(task),
        subset2,
        sid2,
    )


def make_key(
    obj: Dict[str, Any],
    *,
    language_norm: str,
    crosslingual_tasks: Set[str],
    folded: bool,
    fold_subset: bool,
    fold_sample_id: bool,
):
    task = obj.get("task")
    if is_crosslingual_task(task, crosslingual_tasks):
        if folded:
            return mk_crosslingual_key_folded(
                task=task,
                subset=obj.get("subset"),
                sample_id=obj.get("sample_id"),
                fold_subset=fold_subset,
                fold_sample_id=fold_sample_id,
            )
        return mk_crosslingual_key(
            task=task,
            subset=obj.get("subset"),
            sample_id=obj.get("sample_id"),
        )

    if folded:
        return mk_default_key_folded(
            suite=obj.get("suite"),
            task=task,
            language=obj.get("language"),
            subset=obj.get("subset"),
            sample_id=obj.get("sample_id"),
            language_norm=language_norm,
            fold_subset=fold_subset,
            fold_sample_id=fold_sample_id,
        )
    return mk_default_key(
        suite=obj.get("suite"),
        task=task,
        language=obj.get("language"),
        subset=obj.get("subset"),
        sample_id=obj.get("sample_id"),
        language_norm=language_norm,
    )


def validate_has_fields(obj: Dict[str, Any], required: List[str], path_hint: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Record missing fields {missing} in {path_hint}. Record: {obj}")


def load_results_index(
    results_path: str,
    *,
    language_norm: str,
    strict_no_null_key_parts: bool,
    keep_split_field: bool,
    build_folded_index: bool,
    fold_subset: bool,
    fold_sample_id: bool,
    crosslingual_tasks: Set[str],
) -> Tuple[Dict[Tuple, List[Dict[str, Any]]], Dict[Tuple, List[Dict[str, Any]]]]:
    exact_index: Dict[Tuple, List[Dict[str, Any]]] = {}
    folded_index: Dict[Tuple, List[Dict[str, Any]]] = {}

    n = 0
    for res in iter_jsonl_xz(results_path):
        validate_has_fields(res, ["suite", "task", "language", "subset", "sample_id"], results_path)

        key = make_key(
            res,
            language_norm=language_norm,
            crosslingual_tasks=crosslingual_tasks,
            folded=False,
            fold_subset=fold_subset,
            fold_sample_id=fold_sample_id,
        )

        if strict_no_null_key_parts and any(p is None for p in key):
            raise ValueError(f"Results record has null key part(s) for key={key}. Record: {res}")

        payload = {
            "suite": res.get("suite"),
            "task": res.get("task"),
            "language": res.get("language"),
            "subset": res.get("subset"),
            "sample_id": res.get("sample_id"),
            "model_name": res.get("model_name"),
            "metric": res.get("metric"),
            "sample_score": res.get("sample_score"),
            "correct_answer": res.get("correct_answer"),
            "sanity_check_fragment": res.get("sanity_check_fragment"),
        }
        if keep_split_field:
            payload["split"] = res.get("split")

        exact_index.setdefault(key, []).append(payload)

        if build_folded_index:
            fkey = make_key(
                res,
                language_norm=language_norm,
                crosslingual_tasks=crosslingual_tasks,
                folded=True,
                fold_subset=fold_subset,
                fold_sample_id=fold_sample_id,
            )
            folded_index.setdefault(fkey, []).append(payload)

        n += 1

    print(f"[INFO] Loaded {n} result records into RAM.", file=sys.stderr)
    print(f"[INFO] Exact keys: {len(exact_index)}", file=sys.stderr)
    if build_folded_index:
        print(
            f"[INFO] Folded keys: {len(folded_index)} "
            f"(fold_subset={fold_subset}, fold_sample_id={fold_sample_id})",
            file=sys.stderr,
        )

    return exact_index, folded_index


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge slim annotations with slim results JSONL.xz; supports task-specific crosslingual joins (e.g. MMLU)."
    )
    ap.add_argument("--annotations", required=True, help="Slim annotations JSONL.")
    ap.add_argument("--results", required=True, help="Slim results JSONL.xz.")
    ap.add_argument("--output", required=True, help="Output JSONL path.")
    ap.add_argument("--language-norm", choices=["none", "upper", "lower"], default="upper")
    ap.add_argument("--strict-no-null-key-parts", action="store_true")
    ap.add_argument("--require-match", action="store_true", help="Fail if an annotation has no matching results.")
    ap.add_argument("--emit-unmatched-annotations", action="store_true",
                    help="Emit annotation with eval fields None if no match.")
    ap.add_argument("--keep-split-field", action="store_true",
                    help="Keep 'split' from results in output.")

    ap.add_argument("--enable-folding-fallback", action="store_true",
                    help="If exact match fails, try ASCII-folded subset/sample_id key.")
    ap.add_argument("--fold-subset", action="store_true", help="Fold subset for fallback key.")
    ap.add_argument("--fold-sample-id", action="store_true", help="Fold sample_id for fallback key.")
    ap.add_argument("--fail-on-ambiguous-folded-match", action="store_true",
                    help="Fail if folded fallback returns multiple matches after exact failed.")

    ap.add_argument(
        "--crosslingual-annotation-tasks",
        default="MMLU",
        help="Comma-separated tasks whose annotations should propagate across languages/suites. Default: MMLU",
    )

    args = ap.parse_args()

    if not os.path.exists(args.annotations):
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")
    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results file not found: {args.results}")

    crosslingual_tasks = {
        t.strip()
        for t in str(args.crosslingual_annotation_tasks).split(",")
        if t.strip()
    }

    build_folded = args.enable_folding_fallback and (args.fold_subset or args.fold_sample_id)

    exact_index, folded_index = load_results_index(
        args.results,
        language_norm=args.language_norm,
        strict_no_null_key_parts=args.strict_no_null_key_parts,
        keep_split_field=args.keep_split_field,
        build_folded_index=build_folded,
        fold_subset=args.fold_subset,
        fold_sample_id=args.fold_sample_id,
        crosslingual_tasks=crosslingual_tasks,
    )

    ensure_parent_dir(args.output)

    required_ann_fields = ["suite", "task", "language", "subset", "sample_id", "user", "user_id"]

    written = 0
    unmatched = 0
    matched_by = Counter()

    with open(args.output, "w", encoding="utf-8") as out:
        for ann in iter_jsonl(args.annotations):
            validate_has_fields(ann, required_ann_fields, args.annotations)

            key = make_key(
                ann,
                language_norm=args.language_norm,
                crosslingual_tasks=crosslingual_tasks,
                folded=False,
                fold_subset=args.fold_subset,
                fold_sample_id=args.fold_sample_id,
            )

            if args.strict_no_null_key_parts and any(p is None for p in key):
                raise ValueError(f"Annotation record has null key part(s) for key={key}. Record: {ann}")

            matches = exact_index.get(key, [])
            used_folded = False

            if not matches and build_folded:
                fkey = make_key(
                    ann,
                    language_norm=args.language_norm,
                    crosslingual_tasks=crosslingual_tasks,
                    folded=True,
                    fold_subset=args.fold_subset,
                    fold_sample_id=args.fold_sample_id,
                )
                matches = folded_index.get(fkey, [])
                used_folded = bool(matches)

                if used_folded and args.fail_on_ambiguous_folded_match and len(matches) > 1:
                    raise ValueError(
                        f"Ambiguous folded fallback match for annotation key={key} -> folded_key={fkey}: "
                        f"{len(matches)} matches."
                    )

            if not matches:
                unmatched += 1
                if args.require_match:
                    raise ValueError(f"No results match for annotation record. Key={key}. Record: {ann}")
                if args.emit_unmatched_annotations:
                    merged = dict(ann)
                    merged["annotation_suite"] = ann.get("suite")
                    merged["annotation_task"] = ann.get("task")
                    merged["annotation_language"] = ann.get("language")
                    merged["annotation_subset"] = ann.get("subset")
                    merged["annotation_sample_id"] = ann.get("sample_id")
                    merged.update({
                        "model_name": None,
                        "metric": None,
                        "sample_score": None,
                        "correct_answer": None,
                        "sanity_check_fragment": None,
                    })
                    if args.keep_split_field:
                        merged["split"] = None
                    out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    written += 1
                continue

            matched_by["folded" if used_folded else "exact"] += 1

            for res_payload in matches:
                merged = dict(ann)
                merged["annotation_suite"] = ann.get("suite")
                merged["annotation_task"] = ann.get("task")
                merged["annotation_language"] = ann.get("language")
                merged["annotation_subset"] = ann.get("subset")
                merged["annotation_sample_id"] = ann.get("sample_id")
                merged.update(res_payload)
                out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                written += 1

    print(f"[OK] Wrote {written} merged records to: {args.output}", file=sys.stderr)
    print(f"[INFO] Unmatched annotation records: {unmatched}", file=sys.stderr)
    print(f"[INFO] Matched-by: {dict(matched_by)}", file=sys.stderr)
    print(f"[INFO] Crosslingual annotation tasks: {sorted(crosslingual_tasks)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


