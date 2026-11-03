#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnose unmatched annotation records after merging annotations with results.

Join key used (no split):
  (suite, task, language, subset, sample_id)

Outputs:
- unmatched JSONL with extra fields: join_key, reason, candidates (limited)
- stderr summary: counts per reason, and some examples

This script loads the results (JSONL.xz) into RAM (OK for ~6MB compressed) and streams annotations.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


JOIN_FIELDS = ("suite", "task", "language", "subset", "sample_id")


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


def mk_key_no_split(
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
        None if task is None else str(task).strip(),
        norm_language(language, language_norm),
        None if subset is None else str(subset).strip(),
        None if sample_id is None else str(sample_id).strip(),
    )


def validate_has_fields(obj: Dict[str, Any], required: List[str], path_hint: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Record missing fields {missing} in {path_hint}. Record: {obj}")


def load_results_indices(
    results_path: str,
    *,
    language_norm: str,
    strict_no_null_key_parts: bool,
) -> Dict[str, Any]:
    """
    Build multiple indices to diagnose mismatches:
      exact_keys: set((suite, task, lang, subset, sample_id))
      by_sample: sample_id -> set of (suite, task, lang, subset)
      by_suite_task_subset_sample: (suite, task, subset, sample_id) -> set(langs)
      by_suite_task_lang_sample: (suite, task, lang, sample_id) -> set(subsets)
      by_task_lang_subset_sample: (task, lang, subset, sample_id) -> set(suites)
      by_suite_lang_subset_sample: (suite, lang, subset, sample_id) -> set(tasks)
    """
    exact_keys = set()
    by_sample: Dict[str, set] = {}
    by_stss: Dict[Tuple[str, str, str, str], set] = {}
    by_stls: Dict[Tuple[str, str, str, str], set] = {}
    by_tlss: Dict[Tuple[str, str, str, str], set] = {}
    by_slss: Dict[Tuple[str, str, str, str], set] = {}

    n = 0
    for res in iter_jsonl_xz(results_path):
        validate_has_fields(res, ["suite", "task", "language", "subset", "sample_id"], results_path)

        suite = str(res.get("suite")).strip()
        task = str(res.get("task")).strip()
        lang = norm_language(res.get("language"), language_norm)
        subset = str(res.get("subset")).strip()
        sid = str(res.get("sample_id")).strip()

        key = (suite, task, lang, subset, sid)

        if strict_no_null_key_parts and any(p is None for p in key):
            raise ValueError(f"Results record has null key part(s) for key={key}. Record: {res}")

        exact_keys.add(key)

        by_sample.setdefault(sid, set()).add((suite, task, lang, subset))

        by_stss.setdefault((suite, task, subset, sid), set()).add(lang)
        by_stls.setdefault((suite, task, lang, sid), set()).add(subset)
        by_tlss.setdefault((task, lang, subset, sid), set()).add(suite)
        by_slss.setdefault((suite, lang, subset, sid), set()).add(task)

        n += 1

    print(f"[INFO] Results loaded: {n} records, {len(exact_keys)} exact keys.", file=sys.stderr)
    return {
        "exact_keys": exact_keys,
        "by_sample": by_sample,
        "by_stss": by_stss,
        "by_stls": by_stls,
        "by_tlss": by_tlss,
        "by_slss": by_slss,
    }


def diagnose_one(
    key: Tuple[str, str, str, str, str],
    indices: Dict[str, Any],
    *,
    candidates_limit: int,
) -> Tuple[str, Dict[str, Any]]:
    suite, task, lang, subset, sid = key
    exact_keys = indices["exact_keys"]
    by_sample = indices["by_sample"]
    by_stss = indices["by_stss"]
    by_stls = indices["by_stls"]
    by_tlss = indices["by_tlss"]
    by_slss = indices["by_slss"]

    # sanity: caller only uses this when key not in exact_keys
    if sid not in by_sample:
        return "sample_id_not_in_results", {"sample_id": sid}

    # Language mismatch?
    langs = by_stss.get((suite, task, subset, sid))
    if langs:
        # There are results for same suite/task/subset/sample_id, but different language
        return "language_mismatch", {
            "expected_language": lang,
            "available_languages": sorted(list(langs))[:candidates_limit],
        }

    # Subset mismatch?
    subsets = by_stls.get((suite, task, lang, sid))
    if subsets:
        return "subset_mismatch", {
            "expected_subset": subset,
            "available_subsets": sorted(list(subsets))[:candidates_limit],
        }

    # Suite mismatch?
    suites = by_tlss.get((task, lang, subset, sid))
    if suites:
        return "suite_mismatch", {
            "expected_suite": suite,
            "available_suites": sorted(list(suites))[:candidates_limit],
        }

    # Task mismatch?
    tasks = by_slss.get((suite, lang, subset, sid))
    if tasks:
        return "task_mismatch", {
            "expected_task": task,
            "available_tasks": sorted(list(tasks))[:candidates_limit],
        }

    # Otherwise: sample_id exists but combination differs more broadly
    combos = sorted(list(by_sample.get(sid, set())))
    combos = combos[:candidates_limit]
    return "combination_mismatch", {
        "sample_id": sid,
        "available_combinations_for_sample_id": [
            {"suite": s, "task": t, "language": l, "subset": sub} for (s, t, l, sub) in combos
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose unmatched annotation records (join without split).")
    ap.add_argument("--annotations", required=True, help="Slim annotations JSONL.")
    ap.add_argument("--results", required=True, help="Slim results JSONL.xz.")
    ap.add_argument("--out-unmatched", required=True, help="Output JSONL for unmatched annotations + diagnosis.")
    ap.add_argument("--language-norm", choices=["none", "upper", "lower"], default="upper")
    ap.add_argument("--strict-no-null-key-parts", action="store_true")
    ap.add_argument("--candidates-limit", type=int, default=50, help="Max candidates stored per unmatched record.")
    ap.add_argument(
        "--print-top-examples",
        type=int,
        default=10,
        help="Print N example join keys per reason to stderr.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.annotations):
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")
    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results file not found: {args.results}")

    indices = load_results_indices(
        args.results,
        language_norm=args.language_norm,
        strict_no_null_key_parts=args.strict_no_null_key_parts,
    )

    ensure_parent_dir(args.out_unmatched)

    required_ann_fields = ["suite", "task", "language", "subset", "sample_id", "user", "user_id"]

    reason_counts = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = {}

    total = 0
    unmatched = 0

    with open(args.out_unmatched, "w", encoding="utf-8") as out:
        for ann in iter_jsonl(args.annotations):
            total += 1
            validate_has_fields(ann, required_ann_fields, args.annotations)

            key = mk_key_no_split(
                ann.get("suite"),
                ann.get("task"),
                ann.get("language"),
                ann.get("subset"),
                ann.get("sample_id"),
                language_norm=args.language_norm,
            )

            if args.strict_no_null_key_parts and any(p is None for p in key):
                raise ValueError(f"Annotation record has null key part(s) for key={key}. Record: {ann}")

            if key in indices["exact_keys"]:
                continue

            unmatched += 1
            reason, detail = diagnose_one(key, indices, candidates_limit=args.candidates_limit)
            reason_counts[reason] += 1

            payload = dict(ann)
            payload["join_key"] = dict(zip(JOIN_FIELDS, key))
            payload["unmatched_reason"] = reason
            payload["diagnosis"] = detail

            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if len(examples.get(reason, [])) < args.print_top_examples:
                examples.setdefault(reason, []).append(payload["join_key"])

    print(f"[OK] Scanned {total} annotation records.", file=sys.stderr)
    print(f"[INFO] Unmatched: {unmatched}", file=sys.stderr)
    print("[INFO] Reasons:", file=sys.stderr)
    for r, c in reason_counts.most_common():
        print(f"  - {r}: {c}", file=sys.stderr)

    if args.print_top_examples > 0:
        print("[INFO] Example keys per reason:", file=sys.stderr)
        for r, exs in examples.items():
            print(f"  {r}:", file=sys.stderr)
            for k in exs:
                print(f"    {k}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

