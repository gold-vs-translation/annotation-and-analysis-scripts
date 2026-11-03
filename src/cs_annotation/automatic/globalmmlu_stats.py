#!/usr/bin/env python3
"""
Generate planning stats for GlobalMMLU and GlobalMMLU-Lite.

This script is intended for the decision step before another automatic
annotation run. It answers these questions:

1) How many CS/CA-annotated items exist in total?
2) How many of those belong to GlobalMMLU vs GlobalMMLU-Lite?
3) How many GlobalMMLU items are not CS/CA-annotated?
4) How many items exist per GlobalMMLU subject (subset)?
5) How many items exist per GlobalMMLU-Lite category?

Accepted input files: .jsonl and .jsonl.xz
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

VALID_CSL = {"CS", "CA"}


def open_text_auto(path: Path):
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text_auto(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {lineno}: {e}") from e


def collect_jsonl_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.xz"))
    )
    if not files:
        raise FileNotFoundError(f"No .jsonl / .jsonl.xz files found in {path}")
    return files


def normalize_csl_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().upper()
        return v if v in VALID_CSL else None
    return None


def majority_vote_str(values: Iterable[Any]) -> Optional[str]:
    cleaned = [normalize_csl_value(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    counts = Counter(cleaned)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] == most_common[1][1]:
        return None
    return most_common[0][0]


def extract_csl(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for container_key, source in [
        ("culture_ann_ref", "culture_ann_ref.cultural_sensitivity_label"),
        ("culture_ann", "culture_ann.cultural_sensitivity_label"),
    ]:
        container = record.get(container_key)
        if not isinstance(container, dict):
            continue
        value = container.get("cultural_sensitivity_label")
        if isinstance(value, list):
            voted = majority_vote_str(value)
            if voted is not None:
                return voted, source
        else:
            norm = normalize_csl_value(value)
            if norm is not None:
                return norm, source
    return None, None


def load_lite_category_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--lite-category-map must point to a JSON object: {subset: category}")
    out = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError("--lite-category-map must contain only string -> string pairs")
        out[k] = v
    return out


def detect_lite_category(record: Dict[str, Any], subset_to_category: Dict[str, str]) -> Tuple[str, str]:
    direct = record.get("category")
    if isinstance(direct, str) and direct.strip():
        return direct.strip(), "record.category"

    ann = record.get("culture_ann")
    if isinstance(ann, dict):
        value = ann.get("subject_category")
        if isinstance(value, str) and value.strip():
            return value.strip(), "culture_ann.subject_category"

    ref = record.get("culture_ann_ref")
    if isinstance(ref, dict):
        value = ref.get("subject_category")
        if isinstance(value, str) and value.strip():
            return value.strip(), "culture_ann_ref.subject_category"

    subset = record.get("subset")
    if isinstance(subset, str) and subset in subset_to_category:
        return subset_to_category[subset], "mapping_json"

    if isinstance(subset, str) and subset.strip():
        return subset.strip(), "subset_fallback"

    return "__MISSING__", "missing"


def write_counter_csv(path: Path, left_header: str, counter: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([left_header, "n_items"])
        for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([key, value])


def write_rows_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--globalmmlu-dir", required=True)
    ap.add_argument("--globalmmlulite-file", required=True)
    ap.add_argument("--lite-category-map")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    globalmmlu_files = collect_jsonl_files(Path(args.globalmmlu_dir))
    globalmmlulite_files = collect_jsonl_files(Path(args.globalmmlulite_file))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lite_category_map = load_lite_category_map(Path(args.lite_category_map) if args.lite_category_map else None)

    summary: Dict[str, Any] = {
        "inputs": {
            "globalmmlu_files": [str(p) for p in globalmmlu_files],
            "globalmmlulite_files": [str(p) for p in globalmmlulite_files],
            "lite_category_map": str(args.lite_category_map) if args.lite_category_map else None,
        },
        "globalmmlu": {
            "total_items": 0,
            "csl_annotated_items": 0,
            "cs_items": 0,
            "ca_items": 0,
            "not_csl_annotated_items": 0,
        },
        "globalmmlulite": {
            "total_items": 0,
            "csl_annotated_items": 0,
            "cs_items": 0,
            "ca_items": 0,
            "not_csl_annotated_items": 0,
        },
        "combined": {},
    }

    globalmmlu_subject_counts: Counter = Counter()
    globalmmlu_language_counts: Counter = Counter()
    globalmmlu_csl_source_counts: Counter = Counter()
    globalmmlu_file_rows: List[Dict[str, Any]] = []

    lite_category_counts: Counter = Counter()
    lite_language_counts: Counter = Counter()
    lite_csl_source_counts: Counter = Counter()
    lite_category_source_counts: Counter = Counter()
    lite_file_rows: List[Dict[str, Any]] = []

    for file_path in globalmmlu_files:
        file_total = 0
        file_annotated = 0
        file_missing = 0
        file_cs = 0
        file_ca = 0
        for rec in iter_jsonl(file_path):
            file_total += 1
            summary["globalmmlu"]["total_items"] += 1
            lang = rec.get("language")
            if isinstance(lang, str):
                globalmmlu_language_counts[lang] += 1
            subset = rec.get("subset")
            globalmmlu_subject_counts[subset.strip() if isinstance(subset, str) and subset.strip() else "__MISSING__"] += 1
            csl, source = extract_csl(rec)
            if csl is None:
                summary["globalmmlu"]["not_csl_annotated_items"] += 1
                file_missing += 1
            else:
                summary["globalmmlu"]["csl_annotated_items"] += 1
                file_annotated += 1
                globalmmlu_csl_source_counts[source or "unknown"] += 1
                if csl == "CS":
                    summary["globalmmlu"]["cs_items"] += 1
                    file_cs += 1
                elif csl == "CA":
                    summary["globalmmlu"]["ca_items"] += 1
                    file_ca += 1
        globalmmlu_file_rows.append({
            "file": str(file_path),
            "total_items": file_total,
            "csl_annotated_items": file_annotated,
            "cs_items": file_cs,
            "ca_items": file_ca,
            "not_csl_annotated_items": file_missing,
        })

    for file_path in globalmmlulite_files:
        file_total = 0
        file_annotated = 0
        file_missing = 0
        file_cs = 0
        file_ca = 0
        for rec in iter_jsonl(file_path):
            file_total += 1
            summary["globalmmlulite"]["total_items"] += 1
            lang = rec.get("language")
            if isinstance(lang, str):
                lite_language_counts[lang] += 1
            category, category_source = detect_lite_category(rec, lite_category_map)
            lite_category_counts[category] += 1
            lite_category_source_counts[category_source] += 1
            csl, source = extract_csl(rec)
            if csl is None:
                summary["globalmmlulite"]["not_csl_annotated_items"] += 1
                file_missing += 1
            else:
                summary["globalmmlulite"]["csl_annotated_items"] += 1
                file_annotated += 1
                lite_csl_source_counts[source or "unknown"] += 1
                if csl == "CS":
                    summary["globalmmlulite"]["cs_items"] += 1
                    file_cs += 1
                elif csl == "CA":
                    summary["globalmmlulite"]["ca_items"] += 1
                    file_ca += 1
        lite_file_rows.append({
            "file": str(file_path),
            "total_items": file_total,
            "csl_annotated_items": file_annotated,
            "cs_items": file_cs,
            "ca_items": file_ca,
            "not_csl_annotated_items": file_missing,
        })

    for key in ["total_items", "csl_annotated_items", "cs_items", "ca_items", "not_csl_annotated_items"]:
        summary["combined"][key] = summary["globalmmlu"][key] + summary["globalmmlulite"][key]

    summary["globalmmlu"]["languages"] = dict(sorted(globalmmlu_language_counts.items()))
    summary["globalmmlu"]["csl_source_counts"] = dict(sorted(globalmmlu_csl_source_counts.items()))
    summary["globalmmlulite"]["languages"] = dict(sorted(lite_language_counts.items()))
    summary["globalmmlulite"]["csl_source_counts"] = dict(sorted(lite_csl_source_counts.items()))
    summary["globalmmlulite"]["category_source_counts"] = dict(sorted(lite_category_source_counts.items()))

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_counter_csv(out_dir / "globalmmlu_subject_counts.csv", "subset", globalmmlu_subject_counts)
    write_counter_csv(out_dir / "globalmmlu_language_counts.csv", "language", globalmmlu_language_counts)
    write_counter_csv(out_dir / "globalmmlulite_category_counts.csv", "category", lite_category_counts)
    write_counter_csv(out_dir / "globalmmlulite_language_counts.csv", "language", lite_language_counts)
    write_rows_csv(
        out_dir / "globalmmlu_file_counts.csv",
        globalmmlu_file_rows,
        ["file", "total_items", "csl_annotated_items", "cs_items", "ca_items", "not_csl_annotated_items"],
    )
    write_rows_csv(
        out_dir / "globalmmlulite_file_counts.csv",
        lite_file_rows,
        ["file", "total_items", "csl_annotated_items", "cs_items", "ca_items", "not_csl_annotated_items"],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()


