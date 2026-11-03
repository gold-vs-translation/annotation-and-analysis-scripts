#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"]).delimiter
    except Exception:
        return ","


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    delim = sniff_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        return [row for row in r]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def summarize_one(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {
        "total": 0,
        "OK": 0,
        "WARN": 0,
        "ERROR": 0,
        "needs_fix_YES": 0,
        "needs_fix_NO": 0,
    }
    for row in rows:
        counts["total"] += 1
        sev = (row.get("severity") or "").strip()
        nf = (row.get("needs_fix") or "").strip()
        if sev in ("OK", "WARN", "ERROR"):
            counts[sev] += 1
        if nf == "YES":
            counts["needs_fix_YES"] += 1
        elif nf == "NO":
            counts["needs_fix_NO"] += 1
    return counts


def write_tsv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def discover_csv_files(in_path: Path) -> List[Path]:
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        return sorted([p for p in in_path.rglob("*.csv") if p.is_file()])
    raise SystemExit(f"--in path not found: {in_path}")


def infer_language(rows: List[Dict[str, str]], fallback_dirname: str) -> str:
    if rows:
        lang = (rows[0].get("language") or "").strip()
        if lang:
            return lang
    return fallback_dirname


def build_to_fix_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if (r.get("needs_fix") or "").strip() == "YES":
            out.append(
                {
                    "sample_id": (r.get("sample_id") or "").strip(),
                    "severity": (r.get("severity") or "").strip(),
                    "missing_fields": (r.get("missing_fields") or "").strip(),
                    "contradictions": (r.get("contradictions") or "").strip(),
                    "format_issues": (r.get("format_issues") or "").strip(),
                    "span_issues": (r.get("span_issues") or "").strip(),
                    "content_sanity_flags": (r.get("content_sanity_flags") or "").strip(),
                    "suggest_required_knowledge": (r.get("suggest_required_knowledge") or "").strip(),
                    "suggest_csl": (r.get("suggest_csl") or "").strip(),
                    "suggest_time_sensitive": (r.get("suggest_time_sensitive") or "").strip(),
                    "suggest_subject_category": (r.get("suggest_subject_category") or "").strip(),
                    "suggest_culture": (r.get("suggest_culture") or "").strip(),
                    "suggest_region": (r.get("suggest_region") or "").strip(),
                    "suggest_country": (r.get("suggest_country") or "").strip(),
                    "suggest_span_fix": (r.get("suggest_span_fix") or "").strip(),
                    "comment": (r.get("comment") or "").strip(),
                }
            )
    return out


def write_to_fix_tsv(path: Path, to_fix: List[Dict[str, str]]) -> None:
    header = [
        "sample_id",
        "severity",
        "missing_fields",
        "contradictions",
        "format_issues",
        "span_issues",
        "content_sanity_flags",
        "suggest_required_knowledge",
        "suggest_csl",
        "suggest_time_sensitive",
        "suggest_subject_category",
        "suggest_culture",
        "suggest_region",
        "suggest_country",
        "suggest_span_fix",
        "comment",
    ]
    rows_out = [[d.get(h, "") for h in header] for d in to_fix]
    write_tsv(path, header, rows_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="CSV file OR directory containing CSVs (recursive)")
    ap.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help=("If --in is a FILE: path to summary TSV file.\n" "If --in is a DIR: output directory for summary.tsv + per-file *.to_fix.tsv."),
    )
    ap.add_argument(
        "--to-fix-out",
        default="",
        help="Only used when --in is a FILE: path to write the single .to_fix.tsv file (optional).",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    csv_files = discover_csv_files(in_path)
    if not csv_files:
        raise SystemExit(f"No CSV files found under: {in_path}")

    summary_header = [
        "review_file",
        "language",
        "total",
        "OK",
        "WARN",
        "ERROR",
        "needs_fix_YES",
        "needs_fix_NO",
    ]

    summary_rows: List[List[str]] = []

    # Case 1: single CSV file in -> summary TSV file out
    if in_path.is_file():
        rows = read_csv_rows(in_path)
        counts = summarize_one(rows)
        lang = infer_language(rows, fallback_dirname=in_path.parent.name)

        summary_rows.append(
            [
                in_path.name,
                lang,
                str(counts["total"]),
                str(counts["OK"]),
                str(counts["WARN"]),
                str(counts["ERROR"]),
                str(counts["needs_fix_YES"]),
                str(counts["needs_fix_NO"]),
            ]
        )

        # write summary TSV to the exact file path
        write_tsv(out_path, summary_header, summary_rows)
        print(f"Wrote summary TSV: {out_path}")

        # optional single to_fix file
        if args.to_fix_out.strip():
            to_fix_path = Path(args.to_fix_out)
        else:
            to_fix_path = out_path.with_suffix(".to_fix.tsv")

        to_fix = build_to_fix_rows(rows)
        write_to_fix_tsv(to_fix_path, to_fix)
        print(f"Wrote to-fix TSV: {to_fix_path}")
        return

    # Case 2: directory in -> directory out (previous behavior)
    out_dir = out_path
    ensure_dir(out_dir)

    for csv_path in csv_files:
        rows = read_csv_rows(csv_path)
        counts = summarize_one(rows)

        lang = infer_language(rows, fallback_dirname=csv_path.parent.name)
        rel = str(csv_path.relative_to(in_path))

        summary_rows.append(
            [
                rel,
                lang,
                str(counts["total"]),
                str(counts["OK"]),
                str(counts["WARN"]),
                str(counts["ERROR"]),
                str(counts["needs_fix_YES"]),
                str(counts["needs_fix_NO"]),
            ]
        )

        to_fix = build_to_fix_rows(rows)
        to_fix_path = out_dir / Path(rel).with_suffix(".to_fix.tsv")
        write_to_fix_tsv(to_fix_path, to_fix)

    summary_path = out_dir / "summary.tsv"
    write_tsv(summary_path, summary_header, summary_rows)
    print(f"Wrote: {summary_path}")
    print(f"Wrote per-review to_fix.tsv files under: {out_dir}")


if __name__ == "__main__":
    main()


