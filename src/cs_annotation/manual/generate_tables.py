#!/usr/bin/env python3
"""
generate_tables.py — Generate all paper tables for the human annotation section.

Reads:
  --post-review   : Final JSONL export from Argilla (post-review/corrected)
  --pre-review-dir: Directory with per-annotator CSVs (pre-review, for IAA + review stats)

Produces:
  Table 1: Label distributions per language (post-review)
  Table 2: Review statistics per annotator (pre-review)
  Table 3: Pre→Post label shift comparison
  Table 4: IAA for DA (pre-review, 3 annotators)
  Table 5: Pairwise Cohen's κ for DA

Output: Markdown tables (stdout) + LaTeX tables (--latex-out directory)
"""

import argparse
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

# ── Target languages (exclude EN/GlobalMMLU reference set) ────────────
TARGET_LANGS = ["DA", "ET", "LT", "NB", "SV"]

# ── Pre-review CSV file patterns ──────────────────────────────────────
# Maps (language, annotator_short_id) -> CSV filename prefix
# The script auto-discovers CSVs by language prefix.


# =====================================================================
#  I/O helpers
# =====================================================================

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_pre_review_csvs(directory: Path) -> dict[str, list[Path]]:
    """Return {language: [csv_path, ...]} for all per-annotator CSVs."""
    result = {}
    for p in sorted(directory.glob("*.csv")):
        lang = p.name[:2].upper()
        if lang in TARGET_LANGS:
            result.setdefault(lang, []).append(p)
    return result


def load_pre_review_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# =====================================================================
#  Table 1: Post-review label distributions
# =====================================================================

def table_label_distributions(post_rows: list[dict], latex_dir: Path | None):
    print("\n" + "=" * 90)
    print("  TABLE 1: Label Distributions per Language (Post-Review, Final)")
    print("=" * 90)

    # --- Required knowledge ---
    rk_cats = ["none", "regional", "cultural", "dialect"]
    print("\n  (a) Required Knowledge Type\n")
    header = f"{'Lang':<6}{'n':>5}"
    for cat in rk_cats:
        header += f"  {cat:>14}"
    print(header)
    print("-" * len(header))

    all_data_rk = []
    for lang in TARGET_LANGS:
        lr = [r for r in post_rows if r["language"] == lang]
        n = len(lr)
        dist = Counter(r["required_knowledge"] for r in lr)
        row = {"lang": lang, "n": n}
        line = f"{lang:<6}{n:>5}"
        for cat in rk_cats:
            c = dist.get(cat, 0)
            pct = c / n * 100 if n else 0
            line += f"  {c:>4} ({pct:4.1f}%)"
            row[cat] = c
        print(line)
        all_data_rk.append(row)

    # Aggregate
    agg_n = sum(r["n"] for r in all_data_rk)
    line = f"{'ALL':<6}{agg_n:>5}"
    for cat in rk_cats:
        c = sum(r[cat] for r in all_data_rk)
        pct = c / agg_n * 100
        line += f"  {c:>4} ({pct:4.1f}%)"
    print("-" * len(header))
    print(line)

    # --- CS/CA ---
    csl_cats = ["CS", "CA"]
    print(f"\n  (b) Cultural Sensitivity Label\n")
    header = f"{'Lang':<6}{'n':>5}"
    for cat in csl_cats:
        header += f"  {cat:>14}"
    print(header)
    print("-" * len(header))

    all_data_csl = []
    for lang in TARGET_LANGS:
        lr = [r for r in post_rows if r["language"] == lang]
        n = len(lr)
        dist = Counter(r["cultural_sensitivity_label"] for r in lr)
        row = {"lang": lang, "n": n}
        line = f"{lang:<6}{n:>5}"
        for cat in csl_cats:
            c = dist.get(cat, 0)
            pct = c / n * 100 if n else 0
            line += f"  {c:>4} ({pct:4.1f}%)"
            row[cat] = c
        print(line)
        all_data_csl.append(row)

    agg_n = sum(r["n"] for r in all_data_csl)
    line = f"{'ALL':<6}{agg_n:>5}"
    for cat in csl_cats:
        c = sum(r[cat] for r in all_data_csl)
        pct = c / agg_n * 100
        line += f"  {c:>4} ({pct:4.1f}%)"
    print("-" * len(header))
    print(line)

    # --- Time sensitive ---
    ts_cats = ["No", "Yes"]
    print(f"\n  (c) Time Sensitivity\n")
    header = f"{'Lang':<6}{'n':>5}"
    for cat in ts_cats:
        header += f"  {cat:>14}"
    print(header)
    print("-" * len(header))

    for lang in TARGET_LANGS:
        lr = [r for r in post_rows if r["language"] == lang]
        n = len(lr)
        dist = Counter(r.get("time_sensitive", "MISSING") for r in lr)
        line = f"{lang:<6}{n:>5}"
        for cat in ts_cats:
            c = dist.get(cat, 0)
            pct = c / n * 100 if n else 0
            line += f"  {c:>4} ({pct:4.1f}%)"
        print(line)

    # --- Subject category ---
    sc_cats = ["STEM", "Humanities", "Social Sciences", "Medical", "Business", "Other"]
    print(f"\n  (d) Subject Category\n")
    header = f"{'Lang':<6}{'n':>5}"
    for cat in sc_cats:
        header += f"  {cat:>10}"
    print(header)
    print("-" * len(header))

    all_data_sc = []
    for lang in TARGET_LANGS:
        lr = [r for r in post_rows if r["language"] == lang]
        n = len(lr)
        dist = Counter(r.get("subject_category", "MISSING") for r in lr)
        row = {"lang": lang, "n": n}
        line = f"{lang:<6}{n:>5}"
        for cat in sc_cats:
            c = dist.get(cat, 0)
            pct = c / n * 100 if n else 0
            line += f"  {c:>3}({pct:4.1f}%)"
            row[cat] = c
        print(line)
        all_data_sc.append(row)

    agg_n = sum(r["n"] for r in all_data_sc)
    line = f"{'ALL':<6}{agg_n:>5}"
    for cat in sc_cats:
        c = sum(r[cat] for r in all_data_sc)
        pct = c / agg_n * 100
        line += f"  {c:>3}({pct:4.1f}%)"
    print("-" * len(header))
    print(line)

    # --- LaTeX output ---
    if latex_dir:
        _write_latex_label_dist(latex_dir, post_rows, all_data_rk, all_data_csl, all_data_sc)


def _write_latex_label_dist(latex_dir, post_rows, rk_data, csl_data, sc_data):
    """Write combined LaTeX table for label distributions."""
    out = latex_dir / "table_label_distributions.tex"
    rk_cats = ["none", "regional", "cultural", "dialect"]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Distribution of human annotation labels per language (post-review). "
                 r"Required knowledge type (Q1), cultural sensitivity label (CS/CA), "
                 r"and subject category are shown with absolute counts and percentages.}")
    lines.append(r"\label{tab:annot-label-dist}")
    lines.append(r"\begin{tabular}{l r rrrr rr}")
    lines.append(r"\toprule")
    lines.append(r"& & \multicolumn{4}{c}{\textbf{Required Knowledge (Q1)}} & \multicolumn{2}{c}{\textbf{CS/CA}} \\")
    lines.append(r"\cmidrule(lr){3-6} \cmidrule(lr){7-8}")
    lines.append(r"\textbf{Lang} & \textbf{n} & \texttt{none} & \texttt{regional} & \texttt{cultural} & \texttt{dialect} & CS & CA \\")
    lines.append(r"\midrule")

    for lang in TARGET_LANGS:
        lr = [r for r in post_rows if r["language"] == lang]
        n = len(lr)
        rk_dist = Counter(r["required_knowledge"] for r in lr)
        csl_dist = Counter(r["cultural_sensitivity_label"] for r in lr)
        cells = [lang, str(n)]
        for cat in rk_cats:
            c = rk_dist.get(cat, 0)
            cells.append(f"{c} ({c/n*100:.0f}\\%)")
        for cat in ["CS", "CA"]:
            c = csl_dist.get(cat, 0)
            cells.append(f"{c} ({c/n*100:.0f}\\%)")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    agg_n = sum(len([r for r in post_rows if r["language"] == l]) for l in TARGET_LANGS)
    agg_rk = Counter(r["required_knowledge"] for r in post_rows if r["language"] in TARGET_LANGS)
    agg_csl = Counter(r["cultural_sensitivity_label"] for r in post_rows if r["language"] in TARGET_LANGS)
    cells = [r"\textbf{All}", str(agg_n)]
    for cat in rk_cats:
        c = agg_rk.get(cat, 0)
        cells.append(f"{c} ({c/agg_n*100:.0f}\\%)")
    for cat in ["CS", "CA"]:
        c = agg_csl.get(cat, 0)
        cells.append(f"{c} ({c/agg_n*100:.0f}\\%)")
    lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [LaTeX] Wrote: {out}")


# =====================================================================
#  Table 2: Pre-review QA statistics
# =====================================================================

def table_review_statistics(pre_csvs: dict[str, list[Path]], latex_dir: Path | None):
    print("\n" + "=" * 90)
    print("  TABLE 2: Automated Review Statistics (Pre-Review)")
    print("=" * 90)

    header = f"{'Annotator':<28}{'Lang':<6}{'n':>5}{'OK':>6}{'WARN':>7}{'ERROR':>7}{'fix':>8}{'ERR%':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    all_rows = []
    for lang in TARGET_LANGS:
        paths = pre_csvs.get(lang, [])
        for p in paths:
            rows = load_pre_review_csv(p)
            n = len(rows)
            ok = sum(1 for r in rows if r.get("severity") == "OK")
            warn = sum(1 for r in rows if r.get("severity") == "WARN")
            error = sum(1 for r in rows if r.get("severity") == "ERROR")
            fix_yes = sum(1 for r in rows if r.get("needs_fix") == "YES")
            err_pct = error / n * 100 if n else 0
            label = p.stem.split("___")[0]
            print(f"{label:<28}{lang:<6}{n:>5}{ok:>6}{warn:>7}{error:>7}{fix_yes:>8}{err_pct:>6.1f}%")
            all_rows.append({
                "label": label, "lang": lang, "n": n,
                "OK": ok, "WARN": warn, "ERROR": error,
                "fix_YES": fix_yes, "err_pct": err_pct,
            })

    # Aggregate
    print("-" * len(header))
    tot = {k: sum(r[k] for r in all_rows) for k in ["n", "OK", "WARN", "ERROR", "fix_YES"]}
    err_pct = tot["ERROR"] / tot["n"] * 100 if tot["n"] else 0
    print(f"{'ALL':<28}{'':<6}{tot['n']:>5}{tot['OK']:>6}{tot['WARN']:>7}{tot['ERROR']:>7}{tot['fix_YES']:>8}{err_pct:>6.1f}%")

    if latex_dir:
        _write_latex_review_stats(latex_dir, all_rows, tot)


def _write_latex_review_stats(latex_dir, all_rows, tot):
    out = latex_dir / "table_review_statistics.tex"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Automated review statistics per annotator (pre-review). "
                 r"\textsc{Error}: issues requiring correction; "
                 r"\textsc{Warn}: minor consistency issues; "
                 r"\textsc{OK}: no issues detected; "
                 r"\textsc{Fix}: items flagged as needing correction.}")
    lines.append(r"\label{tab:annot-review-stats}")
    lines.append(r"\begin{tabular}{l l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Annotator} & \textbf{Lang} & \textbf{n} & \textbf{OK} & \textbf{Warn} & \textbf{Error} & \textbf{Fix} \\")
    lines.append(r"\midrule")
    for r in all_rows:
        lines.append(f"{r['label']} & {r['lang']} & {r['n']} & {r['OK']} & {r['WARN']} & {r['ERROR']} & {r['fix_YES']} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{All}} & & {tot['n']} & {tot['OK']} & {tot['WARN']} & {tot['ERROR']} & {tot['fix_YES']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [LaTeX] Wrote: {out}")


# =====================================================================
#  Table 3: Pre→Post label shift
# =====================================================================

def table_pre_post_comparison(post_rows: list[dict], pre_csvs: dict[str, list[Path]], latex_dir: Path | None):
    print("\n" + "=" * 90)
    print("  TABLE 3: Pre→Post Review Label Shift (Required Knowledge)")
    print("=" * 90)

    # For each single-annotator language + DA (ann_001 = annotator_1)
    # Map pre-review annotator to the post-review user
    pre_map = {
        "DA": "DA_ann001",
        "ET": "ET_ann004",
        "LT": "LT_ann005",
        "NB": "NB_ann006",
        "SV": "SV_ann007",
    }
    rk_cats = ["none", "regional", "cultural", "dialect"]

    header = f"{'Lang':<6}{'Stage':<7}"
    for cat in rk_cats:
        header += f"  {cat:>10}"
    header += f"  {'CS':>6}{'CA':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for lang in TARGET_LANGS:
        # Post
        post_lr = [r for r in post_rows if r["language"] == lang]
        post_rk = Counter(r["required_knowledge"] for r in post_lr)
        post_csl = Counter(r["cultural_sensitivity_label"] for r in post_lr)

        # Pre: find matching CSV
        prefix = pre_map[lang]
        pre_path = None
        for p in pre_csvs.get(lang, []):
            if p.stem.startswith(prefix):
                pre_path = p
                break
        if not pre_path:
            print(f"{lang:<6}{'pre':<7}  (no matching pre-review CSV found)")
            continue

        pre_rows_l = load_pre_review_csv(pre_path)
        pre_rk = Counter(r["annot_required_knowledge"] for r in pre_rows_l)
        pre_csl = Counter(r["annot_csl"] for r in pre_rows_l)

        # Print pre
        line = f"{lang:<6}{'pre':<7}"
        for cat in rk_cats:
            line += f"  {pre_rk.get(cat, 0):>10}"
        line += f"  {pre_csl.get('CS', 0):>6}{pre_csl.get('CA', 0):>6}"
        print(line)

        # Print post
        line = f"{'':>6}{'post':<7}"
        for cat in rk_cats:
            line += f"  {post_rk.get(cat, 0):>10}"
        line += f"  {post_csl.get('CS', 0):>6}{post_csl.get('CA', 0):>6}"
        print(line)

        # Delta
        line = f"{'':>6}{'Δ':<7}"
        for cat in rk_cats:
            d = post_rk.get(cat, 0) - pre_rk.get(cat, 0)
            sign = "+" if d > 0 else ""
            line += f"  {sign}{d:>9}"
        d_cs = post_csl.get("CS", 0) - pre_csl.get("CS", 0)
        d_ca = post_csl.get("CA", 0) - pre_csl.get("CA", 0)
        line += f"  {'+' if d_cs>=0 else ''}{d_cs:>5}{'+' if d_ca>=0 else ''}{d_ca:>5}"
        print(line)
        print()

    if latex_dir:
        _write_latex_pre_post(latex_dir, post_rows, pre_csvs, pre_map)


def _write_latex_pre_post(latex_dir, post_rows, pre_csvs, pre_map):
    out = latex_dir / "table_pre_post_shift.tex"
    rk_cats = ["none", "regional", "cultural", "dialect"]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Label distribution shift from pre-review to post-review for \texttt{required\_knowledge}. "
                 r"Arrows indicate net change after the automated review and correction round.}")
    lines.append(r"\label{tab:annot-pre-post}")
    lines.append(r"\begin{tabular}{l l rrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Lang} & \textbf{Stage} & \texttt{none} & \texttt{regional} & \texttt{cultural} & \texttt{dialect} \\")
    lines.append(r"\midrule")

    for lang in TARGET_LANGS:
        post_lr = [r for r in post_rows if r["language"] == lang]
        post_rk = Counter(r["required_knowledge"] for r in post_lr)

        prefix = pre_map[lang]
        pre_path = None
        for p in pre_csvs.get(lang, []):
            if p.stem.startswith(prefix):
                pre_path = p
                break
        if not pre_path:
            continue

        pre_rows_l = load_pre_review_csv(pre_path)
        pre_rk = Counter(r["annot_required_knowledge"] for r in pre_rows_l)

        cells_pre = [lang, "pre"]
        cells_post = ["", "post"]
        for cat in rk_cats:
            cells_pre.append(str(pre_rk.get(cat, 0)))
            cells_post.append(str(post_rk.get(cat, 0)))
        lines.append(" & ".join(cells_pre) + r" \\")
        lines.append(" & ".join(cells_post) + r" \\")

        # Delta row
        cells_d = ["", r"$\Delta$"]
        for cat in rk_cats:
            d = post_rk.get(cat, 0) - pre_rk.get(cat, 0)
            sign = "+" if d > 0 else ""
            cells_d.append(f"{sign}{d}")
        lines.append(" & ".join(cells_d) + r" \\")
        if lang != TARGET_LANGS[-1]:
            lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [LaTeX] Wrote: {out}")


# =====================================================================
#  Table 4 + 5: IAA for DA
# =====================================================================

def table_iaa_danish(pre_csvs: dict[str, list[Path]], latex_dir: Path | None):
    print("\n" + "=" * 90)
    print("  TABLE 4: Inter-Annotator Agreement — Danish (Pre-Review)")
    print("=" * 90)

    da_paths = pre_csvs.get("DA", [])
    if len(da_paths) < 2:
        print("  Not enough DA annotators for IAA.")
        return

    # Load all DA annotators, index by sample_id
    annotators = {}
    for p in da_paths:
        label = p.stem.split("___")[0]
        rows = load_pre_review_csv(p)
        annotators[label] = {r["sample_id"]: r for r in rows}

    ann_names = sorted(annotators.keys())
    print(f"  Annotators: {ann_names}")

    # Common items
    common = set.intersection(*(set(annotators[a].keys()) for a in ann_names))
    common = sorted(common)
    print(f"  Common items: {len(common)}")

    if len(common) == 0:
        print("  ERROR: No common items.")
        return

    fields = [
        ("annot_required_knowledge", "Required knowledge (Q1)"),
        ("annot_csl", "CS/CA"),
        ("annot_subject_category", "Subject category"),
        ("annot_time_sensitive", "Time sensitivity"),
    ]
    pairs = list(combinations(ann_names, 2))

    # ── Pairwise Cohen's κ ──
    print(f"\n  ── Pairwise Cohen's κ (Table 5) ──")
    pairwise_results = []

    for field_key, field_label in fields:
        print(f"\n  {field_label}:")
        kappas = []
        for a1, a2 in pairs:
            vals = []
            for sid in common:
                v1 = annotators[a1].get(sid, {}).get(field_key)
                v2 = annotators[a2].get(sid, {}).get(field_key)
                if v1 and v2:
                    vals.append((v1, v2))
            if vals:
                l1s, l2s = zip(*vals)
                try:
                    from sklearn.metrics import cohen_kappa_score
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        k = cohen_kappa_score(l1s, l2s)
                        if np.isnan(k):
                            k = None  # degenerate case (single class)
                except:
                    k = None
                agree = sum(1 for x, y in vals if x == y)
                pct = agree / len(vals) * 100
                kappas.append(k)
                k_str = f"{k:.3f}" if k is not None else "—"
                print(f"    {a1} vs {a2}: κ={k_str}  agree={agree}/{len(vals)} ({pct:.1f}%)")
                pairwise_results.append({
                    "field": field_label, "a1": a1, "a2": a2,
                    "kappa": k, "agree": agree, "n": len(vals), "pct": pct,
                })
        valid = [k for k in kappas if k is not None]
        if valid:
            print(f"    Mean κ = {np.mean(valid):.3f}")

    # ── Fleiss' κ ──
    print(f"\n  ── Fleiss' κ (Table 4) ──")
    fleiss_results = []

    for field_key, field_label in fields:
        all_labels = set()
        for a in ann_names:
            for sid in common:
                v = annotators[a].get(sid, {}).get(field_key)
                if v:
                    all_labels.add(v)
        all_labels = sorted(all_labels)
        label_to_idx = {l: i for i, l in enumerate(all_labels)}

        n_items = len(common)
        n_cats = len(all_labels)
        n_raters = len(ann_names)
        matrix = np.zeros((n_items, n_cats), dtype=int)

        for i, sid in enumerate(common):
            for a in ann_names:
                v = annotators[a].get(sid, {}).get(field_key)
                if v and v in label_to_idx:
                    matrix[i, label_to_idx[v]] += 1

        p_j = matrix.sum(axis=0) / (n_items * n_raters)
        P_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = np.mean(P_i)
        P_e = np.sum(p_j ** 2)

        if P_e < 1.0:
            fleiss_k = (P_bar - P_e) / (1 - P_e)
        else:
            fleiss_k = 1.0

        print(f"  {field_label}: Fleiss' κ = {fleiss_k:.3f}")
        fleiss_results.append({"field": field_label, "fleiss_k": fleiss_k})

    # ── Agreement breakdown ──
    print(f"\n  ── Agreement Breakdown ──")
    breakdown_results = {}
    for field_key, field_label in [("annot_required_knowledge", "Q1"), ("annot_csl", "CS/CA")]:
        full = partial = none_ = 0
        for sid in common:
            vals = []
            for a in ann_names:
                v = annotators[a].get(sid, {}).get(field_key)
                if v:
                    vals.append(v)
            if len(vals) == len(ann_names):
                if len(set(vals)) == 1:
                    full += 1
                elif len(set(vals)) == len(vals):
                    none_ += 1
                else:
                    partial += 1
        total = full + partial + none_
        print(f"\n  {field_label}:")
        print(f"    Full agreement ({len(ann_names)}/{len(ann_names)}): {full}/{total} ({full/total*100:.1f}%)")
        print(f"    Majority:  {partial}/{total} ({partial/total*100:.1f}%)")
        print(f"    Full disagreement: {none_}/{total} ({none_/total*100:.1f}%)")
        breakdown_results[field_label] = {"full": full, "partial": partial, "none": none_, "total": total}

    if latex_dir:
        _write_latex_iaa(latex_dir, fleiss_results, pairwise_results, breakdown_results, len(common), ann_names)


def _write_latex_iaa(latex_dir, fleiss_results, pairwise_results, breakdown_results, n_common, ann_names):
    # Table 4: Fleiss + breakdown
    out = latex_dir / "table_iaa_fleiss.tex"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Inter-annotator agreement for Danish (" + str(len(ann_names)) +
                 r" annotators, " + str(n_common) + r" common items, pre-review). "
                 r"Fleiss'~$\kappa$ with agreement breakdown for Q1 and CS/CA.}")
    lines.append(r"\label{tab:iaa-da}")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Label} & \textbf{Fleiss' $\kappa$} & \textbf{Full (3/3)} & \textbf{Majority (2/3)} & \textbf{All differ} \\")
    lines.append(r"\midrule")

    for fr in fleiss_results:
        bd = breakdown_results.get(fr["field"].split("(")[1].rstrip(")") if "(" in fr["field"] else fr["field"])
        if bd:
            lines.append(f"{fr['field']} & {fr['fleiss_k']:.3f} & {bd['full']}/{bd['total']} ({bd['full']/bd['total']*100:.0f}\\%) "
                         f"& {bd['partial']}/{bd['total']} ({bd['partial']/bd['total']*100:.0f}\\%) "
                         f"& {bd['none']}/{bd['total']} ({bd['none']/bd['total']*100:.0f}\\%) \\\\")
        else:
            lines.append(f"{fr['field']} & {fr['fleiss_k']:.3f} & --- & --- & --- \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [LaTeX] Wrote: {out}")

    # Table 5: Pairwise Cohen's κ
    out2 = latex_dir / "table_iaa_pairwise.tex"
    lines2 = []
    lines2.append(r"\begin{table}[t]")
    lines2.append(r"\centering")
    lines2.append(r"\small")
    lines2.append(r"\caption{Pairwise Cohen's~$\kappa$ between Danish annotators (pre-review).}")
    lines2.append(r"\label{tab:iaa-da-pairwise}")
    lines2.append(r"\begin{tabular}{l l r r}")
    lines2.append(r"\toprule")
    lines2.append(r"\textbf{Label} & \textbf{Pair} & \textbf{Cohen's $\kappa$} & \textbf{Agreement} \\")
    lines2.append(r"\midrule")

    prev_field = None
    for pr in pairwise_results:
        fname = pr["field"] if pr["field"] != prev_field else ""
        pair = f"{pr['a1'].split('_')[1]}--{pr['a2'].split('_')[1]}"
        k_str = f"{pr['kappa']:.3f}" if pr['kappa'] is not None else "---"
        lines2.append(f"{fname} & {pair} & {k_str} & {pr['agree']}/{pr['n']} ({pr['pct']:.0f}\\%) \\\\")
        prev_field = pr["field"]

    lines2.append(r"\bottomrule")
    lines2.append(r"\end{tabular}")
    lines2.append(r"\end{table}")
    out2.write_text("\n".join(lines2), encoding="utf-8")
    print(f"  [LaTeX] Wrote: {out2}")


# =====================================================================
#  Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--post-review", required=True,
                    help="Path to final Argilla JSONL export (post-review)")
    ap.add_argument("--pre-review-dir", required=True,
                    help="Directory containing per-annotator pre-review CSVs")
    ap.add_argument("--latex-out", default=None,
                    help="Directory for LaTeX table output (optional)")
    args = ap.parse_args()

    post_path = Path(args.post_review)
    pre_dir = Path(args.pre_review_dir)
    latex_dir = Path(args.latex_out) if args.latex_out else None

    if latex_dir:
        latex_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading post-review JSONL...")
    post_rows = [r for r in load_jsonl(post_path) if r["language"] in TARGET_LANGS]
    print(f"  {len(post_rows)} items across {len(TARGET_LANGS)} languages")

    print("Discovering pre-review CSVs...")
    pre_csvs = discover_pre_review_csvs(pre_dir)
    for lang, paths in sorted(pre_csvs.items()):
        print(f"  {lang}: {[p.name for p in paths]}")

    # Generate tables
    table_label_distributions(post_rows, latex_dir)
    table_review_statistics(pre_csvs, latex_dir)
    table_pre_post_comparison(post_rows, pre_csvs, latex_dir)
    table_iaa_danish(pre_csvs, latex_dir)

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()


