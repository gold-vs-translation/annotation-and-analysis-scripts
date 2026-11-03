#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build balanced CA/CS item subsets for Block A (baseline ranking analysis).

Outputs
-------
- analysis_base_metric_<metric>.jsonl
- stats_language.csv
- stats_language_dataset.csv
- stats_language_subject.csv
- subset_item_inventory.csv
- balanced_subsets_items.csv
- balanced_subsets_model_rows.jsonl
- subset_build_report.csv
- subset_build_report_group_summary.csv
- baseline_subset_stats.xlsx

Notes
-----
Balancing is done per selection_group x language.
selected_per_class = min(available_CA, available_CS) within that cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_CA_VALUES = {
    "ca", "culturally_agnostic", "culturally agnostic", "agnostic",
    "none", "no", "not_sensitive", "not sensitive", "neutral", "0"
}
DEFAULT_CS_VALUES = {
    "cs", "culturally_sensitive", "culturally sensitive", "sensitive", "1"
}
DEFAULT_UNKNOWN_VALUES = {"", "unknown", "unsure", "n/a", "na", "null", "none_provided"}


def normalize_label(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def classify_ca_cs(label_raw: Any) -> Optional[str]:
    lab = normalize_label(label_raw)
    if lab in DEFAULT_UNKNOWN_VALUES:
        return None
    if lab in DEFAULT_CA_VALUES:
        return "CA"
    if lab in DEFAULT_CS_VALUES:
        return "CS"
    return "CS"


def pick_sens_col(df: pd.DataFrame) -> str:
    for c in ["cultural_sensitivity_label_agreement", "cultural_sensitivity_label"]:
        if c in df.columns:
            return c
    raise ValueError("No sensitivity label column found. Expected cultural_sensitivity_label[_agreement].")


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
    if not rows:
        raise ValueError("No rows loaded from JSONL.")
    return pd.DataFrame(rows)


def _sanitize_value_for_json(x: Any) -> Any:
    """Convert pandas/numpy values into JSON-serializable Python objects."""
    # Missing values / nulls
    if x is None:
        return None

    # Containers first: pd.isna(list_like) can raise ambiguous truth-value errors.
    if isinstance(x, dict):
        return {str(k): _sanitize_value_for_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [_sanitize_value_for_json(v) for v in list(x)]

    # Pandas / numpy scalars
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    # Scalar NA handling
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    return x


def sanitize_for_jsonl(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_sanitize_value_for_json)
    return out


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sanitize_for_jsonl(df)
    with path.open("w", encoding="utf-8") as f:
        for rec in out.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")




def normalize_sample_score_per_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize score values to 0..1 per value:
      - if 1 < v <= 100 -> v/100
      - else unchanged
    This preserves already-normalized 0..1 scores while fixing percent-like values.
    """
    out = df.copy()
    out["sample_score"] = pd.to_numeric(out["sample_score"], errors="coerce")

    def _norm(v):
        if pd.isna(v):
            return np.nan
        try:
            fv = float(v)
        except Exception:
            return np.nan
        if 1.0 < fv <= 100.0:
            return fv / 100.0
        return fv

    out["sample_score"] = out["sample_score"].map(_norm)
    return out

def parse_preferred_users(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in values:
        if ":" not in raw:
            raise ValueError(f"Invalid --prefer-user-language value '{raw}'. Expected LANG:USER")
        lang, user = raw.split(":", 1)
        out[lang.strip()] = user.strip()
    return out


def apply_metric_filter(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if "metric" not in df.columns:
        raise ValueError("Column 'metric' not found.")
    out = df[df["metric"].astype(str) == str(metric)].copy()
    if out.empty:
        raise ValueError(f"No rows left after filtering metric={metric!r}")
    out = normalize_sample_score_per_value(out)
    out = out.dropna(subset=["sample_score"])
    return out


def apply_preferred_users(df: pd.DataFrame, preferred: Dict[str, str]) -> pd.DataFrame:
    if not preferred:
        return df.copy()
    if "user" not in df.columns or "language" not in df.columns:
        raise ValueError("Preferred-user filtering requires columns 'user' and 'language'.")
    keep_parts: List[pd.DataFrame] = []
    handled_langs = set(preferred.keys())
    for lang, user in preferred.items():
        part = df[(df["language"].astype(str) == lang) & (df["user"].astype(str) == user)].copy()
        keep_parts.append(part)
    keep_parts.append(df[~df["language"].astype(str).isin(handled_langs)].copy())
    out = pd.concat(keep_parts, ignore_index=True)
    if out.empty:
        raise ValueError("No rows left after preferred-user filtering.")
    return out


def add_group_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "task" not in out.columns:
        raise ValueError("Column 'task' not found.")
    task = out["task"].astype(str)
    out["group_name"] = np.where(task.eq("MMLU"), "MMLU-only", "non-MMLU-only")
    return out


def detect_item_id_cols(df: pd.DataFrame) -> List[str]:
    candidates = [
        "task", "suite", "subset", "language", "sample_id",
        "subject_category", "required_knowledge", "time_sensitive",
        "sanity_check_fragment",
    ]
    cols = [c for c in candidates if c in df.columns]
    if not {"language", "sample_id"}.issubset(set(cols)):
        raise ValueError("Need at least language and sample_id to identify items.")
    return cols


def make_item_inventory(df: pd.DataFrame, sens_col: str) -> pd.DataFrame:
    item_id_cols = detect_item_id_cols(df)
    keep_cols = item_id_cols + [sens_col]
    if "user" in df.columns:
        keep_cols.append("user")
    items = df[keep_cols].drop_duplicates().copy()
    items["ca_cs"] = items[sens_col].apply(classify_ca_cs)
    items = add_group_cols(items)
    return items


def add_total_and_pivot(base: pd.DataFrame, index_cols: List[str]) -> pd.DataFrame:
    counts = (
        base.groupby(index_cols + ["ca_cs"], dropna=False)
        .size()
        .reset_index(name="n_items")
    )
    counts["ca_cs"] = counts["ca_cs"].fillna("UNCLASSIFIED")
    wide = counts.pivot_table(
        index=index_cols,
        columns="ca_cs",
        values="n_items",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None
    for col in ["CA", "CS", "UNCLASSIFIED"]:
        if col not in wide.columns:
            wide[col] = 0
    wide["TOTAL"] = wide[["CA", "CS", "UNCLASSIFIED"]].sum(axis=1)
    return wide


def choose_balanced_items(items: pd.DataFrame, group_name: str, include_subject_balance: bool, random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    work = items[items["group_name"] == group_name].copy()
    work = work[work["ca_cs"].isin(["CA", "CS"])]
    if work.empty:
        return pd.DataFrame(columns=list(items.columns) + ["selection_group"]), pd.DataFrame()

    selected_parts: List[pd.DataFrame] = []
    report_rows: List[Dict[str, Any]] = []

    for language, g_lang in work.groupby("language", dropna=False):
        g_ca = g_lang[g_lang["ca_cs"] == "CA"].copy()
        g_cs = g_lang[g_lang["ca_cs"] == "CS"].copy()
        n_ca = len(g_ca)
        n_cs = len(g_cs)
        target = min(n_ca, n_cs)

        report_base = {
            "selection_group": group_name,
            "language": language,
            "available_CA": int(n_ca),
            "available_CS": int(n_cs),
            "selected_per_class": int(target),
            "subject_balance_requested": bool(include_subject_balance),
        }

        if target == 0:
            report_rows.append({**report_base, "selected_CA": 0, "selected_CS": 0, "selected_total": 0, "subject_balance_used": False, "note": "skipped_no_overlap"})
            continue

        def sample_label_frame(label_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
            if not include_subject_balance or "subject_category" not in label_df.columns:
                idx = rng.choice(label_df.index.to_numpy(), size=target, replace=False)
                return label_df.loc[idx].copy(), False
            counts = label_df["subject_category"].fillna("Unknown").value_counts()
            props = counts / counts.sum()
            alloc = np.floor(props * target).astype(int)
            remainder = int(target - alloc.sum())
            if remainder > 0:
                fractional = (props * target) - alloc
                for subj in fractional.sort_values(ascending=False).index[:remainder]:
                    alloc.loc[subj] += 1
            rows = []
            for subj, n_take in alloc.items():
                if n_take <= 0:
                    continue
                part = label_df[label_df["subject_category"].fillna("Unknown") == subj]
                n_take = min(int(n_take), len(part))
                if n_take > 0:
                    idx = rng.choice(part.index.to_numpy(), size=n_take, replace=False)
                    rows.append(part.loc[idx].copy())
            selected = pd.concat(rows, ignore_index=False) if rows else label_df.iloc[0:0].copy()
            missing = target - len(selected)
            if missing > 0:
                leftover = label_df.drop(index=selected.index, errors="ignore")
                if len(leftover) >= missing:
                    idx = rng.choice(leftover.index.to_numpy(), size=missing, replace=False)
                    selected = pd.concat([selected, leftover.loc[idx].copy()], ignore_index=False)
            selected = selected.drop_duplicates()
            if len(selected) > target:
                idx = rng.choice(selected.index.to_numpy(), size=target, replace=False)
                selected = selected.loc[idx].copy()
            return selected.copy(), True

        sel_ca, used_subject_balance = sample_label_frame(g_ca)
        sel_cs, _ = sample_label_frame(g_cs)
        if len(sel_ca) != target or len(sel_cs) != target:
            raise ValueError(f"Selection size mismatch for group={group_name}, language={language}: CA={len(sel_ca)}, CS={len(sel_cs)}, target={target}")

        selected = pd.concat([sel_ca, sel_cs], ignore_index=True)
        selected["selection_group"] = group_name
        selected_parts.append(selected)
        report_rows.append({
            **report_base,
            "selected_CA": int(len(sel_ca)),
            "selected_CS": int(len(sel_cs)),
            "selected_total": int(len(selected)),
            "subject_balance_used": bool(used_subject_balance),
            "note": "ok",
        })

    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=list(items.columns) + ["selection_group"])
    report_df = pd.DataFrame(report_rows)
    return selected_df, report_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Path to merged analysis JSONL")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--metric", default="acc", type=str, help="Metric to analyze (default: acc)")
    ap.add_argument("--prefer-user-language", action="append", default=[], help="Optional LANG:USER override, repeatable. Example: --prefer-user-language DA:annotator_1")
    ap.add_argument("--include-all-group", action="store_true", help="Also build an ALL group (optional).")
    ap.add_argument("--balance-by-subject", action="store_true", help="Try proportional subject balancing within each label.")
    ap.add_argument(
        "--exclude-tasks",
        default="Belebele",
        type=str,
        help="Comma-separated task names to exclude entirely (default: Belebele)",
    )
    ap.add_argument("--seed", default=42, type=int, help="Random seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(Path(args.input))
    df = apply_metric_filter(df, args.metric)
    preferred = parse_preferred_users(args.prefer_user_language)
    df = apply_preferred_users(df, preferred)

    # Exclude tasks not relevant for ranking analysis (e.g. Belebele = language proficiency, not exam)
    if args.exclude_tasks:
        exclude_set = {t.strip() for t in args.exclude_tasks.split(",") if t.strip()}
        if exclude_set and "task" in df.columns:
            n_before = len(df)
            df = df[~df["task"].astype(str).isin(exclude_set)].copy()
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                print(f"[INFO] Excluded {n_dropped} rows from tasks: {sorted(exclude_set)}")

    df = add_group_cols(df)

    sens_col = pick_sens_col(df)
    item_inventory = make_item_inventory(df, sens_col=sens_col)

    write_jsonl(df, out_dir / f"analysis_base_metric_{args.metric}.jsonl")

    stats_language = add_total_and_pivot(item_inventory, ["language"])
    stats_language_dataset = add_total_and_pivot(item_inventory, ["language", "suite"])
    subject_cols = ["language"] + (["subject_category"] if "subject_category" in item_inventory.columns else [])
    stats_language_subject = add_total_and_pivot(item_inventory, subject_cols)

    stats_language.to_csv(out_dir / "stats_language.csv", index=False)
    stats_language_dataset.to_csv(out_dir / "stats_language_dataset.csv", index=False)
    stats_language_subject.to_csv(out_dir / "stats_language_subject.csv", index=False)
    item_inventory.to_csv(out_dir / "subset_item_inventory.csv", index=False)

    groups = ["MMLU-only", "non-MMLU-only"]
    if args.include_all_group:
        all_items = item_inventory.copy()
        all_items["group_name"] = "ALL"
        item_inventory_for_select = pd.concat([item_inventory, all_items], ignore_index=True)
        groups.append("ALL")
    else:
        item_inventory_for_select = item_inventory

    selected_parts: List[pd.DataFrame] = []
    report_parts: List[pd.DataFrame] = []
    for i, group_name in enumerate(groups):
        selected, report = choose_balanced_items(
            items=item_inventory_for_select,
            group_name=group_name,
            include_subject_balance=bool(args.balance_by_subject),
            random_seed=args.seed + i,
        )
        if not selected.empty:
            selected_parts.append(selected)
        if not report.empty:
            report_parts.append(report)

    selected_items = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    build_report = pd.concat(report_parts, ignore_index=True) if report_parts else pd.DataFrame()

    if selected_items.empty:
        raise ValueError("No balanced subsets could be built.")

    item_id_cols = detect_item_id_cols(df)
    selected_keys = selected_items[item_id_cols + ["selection_group", "ca_cs"]].drop_duplicates()
    balanced_rows = df.merge(selected_keys, on=item_id_cols, how="inner")

    selected_items.to_csv(out_dir / "balanced_subsets_items.csv", index=False)
    write_jsonl(balanced_rows, out_dir / "balanced_subsets_model_rows.jsonl")
    build_report.to_csv(out_dir / "subset_build_report.csv", index=False)

    if not build_report.empty:
        group_summary = build_report.groupby("selection_group", dropna=False).agg(
            n_languages=("language", "nunique"),
            total_available_CA=("available_CA", "sum"),
            total_available_CS=("available_CS", "sum"),
            total_selected_CA=("selected_CA", "sum"),
            total_selected_CS=("selected_CS", "sum"),
            total_selected_items=("selected_total", "sum"),
        ).reset_index()
        group_summary.to_csv(out_dir / "subset_build_report_group_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "subset_build_report_group_summary.csv", index=False)

    xlsx = out_dir / "baseline_subset_stats.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        stats_language.to_excel(writer, sheet_name="stats_language", index=False)
        stats_language_dataset.to_excel(writer, sheet_name="stats_language_dataset", index=False)
        stats_language_subject.to_excel(writer, sheet_name="stats_language_subject", index=False)
        build_report.to_excel(writer, sheet_name="subset_build_report", index=False)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()


