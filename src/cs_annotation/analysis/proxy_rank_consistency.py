#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Block B: native-vs-proxy ranking consistency.

Design
------
- reads the merged JSONL analysis file
- filters to one metric (default: acc)
- optionally keeps preferred annotators for selected languages
- auto-builds native/proxy pairs from task + language
  * proxy task defaults to MMLU
  * for each language with proxy data, all non-proxy suites in that language
    become native candidates
- computes Table A (scores + ranks) and Table B (Kendall tau-b summary)
- reports variants: ALL, CA, CS

Notes
-----
- CA/CS splits use the same CA/CS labels already present in the input rows.
- If no pairs can be built automatically, the script raises a clear error and
  writes nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from typing import List, Set
import numpy as np
import pandas as pd


# -----------------------------
# I/O helpers
# -----------------------------

def read_jsonl(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        pass

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError("No rows loaded from JSONL.")

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
    except Exception:
        pass

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

# -----------------------------
# CA/CS helpers
# -----------------------------

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
    if lab:
        return "CS"
    return None


def pick_sens_col(df: pd.DataFrame, override: Optional[str] = None) -> Optional[str]:
    if override:
        if override not in df.columns:
            raise ValueError(f"--sens-col '{override}' not found in columns.")
        return override
    for c in ["cultural_sensitivity_label_agreement", "cultural_sensitivity_label"]:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Pair specification
# -----------------------------

def auto_build_pairs(df: pd.DataFrame, proxy_task: str = "MMLU", proxy_languages: Optional[List[str]] = None, exclude_tasks: Optional[List[str]] = None) -> pd.DataFrame:
    required = {"task", "suite", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"auto_build_pairs requires columns: {sorted(missing)}")

    work = df.copy()
    work["task"] = work["task"].astype(str)
    work["suite"] = work["suite"].astype(str)
    work["language"] = work["language"].astype(str)

    # Exclude tasks that should not appear as native benchmarks (e.g. Belebele)
    _exclude = set(exclude_tasks or [])
    _exclude.add(proxy_task)  # proxy task is never native

    proxy_rows = work[work["task"] == str(proxy_task)].copy()
    if proxy_languages:
        proxy_rows = proxy_rows[proxy_rows["language"].isin(proxy_languages)].copy()

    rows: List[Dict[str, str]] = []
    for lang in sorted(proxy_rows["language"].dropna().unique().tolist()):
        proxy_suites = sorted(proxy_rows.loc[proxy_rows["language"] == lang, "suite"].dropna().unique().tolist())
        if not proxy_suites:
            continue
        # usually one proxy suite per language; keep all if multiple exist
        native_suites = sorted(
            work.loc[
                (work["language"] == lang) &
                (~work["task"].isin(_exclude)) &
                (~work["suite"].isin(proxy_suites)),
                "suite"
            ].dropna().unique().tolist()
        )
        for proxy_suite in proxy_suites:
            for native_suite in native_suites:
                rows.append({
                    "language": lang,
                    "suite_native": native_suite,
                    "suite_proxy": proxy_suite,
                })

    cols = ["language", "suite_native", "suite_proxy"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).drop_duplicates().sort_values(cols).reset_index(drop=True)


# -----------------------------
# Ranking helpers
# -----------------------------

def rank_desc_with_ties(scores: pd.Series) -> pd.Series:
    return scores.rank(method="dense", ascending=False)


def bootstrap_kendall_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return np.nan, np.nan

    vals: List[float] = []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        xb = x[idx]
        yb = y[idx]
        try:
            tau, _ = kendalltau(xb, yb, variant="b")
        except TypeError:
            tau, _ = kendalltau(xb, yb)
        if pd.notna(tau):
            vals.append(float(tau))

    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def topk_overlap(rank_a: pd.Series, rank_b: pd.Series, k: int, model_names: pd.Series) -> dict:
    """
    Compute top-k overlap between two rankings.

    Parameters
    ----------
    rank_a, rank_b : pd.Series
        Dense ranks (1 = best). Must be aligned with model_names.
    model_names : pd.Series
        Model identifiers, aligned with rank_a and rank_b.
    k : int
        Number of top positions to compare.

    Returns
    -------
    dict with keys:
        top_k: int
        overlap_count: int
        overlap_fraction: float (0..1)
        models_a: sorted list of model names in top-k of ranking A
        models_b: sorted list of model names in top-k of ranking B
        overlap_models: sorted list of models in both top-k sets
    """
    if k <= 0 or len(rank_a) == 0:
        return {
            "top_k": k,
            "overlap_count": 0,
            "overlap_fraction": np.nan,
            "models_a": [],
            "models_b": [],
            "overlap_models": [],
        }

    # Effective k: can't exceed number of models
    k_eff = min(k, len(rank_a))

    set_a: Set[str] = set(model_names[rank_a <= k_eff].tolist())
    set_b: Set[str] = set(model_names[rank_b <= k_eff].tolist())
    overlap: Set[str] = set_a & set_b

    return {
        "top_k": k_eff,
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / k_eff if k_eff > 0 else np.nan,
        "models_a": sorted(set_a),
        "models_b": sorted(set_b),
        "overlap_models": sorted(overlap),
    }


# -----------------------------
# Core computation
# -----------------------------

def compute_pair_tables(
    df: pd.DataFrame,
    language: str,
    suite_native: str,
    suite_proxy: str,
    subset: str,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df[df["language"].astype(str) == language].copy()
    if d.empty:
        raise ValueError(f"No rows found for language={language!r}")

    if subset in {"CA", "CS"}:
        if "ca_cs" not in d.columns:
            raise ValueError("Subset filtering requires column 'ca_cs'.")
        d = d[d["ca_cs"] == subset].copy()

    dn = d[d["suite"].astype(str) == suite_native].copy()
    dp = d[d["suite"].astype(str) == suite_proxy].copy()

    if dn.empty or dp.empty:
        raise ValueError(
            f"No rows found for pair ({language}, {suite_native}, {suite_proxy}) under subset={subset}."
        )

    native = (
        dn.groupby("model_name", dropna=False)["sample_score"]
        .mean()
        .rename("score_native")
        .reset_index()
    )
    proxy = (
        dp.groupby("model_name", dropna=False)["sample_score"]
        .mean()
        .rename("score_proxy")
        .reset_index()
    )

    table_a = native.merge(proxy, on="model_name", how="inner")
    table_a["rank_native"] = rank_desc_with_ties(table_a["score_native"])
    table_a["rank_proxy"] = rank_desc_with_ties(table_a["score_proxy"])
    table_a.insert(0, "subset", subset)
    table_a.insert(0, "language", language)
    table_a.insert(0, "suite_proxy", suite_proxy)
    table_a.insert(0, "suite_native", suite_native)
    table_a = table_a.sort_values(["rank_native", "model_name"], ascending=[True, True]).reset_index(drop=True)

    x = table_a["score_native"].to_numpy(dtype=float)
    y = table_a["score_proxy"].to_numpy(dtype=float)

    if len(table_a) >= 2:
        try:
            tau, p_value = kendalltau(x, y, variant="b")
        except TypeError:
            tau, p_value = kendalltau(x, y)
        ci_low, ci_high = bootstrap_kendall_ci(x, y, n_boot=n_boot, seed=seed)
    else:
        tau, p_value, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan

    summary = {
        "suite_native": suite_native,
        "suite_proxy": suite_proxy,
        "language": language,
        "subset": subset,
        "n_models": int(len(table_a)),
        "n_items_native": int(dn["sample_id"].nunique()) if "sample_id" in dn.columns else int(len(dn)),
        "n_items_proxy": int(dp["sample_id"].nunique()) if "sample_id" in dp.columns else int(len(dp)),
        "kendall_tau_b": float(tau) if pd.notna(tau) else np.nan,
        "ci_low": float(ci_low) if pd.notna(ci_low) else np.nan,
        "ci_high": float(ci_high) if pd.notna(ci_high) else np.nan,
        "p_value": float(p_value) if pd.notna(p_value) else np.nan,
    }

    # Top-k overlap
    for k in [3, 5]:
        tk = topk_overlap(
            table_a["rank_native"], table_a["rank_proxy"], k=k,
            model_names=table_a["model_name"],
        )
        summary[f"top{k}_overlap_count"] = tk["overlap_count"]
        summary[f"top{k}_overlap_fraction"] = tk["overlap_fraction"]

    return table_a, summary


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Path to merged JSONL analysis file")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--metric", default="acc", type=str, help="Metric to select")
    ap.add_argument(
        "--prefer-user-language",
        action="append",
        default=[],
        help="Optional preferred annotator for a language, e.g. DA:annotator_1. May be passed multiple times.",
    )
    ap.add_argument("--proxy-task", default="MMLU", type=str, help="Task value treated as translated proxy task")
    ap.add_argument(
        "--proxy-languages",
        default=None,
        type=str,
        help="Optional comma-separated subset of proxy languages to consider, e.g. ET,SV,DA",
    )
    ap.add_argument("--sens-col", default=None, type=str, help="Optional CA/CS label column override")
    ap.add_argument(
        "--exclude-tasks",
        default="Belebele",
        type=str,
        help="Comma-separated task names to exclude from native side (default: Belebele)",
    )
    ap.add_argument("--n_boot", default=2000, type=int, help="Bootstrap iterations for CI")
    ap.add_argument("--seed", default=42, type=int, help="Random seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(Path(args.input))
    df = apply_metric_filter(df, args.metric)
    preferred = parse_preferred_users(args.prefer_user_language)
    df = apply_preferred_users(df, preferred)

    required_cols = {"language", "suite", "task", "model_name", "sample_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    sens_col = pick_sens_col(df, override=args.sens_col)
    if sens_col is not None:
        df["ca_cs"] = df[sens_col].apply(classify_ca_cs)
    else:
        df["ca_cs"] = None

    langs = None
    if args.proxy_languages:
        langs = [x.strip() for x in args.proxy_languages.split(",") if x.strip()]

    exclude_tasks = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()] if args.exclude_tasks else []
    pairs = auto_build_pairs(df, proxy_task=args.proxy_task, proxy_languages=langs, exclude_tasks=exclude_tasks)
    if pairs.empty:
        raise ValueError(
            f"No proxy/native pairs could be auto-built. Check task/suite/language values, "
            f"proxy_task={args.proxy_task!r}, exclude_tasks={exclude_tasks!r}."
        )

    pairs.to_csv(out_dir / "proxy_pairs_used.csv", index=False)

    table_a_parts: List[pd.DataFrame] = []
    table_b_rows: List[Dict[str, Any]] = []

    subsets = ["ALL", "CA", "CS"]
    for i, spec in pairs.reset_index(drop=True).iterrows():
        for j, subset in enumerate(subsets):
            try:
                table_a, summary = compute_pair_tables(
                    df=df,
                    language=str(spec["language"]),
                    suite_native=str(spec["suite_native"]),
                    suite_proxy=str(spec["suite_proxy"]),
                    subset=subset,
                    n_boot=args.n_boot,
                    seed=args.seed + (i * 10) + j,
                )
            except ValueError:
                continue
            table_a_parts.append(table_a)
            table_b_rows.append(summary)

    table_a_df = pd.concat(table_a_parts, ignore_index=True) if table_a_parts else pd.DataFrame()
    table_b_df = pd.DataFrame(table_b_rows)
    if not table_b_df.empty:
        table_b_df = table_b_df.sort_values(["language", "suite_native", "suite_proxy", "subset"]).reset_index(drop=True)

    table_a_df.to_csv(out_dir / "proxy_rank_table_A.csv", index=False)
    table_b_df.to_csv(out_dir / "proxy_rank_table_B.csv", index=False)

    with pd.ExcelWriter(out_dir / "proxy_rank_tables.xlsx", engine="openpyxl") as writer:
        table_a_df.to_excel(writer, sheet_name="table_A", index=False)
        table_b_df.to_excel(writer, sheet_name="table_B", index=False)

    print(f"Saved Block B outputs to {out_dir}")


if __name__ == "__main__":
    main()


