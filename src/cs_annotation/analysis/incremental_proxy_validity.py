#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Incremental Proxy Validity Analysis.

Steps
-----
1. Aggregate model-level scores from merged JSONL
2. OLS regression: baseline vs extended (per pairing)
3. CS/CA-stratified regressions (mechanism analysis)
4. Robustness:
   a) Partial/residual correlations
   b) Leave-one-family-out sensitivity

Usage
-----
python incremental_proxy_validity.py \
    --input merged_samplelogs_nativ_globalmmlu_belebele_ann_slim.jsonl \
    --out_dir ./analysis/incremental/ \
    --metric acc

Dependencies: numpy, pandas, scipy, statsmodels
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings("ignore", category=FutureWarning)


# model_name → (active_params_B, family)
MODEL_METADATA: Dict[str, Tuple[float, str]] = {
    # M
    "c4ai-command-a-03-2025-111B-Instruct": (111.0, "Cohere"),
    "Apertus-70B-Instruct-2509":            (70.0,  "Apertus"),
    "Llama-4-Scout-17B-16E-Instruct":       (17.0,  "Llama"),
    "Mistral-Large-Instruct-2411":          (123.0, "Mistral"),
    "Llama-3.1-70B-Instruct":              (70.0,  "Llama"),
    "Qwen2.5-72B-Instruct":                (72.0,  "Qwen"),
    "Xwen-72B-Chat":                        (72.0,  "Xwen"),
    # S
    "aya-expanse-32b-Instruct":             (32.0,  "Cohere"),
    "Phi-4-15B-Instruct":                   (14.7,  "Microsoft"),
    "gpt-oss-20b":                          (3.6,   "OpenAI"),
    "Nemotron-3-Nano-30B-A3B-BF16":         (3.0,   "NVIDIA"),
    "Qwen3-30B-A3B-Instruct-2507":          (3.0,   "Qwen"),
    "EuroLLM-22B-Instruct-Preview":         (22.0,  "EuroLLM"),
    "Mistral-Small-24B-Instruct-2501":      (24.0,  "Mistral"),
    "gemma-3-27b-it":                       (27.0,  "Google"),
    # XS
    "salamandra-7b-instruct":               (7.0,   "BSC"),
    "Teuken-7B-instruct-v0.6":             (7.0,   "openGPT-X"),
    "Llama-3.1-8B-Instruct":               (8.0,   "Llama"),
    "Qwen3-8B":                             (8.0,   "Qwen"),
    "Falcon3-7B-Instruct":                  (7.0,   "TII"),
    "EuroLLM-9B-Instruct":                 (9.0,   "EuroLLM"),
    "aya-expanse-8b":                       (8.0,   "Cohere"),
    # XXS
    "SmolLM2-360M-Instruct":               (0.36,  "HuggingFace"),
    "Qwen3-0.6B":                           (0.6,   "Qwen"),
    "LFM2-700M":                            (0.7,   "Liquid"),
    "gemma-3-1b-it":                        (1.0,   "Google"),
    "Falcon3-1B-Instruct":                  (1.67,  "TII"),
    "Llama-3.2-1B-Instruct":               (1.24,  "Llama"),
    "bloomz-560m":                          (0.56,  "BigScience"),
}


def lookup_model_meta(model_name: str) -> Tuple[float, str]:
    """Look up active params and family. Falls back to substring match."""
    if model_name in MODEL_METADATA:
        return MODEL_METADATA[model_name]
    # Substring fallback
    for key, val in MODEL_METADATA.items():
        if key in model_name or model_name in key:
            return val
    return (np.nan, "Unknown")


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
        raise ValueError("No rows loaded.")
    return pd.DataFrame(rows)


def normalize_score(v):
    if pd.isna(v):
        return np.nan
    try:
        fv = float(v)
    except Exception:
        return np.nan
    if 1.0 < fv <= 100.0:
        return fv / 100.0
    return fv


_CA = {"ca", "culturally_agnostic", "culturally agnostic", "agnostic",
       "none", "no", "not_sensitive", "not sensitive", "neutral", "0"}
_CS = {"cs", "culturally_sensitive", "culturally sensitive", "sensitive", "1"}


def classify_ca_cs(label) -> Optional[str]:
    if label is None:
        return None
    lab = str(label).strip().lower()
    if lab in ("", "unknown", "unsure", "n/a", "na", "null", "none_provided"):
        return None
    if lab in _CA:
        return "CA"
    if lab in _CS:
        return "CS"
    if lab:
        return "CS"
    return None


def aggregate_model_scores(
    df: pd.DataFrame,
    proxy_task: str = "MMLU",
    belebele_task: str = "Belebele",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-model accuracy scores.

    Returns
    -------
    scores_all : DataFrame
        Columns: model_name, task, suite, language, score_all, score_ca, score_cs, n_items, n_ca, n_cs
    model_meta : DataFrame
        Columns: model_name, log_params, family
    """
    # Mean accuracy per (model, suite, language) — overall, CA, CS
    groups = df.groupby(["model_name", "task", "suite", "language"], dropna=False)

    rows = []
    for (model, task, suite, lang), g in groups:
        score_all = g["sample_score"].mean()
        n_items = len(g)

        g_ca = g[g["ca_cs"] == "CA"]
        g_cs = g[g["ca_cs"] == "CS"]
        score_ca = g_ca["sample_score"].mean() if len(g_ca) > 0 else np.nan
        score_cs = g_cs["sample_score"].mean() if len(g_cs) > 0 else np.nan

        rows.append({
            "model_name": model,
            "task": str(task),
            "suite": str(suite),
            "language": str(lang),
            "score_all": float(score_all),
            "score_ca": float(score_ca) if pd.notna(score_ca) else np.nan,
            "score_cs": float(score_cs) if pd.notna(score_cs) else np.nan,
            "n_items": int(n_items),
            "n_ca": int(len(g_ca)),
            "n_cs": int(len(g_cs)),
        })

    scores = pd.DataFrame(rows)

    # Model metadata
    models = sorted(df["model_name"].unique())
    meta_rows = []
    for m in models:
        params, family = lookup_model_meta(m)
        meta_rows.append({
            "model_name": m,
            "active_params_B": params,
            "log_params": float(np.log(params)) if pd.notna(params) and params > 0 else np.nan,
            "family": family,
        })
    model_meta = pd.DataFrame(meta_rows)

    return scores, model_meta


def build_regression_df(
    scores: pd.DataFrame,
    model_meta: pd.DataFrame,
    suite_native: str,
    suite_proxy: str,
    language: str,
    belebele_task: str = "Belebele",
    proxy_task: str = "MMLU",
    proxy_suite: str = "GMMLUX",
) -> pd.DataFrame:
    """
    Build a model-level DataFrame for one native-proxy pairing.

    Columns: model_name, log_params, family, belebele, proxy_all, proxy_ca, proxy_cs,
             native_all, native_ca, native_cs, n_native, n_proxy, gmmlux_en
    """
    # Native scores
    nat = scores[(scores["suite"] == suite_native) & (scores["language"] == language)].copy()
    nat = nat.rename(columns={
        "score_all": "native_all", "score_ca": "native_ca", "score_cs": "native_cs",
        "n_items": "n_native", "n_ca": "n_native_ca", "n_cs": "n_native_cs",
    })[["model_name", "native_all", "native_ca", "native_cs", "n_native", "n_native_ca", "n_native_cs"]]

    # Proxy scores
    prx = scores[(scores["suite"] == suite_proxy) & (scores["language"] == language)].copy()
    prx = prx.rename(columns={
        "score_all": "proxy_all", "score_ca": "proxy_ca", "score_cs": "proxy_cs",
        "n_items": "n_proxy",
    })[["model_name", "proxy_all", "proxy_ca", "proxy_cs", "n_proxy"]]

    # Belebele scores (language proficiency control)
    bel = scores[(scores["task"] == belebele_task) & (scores["language"] == language)].copy()
    if bel.empty:
        bel = scores[scores["task"] == belebele_task].copy()
    bel = bel.rename(columns={"score_all": "belebele"})[["model_name", "belebele"]]
    bel = bel.groupby("model_name", dropna=False)["belebele"].mean().reset_index()

    # GMMLUX English scores (general ability control for sensitivity analysis)
    gen = scores[(scores["suite"] == proxy_suite) & (scores["language"] == "EN")].copy()
    if not gen.empty:
        gen = gen.rename(columns={"score_all": "gmmlux_en"})[["model_name", "gmmlux_en"]]
        gen = gen.groupby("model_name", dropna=False)["gmmlux_en"].mean().reset_index()
    else:
        gen = pd.DataFrame(columns=["model_name", "gmmlux_en"])

    # Merge
    reg = nat.merge(prx, on="model_name", how="inner")
    reg = reg.merge(bel, on="model_name", how="left")
    reg = reg.merge(gen, on="model_name", how="left")
    reg = reg.merge(model_meta, on="model_name", how="left")

    return reg


def run_incremental_regression(
    reg: pd.DataFrame,
    target_col: str = "native_all",
    proxy_col: str = "proxy_all",
    control_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run baseline vs. extended OLS regression.

    Model 1 (baseline): target ~ controls
    Model 2 (extended): target ~ controls + proxy

    Returns summary dict.
    """
    if control_cols is None:
        control_cols = ["log_params", "belebele"]
    cols_base = list(control_cols)
    cols_ext = list(control_cols) + [proxy_col]

    # Drop rows with missing values
    needed = [target_col] + cols_ext
    clean = reg.dropna(subset=needed).copy()
    n = len(clean)

    if n < 5:
        return _empty_regression_result(target_col, proxy_col, n, control_cols=control_cols)

    y = clean[target_col].values
    X_base = sm.add_constant(clean[cols_base].values)
    X_ext = sm.add_constant(clean[cols_ext].values)

    try:
        m1 = sm.OLS(y, X_base).fit()
        m2 = sm.OLS(y, X_ext).fit()
    except Exception as e:
        return _empty_regression_result(target_col, proxy_col, n, note=str(e), control_cols=control_cols)

    # F-test for nested models
    try:
        f_result = anova_lm(m1, m2)
        f_stat = f_result["F"].iloc[1]
        f_pval = f_result["Pr(>F)"].iloc[1]
    except Exception:
        f_stat, f_pval = np.nan, np.nan

    # Proxy coefficient is the last one in m2
    proxy_idx = -1  # last predictor
    beta_proxy = m2.params[proxy_idx]
    se_proxy = m2.bse[proxy_idx]
    t_proxy = m2.tvalues[proxy_idx]
    p_proxy = m2.pvalues[proxy_idx]

    return {
        "target": target_col,
        "proxy_col": proxy_col,
        "n": int(n),
        "R2_base": float(m1.rsquared),
        "R2_adj_base": float(m1.rsquared_adj),
        "AIC_base": float(m1.aic),
        "BIC_base": float(m1.bic),
        "R2_ext": float(m2.rsquared),
        "R2_adj_ext": float(m2.rsquared_adj),
        "AIC_ext": float(m2.aic),
        "BIC_ext": float(m2.bic),
        "delta_R2": float(m2.rsquared - m1.rsquared),
        "delta_R2_adj": float(m2.rsquared_adj - m1.rsquared_adj),
        "F_stat": float(f_stat) if pd.notna(f_stat) else np.nan,
        "F_pval": float(f_pval) if pd.notna(f_pval) else np.nan,
        "beta_proxy": float(beta_proxy),
        "se_proxy": float(se_proxy),
        "t_proxy": float(t_proxy),
        "p_proxy": float(p_proxy),
        "controls": "+".join(control_cols),
        "note": "",
    }
    # Control coefficients from extended model (const + controls + proxy)
    result["beta_const"] = float(m2.params[0])
    for i, col_name in enumerate(control_cols):
        result[f"beta_{col_name}"] = float(m2.params[1 + i])
    return result


def _empty_regression_result(target, proxy_col, n, note="insufficient data", control_cols=None):
    keys = [
        "target", "proxy_col", "n",
        "R2_base", "R2_adj_base", "AIC_base", "BIC_base",
        "R2_ext", "R2_adj_ext", "AIC_ext", "BIC_ext",
        "delta_R2", "delta_R2_adj", "F_stat", "F_pval",
        "beta_proxy", "se_proxy", "t_proxy", "p_proxy",
        "controls", "beta_const", "note",
    ]
    result = {k: np.nan for k in keys}
    result["target"] = target
    result["proxy_col"] = proxy_col
    result["n"] = int(n)
    result["controls"] = "+".join(control_cols) if control_cols else ""
    result["note"] = note
    return result


def partial_correlation(
    reg: pd.DataFrame,
    target_col: str = "native_all",
    proxy_col: str = "proxy_all",
    control_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Residualize target and proxy on controls, then correlate residuals.
    """
    if control_cols is None:
        control_cols = ["log_params", "belebele"]
    controls = list(control_cols)
    needed = [target_col, proxy_col] + controls
    clean = reg.dropna(subset=needed).copy()
    n = len(clean)

    if n < 5:
        return {"target": target_col, "proxy_col": proxy_col, "n": n,
                "partial_r": np.nan, "partial_p": np.nan, "note": "insufficient data"}

    X_ctrl = sm.add_constant(clean[controls].values)
    y_nat = clean[target_col].values
    y_prx = clean[proxy_col].values

    try:
        resid_nat = sm.OLS(y_nat, X_ctrl).fit().resid
        resid_prx = sm.OLS(y_prx, X_ctrl).fit().resid
        r, p = pearsonr(resid_nat, resid_prx)
    except Exception as e:
        return {"target": target_col, "proxy_col": proxy_col, "n": n,
                "partial_r": np.nan, "partial_p": np.nan, "note": str(e)}

    return {
        "target": target_col,
        "proxy_col": proxy_col,
        "n": int(n),
        "partial_r": float(r),
        "partial_p": float(p),
        "note": "",
    }


def leave_one_family_out(
    reg: pd.DataFrame,
    target_col: str = "native_all",
    proxy_col: str = "proxy_all",
    control_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    For each model family, remove all its models and re-run Model 2.
    """
    if control_cols is None:
        control_cols = ["log_params", "belebele"]
    if "family" not in reg.columns:
        return []

    families = sorted(reg["family"].dropna().unique())
    results = []

    for fam in families:
        members = reg[reg["family"] == fam]["model_name"].tolist()
        n_members = len(members)
        subset = reg[reg["family"] != fam].copy()
        n_remain = len(subset.dropna(subset=[target_col, proxy_col] + control_cols))

        if n_remain < 5:
            results.append({
                "excluded_family": fam,
                "n_excluded": n_members,
                "n_remaining": n_remain,
                "beta_proxy": np.nan,
                "se_proxy": np.nan,
                "p_proxy": np.nan,
                "R2_ext": np.nan,
                "delta_R2": np.nan,
                "note": "too few remaining models",
            })
            continue

        res = run_incremental_regression(subset, target_col=target_col, proxy_col=proxy_col,
                                         control_cols=control_cols)

        results.append({
            "excluded_family": fam,
            "n_excluded": int(n_members),
            "n_remaining": int(res["n"]),
            "beta_proxy": res["beta_proxy"],
            "se_proxy": res["se_proxy"],
            "p_proxy": res["p_proxy"],
            "R2_ext": res["R2_ext"],
            "delta_R2": res["delta_R2"],
            "note": res.get("note", ""),
        })

    return results


def auto_build_pairs(
    scores: pd.DataFrame,
    proxy_task: str = "MMLU",
    exclude_tasks: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build native-proxy pairs from the aggregated scores."""
    _exclude = set(exclude_tasks or [])
    _exclude.add(proxy_task)

    proxy_rows = scores[scores["task"] == proxy_task].copy()
    native_rows = scores[~scores["task"].isin(_exclude)].copy()

    pairs = []
    for lang in sorted(proxy_rows["language"].dropna().unique()):
        proxy_suites = sorted(proxy_rows.loc[proxy_rows["language"] == lang, "suite"].unique())
        native_suites = sorted(native_rows.loc[native_rows["language"] == lang, "suite"].unique())

        for ps in proxy_suites:
            for ns in native_suites:
                if ns != ps:
                    pairs.append({
                        "language": lang,
                        "suite_native": ns,
                        "suite_proxy": ps,
                    })

    cols = ["language", "suite_native", "suite_proxy"]
    return pd.DataFrame(pairs, columns=cols).drop_duplicates().sort_values(cols).reset_index(drop=True) if pairs else pd.DataFrame(columns=cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="Incremental Proxy Validity Analysis")
    ap.add_argument("--input", required=True, type=str, help="Path to merged JSONL")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--metric", default="acc", type=str, help="Metric to filter")
    ap.add_argument("--proxy-task", default="MMLU", type=str, help="Proxy task name")
    ap.add_argument("--belebele-task", default="Belebele", type=str, help="Belebele task name")
    ap.add_argument("--exclude-tasks", default="Belebele", type=str,
                     help="Comma-separated tasks to exclude from native side (default: Belebele)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_tasks = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()] if args.exclude_tasks else []

    # --- Load and preprocess ---
    print("[1/7] Loading data...")
    df = read_jsonl(Path(args.input))
    df = df[df["metric"].astype(str) == args.metric].copy()
    df["sample_score"] = pd.to_numeric(df["sample_score"], errors="coerce").map(normalize_score)
    df = df.dropna(subset=["sample_score"])

    # CA/CS labels
    for col in ["cultural_sensitivity_label_agreement", "cultural_sensitivity_label"]:
        if col in df.columns:
            df["ca_cs"] = df[col].apply(classify_ca_cs)
            break
    else:
        df["ca_cs"] = None

    print(f"  {len(df)} rows, {df['model_name'].nunique()} models")

    # --- Step 1: Aggregate ---
    print("[2/7] Aggregating model-level scores...")
    scores, model_meta = aggregate_model_scores(df, proxy_task=args.proxy_task, belebele_task=args.belebele_task)

    scores.to_csv(out_dir / "model_scores_per_suite_language.csv", index=False)
    model_meta.to_csv(out_dir / "model_metadata.csv", index=False)

    # Check metadata coverage
    n_missing = model_meta["log_params"].isna().sum()
    if n_missing > 0:
        missing = model_meta[model_meta["log_params"].isna()]["model_name"].tolist()
        print(f"  [WARN] {n_missing} models without metadata: {missing}")

    # --- Build pairs ---
    print("[3/7] Building native-proxy pairs...")
    pairs = auto_build_pairs(scores, proxy_task=args.proxy_task, exclude_tasks=exclude_tasks)
    if pairs.empty:
        raise ValueError("No pairs could be built.")
    pairs.to_csv(out_dir / "incremental_pairs_used.csv", index=False)
    print(f"  {len(pairs)} pairings")

    # --- Steps 2-4: Run analyses per pairing ---
    print("[4/7] Running incremental regressions...")
    regression_rows = []
    partial_rows = []
    lofo_rows = []

    for _, spec in pairs.iterrows():
        lang = str(spec["language"])
        s_nat = str(spec["suite_native"])
        s_prx = str(spec["suite_proxy"])
        label = f"{s_nat} ({lang})"

        reg = build_regression_df(
            scores, model_meta,
            suite_native=s_nat, suite_proxy=s_prx, language=lang,
            belebele_task=args.belebele_task,
        )

        if reg.empty or len(reg) < 5:
            print(f"  [SKIP] {label}: insufficient models ({len(reg)})")
            continue

        # --- Step 2: Main regressions (ALL, CA, CS) ---
        for target, proxy, subset_label in [
            ("native_all", "proxy_all", "ALL"),
            ("native_ca", "proxy_ca", "CA"),
            ("native_cs", "proxy_cs", "CS"),
        ]:
            res = run_incremental_regression(reg, target_col=target, proxy_col=proxy)
            res["suite_native"] = s_nat
            res["suite_proxy"] = s_prx
            res["language"] = lang
            res["subset"] = subset_label
            regression_rows.append(res)

        # --- Step 4a: Partial correlations ---
        for target, proxy, subset_label in [
            ("native_all", "proxy_all", "ALL"),
            ("native_ca", "proxy_ca", "CA"),
            ("native_cs", "proxy_cs", "CS"),
        ]:
            pres = partial_correlation(reg, target_col=target, proxy_col=proxy)
            pres["suite_native"] = s_nat
            pres["suite_proxy"] = s_prx
            pres["language"] = lang
            pres["subset"] = subset_label
            partial_rows.append(pres)

        # --- Step 4b: Leave-one-family-out (ALL only) ---
        lofo = leave_one_family_out(reg, target_col="native_all", proxy_col="proxy_all")
        for lr in lofo:
            lr["suite_native"] = s_nat
            lr["suite_proxy"] = s_prx
            lr["language"] = lang
        lofo_rows.extend(lofo)

    # --- Step 5: Sensitivity analysis with GMMLUX_EN as additional control ---
    print("[5/7] Running sensitivity analysis (+ GMMLUX_EN control)...")
    sensitivity_rows = []
    controls_extended = ["log_params", "belebele", "gmmlux_en"]

    for _, spec in pairs.iterrows():
        lang = str(spec["language"])
        s_nat = str(spec["suite_native"])
        s_prx = str(spec["suite_proxy"])
        label = f"{s_nat} ({lang})"

        reg = build_regression_df(
            scores, model_meta,
            suite_native=s_nat, suite_proxy=s_prx, language=lang,
            belebele_task=args.belebele_task,
        )

        if reg.empty or len(reg) < 5:
            continue

        # Check if gmmlux_en is available
        if "gmmlux_en" not in reg.columns or reg["gmmlux_en"].isna().all():
            print(f"  [SKIP sensitivity] {label}: no GMMLUX_EN scores")
            continue

        # Run only ALL-items regression with extended controls
        res = run_incremental_regression(
            reg, target_col="native_all", proxy_col="proxy_all",
            control_cols=controls_extended,
        )
        res["suite_native"] = s_nat
        res["suite_proxy"] = s_prx
        res["language"] = lang
        res["subset"] = "ALL"
        sensitivity_rows.append(res)

    # --- Write outputs ---
    print("[6/7] Writing outputs...")

    reg_df = pd.DataFrame(regression_rows)
    if not reg_df.empty:
        col_order = ["suite_native", "suite_proxy", "language", "subset",
                      "n", "controls", "R2_base", "R2_adj_base", "R2_ext", "R2_adj_ext",
                      "delta_R2", "delta_R2_adj", "F_stat", "F_pval",
                      "beta_proxy", "se_proxy", "t_proxy", "p_proxy",
                      "beta_const",
                      "AIC_base", "BIC_base", "AIC_ext", "BIC_ext", "note"]
        col_order = [c for c in col_order if c in reg_df.columns]
        # Also include any beta_* control columns not in the fixed list
        extra_beta = [c for c in reg_df.columns if c.startswith("beta_") and c not in col_order]
        col_order = col_order[:-1] + extra_beta + col_order[-1:]  # insert before "note"
        col_order = [c for c in col_order if c in reg_df.columns]
        reg_df = reg_df[col_order].sort_values(
            ["language", "suite_native", "subset"]).reset_index(drop=True)
    reg_df.to_csv(out_dir / "incremental_regression.csv", index=False)

    partial_df = pd.DataFrame(partial_rows)
    if not partial_df.empty:
        partial_df = partial_df.sort_values(
            ["language", "suite_native", "subset"]).reset_index(drop=True)
    partial_df.to_csv(out_dir / "partial_correlations.csv", index=False)

    lofo_df = pd.DataFrame(lofo_rows)
    if not lofo_df.empty:
        lofo_df = lofo_df.sort_values(
            ["language", "suite_native", "excluded_family"]).reset_index(drop=True)
    lofo_df.to_csv(out_dir / "leave_one_family_out.csv", index=False)

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    if not sensitivity_df.empty:
        sensitivity_df = sensitivity_df.sort_values(
            ["language", "suite_native"]).reset_index(drop=True)
    sensitivity_df.to_csv(out_dir / "sensitivity_gmmlux_en.csv", index=False)

    # Combined Excel
    with pd.ExcelWriter(out_dir / "incremental_proxy_validity.xlsx", engine="openpyxl") as writer:
        reg_df.to_excel(writer, sheet_name="regressions", index=False)
        partial_df.to_excel(writer, sheet_name="partial_corr", index=False)
        lofo_df.to_excel(writer, sheet_name="lofo", index=False)
        if not sensitivity_df.empty:
            sensitivity_df.to_excel(writer, sheet_name="sensitivity_en", index=False)
        model_meta.to_excel(writer, sheet_name="model_meta", index=False)

    # --- Summary ---
    print("[7/7] Summary")
    if not reg_df.empty:
        main_results = reg_df[reg_df["subset"] == "ALL"][
            ["suite_native", "language", "n", "R2_base", "R2_ext",
             "delta_R2", "F_pval", "beta_proxy", "p_proxy"]
        ].copy()
        print("\nMain results (ALL items):")
        print(main_results.to_string(index=False))

    if not sensitivity_df.empty:
        sens_summary = sensitivity_df[
            ["suite_native", "language", "n", "controls", "R2_base", "R2_ext",
             "delta_R2", "F_pval", "beta_proxy", "p_proxy"]
        ].copy()
        print("\nSensitivity (+ GMMLUX_EN control, ALL items):")
        print(sens_summary.to_string(index=False))

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()


