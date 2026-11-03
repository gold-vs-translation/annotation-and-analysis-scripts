#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline ranking analysis for Block A.

Input
-----
JSONL produced by build_balanced_rank_subsets.py:
  balanced_subsets_model_rows.jsonl

Analysis
--------
For each selection_group x language:
- compute model accuracy on CA items
- compute model accuracy on CS items
- rank models within CA and within CS
- estimate ranking consistency using Kendall tau-b
- report p-value from scipy.stats.kendalltau
- report 95% bootstrap CI over models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def read_json_records(path: Path) -> pd.DataFrame:
    """Read either true JSONL or a JSON array file."""
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError('Input file is empty.')

    # First try pandas' JSONL reader (more forgiving than manual line parsing).
    try:
        df = pd.read_json(path, lines=True)
        if not df.empty:
            return df
    except ValueError:
        pass

    # Fallback: strict line-by-line JSONL.
    rows: List[Dict[str, Any]] = []
    try:
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        if rows:
            return pd.DataFrame(rows)
    except json.JSONDecodeError:
        pass

    # Fallback: entire file is one JSON value (typically a list of records).
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f'Could not parse input as JSONL or JSON array: {e}') from e
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        return pd.DataFrame([obj])
    raise ValueError('Unsupported JSON structure. Expected JSONL or a JSON array of records.')




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

def rank_desc_with_ties(scores: pd.Series) -> pd.Series:
    return scores.rank(method='dense', ascending=False)


def bootstrap_kendall_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    vals: List[float] = []
    idx_all = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        try:
            tau, _ = kendalltau(x[idx], y[idx], variant='b')
        except TypeError:
            tau, _ = kendalltau(x[idx], y[idx])
        if pd.notna(tau):
            vals.append(float(tau))
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def compute_group_language_summary(g: pd.DataFrame, n_boot: int, seed: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    acc = (
        g.groupby(['model_name', 'ca_cs'], dropna=False)['sample_score']
        .mean()
        .reset_index()
        .pivot(index='model_name', columns='ca_cs', values='sample_score')
        .reset_index()
    )
    acc = acc.dropna(subset=['CA', 'CS']).copy()
    acc['rank_CA'] = rank_desc_with_ties(acc['CA'])
    acc['rank_CS'] = rank_desc_with_ties(acc['CS'])
    acc = acc.sort_values(['rank_CA', 'model_name'], ascending=[True, True]).reset_index(drop=True)

    x = acc['rank_CA'].to_numpy(dtype=float)
    y = acc['rank_CS'].to_numpy(dtype=float)
    if len(acc) >= 2:
        try:
            tau, p_value = kendalltau(x, y, variant='b')
        except TypeError:
            tau, p_value = kendalltau(x, y)
        ci_low, ci_high = bootstrap_kendall_ci(x, y, n_boot=n_boot, seed=seed)
    else:
        tau, p_value, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan

    summary = {
        'selection_group': g['selection_group'].iloc[0],
        'language': g['language'].iloc[0],
        'n_CA': int(g.loc[g['ca_cs'] == 'CA', 'sample_id'].nunique()),
        'n_CS': int(g.loc[g['ca_cs'] == 'CS', 'sample_id'].nunique()),
        'n_models': int(len(acc)),
        'kendall_tau_b': float(tau) if pd.notna(tau) else np.nan,
        'ci_low': float(ci_low) if pd.notna(ci_low) else np.nan,
        'ci_high': float(ci_high) if pd.notna(ci_high) else np.nan,
        'p_value': float(p_value) if pd.notna(p_value) else np.nan,
    }
    return summary, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, type=str, help='Path to balanced_subsets_model_rows.jsonl')
    ap.add_argument('--out_dir', required=True, type=str, help='Output directory')
    ap.add_argument('--n_boot', default=2000, type=int, help='Bootstrap iterations for CI')
    ap.add_argument('--seed', default=42, type=int, help='Random seed')
    ap.add_argument('--write_debug', action='store_true', help='Also write per-model CA/CS accuracies and ranks')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_json_records(Path(args.input))
    required = {'selection_group', 'language', 'model_name', 'sample_score', 'ca_cs', 'sample_id'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = normalize_sample_score_per_value(df)
    df = df.dropna(subset=['sample_score'])
    df = df[df['ca_cs'].isin(['CA', 'CS'])].copy()

    summary_rows, debug_parts = [], []
    for i, ((selection_group, language), g) in enumerate(df.groupby(['selection_group', 'language'], dropna=False)):
        summary, acc = compute_group_language_summary(g, n_boot=args.n_boot, seed=args.seed + i)
        summary_rows.append(summary)
        if args.write_debug and not acc.empty:
            acc.insert(0, 'language', language)
            acc.insert(0, 'selection_group', selection_group)
            debug_parts.append(acc)

    summary_df = pd.DataFrame(summary_rows).sort_values(['selection_group', 'language']).reset_index(drop=True)
    summary_df.to_csv(out_dir / 'baseline_rank_summary.csv', index=False)
    with pd.ExcelWriter(out_dir / 'baseline_rank_summary.xlsx', engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        if args.write_debug and debug_parts:
            pd.concat(debug_parts, ignore_index=True).to_excel(writer, sheet_name='debug_ranks', index=False)
    print(f'Saved Block A ranking outputs to {out_dir}')


if __name__ == '__main__':
    main()


