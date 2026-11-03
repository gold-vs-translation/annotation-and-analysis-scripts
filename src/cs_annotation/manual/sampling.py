#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


LANGS = ["SV", "ET", "LT", "NB", "DA"]


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dataset_name_from_filename(p: Path) -> str:
    # tries to extract dataset name from file like:
    # text_skolprov_SV_train-gpt-5.2-annotated.jsonl
    name = p.name
    m = re.match(r"text_(.+?)_[A-Z]{2}_.+?-gpt-5\.2-annotated\.jsonl$", name)
    if m:
        return m.group(1)
    # fallback: strip suffixes
    base = name.replace("-gpt-5.2-annotated", "").replace(".jsonl", "")
    return base


def flatten_proxy_labels(row: dict) -> dict:
    ca = row.get("culture_ann") or {}
    return {
        "proxy.required_knowledge": ca.get("required_knowledge"),
        "proxy.subject_category": ca.get("subject_category"),
        "proxy.time_sensitive": ca.get("time_sensitive"),
        "proxy.cultural_sensitivity_label": ca.get("cultural_sensitivity_label"),
    }


def backoff_stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Try stratification with decreasing complexity to avoid over-fragmentation:
      4-way -> 3-way -> 2-way -> 1-way -> random
    """
    rng = random.Random(seed)

    candidates = [
        ["proxy.required_knowledge", "proxy.subject_category", "proxy.time_sensitive", "proxy.cultural_sensitivity_label"],
        ["proxy.required_knowledge", "proxy.subject_category", "proxy.time_sensitive"],
        ["proxy.required_knowledge", "proxy.subject_category"],
        ["proxy.required_knowledge"],
    ]

    def strat_sample(cols: List[str]) -> pd.DataFrame:
        d = df.copy()
        for c in cols:
            d[c] = d[c].fillna("UNK")

        d["__stratum"] = d[cols].astype(str).agg("|".join, axis=1)
        counts = d["__stratum"].value_counts()

        # proportional allocation
        weights = counts / counts.sum()
        alloc = (weights * n).round().astype(int)

        # fix rounding to sum to n
        diff = n - int(alloc.sum())
        if diff != 0:
            order = list(weights.sort_values(ascending=False).index)
            i = 0
            step = 1 if diff > 0 else -1
            while diff != 0 and i < 10000:
                s = order[i % len(order)]
                if step > 0:
                    alloc[s] += 1
                    diff -= 1
                else:
                    if alloc[s] > 0:
                        alloc[s] -= 1
                        diff += 1
                i += 1

        parts = []
        for s, k in alloc.items():
            if k <= 0:
                continue
            sub = d[d["__stratum"] == s]
            k = min(int(k), len(sub))
            if k <= 0:
                continue
            parts.append(sub.sample(n=k, random_state=seed))

        out = pd.concat(parts, ignore_index=True) if parts else d.sample(n=min(n, len(d)), random_state=seed)
        out = out.drop(columns=["__stratum"])
        return out

    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed)  # shuffle

    # Backoff selection: choose first scheme that yields "reasonable" number of strata
    # (avoid extreme fragmentation where most strata are size 1)
    for cols in candidates:
        d = df.copy()
        for c in cols:
            d[c] = d[c].fillna("UNK")
        n_strata = d[cols].astype(str).agg("|".join, axis=1).nunique()
        # heuristic: strata count not too close to n (i.e., avoid mostly-singleton strata)
        if n_strata <= max(10, n // 3):
            return strat_sample(cols)

    # fallback: simple random sample
    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n]].copy()


def allocate_per_dataset(df_lang: pd.DataFrame, total_n: int, min_per_dataset: int, seed: int) -> Dict[str, int]:
    """
    Allocate total_n across datasets in a language.
    - If only one dataset: all
    - If multiple: ensure min_per_dataset each (as possible), rest proportional to dataset size
    """
    ds_counts = df_lang["__dataset"].value_counts().to_dict()
    datasets = list(ds_counts.keys())

    if len(datasets) == 1:
        return {datasets[0]: min(total_n, ds_counts[datasets[0]])}

    alloc = {ds: min(min_per_dataset, ds_counts[ds]) for ds in datasets}
    base = sum(alloc.values())

    if base > total_n:
        # scale down
        scale = total_n / base
        for ds in datasets:
            alloc[ds] = max(1, int(round(alloc[ds] * scale)))
        # fix rounding
        while sum(alloc.values()) > total_n:
            biggest = max(alloc, key=lambda k: alloc[k])
            if alloc[biggest] > 1:
                alloc[biggest] -= 1
            else:
                break
    else:
        remaining = total_n - base
        if remaining > 0:
            cap = {ds: ds_counts[ds] - alloc[ds] for ds in datasets}
            cap_total = sum(max(0, v) for v in cap.values())
            for ds in datasets:
                if cap_total > 0 and cap[ds] > 0:
                    add = int(round(remaining * (cap[ds] / cap_total)))
                    alloc[ds] += min(add, cap[ds])

            # fix rounding to exact total_n
            diff = total_n - sum(alloc.values())
            order = sorted(datasets, key=lambda ds: ds_counts[ds], reverse=True)
            i = 0
            while diff != 0 and i < 10000:
                ds = order[i % len(order)]
                if diff > 0 and alloc[ds] < ds_counts[ds]:
                    alloc[ds] += 1
                    diff -= 1
                elif diff < 0 and alloc[ds] > 0:
                    alloc[ds] -= 1
                    diff += 1
                i += 1

    return alloc


def remove_proxy_annotations(row: dict) -> dict:
    # IMPORTANT: do not leak model labels to annotators
    row = dict(row)
    row.pop("culture_ann", None)
    row.pop("culture_ann_ref", None)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated-dir", required=True, help="Directory with *-gpt-5.2-annotated.jsonl files")
    ap.add_argument("--out-dir", required=True, help="Output directory for cult_ann_split_XX.jsonl")
    ap.add_argument("--n-per-lang", type=int, default=100)
    ap.add_argument("--min-per-dataset", type=int, default=40, help="Only applies if a language has >1 dataset")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    annotated_dir = Path(args.annotated_dir)
    out_dir = Path(args.out_dir)

    files = sorted(annotated_dir.glob("*-gpt-5.2-annotated.jsonl"))
    if not files:
        raise FileNotFoundError(f"No annotated files found in {annotated_dir}")

    # Load all rows, tag dataset by filename
    all_rows = []
    for p in files:
        ds_name = dataset_name_from_filename(p)
        rows = read_jsonl(p)
        for r in rows:
            r["__dataset"] = ds_name
            r.update(flatten_proxy_labels(r))
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df["language"] = df["language"].astype(str).str.upper()

    # Basic sanity check: uniqueness key
    key_cols = ["suite", "task", "id", "subset", "split", "language"]
    if df.duplicated(subset=key_cols).any():
        dups = df[df.duplicated(subset=key_cols, keep=False)][key_cols].head(10)
        raise ValueError(f"Found duplicate items by key {key_cols}. Example:\n{dups}")

    for lang in LANGS:
        df_lang = df[df["language"] == lang].copy()
        if df_lang.empty:
            print(f"[WARN] No data for language={lang}; skipping.")
            continue

        total_n = min(args.n_per_lang, len(df_lang))
        alloc = allocate_per_dataset(df_lang, total_n, args.min_per_dataset, seed=args.seed)

        picked_parts = []
        for ds_name, k in alloc.items():
            block = df_lang[df_lang["__dataset"] == ds_name].copy()
            if block.empty or k <= 0:
                continue
            samp = backoff_stratified_sample(block, n=min(k, len(block)), seed=args.seed)
            picked_parts.append(samp)

        picked = pd.concat(picked_parts, ignore_index=True) if picked_parts else df_lang.sample(n=total_n, random_state=args.seed)

        # Ensure exact total_n (due to rare corner cases)
        if len(picked) > total_n:
            picked = picked.sample(n=total_n, random_state=args.seed)
        elif len(picked) < total_n and len(df_lang) >= total_n:
            missing = total_n - len(picked)
            remainder = df_lang.drop(index=picked.index, errors="ignore")
            if len(remainder) >= missing:
                picked = pd.concat([picked, remainder.sample(n=missing, random_state=args.seed)], ignore_index=True)

        # Export rows WITHOUT proxy labels/fields
        cols_to_drop = [c for c in picked.columns if c.startswith("proxy.") or c in ["__dataset"]]
        picked_export = picked.drop(columns=cols_to_drop, errors="ignore").to_dict(orient="records")
        picked_export = [remove_proxy_annotations(r) for r in picked_export]

        out_path = out_dir / f"cult_ann_split_{lang}.jsonl"
        write_jsonl(out_path, picked_export)

        # Print a small summary for sanity
        print(f"\n[OK] {lang}: wrote {len(picked_export)} -> {out_path}")
        print("  Dataset allocation:", alloc)

    print("\nDone.")


if __name__ == "__main__":
    main()


