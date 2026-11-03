#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reads dev or test JSONL (from step 2)
Evaluates (human vs GPT):
Conf_csl_provided
Conf_csl_derived (human derived from step 1, GPT derived from GPT required_knowledge)
Conf_required_knowledge (human required_knowledge_agreement vs GPT required_knowledge)
Mismatch sheets: Mismatch_required_knowledge, Mismatch_csl_derived, Mismatch_csl_provided
Writes Excel + clear summary (denom/matches/accuracy)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gpt_csl_from_required_knowledge(rk: Any) -> Optional[str]:
    if rk is None:
        return None
    s = str(rk).strip().lower()
    if s == "none":
        return "CA"
    if s in ("cultural", "regional", "dialect"):
        return "CS"
    return None


def compute_match(h: Any, g: Any) -> Optional[bool]:
    if h is None or g is None:
        return None
    return h == g


def confusion_matrix(df: pd.DataFrame, human_col: str, gpt_col: str) -> pd.DataFrame:
    labels = sorted(set(df[human_col].dropna().unique()) | set(df[gpt_col].dropna().unique()))
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for _, r in df.iterrows():
        h = r[human_col]
        g = r[gpt_col]
        if pd.isna(h) or pd.isna(g):
            continue
        mat.loc[h, g] += 1
    mat["__row_total"] = mat.sum(axis=1)
    total_row = mat.sum(axis=0).to_frame().T
    total_row.index = ["__col_total"]
    return pd.concat([mat, total_row], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL split (dev or test) from step2")
    ap.add_argument("--out", dest="out_xlsx", required=True, help="Output Excel")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    rows = read_jsonl(Path(args.in_path))
    logging.info("Loaded %d rows", len(rows))

    out_rows = []
    for r in rows:
        ref = r.get("culture_ann_ref") or {}
        ann = r.get("culture_ann") or {}

        human_rk = ref.get("required_knowledge_agreement")
        human_csl = ref.get("cultural_sensitivity_label_agreement")

        gpt_rk = ann.get("required_knowledge")
        gpt_csl_rule = gpt_csl_from_required_knowledge(gpt_rk)
        gpt_csl_provided = ann.get("cultural_sensitivity_label")  # optional debug

        out_rows.append(
            {
                "id": r.get("id"),
                "subset": r.get("subset"),
                "language": r.get("language"),
                "human_required_knowledge": human_rk,
                "gpt_required_knowledge": gpt_rk,
                "human_csl": human_csl,
                "gpt_csl_rule": gpt_csl_rule,
                "gpt_csl_provided": gpt_csl_provided,
            }
        )

    df = pd.DataFrame(out_rows)

    # matches + summary
    df["match_required_knowledge"] = df.apply(lambda rr: compute_match(rr["human_required_knowledge"], rr["gpt_required_knowledge"]), axis=1)
    df["match_csl_rule"] = df.apply(lambda rr: compute_match(rr["human_csl"], rr["gpt_csl_rule"]), axis=1)
    df["match_csl_provided"] = df.apply(lambda rr: compute_match(rr["human_csl"], rr["gpt_csl_provided"]), axis=1)

    summary = [("n_items", len(df))]
    for name, col in [("required_knowledge", "match_required_knowledge"), ("csl_rule", "match_csl_rule"), ("csl_provided", "match_csl_provided")]:
        denom = df[col].notna().sum()
        matches = (df[col] == True).sum()  # noqa: E712
        acc = (matches / denom) if denom else None
        summary.extend(
            [
                (f"{name}_denom", int(denom)),
                (f"{name}_matches", int(matches)),
                (f"{name}_accuracy", acc),
            ]
        )
    summary_df = pd.DataFrame(summary, columns=["metric", "value"])

    # confusions
    conf_csl_rule = confusion_matrix(df, "human_csl", "gpt_csl_rule")
    conf_csl_provided = confusion_matrix(df, "human_csl", "gpt_csl_provided")
    conf_rk = confusion_matrix(df, "human_required_knowledge", "gpt_required_knowledge")

    # mismatches
    mm_rk = df[df["match_required_knowledge"] == False][
        ["id", "subset", "language", "human_required_knowledge", "gpt_required_knowledge", "human_csl", "gpt_csl_rule"]
    ]
    mm_csl_rule = df[df["match_csl_rule"] == False][
        ["id", "subset", "language", "human_csl", "gpt_csl_rule", "human_required_knowledge", "gpt_required_knowledge"]
    ]
    mm_csl_prov = df[df["match_csl_provided"] == False][["id", "subset", "language", "human_csl", "gpt_csl_provided"]]

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        conf_csl_rule.to_excel(writer, sheet_name="Conf_csl_rule")
        conf_csl_provided.to_excel(writer, sheet_name="Conf_csl_provided")
        conf_rk.to_excel(writer, sheet_name="Conf_required_knowledge")

        mm_rk.to_excel(writer, sheet_name="Mismatch_required_knowledge", index=False)
        mm_csl_rule.to_excel(writer, sheet_name="Mismatch_csl_rule", index=False)
        mm_csl_prov.to_excel(writer, sheet_name="Mismatch_csl_provided", index=False)

    logging.info("Wrote Excel: %s", out_xlsx)


if __name__ == "__main__":
    main()


