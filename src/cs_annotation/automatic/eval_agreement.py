"""
Evaluate agreement between:
- culture_ann_ref (multi-annotator lists) -> aggregated via majority vote
- culture_ann (single model annotation)

Outputs an Excel workbook with:
- Summary
- Counts (human vs gpt)
- Confusion matrices for: csl_provided, csl_derived, required_knowledge
- Annotator count distribution + human disagreement stats

NOTE: For a clean, comparable CS/CA, we derive CS/CA deterministically from required_knowledge (majority for humans).
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional: ISO code normalization
try:
    import pycountry  # type: ignore
except Exception:
    pycountry = None

AGREE_FIELDS = ["csl_provided", "csl_derived", "required_knowledge"]
COUNT_FIELDS = ["csl_provided", "csl_derived", "required_knowledge"]

# Some common ISO3->ISO2 fallbacks if pycountry is not installed
ISO3_TO_ISO2_FALLBACK = {
    "USA": "US",
    "GBR": "GB",
    "DEU": "DE",
    "FRA": "FR",
    "ESP": "ES",
    "ITA": "IT",
    "CHN": "CN",
    "JPN": "JP",
    "KOR": "KR",
    "RUS": "RU",
    "IND": "IN",
    "BRA": "BR",
    "CAN": "CA",
    "AUS": "AU",
    "MEX": "MX",
    "ZAF": "ZA",
    "TUR": "TR",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def majority_vote(values: List[Any]) -> Tuple[Optional[Any], bool]:
    """
    Returns (winner, is_tie).
    Treats None/"-" / empty strings as missing and removes them before voting.
    """
    cleaned = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            vv = v.strip()
            if vv == "" or vv == "-":
                continue
            else:
                cleaned.append(vv)
        else:
            cleaned.append(v)

    # For general majority, keep cleaned as-is; caller can prefilter.
    if not cleaned:
        return None, False

    cnt = Counter(cleaned)
    top = cnt.most_common()
    if len(top) == 1:
        return top[0][0], False
    if top[0][1] == top[1][1]:
        # return top[0][0], True
        # Tie: do NOT pick an arbitrary winner; mark ambiguous
        return None, True
    return top[0][0], False


def vote_profile(values: List[str]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None and str(v).strip() != ""]
    n = len(vals)
    if n == 0:
        return {"n": 0, "tie": False, "majority_share": None, "num_labels": 0, "counts": {}}
    cnt = Counter(vals)
    top = cnt.most_common()
    max_count = top[0][1]
    tie = len(top) > 1 and top[1][1] == max_count
    return {
        "n": n,
        "tie": tie,
        "majority_share": max_count / n,
        "num_labels": len(cnt),
        "counts": dict(cnt),
    }


def derive_csl_from_required_knowledge(rk: Optional[str]) -> Optional[str]:
    """
    Deterministically derive CS/CA from required_knowledge:
      - CA iff rk == 'none'
      - CS otherwise
    """
    if rk is None:
        return None
    return "CA" if rk == "none" else "CS"


def normalize_basic(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "null":
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_enum_lower(x: Optional[str]) -> Optional[str]:
    s = normalize_basic(x)
    return s.lower() if s is not None else None


def iso3_to_iso2(code3: str) -> Optional[str]:
    code3 = code3.upper().strip()
    if len(code3) != 3:
        return None
    if pycountry is not None:
        try:
            c = pycountry.countries.get(alpha_3=code3)
            if c and getattr(c, "alpha_2", None):
                return c.alpha_2
        except Exception:
            pass
    return ISO3_TO_ISO2_FALLBACK.get(code3)


def normalize_country(x: Optional[str]) -> Optional[str]:
    """
    Normalize country strings:
    - Prefer "Country Name (ISO2)" if code is present.
    - Convert ISO3 to ISO2 if possible.
    - Otherwise return cleaned name.
    """
    s = normalize_basic(x)
    if s is None:
        return None

    # Extract optional "(CODE)"
    m = re.match(r"^(.*?)(?:\s*\(([^)]+)\))?$", s)
    name = (m.group(1) or "").strip() if m else s
    code = (m.group(2) or "").strip().upper() if m and m.group(2) else ""

    # Harmonize some common country names
    name_norm = re.sub(r"\s+", " ", name).strip()

    # If code looks like ISO3, convert
    if code and len(code) == 3:
        code2 = iso3_to_iso2(code)
        if code2:
            code = code2

    # If code looks like ISO2, standardize
    if code and len(code) == 2:
        return f"{name_norm} ({code})"

    # If no code but name matches a country and pycountry available, you could optionally add code.
    # Keep it conservative; just return name_norm.
    return name_norm


def get_human_agg(ref: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate culture_ann_ref (multi-annotator) to item-level labels using majority vote.
    """
    out: Dict[str, Any] = {}

    # csl_provided (diagnostic only): whatever the dataset provides
    csl = ref.get("cultural_sensitivity_label")
    if isinstance(csl, list):
        winner, tie = majority_vote([normalize_basic(v) for v in csl if v is not None])
        out["csl_provided"] = winner
        out["_tie_csl"] = tie
    else:
        out["csl_provided"] = normalize_basic(csl)
        out["_tie_csl"] = False

    # required_knowledge, time_sensitive are typically lists of annotator judgments
    rk = ref.get("required_knowledge")
    if isinstance(rk, list):
        # keep literal 'none' as meaningful label
        rk_vals = [normalize_enum_lower(v) for v in rk if v is not None and str(v).strip() != ""]
        winner, tie = majority_vote(rk_vals)
        out["required_knowledge"] = winner
        out["_tie_rk"] = tie
        prof = vote_profile(rk_vals)
        out["_n_annotators_rk"] = prof["n"]
        out["_rk_majority_share"] = prof["majority_share"]
        out["_rk_num_labels"] = prof["num_labels"]
    else:
        out["required_knowledge"] = normalize_enum_lower(rk)
        out["_tie_rk"] = False
        out["_n_annotators_rk"] = 1 if out["required_knowledge"] is not None else 0
        out["_rk_majority_share"] = 1.0 if out["required_knowledge"] is not None else None
        out["_rk_num_labels"] = 1 if out["required_knowledge"] is not None else 0

    # csl_derived (used for agreement): derived from majority required_knowledge
    out["csl_derived"] = derive_csl_from_required_knowledge(out.get("required_knowledge"))

    ts = ref.get("time_sensitive")
    if isinstance(ts, list):
        ts_vals = [normalize_basic(v) for v in ts if v is not None and str(v).strip() != ""]
        winner, tie = majority_vote(ts_vals)
        out["time_sensitive"] = winner
        out["_tie_ts"] = tie
    else:
        out["time_sensitive"] = normalize_basic(ts)
        out["_tie_ts"] = False

    # culture/region/country: lists -> majority (primary)
    culture = ref.get("culture")
    if isinstance(culture, list):
        vals = [normalize_basic(v) for v in culture if normalize_basic(v) is not None]
        winner, tie = majority_vote(vals)
        out["culture"] = winner
        out["_tie_culture"] = tie
    else:
        out["culture"] = normalize_basic(culture)
        out["_tie_culture"] = False

    region = ref.get("region")
    if isinstance(region, list):
        vals = [normalize_basic(v) for v in region if normalize_basic(v) is not None]
        winner, tie = majority_vote(vals)
        out["region"] = winner
        out["_tie_region"] = tie
    else:
        out["region"] = normalize_basic(region)
        out["_tie_region"] = False

    country = ref.get("country")
    if isinstance(country, list):
        vals = [normalize_country(v) for v in country if normalize_country(v) is not None]
        winner, tie = majority_vote(vals)
        out["country"] = winner
        out["_tie_country"] = tie
    else:
        out["country"] = normalize_country(country)
        out["_tie_country"] = False

    return out


def get_gpt_ann(ann: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # keep provided GPT CS/CA (diagnostic)
    out["csl_provided"] = normalize_basic(ann.get("cultural_sensitivity_label"))
    out["required_knowledge"] = normalize_enum_lower(ann.get("required_knowledge"))
    out["time_sensitive"] = normalize_basic(ann.get("time_sensitive"))
    out["culture"] = normalize_basic(ann.get("culture"))
    out["region"] = normalize_basic(ann.get("region"))
    out["country"] = normalize_country(ann.get("country"))
    # derived GPT CS/CA for clean comparison
    out["csl_derived"] = derive_csl_from_required_knowledge(out.get("required_knowledge"))
    return out


def compute_match(h: Any, g: Any) -> Optional[bool]:
    if h is None or g is None:
        return None
    return h == g


def confusion_matrix(rows: List[Dict[str, Any]], field: str) -> pd.DataFrame:
    """
    Build confusion matrix table: human (rows) x gpt (cols)
    """
    labels = set()
    for r in rows:
        labels.add(r.get(f"human_{field}"))
        labels.add(r.get(f"gpt_{field}"))
    labels.discard(None)
    labels = sorted(labels)

    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for r in rows:
        h = r.get(f"human_{field}")
        g = r.get(f"gpt_{field}")
        if h is None or g is None:
            continue
        if h not in mat.index:
            mat.loc[h] = 0
        if g not in mat.columns:
            mat[g] = 0
        mat.loc[h, g] += 1

    # add totals
    mat["__row_total"] = mat.sum(axis=1)
    total_row = mat.sum(axis=0).to_frame().T
    total_row.index = ["__col_total"]
    out = pd.concat([mat, total_row], axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL with culture_ann_ref and culture_ann")
    ap.add_argument("--out", dest="out_xlsx", required=True, help="Output Excel .xlsx")
    ap.add_argument("--skip-ties", action="store_true", help="If set, exclude items with majority ties from agreement denominators.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    rows = read_jsonl(in_path)

    merged_rows: List[Dict[str, Any]] = []
    for rec in rows:
        ref = rec.get("culture_ann_ref") or {}
        ann = rec.get("culture_ann") or {}

        h = get_human_agg(ref)
        g = get_gpt_ann(ann)

        row = {
            "id": rec.get("id"),
            "subset": rec.get("subset"),
            "task": rec.get("task"),
            "suite": rec.get("suite"),
            "language": rec.get("language"),
            "text": rec.get("text", ""),
        }
        # store both provided and derived CS/CA, plus other fields
        for f in COUNT_FIELDS:
            row[f"human_{f}"] = h.get(f)
            row[f"gpt_{f}"] = g.get(f)

        # tie flags
        row["tie_required_knowledge"] = bool(h.get("_tie_rk"))
        row["tie_time_sensitive"] = bool(h.get("_tie_ts"))
        row["tie_culture"] = bool(h.get("_tie_culture"))
        row["tie_region"] = bool(h.get("_tie_region"))
        row["tie_country"] = bool(h.get("_tie_country"))
        row["tie_csl_provided"] = bool(h.get("_tie_csl"))
        # annotator/disagreement stats (based on required_knowledge votes)
        row["human_n_annotators"] = h.get("_n_annotators_rk")
        row["human_rk_majority_share"] = h.get("_rk_majority_share")
        row["human_rk_num_labels"] = h.get("_rk_num_labels")

        # diagnostic: provided vs derived (human)
        row["human_csl_provided_vs_derived_mismatch"] = (
            row["human_csl_provided"] is not None and row["human_csl_derived"] is not None and row["human_csl_provided"] != row["human_csl_derived"]
        )

        # diagnostic: provided vs derived (gpt)
        row["gpt_csl_provided_vs_derived_mismatch"] = (
            row["gpt_csl_provided"] is not None and row["gpt_csl_derived"] is not None and row["gpt_csl_provided"] != row["gpt_csl_derived"]
        )

        # matches (raw)
        for f in AGREE_FIELDS:
            row[f"match_{f}"] = compute_match(row[f"human_{f}"], row[f"gpt_{f}"])

        # same CS/CA but detail mismatch (conditional)
        cs_same = row["human_csl_derived"] is not None and row["gpt_csl_derived"] is not None and row["human_csl_derived"] == row["gpt_csl_derived"]
        row["cs_same"] = cs_same

        merged_rows.append(row)

    df = pd.DataFrame(merged_rows)

    # Summary stats
    summary = []
    total = len(df)
    summary.append(("n_items", total))

    for f in AGREE_FIELDS:
        tie_col = None
        # derived CS/CA depends on majority required_knowledge
        if f == "csl_derived":
            tie_col = "tie_required_knowledge"
        elif f == "required_knowledge":
            tie_col = "tie_required_knowledge"
        elif f == "csl_provided":
            tie_col = "tie_csl_provided"
        elif f == "time_sensitive":
            tie_col = "tie_time_sensitive"
        elif f == "culture":
            tie_col = "tie_culture"
        elif f == "region":
            tie_col = "tie_region"
        elif f == "country":
            tie_col = "tie_country"

        usable = df.copy()
        if args.skip_ties and tie_col:
            usable = usable[~usable[tie_col]]

        denom = usable[f"match_{f}"].notna().sum()
        matches = (usable[f"match_{f}"] == True).sum()  # noqa: E712
        summary.append((f"{f}_denom", int(denom)))
        summary.append((f"{f}_matches", int(matches)))
        summary.append((f"{f}_accuracy", (matches / denom) if denom else None))

    # Human disagreement overview
    summary.append(("human_n_annotators_min", int(df["human_n_annotators"].min())))
    summary.append(("human_n_annotators_max", int(df["human_n_annotators"].max())))
    summary.append(("human_unanimous_n", int((df["human_rk_majority_share"] == 1.0).sum())))
    summary.append(("human_tie_n", int(df["tie_required_knowledge"].sum())))
    summary.append(("human_avg_majority_share", float(df["human_rk_majority_share"].dropna().mean())))

    # CS/CA same but details differ
    cs_same_n = int(df["cs_same"].sum())
    summary.append(("cs_same_n", cs_same_n))

    # Diagnostics: how often provided vs derived differ (human and gpt)
    summary.append(("human_csl_provided_vs_derived_mismatch_n", int(df["human_csl_provided_vs_derived_mismatch"].sum())))
    summary.append(("gpt_csl_provided_vs_derived_mismatch_n", int(df["gpt_csl_provided_vs_derived_mismatch"].sum())))

    summary_df = pd.DataFrame(summary, columns=["metric", "value"])

    # Counts tables
    counts_rows = []
    for f in COUNT_FIELDS:
        human_counts = df[f"human_{f}"].value_counts(dropna=False).to_dict()
        gpt_counts = df[f"gpt_{f}"].value_counts(dropna=False).to_dict()
        keys = set(human_counts.keys()) | set(gpt_counts.keys())
        for k in sorted(keys, key=lambda x: str(x)):
            counts_rows.append(
                {
                    "field": f,
                    "label": k,
                    "human_count": int(human_counts.get(k, 0)),
                    "gpt_count": int(gpt_counts.get(k, 0)),
                }
            )
    counts_df = pd.DataFrame(counts_rows)

    # Confusion matrices per field
    # We include both derived (used for agreement) and provided (diagnostic)
    # conf_fields = list(AGREE_FIELDS) + ["csl_provided"]
    # confusions = {f: confusion_matrix(merged_rows, f) for f in conf_fields}
    conf_fields = ["csl_provided", "csl_derived", "required_knowledge"]
    confusions = {f: confusion_matrix(merged_rows, f) for f in conf_fields}

    # Mismatch detail tables
    # 1) All mismatches per field
    mismatches = {}
    for f in AGREE_FIELDS:
        mm = df[df[f"match_{f}"] == False].copy()  # noqa: E712
        mismatches[f] = mm[
            [
                "id",
                "subset",
                "language",
                f"human_{f}",
                f"gpt_{f}",
                "human_csl_derived",
                "gpt_csl_derived",
                "human_csl_provided",
                "gpt_csl_provided",
                "human_n_annotators",
                "human_rk_majority_share",
                "tie_required_knowledge",
            ]
        ]

    # Annotator count distribution
    annot_counts = df["human_n_annotators"].value_counts(dropna=False).sort_index().reset_index()
    annot_counts.columns = ["n_annotators", "n_items"]

    # Human disagreement table (per item, compact)
    human_disagreement = df[
        [
            "id",
            "subset",
            "language",
            "human_n_annotators",
            "human_required_knowledge",
            "human_rk_majority_share",
            "human_rk_num_labels",
            "tie_required_knowledge",
        ]
    ]

    # Write Excel
    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        counts_df.to_excel(writer, sheet_name="Counts", index=False)
        annot_counts.to_excel(writer, sheet_name="AnnotatorCounts", index=False)
        human_disagreement.to_excel(writer, sheet_name="HumanDisagreement", index=False)

        for f, cm in confusions.items():
            sheet = f"Conf_{f}"[:31]
            cm.to_excel(writer, sheet_name=sheet)

        for f, mm_df in mismatches.items():
            sheet = f"Mismatch_{f}"[:31]
            mm_df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


