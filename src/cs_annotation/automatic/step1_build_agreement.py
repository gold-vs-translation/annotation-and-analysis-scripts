"""
Step-1: Agreement human multi-annotator lists via majority vote + Human-Derived CS/CA + Tie-Split

Reads the annotated JSONL (with culture_ann_ref + culture_ann).
Calculates human majority (required_knowledge + time_sensitive; optional culture/region/country depending on required_knowledge).

Handle ties carefully:
  * Keep and resolve ONLY ties of the form: none vs exactly one of {cultural, regional, dialect}
    (no additional max-ties among non-none labels).
    -> required_knowledge_agreement = that non-none label (resolved)
    -> cultural_sensitivity_label_derived = CS
  * Send all other ties (tie-of-ties / complex ties) into a separate JSONL.

Discard (= in --out-ties): “Tie of ties” or complex Ties, e.g.
['regional','cultural','dialect'] (alle gleich häufig)
['cultural','cultural','regional','dialect','regional','none'] (cultural vs regional Tie)
['none','none','cultural','cultural','regional','regional'] (none/cultural/regional Tie)

- required_knowledge_agreement:
  * unique majority -> winner
  * tie only between {"none", X} where X in {cultural, regional, dialect} -> resolve to X (keep)
  * all other ties (tie-of-ties / complex) -> send to out-ties

- cultural_sensitivity_label_agreement (CS/CA):
  * CA if #none > n/2, else CS (if there are valid votes)
  This matches the observed provided-label rule.

Calculates cultural_sensitivity_label_derived from Human-Majority required_knowledge
Compute stats to understand how provided cultural_sensitivity_label might be derived
Writes:
  --out-agreed (only items without tie in required_knowledge)
  --out-ties (tie items)

Writes Excel with:
Summary (including how many kept vs ties, tie types, mean majority share, etc.)
Counts (human agreement label counts)
AnnotatorCounts (distribution of annotators per item)
HumanDisagreement (per item: n_annotators, majority_share, num_labels, tie-type)
ProvidedCSL_Hypotheses (test your hypotheses: which rule best explains provided CS/CA)
Conf_ProvidedCSL_vs_Rules (confusion matrices of hypotheses against provided CS/CA)

"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

NON_NONE = {"cultural", "regional", "dialect"}
ALL_RK = NON_NONE | {"none"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_basic(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "null":
        return None
    s = re.sub(r"\s+", " ", s)
    return s


def norm_lower(x: Any) -> Optional[str]:
    s = norm_basic(x)
    return s.lower() if s is not None else None


def norm_csl(x: Any) -> Optional[str]:
    s = norm_basic(x)
    if s is None:
        return None
    s = s.upper()
    return s if s in ("CS", "CA") else None


def vote_counts(values: List[Optional[str]]) -> Tuple[Dict[str, int], int]:
    cleaned = [v for v in values if v is not None and v != "" and v != "-"]
    cnt = Counter(cleaned)
    return dict(cnt), len(cleaned)


def majority_share(counts: Dict[str, int], n: int) -> Optional[float]:
    if n <= 0 or not counts:
        return None
    return max(counts.values()) / n


def resolve_required_knowledge(counts: Dict[str, int], n: int) -> Dict[str, Any]:
    """
    Resolve required_knowledge agreement.

    Returns:
      rk_agreement: str|None
      tie_type: "none" | "tie_none_vs_non_none_resolved" | "complex_tie" | "missing"
      resolved_from_tie: bool
      is_complex_tie: bool
      max_labels: list[str] (labels tied at max, if any)
    """
    if n == 0 or not counts:
        return {
            "rk_agreement": None,
            "tie_type": "missing",
            "resolved_from_tie": False,
            "is_complex_tie": True,
            "max_labels": [],
        }

    max_count = max(counts.values())
    max_labels = sorted([k for k, v in counts.items() if v == max_count])

    if len(max_labels) == 1:
        return {
            "rk_agreement": max_labels[0],
            "tie_type": "none",
            "resolved_from_tie": False,
            "is_complex_tie": False,
            "max_labels": [],
        }

    # Tie case
    s = set(max_labels)

    # Allowed: tie between none and exactly one non-none label
    if len(s) == 2 and "none" in s:
        other = next(iter(s - {"none"}))
        if other in NON_NONE:
            return {
                "rk_agreement": other,  # resolve to non-none
                "tie_type": "tie_none_vs_non_none_resolved",
                "resolved_from_tie": True,
                "is_complex_tie": False,
                "max_labels": ["none", other],
            }

    # Otherwise: complex tie (tie-of-ties)
    return {
        "rk_agreement": None,
        "tie_type": "complex_tie",
        "resolved_from_tie": False,
        "is_complex_tie": True,
        "max_labels": max_labels,
    }


def csl_rule_strict_majority_none(rk_counts: Dict[str, int], n: int) -> Optional[str]:
    """
    Provided-CS/CA rule:
      CA if #none > n/2 else CS (if n>0)
    """
    if n <= 0:
        return None
    n_none = rk_counts.get("none", 0)
    return "CA" if n_none > (n / 2) else "CS"


def majority_label(values: List[Optional[str]]) -> Tuple[Optional[str], bool]:
    """
    Generic majority; tie => None.
    """
    counts, n = vote_counts(values)
    if n == 0:
        return None, False
    max_count = max(counts.values())
    max_labels = [k for k, v in counts.items() if v == max_count]
    if len(max_labels) == 1:
        return max_labels[0], False
    return None, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL with culture_ann_ref + culture_ann")
    ap.add_argument("--out-agreed", required=True, help="Output JSONL with agreement fields (kept items)")
    ap.add_argument("--out-ties", required=True, help="Output JSONL with complex RK ties (dropped)")
    ap.add_argument("--out-xlsx", required=True, help="Output Excel with human stats + provided-rule sanity check")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    rows = read_jsonl(Path(args.in_path))
    logging.info("Loaded %d rows", len(rows))

    agreed_rows: List[Dict[str, Any]] = []
    tie_rows: List[Dict[str, Any]] = []

    stats_rows: List[Dict[str, Any]] = []
    tie_type_counter = Counter()
    provided_vs_rule_mismatch = 0

    for rec in rows:
        ref = rec.get("culture_ann_ref") or {}

        # required_knowledge votes (normalized)
        rk_raw = ref.get("required_knowledge") or []
        rk_votes = [norm_lower(v) for v in rk_raw]
        rk_votes = [v for v in rk_votes if (v in ALL_RK) or (v is None)]

        rk_counts, n_valid = vote_counts(rk_votes)
        rk_share = majority_share(rk_counts, n_valid)
        rk_num_labels = len([k for k, v in rk_counts.items() if v > 0])

        rk_res = resolve_required_knowledge(rk_counts, n_valid)
        tie_type_counter[rk_res["tie_type"]] += 1

        # CS/CA agreement via strict majority none (works also for ties)
        csl_agreement = csl_rule_strict_majority_none(rk_counts, n_valid)

        # Compare to provided CS/CA (diagnostic)
        csl_provided = norm_csl(ref.get("cultural_sensitivity_label"))
        if csl_provided is not None and csl_agreement is not None and csl_provided != csl_agreement:
            provided_vs_rule_mismatch += 1

        # Drop only complex ties for required_knowledge agreement
        if rk_res["is_complex_tie"]:
            # still attach diagnostics
            ref["required_knowledge_vote_counts"] = rk_counts
            ref["n_annotators_required_knowledge"] = n_valid
            ref["required_knowledge_majority_share"] = rk_share
            ref["required_knowledge_num_labels"] = rk_num_labels
            ref["required_knowledge_tie_type"] = rk_res["tie_type"]
            ref["required_knowledge_max_labels"] = rk_res["max_labels"]
            ref["cultural_sensitivity_label_agreement"] = csl_agreement  # optional, for inspection
            tie_rows.append(rec)
            continue

        # Time-sensitive agreement (majority; tie -> "")
        ts_raw = ref.get("time_sensitive") or []
        ts_votes = [norm_basic(v) for v in ts_raw]
        ts_winner, ts_tie = majority_label(ts_votes)
        ts_agreement = ts_winner if (ts_winner is not None and not ts_tie) else ""

        # culture/region/country agreement conditional on RK agreement
        culture_agreement = ""
        region_agreement = ""
        country_agreement = ""

        rk_agreement = rk_res["rk_agreement"]

        if rk_agreement == "cultural":
            cul_votes = [norm_basic(v) for v in (ref.get("culture") or [])]
            cul_w, cul_t = majority_label(cul_votes)
            culture_agreement = cul_w or ""
        elif rk_agreement == "regional":
            reg_votes = [norm_basic(v) for v in (ref.get("region") or [])]
            reg_w, reg_t = majority_label(reg_votes)
            region_agreement = reg_w or ""
            ctry_votes = [norm_basic(v) for v in (ref.get("country") or [])]
            ctry_w, ctry_t = majority_label(ctry_votes)
            country_agreement = ctry_w or ""

        # Write agreement fields into culture_ann_ref (as requested)
        ref["required_knowledge_agreement"] = rk_agreement
        ref["time_sensitive_agreement"] = ts_agreement
        ref["culture_agreement"] = culture_agreement
        ref["region_agreement"] = region_agreement
        ref["country_agreement"] = country_agreement
        ref["cultural_sensitivity_label_agreement"] = csl_agreement

        # Diagnostics
        ref["required_knowledge_vote_counts"] = rk_counts
        ref["n_annotators_required_knowledge"] = n_valid
        ref["required_knowledge_majority_share"] = rk_share
        ref["required_knowledge_num_labels"] = rk_num_labels
        ref["required_knowledge_tie_type"] = rk_res["tie_type"]
        ref["required_knowledge_max_labels"] = rk_res["max_labels"]
        ref["required_knowledge_resolved_from_tie"] = bool(rk_res["resolved_from_tie"])

        agreed_rows.append(rec)

        # For Excel human stats
        stats_rows.append(
            {
                "id": rec.get("id"),
                "subset": rec.get("subset"),
                "language": rec.get("language"),
                "n_annotators_required_knowledge": n_valid,
                "required_knowledge_majority_share": rk_share,
                "required_knowledge_num_labels": rk_num_labels,
                "required_knowledge_tie_type": rk_res["tie_type"],
                "required_knowledge_resolved_from_tie": bool(rk_res["resolved_from_tie"]),
                "required_knowledge_agreement": rk_agreement,
                "csl_agreement": csl_agreement,
                "csl_provided": csl_provided,
            }
        )

    logging.info("Kept items: %d", len(agreed_rows))
    logging.info("Complex-tie items dropped: %d", len(tie_rows))
    logging.info("Tie-type distribution: %s", dict(tie_type_counter))
    logging.info("Provided vs rule CS/CA mismatches: %d", provided_vs_rule_mismatch)

    write_jsonl(Path(args.out_agreed), agreed_rows)
    write_jsonl(Path(args.out_ties), tie_rows)
    logging.info("Wrote JSONL: %s | %s", args.out_agreed, args.out_ties)

    # ---- Excel ----
    df = pd.DataFrame(stats_rows)

    summary = [
        ("n_items_total", len(rows)),
        ("n_items_kept", len(agreed_rows)),
        ("n_items_complex_ties", len(tie_rows)),
        ("provided_vs_rule_mismatch_n", provided_vs_rule_mismatch),
    ]
    for k, v in sorted(tie_type_counter.items()):
        summary.append((f"rk_tie_type_{k}", int(v)))

    if len(df) > 0:
        summary.extend(
            [
                ("human_n_annotators_min", int(df["n_annotators_required_knowledge"].min())),
                ("human_n_annotators_max", int(df["n_annotators_required_knowledge"].max())),
                ("human_unanimous_n", int((df["required_knowledge_majority_share"] == 1.0).sum())),
                ("human_tie_resolved_n", int((df["required_knowledge_resolved_from_tie"] == True).sum())),  # noqa: E712
                ("human_avg_majority_share", float(df["required_knowledge_majority_share"].dropna().mean())),
            ]
        )
    summary_df = pd.DataFrame(summary, columns=["metric", "value"])

    # Counts
    counts_rows = []
    for field in ["required_knowledge_agreement", "csl_agreement", "csl_provided"]:
        vc = df[field].value_counts(dropna=False).to_dict() if len(df) else {}
        for label, c in sorted(vc.items(), key=lambda x: str(x[0])):
            counts_rows.append({"field": field, "label": label, "count": int(c)})
    counts_df = pd.DataFrame(counts_rows)

    # AnnotatorCounts
    annot_counts = df["n_annotators_required_knowledge"].value_counts(dropna=False).sort_index().reset_index()
    annot_counts.columns = ["n_annotators", "n_items"]

    # HumanDisagreement
    human_disagreement = df[
        [
            "id",
            "subset",
            "language",
            "n_annotators_required_knowledge",
            "required_knowledge_agreement",
            "required_knowledge_majority_share",
            "required_knowledge_num_labels",
            "required_knowledge_tie_type",
            "required_knowledge_resolved_from_tie",
            "csl_agreement",
            "csl_provided",
        ]
    ]

    # Sanity check table: provided vs rule
    sanity = df.dropna(subset=["csl_agreement", "csl_provided"]).copy()
    sanity["provided_equals_rule"] = sanity["csl_provided"] == sanity["csl_agreement"]
    sanity_summary = sanity["provided_equals_rule"].value_counts(dropna=False).reset_index()
    sanity_summary.columns = ["provided_equals_rule", "count"]

    out_xlsx = Path(args.out_xlsx)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        counts_df.to_excel(writer, sheet_name="Counts", index=False)
        annot_counts.to_excel(writer, sheet_name="AnnotatorCounts", index=False)
        human_disagreement.to_excel(writer, sheet_name="HumanDisagreement", index=False)
        sanity_summary.to_excel(writer, sheet_name="Provided_vs_Rule", index=False)

    logging.info("Wrote Excel: %s", out_xlsx)


if __name__ == "__main__":
    main()


