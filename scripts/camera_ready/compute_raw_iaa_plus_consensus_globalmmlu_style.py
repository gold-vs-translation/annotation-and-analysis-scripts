#!/usr/bin/env python3
"""
Compute raw inter-annotator agreement (IAA) for multilingual CS/CA annotations.

Input structure:
  ROOT/
    DA/
      ann0172___<uuid>.jsonl
      ann3090___<uuid>.jsonl
      ann6129___<uuid>.jsonl
    ET/
      ...
    LT/
      ...
    NB/
      ...
    SV/
      ...

This script keeps three levels separate:

A) IAA on explicit cultural_sensitivity_label
   - Uses the annotators' actual CS/CA labels.

B) IAA on annotator-level derived CS/CA from required_knowledge
   - Per annotator:
       required_knowledge == "none" -> CA
       required_knowledge in {"cultural", "regional", "dialect"} -> CS
   - Agreement is computed over these derived annotator-level labels.

C) Item-level GlobalMMLU-style derived CS/CA consensus
   - Per item, from all required_knowledge votes:
       CA iff #none > n_valid / 2
       else CS
   - This is a consensus label, not an IAA metric.

Outputs:
  raw_votes_long.tsv
  raw_iaa_summary.tsv
  raw_item_level_consensus.tsv
  raw_item_disagreements.tsv
  raw_invalid_or_missing_summary.tsv
"""

import argparse
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import numpy as np

VALID_RK = {"none", "cultural", "regional", "dialect"}
NON_NONE_RK = {"cultural", "regional", "dialect"}
VALID_CSL = {"CA", "CS"}

FIELDS_FOR_IAA = [
    "cultural_sensitivity_label",
    "cultural_sensitivity_from_required_knowledge",
    "required_knowledge",
    "time_sensitive",
    "subject_category",
]

FIELD_LABELS = {
    "cultural_sensitivity_label": ["CA", "CS"],
    "cultural_sensitivity_from_required_knowledge": ["CA", "CS"],
    "required_knowledge": ["none", "cultural", "regional", "dialect"],
    "time_sensitive": ["No", "Yes"],
    # subject_category uses observed labels
}

ANNOTATOR_NAMES = {
    'ann0172': 'DA_annotator_01',
    'ann3090': 'DA_annotator_02',
    'ann6129': 'DA_annotator_03',
    'ann7708': 'ET_annotator_03',
    'ann4956': 'LT_annotator_01',
    'ann8933': 'NB_annotator_03',
    'ann8395': 'SV_annotator_02',
    'ann3615': 'SV_annotator_01',
    'ann6325': 'LT_annotator_02',
    'ann4477': 'NB_annotator_01',
    'ann9154': 'SV_annotator_03',
    'ann8062': 'LT_annotator_03',
    'ann1410': 'ET_annotator_01',
    'ann6023': 'ET_annotator_02',
    'ann7101': 'NB_annotator_02',
}


def parse_file_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
    m = re.match(r"(ann\d+)___(.+)\.jsonl$", path.name)
    if not m:
        return None, None
    return m.group(1), m.group(2)

def norm_basic(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "null":
        return None
    s = re.sub(r"\s+", " ", s)
    return s

def norm_required_knowledge(x: Any) -> Optional[str]:
    if x is None:
        return None

    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "null":
        return None

    s_low = re.sub(r"\s+", " ", s).lower()

    if s_low in {"none", "cultural", "regional", "dialect"}:
        return s_low

    return None


def norm_csl(x: Any) -> Optional[str]:
    s = norm_basic(x)
    if s is None:
        return None

    s_up = s.upper()
    aliases = {
        "CULTURALLY SENSITIVE": "CS",
        "CULTURALLY AGNOSTIC": "CA",
        "SENSITIVE": "CS",
        "AGNOSTIC": "CA",
    }
    s_up = aliases.get(s_up, s_up)

    if s_up in VALID_CSL:
        return s_up

    return None


def norm_yes_no(x: Any) -> Optional[str]:
    s = norm_basic(x)
    if s is None:
        return None

    s_low = s.lower()
    if s_low in {"yes", "y", "true"}:
        return "Yes"
    if s_low in {"no", "n", "false"}:
        return "No"

    return None


def norm_subject_category(x: Any) -> Optional[str]:
    s = norm_basic(x)
    if s is None:
        return None
    return s


def extract_response(item: Dict[str, Any], field: str, expected_user_id: str) -> Any:
    responses = item.get("responses", {})
    values = responses.get(field, [])

    if not values:
        return None

    matching = [v for v in values if v.get("user_id") == expected_user_id]
    if matching:
        values = matching

    if len(values) == 1:
        return values[0].get("value")

    # Rare case: multiple values for the same field/user.
    # Return the first non-empty value, otherwise None.
    for v in values:
        raw = v.get("value")
        if raw is not None and str(raw).strip() != "":
            return raw

    return None


def derive_csl_from_single_rk(rk: Optional[str]) -> Optional[str]:
    """
    Annotator-level derived CS/CA for IAA.

    This is used for B:
      none -> CA
      cultural/regional/dialect -> CS
    """
    if rk is None:
        return None
    if rk == "none":
        return "CA"
    if rk in NON_NONE_RK:
        return "CS"
    return None


def derive_globalmmlu_csl_from_rk_votes(rk_votes: List[Optional[str]]) -> Optional[str]:
    """
    Item-level GlobalMMLU-style CS/CA consensus.

    CA iff strict majority of valid required_knowledge votes is none.
    Else CS.

    Examples with 3 annotators:
      none, none, regional -> CA
      none, regional, regional -> CS
      none, cultural, regional -> CS
      cultural, regional, dialect -> CS
    """
    cleaned = [v for v in rk_votes if v in VALID_RK]
    n = len(cleaned)

    if n == 0:
        return None

    n_none = sum(v == "none" for v in cleaned)
    return "CA" if n_none > (n / 2) else "CS"


def vote_counts(values: List[Optional[str]]) -> Tuple[Dict[str, int], int]:
    cleaned = [v for v in values if v is not None]
    counts = Counter(cleaned)
    return dict(counts), len(cleaned)


def simple_majority(values: List[Optional[str]]) -> Tuple[Optional[str], str, Optional[float], Dict[str, int]]:
    """
    Generic majority for item-level inspection.

    Returns:
      winner, tie_type, majority_share, counts
    """
    counts, n = vote_counts(values)

    if n == 0:
        return None, "missing", None, counts

    max_count = max(counts.values())
    winners = sorted([k for k, v in counts.items() if v == max_count])
    majority_share = max_count / n

    if len(winners) == 1:
        return winners[0], "none", majority_share, counts

    return None, "tie", majority_share, counts


def resolve_required_knowledge_consensus(
    rk_votes: List[Optional[str]],
) -> Tuple[Optional[str], str, Optional[float], Dict[str, int], List[str]]:
    """
    Required-knowledge consensus following the GlobalMMLU-Lite script style:

    - unique majority -> winner
    - tie of exactly {none, one_non_none} -> resolve to the non-none label
    - all other ties -> complex_tie, no consensus winner
    """
    cleaned = [v for v in rk_votes if v in VALID_RK]
    counts = Counter(cleaned)
    n = len(cleaned)

    if n == 0:
        return None, "missing", None, dict(counts), []

    max_count = max(counts.values())
    max_labels = sorted([k for k, v in counts.items() if v == max_count])
    majority_share = max_count / n

    if len(max_labels) == 1:
        return max_labels[0], "none", majority_share, dict(counts), max_labels

    s = set(max_labels)
    if len(s) == 2 and "none" in s:
        other = next(iter(s - {"none"}))
        if other in NON_NONE_RK:
            return other, "tie_none_vs_non_none_resolved", majority_share, dict(counts), max_labels

    return None, "complex_tie", majority_share, dict(counts), max_labels


def read_votes(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    invalid_rows = []

    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir():
            continue

        language_from_dir = lang_dir.name

        for path in sorted(lang_dir.glob("*.jsonl")):
            annotator_id, user_id = parse_file_name(path)
            if annotator_id is None or user_id is None:
                continue

            annotator_name = ANNOTATOR_NAMES.get(annotator_id, annotator_id)

            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    item = json.loads(line)
                    metadata = item.get("metadata", {})
                    sample_id = metadata.get("sample_id") or item.get("id")
                    language = metadata.get("language") or language_from_dir

                    raw_rk = extract_response(item, "required_knowledge", user_id)
                    raw_csl = extract_response(item, "cultural_sensitivity_label", user_id)
                    raw_ts = extract_response(item, "time_sensitive", user_id)
                    raw_subj = extract_response(item, "subject_category", user_id)

                    rk = norm_required_knowledge(raw_rk)
                    csl = norm_csl(raw_csl)
                    ts = norm_yes_no(raw_ts)
                    subj = norm_subject_category(raw_subj)

                    # Track invalid non-empty values.
                    if raw_rk is not None and norm_basic(raw_rk) is not None and rk is None:
                        invalid_rows.append({
                            "language": language,
                            "sample_id": sample_id,
                            "annotator_id": annotator_id,
                            "field": "required_knowledge",
                            "raw_value": raw_rk,
                            "file": str(path),
                            "line_no": line_no,
                        })
                    if raw_csl is not None and norm_basic(raw_csl) is not None and csl is None:
                        invalid_rows.append({
                            "language": language,
                            "sample_id": sample_id,
                            "annotator_id": annotator_id,
                            "field": "cultural_sensitivity_label",
                            "raw_value": raw_csl,
                            "file": str(path),
                            "line_no": line_no,
                        })
                    if raw_ts is not None and norm_basic(raw_ts) is not None and ts is None:
                        invalid_rows.append({
                            "language": language,
                            "sample_id": sample_id,
                            "annotator_id": annotator_id,
                            "field": "time_sensitive",
                            "raw_value": raw_ts,
                            "file": str(path),
                            "line_no": line_no,
                        })

                    rows.append({
                        "language": language,
                        "sample_id": sample_id,
                        "suite": metadata.get("suite"),
                        "task": metadata.get("task"),
                        "subset": metadata.get("subset"),
                        "annotator_id": annotator_id,
                        "annotator_name": annotator_name,
                        "user_id": user_id,
                        "status": item.get("status"),
                        "required_knowledge": rk,
                        "cultural_sensitivity_label": csl,
                        "cultural_sensitivity_from_required_knowledge": derive_csl_from_single_rk(rk),
                        "time_sensitive": ts,
                        "subject_category": subj,
                    })

    return pd.DataFrame(rows), pd.DataFrame(invalid_rows)


def pairwise_agreement(values: List[str]) -> float:
    pairs = list(combinations(values, 2))
    if not pairs:
        return float("nan")
    return sum(a == b for a, b in pairs) / len(pairs)


def fleiss_kappa_complete(matrix: pd.DataFrame, labels: List[str]) -> Optional[float]:
    """
    Fleiss' kappa for complete data with fixed number of raters per item.
    matrix: rows = items, columns = annotators.
    """
    if matrix.empty:
        return None

    n_items = len(matrix)
    n_raters = matrix.shape[1]

    if n_items == 0 or n_raters < 2:
        return None

    count_rows = []
    for _, row in matrix.iterrows():
        cnt = Counter(row.tolist())
        count_rows.append([cnt.get(label, 0) for label in labels])

    counts = pd.DataFrame(count_rows, columns=labels)
    n_per_item = counts.sum(axis=1)

    if not (n_per_item == n_raters).all():
        return None

    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()

    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = (p_j ** 2).sum()

    if math.isclose(1 - p_e, 0.0):
        return None

    return float((p_bar - p_e) / (1 - p_e))


def krippendorff_alpha_nominal_complete(matrix: pd.DataFrame, labels: List[str]) -> Optional[float]:
    """
    Nominal Krippendorff alpha for complete data using pairwise disagreement.
    """
    if matrix.empty:
        return None

    observed_pairwise_agreements = []
    all_values = []

    for _, row in matrix.iterrows():
        vals = row.tolist()
        all_values.extend(vals)
        observed_pairwise_agreements.append(pairwise_agreement(vals))

    p_observed_agreement = sum(observed_pairwise_agreements) / len(observed_pairwise_agreements)
    d_observed = 1 - p_observed_agreement

    counts = Counter(all_values)
    total = sum(counts.values())

    if total == 0:
        return None

    probs = [counts.get(label, 0) / total for label in labels]
    d_expected = 1 - sum(p ** 2 for p in probs)

    if math.isclose(d_expected, 0.0):
        return None

    return float(1 - d_observed / d_expected)


def gwet_ac1_complete(matrix: pd.DataFrame, labels: List[str]) -> Optional[float]:
    """
    Gwet's AC1, useful as a prevalence-robust complement to kappa.
    """
    if matrix.empty:
        return None

    q = len(labels)
    if q < 2:
        return None

    item_pairwise_agreements = []
    all_values = []

    for _, row in matrix.iterrows():
        vals = row.tolist()
        all_values.extend(vals)
        item_pairwise_agreements.append(pairwise_agreement(vals))

    p_a = sum(item_pairwise_agreements) / len(item_pairwise_agreements)

    counts = Counter(all_values)
    total = sum(counts.values())

    if total == 0:
        return None

    probs = [counts.get(label, 0) / total for label in labels]
    p_e = sum(p * (1 - p) for p in probs) / (q - 1)

    if math.isclose(1 - p_e, 0.0):
        return None

    return float((p_a - p_e) / (1 - p_e))


def cohen_kappa_pair(a: List[str], b: List[str], labels: List[str]) -> Optional[float]:
    if len(a) == 0:
        return None

    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n

    ca = Counter(a)
    cb = Counter(b)
    p_e = sum((ca.get(label, 0) / n) * (cb.get(label, 0) / n) for label in labels)

    if math.isclose(1 - p_e, 0.0):
        return None

    return float((p_o - p_e) / (1 - p_e))


BOOTSTRAP_METRICS = [
    "all_annotators_same_pct",
    "pairwise_agreement",
    "mean_majority_share",
    "fleiss_kappa",
    "krippendorff_alpha_nominal",
    "gwet_ac1",
    "pabak_binary",
]


def stable_seed(base_seed: int, *parts: str) -> int:
    """
    Stable deterministic seed independent of Python's randomized hash().
    """
    acc = int(base_seed) % (2**32 - 1)
    text = "|".join(str(p) for p in parts)
    for ch in text:
        acc = (acc * 131 + ord(ch)) % (2**32 - 1)
    return acc


def metric_bundle_from_complete_matrix(
    complete: pd.DataFrame,
    labels: List[str],
) -> Dict[str, Optional[float]]:
    """
    Compute the bootstrap-target metrics on a complete item x annotator matrix.
    """
    if complete.empty:
        return {m: None for m in BOOTSTRAP_METRICS}

    item_pairwise = complete.apply(lambda row: pairwise_agreement(row.tolist()), axis=1)
    all_same = complete.apply(lambda row: len(set(row.tolist())) == 1, axis=1)
    majority_share = complete.apply(
        lambda row: Counter(row.tolist()).most_common(1)[0][1] / len(row.tolist()),
        axis=1,
    )

    p_pairwise = float(item_pairwise.mean())
    pabak_binary = 2 * p_pairwise - 1 if len(labels) == 2 else None

    return {
        "all_annotators_same_pct": float(100 * all_same.sum() / len(complete)),
        "pairwise_agreement": p_pairwise,
        "mean_majority_share": float(majority_share.mean()),
        "fleiss_kappa": fleiss_kappa_complete(complete, labels),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal_complete(complete, labels),
        "gwet_ac1": gwet_ac1_complete(complete, labels),
        "pabak_binary": pabak_binary,
    }


def bootstrap_cis_for_complete_matrix(
    complete: pd.DataFrame,
    labels: List[str],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> Dict[str, Optional[float]]:
    """
    Percentile bootstrap CIs by resampling items with replacement.

    Unit of resampling: item rows.
    Do NOT resample individual ratings, because annotator dependence is within item.
    """
    out: Dict[str, Optional[float]] = {}

    if n_bootstrap <= 0 or complete.empty:
        return out

    n_items = len(complete)
    rng = np.random.default_rng(seed)

    estimates = {metric: [] for metric in BOOTSTRAP_METRICS}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_items, size=n_items)
        sample = complete.iloc[idx].reset_index(drop=True)

        vals = metric_bundle_from_complete_matrix(sample, labels)

        for metric, value in vals.items():
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            estimates[metric].append(float(value))

    alpha = (100.0 - ci_level) / 2.0
    lo_q = alpha
    hi_q = 100.0 - alpha

    for metric, vals in estimates.items():
        out[f"{metric}_ci_low"] = None
        out[f"{metric}_ci_high"] = None
        out[f"{metric}_ci_n"] = len(vals)

        if len(vals) > 0:
            out[f"{metric}_ci_low"] = float(np.percentile(vals, lo_q))
            out[f"{metric}_ci_high"] = float(np.percentile(vals, hi_q))

    out["bootstrap_n_requested"] = int(n_bootstrap)
    out["bootstrap_ci_level"] = float(ci_level)

    return out

def summarize_iaa_for_field(
    votes: pd.DataFrame,
    language: str,
    field: str,
    n_bootstrap: int = 0,
    ci_level: float = 95.0,
    seed: int = 1,
) -> Dict[str, Any]:
    sub = votes[votes["language"] == language].copy()

    wide = sub.pivot_table(
        index="sample_id",
        columns="annotator_id",
        values=field,
        aggfunc="first",
    )

    n_items_total = len(wide)
    n_annotators_seen = len(wide.columns)

    complete = wide.dropna(axis=0, how="any")
    n_items_complete = len(complete)

    observed_labels = sorted({
        x
        for x in complete.to_numpy().ravel().tolist()
        if pd.notna(x)
    })

    labels = FIELD_LABELS.get(field) or observed_labels

    base = {
        "language": language,
        "field": field,
        "annotators": ",".join(sorted(wide.columns.tolist())),
        "n_annotators_seen": n_annotators_seen,
        "n_items_total": int(n_items_total),
        "n_items_complete": int(n_items_complete),
        "coverage_complete_pct": 100 * n_items_complete / n_items_total if n_items_total else None,
    }

    if n_items_complete == 0 or len(labels) == 0:
        return base

    item_pairwise = complete.apply(lambda row: pairwise_agreement(row.tolist()), axis=1)
    all_same = complete.apply(lambda row: len(set(row.tolist())) == 1, axis=1)
    majority_share = complete.apply(
        lambda row: Counter(row.tolist()).most_common(1)[0][1] / len(row.tolist()),
        axis=1,
    )

    n_all_same = int(all_same.sum())

    if complete.shape[1] == 3:
        n_2_vs_1 = int((majority_share == 2 / 3).sum())
        n_three_way = int((majority_share == 1 / 3).sum())
    else:
        n_2_vs_1 = None
        n_three_way = None

    pair_kappas = []
    for ann_a, ann_b in combinations(complete.columns.tolist(), 2):
        pair = complete[[ann_a, ann_b]].dropna()
        k = cohen_kappa_pair(pair[ann_a].tolist(), pair[ann_b].tolist(), labels)
        if k is not None:
            pair_kappas.append(k)

    pooled_counts = Counter(complete.to_numpy().ravel().tolist())
    label_distribution = "; ".join(
        f"{label}={pooled_counts.get(label, 0)}" for label in labels
    )

    p_pairwise = float(item_pairwise.mean())
    pabak_binary = 2 * p_pairwise - 1 if len(labels) == 2 else None

    base.update({
        "n_all_annotators_same": n_all_same,
        "all_annotators_same_pct": 100 * n_all_same / n_items_complete,
        "pairwise_agreement": p_pairwise,
        "mean_majority_share": float(majority_share.mean()),
        "n_2_vs_1": n_2_vs_1,
        "n_three_way_tie": n_three_way,
        "fleiss_kappa": fleiss_kappa_complete(complete, labels),
        "krippendorff_alpha_nominal": krippendorff_alpha_nominal_complete(complete, labels),
        "gwet_ac1": gwet_ac1_complete(complete, labels),
        "pabak_binary": pabak_binary,
        "pairwise_cohen_kappa_mean": float(sum(pair_kappas) / len(pair_kappas)) if pair_kappas else None,
        "pairwise_cohen_kappa_min": float(min(pair_kappas)) if pair_kappas else None,
        "pairwise_cohen_kappa_max": float(max(pair_kappas)) if pair_kappas else None,
        "label_distribution_complete_ratings": label_distribution,
    })
    
    if n_bootstrap and n_bootstrap > 0:
        ci_seed = stable_seed(seed, language, field)
        base.update(
            bootstrap_cis_for_complete_matrix(
                complete=complete,
                labels=labels,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=ci_seed,
            )
        )

    return base


def build_item_level_consensus(votes: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (language, sample_id), g in votes.groupby(["language", "sample_id"], dropna=False):
        g = g.sort_values("annotator_id")

        rk_votes = g["required_knowledge"].tolist()
        explicit_csl_votes = g["cultural_sensitivity_label"].tolist()
        derived_csl_votes = g["cultural_sensitivity_from_required_knowledge"].tolist()

        rk_consensus, rk_tie_type, rk_majority_share, rk_counts, rk_max_labels = (
            resolve_required_knowledge_consensus(rk_votes)
        )

        explicit_csl_consensus, explicit_csl_tie_type, explicit_csl_majority_share, explicit_csl_counts = (
            simple_majority(explicit_csl_votes)
        )

        derived_csl_majority, derived_csl_tie_type, derived_csl_majority_share, derived_csl_counts = (
            simple_majority(derived_csl_votes)
        )

        globalmmlu_csl = derive_globalmmlu_csl_from_rk_votes(rk_votes)

        rows.append({
            "language": language,
            "sample_id": sample_id,
            "suite": g["suite"].dropna().iloc[0] if g["suite"].notna().any() else None,
            "task": g["task"].dropna().iloc[0] if g["task"].notna().any() else None,
            "subset": g["subset"].dropna().iloc[0] if g["subset"].notna().any() else None,
            "annotators": ",".join(g["annotator_id"].tolist()),
            "n_annotators": int(g["annotator_id"].nunique()),

            "required_knowledge_votes": " | ".join(
                f"{r.annotator_id}={r.required_knowledge}" for r in g.itertuples()
            ),
            "required_knowledge_vote_counts": json.dumps(rk_counts, ensure_ascii=False, sort_keys=True),
            "required_knowledge_consensus": rk_consensus,
            "required_knowledge_tie_type": rk_tie_type,
            "required_knowledge_majority_share": rk_majority_share,
            "required_knowledge_max_labels": json.dumps(rk_max_labels, ensure_ascii=False),

            "explicit_csl_votes": " | ".join(
                f"{r.annotator_id}={r.cultural_sensitivity_label}" for r in g.itertuples()
            ),
            "explicit_csl_vote_counts": json.dumps(explicit_csl_counts, ensure_ascii=False, sort_keys=True),
            "explicit_csl_majority_consensus": explicit_csl_consensus,
            "explicit_csl_tie_type": explicit_csl_tie_type,
            "explicit_csl_majority_share": explicit_csl_majority_share,

            "derived_csl_votes_from_rk": " | ".join(
                f"{r.annotator_id}={r.cultural_sensitivity_from_required_knowledge}" for r in g.itertuples()
            ),
            "derived_csl_vote_counts": json.dumps(derived_csl_counts, ensure_ascii=False, sort_keys=True),
            "derived_csl_simple_majority": derived_csl_majority,
            "derived_csl_simple_majority_tie_type": derived_csl_tie_type,
            "derived_csl_simple_majority_share": derived_csl_majority_share,

            # This is the GlobalMMLU-style consensus label:
            "globalmmlu_derived_csl_consensus_from_rk": globalmmlu_csl,
        })

    return pd.DataFrame(rows)


def build_item_disagreement_table(votes: pd.DataFrame) -> pd.DataFrame:
    rows = []

    key_fields = [
        "cultural_sensitivity_label",
        "cultural_sensitivity_from_required_knowledge",
        "required_knowledge",
    ]

    for (language, sample_id), g in votes.groupby(["language", "sample_id"], dropna=False):
        g = g.sort_values("annotator_id")

        row = {
            "language": language,
            "sample_id": sample_id,
            "annotators": ",".join(g["annotator_id"].tolist()),
        }

        has_disagreement = False

        for field in key_fields:
            vals = g[field].tolist()
            valid_vals = [v for v in vals if v is not None]

            counts = Counter(valid_vals)
            n_labels = len(counts)

            if len(valid_vals) >= 2 and n_labels > 1:
                has_disagreement = True

            row[f"{field}_votes"] = " | ".join(
                f"{r.annotator_id}={getattr(r, field)}" for r in g.itertuples()
            )
            row[f"{field}_vote_counts"] = json.dumps(dict(counts), ensure_ascii=False, sort_keys=True)
            row[f"{field}_n_labels"] = n_labels
            row[f"{field}_all_same"] = bool(n_labels == 1 and len(valid_vals) == len(vals))

        if has_disagreement:
            rows.append(row)

    return pd.DataFrame(rows)


def build_invalid_or_missing_summary(votes: pd.DataFrame, invalid_values: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (language, annotator_id), g in votes.groupby(["language", "annotator_id"], dropna=False):
        row = {
            "language": language,
            "annotator_id": annotator_id,
            "annotator_name": g["annotator_name"].iloc[0],
            "n_rows": len(g),
        }

        for field in FIELDS_FOR_IAA:
            row[f"missing_{field}"] = int(g[field].isna().sum())

        rows.append(row)

    out = pd.DataFrame(rows)

    if not invalid_values.empty:
        invalid_summary = (
            invalid_values.groupby(["language", "annotator_id", "field"], dropna=False)
            .size()
            .reset_index(name="n_invalid_values")
        )
        invalid_summary["metric"] = "invalid_values"
    else:
        invalid_summary = pd.DataFrame(columns=[
            "language", "annotator_id", "field", "n_invalid_values", "metric"
        ])

    return out, invalid_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="Root directory with language subdirectories and three annotator JSONL files per language.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for TSV reports.",
    )

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of item-level bootstrap resamples for confidence intervals. Use 0 to disable.",
    )
    ap.add_argument(
        "--ci-level",
        type=float,
        default=95.0,
        help="Bootstrap confidence interval level, e.g. 95 for 95%% CIs.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260501,
        help="Base random seed for bootstrap resampling.",
    )    
    
    
    
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    votes, invalid_values = read_votes(root)

    if votes.empty:
        raise SystemExit("No votes found. Check --root path and JSONL filenames.")

    votes_out = out_dir / "raw_votes_long.tsv"
    votes.to_csv(votes_out, sep="\t", index=False)

    summary_rows = []
    for language in sorted(votes["language"].dropna().unique()):
        for field in FIELDS_FOR_IAA:
            summary_rows.append(
                summarize_iaa_for_field(
                    votes,
                    language,
                    field,
                    n_bootstrap=args.bootstrap,
                    ci_level=args.ci_level,
                    seed=args.seed,
                )
            )

    iaa_summary = pd.DataFrame(summary_rows)
    iaa_summary_out = out_dir / "raw_iaa_summary.tsv"
    iaa_summary.to_csv(iaa_summary_out, sep="\t", index=False)

    consensus = build_item_level_consensus(votes)
    consensus_out = out_dir / "raw_item_level_consensus.tsv"
    consensus.to_csv(consensus_out, sep="\t", index=False)

    disagreements = build_item_disagreement_table(votes)
    disagreements_out = out_dir / "raw_item_disagreements.tsv"
    disagreements.to_csv(disagreements_out, sep="\t", index=False)

    missing_summary, invalid_summary = build_invalid_or_missing_summary(votes, invalid_values)
    missing_out = out_dir / "raw_missing_summary.tsv"
    invalid_out = out_dir / "raw_invalid_values.tsv"
    invalid_summary_out = out_dir / "raw_invalid_values_summary.tsv"

    missing_summary.to_csv(missing_out, sep="\t", index=False)
    invalid_values.to_csv(invalid_out, sep="\t", index=False)
    invalid_summary.to_csv(invalid_summary_out, sep="\t", index=False)

    print("Wrote:")
    print(f"  {votes_out}")
    print(f"  {iaa_summary_out}")
    print(f"  {consensus_out}")
    print(f"  {disagreements_out}")
    print(f"  {missing_out}")
    print(f"  {invalid_out}")
    print(f"  {invalid_summary_out}")

    print()
    print("Key rows to inspect in raw_iaa_summary.tsv:")
    print("  field == cultural_sensitivity_label")
    print("  field == cultural_sensitivity_from_required_knowledge")
    print("  field == required_knowledge")


if __name__ == "__main__":
    main()