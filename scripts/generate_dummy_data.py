#!/usr/bin/env python3
"""Generate synthetic dummy data for all pipeline stages.

This enables reviewers to run every script without access to the real data.
All content is synthetic and does not represent real exam items.
"""

import csv
import json
import random
import os
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data" / "dummy"

LANGUAGES = ["DA", "ET", "LT", "NB", "SV"]
SUITES = {
    "DA": "DanishCitizenTest",
    "ET": "Exam-ET",
    "LT": "INCLUDE-base",
    "NB": "NRK-Quiz-QA",
    "SV": "SwedishFacts",
}
SUBSETS = {
    "DA": ["citizenship"],
    "ET": ["ajalugu", "bioloogia", "matemaatika", "geograafia", "filosoofia"],
    "LT": ["history", "science", "math"],
    "NB": ["general_knowledge"],
    "SV": ["general"],
}
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "philosophy", "high_school_world_history",
    "high_school_biology", "college_mathematics", "clinical_knowledge",
    "business_ethics", "sociology", "marketing",
]
RK_VALUES = ["none", "cultural", "regional", "dialect"]
CSL_VALUES = ["CS", "CA"]
TS_VALUES = ["Yes", "No"]
SUBJECT_CATS = ["STEM", "Humanities", "Social Sciences", "Medical", "Business", "Other"]
CULTURES = ["Western Culture", "Eastern Asian Culture", "Middle Eastern Culture", None]
REGIONS = ["North America", "Europe", "Asia", None]
COUNTRIES = ["United States (US)", "Germany (DE)", "Denmark (DK)", None]

DUMMY_TEXTS = [
    "What is the capital of {country}?\n(A) Berlin\n(B) Paris\n(C) London\n(D) Rome\n=> (A)",
    "Which philosopher argued for the categorical imperative?\n(A) Hume\n(B) Kant\n(C) Locke\n(D) Mill\n=> (B)",
    "Calculate the integral of x^2 dx from 0 to 1.\n(A) 1/3\n(B) 1/2\n(C) 1\n(D) 2/3\n=> (A)",
    "Which organ produces insulin?\n(A) Liver\n(B) Pancreas\n(C) Kidney\n(D) Heart\n=> (B)",
    "In which year did World War II end?\n(A) 1943\n(B) 1944\n(C) 1945\n(D) 1946\n=> (C)",
]
COUNTRY_NAMES = ["Denmark", "Estonia", "Lithuania", "Norway", "Sweden", "Germany", "France"]


def make_culture_ann(rk=None):
    rk = rk or random.choice(RK_VALUES)
    csl = "CA" if rk == "none" else "CS"
    ts = random.choice(TS_VALUES)
    sc = random.choice(SUBJECT_CATS)
    culture = random.choice(CULTURES) if rk == "cultural" else None
    region = random.choice(REGIONS) if rk == "regional" else None
    country = random.choice(COUNTRIES) if rk == "regional" else None

    return {
        "required_knowledge": rk,
        "time_sensitive": ts,
        "cultural_sensitivity_label": csl,
        "subject_category": sc,
        "culture": culture,
        "region": region,
        "country": country,
        "references": [],
        "other_references": [],
        "explanation": f"Synthetic item. Required knowledge: {rk}.",
    }


def make_culture_ann_ref(n_annotators=3):
    """Multi-annotator reference (lists of labels)."""
    rk_list = [random.choice(RK_VALUES) for _ in range(n_annotators)]
    ts_list = [random.choice(TS_VALUES) for _ in range(n_annotators)]
    csl_list = [("CA" if rk == "none" else "CS") for rk in rk_list]

    from collections import Counter
    rk_winner = Counter(rk_list).most_common(1)[0][0]

    return {
        "required_knowledge": rk_list,
        "time_sensitive": ts_list,
        "cultural_sensitivity_label": csl_list,
        "required_knowledge_agreement": rk_winner,
        "time_sensitive_agreement": Counter(ts_list).most_common(1)[0][0],
        "cultural_sensitivity_label_agreement": "CA" if rk_winner == "none" else "CS",
    }


def gen_item(suite, task, lang, subset, idx):
    text = random.choice(DUMMY_TEXTS).replace("{country}", random.choice(COUNTRY_NAMES))
    return {
        "id": f"{subset}/{idx}",
        "suite": suite,
        "task": task,
        "subset": subset,
        "split": "test",
        "language": lang,
        "text": text,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):>4d} rows -> {path.relative_to(DATA_DIR)}")


def gen_annotated_datasets():
    """Stage A: GPT-annotated JSONL files per dataset."""
    out_dir = DATA_DIR / "annotated"
    for lang in LANGUAGES:
        suite = SUITES[lang]
        subsets = SUBSETS[lang]
        rows = []
        for i in range(50):
            subset = random.choice(subsets)
            item = gen_item(suite, suite, lang, subset, i)
            item["culture_ann"] = make_culture_ann()
            rows.append(item)
        fname = f"text_{suite.lower().replace('-', '_')}_{lang}_test-annotated.jsonl"
        write_jsonl(out_dir / fname, rows)
    return out_dir


def gen_globalmmlu_lite():
    """Stage B: GlobalMMLU-Lite test file with human multi-annotator refs."""
    out_dir = DATA_DIR / "globalmmlulite"
    rows = []
    for i in range(200):
        subset = random.choice(MMLU_SUBJECTS)
        item = gen_item("Global-MMLU-Lite", "Global-MMLU-Lite", "EN", subset, i)
        item["culture_ann_ref"] = make_culture_ann_ref(n_annotators=3)
        item["culture_ann"] = make_culture_ann()
        rows.append(item)
    write_jsonl(out_dir / "text_globalmmlulite_EN_test.jsonl", rows)

    # Also make agreed + splits
    agreed = [r for r in rows if not any(
        isinstance(r["culture_ann_ref"].get("required_knowledge"), list)
        and len(set(r["culture_ann_ref"]["required_knowledge"])) > 2
        for _ in [None]
    )]
    write_jsonl(out_dir / "text_globalmmlulite_EN_test_agreed.jsonl", agreed)
    dev = agreed[:60]
    test = agreed[60:]
    write_jsonl(out_dir / "text_globalmmlulite_EN_test_devsplit.jsonl", dev)
    write_jsonl(out_dir / "text_globalmmlulite_EN_test_testsplit.jsonl", test)
    return out_dir


def gen_human_annotations():
    """Stage C: Human annotation export (post-review JSONL)."""
    out_dir = DATA_DIR / "human_annotations"
    rows = []
    annotators = {
        "DA": ("annotator_1", "uuid-ann-001"),
        "ET": ("annotator_3", "uuid-ann-004"),
        "LT": ("annotator_2", "uuid-ann-005"),
        "NB": ("annotator_5", "uuid-ann-006"),
        "SV": ("annotator_4", "uuid-ann-007"),
    }
    for lang in LANGUAGES:
        suite = SUITES[lang]
        user, uid = annotators[lang]
        subsets = SUBSETS[lang]
        for i in range(20):
            subset = random.choice(subsets)
            rk = random.choice(RK_VALUES)
            rows.append({
                "user": user,
                "user_id": uid,
                "task": suite,
                "suite": suite,
                "subset": subset,
                "language": lang,
                "sample_id": f"{subset}/{i}",
                "required_knowledge": rk,
                "cultural_sensitivity_label": "CA" if rk == "none" else "CS",
                "time_sensitive": random.choice(TS_VALUES),
                "subject_category": random.choice(SUBJECT_CATS),
                "references": [],
                "explanation": "Synthetic annotation.",
            })

    write_jsonl(out_dir / "all_user_annotations.jsonl", rows)

    # DA has 3 annotators for IAA
    da_extra = []
    for ann_idx in [2, 3]:
        user = f"annotator_{ann_idx}"
        uid = f"uuid-ann-00{ann_idx}"
        for i in range(20):
            subset = random.choice(SUBSETS["DA"])
            rk = random.choice(RK_VALUES)
            da_extra.append({
                "user": user, "user_id": uid,
                "task": "DanishCitizenTest", "suite": "DanishCitizenTest",
                "subset": subset, "language": "DA",
                "sample_id": f"{subset}/{i}",
                "required_knowledge": rk,
                "cultural_sensitivity_label": "CA" if rk == "none" else "CS",
                "time_sensitive": random.choice(TS_VALUES),
                "subject_category": random.choice(SUBJECT_CATS),
                "references": [], "explanation": "Synthetic annotation.",
            })
    write_jsonl(out_dir / "all_user_annotations_with_da_iaa.jsonl", rows + da_extra)
    return out_dir


def gen_pre_review_csvs():
    """Stage D: Pre-review CSVs for QA/validation tables."""
    out_dir = DATA_DIR / "pre_review_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotator_map = {
        "DA": [("DA_ann001___uuid-ann-001", 20), ("DA_ann002___uuid-ann-002", 20), ("DA_ann003___uuid-ann-003", 20)],
        "ET": [("ET_ann004___uuid-ann-004", 20)],
        "LT": [("LT_ann005___uuid-ann-005", 20)],
        "NB": [("NB_ann006___uuid-ann-006", 20)],
        "SV": [("SV_ann007___uuid-ann-007", 20)],
    }
    severity_choices = ["OK", "WARN", "ERROR"]
    fields = [
        "sample_id", "annot_required_knowledge", "annot_csl",
        "annot_time_sensitive", "annot_subject_category",
        "severity", "needs_fix",
        "missing_fields", "contradictions", "format_issues",
        "span_issues", "content_sanity_flags",
        "suggest_required_knowledge", "suggest_csl",
        "suggest_time_sensitive", "suggest_subject_category",
        "comment",
    ]
    for lang, annotators in annotator_map.items():
        subsets = SUBSETS[lang]
        for ann_prefix, n_items in annotators:
            fname = f"{ann_prefix}.csv"
            rows = []
            for i in range(n_items):
                subset = random.choice(subsets)
                rk = random.choice(RK_VALUES)
                sev = random.choice(severity_choices)
                rows.append({
                    "sample_id": f"{subset}/{i}",
                    "annot_required_knowledge": rk,
                    "annot_csl": "CA" if rk == "none" else "CS",
                    "annot_time_sensitive": random.choice(TS_VALUES),
                    "annot_subject_category": random.choice(SUBJECT_CATS),
                    "severity": sev,
                    "needs_fix": "YES" if sev == "ERROR" else "NO",
                    "missing_fields": "",
                    "contradictions": "",
                    "format_issues": "",
                    "span_issues": "",
                    "content_sanity_flags": "",
                    "suggest_required_knowledge": "",
                    "suggest_csl": "",
                    "suggest_time_sensitive": "",
                    "suggest_subject_category": "",
                    "comment": "Synthetic QA review.",
                })
            path = out_dir / fname
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"  Wrote {len(rows):>4d} rows -> {path.relative_to(DATA_DIR)}")
    return out_dir


def gen_merged_with_results():
    """Stage E+F: Merged annotations with evaluation results."""
    out_dir = DATA_DIR / "merged"
    models = ["model_a", "model_b", "model_c", "model_d", "model_e"]
    rows = []
    for lang in LANGUAGES:
        suite = SUITES[lang]
        subsets = SUBSETS[lang]
        for model in models:
            for i in range(30):
                subset = random.choice(subsets)
                rk = random.choice(RK_VALUES)
                rows.append({
                    "suite": suite,
                    "task": suite,
                    "language": lang,
                    "subset": subset,
                    "sample_id": f"{subset}/{i}",
                    "model_name": model,
                    "metric": "acc",
                    "sample_score": random.choice([0, 1]),
                    "user": f"annotator_{LANGUAGES.index(lang) + 1}",
                    "user_id": f"uuid-ann-00{LANGUAGES.index(lang) + 1}",
                    "required_knowledge": rk,
                    "cultural_sensitivity_label": "CA" if rk == "none" else "CS",
                    "time_sensitive": random.choice(TS_VALUES),
                    "subject_category": random.choice(SUBJECT_CATS),
                })
    write_jsonl(out_dir / "all_merged_samplelogs.jsonl", rows)

    # Also make a slim annotations-only file
    seen = set()
    ann_rows = []
    for r in rows:
        key = (r["suite"], r["task"], r["language"], r["subset"], r["sample_id"], r["user"])
        if key not in seen:
            seen.add(key)
            ann_rows.append({k: r[k] for k in [
                "suite", "task", "language", "subset", "sample_id",
                "user", "user_id", "required_knowledge",
                "cultural_sensitivity_label", "time_sensitive", "subject_category",
            ]})
    write_jsonl(out_dir / "all_annotations_slim.jsonl", ann_rows)
    return out_dir


def gen_sampling_input():
    """Stage C pre-step: annotated files for sampling.py."""
    # sampling.py reads *-annotated.jsonl from annotated dir
    # We already have those from gen_annotated_datasets
    pass


def gen_analyze_datasets_aliases():
    """Create filename aliases so analyze_datasets.py finds the dummy files."""
    import lzma
    src_dir = DATA_DIR / "annotated"
    aliases = {
        "text_swedishfacts_SV_train-annotated.jsonl": "text_swedishfacts_SV_test-annotated.jsonl",
        "text_skolprov_SV_train-annotated.jsonl": "text_swedishfacts_SV_test-annotated.jsonl",
        "text_DCT_DA_train-annotated.jsonl": "text_danishcitizentest_DA_test-annotated.jsonl",
        "text_examet_ET_train-annotated.jsonl": "text_exam_et_ET_test-annotated.jsonl",
        "text_includebase_LT_test-annotated.jsonl": "text_include_base_LT_test-annotated.jsonl",
        "text_includebase_ET_test-annotated.jsonl": "text_include_base_LT_test-annotated.jsonl",
        "text_nrkquizqa_NB_test-annotated.jsonl": "text_nrk_quiz_qa_NB_test-annotated.jsonl",
    }
    for alias, target in aliases.items():
        dst = src_dir / alias
        if not dst.exists():
            src = src_dir / target
            if src.exists():
                import shutil
                shutil.copy2(src, dst)
                print(f"  Alias: {alias} -> {target}")

    # GlobalMMLU-Lite
    lite_src = DATA_DIR / "globalmmlulite" / "text_globalmmlulite_EN_test_testsplit.jsonl"
    lite_dst = src_dir / "text_globalmmlulite_EN_test-annotated.jsonl"
    if lite_src.exists() and not lite_dst.exists():
        import shutil
        shutil.copy2(lite_src, lite_dst)
        print(f"  Alias: text_globalmmlulite_EN_test-annotated.jsonl")

    # GlobalMMLU .jsonl.xz
    xz_dst = src_dir / "text_globalmmlu_EN_test.jsonl.xz"
    if not xz_dst.exists():
        rows = []
        for i in range(20):
            rk = random.choice(RK_VALUES)
            rows.append(json.dumps({
                "id": f"test/{i}", "suite": "Global-MMLU", "task": "Global-MMLU",
                "subset": random.choice(MMLU_SUBJECTS), "language": "EN",
                "text": f"Synthetic question {i}?",
                "culture_ann": make_culture_ann(rk),
            }) + "\n")
        import lzma as _lzma
        with _lzma.open(str(xz_dst), "wt", encoding="utf-8") as f:
            f.writelines(rows)
        print(f"  Created: text_globalmmlu_EN_test.jsonl.xz")


def main():
    print("Generating dummy data for all pipeline stages...")
    print(f"Output directory: {DATA_DIR}\n")
    gen_annotated_datasets()
    gen_globalmmlu_lite()
    gen_human_annotations()
    gen_pre_review_csvs()
    gen_merged_with_results()
    gen_analyze_datasets_aliases()
    print("\nDone! All dummy data generated.")
    print(f"Total size: {sum(f.stat().st_size for f in DATA_DIR.rglob('*') if f.is_file()) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
