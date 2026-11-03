#!/usr/bin/env python3
"""
MC-Exam-Style Dataset Analyzer
===============================
Reads JSONL datasets, creates MMLU subject/category mappings, and generates
statistics tables for cross-benchmark comparison in the Project paper.

Usage:
    python analyze_datasets.py [--output-dir OUTPUT_DIR]

Output:
    - table1_dataset_overview.csv       (item counts, subjects, categories)
    - table2_cultural_annotations.csv   (CA/CS, required_knowledge, time_sensitive)
    - subject_mapping.csv               (native subjects → MMLU subjects/categories + confidence)
    - cross_validation.csv              (our mapping vs GPT-5.2 subject_category agreement)
    - cross_validation_summary.txt      (human-readable agreement report)
    - confidence_summary.csv            (how many items at each confidence level)
    - example_items_per_mapping.csv     (sample items per subject mapping for appendix)
    - full_subject_stats.csv            (detailed per-subject item counts)
"""

import json
import lzma
import os
import csv
import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset specifications: paths are resolved relative to --data-dir at runtime.
DATASET_SPECS = [
    {"name": "SwedishFacts",      "suite": "SwedishFacts",      "language": "SV", "file": "text_swedishfacts_SV_train-annotated.jsonl"},
    {"name": "Skolprov",          "suite": "Skolprov",          "language": "SV", "file": "text_skolprov_SV_train-annotated.jsonl"},
    {"name": "DanishCitizenTest", "suite": "DanishCitizenTest", "language": "DA", "file": "text_DCT_DA_train-annotated.jsonl"},
    {"name": "Exam-ET",           "suite": "Exam-ET",           "language": "ET", "file": "text_examet_ET_train-annotated.jsonl"},
    {"name": "INCLUDE-base (LT)", "suite": "INCLUDE-base",      "language": "LT", "file": "text_includebase_LT_test-annotated.jsonl"},
    {"name": "INCLUDE-base (ET)", "suite": "INCLUDE-base",      "language": "ET", "file": "text_includebase_ET_test-annotated.jsonl"},
    {"name": "NRK-Quiz-QA",       "suite": "NRK-Quiz-QA",       "language": "NB", "file": "text_nrkquizqa_NB_test-annotated.jsonl"},
    {"name": "Global-MMLU-Lite",  "suite": "Global-MMLU-Lite",  "language": "EN", "file": "text_globalmmlulite_EN_test-annotated.jsonl"},
    {"name": "Global-MMLU",       "suite": "Global-MMLU",       "language": "EN", "file": "text_globalmmlu_EN_test.jsonl.xz"},
]


def resolve_datasets(data_dir: Path) -> list:
    """Resolve dataset specs to full paths relative to data_dir."""
    datasets = []
    for spec in DATASET_SPECS:
        ds = dict(spec)
        ds["path"] = str(data_dir / spec["file"])
        datasets.append(ds)
    return datasets

# MMLU subject → category mapping
MMLU_SUBJECT_TO_CATEGORY = {
    'abstract_algebra': 'stem',
    'anatomy': 'stem',
    'astronomy': 'stem',
    'business_ethics': 'business',
    'clinical_knowledge': 'medical',
    'college_biology': 'stem',
    'college_chemistry': 'stem',
    'college_computer_science': 'stem',
    'college_mathematics': 'stem',
    'college_medicine': 'medical',
    'college_physics': 'stem',
    'computer_security': 'stem',
    'conceptual_physics': 'stem',
    'econometrics': 'social_sciences',
    'electrical_engineering': 'stem',
    'elementary_mathematics': 'stem',
    'formal_logic': 'humanities',
    'global_facts': 'other',
    'high_school_biology': 'stem',
    'high_school_chemistry': 'stem',
    'high_school_computer_science': 'stem',
    'high_school_european_history': 'humanities',
    'high_school_geography': 'social_sciences',
    'high_school_government_and_politics': 'social_sciences',
    'high_school_macroeconomics': 'social_sciences',
    'high_school_mathematics': 'stem',
    'high_school_microeconomics': 'social_sciences',
    'high_school_physics': 'stem',
    'high_school_psychology': 'social_sciences',
    'high_school_statistics': 'stem',
    'high_school_us_history': 'humanities',
    'high_school_world_history': 'humanities',
    'human_aging': 'medical',
    'human_sexuality': 'social_sciences',
    'international_law': 'humanities',
    'jurisprudence': 'humanities',
    'logical_fallacies': 'humanities',
    'machine_learning': 'stem',
    'management': 'business',
    'marketing': 'business',
    'medical_genetics': 'medical',
    'miscellaneous': 'other',
    'moral_disputes': 'humanities',
    'moral_scenarios': 'humanities',
    'nutrition': 'medical',
    'philosophy': 'humanities',
    'prehistory': 'humanities',
    'professional_accounting': 'business',
    'professional_law': 'humanities',
    'professional_medicine': 'medical',
    'professional_psychology': 'social_sciences',
    'public_relations': 'social_sciences',
    'security_studies': 'social_sciences',
    'sociology': 'social_sciences',
    'us_foreign_policy': 'social_sciences',
    'virology': 'medical',
    'world_religions': 'humanities',
}

# All valid MMLU categories
MMLU_CATEGORIES = sorted(set(MMLU_SUBJECT_TO_CATEGORY.values()))

# Normalisation map: MMLU category names ↔ GPT-5.2 subject_category names
# MMLU uses lowercase, GPT-5.2 uses title case.
MMLU_CAT_TO_ANN_CAT = {
    "stem": "STEM",
    "humanities": "Humanities",
    "social_sciences": "Social Sciences",
    "business": "Business",
    "medical": "Medical",
    "other": "Other",
}
ANN_CAT_TO_MMLU_CAT = {v: k for k, v in MMLU_CAT_TO_ANN_CAT.items()}

# -------------------------------------------------------------------------
# Translation dictionaries: native subject names → English
# -------------------------------------------------------------------------

SKOLPROV_SUBJECT_TRANSLATION = {
    "högskoleprovet": "Swedish Scholastic Aptitude Test",
    "kvantitativ": "Quantitative Reasoning",
    "verbal": "Verbal Reasoning",
    "ord": "Vocabulary (Word Comprehension)",
    "las": "Reading Comprehension",
    "läs": "Reading Comprehension",
    "mek": "Sentence Completion",
    "elk": "English Reading Comprehension",
    "nog": "Quantitative Comparisons (Sufficiency)",
    "dtk": "Diagrams, Tables & Maps",
    "xyz": "Quantitative Problem Solving",
    "kvant": "Quantitative Reasoning",
    "kunskapsprov_apotekare": "Professional Knowledge Test: Pharmacist",
    "kunskapsprov_audionom": "Professional Knowledge Test: Audiologist",
    "kunskapsprov_läkare": "Professional Knowledge Test: Physician",
    "kunskapsprov_tandläkare": "Professional Knowledge Test: Dentist",
    "matematik_och_fysikprovet": "Mathematics and Physics Test",
}

EXAMET_SUBJECT_TRANSLATION = {
    "ajalugu": "History",
    "bioloogia": "Biology",
    "eesti_keel": "Estonian Language",
    "eesti": "Estonian Language",
    "füüsika": "Physics",
    "geograafia": "Geography",
    "keemia": "Chemistry",
    "kirjandus": "Literature",
    "kunstiajalugu": "Art History",
    "matemaatika": "Mathematics",
    "muusika": "Music",
    "filosoofia": "Philosophy",
    "inglise_keel": "English Language",
    "saksa_keel": "German Language",
    "prantsuse_keel": "French Language",
    "vene_keel": "Russian Language",
    "ühiskonnaõpetus": "Social Studies",
    "yhiskonnaopetus": "Social Studies",
    "informaatika": "Computer Science",
    "majandus": "Economics",
    "riigikaitseõpetus": "National Defense",
    "loodusõpetus": "Natural Sciences",
    "eesti_keel_teise_keelena_gümnaasium": "Estonian as Second Language (Gymnasium)",
    "eesti_keel_teise_keelena_põhikool": "Estonian as Second Language (Basic School)",
}

DCT_SUBJECT_TRANSLATION = {
    "citizenship": "Citizenship Test",
    "indfødsretsprøve": "Citizenship Test",
    "medborgerskabsprøve": "Citizenship Examination",
    "residence": "Residence Permit Test",
}

# -------------------------------------------------------------------------
# Mapping: native subjects → MMLU subject(s), category, and confidence
# -------------------------------------------------------------------------
# Confidence levels:
#   "direct"         – 1:1 MMLU subject match (e.g. filosoofia → philosophy)
#   "approximate"    – plausible mapping but not exact (granularity/scope mismatch)
#   "no_equivalent"  – no MMLU subject covers this; category-level mapping only

NATIVE_SUBJECT_TO_MMLU = {
    # --- Skolprov ---
    "högskoleprovet": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "other", "confidence": "no_equivalent", "note": "Mixed aptitude test; no single MMLU subject covers this"},
    "ord": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Vocabulary/word comprehension; no MMLU equivalent for L1 vocab tests"},
    "las": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Reading comprehension (SV); MMLU has no reading comprehension subject"},
    "läs": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Reading comprehension (SV); MMLU has no reading comprehension subject"},
    "mek": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Sentence completion (SV); tests linguistic skill, not domain knowledge"},
    "elk": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "English reading comprehension; language proficiency test"},
    "nog": {"mmlu_subjects": ["high_school_mathematics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Data sufficiency / quantitative; tests reasoning more than curriculum math"},
    "dtk": {"mmlu_subjects": ["high_school_statistics", "high_school_geography"], "mmlu_category": "stem", "confidence": "approximate", "note": "Diagrams, tables, maps; spans multiple MMLU subjects"},
    "xyz": {"mmlu_subjects": ["elementary_mathematics", "high_school_mathematics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Quantitative problem solving; difficulty range spans two MMLU levels"},
    "kvant": {"mmlu_subjects": ["high_school_mathematics", "elementary_mathematics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Quantitative reasoning; difficulty range spans two MMLU levels"},
    "kvantitativ": {"mmlu_subjects": ["high_school_mathematics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Quantitative reasoning"},
    "verbal": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Verbal reasoning; tests linguistic aptitude, not domain knowledge"},
    "kunskapsprov_apotekare": {"mmlu_subjects": ["clinical_knowledge", "nutrition"], "mmlu_category": "medical", "confidence": "approximate", "note": "Professional pharmacist exam; covers pharmacology beyond MMLU scope"},
    "kunskapsprov_audionom": {"mmlu_subjects": ["clinical_knowledge", "anatomy"], "mmlu_category": "medical", "confidence": "approximate", "note": "Professional audiologist exam; specialized beyond MMLU granularity"},
    "kunskapsprov_läkare": {"mmlu_subjects": ["professional_medicine", "clinical_knowledge", "college_medicine"], "mmlu_category": "medical", "confidence": "approximate", "note": "Professional physician exam; broad medical scope across multiple MMLU subjects"},
    "kunskapsprov_tandläkare": {"mmlu_subjects": ["professional_medicine", "clinical_knowledge", "anatomy"], "mmlu_category": "medical", "confidence": "approximate", "note": "Professional dentist exam; dental-specific content not in MMLU"},
    "matematik_och_fysikprovet": {"mmlu_subjects": ["high_school_mathematics", "high_school_physics", "college_physics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Combined math+physics; spans multiple MMLU subjects and levels"},
    # --- Exam-ET ---
    "ajalugu": {"mmlu_subjects": ["high_school_european_history", "high_school_world_history"], "mmlu_category": "humanities", "confidence": "approximate", "note": "History (ET); may include Estonian-specific history not in MMLU"},
    "bioloogia": {"mmlu_subjects": ["high_school_biology", "college_biology"], "mmlu_category": "stem", "confidence": "approximate", "note": "Biology (ET); difficulty level spans two MMLU subjects"},
    "eesti_keel": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Estonian language exam; no MMLU equivalent for L1 language exams"},
    "eesti": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Estonian language exam; no MMLU equivalent"},
    "füüsika": {"mmlu_subjects": ["high_school_physics", "college_physics", "conceptual_physics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Physics (ET); difficulty spans three MMLU subjects"},
    "geograafia": {"mmlu_subjects": ["high_school_geography"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Geography (ET); closest single MMLU match"},
    "keemia": {"mmlu_subjects": ["high_school_chemistry", "college_chemistry"], "mmlu_category": "stem", "confidence": "approximate", "note": "Chemistry (ET); difficulty level spans two MMLU subjects"},
    "kirjandus": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Literature (ET); MMLU has no literature subject"},
    "kunstiajalugu": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Art History (ET); MMLU has no art history subject"},
    "matemaatika": {"mmlu_subjects": ["high_school_mathematics", "college_mathematics", "elementary_mathematics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Mathematics (ET); difficulty range spans three MMLU levels"},
    "muusika": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Music (ET); MMLU has no music subject"},
    "filosoofia": {"mmlu_subjects": ["philosophy"], "mmlu_category": "humanities", "confidence": "direct", "note": "Philosophy (ET); direct MMLU equivalent exists"},
    "inglise_keel": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "English language (ET); L2 proficiency test, not in MMLU"},
    "ühiskonnaõpetus": {"mmlu_subjects": ["high_school_government_and_politics", "sociology"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Social studies (ET); spans multiple MMLU subjects"},
    "yhiskonnaopetus": {"mmlu_subjects": ["high_school_government_and_politics", "sociology"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Social studies (ET); spans multiple MMLU subjects"},
    "informaatika": {"mmlu_subjects": ["high_school_computer_science", "college_computer_science"], "mmlu_category": "stem", "confidence": "approximate", "note": "Computer science (ET); difficulty spans two MMLU levels"},
    "majandus": {"mmlu_subjects": ["high_school_macroeconomics", "high_school_microeconomics", "econometrics"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Economics (ET); spans macro, micro, and econometrics in MMLU"},
    "loodusõpetus": {"mmlu_subjects": ["high_school_biology", "conceptual_physics"], "mmlu_category": "stem", "confidence": "approximate", "note": "Natural sciences (ET); interdisciplinary, spans biology and physics"},
    "eesti_keel_teise_keelena_gümnaasium": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Estonian as L2, gymnasium; language proficiency test, not in MMLU"},
    "eesti_keel_teise_keelena_põhikool": {"mmlu_subjects": ["miscellaneous"], "mmlu_category": "humanities", "confidence": "no_equivalent", "note": "Estonian as L2, basic school; language proficiency test, not in MMLU"},
    # --- DanishCitizenTest ---
    "citizenship": {"mmlu_subjects": ["high_school_government_and_politics", "sociology"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Danish citizenship test; tests Denmark-specific civics and culture"},
    "indfødsretsprøve": {"mmlu_subjects": ["high_school_government_and_politics"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Danish naturalization test; country-specific content"},
    "medborgerskabsprøve": {"mmlu_subjects": ["high_school_government_and_politics"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Danish citizenship examination; country-specific content"},
    "residence": {"mmlu_subjects": ["high_school_government_and_politics", "sociology"], "mmlu_category": "social_sciences", "confidence": "approximate", "note": "Danish residence permit test; tests Denmark-specific civics and culture"},
    # --- Datasets without native subjects ---
    "main": {"mmlu_subjects": [], "mmlu_category": "mixed", "confidence": "no_equivalent", "note": "No native subject; use culture_ann.subject_category"},
}


# =============================================================================
# HELPERS
# =============================================================================

def load_jsonl(path: str):
    """Load JSONL (optionally xz-compressed). Returns list of dicts."""
    items = []
    open_fn = lzma.open if path.endswith('.xz') else open
    with open_fn(path, 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping line {line_no} in {Path(path).name}: {e}")
    return items


def get_culture_ann(item):
    """Return the culture annotation dict (GPT-5.2 or human), or None."""
    ann = item.get("culture_ann")
    if ann is not None and not (isinstance(ann, float) and math.isnan(ann)):
        return ann
    ref = item.get("culture_ann_ref")
    if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
        return ref
    return None


def get_subject_category(item):
    """Extract subject_category from culture_ann (GPT-5.2) if available."""
    ann = item.get("culture_ann")
    if ann and isinstance(ann, dict):
        return ann.get("subject_category")
    return None


def translate_subject(suite: str, subject: str) -> str:
    """Translate a native subject name to English."""
    subj_lower = subject.lower().strip()
    if suite == "Skolprov":
        return SKOLPROV_SUBJECT_TRANSLATION.get(subj_lower, f"[UNKNOWN: {subject}]")
    elif suite == "Exam-ET":
        return EXAMET_SUBJECT_TRANSLATION.get(subj_lower, f"[UNKNOWN: {subject}]")
    elif suite == "DanishCitizenTest":
        return DCT_SUBJECT_TRANSLATION.get(subj_lower, f"[UNKNOWN: {subject}]")
    else:
        return subject.replace("_", " ").title()


def get_mmlu_mapping(suite: str, subject: str):
    """Return MMLU mapping dict for a native subject, or None."""
    subj_lower = subject.lower().strip()

    # Direct MMLU subjects (Global-MMLU, Global-MMLU-Lite)
    if suite in ("Global-MMLU", "Global-MMLU-Lite"):
        if subj_lower in MMLU_SUBJECT_TO_CATEGORY:
            return {
                "mmlu_subjects": [subj_lower],
                "mmlu_category": MMLU_SUBJECT_TO_CATEGORY[subj_lower],
                "confidence": "direct",
                "note": "Direct MMLU subject",
            }
        return None

    return NATIVE_SUBJECT_TO_MMLU.get(subj_lower)


def truncate_text(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_dataset(ds_config: dict) -> dict:
    """Analyze a single dataset and return structured results."""
    name = ds_config["name"]
    suite = ds_config["suite"]
    language = ds_config["language"]
    path = ds_config["path"]

    print(f"\n{'='*60}")
    print(f"Processing: {name} ({language})")
    print(f"  Path: {path}")

    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        return None

    items = load_jsonl(path)
    print(f"  Loaded {len(items)} items")

    if not items:
        return None

    subset_counter = Counter()
    subject_category_counter = Counter()
    ca_count = 0
    cs_count = 0
    other_sensitivity = Counter()
    required_knowledge_counter = Counter()
    time_sensitive_counter = Counter()
    example_items = {}

    # Cross-validation: per (subset, ann_category) item counts
    subset_ann_cat_counts = defaultdict(Counter)

    for item in items:
        subset = item.get("subset", "unknown")
        subset_counter[subset] += 1

        if subset not in example_items:
            example_items[subset] = item

        subj_cat = get_subject_category(item)
        if subj_cat:
            subject_category_counter[subj_cat] += 1
            subset_ann_cat_counts[subset][subj_cat] += 1

        ann = get_culture_ann(item)
        if ann and isinstance(ann, dict):
            csl = ann.get("cultural_sensitivity_label", "")
            if csl == "CA":
                ca_count += 1
            elif csl == "CS":
                cs_count += 1
            elif csl:
                other_sensitivity[csl] += 1

            rk = ann.get("required_knowledge")
            if rk is not None:
                if isinstance(rk, list):
                    for v in rk:
                        if v:
                            required_knowledge_counter[v] += 1
                elif isinstance(rk, str):
                    required_knowledge_counter[rk] += 1

            ts = ann.get("time_sensitive")
            if ts is not None:
                if isinstance(ts, list):
                    for v in ts:
                        if v:
                            time_sensitive_counter[v] += 1
                elif isinstance(ts, str):
                    time_sensitive_counter[ts] += 1

            rk_agree = ann.get("required_knowledge_agreement")
            if rk_agree and not rk:
                required_knowledge_counter[rk_agree] += 1
            ts_agree = ann.get("time_sensitive_agreement")
            if ts_agree and not ts:
                time_sensitive_counter[ts_agree] += 1

    # Build subject mappings
    subject_mappings = []
    for subset, count in sorted(subset_counter.items()):
        en_translation = translate_subject(suite, subset)
        mapping = get_mmlu_mapping(suite, subset)

        subject_mappings.append({
            "native_subject": subset,
            "en_translation": en_translation,
            "item_count": count,
            "mmlu_subjects": mapping["mmlu_subjects"] if mapping else [],
            "mmlu_category": mapping["mmlu_category"] if mapping else "unmapped",
            "confidence": mapping.get("confidence", "unmapped") if mapping else "unmapped",
            "mapping_note": mapping["note"] if mapping else "No mapping found",
        })

    mmlu_conforming_subjects = set()
    mmlu_conforming_categories = set()
    for sm in subject_mappings:
        for ms in sm["mmlu_subjects"]:
            if ms in MMLU_SUBJECT_TO_CATEGORY:
                mmlu_conforming_subjects.add(ms)
        if sm["mmlu_category"] in MMLU_CATEGORIES:
            mmlu_conforming_categories.add(sm["mmlu_category"])

    ann_categories = set(subject_category_counter.keys())

    result = {
        "name": name,
        "suite": suite,
        "language": language,
        "total_items": len(items),
        "subsets": dict(subset_counter),
        "subject_mappings": subject_mappings,
        "mmlu_conforming_subjects": sorted(mmlu_conforming_subjects),
        "mmlu_conforming_categories": sorted(mmlu_conforming_categories),
        "annotated_subject_categories": dict(subject_category_counter),
        "ann_categories_set": sorted(ann_categories),
        "ca_count": ca_count,
        "cs_count": cs_count,
        "other_sensitivity": dict(other_sensitivity),
        "required_knowledge": dict(required_knowledge_counter),
        "time_sensitive": dict(time_sensitive_counter),
        "example_items": {k: truncate_text(v.get("text", ""), 200) for k, v in example_items.items()},
        "subset_ann_cat_counts": {k: dict(v) for k, v in subset_ann_cat_counts.items()},
    }

    print(f"  Subsets: {len(subset_counter)}")
    for s, c in sorted(subset_counter.items(), key=lambda x: -x[1])[:10]:
        print(f"    {s}: {c}")
    if len(subset_counter) > 10:
        print(f"    ... and {len(subset_counter) - 10} more")
    print(f"  MMLU-conforming subjects: {len(mmlu_conforming_subjects)}")
    print(f"  MMLU-conforming categories: {len(mmlu_conforming_categories)}")
    print(f"  Annotated subject_categories: {dict(subject_category_counter)}")
    print(f"  CA: {ca_count}, CS: {cs_count}")

    return result


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_table1(results: list, output_dir: str):
    """Table 1: Dataset overview."""
    path = os.path.join(output_dir, "table1_dataset_overview.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Language", "Total Items",
            "Num Native Subjects/Subsets",
            "Num MMLU-Conforming Subjects",
            "Num MMLU-Conforming Categories",
            "Num Annotated subject_categories",
            "Native Subjects (original)",
            "Native Subjects (EN translation)",
            "MMLU Categories Covered",
            "Annotated subject_categories",
        ])
        for r in results:
            if r is None:
                continue
            native_subjects_orig = "; ".join(sorted(r["subsets"].keys()))
            native_subjects_en = "; ".join(
                sm["en_translation"] for sm in sorted(r["subject_mappings"], key=lambda x: x["native_subject"])
            )
            writer.writerow([
                r["name"], r["language"], r["total_items"],
                len(r["subsets"]),
                len(r["mmlu_conforming_subjects"]),
                len(r["mmlu_conforming_categories"]),
                len(r["ann_categories_set"]),
                native_subjects_orig, native_subjects_en,
                "; ".join(r["mmlu_conforming_categories"]),
                "; ".join(r["ann_categories_set"]),
            ])
    print(f"\n[OK] Table 1 written to: {path}")


def write_table2(results: list, output_dir: str):
    """Table 2: Cultural annotation statistics."""
    path = os.path.join(output_dir, "table2_cultural_annotations.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Language", "Total Items",
            "CA Count", "CS Count", "Other Sensitivity Labels",
            "Num Distinct required_knowledge Labels",
            "required_knowledge Distribution",
            "time_sensitive Yes", "time_sensitive No",
        ])
        for r in results:
            if r is None:
                continue
            rk = r["required_knowledge"]
            ts = r["time_sensitive"]
            other_sens = "; ".join(f"{k}={v}" for k, v in r["other_sensitivity"].items()) or "—"
            rk_dist = "; ".join(f"{k}={v}" for k, v in sorted(rk.items(), key=lambda x: -x[1]))
            writer.writerow([
                r["name"], r["language"], r["total_items"],
                r["ca_count"], r["cs_count"], other_sens,
                len(rk), rk_dist if rk_dist else "—",
                ts.get("Yes", 0), ts.get("No", 0),
            ])
    print(f"[OK] Table 2 written to: {path}")


def write_subject_mapping(results: list, output_dir: str):
    """Subject mapping table with confidence levels."""
    path = os.path.join(output_dir, "subject_mapping.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Language",
            "Native Subject (Original)", "Native Subject (EN)",
            "Item Count",
            "Mapped MMLU Subjects", "Mapped MMLU Category",
            "Mapping Confidence",
            "Mapping Note",
        ])
        for r in results:
            if r is None:
                continue
            for sm in sorted(r["subject_mappings"], key=lambda x: -x["item_count"]):
                writer.writerow([
                    r["name"], r["language"],
                    sm["native_subject"], sm["en_translation"],
                    sm["item_count"],
                    "; ".join(sm["mmlu_subjects"]) if sm["mmlu_subjects"] else "—",
                    sm["mmlu_category"],
                    sm["confidence"],
                    sm["mapping_note"],
                ])
    print(f"[OK] Subject mapping written to: {path}")


def write_cross_validation(results: list, output_dir: str):
    """Cross-validate our manual category mapping against GPT-5.2 subject_category.

    For each native subject, compare:
      - Our assigned mmlu_category (from the mapping dict)
      - The distribution of GPT-5.2's subject_category annotations for items in that subject

    This reveals whether our heuristic mapping agrees with the independent annotation.
    """
    path = os.path.join(output_dir, "cross_validation.csv")
    summary_path = os.path.join(output_dir, "cross_validation_summary.txt")

    total_items_checked = 0
    total_items_agree = 0
    total_items_disagree = 0
    total_items_no_ann = 0
    disagreement_details = []

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Language",
            "Native Subject",
            "Our Mapped MMLU Category",
            "Mapping Confidence",
            "Total Items",
            "Items with subject_category Annotation",
            "GPT-5.2 subject_category Distribution",
            "Majority GPT-5.2 Category",
            "Majority GPT-5.2 Category (normalised to MMLU)",
            "Agreement (our_cat == majority_ann_cat)",
            "Agreement Rate (%)",
        ])

        for r in results:
            if r is None:
                continue

            ann_counts = r.get("subset_ann_cat_counts", {})

            for sm in sorted(r["subject_mappings"], key=lambda x: -x["item_count"]):
                subset = sm["native_subject"]
                our_cat = sm["mmlu_category"]
                confidence = sm["confidence"]
                item_count = sm["item_count"]
                cat_dist = ann_counts.get(subset, {})
                annotated_count = sum(cat_dist.values())

                if annotated_count == 0:
                    total_items_no_ann += item_count
                    writer.writerow([
                        r["name"], r["language"], subset, our_cat, confidence,
                        item_count, 0, "—", "—", "—", "N/A (no annotations)", "—",
                    ])
                    continue

                # Find majority GPT-5.2 category
                majority_ann_cat = max(cat_dist, key=cat_dist.get)
                majority_ann_cat_normalised = ANN_CAT_TO_MMLU_CAT.get(majority_ann_cat, majority_ann_cat.lower())

                # For "mixed" (datasets without native subjects), skip agreement
                if our_cat == "mixed":
                    agrees = "N/A (mixed)"
                    agree_rate = "—"
                else:
                    # Items whose GPT-5.2 annotation matches our mapped category
                    matching_ann_cats = set()
                    for ann_cat, mmlu_cat in ANN_CAT_TO_MMLU_CAT.items():
                        if mmlu_cat == our_cat:
                            matching_ann_cats.add(ann_cat)

                    items_agreeing = sum(cat_dist.get(ac, 0) for ac in matching_ann_cats)
                    agree_rate_val = (items_agreeing / annotated_count * 100) if annotated_count > 0 else 0

                    agrees = "YES" if majority_ann_cat_normalised == our_cat else "NO"
                    agree_rate = f"{agree_rate_val:.1f}"

                    total_items_checked += annotated_count
                    total_items_agree += items_agreeing
                    total_items_disagree += (annotated_count - items_agreeing)

                    if agrees == "NO":
                        disagreement_details.append({
                            "dataset": r["name"],
                            "subject": subset,
                            "our_cat": our_cat,
                            "gpt_majority": majority_ann_cat_normalised,
                            "gpt_dist": cat_dist,
                            "item_count": annotated_count,
                        })

                dist_str = "; ".join(f"{k}={v}" for k, v in sorted(cat_dist.items(), key=lambda x: -x[1]))

                writer.writerow([
                    r["name"], r["language"], subset, our_cat, confidence,
                    item_count, annotated_count, dist_str,
                    majority_ann_cat, majority_ann_cat_normalised,
                    agrees, agree_rate,
                ])

    print(f"[OK] Cross-validation written to: {path}")

    # Write human-readable summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Cross-Validation Summary: Our Mapping vs GPT-5.2 subject_category\n")
        f.write("=" * 70 + "\n\n")

        if total_items_checked > 0:
            overall_rate = total_items_agree / total_items_checked * 100
            f.write(f"Total items cross-validated:   {total_items_checked}\n")
            f.write(f"Items where categories agree:  {total_items_agree} ({overall_rate:.1f}%)\n")
            f.write(f"Items where categories differ: {total_items_disagree} ({100-overall_rate:.1f}%)\n")
            f.write(f"Items without annotation:      {total_items_no_ann}\n\n")
        else:
            f.write("No items with annotations could be cross-validated.\n\n")

        if disagreement_details:
            f.write(f"Disagreements ({len(disagreement_details)} subjects where our mapping != GPT-5.2 majority):\n")
            f.write("-" * 70 + "\n")
            for d in disagreement_details:
                f.write(f"\n  Dataset: {d['dataset']}\n")
                f.write(f"  Subject: {d['subject']}\n")
                f.write(f"  Our category:     {d['our_cat']}\n")
                f.write(f"  GPT-5.2 majority: {d['gpt_majority']}\n")
                f.write(f"  GPT-5.2 full distribution: {d['gpt_dist']}\n")
                f.write(f"  Annotated items:  {d['item_count']}\n")
        else:
            f.write("No disagreements found — all mappings agree with GPT-5.2!\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("Interpretation:\n")
        f.write("  - High agreement validates the manual category mapping.\n")
        f.write("  - Disagreements highlight subjects where the mapping is debatable\n")
        f.write("    and should be discussed in the paper.\n")
        f.write("  - 'mixed' category (datasets without native subjects) cannot be\n")
        f.write("    cross-validated at subject level; use item-level annotations.\n")

    print(f"[OK] Cross-validation summary written to: {summary_path}")


def write_confidence_summary(results: list, output_dir: str):
    """Summary of mapping confidence levels across all datasets."""
    path = os.path.join(output_dir, "confidence_summary.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Language",
            "Items: direct", "Items: approximate", "Items: no_equivalent", "Items: unmapped",
            "Subjects: direct", "Subjects: approximate", "Subjects: no_equivalent", "Subjects: unmapped",
        ])
        for r in results:
            if r is None:
                continue
            item_counts = Counter()
            subj_counts = Counter()
            for sm in r["subject_mappings"]:
                conf = sm["confidence"]
                item_counts[conf] += sm["item_count"]
                subj_counts[conf] += 1
            writer.writerow([
                r["name"], r["language"],
                item_counts.get("direct", 0),
                item_counts.get("approximate", 0),
                item_counts.get("no_equivalent", 0),
                item_counts.get("unmapped", 0),
                subj_counts.get("direct", 0),
                subj_counts.get("approximate", 0),
                subj_counts.get("no_equivalent", 0),
                subj_counts.get("unmapped", 0),
            ])
    print(f"[OK] Confidence summary written to: {path}")


def write_example_items(results: list, output_dir: str):
    """Example items per subject mapping for the appendix."""
    path = os.path.join(output_dir, "example_items_per_mapping.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Language", "Native Subject",
            "Mapped MMLU Category", "Mapping Confidence",
            "Example Item (truncated)",
        ])
        for r in results:
            if r is None:
                continue
            for sm in sorted(r["subject_mappings"], key=lambda x: x["native_subject"]):
                example = r["example_items"].get(sm["native_subject"], "—")
                writer.writerow([
                    r["name"], r["language"],
                    sm["native_subject"], sm["mmlu_category"], sm["confidence"],
                    example,
                ])
    print(f"[OK] Example items written to: {path}")


def write_full_subject_stats(results: list, output_dir: str):
    """Detailed per-subject item counts with annotation breakdown."""
    path = os.path.join(output_dir, "full_subject_stats.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Language", "Subject/Subset", "Item Count",
            "GPT-5.2 subject_category Distribution",
        ])
        for r in results:
            if r is None:
                continue
            ann_counts = r.get("subset_ann_cat_counts", {})
            for subset, count in sorted(r["subsets"].items(), key=lambda x: -x[1]):
                cat_dist = ann_counts.get(subset, {})
                dist_str = "; ".join(f"{k}={v}" for k, v in sorted(cat_dist.items(), key=lambda x: -x[1]))
                writer.writerow([
                    r["name"], r["language"], subset, count,
                    dist_str if dist_str else "—",
                ])
    print(f"[OK] Full subject stats written to: {path}")


def write_unmapped_subjects_report(results: list, output_dir: str):
    """Report subjects that could not be mapped."""
    path = os.path.join(output_dir, "unmapped_subjects.txt")
    unmapped = []
    for r in results:
        if r is None:
            continue
        for sm in r["subject_mappings"]:
            if sm["mmlu_category"] == "unmapped" or "[UNKNOWN" in sm["en_translation"]:
                unmapped.append((r["name"], r["language"], sm["native_subject"], sm["en_translation"], sm["item_count"]))

    with open(path, 'w', encoding='utf-8') as f:
        f.write("Unmapped or Unknown Subjects\n")
        f.write("=" * 60 + "\n\n")
        if not unmapped:
            f.write("All subjects were successfully mapped!\n")
        else:
            f.write(f"{'Dataset':<25} {'Lang':<5} {'Native Subject':<25} {'EN Translation':<35} {'Count'}\n")
            f.write("-" * 100 + "\n")
            for ds, lang, native, en, count in unmapped:
                f.write(f"{ds:<25} {lang:<5} {native:<25} {en:<35} {count}\n")
            f.write(f"\nTotal unmapped: {len(unmapped)} subjects\n")
            f.write("\nAction needed: Add these subjects to the translation/mapping dictionaries\n")
            f.write("in analyze_datasets.py and re-run.\n")
    print(f"[OK] Unmapped subjects report written to: {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze MC-Exam-Style Datasets for MMLU Mapping")
    parser.add_argument("--data-dir", "-d", required=True,
                        help="Base directory containing annotated JSONL files")
    parser.add_argument("--output-dir", "-o", default="./output_stats",
                        help="Directory for output files (default: ./output_stats)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    DATASETS = resolve_datasets(Path(args.data_dir))

    print("MC-Exam-Style Dataset Analyzer")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Datasets to process: {len(DATASETS)}")

    results = []
    for ds in DATASETS:
        result = analyze_dataset(ds)
        results.append(result)

    print("\n" + "=" * 60)
    print("Writing output files...")
    write_table1(results, output_dir)
    write_table2(results, output_dir)
    write_subject_mapping(results, output_dir)
    write_cross_validation(results, output_dir)
    write_confidence_summary(results, output_dir)
    write_example_items(results, output_dir)
    write_full_subject_stats(results, output_dir)
    write_unmapped_subjects_report(results, output_dir)

    valid = [r for r in results if r is not None]
    total_items = sum(r["total_items"] for r in valid)
    print(f"\nDone! Processed {len(valid)}/{len(DATASETS)} datasets, {total_items} total items.")
    print(f"Output files in: {output_dir}/")


if __name__ == "__main__":
    main()


