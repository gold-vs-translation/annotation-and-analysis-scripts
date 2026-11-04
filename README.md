# Cultural Sensitivity Annotation Pipeline

Code and data processing pipeline for annotating cultural sensitivity in exam-style multiple-choice benchmarks across five Nordic / Baltic languages (DA, ET, LT, NB, SV) plus English reference data (Global-MMLU / Global-MMLU-Lite).


## Overview

This repository implements a complete annotation and analysis pipeline:

1. **Automatic annotation** of MC-exam items using GPT-5.2 structured outputs
2. **Prompt calibration** via a three-step human-agreement → dev/test → evaluation loop on Global-MMLU-Lite
3. **Human annotation** through an Argilla-based UI with stratified sampling
4. **Quality assurance** using LLM-based validation of human annotations
5. **Dataset harmonization** converting heterogeneous benchmark formats into a unified slim schema
6. **Ranking analysis** comparing model rankings on culturally sensitive (CS) vs. culturally agnostic (CA) subsets
7. **Incremental proxy validity** testing whether translated benchmarks predict native-benchmark performance

## Repository Structure

```
cs-annotation/
├── pyproject.toml                          # Package metadata and dependencies
├── README.md                               # This file
├── docs/
│   └── PIPELINE.md                         # Detailed pipeline documentation
├── scripts/
│   ├── generate_dummy_data.py              # Generate synthetic test data
│   ├── run_pipeline_demo.sh                # End-to-end demo with dummy data
│   └── split_by_users.sh                   # Split Argilla export by annotator
├── data/
│   └── dummy/                              # Synthetic data for testing
│       ├── annotated/                      # GPT-annotated JSONL files
│       ├── globalmmlulite/                 # Global-MMLU-Lite splits
│       ├── human_annotations/              # Human annotation exports
│       ├── pre_review_csvs/                # QA review CSVs
│       └── merged/                         # Merged annotations + eval logs
└── src/cs_annotation/
    ├── automatic/                          # Automatic annotation pipeline
    │   ├── annotation_gpt.py               # GPT-5.2 annotation (sync + batch)
    │   ├── step1_build_agreement.py        # Human agreement + CS/CA derivation
    │   ├── step2_dev_test_split.py         # Stratified dev/test split
    │   ├── step3_eval_split.py             # Evaluation + confusion matrices
    │   ├── eval_agreement.py               # Human vs. GPT agreement analysis
    │   ├── globalmmlu_stats.py             # Global-MMLU coverage statistics
    │   └── human_metaeval_merge.py         # Meta-evaluation merge (500 items)
    ├── manual/                             # Human annotation tools
    │   ├── annotation_human.py             # Argilla dataset creation + upload
    │   ├── sampling.py                     # Stratified sampling for annotation
    │   ├── generate_tables.py              # Paper tables (IAA, distributions)
    │   ├── filter_by_user.py               # Filter Argilla exports per user
    │   └── dedup.py                        # Deduplicate JSONL items
    ├── validation/                         # Quality assurance
    │   ├── validate_annotations.py         # LLM-based annotation validation
    │   └── summarize_reviews.py            # Aggregate QA reviews
    ├── datasets/                           # Data harmonization
    │   ├── analyze_datasets.py             # MC-exam → MMLU mapping + stats
    │   ├── to_slim_globalmmlu.py           # Global-MMLU → slim format
    │   ├── to_slim_native.py               # Native benchmarks → slim format
    │   ├── to_slim_belebele.py             # Belebele → slim format
    │   ├── merge_annotations_slim.py       # Merge per-user annotations
    │   ├── merge_with_results.py           # Join annotations + eval logs
    │   └── diagnose_unmatched.py           # Diagnose join failures
    └── analysis/                           # Ranking & validity analysis
        ├── build_balanced_subsets.py        # Balanced CA/CS subsets
        ├── baseline_rank.py                # Baseline ranking (CA vs CS)
        ├── proxy_rank_consistency.py        # Proxy ranking consistency
        ├── compare_variants.py             # Cross-suite ranking comparison
        └── incremental_proxy_validity.py   # Incremental R² regression
```

For an overview of the ranking analyses, incremental proxy validity, robustness checks, and a reproducible smoke test, see the [Analysis Guide](docs/ANALYSIS_GUIDE.md).

The released annotations, merged model scores, and reference outputs are available in the [companion data repository](https://github.com/gold-vs-translation/cs-annotation-data).


## Setup

### Requirements

- Python 3.10+
- An OpenAI API key (for automatic annotation and validation)
- Argilla server (for human annotation; optional for running analysis scripts)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# For human annotation tools:
pip install -e ".[human]"

# For development:
pip install -e ".[all]"
```

### Running with Dummy Data

Generate synthetic test data and run the pipeline demo:

```bash
# Generate dummy data
python scripts/generate_dummy_data.py

# Run the demo pipeline (no API key needed for analysis scripts)
bash scripts/run_pipeline_demo.sh
```

## Pipeline Stages

### Stage A: Automatic GPT Annotation

Annotates MC-exam items with cultural sensitivity labels using GPT-5.2 structured outputs. Supports both synchronous (debug) and batch (production) modes.

```bash
export OPENAI_API_KEY="sk-..."

python -m cs_annotation.automatic.annotation_gpt \
  --in data/dummy/annotated/input.jsonl \
  --out data/dummy/annotated/output.jsonl \
  --lang EN \
  --mode sync \
  --model gpt-5.2 \
  --reasoning-effort high \
  --cache-path cache/annotation.sqlite \
  --log INFO
```

### Stage B: Prompt Calibration (3-Step Pipeline)

```bash
# Step 1: Build human agreement from multi-annotator data
python -m cs_annotation.automatic.step1_build_agreement \
  --in data/dummy/globalmmlulite/text_globalmmlulite_EN_test.jsonl \
  --out-agreed data/dummy/globalmmlulite/agreed.jsonl \
  --out-ties data/dummy/globalmmlulite/ties.jsonl \
  --out-xlsx output/step1_stats.xlsx

# Step 2: Stratified dev/test split
python -m cs_annotation.automatic.step2_dev_test_split \
  --in data/dummy/globalmmlulite/agreed.jsonl \
  --out-dev data/dummy/globalmmlulite/dev.jsonl \
  --out-test data/dummy/globalmmlulite/test.jsonl \
  --seed 13 --dev-frac 0.30

# Step 3: Evaluate GPT annotations against human reference
python -m cs_annotation.automatic.step3_eval_split \
  --in data/dummy/globalmmlulite/text_globalmmlulite_EN_test_devsplit.jsonl \
  --out output/eval.xlsx
```

### Stage C: Human Annotation

```bash
export ARGILLA_API_URL="https://your-instance.hf.space"
export ARGILLA_API_KEY="your-key"

# Stratified sampling
python src/cs_annotation/manual/sampling.py \
  --annotated-dir data/dummy/annotated \
  --out-dir data/dummy/sampled \
  --n-per-lang 100 --min-per-dataset 40 --seed 13

# Upload to Argilla
python src/cs_annotation/manual/annotation_human.py \
  --base-dir data/dummy/sampled \
  --langs "DA,ET,LT,NB,SV"
```

### Stage D: Quality Assurance

```bash
# Validate annotations
python src/cs_annotation/validation/validate_annotations.py \
  --in data/dummy/human_annotations/ \
  --out output/reviews/ \
  --mode sync --model gpt-5.2

# Generate paper tables (IAA, label distributions)
python src/cs_annotation/manual/generate_tables.py \
  --post-review data/dummy/human_annotations/all_user_annotations.jsonl \
  --pre-review-dir data/dummy/pre_review_csvs/ \
  --latex-out output/tables/
```

### Stage E+F: Data Harmonization & Merge

```bash
# Convert to slim format
python src/cs_annotation/datasets/to_slim_native.py \
  --input-dir data/dummy/annotated/ \
  --pattern "*.jsonl" \
  --output data/dummy/merged/native_slim.jsonl

# Merge with evaluation results
python src/cs_annotation/datasets/merge_with_results.py \
  --annotations data/dummy/merged/all_annotations_slim.jsonl \
  --results data/dummy/merged/results.jsonl.xz \
  --output data/dummy/merged/merged.jsonl
```

### Stage G: Ranking Analysis

```bash
# Build balanced CA/CS subsets
python src/cs_annotation/analysis/build_balanced_subsets.py \
  --input data/dummy/merged/all_merged_samplelogs.jsonl \
  --out_dir output/ranking/

# Baseline ranking
python src/cs_annotation/analysis/baseline_rank.py \
  --input output/ranking/balanced_subsets_model_rows.jsonl \
  --out_dir output/ranking/

# Incremental proxy validity
python src/cs_annotation/analysis/incremental_proxy_validity.py \
  --input data/dummy/merged/all_merged_samplelogs.jsonl \
  --out_dir output/incremental/
```

## Annotation Schema

Each item is annotated with the following fields:

| Field | Values | Description |
|-------|--------|-------------|
| `required_knowledge` | `none`, `cultural`, `regional`, `dialect` | Type of culturally specific knowledge needed |
| `cultural_sensitivity_label` | `CS`, `CA` | Culturally sensitive vs. agnostic |
| `time_sensitive` | `Yes`, `No` | Whether the answer changes over time |
| `subject_category` | `STEM`, `Humanities`, `Social Sciences`, `Medical`, `Business`, `Other` | Academic domain |
| `culture` | e.g., `Western Culture` | Primary culture (if `cultural`) |
| `region` | e.g., `Europe` | Primary region (if `regional`) |
| `country` | e.g., `Denmark (DK)` | Primary country (if `regional`) |
| `references` | List of `{start, end, span, label}` | Trigger spans in the item text |
| `explanation` | String (≤250 chars) | Brief justification |

## Key Design Decisions

- **CS/CA derivation**: `CA` iff `required_knowledge == "none"`; `CS` otherwise. This deterministic rule avoids conflating annotator judgment on two correlated dimensions.
- **IAA on pre-review data**: Inter-annotator agreement is computed on pre-review annotations to measure genuine independent agreement, not the effect of the review process.
- **Stratified sampling**: Human annotation items are sampled to ensure coverage across datasets, cultural sensitivity levels, and subject categories.
- **Proxy validity**: Incremental R² regression tests whether translated (proxy) benchmark scores predict native benchmark performance beyond what model size alone explains.

## Dependencies

Core: `openai`, `pydantic`, `pandas`, `openpyxl`, `numpy`, `scikit-learn`

Human annotation: `argilla`, `pycountry`

See `pyproject.toml` for exact version constraints.

## License

MIT
