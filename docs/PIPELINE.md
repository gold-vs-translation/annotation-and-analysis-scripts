# Pipeline Documentation

This document describes the complete data processing pipeline in detail,
including data flow, intermediate artifacts, and methodological notes.

## Pipeline Overview

```
                    ┌─────────────────────┐
                    │  Raw MC-Exam Items  │
                    │  (JSONL per dataset) │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌──────────────┐ ┌────────────┐ ┌─────────────────┐
    │  Stage A:    │ │ Stage B:   │ │ Stage C:        │
    │  GPT-5.2     │ │ Prompt     │ │ Human           │
    │  Annotation  │ │ Tuning     │ │ Annotation      │
    │  (all items) │ │ (EN only)  │ │ (100/lang)      │
    └──────┬───────┘ └─────┬──────┘ └───────┬─────────┘
           │               │                │
           │               ▼                ▼
           │     ┌──────────────┐  ┌──────────────────┐
           │     │  Calibrated  │  │  Stage D: QA     │
           │     │  Prompt      │  │  + Paper Tables  │
           │     └──────────────┘  └────────┬─────────┘
           │                                │
           ▼                                ▼
    ┌──────────────────────────────────────────────┐
    │  Stage E+F: Data Harmonization & Merge       │
    │  (slim format + join with eval results)      │
    └──────────────────────┬───────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌──────────────┐ ┌──────────┐ ┌────────────────┐
    │  Stage G:    │ │ Dataset  │ │ Incremental    │
    │  CA/CS       │ │ Analysis │ │ Proxy Validity │
    │  Ranking     │ │ (Tables) │ │ (R² analysis)  │
    └──────────────┘ └──────────┘ └────────────────┘
```

## Stage A: Automatic GPT Annotation

**Script**: `src/cs_annotation/automatic/annotation_gpt.py`

**Purpose**: Annotate each MC-exam item with cultural sensitivity labels using
GPT-5.2 structured outputs (JSON schema enforcement).

**Key features**:
- SQLite-based deduplication cache (content-addressed via SHA-256 fingerprint)
- Sync mode for debugging, batch mode for production
- Automatic retry with higher `max_output_tokens` on truncation
- Prompt caching via OpenAI's `prompt_cache_key` parameter

**Input**: JSONL with fields `id`, `suite`, `task`, `subset`, `language`, `text`

**Output**: Same JSONL with added `culture_ann` object containing all annotation fields.

## Stage B: Prompt Calibration

Three scripts form a calibration loop using Global-MMLU-Lite (English, ~3000 items
with 3+ human annotators per item):

### Step 1: `step1_build_agreement.py`
- Computes majority vote across human annotators
- Derives CS/CA deterministically from `required_knowledge`
- Exports tie cases separately for manual review
- Outputs agreement statistics (XLSX)

### Step 2: `step2_dev_test_split.py`
- Stratified split ensuring minimum items per MMLU subject
- Fixed seed for reproducibility

### Step 3: `step3_eval_split.py`
- Compares GPT annotations against human reference
- Produces confusion matrices for `required_knowledge`, `csl_rule`, `csl_provided`
- Identifies systematic error patterns for prompt refinement

## Stage C: Human Annotation

### Sampling (`sampling.py`)
- Reads GPT-annotated files and samples 100 items per language
- Stratifies by: `cultural_sensitivity_label`, `required_knowledge`, `subject_category`, `time_sensitive`
- Ensures minimum coverage per dataset (≥40 items)

### Argilla Upload (`annotation_human.py`)
- Creates Argilla datasets with the full annotation schema
- Includes detailed guidelines (embedded in source)
- Supports span annotation for trigger highlights

## Stage D: Quality Assurance

### Validation (`validate_annotations.py`)
- Uses GPT-5.2 to review each human annotation
- Checks: missing fields, contradictions, format issues, span alignment, content sanity
- Outputs severity levels: OK / WARN / ERROR
- Suggests corrections where applicable

### Paper Tables (`generate_tables.py`)
- **Table 1**: Label distributions per language (post-review)
- **Table 2**: QA review statistics per annotator (pre-review)
- **Table 3**: Pre→Post label shift (effect of review round)
- **Table 4**: Fleiss' κ for DA (3 annotators, pre-review)
- **Table 5**: Pairwise Cohen's κ for DA

**Methodological note**: IAA is computed on pre-review data to measure genuine
independent agreement. Computing IAA on post-review data would conflate
annotator agreement with the review process.

## Stage E+F: Data Harmonization

### Slim Format Converters
Three scripts normalize heterogeneous benchmark formats into a unified schema:
- `to_slim_globalmmlu.py`: Global-MMLU (with multi-annotator `culture_ann_ref`)
- `to_slim_native.py`: GPT-annotated native benchmarks
- `to_slim_belebele.py`: Belebele (unannotated, null annotation fields)

### Merge Pipeline
1. `merge_annotations_slim.py`: Combine per-user Argilla exports into one JSONL
2. `merge_with_results.py`: Join annotations with LM Eval Harness sample logs
3. `diagnose_unmatched.py`: Analyze join failures (language/subset/suite mismatches)

## Stage G: Ranking Analysis

### Balanced Subsets (`build_balanced_subsets.py`)
- Creates CA and CS subsets with matched item counts per model
- Ensures fair comparison by controlling for subset size

### Baseline Ranking (`baseline_rank.py`)
- Computes model accuracy on ALL / CA / CS subsets
- Wilson confidence intervals
- Kendall τ_b rank correlation between CA and CS rankings

### Proxy Rank Consistency (`proxy_rank_consistency.py`)
- Tests whether model rankings on translated benchmarks match native benchmarks
- Per-language and cross-language analysis

### Incremental Proxy Validity (`incremental_proxy_validity.py`)
- Hierarchical regression: does proxy score predict native score beyond model size?
- Controls: log(parameters), Belebele score, optionally Global-MMLU-X EN score
- Partial correlations and leave-one-family-out robustness checks

## Data Formats

### Input JSONL (raw items)
```json
{
  "id": "citizenship/2020/summer/27",
  "suite": "DanishCitizenTest",
  "task": "DanishCitizenTest",
  "subset": "citizenship",
  "split": "test",
  "language": "DA",
  "text": "Hvad hedder Danmarks hovedstad?\n(A) Berlin\n(B) København\n..."
}
```

### Annotated JSONL (after Stage A)
```json
{
  "id": "...",
  "suite": "...",
  "culture_ann": {
    "required_knowledge": "regional",
    "cultural_sensitivity_label": "CS",
    "time_sensitive": "No",
    "subject_category": "Social Sciences",
    "culture": null,
    "region": "Europe",
    "country": "Denmark (DK)",
    "references": [{"start": 0, "end": 35, "span": "Danmarks hovedstad", "label": "Geographic"}],
    "other_references": [],
    "explanation": "Requires geographic knowledge of Denmark's capital."
  }
}
```

### Slim Format (after Stage E)
```json
{
  "user": "annotator_1",
  "user_id": "uuid-ann-001",
  "suite": "DanishCitizenTest",
  "task": "DanishCitizenTest",
  "subset": "citizenship",
  "language": "DA",
  "sample_id": "citizenship/2020/summer/27",
  "required_knowledge": "regional",
  "cultural_sensitivity_label": "CS",
  "time_sensitive": "No",
  "subject_category": "Social Sciences"
}
```

### Merged with Results (after Stage F)
```json
{
  "suite": "DanishCitizenTest",
  "task": "DanishCitizenTest",
  "language": "DA",
  "subset": "citizenship",
  "sample_id": "citizenship/2020/summer/27",
  "model_name": "llama-3-8b",
  "sample_score": 1,
  "required_knowledge": "regional",
  "cultural_sensitivity_label": "CS"
}
```
