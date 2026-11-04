# Analysis guide — Gold vs. Translation

Companion to [Gold vs. Translation (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.32/). This guide focuses on **using the released data and checking the central analysis path**, rather than repeating costly GPT annotation or 29-model evaluation runs.

**Inputs:** [cs-annotation-data](https://github.com/gold-vs-translation/cs-annotation-data) · **Evaluation tasks:** [LM Evaluation Harness fork](https://github.com/gold-vs-translation/lm-evaluation-harness) · **Pipeline background:** [PIPELINE.md](PIPELINE.md).

## Which analysis answers which question?

| Analysis | Research question | Relevant code | Data/output |
|---|---|---|---|
| **CS/CA ranking consistency** | Do the same models rank highly on culturally sensitive (CS) versus culturally agnostic (CA) native items? | `src/cs_annotation/analysis/build_balanced_subsets.py`, `baseline_rank.py` | Merged scores and annotation labels |
| **Proxy ranking consistency** | Do model rankings on translated proxies align with rankings on native benchmarks? | `src/cs_annotation/analysis/proxy_rank_consistency.py`, `compare_variants.py` | Merged model scores |
| **Incremental proxy validity (central test)** | Does translated-benchmark performance explain *additional* variation in native performance after controls? | `scripts/camera_ready/incremental_proxy_validity_bootstrap.py` (final), earlier version in `src/cs_annotation/analysis/incremental_proxy_validity.py` | Merged scores; `results/proxy_validity/` reference outputs |
| **Human IAA / meta-evaluation** | How consistently were items annotated, and how well do GPT labels agree with human majority labels? | `scripts/camera_ready/compute_raw_iaa_plus_consensus_globalmmlu_style.py`, `recompute_native_meta_eval_camera_ready.py` | `results/iaa/`, `results/meta_eval/` |

## 1. CS/CA ranking analysis

**Why.** A benchmark's aggregate accuracy may conceal systematic differences between culturally sensitive (CS) and culturally agnostic (CA) items. Comparing model rankings across these subsets tests whether the distinction changes the relative ordering of models.

**How.** Items are assigned CS/CA labels from the annotation data. `build_balanced_subsets.py` can construct subsets with comparable item counts; `baseline_rank.py` aggregates model scores and compares rankings (including Kendall's tie-aware τ_b and Wilson intervals). The proxy-ranking scripts compare translated-proxy rankings against native rankings. Consult the paper for the precise grouping, stratification, uncertainty procedure and reported tables.

**Entry points.** `src/cs_annotation/analysis/` contains these scripts. They are included for method transparency, but this release does not claim every historical ranking variant was independently rerun using the redacted public JSONL. Check each script's CLI and required columns before use.

## 2. Incremental proxy validity (central analysis)

**Why.** A translated MMLU-like benchmark might correlate with native scores simply because stronger/larger models do well on both. Incremental validity asks whether the translated proxy still helps predict performance on a native benchmark **after controlling for general factors**.

**How.** For each native benchmark/language pairing, item-level scores are aggregated to model-level performance. The baseline regression predicts native performance from `log(model parameters)` and the corresponding **Belebele** performance. The extended regression adds the translated MMLU proxy score. The improvement is measured by **ΔR² = R²_extended − R²_baseline**, with a nested-model F-test and the proxy coefficient (`beta_proxy`). The optional English G-MMLUX control tests sensitivity to more general MMLU knowledge. Model-level bootstrap (5,000 resamples with seed 42 in the final analysis) and leave-one-model-family-out (LOFO) checks assess stability. These uncertainty checks are important: a positive in-sample ΔR² is not by itself evidence of a robust effect.

**Final implementation.** `scripts/camera_ready/incremental_proxy_validity_bootstrap.py` accepts `--input`, `--out_dir`, `--metric`, `--proxy-task`, `--belebele-task`, `--exclude-tasks` and optional `--bootstrap`, `--n-boot`, `--bootstrap-seed`, `--bootstrap-ci`. Its input is **plain JSONL**, not directly `.jsonl.gz`.

### Reproduce the successful small smoke check

Run from a checkout of `annotation-and-analysis-scripts` with the companion data repository at `../cs-annotation-data`. First install dependencies using the repository's `pyproject.toml`. The commands below create only temporary files, outside the tracked repositories.

```bash
mkdir -p /tmp/gvt-proxy-smoke
python - <<'PY'
import gzip, json
from collections import defaultdict
from pathlib import Path
src = Path('../cs-annotation-data/data/merged_samplelogs_nativ_globalmmlu_belebele_ann_slim.jsonl.gz')
dst = Path('/tmp/gvt-proxy-smoke/input.jsonl')
keep = {('DanishCitizenTest','DA'), ('GMMLUX','DA'), ('GMMLUX','EN'), ('Belebele','DA')}
seen = defaultdict(set)
with gzip.open(src, 'rt', encoding='utf-8') as inp, dst.open('w', encoding='utf-8') as out:
    for line in inp:
        row = json.loads(line)
        group = (row.get('suite'), row.get('language'))
        if group not in keep or row.get('metric') != 'acc':
            continue
        item = (row.get('subset'), str(row.get('sample_id')))
        if item not in seen[group]:
            if len(seen[group]) >= 50:
                continue
            seen[group].add(item)
        out.write(line)
print({str(k): len(v) for k, v in sorted(seen.items())})
PY

python scripts/camera_ready/incremental_proxy_validity_bootstrap.py \
  --input /tmp/gvt-proxy-smoke/input.jsonl \
  --out_dir /tmp/gvt-proxy-smoke/results \
  --metric acc
```

**Observed smoke result (TUD, November 2026):** the selection yielded **5,800 rows** (50 items in each of four benchmark/language groups across 29 models). The script exited with code 0 and wrote `incremental_regression.csv`, `incremental_proxy_validity.xlsx`, `incremental_pairs_used.csv`, `leave_one_family_out.csv`, `model_metadata.csv`, `model_scores_per_suite_language.csv`, `partial_correlations.csv` and `sensitivity_gmmlux_en.csv`. One example pairing was DanishCitizenTest/DA. **These sampled regression estimates must not be cited as the paper's full-data results.**

### Optional bootstrap smoke

To check only that bootstrap code executes, run the same command with `--bootstrap --n-boot 10 --bootstrap-seed 42` and a fresh output folder. Ten resamples are a software smoke check, **not** an uncertainty estimate for scientific reporting. The final frozen tables with the 5,000-resample analysis are in the [data repository](https://github.com/gold-vs-translation/cs-annotation-data/tree/main/results/proxy_validity).

## 3. Human annotation and meta-evaluation

The human reference is the 500-item majority-gold file. Individual IAA votes are pseudonymized (three votes per item) and frozen outputs are available in the data repository. The Camera-ready IAA script was originally written to ingest per-annotator JSONL files; direct reproduction *from the public `raw_votes_long.tsv`* has **not** been established by the smoke test above. Do not claim the raw-file ingestion can run unchanged on the public TSV. The meta-evaluation explicitly distinguishes directly judged CS/CA labels from CS/CA derived from `required_knowledge` and documents the handling of unresolved human majority ties.

## 4. Scope of reproduction

**Smoke-tested:** core incremental-validity data ingestion, score aggregation, nested regressions, English-control sensitivity, output generation using a representative sample of the released data. **Released but not comprehensively rerun:** all ranking variants, full bootstrap CIs, full-data paper-table regeneration, original model evaluations and API-based annotation. This distinction is deliberate: the released merged scores and frozen outputs let readers inspect and extend the analyses without needing to repeat expensive evaluations.
