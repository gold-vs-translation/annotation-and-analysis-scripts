#!/usr/bin/env bash
# run_pipeline_demo.sh — Run analysis pipeline stages on dummy data.
#
# Usage:
#   bash scripts/run_pipeline_demo.sh
#
# Prerequisites:
#   pip install -e .
#   python scripts/generate_dummy_data.py
#
# NOTE: Stages requiring an OpenAI API key (annotation_gpt.py,
# validate_annotations.py) are skipped in this demo. All analysis
# and table-generation scripts run without API access.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_DIR/data/dummy"
OUT_DIR="$REPO_DIR/output/demo"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " Cultural Sensitivity Annotation Pipeline — Demo Run"
echo "============================================================"
echo " Data: $DATA_DIR"
echo " Output: $OUT_DIR"
echo ""

# ── Stage B: Prompt Calibration (Step 3 — eval only) ──────────
echo "[Stage B] Evaluating GPT annotations on dev split..."
python -m cs_annotation.automatic.step3_eval_split \
  --in "$DATA_DIR/globalmmlulite/text_globalmmlulite_EN_test_devsplit.jsonl" \
  --out "$OUT_DIR/dev_eval.xlsx" \
  --log WARNING \
  2>&1 || echo "  (step3_eval_split requires culture_ann + culture_ann_ref; skipping if format mismatch)"

echo ""

# ── Stage D: Paper Tables ─────────────────────────────────────
echo "[Stage D] Generating paper tables from human annotations..."
python src/cs_annotation/manual/generate_tables.py \
  --post-review "$DATA_DIR/human_annotations/all_user_annotations.jsonl" \
  --pre-review-dir "$DATA_DIR/pre_review_csvs/" \
  --latex-out "$OUT_DIR/tables/" \
  2>&1 || echo "  (generate_tables encountered an issue; see output above)"

echo ""

# ── Stage H: Dataset Analysis ─────────────────────────────────
echo "[Stage H] Analyzing dataset structure and MMLU mapping..."
python src/cs_annotation/datasets/analyze_datasets.py \
  --data-dir "$DATA_DIR/annotated" \
  --output-dir "$OUT_DIR/dataset_analysis/" \
  2>&1 || echo "  (analyze_datasets encountered an issue; see output above)"

echo ""

# ── Stage G: Ranking Analysis ─────────────────────────────────
echo "[Stage G] Building balanced CA/CS subsets..."
python src/cs_annotation/analysis/build_balanced_subsets.py \
  --input "$DATA_DIR/merged/all_merged_samplelogs.jsonl" \
  --out_dir "$OUT_DIR/ranking/" \
  --metric acc \
  2>&1 || echo "  (build_balanced_subsets encountered an issue; see output above)"

echo ""

echo "[Stage G] Computing baseline CA/CS rankings..."
if [ -f "$OUT_DIR/ranking/balanced_subsets_model_rows.jsonl" ]; then
  python src/cs_annotation/analysis/baseline_rank.py \
    --input "$OUT_DIR/ranking/balanced_subsets_model_rows.jsonl" \
    --out_dir "$OUT_DIR/ranking/" \
    2>&1 || echo "  (baseline_rank encountered an issue)"
else
  echo "  (skipping: balanced_subsets_model_rows.jsonl not found)"
fi

echo ""
echo "============================================================"
echo " Demo complete!"
echo ""
echo " Generated outputs in: $OUT_DIR/"
find "$OUT_DIR" -type f 2>/dev/null | sort | while read f; do
  echo "  $(du -h "$f" | cut -f1)  ${f#$OUT_DIR/}"
done
echo "============================================================"
