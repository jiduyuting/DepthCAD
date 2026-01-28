#!/bin/bash

# ============================================================================
# HYPIR-Enhanced DepthCAD Evaluation Script
# ============================================================================
#
# This script evaluates depth predictions from models trained with train_hypir.py.
#
# Features:
# - No amplitude masking (evaluates all pixels)
# - Automatic GT file matching
# - Outputs per-image and aggregated metrics
#
# ============================================================================

# ===== Configuration =====

# Prediction directory (output from inference_hypir.sh)
PRED_DIR="pbrt/data_hypir_30000"

# Output directory for evaluation results
OUT_DIR="out_pbrt/eval_hypir_30000"

# Ground truth directory
GT_DIR="/data/pre_student/hcy/pbrt/gt_depth"

# Create output directory
mkdir -p "$OUT_DIR"

# ===== Header =====
echo "=========================================="
echo "HYPIR-Enhanced DepthCAD Evaluation"
echo "=========================================="
echo "Prediction directory: $PRED_DIR"
echo "Ground truth directory: $GT_DIR"
echo "Output directory: $OUT_DIR"
echo "=========================================="
echo ""

# ===== Run Evaluation =====
python eval_hypir.py \
    --pred_dir "$PRED_DIR" \
    --out_dir "$OUT_DIR" \
    --gt_dir "$GT_DIR"

# ===== Display Results =====
if [ -f "$OUT_DIR/result_metrics.txt" ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation Results"
    echo "=========================================="
    cat "$OUT_DIR/result_metrics.txt"
    echo "=========================================="
    echo ""
    echo "Detailed per-image metrics saved to:"
    echo "  $OUT_DIR/per_image_metrics.csv"
    echo "  $OUT_DIR/result_metrics.txt"
fi
