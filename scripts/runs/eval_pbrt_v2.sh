#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# DepthCAD Evaluation Script for PBRT Dataset V2 Model
# 使用 scripts/eval_pbrt.py 评估 V2 模型的推理结果

# ====== 修改这里来指定预测结果目录 ======
# 默认为 inference_pbrt_v2.sh 的输出目录
# 可以通过环境变量覆盖
PRED_DIR="${PRED_DIR:-/data/pre_student/GJ/DepthCAD/pbrt/v2_latest}"

# 输出目录 (默认与预测目录相同)
OUT_DIR="${OUT_DIR:-$PRED_DIR}"

# 测试集列表
TEST_LIST_PATH="/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt"

echo "=========================================="
echo "DepthCAD V2 Evaluation for PBRT dataset"
echo "=========================================="
echo "Prediction directory: $PRED_DIR"
echo "Output directory: $OUT_DIR"
echo "Test list: $TEST_LIST_PATH"
echo "=========================================="

python scripts/eval_pbrt.py \
    --test_list_path "$TEST_LIST_PATH" \
    --out_dir "$OUT_DIR" \
    --pred_dir "$PRED_DIR"

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUT_DIR/result_metrics.txt"
echo "=========================================="
