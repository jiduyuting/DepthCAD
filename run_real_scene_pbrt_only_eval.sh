#!/usr/bin/env bash
set -euo pipefail

REAL_ROOT=${REAL_ROOT:-/data/pre_student/hcy/datasets/pbrt/Real}
VAL_SCENE=${VAL_SCENE:?Set VAL_SCENE, for example lab307 or library}

PYTHON_BIN=${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}
CHECKPOINT=${CHECKPOINT:-output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt}
RAW9_TRANSFORM=${RAW9_TRANSFORM:-auto}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-pbrt_only_eval_${VAL_SCENE}}
SPLIT_JSON=${SPLIT_JSON:-output/${EXPERIMENT_NAME}/split.json}
OUTPUT_DIR=${OUTPUT_DIR:-output/${EXPERIMENT_NAME}/real_val_selftest_auto}

"${PYTHON_BIN}" make_real_scene_holdout_split.py \
  --raw_dir "${REAL_ROOT}/noise" \
  --depth_dir "${REAL_ROOT}/depth" \
  --val_scenes "${VAL_SCENE}" \
  --output_json "${SPLIT_JSON}"

REAL_ROOT="${REAL_ROOT}" \
CHECKPOINT="${CHECKPOINT}" \
SPLIT_JSON="${SPLIT_JSON}" \
EVAL_SPLIT=val \
RAW9_TRANSFORM="${RAW9_TRANSFORM}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash run_pbrt_real_threshold_amp_depth_selftest.sh

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-depthcad} \
"${PYTHON_BIN}" analyze_real_val_failures.py \
  --selftest_dir "${OUTPUT_DIR}" \
  --top_k 12
