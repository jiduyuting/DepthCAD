#!/usr/bin/env bash
set -euo pipefail

REAL_ROOT=${REAL_ROOT:-/data/pre_student/hcy/datasets/pbrt/Real}
MODEL_DIR=${MODEL_DIR:-output/pbrt_real_threshold_amp_depth_finetune_replay_auto_boundary_v2_e15_p8_keepamp_rw030}
CHECKPOINT=${CHECKPOINT:-${MODEL_DIR}/best.pt}
SPLIT_JSON=${SPLIT_JSON:-${MODEL_DIR}/split.json}
RAW9_TRANSFORM=${RAW9_TRANSFORM:-auto}

REAL_SELFTEST_DIR=${REAL_SELFTEST_DIR:-${MODEL_DIR}/real_val_selftest_auto}
PBRT_EVAL_DIR=${PBRT_EVAL_DIR:-${MODEL_DIR}/eval_pbrt_val97_endpoint}

echo "Checkpoint: ${CHECKPOINT}"
echo "Split JSON: ${SPLIT_JSON}"
echo "Real self-test dir: ${REAL_SELFTEST_DIR}"
echo "PBRT eval dir: ${PBRT_EVAL_DIR}"

REAL_ROOT="${REAL_ROOT}" \
CHECKPOINT="${CHECKPOINT}" \
SPLIT_JSON="${SPLIT_JSON}" \
EVAL_SPLIT=val \
RAW9_TRANSFORM="${RAW9_TRANSFORM}" \
OUTPUT_DIR="${REAL_SELFTEST_DIR}" \
bash run_pbrt_real_threshold_amp_depth_selftest.sh

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-depthcad} \
/home/lab507/anaconda3/envs/SVDC/bin/python scripts/analysis/analyze_real_val_failures.py \
  --selftest_dir "${REAL_SELFTEST_DIR}" \
  --top_k 12

CHECKPOINT="${CHECKPOINT}" \
OUTPUT_DIR="${PBRT_EVAL_DIR}" \
bash run_pbrt_val97_flow_eval.sh
