#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
TRAIN_LIST="${TRAIN_LIST:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq/train.txt}"
VAL_LIST="${VAL_LIST:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq/val.txt}"
TEST_LIST="${TEST_LIST:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq/test.txt}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine}"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-5e-5}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
RES_BLOCKS="${RES_BLOCKS:-1}"
PROPAGATION_STEPS="${PROPAGATION_STEPS:-6}"
REFINE_DILATE_RADIUS="${REFINE_DILATE_RADIUS:-3}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-1.5}"
SELECTION_METRIC="${SELECTION_METRIC:-hole}"
SAVE_EVERY="${SAVE_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-20}"
AMP="${AMP:-1}"
RESUME="${RESUME:-0}"
EVAL_AFTER="${EVAL_AFTER:-1}"
VISUALIZE="${VISUALIZE:-1}"
VIS_MAX_SAMPLES="${VIS_MAX_SAMPLES:-12}"

train_args=(
  "${ROOT_DIR}/train_depth_flow_propagation_refine.py"
  --cache_dir "${CACHE_DIR}"
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --train_list "${TRAIN_LIST}"
  --val_list "${VAL_LIST}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --base_channels "${BASE_CHANNELS}"
  --res_blocks "${RES_BLOCKS}"
  --propagation_steps "${PROPAGATION_STEPS}"
  --refine_dilate_radius "${REFINE_DILATE_RADIUS}"
  --residual_scale "${RESIDUAL_SCALE}"
  --selection_metric "${SELECTION_METRIC}"
  --save_every "${SAVE_EVERY}"
  --log_every "${LOG_EVERY}"
)

if [[ -n "${DEVICE}" ]]; then
  train_args+=(--device "${DEVICE}")
fi
if [[ "${AMP}" == "1" ]]; then
  train_args+=(--amp)
fi
if [[ "${RESUME}" == "1" ]]; then
  train_args+=(--resume)
fi

echo "Training Flow-anchor propagation refine"
echo "  checkpoint: ${PRETRAINED_CHECKPOINT}"
echo "  output:     ${OUTPUT_DIR}"
echo "  train/val:  ${TRAIN_LIST} / ${VAL_LIST}"
"${PYTHON_BIN}" -u "${train_args[@]}"

if [[ "${EVAL_AFTER}" == "1" ]]; then
  EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${OUTPUT_DIR}/best.pt}"
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval_test_best}"
  eval_args=(
    "${ROOT_DIR}/eval_depth_flow_propagation_refine.py"
    --checkpoint "${EVAL_CHECKPOINT}"
    --sample_list "${TEST_LIST}"
    --split all
    --output_dir "${EVAL_OUTPUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
  )
  if [[ -n "${DEVICE}" ]]; then
    eval_args+=(--device "${DEVICE}")
  fi
  if [[ "${VISUALIZE}" == "1" ]]; then
    eval_args+=(--visualize --vis_rank best_worst_hole --vis_max_samples "${VIS_MAX_SAMPLES}")
  fi

  echo "Evaluating on PBRT100 seed123 holdout"
  echo "  checkpoint: ${EVAL_CHECKPOINT}"
  echo "  output:     ${EVAL_OUTPUT_DIR}"
  "${PYTHON_BIN}" -u "${eval_args[@]}"
fi
