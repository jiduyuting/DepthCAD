#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"

CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
TRAIN_LIST="${TRAIN_LIST:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq/train.txt}"
VAL_LIST="${VAL_LIST:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq/val.txt}"
TEST_LIST="${TEST_LIST:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq/test.txt}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}"
ANCHOR_CACHE_DIR="${ANCHOR_CACHE_DIR:?ANCHOR_CACHE_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine_cached}"

DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-2e-5}"
SELECTION_METRIC="${SELECTION_METRIC:-composite}"
AMP="${AMP:-1}"
RESUME="${RESUME:-1}"
EVAL_AFTER="${EVAL_AFTER:-1}"
VISUALIZE="${VISUALIZE:-1}"

train_args=(
  "${ROOT_DIR}/scripts/train_depth_flow_propagation_refine.py"
  --cache_dir "${CACHE_DIR}"
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --anchor_cache_dir "${ANCHOR_CACHE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --train_list "${TRAIN_LIST}"
  --val_list "${VAL_LIST}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --selection_metric "${SELECTION_METRIC}"
  --device "${DEVICE}"
)
if [[ "${AMP}" == "1" ]]; then train_args+=(--amp); fi
if [[ "${RESUME}" == "1" ]]; then train_args+=(--resume); fi

"${PYTHON_BIN}" -u "${train_args[@]}"

if [[ "${EVAL_AFTER}" == "1" ]]; then
  eval_args=(
    "${ROOT_DIR}/scripts/eval_depth_flow_propagation_refine.py"
    --checkpoint "${OUTPUT_DIR}/best.pt"
    --cache_dir "${CACHE_DIR}"
    --sample_list "${TEST_LIST}"
    --split all
    --output_dir "${OUTPUT_DIR}/eval_test_best"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --anchor_cache_dir "${ANCHOR_CACHE_DIR}"
  )
  if [[ "${VISUALIZE}" == "1" ]]; then
    eval_args+=(--visualize --vis_rank best_worst_hole --vis_max_samples "${VIS_MAX_SAMPLES:-12}")
  fi
  "${PYTHON_BIN}" -u "${eval_args[@]}"
fi
