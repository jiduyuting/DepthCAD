#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/flow_exp_hole_distance}"
DEVICE="${DEVICE:-cuda:0}"
AMP="${AMP:-1}"

extra_args=()
if [[ "${AMP}" == "1" ]]; then
  extra_args+=(--amp)
fi

echo "[exp-c] Flow with normalized hole-distance input, trained from scratch"
echo "[exp-c] output=${OUTPUT_DIR}"

"${PYTHON_BIN}" -u scripts/train_depth_flow_restoration.py \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_list "${LIST_DIR}/train.txt" \
  --val_list "${LIST_DIR}/val.txt" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS:-120}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --num_workers "${WORKERS:-0}" \
  --lr "${LR:-1e-4}" \
  --backbone transformer_bottleneck \
  --input_mode noisy_iq_amp \
  --anchor_mode noisy_ns \
  --eval_sampling_mode endpoint \
  --include_hole_distance \
  --selection_metric global \
  --hole_weight 5.0 \
  --valid_weight 1.0 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  "${extra_args[@]}"
