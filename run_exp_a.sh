#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/flow_exp_conservative}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}"
DEVICE="${DEVICE:-cuda:0}"
AMP="${AMP:-1}"

extra_args=()
if [[ "${AMP}" == "1" ]]; then
  extra_args+=(--amp)
fi

mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${OUTPUT_DIR}/last.pt" ]]; then
  [[ -f "${BASE_CHECKPOINT}" ]] || { echo "Missing baseline checkpoint: ${BASE_CHECKPOINT}" >&2; exit 2; }
  cp "${BASE_CHECKPOINT}" "${OUTPUT_DIR}/last.pt"
fi

echo "[exp-a] Conservative fine-tune from the epoch-108 Flow checkpoint"
echo "[exp-a] output=${OUTPUT_DIR}"

"${PYTHON_BIN}" -u train_depth_flow_restoration.py \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_list "${LIST_DIR}/train.txt" \
  --val_list "${LIST_DIR}/val.txt" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS:-120}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --num_workers "${WORKERS:-0}" \
  --lr "${LR:-2e-6}" \
  --backbone transformer_bottleneck \
  --input_mode noisy_iq_amp \
  --anchor_mode noisy_ns \
  --eval_sampling_mode endpoint \
  --resume \
  --selection_metric global \
  --hole_weight 5.0 \
  --valid_weight 1.0 \
  --grad_weight 0.5 \
  --smooth_weight 0.02 \
  --anchor_weight 0.02 \
  "${extra_args[@]}"
