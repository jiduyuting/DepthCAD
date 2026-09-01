#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
MANIFEST="${MANIFEST:-output/full_pbrt_manifest_available_iq.json}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2}"

"${PYTHON_BIN}" scripts/make_full_pbrt_flow_lists.py \
  --manifest "${MANIFEST}" \
  --output_dir "${LIST_DIR}"

args=(
  scripts/train_depth_flow_restoration.py
  --cache_dir "${CACHE_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --train_list "${LIST_DIR}/train.txt"
  --val_list "${LIST_DIR}/val.txt"
  --device "${DEVICE}"
  --epochs "${EPOCHS:-120}"
  --batch_size "${BATCH_SIZE:-4}"
  --num_workers "${WORKERS:-0}"
  --lr "${LR:-1e-4}"
  --backbone "${BACKBONE:-transformer_bottleneck}"
  --input_mode "${INPUT_MODE:-noisy_iq_amp}"
  --anchor_mode "${ANCHOR_MODE:-noisy_ns}"
  --eval_sampling_mode "${SAMPLING_MODE:-endpoint}"
)

if [[ "${RESUME:-1}" == "1" && -f "${OUTPUT_DIR}/last.pt" ]]; then
  args+=(--resume)
fi

echo "[flow] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} GPU=${GPU} DEVICE=${DEVICE}"
echo "[flow] output=${OUTPUT_DIR}"
"${PYTHON_BIN}" -u "${args[@]}"
