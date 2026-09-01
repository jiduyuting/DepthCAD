#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTHOR_ROOT="${DEPTHOR_ROOT:-/data/pre_student/GJ/Depthor}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/py310/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
DAV2_CHECKPOINT="${DAV2_CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/depthor/depth_anything_v2_vits.pth}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/depthor/depthor_zju_large.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_completion_baselines/depthor_pbrt_train}"
GPU="${GPU:-2}"
DEVICE="${DEVICE:-cuda:${GPU}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-0.0003}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HOLE_WEIGHT="${HOLE_WEIGHT:-4.0}"
EVAL_EVERY="${EVAL_EVERY:-1}"
LIMIT="${LIMIT:-}"
AMP="${AMP:-0}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -d "${DEPTHOR_ROOT}" ]] || { echo "Missing DEPTHOR repo: ${DEPTHOR_ROOT}"; exit 2; }
[[ -f "${DAV2_CHECKPOINT}" ]] || { echo "Missing DAV2 checkpoint: ${DAV2_CHECKPOINT}"; exit 2; }
[[ -f "${INIT_CHECKPOINT}" ]] || { echo "Missing DEPTHOR init checkpoint: ${INIT_CHECKPOINT}"; exit 2; }
[[ -d "${CACHE_ROOT}" ]] || { echo "Missing PBRT cache: ${CACHE_ROOT}"; exit 2; }
[[ -f "${SPLIT_JSON}" ]] || { echo "Missing split JSON: ${SPLIT_JSON}"; exit 2; }

export PYTHONPATH="${DEPTHOR_ROOT}/exts:${DEPTHOR_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -c 'import torch, timm, BpOps' || {
  echo "DEPTHOR dependencies are incomplete: timm and compiled BpOps are required."
  exit 2
}

ARGS=(
  "${ROOT_DIR}/train_depthor_pbrt.py"
  --depthor_root "${DEPTHOR_ROOT}"
  --dav2_checkpoint "${DAV2_CHECKPOINT}"
  --init_checkpoint "${INIT_CHECKPOINT}"
  --cache_root "${CACHE_ROOT}"
  --split_json "${SPLIT_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --hole_weight "${HOLE_WEIGHT}"
  --eval_every "${EVAL_EVERY}"
)

if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--train_limit "${LIMIT}" --val_limit "${LIMIT}")
fi
if [[ "${AMP}" == "1" ]]; then
  ARGS+=(--amp)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
