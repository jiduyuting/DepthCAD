#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTHOR_ROOT="${DEPTHOR_ROOT:-/data/pre_student/GJ/Depthor}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/py310/bin/python}"
CHECKPOINT="${CHECKPOINT:-}"
DAV2_CHECKPOINT="${DAV2_CHECKPOINT:-${DEPTHOR_ROOT}/checkpoints/depth_anything_v2_vits.pth}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_completion_baselines/depthor}"
DEVICE="${DEVICE:-cuda:0}"
N_BINS="${N_BINS:-256}"
LIMIT="${LIMIT:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -n "${CHECKPOINT}" && -f "${CHECKPOINT}" ]] || {
  echo "Set CHECKPOINT to a DEPTHOR weight file."; exit 2;
}
[[ -f "${DAV2_CHECKPOINT}" ]] || { echo "Missing DAV2-small checkpoint: ${DAV2_CHECKPOINT}"; exit 2; }
export PYTHONPATH="${DEPTHOR_ROOT}/exts:${DEPTHOR_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -c 'import torch, timm, BpOps' || {
  echo "DEPTHOR dependencies are incomplete: timm and compiled BpOps are required."; exit 2;
}

EXTRA_ARGS=()
[[ -n "${LIMIT}" ]] && EXTRA_ARGS+=(--limit "${LIMIT}")
"${PYTHON_BIN}" "${ROOT_DIR}/eval_depthor_full_pbrt.py" \
  --depthor_root "${DEPTHOR_ROOT}" --checkpoint "${CHECKPOINT}" \
  --dav2_checkpoint "${DAV2_CHECKPOINT}" --device "${DEVICE}" \
  --cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --n_bins "${N_BINS}" \
  --save_predictions "${EXTRA_ARGS[@]}"
