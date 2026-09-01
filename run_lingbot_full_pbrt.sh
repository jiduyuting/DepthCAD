#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_ROOT="${LINGBOT_ROOT:-/data/pre_student/GJ/lingbot-depth}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
MODEL="${MODEL:-robbyant/lingbot-depth-postrain-dc-vitl14}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_completion_baselines/lingbot_dc_zero_shot}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -d "${LINGBOT_ROOT}" ]] || { echo "Missing LingBot repo: ${LINGBOT_ROOT}"; exit 2; }

if [[ -d "${MODEL}" && -f "${MODEL}/model.pt" ]]; then
  MODEL="${MODEL}/model.pt"
fi

export PYTHONPATH="${LINGBOT_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -c 'import torch, xformers; from mdm.model.v2 import MDMModel' || {
  echo "LingBot dependency check failed. Recommended environment: depthcad."; exit 2;
}

if [[ "${MODEL}" == robbyant/* ]]; then
  echo "LingBot model: ${MODEL}"
  echo "The first run downloads model.pt (~1.28 GB); interrupted downloads resume from the Hugging Face cache."
fi

EXTRA_ARGS=()
[[ -n "${LIMIT}" ]] && EXTRA_ARGS+=(--limit "${LIMIT}")
"${PYTHON_BIN}" "${ROOT_DIR}/eval_lingbot_full_pbrt.py" \
  --lingbot_root "${LINGBOT_ROOT}" --model "${MODEL}" --device "${DEVICE}" \
  --cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_predictions "${EXTRA_ARGS[@]}"
