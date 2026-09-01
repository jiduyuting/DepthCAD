#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LDCM_ROOT="${LDCM_ROOT:-/data/pre_student/GJ/LDCM}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/lingbot-world/bin/python}"
MODEL="${MODEL:-pkqbajng/LDCM}"
MOGE_MODEL="${MOGE_MODEL:-Ruicheng/moge-2-vits-normal}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_completion_baselines/ldcm_zero_shot}"
DEVICE="${DEVICE:-cuda:0}"
LIMIT="${LIMIT:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -d "${LDCM_ROOT}" ]] || { echo "Missing LDCM repo: ${LDCM_ROOT}"; exit 2; }

if [[ -d "${MODEL}" && -f "${MODEL}/ldcm.pt" ]]; then
  MODEL="${MODEL}/ldcm.pt"
fi
if [[ -d "${MOGE_MODEL}" && -f "${MOGE_MODEL}/model.pt" ]]; then
  MOGE_MODEL="${MOGE_MODEL}/model.pt"
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${LDCM_ROOT}:${ROOT_DIR}"
"${PYTHON_BIN}" -c 'import torch, ldcm, utils3d; assert hasattr(utils3d.pt, "intrinsics_from_focal_center")' || {
  echo "LDCM dependency check failed. Install the pinned utils3d commit from the LDCM README into the selected Python environment."; exit 2;
}

EXTRA_ARGS=()
[[ -n "${LIMIT}" ]] && EXTRA_ARGS+=(--limit "${LIMIT}")
"${PYTHON_BIN}" "${ROOT_DIR}/eval_ldcm_full_pbrt.py" \
  --ldcm_root "${LDCM_ROOT}" --model "${MODEL}" --moge_model "${MOGE_MODEL}" \
  --cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" --save_predictions "${EXTRA_ARGS[@]}"
