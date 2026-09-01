#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMD_ROOT="${DMD_ROOT:-/data/pre_student/GJ/DMD3Cpp}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/llava/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/dmd3c/result_ema.pth}"
UNIFORMAT_DIR="${UNIFORMAT_DIR:-${ROOT_DIR}/output/depth_completion_baselines/uniformat_full_pbrt_val}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
SPLIT="${SPLIT:-test}"
RUN_NAME="${RUN_NAME:-DMD3C_PBRT_FULL}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${ROOT_DIR}/output/depth_completion_baselines/dmd3c/summary.json}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing DMD3C checkpoint: ${CHECKPOINT}"
  echo "Run: bash scripts/download_dmd3c_weights.sh"
  exit 2
}
CHECKPOINT="$(realpath "${CHECKPOINT}")"
UNIFORMAT_DIR="$(realpath -m "${UNIFORMAT_DIR}")"
CACHE_ROOT="$(realpath "${CACHE_ROOT}")"
SPLIT_JSON="$(realpath "${SPLIT_JSON}")"
SUMMARY_OUTPUT="$(realpath -m "${SUMMARY_OUTPUT}")"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-dmd3c}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
TORCH_LIB="$("${PYTHON_BIN}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
CONDA_LIB="$("${PYTHON_BIN}" -c 'import os, sys; print(os.path.join(sys.prefix, "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${DMD_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -c 'import hydra, omegaconf, BpOps; from depth_anything_3.api import DepthAnything3' || {
  echo "DMD3C dependencies are incomplete. Run: bash scripts/setup_dmd3c_llava.sh"
  exit 2
}

EXPORT_ARGS=()
[[ -n "${LIMIT}" ]] && EXPORT_ARGS+=(--limit "${LIMIT}")
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/export_pbrt_depth_completion_uniformat.py" \
  --cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --split "${SPLIT}" \
  --output_dir "${UNIFORMAT_DIR}" --overwrite "${EXPORT_ARGS[@]}"

cd "${DMD_ROOT}"
"${PYTHON_BIN}" test.py \
  "gpus=[${GPU}]" "gpu_id=${GPU}" "name=${RUN_NAME}" "++chpt=${CHECKPOINT}" \
  net=PMP_Residual_Norm_fast num_workers=4 data=UNI \
  "data.path=${UNIFORMAT_DIR}" test_batch_size=1 metric=MetricALL ++save=true

cd "${ROOT_DIR}"
"${PYTHON_BIN}" summarize_depth_completion_predictions.py \
  --method DMD3C --prediction_root "${DMD_ROOT}/results/${RUN_NAME}/test" \
  --prediction_format indexed_png --index_json "${UNIFORMAT_DIR}/index.json" \
  --cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --split "${SPLIT}" \
  --prediction_scale 0.00390625 \
  --output "${SUMMARY_OUTPUT}" "${EXPORT_ARGS[@]}"
