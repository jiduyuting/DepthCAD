#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DMD_ROOT="${DMD_ROOT:-/data/pre_student/GJ/DMD3Cpp}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/llava/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/dmd3c/result_ema.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_completion_baselines/dmd3c_pbrt_train}"
RUN_NAME="${RUN_NAME:-dmd3c_pbrt_supervised_seed123}"
GPU="${GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-0.001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TEST_EPOCH="${TEST_EPOCH:-25}"
TEST_ITER="${TEST_ITER:-1000}"
LIMIT="${LIMIT:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -d "${DMD_ROOT}" ]] || { echo "Missing DMD3C repo: ${DMD_ROOT}"; exit 2; }
[[ -d "${CACHE_ROOT}" ]] || { echo "Missing PBRT cache: ${CACHE_ROOT}"; exit 2; }
[[ -f "${SPLIT_JSON}" ]] || { echo "Missing split JSON: ${SPLIT_JSON}"; exit 2; }

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-dmd3c-train}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
TORCH_LIB="$("${PYTHON_BIN}" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
CONDA_LIB="$("${PYTHON_BIN}" -c 'import os, sys; print(os.path.join(sys.prefix, "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${DMD_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -c 'import hydra, omegaconf, BpOps; from depth_anything_3.api import DepthAnything3' || {
  echo "DMD3C dependencies are incomplete. Run: bash scripts/setup_dmd3c_llava.sh"
  exit 2
}

LIMIT_SUFFIX="full"
EXPORT_ARGS=(--cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --overwrite)
if [[ -n "${LIMIT}" ]]; then
  LIMIT_SUFFIX="limit${LIMIT}"
  EXPORT_ARGS+=(--limit "${LIMIT}")
fi

TRAIN_UNIFORMAT_DIR="${TRAIN_UNIFORMAT_DIR:-${OUTPUT_DIR}/uniformat_train_${LIMIT_SUFFIX}}"
VAL_UNIFORMAT_DIR="${VAL_UNIFORMAT_DIR:-${OUTPUT_DIR}/uniformat_val_${LIMIT_SUFFIX}}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/export_pbrt_depth_completion_uniformat.py" \
  "${EXPORT_ARGS[@]}" --split train --output_dir "${TRAIN_UNIFORMAT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/export_pbrt_depth_completion_uniformat.py" \
  "${EXPORT_ARGS[@]}" --split test --output_dir "${VAL_UNIFORMAT_DIR}"

TRAIN_ARGS=(
  "${DMD_ROOT}/train_distill.py"
  "gpus=[${GPU}]"
  "gpu_id=${GPU}"
  "name=${RUN_NAME}"
  "data=UNI"
  "net=PMP_Residual_Norm_fast"
  "loss=MSMSE_pssil"
  "metric=MetricALL"
  "data.path=${TRAIN_UNIFORMAT_DIR}"
  "data.trainset.path=${TRAIN_UNIFORMAT_DIR}"
  "data.testset.path=${VAL_UNIFORMAT_DIR}"
  "train_batch_size=${BATCH_SIZE}"
  "test_batch_size=${TEST_BATCH_SIZE}"
  "num_workers=${NUM_WORKERS}"
  "nepoch=${EPOCHS}"
  "test_epoch=${TEST_EPOCH}"
  "test_iter=${TEST_ITER}"
  "lr=${LR}"
  "hydra.run.dir=${OUTPUT_DIR}/hydra/${RUN_NAME}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing DMD3C init checkpoint: ${CHECKPOINT}"; exit 2; }
  TRAIN_ARGS+=("++chpt=${CHECKPOINT}")
fi

cd "${OUTPUT_DIR}"
"${PYTHON_BIN}" "${TRAIN_ARGS[@]}"

echo "DMD3C PBRT training outputs: ${OUTPUT_DIR}"
echo "Checkpoints: ${OUTPUT_DIR}/checkpoints/${RUN_NAME}"
