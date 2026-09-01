#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETIONFORMER_ROOT="${COMPLETIONFORMER_ROOT:-/data/pre_student/hcy/CompletionFormer}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/cformer/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
SOURCE_SPLIT="${SOURCE_SPLIT:-${ROOT_DIR}/output/full_pbrt_manifest_available_iq.json}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-72}"
LR="${LR:-0.001}"
RUN_NAME="${RUN_NAME:-completionformer_full_pbrt_amp3_depth_seed123}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output/completionformer_full_pbrt/train_logs/}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_completionformer_pbrt.py" \
  --cache_root "${CACHE_ROOT}" \
  --source_split "${SOURCE_SPLIT}" \
  --output "${SPLIT_JSON}"

ADAPTER_SOURCE="${ROOT_DIR}/integrations/completionformer/pbrtfull.py"
ADAPTER_TARGET="${COMPLETIONFORMER_ROOT}/src/data/pbrtfull.py"
if [[ ! -f "${ADAPTER_TARGET}" ]] || ! cmp -s "${ADAPTER_SOURCE}" "${ADAPTER_TARGET}"; then
  echo "CompletionFormer adapter is not installed or is outdated."
  echo "Run: cp '${ADAPTER_SOURCE}' '${ADAPTER_TARGET}'"
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${COMPLETIONFORMER_ROOT}/src"
export PYTHONPATH="${ROOT_DIR}/integrations/completionformer/compat:${COMPLETIONFORMER_ROOT}/src/model/deformconv/build/lib.linux-x86_64-3.8${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" main.py \
  --data_name PBRTFull \
  --dir_data "${CACHE_ROOT}" \
  --split_json "${SPLIT_JSON}" \
  --patch_height 256 \
  --patch_width 256 \
  --gpus "${GPU}" \
  --seed 123 \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --milestones 36 48 56 64 \
  --lr "${LR}" \
  --max_depth 10.0 \
  --loss '1.0*L1+1.0*L2' \
  --num_threads 4 \
  --log_dir "${LOG_DIR}" \
  --save "${RUN_NAME}"
