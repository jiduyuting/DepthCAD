#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/control/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/lab507/anaconda3/envs/control/bin/accelerate}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/stable-diffusion-2-1}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/pbrt_dataset}"
DATASET_CONFIG="${DATASET_CONFIG:-sd21_full_pbrt_train}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depthcad_sd21_full_pbrt}"
RESOLUTION="${RESOLUTION:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
EPOCHS="${EPOCHS:-500}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-5000}"
GPU="${GPU:-0}"
RESUME="${RESUME:-0}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -x "${ACCELERATE_BIN}" ]] || { echo "Missing accelerate: ${ACCELERATE_BIN}" >&2; exit 2; }
[[ -f "${MODEL_DIR}/model_index.json" ]] || { echo "Missing SD2.1 model: ${MODEL_DIR}" >&2; exit 2; }
[[ -d "${DATASET_DIR}/data/ideal_IQ_sd21_full_pbrt_train" ]] || {
  echo "Missing converted SD2.1 dataset. Run scripts/convert_unified_pbrt_to_sd21.py first." >&2
  exit 2
}

DATASETS_VERSION="$(${PYTHON_BIN} -c 'import datasets; print(datasets.__version__)')"
if [[ "${DATASETS_VERSION%%.*}" -ge 4 ]]; then
  echo "datasets ${DATASETS_VERSION} no longer supports pbrt_dataset.py loading scripts." >&2
  echo "Run: ${PYTHON_BIN} -m pip install 'datasets<4'" >&2
  exit 2
fi

args=(
  train_pbrt.py
  --pretrained_model_name_or_path "${MODEL_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --dataset_name "${DATASET_DIR}"
  --dataset_config "${DATASET_CONFIG}"
  --resolution "${RESOLUTION}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --gradient_checkpointing
  --enable_xformers_memory_efficient_attention
  --set_grads_to_none
  --learning_rate "${LEARNING_RATE}"
  --num_train_epochs "${EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --mixed_precision "${MIXED_PRECISION:-fp16}"
)
if [[ "${USE_8BIT_ADAM}" == "1" ]]; then
  args+=(--use_8bit_adam)
fi
if [[ "${RESUME}" == "1" ]]; then
  args+=(--resume_from_checkpoint latest)
fi

echo "[depthcad-sd21] GPU=${GPU} model=${MODEL_DIR}"
echo "[depthcad-sd21] config=${DATASET_CONFIG} resolution=${RESOLUTION}"
echo "[depthcad-sd21] output=${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${ACCELERATE_BIN}" launch --num_processes 1 "${args[@]}"
