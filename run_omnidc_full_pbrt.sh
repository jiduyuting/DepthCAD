#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNI_ROOT="${OMNI_ROOT:-/data/pre_student/GJ/OMNI-DC}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/cformer/bin/python}"
CHECKPOINT="${CHECKPOINT:-${OMNI_ROOT}/checkpoints/modelv1.1_best_72epochs.pt}"
DAV2_CHECKPOINT="${DAV2_CHECKPOINT:-${OMNI_ROOT}/src/depth_models/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth}"
UNIFORMAT_DIR="${UNIFORMAT_DIR:-${ROOT_DIR}/output/depth_completion_baselines/uniformat_full_pbrt_val}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
SPLIT="${SPLIT:-test}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${ROOT_DIR}/output/depth_completion_baselines/omnidc_zero_shot/summary.json}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output/depth_completion_baselines/omnidc_runs/}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing OMNI-DC checkpoint: ${CHECKPOINT}"; exit 2; }
[[ -f "${DAV2_CHECKPOINT}" ]] || { echo "Missing Depth Anything V2 checkpoint: ${DAV2_CHECKPOINT}"; exit 2; }
CHECKPOINT="$(realpath "${CHECKPOINT}")"
DAV2_CHECKPOINT="$(realpath "${DAV2_CHECKPOINT}")"
UNIFORMAT_DIR="$(realpath -m "${UNIFORMAT_DIR}")"
CACHE_ROOT="$(realpath "${CACHE_ROOT}")"
SPLIT_JSON="$(realpath "${SPLIT_JSON}")"
SUMMARY_OUTPUT="$(realpath -m "${SUMMARY_OUTPUT}")"
LOG_DIR="$(realpath -m "${LOG_DIR}")"
"${PYTHON_BIN}" -c 'import torch, apex' || { echo "OMNI-DC requires torch and apex."; exit 2; }

EXPORT_ARGS=()
[[ -n "${LIMIT}" ]] && EXPORT_ARGS+=(--limit "${LIMIT}")
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/export_pbrt_depth_completion_uniformat.py" \
  --cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --split "${SPLIT}" \
  --output_dir "${UNIFORMAT_DIR}" --overwrite "${EXPORT_ARGS[@]}"

mkdir -p "${LOG_DIR}"
cd "${OMNI_ROOT}/src"
"${PYTHON_BIN}" main.py \
  --dir_data "${UNIFORMAT_DIR}" --val_data_name Uniformat --max_depth 10.0 \
  --data_normalize_median 1 --num_resolution 3 \
  --multi_resolution_learnable_gradients_weights uniform --load_dav2 1 \
  --gpus "${GPU}" --GRU_iters 1 --optim_layer_input_clamp 1.0 \
  --depth_activation_format exp --whiten_sparse_depths 1 --gru_internal_whiten_method median \
  --log_dir "${LOG_DIR}/" --save pbrt_full_v11 --backbone_mode rgbd \
  --pred_confidence_input 1 --pretrain "${CHECKPOINT}" --save_result_only --test_only

PREDICTION_DIR="$(find "${LOG_DIR}" -type d -path '*/test/epoch*' 2>/dev/null | sort | tail -n 1)"
if [[ -z "${PREDICTION_DIR}" ]]; then
  PREDICTION_DIR="$(find "${LOG_DIR}"* -type d -path '*/test/epoch*' 2>/dev/null | sort | tail -n 1)"
fi
[[ -n "${PREDICTION_DIR}" ]] || { echo "Could not locate OMNI-DC predictions under ${LOG_DIR}"; exit 3; }
cd "${ROOT_DIR}"
"${PYTHON_BIN}" summarize_depth_completion_predictions.py \
  --method OMNI-DC-v1.1-zero-shot --prediction_root "${PREDICTION_DIR}" \
  --prediction_format indexed_png --index_json "${UNIFORMAT_DIR}/index.json" \
  --cache_root "${CACHE_ROOT}" --split_json "${SPLIT_JSON}" --split "${SPLIT}" \
  --prediction_scale 0.00390625 \
  --output "${SUMMARY_OUTPUT}" "${EXPORT_ARGS[@]}"
