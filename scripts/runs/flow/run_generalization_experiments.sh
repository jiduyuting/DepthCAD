#!/usr/bin/env bash
set -euo pipefail

# One-click protocol runner for PBRT unseen-scene and PBRT -> FLAT tests.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts/flow:${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
GPU="${GPU:-0}"
PBRT_CACHE="${PBRT_CACHE:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
PBRT_LISTS="${PBRT_LISTS:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
FLAT_CACHE="${FLAT_CACHE:-${ROOT_DIR}/depth_completion_cache/flat_flow_pseudo_gt}"
FLAT_DATA="${FLAT_DATA:-${ROOT_DIR}/flat_dataset/data}"
FLAT_DEPTH="${FLAT_DEPTH:-/data/pre_student/hcy/ControlNet/data}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT_DIR}/output/parallel_high_gain/20260902_181031/valid_focus/best_global.pt}"
HOLDOUT_SCENE="${HOLDOUT_SCENE:-contemporary-bathroom}"
MODE="${MODE:-pilot}"
EPOCHS="${EPOCHS:-5}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/output/generalization_experiments/$(date +%Y%m%d_%H%M%S)}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"

[[ -d "${PBRT_CACHE}" ]] || { echo "Missing PBRT cache: ${PBRT_CACHE}" >&2; exit 2; }
[[ -f "${PBRT_LISTS}/train.txt" ]] || { echo "Missing PBRT lists: ${PBRT_LISTS}" >&2; exit 2; }
[[ -f "${BASE_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${BASE_CHECKPOINT}" >&2; exit 2; }
if [[ "${MODE}" == "full" ]]; then EPOCHS="${FULL_EPOCHS:-24}"; fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/lists" "${RUN_ROOT}/scene_baseline" "${RUN_ROOT}/scene_mixed_mask"

if [[ ! -f "${FLAT_CACHE}/manifest.json" ]]; then
  "${PYTHON_BIN}" scripts/flow/prepare_flat_flow_cache.py \
    --flat_data_root "${FLAT_DATA}" --depth_root "${FLAT_DEPTH}" --output_root "${FLAT_CACHE}" \
    >"${RUN_ROOT}/logs/prepare_flat.log" 2>&1
fi

"${PYTHON_BIN}" scripts/flow/make_scene_holdout_flow_lists.py \
  --cache_dir "${PBRT_CACHE}" --source_dir "${PBRT_LISTS}" \
  --holdout_scene "${HOLDOUT_SCENE}" --output_dir "${RUN_ROOT}/lists/pbrt_${HOLDOUT_SCENE}" \
  >"${RUN_ROOT}/logs/make_scene_lists.log" 2>&1

TRAIN_LIST="${RUN_ROOT}/lists/pbrt_${HOLDOUT_SCENE}/train.txt"
VAL_LIST="${RUN_ROOT}/lists/pbrt_${HOLDOUT_SCENE}/val.txt"
TEST_LIST="${RUN_ROOT}/lists/pbrt_${HOLDOUT_SCENE}/test.txt"

run_scene() {
  local name="$1"; shift
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u scripts/flow/train_depth_flow_restoration.py \
    --cache_dir "${PBRT_CACHE}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
    --output_dir "${RUN_ROOT}/${name}" --device cuda:0 --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" --backbone transformer_bottleneck \
    --input_mode noisy_iq_amp --anchor_mode noisy_ns --eval_sampling_mode endpoint \
    --selection_metric global --hole_weight 6 --valid_weight 1 --grad_weight 0.5 \
    --smooth_weight 0.02 --endpoint_weight 2 --base_channels 48 --res_blocks 2 --amp \
    "$@" >"${RUN_ROOT}/logs/${name}.train.log" 2>&1
}

run_scene scene_baseline
run_scene scene_mixed_mask --mask_augment --mask_augment_probability 0.5 \
  --mask_augment_block_sizes 4 8 12 16 --mask_augment_hole_ratios 0.05 0.30 \
  --mask_augment_noise_depth_root /data/pre_student/hcy/pbrt/noise_depth

eval_checkpoint() {
  local name="$1"; local checkpoint="$2"; local cache="$3"; local list="$4"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u scripts/flow/eval_depth_flow_restoration.py \
    --checkpoint "${checkpoint}" --cache_dir "${cache}" --sample_list "${list}" \
    --split all --output_dir "${RUN_ROOT}/${name}" --batch_size "${BATCH_SIZE}" \
    --num_workers "${WORKERS}" --device cuda:0 --sampling_mode endpoint \
    >"${RUN_ROOT}/logs/${name}.eval.log" 2>&1
}

eval_checkpoint scene_baseline_holdout "${RUN_ROOT}/scene_baseline/best_global.pt" "${PBRT_CACHE}" "${TEST_LIST}"
eval_checkpoint scene_mixed_mask_holdout "${RUN_ROOT}/scene_mixed_mask/best_global.pt" "${PBRT_CACHE}" "${TEST_LIST}"
eval_checkpoint full_pbrt_to_flat "${BASE_CHECKPOINT}" "${FLAT_CACHE}" "${FLAT_CACHE}/test.txt"

cat >"${RUN_ROOT}/README.txt" <<EOF
holdout_scene=${HOLDOUT_SCENE}
mode=${MODE}
pbrt_cache=${PBRT_CACHE}
flat_cache=${FLAT_CACHE}
base_checkpoint=${BASE_CHECKPOINT}
The FLAT target is paired pseudo-GT, not physical ground truth.
EOF
echo "Results: ${RUN_ROOT}"
