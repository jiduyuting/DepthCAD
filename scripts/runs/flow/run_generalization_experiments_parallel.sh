#!/usr/bin/env bash
set -euo pipefail

# Parallel pilot/full runner: two held-out scenes x baseline/mixed-mask.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts/flow:${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
GPUS="${GPUS:-0,1,2,3}"; IFS=',' read -r GPU0 GPU1 GPU2 GPU3 <<< "${GPUS}"
PBRT_CACHE="${PBRT_CACHE:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
PBRT_LISTS="${PBRT_LISTS:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
FLAT_CACHE="${FLAT_CACHE:-${ROOT_DIR}/depth_completion_cache/flat_flow_pseudo_gt}"
FLAT_DATA="${FLAT_DATA:-${ROOT_DIR}/flat_dataset/data}"; FLAT_DEPTH="${FLAT_DEPTH:-/data/pre_student/hcy/ControlNet/data}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT_DIR}/output/parallel_high_gain/20260902_181031/valid_focus/best_global.pt}"
MODE="${MODE:-pilot}"; EPOCHS="${EPOCHS:-5}"; RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/output/generalization_parallel/$(date +%Y%m%d_%H%M%S)}"
WORKERS="${WORKERS:-2}"; BATCH_SIZE="${BATCH_SIZE:-4}"
[[ -d "${PBRT_CACHE}" && -f "${PBRT_LISTS}/train.txt" && -f "${BASE_CHECKPOINT}" ]] || { echo "Missing PBRT cache/lists/checkpoint" >&2; exit 2; }
if [[ "${MODE}" == "full" ]]; then EPOCHS="${FULL_EPOCHS:-24}"; fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/lists" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/eval"
if [[ ! -f "${FLAT_CACHE}/manifest.json" ]]; then
  "${PYTHON_BIN}" scripts/flow/prepare_flat_flow_cache.py --flat_data_root "${FLAT_DATA}" --depth_root "${FLAT_DEPTH}" --output_root "${FLAT_CACHE}" >"${RUN_ROOT}/logs/prepare_flat.log" 2>&1
fi
for scene in contemporary-bathroom white-room; do
  "${PYTHON_BIN}" scripts/flow/make_scene_holdout_flow_lists_v2.py --cache_dir "${PBRT_CACHE}" --source_dir "${PBRT_LISTS}" --holdout_scene "${scene}" --output_dir "${RUN_ROOT}/lists/${scene}" >"${RUN_ROOT}/logs/make_lists_${scene}.log" 2>&1
done

train_one() {
  local gpu="$1" name="$2" scene="$3" mixed="$4"; local extra=()
  if [[ "${mixed}" == "1" ]]; then extra=(--mask_augment --mask_augment_probability 0.5 --mask_augment_block_sizes 4 8 12 16 --mask_augment_hole_ratios 0.05 0.30 --mask_augment_noise_depth_root /data/pre_student/hcy/pbrt/noise_depth); fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/flow/train_depth_flow_restoration.py --cache_dir "${PBRT_CACHE}" --train_list "${RUN_ROOT}/lists/${scene}/train.txt" --val_list "${RUN_ROOT}/lists/${scene}/val.txt" --output_dir "${RUN_ROOT}/checkpoints/${name}" --device cuda:0 --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" --backbone transformer_bottleneck --input_mode noisy_iq_amp --anchor_mode noisy_ns --eval_sampling_mode endpoint --selection_metric global --hole_weight 6 --valid_weight 1 --grad_weight 0.5 --smooth_weight 0.02 --endpoint_weight 2 --base_channels 48 --res_blocks 2 --amp "${extra[@]}" >"${RUN_ROOT}/logs/${name}.train.log" 2>&1
}
declare -a PIDS=()
train_one "${GPU0}" contemporary_baseline contemporary-bathroom 0 & PIDS+=("$!")
train_one "${GPU1}" contemporary_mixed contemporary-bathroom 1 & PIDS+=("$!")
train_one "${GPU2}" white_baseline white-room 0 & PIDS+=("$!")
train_one "${GPU3}" white_mixed white-room 1 & PIDS+=("$!")
status=0; for pid in "${PIDS[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "Training failed; inspect ${RUN_ROOT}/logs" >&2; exit 1; }

eval_one() {
  local gpu="$1" name="$2" checkpoint="$3" list="$4" cache="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/flow/eval_depth_flow_restoration.py --checkpoint "${checkpoint}" --cache_dir "${cache}" --sample_list "${list}" --split all --output_dir "${RUN_ROOT}/eval/${name}" --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" --device cuda:0 --sampling_mode endpoint >"${RUN_ROOT}/logs/${name}.eval.log" 2>&1
}
declare -a EPIDS=()
eval_one "${GPU0}" contemporary_baseline_holdout "${RUN_ROOT}/checkpoints/contemporary_baseline/best_global.pt" "${RUN_ROOT}/lists/contemporary-bathroom/test.txt" "${PBRT_CACHE}" & EPIDS+=("$!")
eval_one "${GPU1}" contemporary_mixed_holdout "${RUN_ROOT}/checkpoints/contemporary_mixed/best_global.pt" "${RUN_ROOT}/lists/contemporary-bathroom/test.txt" "${PBRT_CACHE}" & EPIDS+=("$!")
eval_one "${GPU2}" white_baseline_holdout "${RUN_ROOT}/checkpoints/white_baseline/best_global.pt" "${RUN_ROOT}/lists/white-room/test.txt" "${PBRT_CACHE}" & EPIDS+=("$!")
eval_one "${GPU3}" white_mixed_holdout "${RUN_ROOT}/checkpoints/white_mixed/best_global.pt" "${RUN_ROOT}/lists/white-room/test.txt" "${PBRT_CACHE}" & EPIDS+=("$!")
for pid in "${EPIDS[@]}"; do wait "${pid}" || status=1; done
eval_one "${GPU0}" full_pbrt_to_flat "${BASE_CHECKPOINT}" "${FLAT_CACHE}/test.txt" "${FLAT_CACHE}" || status=1
cat >"${RUN_ROOT}/README.txt" <<EOF
mode=${MODE}
epochs=${EPOCHS}
pbrt_cache=${PBRT_CACHE}
flat_cache=${FLAT_CACHE}
base_checkpoint=${BASE_CHECKPOINT}
This run compares baseline vs mixed-mask on two held-out PBRT scenes and evaluates PBRT->FLAT zero-shot.
FLAT targets are paired pseudo-GT, not physical ground truth.
EOF
echo "Results: ${RUN_ROOT}"; exit "${status}"
