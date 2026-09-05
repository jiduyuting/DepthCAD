#!/usr/bin/env bash
set -euo pipefail

# Critical four-GPU study: preserve I/Q phase geometry and test cross-domain
# generalization without another unprincipled mask-parameter sweep.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts/flow:${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r GPU0 GPU1 GPU2 GPU3 <<< "${GPUS}"
PBRT_CACHE="${PBRT_CACHE:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
PBRT_LISTS="${PBRT_LISTS:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
# Train/val come from SPLIT_DIR; test stays on PBRT_LISTS. Defaulting SPLIT_DIR
# to PBRT_LISTS reproduces the old (leaky) protocol, so always set it to a
# scene-holdout dir from make_protocol_holdout_flow_lists.py for real runs --
# the default val shares all 40 camera views with train (98.6% of val frames
# have a +/-1 train neighbour), so selecting on it is anti-correlated with
# generalization.
SPLIT_DIR="${SPLIT_DIR:-${PBRT_LISTS}}"
FLAT_CACHE="${FLAT_CACHE:-${ROOT_DIR}/depth_completion_cache/flat_flow_matched_pbrt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${ROOT_DIR}/output/parallel_high_gain/20260902_181031/valid_focus/best_global.pt}"
MODE="${MODE:-pilot}"
RUN_ONLY="${RUN_ONLY:-all}"
EPOCHS="${EPOCHS:-8}"
if [[ "${MODE}" == "full" ]]; then EPOCHS="${FULL_EPOCHS:-24}"; fi
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/output/critical_generalization/$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-2}"

[[ -d "${PBRT_CACHE}" && -f "${PBRT_LISTS}/test.txt" ]] || { echo "Missing PBRT cache/test list" >&2; exit 2; }
[[ -f "${SPLIT_DIR}/train.txt" && -f "${SPLIT_DIR}/val.txt" ]] || { echo "Missing train/val lists under SPLIT_DIR=${SPLIT_DIR}" >&2; exit 2; }
[[ -f "${INIT_CHECKPOINT}" ]] || { echo "Missing INIT_CHECKPOINT=${INIT_CHECKPOINT}" >&2; exit 2; }
[[ -f "${FLAT_CACHE}/test.txt" ]] || { echo "Missing FLAT matched cache; run prepare_flat_matched_flow_cache.py first" >&2; exit 2; }
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/eval"

train_one() {
  local gpu="$1" name="$2" iq_norm="$3" distance="$4" velocity="$5" recon="$6" endpoint="$7"
  local distance_arg=()
  [[ "${distance}" == "1" ]] && distance_arg=(--include_hole_distance)
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/flow/train_depth_flow_restoration.py \
    --cache_dir "${PBRT_CACHE}" --train_list "${SPLIT_DIR}/train.txt" --val_list "${SPLIT_DIR}/val.txt" \
    --output_dir "${RUN_ROOT}/checkpoints/${name}" --device cuda:0 --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" --backbone transformer_bottleneck \
    --input_mode noisy_iq_amp --iq_normalization "${iq_norm}" --anchor_mode noisy_ns \
    --eval_sampling_mode endpoint --selection_metric global --hole_weight 1 --valid_weight 1 \
    --velocity_weight "${velocity}" --recon_weight "${recon}" --endpoint_weight "${endpoint}" \
    --grad_weight 0.5 --smooth_weight 0.02 --base_channels 48 --res_blocks 2 --amp \
    --init_checkpoint "${INIT_CHECKPOINT}" "${distance_arg[@]}" >"${RUN_ROOT}/logs/${name}.train.log" 2>&1
}

# A skipped branch must exit 0, otherwise `wait` below sees a failure and the
# whole script aborts before the eval stage (this is what silently killed the
# eval for every RUN_ONLY=<single branch> run).
run_selected() {
  if [[ "${RUN_ONLY}" == "all" || "${RUN_ONLY}" == "$1" ]]; then
    shift
    train_one "$@"
  fi
}
run_selected channel_control "${GPU0}" channel_control channel 0 1 1 2 & P0=$!
run_selected pairwise_phase "${GPU1}" pairwise_phase pairwise 0 1 1 2 & P1=$!
run_selected pairwise_distance "${GPU2}" pairwise_distance pairwise 1 1 1 2 & P2=$!
run_selected pairwise_endpoint "${GPU3}" pairwise_endpoint pairwise 0 0 0 1 & P3=$!
status=0
for pid in "$P0" "$P1" "$P2" "$P3"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "Training failed; inspect ${RUN_ROOT}/logs" >&2; exit 1; }

eval_one() {
  local gpu="$1" name="$2" checkpoint="$3" list="$4" cache="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/flow/eval_depth_flow_restoration.py \
    --checkpoint "${checkpoint}" --cache_dir "${cache}" --sample_list "${list}" --split all \
    --output_dir "${RUN_ROOT}/eval/${name}" --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" \
    --device cuda:0 --sampling_mode endpoint >"${RUN_ROOT}/logs/${name}.eval.log" 2>&1
}

eval_pids=()
eval_names=()
for spec in \
  "channel_control:${GPU0}" "pairwise_phase:${GPU1}" \
  "pairwise_distance:${GPU2}" "pairwise_endpoint:${GPU3}"; do
  name="${spec%%:*}"; gpu="${spec##*:}"
  [[ "${RUN_ONLY}" == "all" || "${RUN_ONLY}" == "${name}" ]] || continue
  ckpt="${RUN_ROOT}/checkpoints/${name}/best_global.pt"
  [[ -f "${ckpt}" ]] || { echo "Missing checkpoint ${ckpt}; training produced no best_global.pt" >&2; exit 1; }
  eval_one "${gpu}" "${name}_pbrt100" "${ckpt}" "${PBRT_LISTS}/test.txt" "${PBRT_CACHE}" &
  eval_pids+=($!); eval_names+=("${name}_pbrt100")
  eval_one "${gpu}" "${name}_flat_matched" "${ckpt}" "${FLAT_CACHE}/test.txt" "${FLAT_CACHE}" &
  eval_pids+=($!); eval_names+=("${name}_flat_matched")
done
# A bare `wait` always returns 0, which would let a failed eval print a success
# banner and a README claiming results exist. Check each job explicitly.
eval_status=0
for i in "${!eval_pids[@]}"; do
  wait "${eval_pids[$i]}" || { echo "Eval failed: ${eval_names[$i]} (see ${RUN_ROOT}/logs/${eval_names[$i]}.eval.log)" >&2; eval_status=1; }
done
(( eval_status == 0 )) || { echo "Evaluation failed; inspect ${RUN_ROOT}/logs" >&2; exit 1; }

printf '%s\n' "mode=${MODE}" "epochs=${EPOCHS}" "run_only=${RUN_ONLY}" "split_dir=${SPLIT_DIR}" \
  "init_checkpoint=${INIT_CHECKPOINT}" \
  "PBRT100 and FLAT matched evaluations are under ${RUN_ROOT}/eval" >"${RUN_ROOT}/README.txt"
echo "Results: ${RUN_ROOT}"
