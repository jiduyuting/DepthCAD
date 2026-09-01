#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# This runner keeps the existing PBRT100 test split untouched. Training selects
# checkpoints on the 990-sample validation split; the 100-sample test split is
# only evaluated after each job finishes.
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
SUMMARY_PYTHON="${SUMMARY_PYTHON:-python3}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
ANCHOR_CACHE_DIR="${ANCHOR_CACHE_DIR:-${ROOT_DIR}/output/flow_anchor_cache_epoch108}"
BASE_FLOW_CHECKPOINT="${BASE_FLOW_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT_DIR}/output/next_depth_experiments/$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${EXPERIMENT_ROOT}/logs"

GPU_B="${GPU_B:-0}"
GPU_A="${GPU_A:-1}"
GPU_C="${GPU_C:-2}"
DEVICE_MODE="${DEVICE_MODE:-auto}"
RUN_B_EVAL="${RUN_B_EVAL:-1}"
RUN_A="${RUN_A:-1}"
RUN_C="${RUN_C:-1}"
ALLOW_CPU="${ALLOW_CPU:-0}"

mkdir -p "${LOG_DIR}"

if [[ "${DEVICE_MODE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE_MODE="cuda"
  else
    DEVICE_MODE="cpu"
  fi
fi

if [[ "${DEVICE_MODE}" == "cuda" ]]; then
  DEVICE_B="cuda:0"
  DEVICE_A="cuda:0"
  DEVICE_C="cuda:0"
else
  echo "[next-round] CUDA unavailable; using CPU. Set DEVICE_MODE=cuda when a driver is available."
  DEVICE_B="cpu"
  DEVICE_A="cpu"
  DEVICE_C="cpu"
fi

if [[ "${DEVICE_MODE}" != "cuda" && "${ALLOW_CPU}" != "1" && ( "${RUN_A}" == "1" || "${RUN_C}" == "1" ) ]]; then
  echo "[next-round] Refusing long CPU training. Use a CUDA node or explicitly set ALLOW_CPU=1." >&2
  exit 2
fi

echo "[next-round] root=${EXPERIMENT_ROOT}"
echo "[next-round] device=${DEVICE_MODE}; B=${GPU_B}, A=${GPU_A}, C=${GPU_C}"
echo "[next-round] logs=${LOG_DIR}"

run_b_evaluation() {
  local out_root="${EXPERIMENT_ROOT}/b_checkpoints"
  mkdir -p "${out_root}"
  local checkpoint name duplicate prior
  local evaluated=()
  for checkpoint in \
    "${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine/best.pt" \
    "${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine/best_hole.pt" \
    "${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine/best_global.pt"; do
    [[ -f "${checkpoint}" ]] || { echo "[next-round][B] missing ${checkpoint}" >&2; continue; }
    name="$(basename "${checkpoint}" .pt)"
    duplicate=""
    for prior in "${evaluated[@]}"; do
      if cmp -s "${checkpoint}" "${prior}"; then
        duplicate="${prior}"
        break
      fi
    done
    if [[ -n "${duplicate}" ]]; then
      echo "[next-round][B] skipping ${name}; byte-identical to $(basename "${duplicate}")"
      continue
    fi
    evaluated+=("${checkpoint}")
    "${PYTHON_BIN}" -u eval_depth_flow_propagation_refine.py \
      --checkpoint "${checkpoint}" \
      --cache_dir "${CACHE_DIR}" \
      --sample_list "${LIST_DIR}/test.txt" \
      --split all \
      --output_dir "${out_root}/${name}" \
      --batch_size "${BATCH_SIZE_B:-4}" \
      --num_workers "${WORKERS_B:-0}" \
      --device "${DEVICE_B}" \
      --anchor_cache_dir "${ANCHOR_CACHE_DIR}" \
      --vis_rank best_worst_hole \
      --visualize \
      --vis_max_samples 12
  done

  checkpoint="${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine/best.pt"
  if [[ -f "${checkpoint}" ]]; then
    "${PYTHON_BIN}" -u eval_depth_flow_propagation_refine.py \
      --checkpoint "${checkpoint}" \
      --cache_dir "${CACHE_DIR}" \
      --sample_list "${LIST_DIR}/test.txt" \
      --split all \
      --output_dir "${out_root}/best_preserve_observed" \
      --batch_size "${BATCH_SIZE_B:-4}" \
      --num_workers "${WORKERS_B:-0}" \
      --device "${DEVICE_B}" \
      --anchor_cache_dir "${ANCHOR_CACHE_DIR}" \
      --preserve_observed
  fi
}

run_a_training_and_evaluation() {
  local out_root="${EXPERIMENT_ROOT}/a_conservative"
  DEVICE="${DEVICE_A}" BASE_CHECKPOINT="${BASE_FLOW_CHECKPOINT}" \
    OUTPUT_DIR="${out_root}" EPOCHS="${EPOCHS_A:-120}" \
    BATCH_SIZE="${BATCH_SIZE_A:-4}" LR="${LR_A:-2e-6}" \
    AMP="${AMP_A:-1}" VISUALIZE=0 \
    bash run_exp_a.sh
  [[ -f "${out_root}/best_hole.pt" ]] || { echo "[next-round][A] best_hole.pt missing" >&2; return 1; }
  for preserve in raw preserve_observed; do
    local args=()
    [[ "${preserve}" == "preserve_observed" ]] && args+=(--preserve_observed)
    "${PYTHON_BIN}" -u eval_depth_flow_restoration.py \
      --checkpoint "${out_root}/best_hole.pt" \
      --cache_dir "${CACHE_DIR}" \
      --sample_list "${LIST_DIR}/test.txt" \
      --split all \
      --output_dir "${out_root}/eval_test_${preserve}" \
      --batch_size "${BATCH_SIZE_A:-4}" \
      --num_workers "${WORKERS_A:-0}" \
      --device "${DEVICE_A}" \
      --sampling_mode endpoint \
      "${args[@]}"
  done
}

run_c_training_and_evaluation() {
  local out_root="${EXPERIMENT_ROOT}/c_hole_distance"
  DEVICE="${DEVICE_C}" OUTPUT_DIR="${out_root}" \
    EPOCHS="${EPOCHS_C:-120}" BATCH_SIZE="${BATCH_SIZE_C:-2}" \
    LR="${LR_C:-1e-4}" WORKERS="${WORKERS_C:-0}" AMP="${AMP_C:-1}" \
    bash run_exp_c.sh
  [[ -f "${out_root}/best_hole.pt" ]] || { echo "[next-round][C] best_hole.pt missing" >&2; return 1; }
  for preserve in raw preserve_observed; do
    local args=()
    [[ "${preserve}" == "preserve_observed" ]] && args+=(--preserve_observed)
    "${PYTHON_BIN}" -u eval_depth_flow_restoration.py \
      --checkpoint "${out_root}/best_hole.pt" \
      --cache_dir "${CACHE_DIR}" \
      --sample_list "${LIST_DIR}/test.txt" \
      --split all \
      --output_dir "${out_root}/eval_test_${preserve}" \
      --batch_size "${BATCH_SIZE_C:-2}" \
      --num_workers "${WORKERS_C:-0}" \
      --device "${DEVICE_C}" \
      --sampling_mode endpoint \
      "${args[@]}"
  done
}

pids=()
labels=()
if [[ "${RUN_B_EVAL}" == "1" ]]; then
  if [[ "${DEVICE_MODE}" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_B}" run_b_evaluation >"${LOG_DIR}/b_checkpoints.log" 2>&1 &
  else
    run_b_evaluation >"${LOG_DIR}/b_checkpoints.log" 2>&1 &
  fi
  pids+=("$!"); labels+=("B-checkpoints")
fi
if [[ "${RUN_A}" == "1" ]]; then
  if [[ "${DEVICE_MODE}" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_A}" run_a_training_and_evaluation >"${LOG_DIR}/a_conservative.log" 2>&1 &
  else
    run_a_training_and_evaluation >"${LOG_DIR}/a_conservative.log" 2>&1 &
  fi
  pids+=("$!"); labels+=("A-conservative")
fi
if [[ "${RUN_C}" == "1" ]]; then
  if [[ "${DEVICE_MODE}" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_C}" run_c_training_and_evaluation >"${LOG_DIR}/c_hole_distance.log" 2>&1 &
  else
    run_c_training_and_evaluation >"${LOG_DIR}/c_hole_distance.log" 2>&1 &
  fi
  pids+=("$!"); labels+=("C-hole-distance")
fi

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    echo "[next-round] ${labels[index]} failed; inspect ${LOG_DIR}/${labels[index]}.log" >&2
    status=1
  else
    echo "[next-round] ${labels[index]} finished"
  fi
done

comparison_dir="${EXPERIMENT_ROOT}/comparison"
mkdir -p "${comparison_dir}"
summary_args=(
  --output_dir "${comparison_dir}"
  --no-include_defaults
  --run "CompletionFormer:${ROOT_DIR}/output/initial_model_eval_20260827/completionformer_test100/summary.json:PBRT supervised"
  --run "DMD3C:${ROOT_DIR}/output/pbrt100_depth_completion/dmd3c/summary_20260827.json:official checkpoint"
  --run "OMNI-DC:${ROOT_DIR}/output/pbrt100_depth_completion/omnidc_zero_shot/summary_20260827.json:official zero-shot"
  --run "LDCM:${ROOT_DIR}/output/pbrt100_depth_completion/ldcm_zero_shot/summary.json:official zero-shot"
  --run "LingBot-Depth:${ROOT_DIR}/output/pbrt100_depth_completion/lingbot_dc_zero_shot/summary.json:official zero-shot"
  --run "DEPTHOR:${ROOT_DIR}/output/pbrt100_depth_completion/depthor/summary.json:PBRT trained"
  --run "RGBD-Imaging:${ROOT_DIR}/output/pbrt100_depth_completion/rgbd_lfrd2/rgbd_initial_summary.json:PBRT trained"
  --run "LFRD2:${ROOT_DIR}/output/initial_model_eval_20260827/lfrd2_best.json:PBRT trained"
)

add_candidate() {
  local name="$1" path="$2" note="$3"
  if [[ -f "${path}" ]]; then
    summary_args+=(--selected "${name}:${path}:${note}")
  fi
}

add_candidate "B-best" "${EXPERIMENT_ROOT}/b_checkpoints/best/summary.json" "Exp B best checkpoint"
add_candidate "B-best-hole" "${EXPERIMENT_ROOT}/b_checkpoints/best_hole/summary.json" "Exp B hole-selected checkpoint"
add_candidate "B-best-global" "${EXPERIMENT_ROOT}/b_checkpoints/best_global/summary.json" "Exp B global-selected checkpoint"
add_candidate "B-best-preserve-observed" "${EXPERIMENT_ROOT}/b_checkpoints/best_preserve_observed/summary.json" "Exp B with observed pixels preserved"
add_candidate "A-conservative" "${EXPERIMENT_ROOT}/a_conservative/eval_test_raw/summary.json" "Exp A; checkpoint selected by validation Hole MAE"
add_candidate "A-conservative-preserve-observed" "${EXPERIMENT_ROOT}/a_conservative/eval_test_preserve_observed/summary.json" "Exp A validation-hole checkpoint with observed pixels preserved"
add_candidate "C-hole-distance" "${EXPERIMENT_ROOT}/c_hole_distance/eval_test_raw/summary.json" "Exp C; checkpoint selected by validation Hole MAE"
add_candidate "C-hole-distance-preserve-observed" "${EXPERIMENT_ROOT}/c_hole_distance/eval_test_preserve_observed/summary.json" "Exp C validation-hole checkpoint with observed pixels preserved"

"${SUMMARY_PYTHON}" -u scripts/analysis/summarize_pbrt100_depth_completion_comparison.py "${summary_args[@]}" \
  >"${LOG_DIR}/comparison.log" 2>&1 || status=1
printf '%s\n' "${EXPERIMENT_ROOT}" >"${EXPERIMENT_ROOT}/RUN_ROOT.txt"
echo "[next-round] comparison: ${comparison_dir}/summary.md"
exit "${status}"
