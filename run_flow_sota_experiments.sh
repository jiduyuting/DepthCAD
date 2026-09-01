#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
NOISE_DEPTH_ROOT="${NOISE_DEPTH_ROOT:-/data/pre_student/hcy/pbrt/noise_depth}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}"
CURRENT_REFINE_CHECKPOINT="${CURRENT_REFINE_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine/best.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/output/flow_sota_experiments/$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-auto}"
MODE="${MODE:-full}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-140}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-40}"

mkdir -p "${RUN_ROOT}/stage1" "${RUN_ROOT}/stage2" "${RUN_ROOT}/logs" "${RUN_ROOT}/test_eval"

for path in "${CACHE_DIR}" "${LIST_DIR}/train.txt" "${LIST_DIR}/val.txt" "${LIST_DIR}/test.txt" "${NOISE_DEPTH_ROOT}" "${BASE_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done

if [[ "${GPUS}" == "cpu" ]]; then
  GPU_IDS=("cpu")
  TRAIN_DEVICE="cpu"
  AMP_FLAG=""
elif [[ "${GPUS}" == "auto" ]]; then
  GPU_COUNT="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')"
  if (( GPU_COUNT == 0 )); then
    echo "No CUDA GPU is visible. Set GPUS=0,1,... on a GPU node, or use MODE=smoke GPUS=cpu." >&2
    exit 2
  fi
  GPU_IDS=()
  for ((index=0; index<GPU_COUNT; index++)); do GPU_IDS+=("${index}"); done
  TRAIN_DEVICE="cuda:0"
  AMP_FLAG="--amp"
else
  IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
  TRAIN_DEVICE="cuda:0"
  AMP_FLAG="--amp"
fi

if [[ "${TRAIN_DEVICE}" == "cpu" && "${MODE}" != "smoke" && "${ALLOW_CPU:-0}" != "1" ]]; then
  echo "Full training on CPU is disabled. Use MODE=smoke or set ALLOW_CPU=1 explicitly." >&2
  exit 2
fi

TRAIN_LIST="${LIST_DIR}/train.txt"
VAL_LIST="${LIST_DIR}/val.txt"
TEST_LIST="${LIST_DIR}/test.txt"
if [[ "${MODE}" == "smoke" ]]; then
  TRAIN_LIST="${RUN_ROOT}/smoke_train.txt"
  VAL_LIST="${RUN_ROOT}/smoke_val.txt"
  TEST_LIST="${RUN_ROOT}/smoke_test.txt"
  head -n "${SMOKE_TRAIN_SAMPLES:-8}" "${LIST_DIR}/train.txt" > "${TRAIN_LIST}"
  head -n "${SMOKE_VAL_SAMPLES:-4}" "${LIST_DIR}/val.txt" > "${VAL_LIST}"
  head -n "${SMOKE_TEST_SAMPLES:-4}" "${LIST_DIR}/test.txt" > "${TEST_LIST}"
  STAGE1_EPOCHS=109
  STAGE2_EPOCHS=1
  BATCH_SIZE=1
  WORKERS=0
fi

${PYTHON_BIN} scripts/audit_flow_protocol.py \
  --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --test_list "${TEST_LIST}" \
  --max_samples "${AUDIT_MAX_SAMPLES:-1000}" --output "${RUN_ROOT}/protocol_audit.json" \
  >"${RUN_ROOT}/logs/protocol_audit.log"

run_on_gpu() {
  local gpu="$1"
  shift
  if [[ "${gpu}" == "cpu" ]]; then
    "$@"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" "$@"
  fi
}

run_wave() {
  local jobs=("$@")
  local offset=0
  while (( offset < ${#jobs[@]} )); do
    local pids=()
    local labels=()
    for ((slot=0; slot<${#GPU_IDS[@]} && offset<${#jobs[@]}; slot++, offset++)); do
      local spec="${jobs[$offset]}"
      local label="${spec%%::*}"
      local command="${spec#*::}"
      echo "[launch] ${label} on physical GPU ${GPU_IDS[$slot]}"
      run_on_gpu "${GPU_IDS[$slot]}" bash -lc "${command}" >"${RUN_ROOT}/logs/${label}.log" 2>&1 &
      pids+=("$!")
      labels+=("${label}")
    done
    local status=0
    for ((i=0; i<${#pids[@]}; i++)); do
      if ! wait "${pids[$i]}"; then
        echo "[failed] ${labels[$i]}; see ${RUN_ROOT}/logs/${labels[$i]}.log" >&2
        status=1
      else
        echo "[done] ${labels[$i]}"
      fi
    done
    (( status == 0 )) || exit 1
  done
}

prepare_resume() {
  local output="$1"
  mkdir -p "${output}"
  if [[ ! -f "${output}/last.pt" ]]; then cp "${BASE_CHECKPOINT}" "${output}/last.pt"; fi
}

FLOW_CORE="${ROOT_DIR}/train_depth_flow_restoration.py --cache_dir ${CACHE_DIR} --train_list ${TRAIN_LIST} --val_list ${VAL_LIST} --device ${TRAIN_DEVICE} --batch_size ${BATCH_SIZE} --num_workers ${WORKERS} --backbone transformer_bottleneck --input_mode noisy_iq_amp --anchor_mode noisy_ns --eval_sampling_mode endpoint --selection_metric composite --hole_weight 5 --valid_weight 1 --grad_weight 0.5 --smooth_weight 0.02 --endpoint_weight 2 --val_mask_augment --mask_augment_block_sizes 4 8 12 --mask_augment_hole_ratios 0.15 0.20 --mask_augment_noise_depth_root ${NOISE_DEPTH_ROOT} ${AMP_FLAG}"
FLOW_AUG="${FLOW_CORE} --mask_augment --mask_augment_probability 0.50"

E0="${RUN_ROOT}/stage1/e0_endpoint2_fixed_mask"
E1="${RUN_ROOT}/stage1/e1_endpoint2_t025"
E2="${RUN_ROOT}/stage1/e2_endpoint2_t050_boundary"
E3="${RUN_ROOT}/stage1/e3_endpoint4_t025"
prepare_resume "${E0}"
prepare_resume "${E1}"
prepare_resume "${E2}"
prepare_resume "${E3}"

run_wave \
  "e0::${PYTHON_BIN} -u ${FLOW_CORE} --output_dir ${E0} --epochs ${STAGE1_EPOCHS} --lr 1e-5 --t0_sample_probability 0.25 --boundary_weight 0.5 --resume" \
  "e1::${PYTHON_BIN} -u ${FLOW_AUG} --output_dir ${E1} --epochs ${STAGE1_EPOCHS} --lr 1e-5 --t0_sample_probability 0.25 --boundary_weight 0.5 --resume" \
  "e2::${PYTHON_BIN} -u ${FLOW_AUG} --output_dir ${E2} --epochs ${STAGE1_EPOCHS} --lr 1e-5 --t0_sample_probability 0.50 --boundary_weight 1.0 --hard_sampling --hard_loss_gamma 1.0 --resume" \
  "e3::${PYTHON_BIN} -u ${FLOW_AUG} --output_dir ${E3} --epochs ${STAGE1_EPOCHS} --lr 1e-5 --endpoint_weight 4 --t0_sample_probability 0.25 --boundary_weight 0.5 --resume"

SELECT_JSON="${RUN_ROOT}/selected_flow.json"
SELECTED_FLOW="$(${PYTHON_BIN} scripts/manage_flow_sota_experiments.py select --root "${RUN_ROOT}" --output "${SELECT_JSON}")"
echo "[selected flow] ${SELECTED_FLOW}"

ANCHOR_CACHE="${RUN_ROOT}/anchor_cache"
run_on_gpu "${GPU_IDS[0]}" "${PYTHON_BIN}" -u cache_flow_anchors.py \
  --cache_dir "${CACHE_DIR}" --pretrained_checkpoint "${SELECTED_FLOW}" --output_dir "${ANCHOR_CACHE}" \
  --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --test_list "${TEST_LIST}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" --device "${TRAIN_DEVICE}" ${AMP_FLAG} \
  >"${RUN_ROOT}/logs/cache_anchors.log" 2>&1

REFINE_COMMON="${ROOT_DIR}/train_depth_flow_propagation_refine.py --cache_dir ${CACHE_DIR} --pretrained_checkpoint ${SELECTED_FLOW} --anchor_cache_dir ${ANCHOR_CACHE} --train_list ${TRAIN_LIST} --val_list ${VAL_LIST} --device ${TRAIN_DEVICE} --epochs ${STAGE2_EPOCHS} --batch_size ${BATCH_SIZE} --num_workers ${WORKERS} --lr 2e-5 --selection_metric composite --mask_weight 5 --valid_weight 1 --coarse_weight 0.5 --boundary_grad_weight 0.75 --boundary_l1_weight 0.2 ${AMP_FLAG}"
R1="${RUN_ROOT}/stage2/r1_local_refine"
R2="${RUN_ROOT}/stage2/r2_global_refine"
R3="${RUN_ROOT}/stage2/r3_global_refine_strong"
R1_RESUME=""; [[ -f "${R1}/last.pt" ]] && R1_RESUME="--resume"
R2_RESUME=""; [[ -f "${R2}/last.pt" ]] && R2_RESUME="--resume"
R3_RESUME=""; [[ -f "${R3}/last.pt" ]] && R3_RESUME="--resume"
run_wave \
  "r1::${PYTHON_BIN} -u ${REFINE_COMMON} --output_dir ${R1} ${R1_RESUME}" \
  "r2::${PYTHON_BIN} -u ${REFINE_COMMON} --output_dir ${R2} --global_refine --anchor_weight 0.05 ${R2_RESUME}" \
  "r3::${PYTHON_BIN} -u ${REFINE_COMMON} --output_dir ${R3} --global_refine --anchor_weight 0.05 --res_blocks 2 --propagation_steps 8 --lr 1e-5 ${R3_RESUME}"

EVAL_JOBS=()
if [[ -f "${CURRENT_REFINE_CHECKPOINT}" ]]; then
  EVAL_JOBS+=("eval_current_refine::${PYTHON_BIN} -u ${ROOT_DIR}/eval_depth_flow_propagation_refine.py --checkpoint ${CURRENT_REFINE_CHECKPOINT} --sample_list ${TEST_LIST} --split all --output_dir ${RUN_ROOT}/test_eval/current_refine_baseline --batch_size ${BATCH_SIZE} --num_workers ${WORKERS} --device ${TRAIN_DEVICE}")
fi
for experiment in "${E0}" "${E1}" "${E2}" "${E3}"; do
  name="$(basename "${experiment}")"
  EVAL_JOBS+=("eval_${name}::${PYTHON_BIN} -u ${ROOT_DIR}/eval_depth_flow_restoration.py --checkpoint ${experiment}/best.pt --sample_list ${TEST_LIST} --split all --output_dir ${RUN_ROOT}/test_eval/${name} --batch_size ${BATCH_SIZE} --num_workers ${WORKERS} --device ${TRAIN_DEVICE} --sampling_mode endpoint")
done
for experiment in "${R1}" "${R2}" "${R3}"; do
  name="$(basename "${experiment}")"
  EVAL_JOBS+=("eval_${name}::${PYTHON_BIN} -u ${ROOT_DIR}/eval_depth_flow_propagation_refine.py --checkpoint ${experiment}/best.pt --sample_list ${TEST_LIST} --split all --output_dir ${RUN_ROOT}/test_eval/${name} --batch_size ${BATCH_SIZE} --num_workers ${WORKERS} --device ${TRAIN_DEVICE} --anchor_cache_dir ${ANCHOR_CACHE}")
done
run_wave "${EVAL_JOBS[@]}"

${PYTHON_BIN} scripts/manage_flow_sota_experiments.py summarize \
  --root "${RUN_ROOT}" --output "${RUN_ROOT}/summary.json"
echo "Results: ${RUN_ROOT}/summary.md"
