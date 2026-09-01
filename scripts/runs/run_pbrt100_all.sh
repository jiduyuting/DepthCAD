#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

# The runner intentionally keeps going when an external baseline is unavailable.
# Its final comparison marks that method as missing instead of hiding the reason.
RUN_LOG="${RUN_LOG:-${ROOT_DIR}/output/pbrt100_depth_completion/run.log}"
mkdir -p "$(dirname "${RUN_LOG}")"
: > "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

PYTHON_BASE="${PYTHON_BASE:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
PYTHON_RGBD="${PYTHON_RGBD:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
PYTHON_LFRD2="${PYTHON_LFRD2:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
PYTHON_CFORMER="${PYTHON_CFORMER:-/home/lab507/anaconda3/envs/cformer/bin/python}"
PYTHON_LDCM="${PYTHON_LDCM:-/home/lab507/anaconda3/envs/lingbot-world/bin/python}"
PYTHON_OMNI="${PYTHON_OMNI:-/home/lab507/anaconda3/envs/cformer/bin/python}"
PYTHON_DMD3C="${PYTHON_DMD3C:-/home/lab507/anaconda3/envs/llava/bin/python}"
PYTHON_DEPTHOR="${PYTHON_DEPTHOR:-/home/lab507/anaconda3/envs/py310/bin/python}"
PYTHON_FLOW="${PYTHON_FLOW:-/home/lab507/anaconda3/envs/SVDC/bin/python}"

PBRT100_CACHE="${PBRT100_CACHE:-${ROOT_DIR}/depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq}"
PBRT100_SPLIT="${PBRT100_SPLIT:-${ROOT_DIR}/output/pbrt100_depth_completion/split.json}"
UNIFIED_MANIFEST="${UNIFIED_MANIFEST:-${ROOT_DIR}/output/full_pbrt_manifest_available_iq.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/output/pbrt100_depth_completion}"

RUN_UNIFIED_TRAIN="${RUN_UNIFIED_TRAIN:-1}"
RUN_FLOW="${RUN_FLOW:-1}"
TRAIN_FLOW="${TRAIN_FLOW:-1}"
RUN_COMPLETIONFORMER="${RUN_COMPLETIONFORMER:-1}"
RUN_DMD3C="${RUN_DMD3C:-1}"
RUN_OMNI="${RUN_OMNI:-1}"
RUN_LDCM="${RUN_LDCM:-1}"
RUN_LINGBOT="${RUN_LINGBOT:-1}"
RUN_DEPTHOR="${RUN_DEPTHOR:-1}"
TRAIN_COMPLETIONFORMER="${TRAIN_COMPLETIONFORMER:-1}"
TRAIN_DEPTHOR="${TRAIN_DEPTHOR:-1}"
RESUME="${RESUME:-1}"
DEVICE="${DEVICE:-auto}"

if [[ "${DEVICE}" == "auto" ]]; then
  if "${PYTHON_BASE}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

status=0
run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  if "$@"; then
    echo "[ok] ${name}"
  else
    echo "[failed] ${name}; continuing" >&2
    status=1
  fi
}

skip_step() {
  echo "[skip] $1"
}

has_file() { [[ -f "$1" ]]; }
has_dir() { [[ -d "$1" ]]; }

echo "PBRT100 runner"
echo "root: ${ROOT_DIR}"
echo "device: ${DEVICE}"
echo "holdout cache: ${PBRT100_CACHE}"
echo "unified manifest: ${UNIFIED_MANIFEST}"

if [[ ! -f "${UNIFIED_MANIFEST}" || "${REBUILD_MANIFEST:-1}" == "1" ]]; then
  run_manifest_python="${PYTHON_BASE}"
  "${run_manifest_python}" scripts/build_available_iq_manifest.py --output "${UNIFIED_MANIFEST}"
fi

if ! has_dir "${PBRT100_CACHE}"; then
  echo "Missing PBRT100 cache: ${PBRT100_CACHE}" >&2
  exit 2
fi

run_step "Create PBRT100 split" \
  "${PYTHON_BASE}" scripts/make_pbrt100_completion_split.py \
    --cache_root "${PBRT100_CACHE}" \
    --manifest "${ROOT_DIR}/__missing_pbrt100_manifest.json" \
    --output "${PBRT100_SPLIT}" \
    --expected_count 100

if ! has_file "${UNIFIED_MANIFEST}"; then
  echo "Missing unified manifest: ${UNIFIED_MANIFEST}" >&2
  echo "Create it with scripts/build_unified_pbrt_manifest.py before training RGBD/LFRD2." >&2
  RUN_UNIFIED_TRAIN=0
fi

if [[ "${RUN_UNIFIED_TRAIN}" == "1" ]]; then
  unified_args=(
    MANIFEST="${UNIFIED_MANIFEST}"
    DEVICE="${DEVICE}"
    RESUME="${RESUME}"
    RUN_EVAL=0
    EPOCHS="${EPOCHS:-200}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    WORKERS="${WORKERS:-0}"
  )
  run_step "Train RGBD-Imaging and LFRD2" env "${unified_args[@]}" bash scripts/runs/run_train_rgbd_lfrd2.sh
else
  skip_step "Unified RGBD/LFRD2 training disabled"
fi

RGBD_CKPT="${RGBD_CKPT:-${ROOT_DIR}/output/rgbd_imaging_full_pbrt/checkpoint_best.pth}"
LFRD2_CKPT="${LFRD2_CKPT:-${ROOT_DIR}/output/lfrd2_full_pbrt/checkpoint_best_net.pth}"
if [[ -f "${RGBD_CKPT}" && -f "${LFRD2_CKPT}" ]]; then
  run_step "Evaluate RGBD-Imaging and LFRD2" \
    "${PYTHON_LFRD2}" scripts/eval_unified_baselines.py \
      --manifest "${UNIFIED_MANIFEST}" --model both --device "${DEVICE}" \
      --workers "${WORKERS:-0}" --batch_size "${EVAL_BATCH_SIZE:-4}" \
      --rgbd_checkpoint "${RGBD_CKPT}" --lfrd2_checkpoint "${LFRD2_CKPT}" \
      --output "${OUTPUT_ROOT}/rgbd_lfrd2/summary.json"
else
  skip_step "RGBD/LFRD2 evaluation (checkpoint missing)"
fi

FLOW_CKPT="${FLOW_CKPT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}"
FLOW_LIST_DIR="${FLOW_LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
FLOW_OUTPUT="${FLOW_OUTPUT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2}"
if [[ "${TRAIN_FLOW}" == "1" && -f "${UNIFIED_MANIFEST}" ]]; then
  run_step "Prepare Ours-Flow lists" \
    "${PYTHON_BASE}" scripts/flow/make_full_pbrt_flow_lists.py \
      --manifest "${UNIFIED_MANIFEST}" --output_dir "${FLOW_LIST_DIR}"
  flow_resume=()
  if [[ "${RESUME}" == "1" && -f "${FLOW_OUTPUT}/last.pt" ]]; then
    flow_resume+=(--resume)
  fi
  run_step "Train Ours-Flow-FullPBRT" \
    "${PYTHON_FLOW}" -u scripts/flow/train_depth_flow_restoration.py \
      --cache_dir "${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq" \
      --output_dir "${FLOW_OUTPUT}" \
      --train_list "${FLOW_LIST_DIR}/train.txt" \
      --val_list "${FLOW_LIST_DIR}/val.txt" \
      --device "${DEVICE}" \
      --epochs "${FLOW_EPOCHS:-120}" \
      --batch_size "${FLOW_BATCH_SIZE:-4}" \
      --num_workers "${FLOW_WORKERS:-0}" \
      --lr "${FLOW_LR:-1e-4}" \
      --backbone "${FLOW_BACKBONE:-transformer_bottleneck}" \
      --input_mode "${FLOW_INPUT_MODE:-noisy_iq_amp}" \
      --anchor_mode "${FLOW_ANCHOR_MODE:-noisy_ns}" \
      --eval_sampling_mode "${FLOW_SAMPLING_MODE:-endpoint}" \
      "${flow_resume[@]}"
  FLOW_CKPT="${FLOW_OUTPUT}/best.pt"
fi
if [[ "${RUN_FLOW}" == "1" && -f "${FLOW_CKPT}" ]]; then
  run_step "Evaluate Ours-Flow-FullPBRT" \
    "${PYTHON_FLOW}" scripts/flow/eval_flow_unified_test.py \
      --manifest "${UNIFIED_MANIFEST}" --checkpoint "${FLOW_CKPT}" \
      --device "${DEVICE}" --workers "${WORKERS:-0}" \
      --output "${OUTPUT_ROOT}/ours_flow/summary.json"
else
  skip_step "Ours-Flow-FullPBRT inference (checkpoint missing or disabled)"
fi

CF_ROOT="${COMPLETIONFORMER_ROOT:-/data/pre_student/hcy/CompletionFormer}"
CF_CKPT="${COMPLETIONFORMER_CHECKPOINT:-}"
if [[ "${TRAIN_COMPLETIONFORMER}" == "1" ]]; then
  if has_dir "${CF_ROOT}"; then
    run_step "Train CompletionFormer on PBRT" \
      env CACHE_ROOT="${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq" \
        SPLIT_JSON="${ROOT_DIR}/output/completionformer_full_pbrt/split.json" \
        COMPLETIONFORMER_ROOT="${CF_ROOT}" PYTHON_BIN="${PYTHON_CFORMER}" \
        bash scripts/runs/run_completionformer_full_pbrt.sh
    if [[ -z "${CF_CKPT}" ]]; then
      CF_CKPT="$(find "${ROOT_DIR}/output/completionformer_full_pbrt" -type f \( -name '*.pth' -o -name '*.pt' \) -print 2>/dev/null | sort | tail -n 1)"
      [[ -n "${CF_CKPT}" ]] && echo "[runner] discovered CompletionFormer checkpoint: ${CF_CKPT}"
    fi
  else
    skip_step "CompletionFormer training (repo missing: ${CF_ROOT})"
  fi
fi
if [[ "${RUN_COMPLETIONFORMER}" == "1" && -n "${CF_CKPT}" && -f "${CF_CKPT}" && -d "${CF_ROOT}" ]]; then
  run_step "Evaluate CompletionFormer" \
    "${PYTHON_CFORMER}" scripts/eval_completionformer_full_pbrt.py \
      --completionformer_root "${CF_ROOT}" --checkpoint "${CF_CKPT}" \
      --cache_root "${PBRT100_CACHE}" --split_json "${PBRT100_SPLIT}" \
      --output_dir "${OUTPUT_ROOT}/completionformer" --device "${DEVICE}" \
      --num_workers "${WORKERS:-0}" --save_predictions
else
  skip_step "CompletionFormer inference (set COMPLETIONFORMER_CHECKPOINT and ensure repo exists)"
fi

if [[ "${RUN_DMD3C}" == "1" && -f "${DMD3C_CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/dmd3c/result_ema.pth}" && -d "${DMD_ROOT:-/data/pre_student/GJ/DMD3Cpp}" ]]; then
  run_step "Evaluate DMD3C" \
    env DMD_ROOT="${DMD_ROOT:-/data/pre_student/GJ/DMD3Cpp}" \
      CHECKPOINT="${DMD3C_CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/dmd3c/result_ema.pth}" \
      CACHE_ROOT="${PBRT100_CACHE}" SPLIT_JSON="${PBRT100_SPLIT}" SPLIT=test \
      SUMMARY_OUTPUT="${OUTPUT_ROOT}/dmd3c/summary.json" \
      UNIFORMAT_DIR="${OUTPUT_ROOT}/uniformat_full_pbrt" \
      PYTHON_BIN="${PYTHON_DMD3C}" bash scripts/runs/run_dmd3c_full_pbrt.sh
else
  skip_step "DMD3C inference (repo or checkpoint missing)"
fi

if [[ "${RUN_OMNI}" == "1" && -d "${OMNI_ROOT:-/data/pre_student/GJ/OMNI-DC}" ]]; then
  run_step "Evaluate OMNI-DC" \
    env OMNI_ROOT="${OMNI_ROOT:-/data/pre_student/GJ/OMNI-DC}" \
      CACHE_ROOT="${PBRT100_CACHE}" SPLIT_JSON="${PBRT100_SPLIT}" SPLIT=test \
      SUMMARY_OUTPUT="${OUTPUT_ROOT}/omnidc_zero_shot/summary.json" \
      UNIFORMAT_DIR="${OUTPUT_ROOT}/uniformat_full_pbrt" LOG_DIR="${OUTPUT_ROOT}/omnidc_runs" \
      PYTHON_BIN="${PYTHON_OMNI}" bash scripts/runs/run_omnidc_full_pbrt.sh
else
  skip_step "OMNI-DC inference (repo missing)"
fi

if [[ "${RUN_LDCM}" == "1" && -d "${LDCM_ROOT:-/data/pre_student/GJ/LDCM}" ]]; then
  run_step "Evaluate LDCM" \
    env LDCM_ROOT="${LDCM_ROOT:-/data/pre_student/GJ/LDCM}" \
      CACHE_ROOT="${PBRT100_CACHE}" SPLIT_JSON="${PBRT100_SPLIT}" SPLIT=test \
      OUTPUT_DIR="${OUTPUT_ROOT}/ldcm_zero_shot" PYTHON_BIN="${PYTHON_LDCM}" \
      bash scripts/runs/run_ldcm_full_pbrt.sh
else
  skip_step "LDCM inference (repo missing)"
fi

if [[ "${RUN_LINGBOT}" == "1" && -d "${LINGBOT_ROOT:-/data/pre_student/GJ/lingbot-depth}" ]]; then
  run_step "Evaluate LingBot-Depth" \
    env LINGBOT_ROOT="${LINGBOT_ROOT:-/data/pre_student/GJ/lingbot-depth}" \
      CACHE_ROOT="${PBRT100_CACHE}" SPLIT_JSON="${PBRT100_SPLIT}" SPLIT=test \
      OUTPUT_DIR="${OUTPUT_ROOT}/lingbot_dc_zero_shot" PYTHON_BIN="${PYTHON_BASE}" \
      bash scripts/runs/run_lingbot_full_pbrt.sh
else
  skip_step "LingBot-Depth inference (repo missing)"
fi

DEPTHOR_ROOT_VALUE="${DEPTHOR_ROOT:-/data/pre_student/GJ/Depthor}"
DEPTHOR_CKPT="${DEPTHOR_CHECKPOINT:-}"
if [[ "${TRAIN_DEPTHOR}" == "1" && -d "${DEPTHOR_ROOT_VALUE}" ]]; then
  run_step "Train DEPTHOR on PBRT" \
    env DEPTHOR_ROOT="${DEPTHOR_ROOT_VALUE}" PYTHON_BIN="${PYTHON_DEPTHOR}" \
      CACHE_ROOT="${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq" \
      SPLIT_JSON="${ROOT_DIR}/output/completionformer_full_pbrt/split.json" \
      bash scripts/runs/run_depthor_pbrt_train.sh
  if [[ -z "${DEPTHOR_CKPT}" && -f "${ROOT_DIR}/output/depth_completion_baselines/depthor_pbrt_train/best_weights.pt" ]]; then
    DEPTHOR_CKPT="${ROOT_DIR}/output/depth_completion_baselines/depthor_pbrt_train/best_weights.pt"
    echo "[runner] discovered DEPTHOR checkpoint: ${DEPTHOR_CKPT}"
  fi
fi
if [[ "${RUN_DEPTHOR}" == "1" && -n "${DEPTHOR_CKPT}" && -f "${DEPTHOR_CKPT}" && -d "${DEPTHOR_ROOT_VALUE}" ]]; then
  run_step "Evaluate DEPTHOR" \
    env DEPTHOR_ROOT="${DEPTHOR_ROOT_VALUE}" CHECKPOINT="${DEPTHOR_CKPT}" \
      CACHE_ROOT="${PBRT100_CACHE}" SPLIT_JSON="${PBRT100_SPLIT}" SPLIT=test \
      OUTPUT_DIR="${OUTPUT_ROOT}/depthor" PYTHON_BIN="${PYTHON_DEPTHOR}" \
      DEVICE="${DEVICE}" bash scripts/runs/run_depthor_full_pbrt.sh
else
  skip_step "DEPTHOR inference (set DEPTHOR_CHECKPOINT and ensure repo/checkpoints exist)"
fi

run_step "Build PBRT100 comparison" \
  "${PYTHON_BASE}" scripts/analysis/summarize_pbrt100_depth_completion_comparison.py \
    --output_dir "${OUTPUT_ROOT}/comparison" --expected_samples 100 \
    --selected "Ours-Flow-FullPBRT:${OUTPUT_ROOT}/ours_flow/summary.json:unified flow evaluation"

echo
echo "Finished. Summary: ${OUTPUT_ROOT}/comparison/summary.md"
echo "Run log: ${RUN_LOG}"
exit "${status}"
