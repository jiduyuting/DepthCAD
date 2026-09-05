#!/usr/bin/env bash
set -euo pipefail

# Short, warm-started follow-up experiments for the best large Flow model.
# GPU 2 is intentionally untouched because it is used by another job.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts/flow:${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT_DIR}/output/parallel_extra/20260831_221704/large_b48/best_hole.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/output/parallel_next/$(date +%Y%m%d_%H%M%S)}"

GPU_CACHED="${GPU_CACHED:-0}"
GPU_MIX="${GPU_MIX:-1}"
GPU_IQ="${GPU_IQ:-3}"
WORKERS="${WORKERS:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-24}"
IQ_EPOCHS="${IQ_EPOCHS:-24}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SETSID_BIN="${SETSID_BIN:-setsid}"

command -v "${SETSID_BIN}" >/dev/null 2>&1 || {
  echo "Missing required command: ${SETSID_BIN}" >&2
  exit 2
}

for path in \
  "${CACHE_DIR}" \
  "${LIST_DIR}/train.txt" \
  "${LIST_DIR}/val.txt" \
  "${LIST_DIR}/test.txt" \
  "${BASE_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run directory already exists: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/cached_only" \
  "${RUN_ROOT}/mix05" "${RUN_ROOT}/iq_only"

# Warm-start a branch while resetting the epoch/selection counters. This makes
# best_hole.pt belong to the new validation protocol instead of inheriting the
# mixed stress-validation score from the previous run.
prepare_warm_start() {
  local out_dir="$1"
  "${PYTHON_BIN}" - "${BASE_CHECKPOINT}" "${out_dir}/last.pt" <<'PY'
import sys
import torch

source, target = sys.argv[1:]
state = torch.load(source, map_location="cpu")
state["epoch"] = 0
state["metrics"] = {}
torch.save(state, target)
PY
}

prepare_warm_start "${RUN_ROOT}/cached_only"
prepare_warm_start "${RUN_ROOT}/mix05"

COMMON_ARGS=(
  --cache_dir "${CACHE_DIR}"
  --train_list "${LIST_DIR}/train.txt"
  --val_list "${LIST_DIR}/val.txt"
  --anchor_mode noisy_ns
  --backbone transformer_bottleneck
  --transformer_layers 2
  --transformer_heads 8
  --transformer_pool 2
  --eval_sampling_mode endpoint
  --grad_weight 0.5
  --smooth_weight 0.02
  --endpoint_weight 2
  --t0_sample_probability 0.5
  --selection_metric global
  --amp
)

COMMON_IQ_ARGS=(
  "${COMMON_ARGS[@]}"
  --input_mode noisy_iq
)

JOB_PIDS=()
cleanup_jobs() {
  local pid
  for pid in "${JOB_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
handle_interrupt() {
  echo "[interrupt] stopping child process groups..." >&2
  cleanup_jobs
  exit 130
}
trap handle_interrupt INT TERM

echo "Run root: ${RUN_ROOT}"
echo "GPU assignment: cached_only=${GPU_CACHED}, mix05=${GPU_MIX}, iq_only=${GPU_IQ}; GPU2 untouched."
echo "Warm-start checkpoint: ${BASE_CHECKPOINT}"

echo "[launch] cached_only: match fixed test-mask distribution"
"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_CACHED}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/flow/train_depth_flow_restoration.py" \
  "${COMMON_ARGS[@]}" \
  --input_mode noisy_iq_amp \
  --output_dir "${RUN_ROOT}/cached_only" \
  --device cuda:0 \
  --resume \
  --epochs "${FINETUNE_EPOCHS}" \
  --batch_size 2 \
  --num_workers "${WORKERS}" \
  --lr 2e-5 \
  --base_channels 48 \
  --res_blocks 2 \
  --hole_weight 6 \
  --valid_weight 1 \
  --boundary_weight 1 \
  >"${RUN_ROOT}/logs/cached_only.train.log" 2>&1 &
PID_CACHED=$!
JOB_PIDS+=("${PID_CACHED}")

echo "[launch] mix05: cached + moderate raw-IQ mask augmentation"
"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_MIX}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/flow/train_depth_flow_restoration.py" \
  "${COMMON_ARGS[@]}" \
  --input_mode noisy_iq_amp \
  --output_dir "${RUN_ROOT}/mix05" \
  --device cuda:0 \
  --resume \
  --epochs "${FINETUNE_EPOCHS}" \
  --batch_size 2 \
  --num_workers "${WORKERS}" \
  --lr 2e-5 \
  --base_channels 48 \
  --res_blocks 2 \
  --mask_augment \
  --mask_augment_probability 0.5 \
  --mask_augment_block_sizes 4 8 12 16 \
  --mask_augment_hole_ratios 0.15 0.20 \
  --mask_augment_noise_depth_root /data/pre_student/hcy/pbrt/noise_depth \
  --hole_weight 6 \
  --valid_weight 1 \
  --boundary_weight 1 \
  >"${RUN_ROOT}/logs/mix05.train.log" 2>&1 &
PID_MIX=$!
JOB_PIDS+=("${PID_MIX}")

echo "[launch] iq_only: ablate amplitude while retaining raw IQ channels"
"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_IQ}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/flow/train_depth_flow_restoration.py" \
  "${COMMON_IQ_ARGS[@]}" \
  --output_dir "${RUN_ROOT}/iq_only" \
  --device cuda:0 \
  --epochs "${IQ_EPOCHS}" \
  --batch_size 2 \
  --num_workers "${WORKERS}" \
  --lr 1e-4 \
  --base_channels 48 \
  --res_blocks 2 \
  --mask_augment \
  --mask_augment_probability 0.5 \
  --mask_augment_block_sizes 4 8 12 16 \
  --mask_augment_hole_ratios 0.15 0.20 \
  --mask_augment_noise_depth_root /data/pre_student/hcy/pbrt/noise_depth \
  --hole_weight 6 \
  --valid_weight 1 \
  --boundary_weight 1 \
  >"${RUN_ROOT}/logs/iq_only.train.log" 2>&1 &
PID_IQ=$!
JOB_PIDS+=("${PID_IQ}")

status=0
for item in "cached_only:${PID_CACHED}" "mix05:${PID_MIX}" "iq_only:${PID_IQ}"; do
  name="${item%%:*}"
  pid="${item##*:}"
  if wait "${pid}"; then
    echo "[done] ${name} training"
  else
    echo "[failed] ${name} training; see ${RUN_ROOT}/logs/${name}.train.log" >&2
    status=1
  fi
done

run_eval() {
  local name="$1"
  local gpu="$2"
  local checkpoint="${RUN_ROOT}/${name}/best_hole.pt"
  [[ -f "${checkpoint}" ]] || {
    echo "[skip] ${name}: missing ${checkpoint}" >&2
    return 0
  }
  for mode in raw preserve_observed; do
    local extra=()
    [[ "${mode}" == "preserve_observed" ]] && extra+=(--preserve_observed)
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u \
      "${ROOT_DIR}/scripts/flow/eval_depth_flow_restoration.py" \
      --checkpoint "${checkpoint}" \
      --cache_dir "${CACHE_DIR}" \
      --sample_list "${LIST_DIR}/test.txt" \
      --split all \
      --output_dir "${RUN_ROOT}/${name}/eval_test_${mode}" \
      --batch_size "${EVAL_BATCH_SIZE}" \
      --num_workers "${WORKERS}" \
      --device cuda:0 \
      --sampling_mode endpoint \
      --visualize \
      --vis_rank best_worst_hole \
      --vis_max_samples 12 \
      "${extra[@]}" \
      >"${RUN_ROOT}/logs/${name}.eval_${mode}.log" 2>&1
  done
}

echo "[eval] Starting fixed-test evaluation."
run_eval cached_only "${GPU_CACHED}" & PID_EVAL_CACHED=$!
run_eval mix05 "${GPU_MIX}" & PID_EVAL_MIX=$!
run_eval iq_only "${GPU_IQ}" & PID_EVAL_IQ=$!
JOB_PIDS+=("${PID_EVAL_CACHED}" "${PID_EVAL_MIX}" "${PID_EVAL_IQ}")

for item in "cached_only:${PID_EVAL_CACHED}" "mix05:${PID_EVAL_MIX}" "iq_only:${PID_EVAL_IQ}"; do
  name="${item%%:*}"
  pid="${item##*:}"
  if wait "${pid}"; then
    echo "[done] ${name} evaluation"
  else
    echo "[failed] ${name} evaluation; inspect ${RUN_ROOT}/logs/${name}.eval_*.log" >&2
    status=1
  fi
done

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for name in ("cached_only", "mix05", "iq_only"):
    for mode in ("raw", "preserve_observed"):
        path = root / name / f"eval_test_{mode}" / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        aggregate = payload.get("aggregate", payload)
        rows.append({
            "experiment": name,
            "mode": mode,
            "hole_mae": aggregate.get("model_hole_mae"),
            "global_mae": aggregate.get("model_global_mae"),
            "valid_mae": aggregate.get("model_valid_mae"),
            "summary": str(path),
        })

(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (root / "summary.tsv").open("w") as f:
    f.write("experiment\tmode\thole_mae\tglobal_mae\tvalid_mae\tsummary\n")
    for row in rows:
        f.write("{experiment}\t{mode}\t{hole_mae}\t{global_mae}\t{valid_mae}\t{summary}\n".format(**row))
print(f"Wrote {root / 'summary.tsv'}")
PY

echo "Results: ${RUN_ROOT}/summary.tsv"
exit "${status}"
