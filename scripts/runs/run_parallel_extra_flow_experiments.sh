#!/usr/bin/env bash
set -euo pipefail

# Run three independent Stage-1 experiments on GPU 0/1/3 while GPU 2 is used
# by the existing Stage-2 job.  Each experiment is evaluated on the fixed test
# split after training, both with raw predictions and observed-pixel merging.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT_DIR}/output/flow_sota_experiments/main_v1/stage1/e2_endpoint2_t050_boundary/best.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/output/parallel_extra/$(date +%Y%m%d_%H%M%S)}"
GPU_MASK="${GPU_MASK:-0}"
GPU_CAPACITY="${GPU_CAPACITY:-1}"
GPU_DISTANCE="${GPU_DISTANCE:-3}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LARGE_BATCH_SIZE="${LARGE_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EPOCHS_SCRATCH="${EPOCHS_SCRATCH:-140}"
EPOCHS_RESUME="${EPOCHS_RESUME:-220}"
SETSID_BIN="${SETSID_BIN:-setsid}"

command -v "${SETSID_BIN}" >/dev/null 2>&1 || {
  echo "Missing required command: ${SETSID_BIN}" >&2
  exit 2
}

# Each job gets its own process group.  Ctrl-C/TERM then stops the Python
# process and its DataLoader workers instead of orphaning them under PID 1.
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

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run directory already exists: ${RUN_ROOT}" >&2
  echo "Choose a new RUN_ROOT to avoid overwriting an old experiment." >&2
  exit 2
fi

for path in \
  "${CACHE_DIR}" \
  "${LIST_DIR}/train.txt" \
  "${LIST_DIR}/val.txt" \
  "${LIST_DIR}/test.txt" \
  "${BASE_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/mask_match" \
  "${RUN_ROOT}/large_b48" "${RUN_ROOT}/hole_distance_focus"
cp "${BASE_CHECKPOINT}" "${RUN_ROOT}/mask_match/last.pt"

COMMON_ARGS=(
  --cache_dir "${CACHE_DIR}"
  --train_list "${LIST_DIR}/train.txt"
  --val_list "${LIST_DIR}/val.txt"
  --input_mode noisy_iq_amp
  --anchor_mode noisy_ns
  --backbone transformer_bottleneck
  --transformer_layers 2
  --transformer_heads 8
  --transformer_pool 2
  --eval_sampling_mode endpoint
  --mask_augment
  --mask_augment_probability 1.0
  --mask_augment_block_sizes 8 12 16 24
  --mask_augment_hole_ratios 0.15 0.20
  --mask_augment_noise_depth_root /data/pre_student/hcy/pbrt/noise_depth
  --val_mask_augment
  --grad_weight 0.5
  --smooth_weight 0.02
  --endpoint_weight 2
  --t0_sample_probability 0.5
  --selection_metric hole
  --amp
)

echo "Run root: ${RUN_ROOT}"
echo "Stage-2 remains untouched; using GPU ${GPU_MASK}, ${GPU_CAPACITY}, ${GPU_DISTANCE}."

echo "[launch] mask_match on GPU ${GPU_MASK}"
"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_MASK}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/train_depth_flow_restoration.py" \
  "${COMMON_ARGS[@]}" \
  --output_dir "${RUN_ROOT}/mask_match" \
  --device cuda:0 \
  --resume \
  --epochs "${EPOCHS_RESUME}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${WORKERS}" \
  --lr 5e-6 \
  --base_channels 32 \
  --res_blocks 2 \
  --hole_weight 8 \
  --valid_weight 1 \
  --boundary_weight 1.5 \
  >"${RUN_ROOT}/logs/mask_match.train.log" 2>&1 &
PID_MASK=$!
JOB_PIDS+=("${PID_MASK}")

echo "[launch] large_b48 on GPU ${GPU_CAPACITY}"
"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_CAPACITY}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/train_depth_flow_restoration.py" \
  "${COMMON_ARGS[@]}" \
  --output_dir "${RUN_ROOT}/large_b48" \
  --device cuda:0 \
  --epochs "${EPOCHS_SCRATCH}" \
  --batch_size "${LARGE_BATCH_SIZE}" \
  --num_workers "${WORKERS}" \
  --lr 1e-4 \
  --base_channels 48 \
  --res_blocks 2 \
  --hole_weight 6 \
  --valid_weight 1 \
  --boundary_weight 1 \
  >"${RUN_ROOT}/logs/large_b48.train.log" 2>&1 &
PID_CAPACITY=$!
JOB_PIDS+=("${PID_CAPACITY}")

echo "[launch] hole_distance_focus on GPU ${GPU_DISTANCE}"
"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_DISTANCE}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/train_depth_flow_restoration.py" \
  "${COMMON_ARGS[@]}" \
  --output_dir "${RUN_ROOT}/hole_distance_focus" \
  --device cuda:0 \
  --epochs "${EPOCHS_SCRATCH}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${WORKERS}" \
  --lr 1e-4 \
  --base_channels 32 \
  --res_blocks 2 \
  --include_hole_distance \
  --hole_weight 10 \
  --valid_weight 0.5 \
  --boundary_weight 2 \
  --hard_sampling \
  --hard_sampling_gamma 2 \
  --hard_loss_gamma 2 \
  >"${RUN_ROOT}/logs/hole_distance_focus.train.log" 2>&1 &
PID_DISTANCE=$!
JOB_PIDS+=("${PID_DISTANCE}")

status=0
for item in \
  "mask_match:${PID_MASK}" \
  "large_b48:${PID_CAPACITY}" \
  "hole_distance_focus:${PID_DISTANCE}"; do
  name="${item%%:*}"
  pid="${item##*:}"
  if wait "${pid}"; then
    echo "[done] ${name} training"
  else
    echo "[failed] ${name} training; see ${RUN_ROOT}/logs/${name}.train.log" >&2
    status=1
  fi
done

echo "[eval] Starting fixed-test evaluation for completed jobs."

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
      "${ROOT_DIR}/scripts/eval_depth_flow_restoration.py" \
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

run_eval mask_match "${GPU_MASK}" & PID_EVAL_MASK=$!
run_eval large_b48 "${GPU_CAPACITY}" & PID_EVAL_CAPACITY=$!
run_eval hole_distance_focus "${GPU_DISTANCE}" & PID_EVAL_DISTANCE=$!
JOB_PIDS+=("${PID_EVAL_MASK}" "${PID_EVAL_CAPACITY}" "${PID_EVAL_DISTANCE}")

for item in \
  "mask_match:${PID_EVAL_MASK}" \
  "large_b48:${PID_EVAL_CAPACITY}" \
  "hole_distance_focus:${PID_EVAL_DISTANCE}"; do
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
for name in ("mask_match", "large_b48", "hole_distance_focus"):
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
