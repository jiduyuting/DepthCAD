#!/usr/bin/env bash
set -euo pipefail

# Parallel high-value follow-up for mix05.
# GPU0/1: warm-start fine-tunes. GPU3: validation-calibrated soft blending.
# GPU2 is intentionally untouched.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts/flow:${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT_DIR}/output/parallel_next/20260902_103459/mix05/best_hole.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/output/parallel_high_gain/$(date +%Y%m%d_%H%M%S)}"

GPU_FINE="${GPU_FINE:-0}"
GPU_VALID="${GPU_VALID:-1}"
GPU_BLEND="${GPU_BLEND:-3}"
WORKERS="${WORKERS:-2}"
FINE_EPOCHS="${FINE_EPOCHS:-24}"
VALID_EPOCHS="${VALID_EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
SETSID_BIN="${SETSID_BIN:-setsid}"

for path in "${CACHE_DIR}" "${LIST_DIR}/train.txt" "${LIST_DIR}/val.txt" \
  "${LIST_DIR}/test.txt" "${BASE_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done
command -v "${SETSID_BIN}" >/dev/null 2>&1 || { echo "Missing ${SETSID_BIN}" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run directory exists: ${RUN_ROOT}" >&2; exit 2; }
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/mix05_long" "${RUN_ROOT}/valid_focus" "${RUN_ROOT}/soft_blend"

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
prepare_warm_start "${RUN_ROOT}/mix05_long"
prepare_warm_start "${RUN_ROOT}/valid_focus"

COMMON=(
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
  --mask_augment_probability 0.5
  --mask_augment_block_sizes 4 8 12 16
  --mask_augment_hole_ratios 0.15 0.20
  --mask_augment_noise_depth_root /data/pre_student/hcy/pbrt/noise_depth
  --grad_weight 0.5
  --smooth_weight 0.02
  --t0_sample_probability 0.5
  --selection_metric global
  --amp
  --base_channels 48
  --res_blocks 2
  --hole_weight 6
  --valid_weight 1
  --boundary_weight 1
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
trap 'cleanup_jobs; exit 130' INT TERM

echo "Run root: ${RUN_ROOT}"
echo "GPU assignment: mix05_long=${GPU_FINE}, valid_focus=${GPU_VALID}, soft_blend=${GPU_BLEND}; GPU2 untouched."

"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_FINE}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/flow/train_depth_flow_restoration.py" \
  "${COMMON[@]}" \
  --output_dir "${RUN_ROOT}/mix05_long" \
  --device cuda:0 --resume --epochs "${FINE_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" --lr 5e-6 \
  --endpoint_weight 2 \
  >"${RUN_ROOT}/logs/mix05_long.train.log" 2>&1 &
PID_FINE=$!
JOB_PIDS+=("${PID_FINE}")

"${SETSID_BIN}" env CUDA_VISIBLE_DEVICES="${GPU_VALID}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/flow/train_depth_flow_restoration.py" \
  "${COMMON[@]}" \
  --output_dir "${RUN_ROOT}/valid_focus" \
  --device cuda:0 --resume --epochs "${VALID_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" --num_workers "${WORKERS}" --lr 1e-5 \
  --valid_weight 2 --endpoint_weight 4 --boundary_weight 1.5 \
  >"${RUN_ROOT}/logs/valid_focus.train.log" 2>&1 &
PID_VALID=$!
JOB_PIDS+=("${PID_VALID}")

echo "[launch] soft blend calibration on GPU ${GPU_BLEND}"
CUDA_VISIBLE_DEVICES="${GPU_BLEND}" "${PYTHON_BIN}" -u \
  "${ROOT_DIR}/scripts/flow/eval_flow_soft_blend.py" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --cache_dir "${CACHE_DIR}" \
  --val_list "${LIST_DIR}/val.txt" \
  --test_list "${LIST_DIR}/test.txt" \
  --output_dir "${RUN_ROOT}/soft_blend" \
  --device cuda:0 --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${WORKERS}" \
  >"${RUN_ROOT}/logs/soft_blend.log" 2>&1 &
PID_BLEND=$!
JOB_PIDS+=("${PID_BLEND}")

status=0
for item in "mix05_long:${PID_FINE}" "valid_focus:${PID_VALID}" "soft_blend:${PID_BLEND}"; do
  name="${item%%:*}"; pid="${item##*:}"
  if wait "${pid}"; then echo "[done] ${name}"; else echo "[failed] ${name}" >&2; status=1; fi
done

run_eval() {
  local name="$1"; local gpu="$2"; local checkpoint="${RUN_ROOT}/${name}/best_global.pt"
  [[ -f "${checkpoint}" ]] || checkpoint="${RUN_ROOT}/${name}/best_hole.pt"
  [[ -f "${checkpoint}" ]] || { echo "[skip] ${name}: no checkpoint" >&2; return 0; }
  for mode in raw preserve_observed; do
    local extra=(); [[ "${mode}" == "preserve_observed" ]] && extra+=(--preserve_observed)
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u \
      "${ROOT_DIR}/scripts/flow/eval_depth_flow_restoration.py" \
      --checkpoint "${checkpoint}" --cache_dir "${CACHE_DIR}" \
      --sample_list "${LIST_DIR}/test.txt" --split all \
      --output_dir "${RUN_ROOT}/${name}/eval_test_${mode}" \
      --batch_size "${EVAL_BATCH_SIZE}" --num_workers "${WORKERS}" \
      --device cuda:0 --sampling_mode endpoint "${extra[@]}" \
      >"${RUN_ROOT}/logs/${name}.eval_${mode}.log" 2>&1
  done
}

run_eval mix05_long "${GPU_FINE}" & P0=$!
run_eval valid_focus "${GPU_VALID}" & P1=$!
wait "$P0" "$P1" || status=1

"${PYTHON_BIN}" - "${RUN_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); rows = []
for name in ("mix05_long", "valid_focus"):
    for mode in ("raw", "preserve_observed"):
        p = root / name / f"eval_test_{mode}" / "summary.json"
        if not p.exists(): continue
        a = json.loads(p.read_text()).get("aggregate", {})
        rows.append({"experiment": name, "mode": mode,
                     "global_mae": a.get("model_global_mae"),
                     "hole_mae": a.get("model_hole_mae"),
                     "valid_mae": a.get("model_valid_mae"), "summary": str(p)})
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (root / "summary.tsv").open("w") as f:
    f.write("experiment\tmode\tglobal_mae\thole_mae\tvalid_mae\tsummary\n")
    for r in rows: f.write("{experiment}\t{mode}\t{global_mae}\t{hole_mae}\t{valid_mae}\t{summary}\n".format(**r))
print(f"Wrote {root / 'summary.tsv'}")
PY

echo "Results: ${RUN_ROOT}/summary.tsv"
exit "${status}"
