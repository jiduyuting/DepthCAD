#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts/flow:${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
LIST_DIR="${LIST_DIR:-${ROOT_DIR}/output/full_pbrt_flow_lists_iq}"
ANCHOR_CACHE_DIR="${ANCHOR_CACHE_DIR:-${ROOT_DIR}/output/flow_anchor_cache_epoch108}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine}"
FLOW_CHECKPOINT="${FLOW_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}"

if pgrep -af '[t]rain_depth_flow_propagation_refine.py' >/dev/null; then
  echo "A propagation-refine process is already running; refusing to start a second one." >&2
  exit 2
fi

EXPECTED_ANCHORS="$(${PYTHON_BIN} -c 'from pathlib import Path; import sys; print(sum(1 for name in sys.argv[1:] for line in Path(name).read_text().splitlines() if line.strip()))' \
  "${LIST_DIR}/train.txt" "${LIST_DIR}/val.txt" "${LIST_DIR}/test.txt")"
CACHED_ANCHORS="$(find "${ANCHOR_CACHE_DIR}" -maxdepth 1 -type f -name '*.npy' 2>/dev/null | wc -l)"
if [[ "${CACHED_ANCHORS}" -ge "${EXPECTED_ANCHORS}" ]]; then
  echo "[exp-b-cached] Reusing ${CACHED_ANCHORS}/${EXPECTED_ANCHORS} cached Flow anchors"
else
  echo "[exp-b-cached] Precomputing frozen Flow anchors (${CACHED_ANCHORS}/${EXPECTED_ANCHORS} ready)"
  "${PYTHON_BIN}" -u scripts/flow/cache_flow_anchors.py \
    --cache_dir "${CACHE_DIR}" \
    --pretrained_checkpoint "${FLOW_CHECKPOINT}" \
    --output_dir "${ANCHOR_CACHE_DIR}" \
    --train_list "${LIST_DIR}/train.txt" \
    --val_list "${LIST_DIR}/val.txt" \
    --test_list "${LIST_DIR}/test.txt" \
    --batch_size "${ANCHOR_BATCH_SIZE:-4}" \
    --num_workers "${ANCHOR_WORKERS:-0}" \
    --device cuda:0
fi

echo "[exp-b-cached] Resuming propagation-refine with cached anchors"
DEVICE=cuda:0 \
EPOCHS="${EPOCHS:-40}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
LR="${LR:-2e-5}" \
SELECTION_METRIC="${SELECTION_METRIC:-composite}" \
AMP="${AMP:-1}" \
EVAL_AFTER="${EVAL_AFTER:-1}" \
VISUALIZE="${VISUALIZE:-1}" \
PRETRAINED_CHECKPOINT="${FLOW_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
ANCHOR_CACHE_DIR="${ANCHOR_CACHE_DIR}" \
  bash scripts/runs/flow/run_depth_flow_propagation_refine_full_pbrt_cached.sh
