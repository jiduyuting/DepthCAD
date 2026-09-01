#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/cformer/bin/python}"
export COMPLETIONFORMER_ROOT="${COMPLETIONFORMER_ROOT:-/data/pre_student/hcy/CompletionFormer}"
export CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
export SOURCE_SPLIT="${SOURCE_SPLIT:-${ROOT_DIR}/output/full_pbrt_manifest_available_iq.json}"
export SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
export GPU="${GPU:-0}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export EPOCHS="${EPOCHS:-72}"
export LR="${LR:-0.001}"
export RUN_NAME="${RUN_NAME:-completionformer_full_pbrt_amp3_depth_seed123}"
export LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output/completionformer_full_pbrt/train_logs/}"

echo "[completionformer] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} GPU=${GPU}"
echo "[completionformer] root=${COMPLETIONFORMER_ROOT}"
bash scripts/runs/run_completionformer_full_pbrt.sh
