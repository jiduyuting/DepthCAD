#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/py310/bin/python}"
export DEPTHOR_ROOT="${DEPTHOR_ROOT:-/data/pre_student/GJ/Depthor}"
export CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq}"
export SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/completionformer_full_pbrt/split.json}"
export GPU="${GPU:-0}"
export DEVICE="${DEVICE:-cuda:0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export EPOCHS="${EPOCHS:-30}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

echo "[depthor] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} GPU=${GPU} DEVICE=${DEVICE}"
echo "[depthor] root=${DEPTHOR_ROOT}"
bash scripts/runs/run_depthor_pbrt_train.sh
