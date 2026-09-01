#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
set -euo pipefail

ROOT="/data/pre_student/hcy/datasets/pbrt"
OUT="output/rgbd_imaging_pbrt"
PYTHON="${PYTHON:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON}" scripts/train_rgbd_pbrt.py \
  --dataset "${ROOT}" \
  --output "${OUT}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE:-6}" \
  --epochs "${EPOCHS:-200}" \
  --workers "${WORKERS:-4}" \
  --save-every 10

"${PYTHON}" scripts/predict_rgbd_pbrt.py \
  --dataset "${ROOT}" \
  --checkpoint "${OUT}/checkpoint_best.pth" \
  --output "${OUT}/inference_test" \
  --device "${DEVICE}"
