#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/pre_student/hcy/datasets/pbrt"
OUT="output/rgbd_imaging_pbrt"
PYTHON="${PYTHON:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

"${PYTHON}" train_rgbd_pbrt.py \
  --dataset "${ROOT}" \
  --output "${OUT}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE:-6}" \
  --epochs "${EPOCHS:-200}" \
  --workers "${WORKERS:-4}" \
  --save-every 10

"${PYTHON}" predict_rgbd_pbrt.py \
  --dataset "${ROOT}" \
  --checkpoint "${OUT}/checkpoint_best.pth" \
  --output "${OUT}/inference_test" \
  --device "${DEVICE}"
