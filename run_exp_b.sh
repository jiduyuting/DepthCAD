#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "[exp-b] Flow-anchor propagation + refinement"

DEVICE=cuda:0 \
EPOCHS="${EPOCHS:-40}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
LR="${LR:-2e-5}" \
SELECTION_METRIC="${SELECTION_METRIC:-composite}" \
AMP="${AMP:-1}" \
EVAL_AFTER="${EVAL_AFTER:-1}" \
VISUALIZE="${VISUALIZE:-1}" \
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt}" \
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_propagation_refine}" \
  bash run_depth_flow_propagation_refine_full_pbrt.sh
