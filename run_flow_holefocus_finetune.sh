#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_OUTPUT="${BASE_OUTPUT:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/depth_flow_full_pbrt_iq_endpoint_w2_holefocus}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
GPU="${GPU:-3}"

mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${OUTPUT_DIR}/last.pt" ]]; then
  [[ -f "${BASE_OUTPUT}/best_hole.pt" ]] || { echo "Missing ${BASE_OUTPUT}/best_hole.pt"; exit 2; }
  cp "${BASE_OUTPUT}/best_hole.pt" "${OUTPUT_DIR}/last.pt"
fi

exec env CUDA_VISIBLE_DEVICES="${GPU}" GPU=0 DEVICE=cuda:0 \
  "${PYTHON_BIN}" -u "${ROOT_DIR}/train_depth_flow_restoration.py" \
  --cache_dir "${ROOT_DIR}/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq" \
  --output_dir "${OUTPUT_DIR}" \
  --train_list "${ROOT_DIR}/output/full_pbrt_flow_lists_iq/train.txt" \
  --val_list "${ROOT_DIR}/output/full_pbrt_flow_lists_iq/val.txt" \
  --device cuda:0 --epochs "${EPOCHS:-150}" --batch_size "${BATCH_SIZE:-4}" \
  --num_workers "${WORKERS:-0}" --lr "${LR:-2e-5}" \
  --backbone transformer_bottleneck --input_mode noisy_iq_amp --anchor_mode noisy_ns \
  --eval_sampling_mode endpoint --resume --selection_metric hole \
  --hard_sampling --hard_sampling_gamma "${HARD_SAMPLING_GAMMA:-2.0}" \
  --hard_loss_gamma "${HARD_LOSS_GAMMA:-2.0}" \
  --hard_loss_area_scale "${HARD_LOSS_AREA_SCALE:-0.10}" \
  --hard_loss_max_weight "${HARD_LOSS_MAX_WEIGHT:-4.0}" \
  --boundary_weight "${BOUNDARY_WEIGHT:-1.0}" \
  --boundary_px "${BOUNDARY_PX:-3.0}"
