#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts/flow:${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/control/bin/python}"

RAW_DIR="${RAW_DIR:-raw}"
DEPTH_DIR="${DEPTH_DIR:-depth}"
CHECKPOINT="${CHECKPOINT:-output/real_raw9_flow_finetune_after_synth_realhole_e20_lr5e6/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/real_raw9_flow_infer_split_added_anchor_ns}"
AMP_MODE="${AMP_MODE:-iq6}"
RAW9_TRANSFORM="${RAW9_TRANSFORM:-checkpoint}"
DEPTH_UNIT="${DEPTH_UNIT:-auto}"

SAMPLES=("$@")
if [ ${#SAMPLES[@]} -eq 0 ]; then
  SAMPLES=(33 34 41 42)
fi

echo "Python: ${PYTHON_BIN}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "Raw9 transform: ${RAW9_TRANSFORM}"
echo "Depth unit: ${DEPTH_UNIT}"
echo "Samples: ${SAMPLES[*]}"

"${PYTHON_BIN}" scripts/flow/infer_real_raw9_flow.py \
  --raw_dir "${RAW_DIR}" \
  --depth_dir "${DEPTH_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --amplitude_mode "${AMP_MODE}" \
  --raw9_transform "${RAW9_TRANSFORM}" \
  --depth_unit "${DEPTH_UNIT}" \
  --hole_mask_mode amp_speckle_cleaned \
  --clean_dilate 1 \
  --speckle_link_radius 2 \
  --split_added_fill \
  --split_added_mode anchor_ns \
  --split_added_inpaint_radius 5 \
  --samples "${SAMPLES[@]}"
