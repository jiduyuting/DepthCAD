#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
set -euo pipefail

RAW9_TRANSFORM=${RAW9_TRANSFORM:-flip_lr}
OUTPUT_DIR=${OUTPUT_DIR:-output/pbrt_real_threshold_amp_depth_finetune_replay_fliplr_e30_p8_keepamp_rw030}

export RAW9_TRANSFORM
export OUTPUT_DIR

exec bash scripts/runs/run_pbrt_real_threshold_amp_depth_finetune_replay.sh
