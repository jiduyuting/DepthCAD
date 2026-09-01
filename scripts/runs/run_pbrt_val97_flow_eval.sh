#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}
CHECKPOINT=${CHECKPOINT:-output/pbrt_real_threshold_amp_depth_finetune_replay_e30_p8_keepamp_rw030/best.pt}
PBRT_SPLIT_JSON=${PBRT_SPLIT_JSON:-output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/split.json}
SAMPLE_LIST=${SAMPLE_LIST:-/tmp/depthcad_pbrt_val_97.txt}
OUTPUT_DIR=${OUTPUT_DIR:-"$(dirname "${CHECKPOINT}")/eval_pbrt_val97_endpoint"}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}

"${PYTHON_BIN}" -c "import json; d=json.load(open('${PBRT_SPLIT_JSON}')); open('${SAMPLE_LIST}', 'w').write('\\n'.join(d['val']) + '\\n'); print(len(d['val']))"

"${PYTHON_BIN}" scripts/eval_depth_flow_restoration.py \
  --checkpoint "${CHECKPOINT}" \
  --sample_list "${SAMPLE_LIST}" \
  --split all \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --sampling_mode endpoint
