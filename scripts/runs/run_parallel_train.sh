#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

FLOW_GPU="${FLOW_GPU:-1}"
COMPLETIONFORMER_GPU="${COMPLETIONFORMER_GPU:-2}"
DEPTHOR_GPU="${DEPTHOR_GPU:-3}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output/parallel_train_logs}"
mkdir -p "${LOG_DIR}"

/home/lab507/anaconda3/envs/depthcad/bin/python scripts/build_available_iq_manifest.py

echo "Starting parallel training: Flow=${FLOW_GPU}, CompletionFormer=${COMPLETIONFORMER_GPU}, DEPTHOR=${DEPTHOR_GPU}"

CUDA_VISIBLE_DEVICES="${FLOW_GPU}" GPU=0 DEVICE=cuda:0 \
  bash train_flow.sh >"${LOG_DIR}/flow.log" 2>&1 &
flow_pid=$!

CUDA_VISIBLE_DEVICES="${COMPLETIONFORMER_GPU}" GPU=0 \
  bash train_completionformer.sh >"${LOG_DIR}/completionformer.log" 2>&1 &
completionformer_pid=$!

CUDA_VISIBLE_DEVICES="${DEPTHOR_GPU}" GPU=0 DEVICE=cuda:0 \
  bash train_depthor.sh >"${LOG_DIR}/depthor.log" 2>&1 &
depthor_pid=$!

echo "PIDs: flow=${flow_pid}, completionformer=${completionformer_pid}, depthor=${depthor_pid}"
echo "Logs: ${LOG_DIR}/*.log"

status=0
wait "${flow_pid}" || { echo "Flow failed; see ${LOG_DIR}/flow.log" >&2; status=1; }
wait "${completionformer_pid}" || { echo "CompletionFormer failed; see ${LOG_DIR}/completionformer.log" >&2; status=1; }
wait "${depthor_pid}" || { echo "DEPTHOR failed; see ${LOG_DIR}/depthor.log" >&2; status=1; }

exit "${status}"
