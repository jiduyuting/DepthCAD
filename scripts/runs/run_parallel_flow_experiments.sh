#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
GPU_C="${GPU_C:-2}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output}"
mkdir -p "${LOG_DIR}"

echo "Starting parallel Flow experiments: A=${GPU_A}, B=${GPU_B}, C=${GPU_C}"
echo "Logs: ${LOG_DIR}/exp_[abc].log"

CUDA_VISIBLE_DEVICES="${GPU_A}" bash scripts/runs/run_exp_a.sh >"${LOG_DIR}/exp_a.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES="${GPU_B}" bash scripts/runs/run_exp_b.sh >"${LOG_DIR}/exp_b.log" 2>&1 &
pid_b=$!
CUDA_VISIBLE_DEVICES="${GPU_C}" bash scripts/runs/run_exp_c.sh >"${LOG_DIR}/exp_c.log" 2>&1 &
pid_c=$!

echo "PIDs: exp_a=${pid_a}, exp_b=${pid_b}, exp_c=${pid_c}"

status=0
wait "${pid_a}" || { echo "exp_a failed; see ${LOG_DIR}/exp_a.log" >&2; status=1; }
wait "${pid_b}" || { echo "exp_b failed; see ${LOG_DIR}/exp_b.log" >&2; status=1; }
wait "${pid_c}" || { echo "exp_c failed; see ${LOG_DIR}/exp_c.log" >&2; status=1; }

exit "${status}"
