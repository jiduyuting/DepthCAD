#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_BIN="${HF_BIN:-/home/lab507/.local/bin/hf}"
WEIGHT_DIR="${WEIGHT_DIR:-${ROOT_DIR}/output/depth_completion_weights/dmd3c}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

[[ -x "${HF_BIN}" ]] || { echo "Missing Hugging Face CLI: ${HF_BIN}"; exit 2; }
mkdir -p "${WEIGHT_DIR}"

"${HF_BIN}" download Liangyingping/DMD3Cpp-checkpoints result_ema.pth \
  --repo-type dataset --local-dir "${WEIGHT_DIR}"

"${HF_BIN}" download depth-anything/DA3METRIC-LARGE

echo "DMD3C checkpoint: ${WEIGHT_DIR}/result_ema.pth"
echo "DA3METRIC-LARGE is stored in the Hugging Face cache."
