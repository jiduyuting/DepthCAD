#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DMD_ROOT="${DMD_ROOT:-/data/pre_student/GJ/DMD3Cpp}"
PYTHON_BIN="${PYTHON_BIN:-/home/lab507/anaconda3/envs/llava/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}"; exit 2; }
[[ -f "${DMD_ROOT}/exts/setup.py" ]] || { echo "Missing DMD3C repo: ${DMD_ROOT}"; exit 2; }
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || { echo "Missing nvcc: ${CUDA_HOME}/bin/nvcc"; exit 2; }

"${PYTHON_BIN}" -m pip install \
  hydra-core==1.3.2 omegaconf==2.3.0 moviepy==1.0.3 pycolmap==0.4.0 \
  evo==1.34.0 h5py==3.15.1

BUILD_DIR="$(mktemp -d /tmp/dmd3c-bpops.XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
cp "${DMD_ROOT}/exts/setup.py" \
   "${DMD_ROOT}/exts/bp_cuda.cpp" \
   "${DMD_ROOT}/exts/bp_cuda.h" \
   "${DMD_ROOT}/exts/bp_cuda_kernel.cu" \
   "${BUILD_DIR}/"

CUDA_HOME="${CUDA_HOME}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=4 \
  "${PYTHON_BIN}" -m pip install --no-build-isolation --force-reinstall "${BUILD_DIR}"

MPLCONFIGDIR=/tmp/matplotlib-dmd3c PYTHONPATH="${DMD_ROOT}:${ROOT_DIR}" \
  "${PYTHON_BIN}" -c \
  'import hydra, omegaconf, BpOps; from depth_anything_3.api import DepthAnything3; print("DMD3C environment: OK")'
