#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

HF_BIN="${HF_BIN:-/home/lab507/.local/bin/hf}"
GDOWN_BIN="${GDOWN_BIN:-/home/lab507/anaconda3/bin/gdown}"
PYTHON_BASE="${PYTHON_BASE:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
PYTHON_DMD3C="${PYTHON_DMD3C:-/home/lab507/anaconda3/envs/llava/bin/python}"
PYTHON_OMNI="${PYTHON_OMNI:-/home/lab507/anaconda3/envs/cformer/bin/python}"
PYTHON_LDCM="${PYTHON_LDCM:-/home/lab507/anaconda3/envs/lingbot-world/bin/python}"
PYTHON_LINGBOT="${PYTHON_LINGBOT:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
PYTHON_DEPTHOR="${PYTHON_DEPTHOR:-/home/lab507/anaconda3/envs/py310/bin/python}"

WEIGHT_ROOT="${WEIGHT_ROOT:-${ROOT_DIR}/output/depth_completion_weights}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/pbrt100_depth_completion/split.json}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq}"
DOWNLOAD="${DOWNLOAD:-1}"
RUN="${RUN:-0}"

RUN_DMD3C="${RUN_DMD3C:-1}"
RUN_OMNI="${RUN_OMNI:-1}"
RUN_LDCM="${RUN_LDCM:-1}"
RUN_LINGBOT="${RUN_LINGBOT:-1}"
RUN_DEPTHOR="${RUN_DEPTHOR:-1}"

mkdir -p "${WEIGHT_ROOT}" "${ROOT_DIR}/output/depth_completion_baselines"

download_weights() {
  [[ -x "${HF_BIN}" ]] || { echo "Missing HF CLI: ${HF_BIN}" >&2; return 1; }
  [[ -x "${GDOWN_BIN}" ]] || { echo "Missing gdown: ${GDOWN_BIN}" >&2; return 1; }

  if [[ "${RUN_DMD3C}" == "1" ]]; then
    mkdir -p "${WEIGHT_ROOT}/dmd3c"
    "${HF_BIN}" download Liangyingping/DMD3Cpp-checkpoints result_ema.pth \
      --repo-type dataset --local-dir "${WEIGHT_ROOT}/dmd3c"
    "${HF_BIN}" download depth-anything/DA3METRIC-LARGE
  fi

  if [[ "${RUN_OMNI}" == "1" ]]; then
    mkdir -p "${WEIGHT_ROOT}/omnidc"
    "${GDOWN_BIN}" 'https://drive.google.com/uc?id=1ssJYFB3rQD5JEYgG7W6tRJg1hpQKvqPD' \
      -O "${WEIGHT_ROOT}/omnidc/modelv1.1_best_72epochs.pt"
    "${HF_BIN}" download depth-anything/Depth-Anything-V2-Large \
      depth_anything_v2_vitl.pth --local-dir "${WEIGHT_ROOT}/omnidc"
  fi

  if [[ "${RUN_LDCM}" == "1" ]]; then
    mkdir -p "${WEIGHT_ROOT}/ldcm" "${WEIGHT_ROOT}/moge-2-vits-normal"
    "${HF_BIN}" download pkqbajng/LDCM ldcm.pt --local-dir "${WEIGHT_ROOT}/ldcm"
    "${HF_BIN}" download Ruicheng/moge-2-vits-normal model.pt \
      --local-dir "${WEIGHT_ROOT}/moge-2-vits-normal"
  fi

  if [[ "${RUN_LINGBOT}" == "1" ]]; then
    mkdir -p "${WEIGHT_ROOT}/lingbot-depth-dc"
    "${HF_BIN}" download robbyant/lingbot-depth-postrain-dc-vitl14 \
      --local-dir "${WEIGHT_ROOT}/lingbot-depth-dc"
  fi

  if [[ "${RUN_DEPTHOR}" == "1" ]]; then
    mkdir -p "${WEIGHT_ROOT}/depthor"
    "${GDOWN_BIN}" 'https://drive.google.com/uc?id=1oZByVUklbjQHlZTdKFQdwkMKkugi4l6-' \
      -O "${WEIGHT_ROOT}/depthor/depthor_zju_large.pt"
    "${HF_BIN}" download depth-anything/Depth-Anything-V2-Small \
      depth_anything_v2_vits.pth --local-dir "${WEIGHT_ROOT}/depthor"
  fi
}

check_files() {
  local status=0
  for path in \
    "${WEIGHT_ROOT}/dmd3c/result_ema.pth" \
    "${WEIGHT_ROOT}/omnidc/modelv1.1_best_72epochs.pt" \
    "${WEIGHT_ROOT}/omnidc/depth_anything_v2_vitl.pth" \
    "${WEIGHT_ROOT}/ldcm/ldcm.pt" \
    "${WEIGHT_ROOT}/moge-2-vits-normal/model.pt" \
    "${WEIGHT_ROOT}/lingbot-depth-dc/model.pt" \
    "${WEIGHT_ROOT}/depthor/depthor_zju_large.pt" \
    "${WEIGHT_ROOT}/depthor/depth_anything_v2_vits.pth"; do
    if [[ -f "${path}" ]]; then echo "FOUND ${path}"; else echo "MISSING ${path}"; status=1; fi
  done
  return "${status}"
}

run_one() {
  local name="$1"
  shift
  local log="${ROOT_DIR}/output/depth_completion_baselines/${name}_run.log"
  echo "Starting ${name}; log=${log}"
  ( "$@" ) >"${log}" 2>&1 &
  echo "$! ${name}"
}

if [[ "${DOWNLOAD}" == "1" ]]; then
  download_weights || exit 2
fi
check_files || { echo "Weight check failed; download missing files before RUN=1." >&2; [[ "${RUN}" == "1" ]] && exit 2; }

if [[ "${RUN}" == "1" ]]; then
  [[ -f "${SPLIT_JSON}" ]] || { echo "Missing split: ${SPLIT_JSON}" >&2; exit 2; }
  [[ -d "${CACHE_ROOT}" ]] || { echo "Missing cache: ${CACHE_ROOT}" >&2; exit 2; }

  if [[ "${RUN_DMD3C}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${DMD3C_GPU:-1}" GPU=0 CHECKPOINT="${WEIGHT_ROOT}/dmd3c/result_ema.pth" \
      CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT=test \
      SUMMARY_OUTPUT="${ROOT_DIR}/output/depth_completion_baselines/dmd3c/summary.json" \
      PYTHON_BIN="${PYTHON_DMD3C}" bash scripts/runs/run_dmd3c_full_pbrt.sh
  fi
  if [[ "${RUN_OMNI}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${OMNI_GPU:-2}" DEVICE=cuda:0 CHECKPOINT="${WEIGHT_ROOT}/omnidc/modelv1.1_best_72epochs.pt" \
      DAV2_CHECKPOINT="${WEIGHT_ROOT}/omnidc/depth_anything_v2_vitl.pth" \
      CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT=test \
      SUMMARY_OUTPUT="${ROOT_DIR}/output/depth_completion_baselines/omnidc_zero_shot/summary.json" \
      PYTHON_BIN="${PYTHON_OMNI}" bash scripts/runs/run_omnidc_full_pbrt.sh
  fi
  if [[ "${RUN_LDCM}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${LDCM_GPU:-3}" DEVICE=cuda:0 MODEL="${WEIGHT_ROOT}/ldcm/ldcm.pt" \
      MOGE_MODEL="${WEIGHT_ROOT}/moge-2-vits-normal/model.pt" CACHE_ROOT="${CACHE_ROOT}" \
      SPLIT_JSON="${SPLIT_JSON}" SPLIT=test PYTHON_BIN="${PYTHON_LDCM}" \
      bash scripts/runs/run_ldcm_full_pbrt.sh
  fi
  if [[ "${RUN_LINGBOT}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${LINGBOT_GPU:-4}" DEVICE=cuda:0 MODEL="${WEIGHT_ROOT}/lingbot-depth-dc/model.pt" \
      CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT=test \
      PYTHON_BIN="${PYTHON_LINGBOT}" bash scripts/runs/run_lingbot_full_pbrt.sh
  fi
  if [[ "${RUN_DEPTHOR}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${DEPTHOR_GPU:-5}" DEVICE=cuda:0 \
      CHECKPOINT="${WEIGHT_ROOT}/depthor/depthor_zju_large.pt" \
      DAV2_CHECKPOINT="${WEIGHT_ROOT}/depthor/depth_anything_v2_vits.pth" \
      CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT=test \
      PYTHON_BIN="${PYTHON_DEPTHOR}" bash scripts/runs/run_depthor_full_pbrt.sh
  fi
fi
