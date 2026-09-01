#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BASE_PYTHON="${BASE_PYTHON:-/home/lab507/anaconda3/bin/python}"
HF_BIN="${HF_BIN:-/home/lab507/.local/bin/hf}"
GDOWN_BIN="${GDOWN_BIN:-/home/lab507/anaconda3/bin/gdown}"
LLAVA_PYTHON="${LLAVA_PYTHON:-/home/lab507/anaconda3/envs/llava/bin/python}"
CFORMER_PYTHON="${CFORMER_PYTHON:-/home/lab507/anaconda3/envs/cformer/bin/python}"
LDCM_PYTHON="${LDCM_PYTHON:-/home/lab507/anaconda3/envs/lingbot-world/bin/python}"
LINGBOT_PYTHON="${LINGBOT_PYTHON:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
DEPTHOR_PYTHON="${DEPTHOR_PYTHON:-/home/lab507/anaconda3/envs/py310/bin/python}"

WEIGHT_ROOT="${WEIGHT_ROOT:-${ROOT_DIR}/output/depth_completion_weights}"
INSTALL="${INSTALL:-1}"
DOWNLOAD="${DOWNLOAD:-1}"
BUILD_DMD3C="${BUILD_DMD3C:-0}"

mkdir -p "${WEIGHT_ROOT}"

run_pip() {
  local python_bin="$1"
  shift
  "${python_bin}" -m pip install "$@"
}

if [[ "${INSTALL}" == "1" ]]; then
  "${BASE_PYTHON}" -m pip install --user -U "huggingface_hub[cli]" gdown
  run_pip "${LDCM_PYTHON}" -r /data/pre_student/GJ/LDCM/requirements.txt
  run_pip "${LDCM_PYTHON}" -e /data/pre_student/GJ/LDCM
  run_pip "${LINGBOT_PYTHON}" -r /data/pre_student/GJ/lingbot-depth/requirements.txt
  run_pip "${DEPTHOR_PYTHON}" -r /data/pre_student/GJ/Depthor/requirements.txt
  run_pip "${LLAVA_PYTHON}" hydra-core==1.3.2 omegaconf==2.3.0
fi

export PATH="$(dirname "${HF_BIN}"):/home/lab507/anaconda3/bin:${PATH}"
hash -r

if [[ "${DOWNLOAD}" == "1" ]]; then
  HF_BIN="$(command -v hf || true)"
  GDOWN_BIN="$(command -v gdown || true)"
  [[ -n "${HF_BIN}" ]] || { echo "hf CLI is unavailable after installation." >&2; exit 2; }
  [[ -n "${GDOWN_BIN}" ]] || { echo "gdown is unavailable after installation." >&2; exit 2; }

  mkdir -p "${WEIGHT_ROOT}/dmd3c" "${WEIGHT_ROOT}/omnidc" \
    "${WEIGHT_ROOT}/ldcm" "${WEIGHT_ROOT}/moge-2-vits-normal" \
    "${WEIGHT_ROOT}/lingbot-depth-dc" "${WEIGHT_ROOT}/depthor"

  [[ -f "${WEIGHT_ROOT}/dmd3c/result_ema.pth" ]] || \
    "${HF_BIN}" download Liangyingping/DMD3Cpp-checkpoints result_ema.pth \
      --repo-type dataset --local-dir "${WEIGHT_ROOT}/dmd3c"
  "${HF_BIN}" download depth-anything/DA3METRIC-LARGE

  [[ -f "${WEIGHT_ROOT}/omnidc/modelv1.1_best_72epochs.pt" ]] || \
    "${GDOWN_BIN}" 'https://drive.google.com/uc?id=1ssJYFB3rQD5JEYgG7W6tRJg1hpQKvqPD' \
      -O "${WEIGHT_ROOT}/omnidc/modelv1.1_best_72epochs.pt"
  [[ -f "${WEIGHT_ROOT}/omnidc/depth_anything_v2_vitl.pth" ]] || \
    "${HF_BIN}" download depth-anything/Depth-Anything-V2-Large \
      depth_anything_v2_vitl.pth --local-dir "${WEIGHT_ROOT}/omnidc"

  [[ -f "${WEIGHT_ROOT}/ldcm/ldcm.pt" ]] || \
    "${HF_BIN}" download pkqbajng/LDCM ldcm.pt --local-dir "${WEIGHT_ROOT}/ldcm"
  [[ -f "${WEIGHT_ROOT}/moge-2-vits-normal/model.pt" ]] || \
    "${HF_BIN}" download Ruicheng/moge-2-vits-normal model.pt \
      --local-dir "${WEIGHT_ROOT}/moge-2-vits-normal"
  [[ -f "${WEIGHT_ROOT}/lingbot-depth-dc/model.pt" ]] || \
    "${HF_BIN}" download robbyant/lingbot-depth-postrain-dc-vitl14 \
      --local-dir "${WEIGHT_ROOT}/lingbot-depth-dc"

  [[ -f "${WEIGHT_ROOT}/depthor/depthor_zju_large.pt" ]] || \
    "${GDOWN_BIN}" 'https://drive.google.com/uc?id=1oZByVUklbjQHlZTdKFQdwkMKkugi4l6-' \
      -O "${WEIGHT_ROOT}/depthor/depthor_zju_large.pt"
  [[ -f "${WEIGHT_ROOT}/depthor/depth_anything_v2_vits.pth" ]] || \
    "${HF_BIN}" download depth-anything/Depth-Anything-V2-Small \
      depth_anything_v2_vits.pth --local-dir "${WEIGHT_ROOT}/depthor"
fi

echo "-- environment checks --"
for spec in \
  "DMD3C:${LLAVA_PYTHON}:hydra,omegaconf,BpOps" \
  "OMNI-DC:${CFORMER_PYTHON}:torch,apex,mmcv" \
  "LDCM:${LDCM_PYTHON}:torch,cv2" \
  "LingBot:${LINGBOT_PYTHON}:torch,xformers" \
  "DEPTHOR:${DEPTHOR_PYTHON}:torch,timm,BpOps"; do
  IFS=: read -r name python_bin modules <<<"${spec}"
  MODULES="${modules}" "${python_bin}" - <<'PY'
import importlib.util
import os
import sys
missing = [name for name in os.environ['MODULES'].split(',') if importlib.util.find_spec(name) is None]
if missing:
    print(f"MISSING {sys.executable}: {','.join(missing)}")
    raise SystemExit(1)
print(f"OK {sys.executable}: {os.environ['MODULES']}")
PY
done

echo "-- weights --"
find "${WEIGHT_ROOT}" -maxdepth 3 -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.safetensors' \) -print | sort

if [[ "${BUILD_DMD3C}" == "1" ]]; then
  bash scripts/setup_dmd3c_llava.sh
fi
