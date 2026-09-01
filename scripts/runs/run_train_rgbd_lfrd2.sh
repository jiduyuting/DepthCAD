#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON_RGBD="${PYTHON_RGBD:-/home/lab507/anaconda3/envs/depthcad/bin/python}"
PYTHON_LFRD2="${PYTHON_LFRD2:-/home/lab507/anaconda3/envs/SVDC/bin/python}"
MANIFEST="${MANIFEST:-output/full_pbrt_manifest_seed123.json}"
RGBD_OUTPUT="${RGBD_OUTPUT:-output/rgbd_imaging_full_pbrt}"
LFRD2_OUTPUT="${LFRD2_OUTPUT:-output/lfrd2_full_pbrt}"
DEVICE="${DEVICE:-auto}"
MODELS="${MODELS:-both}"
RESUME="${RESUME:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  echo "Create it first with scripts/build_unified_pbrt_manifest.py." >&2
  exit 1
fi

if [[ "${DEVICE}" == "auto" ]]; then
  if "${PYTHON_RGBD}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

resume_rgbd_args=()
resume_lfrd2_args=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "yes" ]]; then
  if [[ -f "${RGBD_OUTPUT}/last.pth" ]]; then
    resume_rgbd_args+=(--resume)
  else
    echo "[runner] RGBD checkpoint not found; starting from epoch 1"
  fi
  if [[ -f "${LFRD2_OUTPUT}/last.pth" ]]; then
    resume_lfrd2_args+=(--resume)
  else
    echo "[runner] LFRD2 checkpoint not found; starting from epoch 1"
  fi
fi

echo "[runner] root=${ROOT}"
echo "[runner] device=${DEVICE}"
echo "[runner] manifest=${MANIFEST}"
echo "[runner] RGBD python=${PYTHON_RGBD}"
echo "[runner] LFRD2 python=${PYTHON_LFRD2}"

case "${MODELS}" in
  rgbd|both)
    "${PYTHON_RGBD}" scripts/train_rgbd_unified_pbrt.py \
      --manifest "${MANIFEST}" \
      --output "${RGBD_OUTPUT}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE:-4}" \
      --epochs "${EPOCHS:-200}" \
      --learning_rate "${RGBD_LR:-1e-4}" \
      --workers "${WORKERS:-0}" \
      "${resume_rgbd_args[@]}"
    ;;
esac

case "${MODELS}" in
  lfrd2|both)
    "${PYTHON_LFRD2}" scripts/train_lfrd2_unified_pbrt.py \
      --manifest "${MANIFEST}" \
      --output "${LFRD2_OUTPUT}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE:-4}" \
      --epochs "${EPOCHS:-200}" \
      --learning_rate "${LFRD2_LR:-5e-5}" \
      --workers "${WORKERS:-0}" \
      "${resume_lfrd2_args[@]}"
    ;;
esac

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "yes" ]]; then
  eval_args=(
    --manifest "${MANIFEST}"
    --output "${EVAL_OUTPUT:-output/pbrt100_depth_completion/rgbd_lfrd2/summary.json}"
    --device "${DEVICE}"
    --batch_size "${EVAL_BATCH_SIZE:-4}"
    --workers "${EVAL_WORKERS:-0}"
    --model "${MODELS}"
    --rgbd_checkpoint "${RGBD_OUTPUT}/checkpoint_best.pth"
    --lfrd2_checkpoint "${LFRD2_OUTPUT}/checkpoint_best_net.pth"
  )
  eval_python="${PYTHON_RGBD}"
  if [[ "${MODELS}" == "lfrd2" || "${MODELS}" == "both" ]]; then
    eval_python="${PYTHON_LFRD2}"
  fi
  "${eval_python}" scripts/eval_unified_baselines.py "${eval_args[@]}"
fi
