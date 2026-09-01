#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

METHODS="${METHODS:-rgbd_lfrd2}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT_DIR}/output/pbrt100_depth_completion/split.json}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cuda:0}"
GPU="${GPU:-0}"

make_split() {
  python scripts/make_pbrt100_completion_split.py \
    --cache_root "${CACHE_ROOT}" \
    --output "${SPLIT_JSON}"
}

run_rgbd_lfrd2() {
  python scripts/eval_unified_baselines.py \
    --manifest output/full_pbrt_manifest_seed123_iq.json \
    --model both \
    --workers 0 \
    --output output/pbrt100_depth_completion/rgbd_lfrd2/summary.json
}

run_ldcm() {
  CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT="${SPLIT}" \
  OUTPUT_DIR="${ROOT_DIR}/output/pbrt100_depth_completion/ldcm_zero_shot" \
  DEVICE="${DEVICE}" bash run_ldcm_full_pbrt.sh
}

run_lingbot() {
  CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT="${SPLIT}" \
  OUTPUT_DIR="${ROOT_DIR}/output/pbrt100_depth_completion/lingbot_dc_zero_shot" \
  DEVICE="${DEVICE}" bash run_lingbot_full_pbrt.sh
}

run_depthor() {
  CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/depthor/depthor_zju_large.pt}" \
  DAV2_CHECKPOINT="${DAV2_CHECKPOINT:-${ROOT_DIR}/output/depth_completion_weights/depthor/depth_anything_v2_vits.pth}" \
  CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT="${SPLIT}" \
  OUTPUT_DIR="${ROOT_DIR}/output/pbrt100_depth_completion/depthor" \
  DEVICE="${DEVICE}" bash run_depthor_full_pbrt.sh
}

run_omnidc() {
  CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT="${SPLIT}" \
  UNIFORMAT_DIR="${ROOT_DIR}/output/pbrt100_depth_completion/uniformat_omnidc" \
  SUMMARY_OUTPUT="${ROOT_DIR}/output/pbrt100_depth_completion/omnidc_zero_shot/summary.json" \
  LOG_DIR="${ROOT_DIR}/output/pbrt100_depth_completion/omnidc_runs" \
  GPU="${GPU}" bash run_omnidc_full_pbrt.sh
}

run_dmd3c() {
  CACHE_ROOT="${CACHE_ROOT}" SPLIT_JSON="${SPLIT_JSON}" SPLIT="${SPLIT}" \
  UNIFORMAT_DIR="${ROOT_DIR}/output/pbrt100_depth_completion/uniformat_dmd3c" \
  RUN_NAME="${RUN_NAME:-DMD3C_PBRT100}" \
  SUMMARY_OUTPUT="${ROOT_DIR}/output/pbrt100_depth_completion/dmd3c/summary.json" \
  GPU="${GPU}" bash run_dmd3c_full_pbrt.sh
}

run_completionformer() {
  local checkpoint="${COMPLETIONFORMER_CHECKPOINT:-}"
  if [[ -z "${checkpoint}" ]]; then
    checkpoint="$(find output/completionformer_full_pbrt/train_logs -name 'model_*.pt' 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
  if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "Missing CompletionFormer checkpoint. Set COMPLETIONFORMER_CHECKPOINT=/path/to/model.pt" >&2
    return 2
  fi
  /home/lab507/anaconda3/envs/cformer/bin/python eval_completionformer_full_pbrt.py \
    --completionformer_root "${COMPLETIONFORMER_ROOT:-/data/pre_student/hcy/CompletionFormer}" \
    --checkpoint "${checkpoint}" \
    --cache_root "${CACHE_ROOT}" \
    --split_json "${SPLIT_JSON}" \
    --output_dir output/pbrt100_depth_completion/completionformer \
    --device "${DEVICE}" \
    --save_predictions
}

summarize() {
  python summarize_pbrt100_depth_completion_comparison.py
}

usage() {
  cat <<'EOF'
Run PBRT100 depth completion baselines.

Examples:
  METHODS="rgbd_lfrd2" bash run_pbrt100_depth_completion_baselines.sh
  METHODS="ldcm lingbot omnidc dmd3c depthor completionformer" GPU=0 bash run_pbrt100_depth_completion_baselines.sh
  METHODS="all" bash run_pbrt100_depth_completion_baselines.sh

Outputs:
  output/pbrt100_depth_completion/*/summary.json
  output/pbrt100_depth_completion/comparison/summary.md
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

make_split

if [[ "${METHODS}" == "all" ]]; then
  METHODS="rgbd_lfrd2 completionformer ldcm lingbot omnidc dmd3c depthor"
fi

for method in ${METHODS}; do
  case "${method}" in
    rgbd_lfrd2) run_rgbd_lfrd2 ;;
    completionformer) run_completionformer ;;
    ldcm) run_ldcm ;;
    lingbot) run_lingbot ;;
    omnidc) run_omnidc ;;
    dmd3c) run_dmd3c ;;
    depthor) run_depthor ;;
    summary) ;;
    *) echo "Unknown method: ${method}" >&2; usage >&2; exit 1 ;;
  esac
done

summarize
