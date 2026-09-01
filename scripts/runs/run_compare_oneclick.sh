#!/usr/bin/env bash
# set -euo pipefail
set +e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runs/run_compare_oneclick.sh [real|pbrt|all]

Default scope is `all`.

Main outputs:
  - Real: output/pbrt_real_new_selection/oneclick_compare/
  - PBRT: output/pbrt_propainter_seed123/ and checkpoint eval dir

Useful env vars:
  REAL_SELECTED_ROOT=output/pbrt_real_new_selection/selected
  REAL_OUTPUT_ROOT=output/pbrt_real_new_selection/oneclick_compare
  REAL_MASK_DIR=/data/pre_student/hcy/datasets/pbrt/Real/noise_masks
  REAL_METHODS="depth_only raw9_satclip raw9_realholes propainter"
  REAL_INCLUDE_DEPTHCAD=1
  REAL_INCLUDE_AFTER_SYNTH=1
  REAL_INCLUDE_PROPAGATION=1
  REAL_DEPTH_SCALE=1.0
  REAL_HOLE_THRESHOLD=0.0
  REAL_VALID_MIN=0.1
  REAL_VALID_MAX=9.9
  REAL_PROPAINTER_HEIGHT=424
  REAL_PROPAINTER_WIDTH=512
  REAL_PROPAINTER_NEIGHBOR_LENGTH=10
  REAL_PROPAINTER_REF_STRIDE=10
  REAL_PROPAINTER_SUBVIDEO_LENGTH=80

  PBRT_CACHE_DIR=depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123
  PBRT_CHECKPOINT=output/pbrt_real_threshold_amp_depth_finetune_replay_e30_p8_keepamp_rw030/best.pt
  PBRT_SPLIT_JSON=output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/split.json
  PBRT_PROPAINTER_CASE=output/pbrt_propainter_seed123
  PBRT_PROPAINTER_EVAL_DIR=output/pbrt_propainter_seed123/evaluation
  PBRT_RUN_PROPAINTER=1
EOF
}

MODE="${1:-all}"
case "$MODE" in
  real|pbrt|all) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown scope: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac

CONTROL_PYTHON_BIN="${CONTROL_PYTHON_BIN:-/home/lab507/anaconda3/envs/control/bin/python}"
PBRT_PYTHON_BIN="${PBRT_PYTHON_BIN:-/home/lab507/anaconda3/envs/SVDC/bin/python}"

REAL_SELECTED_ROOT="${REAL_SELECTED_ROOT:-output/pbrt_real_new_selection/selected}"
REAL_OUTPUT_ROOT="${REAL_OUTPUT_ROOT:-output/pbrt_real_new_selection/oneclick_compare}"
REAL_MASK_DIR="${REAL_MASK_DIR:-/data/pre_student/hcy/datasets/pbrt/Real/noise_masks}"
REAL_METHODS="${REAL_METHODS:-depth_only raw9_satclip raw9_realholes propainter}"
REAL_INCLUDE_DEPTHCAD="${REAL_INCLUDE_DEPTHCAD:-0}"
REAL_INCLUDE_AFTER_SYNTH="${REAL_INCLUDE_AFTER_SYNTH:-0}"
REAL_INCLUDE_PROPAGATION="${REAL_INCLUDE_PROPAGATION:-0}"
REAL_DEPTH_SCALE="${REAL_DEPTH_SCALE:-1.0}"
REAL_HOLE_THRESHOLD="${REAL_HOLE_THRESHOLD:-0.0}"
REAL_VALID_MIN="${REAL_VALID_MIN:-0.1}"
REAL_VALID_MAX="${REAL_VALID_MAX:-9.9}"
REAL_DEPTH_VIS_MIN="${REAL_DEPTH_VIS_MIN:-0.5}"
REAL_DEPTH_VIS_MAX="${REAL_DEPTH_VIS_MAX:-4.5}"
REAL_SKIP_EXISTING="${REAL_SKIP_EXISTING:-1}"
REAL_PROPAINTER_HEIGHT="${REAL_PROPAINTER_HEIGHT:-}"
REAL_PROPAINTER_WIDTH="${REAL_PROPAINTER_WIDTH:-}"
REAL_PROPAINTER_NEIGHBOR_LENGTH="${REAL_PROPAINTER_NEIGHBOR_LENGTH:-10}"
REAL_PROPAINTER_REF_STRIDE="${REAL_PROPAINTER_REF_STRIDE:-10}"
REAL_PROPAINTER_SUBVIDEO_LENGTH="${REAL_PROPAINTER_SUBVIDEO_LENGTH:-80}"

PBRT_CACHE_DIR="${PBRT_CACHE_DIR:-depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123}"
PBRT_CHECKPOINT="${PBRT_CHECKPOINT:-output/pbrt_real_threshold_amp_depth_finetune_replay_e30_p8_keepamp_rw030/best.pt}"
PBRT_SPLIT_JSON="${PBRT_SPLIT_JSON:-output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/split.json}"
PBRT_FINAL_EVAL_DIR="${PBRT_FINAL_EVAL_DIR:-$(dirname "$PBRT_CHECKPOINT")/eval_pbrt_val97_endpoint}"
PBRT_PROPAINTER_CASE="${PBRT_PROPAINTER_CASE:-output/pbrt_propainter_seed123}"
PBRT_PROPAINTER_EVAL_DIR="${PBRT_PROPAINTER_EVAL_DIR:-output/pbrt_propainter_seed123/evaluation}"
PBRT_RUN_PROPAINTER="${PBRT_RUN_PROPAINTER:-1}"
PBRT_PROPAINTER_HEIGHT="${PBRT_PROPAINTER_HEIGHT:-256}"
PBRT_PROPAINTER_WIDTH="${PBRT_PROPAINTER_WIDTH:-256}"
PBRT_PROPAINTER_NEIGHBOR_LENGTH="${PBRT_PROPAINTER_NEIGHBOR_LENGTH:-10}"
PBRT_PROPAINTER_REF_STRIDE="${PBRT_PROPAINTER_REF_STRIDE:-10}"
PBRT_PROPAINTER_SUBVIDEO_LENGTH="${PBRT_PROPAINTER_SUBVIDEO_LENGTH:-80}"

check_bin() {
  local bin="$1"
  local name="$2"
  if [[ ! -x "$bin" ]]; then
    echo "Missing $name: $bin" >&2
    exit 1
  fi
}

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

check_bin "$CONTROL_PYTHON_BIN" "control python"
check_bin "$PBRT_PYTHON_BIN" "PBRT python"
check_path "$PBRT_CACHE_DIR" "PBRT cache dir"
check_path "$PBRT_CHECKPOINT" "PBRT checkpoint"
check_path "$PBRT_SPLIT_JSON" "PBRT split json"
check_path "$REAL_SELECTED_ROOT/depth_m" "Real selected depth dir"
check_path "$REAL_SELECTED_ROOT/raw9_chw" "Real selected raw9 dir"
check_path "$REAL_MASK_DIR" "Real mask dir"

REAL_STAGE_ROOT=""
cleanup() {
  if [[ -n "$REAL_STAGE_ROOT" && -d "$REAL_STAGE_ROOT" ]]; then
    rm -rf "$REAL_STAGE_ROOT"
  fi
}
trap cleanup EXIT

build_real_methods() {
  local -a methods=()
  if [[ -n "${REAL_METHODS:-}" ]]; then
    read -r -a methods <<< "$REAL_METHODS"
  else
    methods=(depth_only raw9_satclip raw9_realholes propainter)
  fi
  if [[ "$REAL_INCLUDE_AFTER_SYNTH" == "1" ]]; then
    methods+=(after_synth)
  fi
  if [[ "$REAL_INCLUDE_PROPAGATION" == "1" ]]; then
    methods+=(propagation)
  fi
  if [[ "$REAL_INCLUDE_DEPTHCAD" == "1" ]]; then
    methods+=(depthcad_depth_gray)
  fi

  local -a dedup=()
  local seen=" "
  local method
  for method in "${methods[@]}"; do
    if [[ "$seen" != *" ${method} "* ]]; then
      dedup+=("$method")
      seen+=" ${method} "
    fi
  done
  REAL_METHODS_ARR=("${dedup[@]}")
}

stage_real_selection() {
  REAL_STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/depthcad-real-stage.XXXXXX")"
  mkdir -p "$REAL_STAGE_ROOT/depth" "$REAL_STAGE_ROOT/iq"

  shopt -s nullglob
  local depth_files=("$REAL_SELECTED_ROOT"/depth_m/*.npy)
  shopt -u nullglob

  if [[ "${#depth_files[@]}" -eq 0 ]]; then
    echo "No selected depth files found under $REAL_SELECTED_ROOT/depth_m" >&2
    exit 1
  fi

  local depth_path stem raw_path
  for depth_path in "${depth_files[@]}"; do
    stem="$(basename "$depth_path" .npy)"
    raw_path="$REAL_SELECTED_ROOT/raw9_chw/${stem}.npy"
    if [[ ! -f "$raw_path" ]]; then
      echo "Missing paired raw9 file for $stem: $raw_path" >&2
      exit 1
    fi
    cp -f "$depth_path" "$REAL_STAGE_ROOT/depth/depth_${stem}.npy"
    cp -f "$raw_path" "$REAL_STAGE_ROOT/iq/iq_${stem}.npy"
  done

  echo "[real] staged ${#depth_files[@]} samples at $REAL_STAGE_ROOT"
}

method_root_for_real() {
  case "$1" in
    depth_only) echo "$REAL_OUTPUT_ROOT/methods/depth_only_flow" ;;
    raw9_satclip) echo "$REAL_OUTPUT_ROOT/methods/raw9_satclip" ;;
    raw9_realholes) echo "$REAL_OUTPUT_ROOT/methods/raw9_realholes" ;;
    after_synth) echo "$REAL_OUTPUT_ROOT/methods/after_synth" ;;
    propagation) echo "$REAL_OUTPUT_ROOT/methods/propagation" ;;
    depthcad_depth_gray) echo "$REAL_OUTPUT_ROOT/methods/depthcad_depth_gray" ;;
    propainter) echo "$REAL_OUTPUT_ROOT/external_inpaint/propainter_run" ;;
    *) return 1 ;;
  esac
}

run_real_compare() {
  build_real_methods
  stage_real_selection

  local -a run_cmd=(
    "$CONTROL_PYTHON_BIN"
    scripts/run_real_capture_method_suite.py
    --data_root "$REAL_STAGE_ROOT"
    --output_root "$REAL_OUTPUT_ROOT"
    --sample_mode all
    --methods
  )
  run_cmd+=("${REAL_METHODS_ARR[@]}")
  run_cmd+=(
    --depth_scale "$REAL_DEPTH_SCALE"
    --hole_depth_threshold "$REAL_HOLE_THRESHOLD"
    --valid_min_depth "$REAL_VALID_MIN"
    --valid_max_depth "$REAL_VALID_MAX"
    --depth_vis_min "$REAL_DEPTH_VIS_MIN"
    --depth_vis_max "$REAL_DEPTH_VIS_MAX"
    --propainter_neighbor_length "$REAL_PROPAINTER_NEIGHBOR_LENGTH"
    --propainter_ref_stride "$REAL_PROPAINTER_REF_STRIDE"
    --propainter_subvideo_length "$REAL_PROPAINTER_SUBVIDEO_LENGTH"
    --no_compare
  )
  if [[ -n "$REAL_PROPAINTER_HEIGHT" ]]; then
    run_cmd+=(--propainter_height "$REAL_PROPAINTER_HEIGHT")
  fi
  if [[ -n "$REAL_PROPAINTER_WIDTH" ]]; then
    run_cmd+=(--propainter_width "$REAL_PROPAINTER_WIDTH")
  fi
  if [[ "$REAL_SKIP_EXISTING" != "0" ]]; then
    run_cmd+=(--skip_existing)
  fi
  if [[ " ${REAL_METHODS_ARR[*]} " == *" depthcad_depth_gray "* ]]; then
    run_cmd+=(--allow_depthcad_cpu)
  fi

  echo "[real] running method suite: ${REAL_METHODS_ARR[*]}"
  "${run_cmd[@]}"

  local -a compare_cmd=(
    "$CONTROL_PYTHON_BIN"
    scripts/compare_realhole_method_outputs.py
    --depth_dir "$REAL_OUTPUT_ROOT/prepared/all/depth_m"
    --mask_dir "$REAL_MASK_DIR"
    --output_dir "$REAL_OUTPUT_ROOT/realhole_method_comparison"
  )
  local method root
  for method in "${REAL_METHODS_ARR[@]}"; do
    root="$(method_root_for_real "$method" || true)"
    if [[ -n "${root:-}" && -e "$root" ]]; then
      compare_cmd+=(--method "${method}=$root")
    fi
  done

  echo "[real] building actual-mask comparison: ${REAL_OUTPUT_ROOT}/realhole_method_comparison"
  "${compare_cmd[@]}"
}

build_pbrt_existing_eval_args() {
  PBRT_EXISTING_EVAL_ARGS=()
  local -a candidates=(
    "final_flow:$PBRT_FINAL_EVAL_DIR"
    "depth_only_flow:output/depth_flow_restoration_noisy_ns_n1000_endpoint/eval_seed123_endpoint"
    "large_resunet:output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_large_res48_b2/eval_seed123_endpoint"
    "residual_restoration:output/depth_restoration_unet_noisy_ns_n1000/eval_seed123"
  )
  local spec name path
  for spec in "${candidates[@]}"; do
    name="${spec%%:*}"
    path="${spec#*:}"
    if [[ -d "$path" && -f "$path/summary.json" ]]; then
      PBRT_EXISTING_EVAL_ARGS+=(--existing_eval "${name}:${path}")
    fi
  done
}

run_pbrt_compare() {
  build_pbrt_existing_eval_args

  echo "[pbrt] evaluating main flow checkpoint"
  CHECKPOINT="$PBRT_CHECKPOINT" \
  OUTPUT_DIR="$PBRT_FINAL_EVAL_DIR" \
  PBRT_SPLIT_JSON="$PBRT_SPLIT_JSON" \
  PYTHON_BIN="$PBRT_PYTHON_BIN" \
    bash scripts/runs/run_pbrt_val97_flow_eval.sh

  if [[ "$PBRT_RUN_PROPAINTER" != "0" ]]; then
    echo "[pbrt] exporting ProPainter case"
    "$CONTROL_PYTHON_BIN" scripts/export_pbrt_propainter_case.py \
      --cache_dir "$PBRT_CACHE_DIR" \
      --output_dir "$PBRT_PROPAINTER_CASE"

    echo "[pbrt] running ProPainter"
    "$CONTROL_PYTHON_BIN" scripts/run_external_inpainting_far_pic.py run-propainter \
      --case "$PBRT_PROPAINTER_CASE" \
      --output_dir "$PBRT_PROPAINTER_CASE/propainter_run" \
      --height "$PBRT_PROPAINTER_HEIGHT" \
      --width "$PBRT_PROPAINTER_WIDTH" \
      --mask_dilation 0 \
      --neighbor_length "$PBRT_PROPAINTER_NEIGHBOR_LENGTH" \
      --ref_stride "$PBRT_PROPAINTER_REF_STRIDE" \
      --subvideo_length "$PBRT_PROPAINTER_SUBVIDEO_LENGTH" \
      --decode

    echo "[pbrt] evaluating ProPainter against available baselines"
    "$CONTROL_PYTHON_BIN" scripts/eval_pbrt_external_inpainting.py \
      --case_dir "$PBRT_PROPAINTER_CASE" \
      --output_dir "$PBRT_PROPAINTER_EVAL_DIR" \
      "${PBRT_EXISTING_EVAL_ARGS[@]}"
  fi
}

if [[ "$MODE" == "real" || "$MODE" == "all" ]]; then
  run_real_compare
fi

if [[ "$MODE" == "pbrt" || "$MODE" == "all" ]]; then
  run_pbrt_compare
fi

echo "Done."
