import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import torch

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from inference_depth_postprocess import opencv_depth_inpaint
from train_depth_flow_restoration import flow_model_in_channels, predict_endpoint_norm, sample_flow
from train_depth_restoration import robust_nonnegative_channels


RAW9_SPATIAL_TRANSFORMS = ("none", "flip_lr", "flip_ud", "rot180")
RAW9_TRANSFORM_CHOICES = ("checkpoint",) + RAW9_SPATIAL_TRANSFORMS + ("auto",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run conditional-flow restoration on paired real raw9/depth .npy files."
    )
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument(
        "--samples",
        type=str,
        nargs="+",
        default=None,
        help="Optional sample stems to run, for example: --samples 33 34 35.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/real_raw9_flow_finetune_iq6_realholes_e40_m8/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/real_raw9_flow_infer_iq6_realholes_e40_m8_best",
    )
    parser.add_argument(
        "--amplitude_mode",
        type=str,
        default="iq6",
        choices=["iq6", "raw_258"],
        help=(
            "iq6 computes sqrt(I^2+Q^2) from channels [0,1],[2,3],[4,5], "
            "matching synthetic cache construction. raw_258 uses channels [2,5,8] directly."
        ),
    )
    parser.add_argument(
        "--raw9_transform",
        type=str,
        default="checkpoint",
        choices=RAW9_TRANSFORM_CHOICES,
        help=(
            "Spatial transform applied to raw9 before amplitude features. "
            "checkpoint uses the raw9_transform saved in the checkpoint args, falling back to none."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["endpoint", "euler"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--hole_depth_threshold", type=float, default=1.0)
    parser.add_argument("--valid_min_depth", type=float, default=1.0)
    parser.add_argument("--valid_max_depth", type=float, default=9.9)
    parser.add_argument(
        "--depth_unit",
        type=str,
        default="auto",
        choices=["auto", "m", "mm"],
        help="Depth unit for paired depth .npy files. auto converts millimeter-like maps to meters.",
    )
    parser.add_argument(
        "--hole_amplitude_mode",
        type=str,
        default="zero",
        choices=["zero", "keep_all"],
        help=(
            "zero keeps historical behavior by clearing amplitude inside repair holes. "
            "keep_all preserves raw/amplitude features in holes, which is needed by "
            "models fine-tuned with synthetic saturation/clipping augmentation."
        ),
    )
    parser.add_argument(
        "--hole_mask_mode",
        type=str,
        default="threshold",
        choices=["threshold", "cleaned", "speckle_cleaned", "amp_speckle_cleaned"],
        help=(
            "threshold uses only invalid/depth-threshold holes. cleaned also absorbs local "
            "speckle/outlier pixels into the repair mask. speckle_cleaned also detects "
            "global sparse speckle clusters. amp_speckle_cleaned further filters those "
            "clusters with raw-amplitude contrast so normal regions are less likely to be "
            "misclassified as holes."
        ),
    )
    parser.add_argument(
        "--clean_outlier_abs",
        type=float,
        default=0.35,
        help="Absolute depth jump in meters used by --hole_mask_mode=cleaned.",
    )
    parser.add_argument(
        "--clean_outlier_mad_scale",
        type=float,
        default=6.0,
        help="Robust local MAD multiplier used by --hole_mask_mode=cleaned.",
    )
    parser.add_argument(
        "--clean_median_ksize",
        type=int,
        default=7,
        help="Odd median-filter kernel for detecting depth speckles in cleaned mask mode.",
    )
    parser.add_argument(
        "--clean_dilate",
        type=int,
        default=1,
        help="Dilation radius applied to the cleaned repair mask.",
    )
    parser.add_argument(
        "--clean_min_component_area",
        type=int,
        default=6,
        help="Remove cleaned-mask components smaller than this area.",
    )
    parser.add_argument(
        "--speckle_window",
        type=int,
        default=11,
        help="Local window used to score sparse speckle clusters in speckle_cleaned mode.",
    )
    parser.add_argument(
        "--speckle_density_threshold",
        type=float,
        default=0.10,
        help="Minimum local anomaly density required to keep a speckle cluster.",
    )
    parser.add_argument(
        "--speckle_residual_abs",
        type=float,
        default=0.18,
        help="Absolute residual threshold used to seed speckle candidates.",
    )
    parser.add_argument(
        "--speckle_link_radius",
        type=int,
        default=2,
        help="Morphological radius used to link sparse speckle candidates into clusters.",
    )
    parser.add_argument(
        "--speckle_min_component_area",
        type=int,
        default=4,
        help="Minimum component area kept in speckle_cleaned mode.",
    )
    parser.add_argument(
        "--speckle_max_component_area",
        type=int,
        default=9000,
        help="Maximum linked speckle component area kept in speckle_cleaned mode.",
    )
    parser.add_argument(
        "--speckle_max_bbox_side",
        type=int,
        default=140,
        help="Reject global speckle components with a larger bbox side to avoid masking object edges.",
    )
    parser.add_argument(
        "--speckle_amp_ring_radius",
        type=int,
        default=7,
        help="Ring radius used to compare component amplitude against nearby reliable pixels.",
    )
    parser.add_argument(
        "--speckle_amp_ratio_min",
        type=float,
        default=2.5,
        help="Minimum seed-amplitude to ring-amplitude ratio kept in amp_speckle_cleaned mode.",
    )
    parser.add_argument(
        "--speckle_amp_delta_min",
        type=float,
        default=4000.0,
        help="Minimum seed-amplitude minus ring-amplitude difference kept in amp_speckle_cleaned mode.",
    )
    parser.add_argument(
        "--speckle_amp_abs_min",
        type=float,
        default=8000.0,
        help="Minimum absolute seed amplitude kept in amp_speckle_cleaned mode when ring evidence is weak.",
    )
    parser.add_argument(
        "--plane_fill",
        action="store_true",
        default=False,
        help="Also save a geometry-first hole-only result using local plane/median component filling.",
    )
    parser.add_argument("--plane_ring_radius", type=int, default=7)
    parser.add_argument("--plane_min_points", type=int, default=24)
    parser.add_argument("--plane_max_component_area", type=int, default=24000)
    parser.add_argument("--plane_max_abs_residual", type=float, default=0.04)
    parser.add_argument("--plane_blend_model", type=float, default=0.0)
    parser.add_argument(
        "--aligned_fill",
        action="store_true",
        default=False,
        help=(
            "Also save a model-first hole-only result whose flow prediction is locally "
            "affine-aligned to reliable depth around each hole component."
        ),
    )
    parser.add_argument("--align_ring_radius", type=int, default=9)
    parser.add_argument("--align_min_points", type=int, default=24)
    parser.add_argument("--align_max_component_area", type=int, default=24000)
    parser.add_argument("--align_max_abs_residual", type=float, default=0.06)
    parser.add_argument("--align_min_pred_std", type=float, default=0.003)
    parser.add_argument("--align_min_scale", type=float, default=0.75)
    parser.add_argument("--align_max_scale", type=float, default=1.25)
    parser.add_argument(
        "--post_clip_mode",
        type=str,
        default="valid_range",
        choices=["none", "valid_range", "valid_percentile"],
        help="Physical post-clipping for real data outputs. Raw predictions are still saved.",
    )
    parser.add_argument(
        "--post_clip_percentiles",
        type=float,
        nargs=2,
        default=[0.5, 99.5],
        help="Percentiles used when --post_clip_mode=valid_percentile.",
    )
    parser.add_argument("--vis_max_samples", type=int, default=1000000)
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument(
        "--split_added_fill",
        action="store_true",
        default=False,
        help=(
            "Fill threshold holes with the flow model first, then repair only the extra "
            "cleaned/amp-speckle pixels with a dedicated postprocess."
        ),
    )
    parser.add_argument(
        "--split_added_mode",
        type=str,
        default="ns",
        choices=["ns", "plane", "anchor_ns"],
        help=(
            "Postprocess used only on extra cleaned/amp-speckle pixels when "
            "--split_added_fill is enabled. anchor_ns builds the guide from raw depth "
            "plus anchor-filled threshold holes, then inserts the flow prediction back "
            "only on the true threshold holes."
        ),
    )
    parser.add_argument(
        "--split_added_inpaint_radius",
        type=int,
        default=3,
        help="OpenCV NS inpaint radius used by --split_added_mode=ns.",
    )
    parser.add_argument(
        "--hybrid_mode",
        type=str,
        default="none",
        choices=["none", "component_fallback"],
        help=(
            "Hybrid postprocess for real holes. component_fallback keeps anchor on small/thin "
            "hole components and uses model only on larger/thicker ones."
        ),
    )
    parser.add_argument(
        "--hybrid_component_min_area",
        type=int,
        default=700,
        help="Minimum connected-component area required to use model output in hybrid mode.",
    )
    parser.add_argument(
        "--hybrid_component_min_radius",
        type=float,
        default=5.5,
        help="Minimum padded distance-transform radius required to use model output in hybrid mode.",
    )
    parser.add_argument(
        "--hybrid_keep_border_anchor",
        action="store_true",
        default=False,
        help="If set, hole components touching the image border always keep anchor in hybrid mode.",
    )
    parser.add_argument(
        "--hybrid_max_mean_abs_anchor_diff",
        type=float,
        default=0.0,
        help=(
            "If >0, reject model output for a hole component when mean |model-anchor| "
            "inside that component exceeds this value in meters."
        ),
    )
    parser.add_argument(
        "--hybrid_max_p95_abs_anchor_diff",
        type=float,
        default=0.0,
        help=(
            "If >0, reject model output for a hole component when p95 |model-anchor| "
            "inside that component exceeds this value in meters."
        ),
    )
    parser.add_argument(
        "--gated_fill",
        action="store_true",
        default=False,
        help=(
            "Also save an anchor-gated hole-only result: anchor + gate * (model-anchor) "
            "inside holes, raw depth outside holes."
        ),
    )
    parser.add_argument(
        "--gate_diff_soft",
        type=float,
        default=0.02,
        help="Use full model residual where |model-anchor| is at or below this value in meters.",
    )
    parser.add_argument(
        "--gate_diff_hard",
        type=float,
        default=0.08,
        help="Fall back to anchor where |model-anchor| is at or above this value in meters.",
    )
    parser.add_argument(
        "--gate_component_max_mean_abs_diff",
        type=float,
        default=0.0,
        help=(
            "If >0, force a hole component to anchor when mean |model-anchor| exceeds "
            "this value in meters."
        ),
    )
    parser.add_argument(
        "--gate_component_max_p95_abs_diff",
        type=float,
        default=0.0,
        help=(
            "If >0, force a hole component to anchor when p95 |model-anchor| exceeds "
            "this value in meters."
        ),
    )
    parser.add_argument(
        "--gate_keep_border_anchor",
        action="store_true",
        default=False,
        help="If set, hole components touching the image border always keep anchor in gated fill.",
    )
    parser.add_argument(
        "--repair_mask_mode",
        type=str,
        default="all",
        choices=["all", "exclude_large_border"],
        help=(
            "Select which sensor holes are repaired in final hole-only outputs. "
            "exclude_large_border preserves large/border hole components as invalid so "
            "they are not hallucinated into dark filled regions."
        ),
    )
    parser.add_argument(
        "--preserve_border_hole_min_area",
        type=int,
        default=1024,
        help="In exclude_large_border mode, preserve border-touching hole components at least this large.",
    )
    parser.add_argument(
        "--preserve_large_hole_min_area",
        type=int,
        default=24000,
        help="In exclude_large_border mode, preserve any hole component at least this large; <=0 disables.",
    )
    parser.add_argument(
        "--preserve_hole_max_bbox_side",
        type=int,
        default=220,
        help="In exclude_large_border mode, preserve hole components with bbox side at least this value; <=0 disables.",
    )
    parser.add_argument(
        "--preserve_holes_as_nan",
        action="store_true",
        default=False,
        help="Write preserved unrepairable holes as NaN in hole-only/gated outputs instead of keeping raw values.",
    )
    return parser.parse_args()


def natural_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def finite_stats(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "min": None,
            "p5": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "min": float(np.min(finite)),
        "p5": float(np.percentile(finite, 5.0)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def safe_mean_abs(values, mask):
    valid = mask & np.isfinite(values)
    if valid.sum() == 0:
        return None
    return float(np.mean(np.abs(values[valid])))


def normalize_depth(depth, center, scale, clip_norm_depth):
    out = (depth - center) / scale
    out = np.nan_to_num(
        out,
        nan=0.0,
        neginf=-float(clip_norm_depth),
        posinf=float(clip_norm_depth),
    )
    return np.clip(out, -float(clip_norm_depth), float(clip_norm_depth)).astype(np.float32)


def move_condition_to_device(condition, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in condition.items()
    }


@torch.no_grad()
def predict_depth(model, condition, ckpt_args, sampling_mode, sample_steps):
    if sampling_mode == "endpoint":
        pred_norm = predict_endpoint_norm(
            model,
            condition,
            int(ckpt_args.get("time_channels", 16)),
            float(ckpt_args.get("max_velocity_norm", 4.0)),
            float(ckpt_args.get("clip_norm_depth", 8.0)),
            float(ckpt_args.get("velocity_scale", 1.0)),
        )
    else:
        pred_norm = sample_flow(
            model,
            condition,
            int(ckpt_args.get("time_channels", 16)),
            float(ckpt_args.get("max_velocity_norm", 4.0)),
            int(sample_steps),
            float(ckpt_args.get("clip_norm_depth", 8.0)),
            float(ckpt_args.get("velocity_scale", 1.0)),
        )

    scale = condition["scale"].view(-1, 1, 1, 1)
    center = condition["center"].view(-1, 1, 1, 1)
    return pred_norm * scale + center


def clip_prediction(pred, depth, valid_mask, args):
    if args.post_clip_mode == "none":
        return pred.astype(np.float32), None

    valid_values = depth[valid_mask & np.isfinite(depth)]
    if valid_values.size == 0:
        lo = float(args.valid_min_depth)
        hi = float(args.valid_max_depth)
    elif args.post_clip_mode == "valid_range":
        lo = float(np.min(valid_values))
        hi = float(np.max(valid_values))
    else:
        lo, hi = np.percentile(valid_values, args.post_clip_percentiles)
        lo = float(lo)
        hi = float(hi)

    lo = max(lo, float(args.valid_min_depth))
    hi = min(hi, float(args.valid_max_depth))
    if hi <= lo:
        lo = float(args.valid_min_depth)
        hi = float(args.valid_max_depth)
    return np.clip(pred, lo, hi).astype(np.float32), [lo, hi]


def image_limits(*arrays):
    values = []
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    values = np.concatenate(values)
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def component_max_radius(component_mask):
    component_mask = np.asarray(component_mask, dtype=np.uint8)
    if component_mask.size == 0 or not component_mask.any():
        return 0.0
    # Pad with zeros so components touching the crop boundary still see a valid exterior.
    padded = np.pad(component_mask, 1, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    return float(np.max(dist[1:-1, 1:-1]))


def fill_small_mask_holes(mask, max_area):
    mask = np.asarray(mask, dtype=bool)
    inv = ~mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv.astype(np.uint8), connectivity=8)
    out = mask.copy()
    h, w = mask.shape
    max_area = int(max_area)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if not touches_border and area <= max_area:
            out[labels == label] = True
    return out


def remove_small_components(mask, min_area):
    mask = np.asarray(mask, dtype=bool)
    min_area = int(min_area)
    if min_area <= 1:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = True
    return out


def dilate_bool(mask, radius):
    radius = int(radius)
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def robust_median_depth(depth, valid, ksize):
    ksize = int(ksize)
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    values = np.asarray(depth, dtype=np.float32)
    fill_value = float(np.median(values[valid])) if valid.any() else 0.0
    dense = np.where(valid, values, fill_value).astype(np.float32)
    try:
        from scipy.ndimage import median_filter

        return median_filter(dense, size=ksize, mode="nearest").astype(np.float32)
    except Exception:
        finite = dense[np.isfinite(dense)]
        if finite.size == 0:
            return dense
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        if hi <= lo:
            return dense
        scaled = np.clip((dense - lo) / (hi - lo) * 65535.0, 0.0, 65535.0).astype(np.uint16)
        filtered = cv2.medianBlur(scaled, ksize).astype(np.float32)
        return filtered / 65535.0 * (hi - lo) + lo


def build_threshold_hole(depth, args):
    depth = np.asarray(depth, dtype=np.float32)
    return (~np.isfinite(depth)) | (depth <= float(args.hole_depth_threshold))


def build_cleaned_hole_mask(depth, args):
    depth = np.asarray(depth, dtype=np.float32)
    threshold_hole = build_threshold_hole(depth, args)
    base_valid = (
        (~threshold_hole)
        & np.isfinite(depth)
        & (depth >= float(args.valid_min_depth))
        & (depth <= float(args.valid_max_depth))
    )
    if base_valid.sum() == 0:
        return threshold_hole, {"added_pixels": 0, "outlier_pixels": 0}

    median = robust_median_depth(depth, base_valid, args.clean_median_ksize)
    abs_residual = np.abs(depth - median)
    residual_values = abs_residual[base_valid & np.isfinite(abs_residual)]
    if residual_values.size == 0:
        robust_threshold = float(args.clean_outlier_abs)
    else:
        med = float(np.median(residual_values))
        mad = float(np.median(np.abs(residual_values - med)))
        robust_threshold = max(float(args.clean_outlier_abs), med + float(args.clean_outlier_mad_scale) * 1.4826 * mad)

    outlier = base_valid & np.isfinite(abs_residual) & (abs_residual > robust_threshold)
    near_existing_hole = dilate_bool(threshold_hole, max(1, int(args.clean_dilate) + 1))
    outlier &= near_existing_hole

    cleaned = threshold_hole | outlier
    cleaned = fill_small_mask_holes(cleaned, max_area=16)
    cleaned = dilate_bool(cleaned, int(args.clean_dilate))
    cleaned = remove_small_components(cleaned, int(args.clean_min_component_area))
    return cleaned, {
        "added_pixels": int(cleaned.sum() - threshold_hole.sum()),
        "outlier_pixels": int(outlier.sum()),
        "robust_threshold": float(robust_threshold),
    }


def box_count(mask, window):
    window = int(window)
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    kernel = np.ones((window, window), dtype=np.float32)
    return cv2.filter2D(mask.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)


def compute_speckle_candidates(depth, args):
    depth = np.asarray(depth, dtype=np.float32)
    threshold_hole = build_threshold_hole(depth, args)
    base_valid = (
        (~threshold_hole)
        & np.isfinite(depth)
        & (depth >= float(args.valid_min_depth))
        & (depth <= float(args.valid_max_depth))
    )
    if base_valid.sum() == 0:
        return threshold_hole, base_valid, None, None, None, float(args.speckle_residual_abs)

    median = robust_median_depth(depth, base_valid, args.clean_median_ksize)
    abs_residual = np.abs(depth - median)
    residual_values = abs_residual[base_valid & np.isfinite(abs_residual)]
    if residual_values.size == 0:
        robust_threshold = float(args.speckle_residual_abs)
    else:
        med = float(np.median(residual_values))
        mad = float(np.median(np.abs(residual_values - med)))
        robust_threshold = max(
            float(args.speckle_residual_abs),
            med + float(args.clean_outlier_mad_scale) * 1.4826 * mad,
        )

    candidate = base_valid & np.isfinite(abs_residual) & (abs_residual > robust_threshold)
    window = max(3, int(args.speckle_window))
    if window % 2 == 0:
        window += 1
    local_count = box_count(candidate, window)
    local_valid = np.maximum(box_count(base_valid, window), 1.0)
    density = local_count / local_valid
    dense_speckle = candidate & (density >= float(args.speckle_density_threshold))
    return threshold_hole, base_valid, abs_residual, candidate, dense_speckle, float(robust_threshold)


def build_speckle_cleaned_hole_mask(depth, args):
    threshold_hole, base_valid, _, candidate, dense_speckle, robust_threshold = compute_speckle_candidates(
        depth, args
    )
    if base_valid.sum() == 0:
        return threshold_hole, {
            "added_pixels": 0,
            "outlier_pixels": 0,
            "speckle_pixels": 0,
            "speckle_candidate_pixels": 0,
        }

    linked_seed = dilate_bool(dense_speckle, int(args.speckle_link_radius))
    linked_seed = fill_small_mask_holes(linked_seed, max_area=16)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        linked_seed.astype(np.uint8), connectivity=8
    )
    linked = np.zeros_like(linked_seed, dtype=bool)
    kept_components = 0
    rejected_large_components = 0
    min_area = int(args.speckle_min_component_area)
    max_area = int(args.speckle_max_component_area)
    max_bbox_side = int(args.speckle_max_bbox_side)
    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox_w = int(stats[label, cv2.CC_STAT_WIDTH])
        bbox_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        seed_count = int(dense_speckle[component].sum())
        if seed_count < min_area:
            continue
        too_large = area > max_area or max(bbox_w, bbox_h) > max_bbox_side
        if too_large:
            rejected_large_components += 1
            continue
        linked[component] = True
        kept_components += 1

    # Preserve the previous near-hole cleanup path; it handles dirty hole borders well.
    near_existing_hole = dilate_bool(threshold_hole, max(1, int(args.clean_dilate) + 1))
    near_hole_outlier = candidate & near_existing_hole

    cleaned = threshold_hole | linked | near_hole_outlier
    cleaned = fill_small_mask_holes(cleaned, max_area=16)
    cleaned = dilate_bool(cleaned, int(args.clean_dilate))
    cleaned = remove_small_components(cleaned, int(args.clean_min_component_area))
    return cleaned, {
        "added_pixels": int(cleaned.sum() - threshold_hole.sum()),
        "outlier_pixels": int(near_hole_outlier.sum()),
        "speckle_pixels": int(linked.sum()),
        "speckle_candidate_pixels": int(candidate.sum()),
        "speckle_dense_seed_pixels": int(dense_speckle.sum()),
        "speckle_kept_components": int(kept_components),
        "speckle_rejected_large_components": int(rejected_large_components),
        "robust_threshold": float(robust_threshold),
        "speckle_density_threshold": float(args.speckle_density_threshold),
    }


def build_amp_speckle_cleaned_hole_mask(depth, amplitude_mean, args):
    threshold_hole, base_valid, _, candidate, dense_speckle, robust_threshold = compute_speckle_candidates(
        depth, args
    )
    if base_valid.sum() == 0:
        return threshold_hole, {
            "added_pixels": 0,
            "outlier_pixels": 0,
            "speckle_pixels": 0,
            "speckle_candidate_pixels": 0,
            "amp_kept_components": 0,
            "amp_rejected_components": 0,
        }

    amplitude_mean = np.asarray(amplitude_mean, dtype=np.float32)
    linked_seed = dilate_bool(dense_speckle, int(args.speckle_link_radius))
    linked_seed = fill_small_mask_holes(linked_seed, max_area=16)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        linked_seed.astype(np.uint8), connectivity=8
    )
    linked = np.zeros_like(linked_seed, dtype=bool)
    kept_components = 0
    rejected_large_components = 0
    amp_kept_components = 0
    amp_rejected_components = 0
    min_area = int(args.speckle_min_component_area)
    max_area = int(args.speckle_max_component_area)
    max_bbox_side = int(args.speckle_max_bbox_side)
    ring_radius = int(args.speckle_amp_ring_radius)
    ring_kernel = np.ones((2 * ring_radius + 1, 2 * ring_radius + 1), dtype=np.uint8)
    amp_ratio_min = float(args.speckle_amp_ratio_min)
    amp_delta_min = float(args.speckle_amp_delta_min)
    amp_abs_min = float(args.speckle_amp_abs_min)

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox_w = int(stats[label, cv2.CC_STAT_WIDTH])
        bbox_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        seed = dense_speckle & component
        seed_count = int(seed.sum())
        if seed_count < min_area:
            continue
        too_large = area > max_area or max(bbox_w, bbox_h) > max_bbox_side
        if too_large:
            rejected_large_components += 1
            continue

        if ring_radius > 0:
            ring = cv2.dilate(component.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
            ring &= (~component) & base_valid
        else:
            ring = np.zeros_like(component, dtype=bool)

        seed_amp_mean = float(np.mean(amplitude_mean[seed])) if seed.any() else 0.0
        ring_amp_mean = float(np.mean(amplitude_mean[ring])) if ring.any() else 0.0
        amp_ratio = seed_amp_mean / max(ring_amp_mean, 1.0)
        amp_delta = seed_amp_mean - ring_amp_mean
        pass_amp = seed_amp_mean >= amp_abs_min and (amp_ratio >= amp_ratio_min or amp_delta >= amp_delta_min)
        if not pass_amp:
            amp_rejected_components += 1
            continue

        linked[component] = True
        kept_components += 1
        amp_kept_components += 1

    near_existing_hole = dilate_bool(threshold_hole, max(1, int(args.clean_dilate) + 1))
    near_hole_outlier = candidate & near_existing_hole

    cleaned = threshold_hole | linked | near_hole_outlier
    cleaned = fill_small_mask_holes(cleaned, max_area=16)
    cleaned = dilate_bool(cleaned, int(args.clean_dilate))
    cleaned = remove_small_components(cleaned, int(args.clean_min_component_area))
    return cleaned, {
        "added_pixels": int(cleaned.sum() - threshold_hole.sum()),
        "outlier_pixels": int(near_hole_outlier.sum()),
        "speckle_pixels": int(linked.sum()),
        "speckle_candidate_pixels": int(candidate.sum()),
        "speckle_dense_seed_pixels": int(dense_speckle.sum()),
        "speckle_kept_components": int(kept_components),
        "speckle_rejected_large_components": int(rejected_large_components),
        "amp_kept_components": int(amp_kept_components),
        "amp_rejected_components": int(amp_rejected_components),
        "robust_threshold": float(robust_threshold),
        "speckle_density_threshold": float(args.speckle_density_threshold),
        "speckle_amp_ratio_min": float(args.speckle_amp_ratio_min),
        "speckle_amp_delta_min": float(args.speckle_amp_delta_min),
        "speckle_amp_abs_min": float(args.speckle_amp_abs_min),
    }


def fit_plane_from_points(xs, ys, zs):
    a = np.stack([xs.astype(np.float32), ys.astype(np.float32), np.ones_like(xs, dtype=np.float32)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(a, zs.astype(np.float32), rcond=None)
    pred = a @ coeff
    residual = float(np.median(np.abs(pred - zs)))
    return coeff.astype(np.float32), residual


def plane_or_median_fill(depth, hole, args, model_pred=None):
    depth = np.asarray(depth, dtype=np.float32)
    hole = np.asarray(hole, dtype=bool)
    output = depth.copy()
    valid = (
        (~hole)
        & np.isfinite(depth)
        & (depth >= float(args.valid_min_depth))
        & (depth <= float(args.valid_max_depth))
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    rows = []
    image_h, image_w = hole.shape
    yy, xx = np.indices(hole.shape)
    ring_radius = int(args.plane_ring_radius)
    kernel = np.ones((2 * ring_radius + 1, 2 * ring_radius + 1), dtype=np.uint8)
    max_area = int(args.plane_max_component_area)

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        ring = dilated & (~component) & valid
        point_count = int(ring.sum())
        method = "none"
        residual = None
        if point_count > 0:
            zs = depth[ring].astype(np.float32)
            median_value = float(np.median(zs))
        else:
            zs = np.asarray([], dtype=np.float32)
            median_value = 0.0

        if area <= max_area and point_count >= int(args.plane_min_points):
            coeff, residual = fit_plane_from_points(xx[ring], yy[ring], zs)
            if residual <= float(args.plane_max_abs_residual):
                comp_x = xx[component].astype(np.float32)
                comp_y = yy[component].astype(np.float32)
                fill_values = coeff[0] * comp_x + coeff[1] * comp_y + coeff[2]
                method = "plane"
            else:
                fill_values = np.full(int(component.sum()), median_value, dtype=np.float32)
                method = "median_residual"
        elif point_count > 0:
            fill_values = np.full(int(component.sum()), median_value, dtype=np.float32)
            method = "median"
        elif model_pred is not None:
            fill_values = np.asarray(model_pred[component], dtype=np.float32)
            method = "model_no_ring"
        else:
            fill_values = np.full(int(component.sum()), 0.0, dtype=np.float32)
            method = "zero_no_ring"

        if model_pred is not None and float(args.plane_blend_model) > 0:
            alpha = float(np.clip(args.plane_blend_model, 0.0, 1.0))
            fill_values = (1.0 - alpha) * fill_values + alpha * np.asarray(model_pred[component], dtype=np.float32)

        fill_values = np.clip(fill_values, float(args.valid_min_depth), float(args.valid_max_depth)).astype(np.float32)
        output[component] = fill_values
        touches_border = bool(x == 0 or y == 0 or (x + w) >= image_w or (y + h) >= image_h)
        rows.append(
            {
                "label": int(label),
                "area": area,
                "bbox": [x, y, w, h],
                "ring_points": point_count,
                "method": method,
                "plane_residual": residual,
                "touches_border": touches_border,
            }
        )

    return output.astype(np.float32), rows


def locally_aligned_model_fill(depth, hole, pred, args):
    depth = np.asarray(depth, dtype=np.float32)
    hole = np.asarray(hole, dtype=bool)
    pred = np.asarray(pred, dtype=np.float32)
    output = np.where(hole, pred, depth).astype(np.float32)
    valid = (
        (~hole)
        & np.isfinite(depth)
        & np.isfinite(pred)
        & (depth >= float(args.valid_min_depth))
        & (depth <= float(args.valid_max_depth))
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    rows = []
    image_h, image_w = hole.shape
    ring_radius = int(args.align_ring_radius)
    kernel = np.ones((2 * ring_radius + 1, 2 * ring_radius + 1), dtype=np.uint8)
    max_area = int(args.align_max_component_area)

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = bool(x == 0 or y == 0 or (x + w) >= image_w or (y + h) >= image_h)
        method = "model"
        scale = 1.0
        offset = 0.0
        residual = None
        point_count = 0

        if area <= max_area and ring_radius > 0:
            dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
            ring = dilated & (~component) & valid
            point_count = int(ring.sum())
            if point_count >= int(args.align_min_points):
                ring_pred = pred[ring].astype(np.float32)
                ring_depth = depth[ring].astype(np.float32)
                pred_std = float(np.std(ring_pred))
                if pred_std >= float(args.align_min_pred_std):
                    design = np.stack([ring_pred, np.ones_like(ring_pred)], axis=1)
                    coeff, _, _, _ = np.linalg.lstsq(design, ring_depth, rcond=None)
                    fit_scale = float(coeff[0])
                    fit_offset = float(coeff[1])
                    fitted_ring = fit_scale * ring_pred + fit_offset
                    residual = float(np.median(np.abs(fitted_ring - ring_depth)))
                    scale_ok = float(args.align_min_scale) <= fit_scale <= float(args.align_max_scale)
                    residual_ok = residual <= float(args.align_max_abs_residual)
                    if scale_ok and residual_ok:
                        scale = fit_scale
                        offset = fit_offset
                        method = "affine"
                    else:
                        offset = float(np.median(ring_depth - ring_pred))
                        shifted_ring = ring_pred + offset
                        residual = float(np.median(np.abs(shifted_ring - ring_depth)))
                        if residual <= float(args.align_max_abs_residual):
                            method = "offset"
                        else:
                            offset = 0.0
                else:
                    offset = float(np.median(ring_depth - ring_pred))
                    shifted_ring = ring_pred + offset
                    residual = float(np.median(np.abs(shifted_ring - ring_depth)))
                    if residual <= float(args.align_max_abs_residual):
                        method = "offset_low_std"
                    else:
                        offset = 0.0

        fill_values = scale * pred[component] + offset
        fill_values = np.clip(fill_values, float(args.valid_min_depth), float(args.valid_max_depth)).astype(np.float32)
        output[component] = fill_values
        rows.append(
            {
                "label": int(label),
                "area": area,
                "bbox": [x, y, w, h],
                "ring_points": point_count,
                "method": method,
                "scale": float(scale),
                "offset": float(offset),
                "align_residual": residual,
                "touches_border": touches_border,
            }
        )

    return output.astype(np.float32), rows


def build_hybrid_hole_only(depth, hole, anchor, pred, args):
    hole = np.asarray(hole, dtype=bool)
    depth = np.asarray(depth, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    if args.hybrid_mode == "none":
        return None, None, []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    use_model_mask = np.zeros_like(hole, dtype=bool)
    component_rows = []
    image_h, image_w = hole.shape
    min_area = int(args.hybrid_component_min_area)
    min_radius = float(args.hybrid_component_min_radius)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component_mask = labels == label
        crop = component_mask[y : y + h, x : x + w]
        max_radius = component_max_radius(crop)
        touches_border = bool(x == 0 or y == 0 or (x + w) >= image_w or (y + h) >= image_h)
        bbox_area = max(1, w * h)
        fill_ratio = float(area) / float(bbox_area)
        component_diff = np.abs(pred[component_mask] - anchor[component_mask]).astype(np.float32)
        finite_diff = component_diff[np.isfinite(component_diff)]
        mean_abs_anchor_diff = float(np.mean(finite_diff)) if finite_diff.size else float("inf")
        p95_abs_anchor_diff = float(np.percentile(finite_diff, 95.0)) if finite_diff.size else float("inf")

        use_model = area >= min_area and max_radius >= min_radius
        if args.hybrid_keep_border_anchor and touches_border:
            use_model = False
        max_mean_diff = float(getattr(args, "hybrid_max_mean_abs_anchor_diff", 0.0))
        if max_mean_diff > 0.0 and mean_abs_anchor_diff > max_mean_diff:
            use_model = False
        max_p95_diff = float(getattr(args, "hybrid_max_p95_abs_anchor_diff", 0.0))
        if max_p95_diff > 0.0 and p95_abs_anchor_diff > max_p95_diff:
            use_model = False
        if use_model:
            use_model_mask[component_mask] = True

        component_rows.append(
            {
                "label": int(label),
                "area": area,
                "bbox": [x, y, w, h],
                "max_radius": max_radius,
                "fill_ratio": fill_ratio,
                "touches_border": touches_border,
                "mean_abs_anchor_diff": mean_abs_anchor_diff,
                "p95_abs_anchor_diff": p95_abs_anchor_diff,
                "use_model": bool(use_model),
            }
        )

    hybrid_fill = np.where(use_model_mask, pred, anchor).astype(np.float32)
    hybrid_hole_only = np.where(hole, hybrid_fill, depth).astype(np.float32)
    return hybrid_hole_only, use_model_mask, component_rows


def build_anchor_gated_hole_only(depth, hole, anchor, pred, args):
    if not bool(getattr(args, "gated_fill", False)):
        return None, None, []

    depth = np.asarray(depth, dtype=np.float32)
    hole = np.asarray(hole, dtype=bool)
    anchor = np.asarray(anchor, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    soft = float(getattr(args, "gate_diff_soft", 0.02))
    hard = float(getattr(args, "gate_diff_hard", 0.08))
    if hard <= soft:
        hard = soft + 1e-6

    diff = np.abs(pred - anchor).astype(np.float32)
    gate_weight = np.clip((hard - diff) / (hard - soft), 0.0, 1.0).astype(np.float32)
    gate_weight[~hole] = 0.0
    gate_weight[~np.isfinite(diff)] = 0.0

    component_rows = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    image_h, image_w = hole.shape
    max_mean_diff = float(getattr(args, "gate_component_max_mean_abs_diff", 0.0))
    max_p95_diff = float(getattr(args, "gate_component_max_p95_abs_diff", 0.0))
    keep_border_anchor = bool(getattr(args, "gate_keep_border_anchor", False))

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = bool(x == 0 or y == 0 or (x + w) >= image_w or (y + h) >= image_h)
        values = diff[component]
        values = values[np.isfinite(values)]
        mean_abs_anchor_diff = float(np.mean(values)) if values.size else float("inf")
        p95_abs_anchor_diff = float(np.percentile(values, 95.0)) if values.size else float("inf")

        force_anchor = False
        if keep_border_anchor and touches_border:
            force_anchor = True
        if max_mean_diff > 0.0 and mean_abs_anchor_diff > max_mean_diff:
            force_anchor = True
        if max_p95_diff > 0.0 and p95_abs_anchor_diff > max_p95_diff:
            force_anchor = True
        if force_anchor:
            gate_weight[component] = 0.0

        component_weights = gate_weight[component]
        component_rows.append(
            {
                "label": int(label),
                "area": area,
                "bbox": [x, y, w, h],
                "touches_border": touches_border,
                "mean_abs_anchor_diff": mean_abs_anchor_diff,
                "p95_abs_anchor_diff": p95_abs_anchor_diff,
                "mean_gate_weight": float(np.mean(component_weights)) if component_weights.size else 0.0,
                "model_pixel_ratio": float(np.mean(component_weights > 0.5)) if component_weights.size else 0.0,
                "force_anchor": bool(force_anchor),
            }
        )

    gated_fill = anchor + gate_weight * (pred - anchor)
    gated_hole_only = np.where(hole, gated_fill, depth).astype(np.float32)
    return gated_hole_only, gate_weight.astype(np.float32), component_rows


def build_repair_hole_mask(hole, args):
    hole = np.asarray(hole, dtype=bool)
    if str(getattr(args, "repair_mask_mode", "all")) == "all" or not hole.any():
        return hole.copy(), np.zeros_like(hole, dtype=bool), []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=8)
    repair_hole = np.zeros_like(hole, dtype=bool)
    preserved_hole = np.zeros_like(hole, dtype=bool)
    rows = []
    image_h, image_w = hole.shape
    border_min_area = int(getattr(args, "preserve_border_hole_min_area", 1024))
    large_min_area = int(getattr(args, "preserve_large_hole_min_area", 24000))
    max_bbox_side = int(getattr(args, "preserve_hole_max_bbox_side", 220))

    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = bool(x == 0 or y == 0 or (x + w) >= image_w or (y + h) >= image_h)
        preserve_reason = None
        if touches_border and area >= border_min_area:
            preserve_reason = "border"
        if large_min_area > 0 and area >= large_min_area:
            preserve_reason = "large_area" if preserve_reason is None else f"{preserve_reason}+large_area"
        if max_bbox_side > 0 and max(w, h) >= max_bbox_side:
            preserve_reason = "large_bbox" if preserve_reason is None else f"{preserve_reason}+large_bbox"

        if preserve_reason is None:
            repair_hole[component] = True
            action = "repair"
        else:
            preserved_hole[component] = True
            action = "preserve"

        rows.append(
            {
                "label": int(label),
                "area": area,
                "bbox": [x, y, w, h],
                "touches_border": touches_border,
                "action": action,
                "preserve_reason": preserve_reason,
            }
        )

    return repair_hole, preserved_hole, rows


def apply_preserved_holes(output, preserved_hole, args):
    output = np.asarray(output, dtype=np.float32).copy()
    preserved_hole = np.asarray(preserved_hole, dtype=bool)
    if preserved_hole.any() and bool(getattr(args, "preserve_holes_as_nan", False)):
        output[preserved_hole] = np.nan
    return output.astype(np.float32)


def fill_threshold_then_added(depth, threshold_hole, full_hole, pred, anchor, args):
    depth = np.asarray(depth, dtype=np.float32)
    threshold_hole = np.asarray(threshold_hole, dtype=bool)
    full_hole = np.asarray(full_hole, dtype=bool)
    pred = np.asarray(pred, dtype=np.float32)

    base_threshold = np.where(threshold_hole, pred, depth).astype(np.float32)
    added_only = full_hole & (~threshold_hole)
    if not added_only.any():
        return base_threshold, []

    if args.split_added_mode == "plane":
        split_hole_only, added_rows = plane_or_median_fill(
            base_threshold,
            added_only,
            args,
            model_pred=pred,
        )
        return split_hole_only.astype(np.float32), added_rows

    if args.split_added_mode == "anchor_ns":
        split_fill = opencv_depth_inpaint(
            np.asarray(anchor, dtype=np.float32),
            added_only,
            method="ns",
            radius=max(1, int(args.split_added_inpaint_radius)),
        ).astype(np.float32)
        split_hole_only = depth.copy().astype(np.float32)
        split_hole_only[threshold_hole] = pred[threshold_hole]
        split_hole_only[added_only] = split_fill[added_only]
        added_rows = [
            {
                "method": "anchor_ns",
                "radius": max(1, int(args.split_added_inpaint_radius)),
                "area": int(added_only.sum()),
            }
        ]
        return split_hole_only.astype(np.float32), added_rows

    radius = max(1, int(args.split_added_inpaint_radius))
    split_fill = opencv_depth_inpaint(
        base_threshold,
        added_only,
        method="ns",
        radius=radius,
    ).astype(np.float32)
    split_hole_only = np.where(added_only, split_fill, base_threshold).astype(np.float32)
    added_rows = [
        {
            "method": "ns",
            "radius": radius,
            "area": int(added_only.sum()),
        }
    ]
    return split_hole_only, added_rows


def save_visualization(
    path,
    name,
    depth,
    hole,
    anchor,
    pred,
    hole_only,
    split_hole_only=None,
    hybrid_hole_only=None,
    hybrid_use_model_mask=None,
    plane_hole_only=None,
    aligned_hole_only=None,
    gated_hole_only=None,
    gated_weight=None,
    repair_hole=None,
    preserved_hole=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(path))
    limit_arrays = [depth[~hole], anchor, pred, hole_only]
    if split_hole_only is not None:
        limit_arrays.append(split_hole_only)
    if hybrid_hole_only is not None:
        limit_arrays.append(hybrid_hole_only)
    if plane_hole_only is not None:
        limit_arrays.append(plane_hole_only)
    if aligned_hole_only is not None:
        limit_arrays.append(aligned_hole_only)
    if gated_hole_only is not None:
        limit_arrays.append(gated_hole_only)
    vmin, vmax = image_limits(*limit_arrays)
    delta = np.abs(pred - anchor)
    dmax = float(np.percentile(delta[np.isfinite(delta)], 98.0)) if np.isfinite(delta).any() else 1.0
    dmax = max(dmax, 1e-6)

    if hybrid_hole_only is None or hybrid_use_model_mask is None:
        panels = [
            ("raw depth", depth, "viridis", vmin, vmax),
            ("hole mask", hole.astype(np.float32), "gray", 0.0, 1.0),
            ("NS anchor", anchor, "viridis", vmin, vmax),
            ("flow restored", pred, "viridis", vmin, vmax),
            ("hole-only blend", hole_only, "viridis", vmin, vmax),
            ("|flow-anchor|", delta, "magma", 0.0, dmax),
        ]
        if split_hole_only is not None:
            panels.insert(5, ("split-added hole-only", split_hole_only, "viridis", vmin, vmax))
        if plane_hole_only is not None:
            panels.insert(5, ("plane hole-only", plane_hole_only, "viridis", vmin, vmax))
        if aligned_hole_only is not None:
            panels.insert(5, ("aligned hole-only", aligned_hole_only, "viridis", vmin, vmax))
        if gated_hole_only is not None:
            panels.insert(5, ("gated hole-only", gated_hole_only, "viridis", vmin, vmax))
            if gated_weight is not None:
                panels.insert(6, ("gate weight", gated_weight.astype(np.float32), "gray", 0.0, 1.0))
        if repair_hole is not None:
            panels.insert(2, ("repair mask", repair_hole.astype(np.float32), "gray", 0.0, 1.0))
        if preserved_hole is not None and np.asarray(preserved_hole, dtype=bool).any():
            panels.insert(3, ("preserved hole", preserved_hole.astype(np.float32), "gray", 0.0, 1.0))
        ncols = int(np.ceil(len(panels) / 2.0))
        fig, axes = plt.subplots(2, ncols, figsize=(4.8 * ncols, 8), constrained_layout=True)
    else:
        panels = [
            ("raw depth", depth, "viridis", vmin, vmax),
            ("hole mask", hole.astype(np.float32), "gray", 0.0, 1.0),
            ("NS anchor", anchor, "viridis", vmin, vmax),
            ("flow restored", pred, "viridis", vmin, vmax),
            ("model hole-only", hole_only, "viridis", vmin, vmax),
            ("hybrid hole-only", hybrid_hole_only, "viridis", vmin, vmax),
            ("|flow-anchor|", delta, "magma", 0.0, dmax),
            ("hybrid use-model", hybrid_use_model_mask.astype(np.float32), "gray", 0.0, 1.0),
        ]
        if split_hole_only is not None:
            panels.insert(6, ("split-added hole-only", split_hole_only, "viridis", vmin, vmax))
        if aligned_hole_only is not None:
            panels.insert(6, ("aligned hole-only", aligned_hole_only, "viridis", vmin, vmax))
        if gated_hole_only is not None:
            panels.insert(6, ("gated hole-only", gated_hole_only, "viridis", vmin, vmax))
            if gated_weight is not None:
                panels.insert(7, ("gate weight", gated_weight.astype(np.float32), "gray", 0.0, 1.0))
        if repair_hole is not None:
            panels.insert(2, ("repair mask", repair_hole.astype(np.float32), "gray", 0.0, 1.0))
        if preserved_hole is not None and np.asarray(preserved_hole, dtype=bool).any():
            panels.insert(3, ("preserved hole", preserved_hole.astype(np.float32), "gray", 0.0, 1.0))
        ncols = int(np.ceil(len(panels) / 2.0))
        fig, axes = plt.subplots(2, ncols, figsize=(4.8 * ncols, 8), constrained_layout=True)

    for ax, (title, image, cmap_name, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap_name, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(name)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def collect_pairs(raw_dir, depth_dir):
    raw_paths = {
        os.path.splitext(os.path.basename(path))[0]: path for path in glob(os.path.join(raw_dir, "*.npy"))
    }
    depth_paths = {
        os.path.splitext(os.path.basename(path))[0]: path for path in glob(os.path.join(depth_dir, "*.npy"))
    }
    stems = sorted(set(raw_paths) & set(depth_paths), key=natural_key)
    return [(stem, raw_paths[stem], depth_paths[stem]) for stem in stems]


def filter_pairs_by_samples(pairs, samples):
    if not samples:
        return pairs
    wanted = {str(sample) for sample in samples}
    filtered = [pair for pair in pairs if pair[0] in wanted]
    found = {pair[0] for pair in filtered}
    missing = sorted(wanted - found, key=natural_key)
    if missing:
        raise FileNotFoundError(f"Requested samples not found in paired raw/depth data: {missing}")
    return filtered


def depth_to_meters(depth, unit="auto"):
    depth = np.asarray(depth, dtype=np.float32)
    unit = str(unit)
    if unit == "m":
        return depth
    if unit == "mm":
        return depth / 1000.0
    if unit != "auto":
        raise ValueError(f"Unknown depth unit: {unit}")

    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return depth
    median = float(np.median(finite))
    p95 = float(np.percentile(finite, 95.0))
    if median > 20.0 or p95 > 100.0:
        return depth / 1000.0
    return depth


def build_model(checkpoint, ckpt_args, device):
    condition_channels = 4 + int(bool(ckpt_args.get("include_hole_distance", False))) + 4
    time_channels = int(ckpt_args.get("time_channels", 16))
    in_channels = flow_model_in_channels(condition_channels, time_channels)
    model = build_depth_backbone(
        ckpt_args.get("backbone", "resunet"),
        in_channels=in_channels,
        base_channels=int(ckpt_args.get("base_channels", 32)),
        out_channels=1,
        res_blocks=int(ckpt_args.get("res_blocks", 2)),
        transformer_layers=int(ckpt_args.get("transformer_layers", 2)),
        transformer_heads=int(ckpt_args.get("transformer_heads", 8)),
        transformer_mlp_ratio=float(ckpt_args.get("transformer_mlp_ratio", 4.0)),
        transformer_pool=int(ckpt_args.get("transformer_pool", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def raw9_to_amplitude(raw9, mode):
    raw9 = raw9.astype(np.float32)
    if raw9.shape[0] != 9:
        raise ValueError(f"Expected raw shape (9,H,W), got {raw9.shape}")
    if mode == "iq6":
        i_channels = np.stack([raw9[0], raw9[2], raw9[4]], axis=0)
        q_channels = np.stack([raw9[1], raw9[3], raw9[5]], axis=0)
        amplitude = np.sqrt(i_channels**2 + q_channels**2).astype(np.float32)
    elif mode == "raw_258":
        amplitude = raw9[[2, 5, 8]].astype(np.float32)
    else:
        raise ValueError(f"Unknown amplitude_mode: {mode}")
    amplitude = np.nan_to_num(amplitude, nan=0.0, neginf=0.0, posinf=0.0)
    amplitude = np.maximum(amplitude, 0.0)
    amplitude_mean = amplitude.mean(axis=0).astype(np.float32)
    return amplitude, amplitude_mean


def apply_raw9_spatial_transform(raw9, mode):
    mode = str(mode or "none")
    raw9 = np.asarray(raw9, dtype=np.float32)
    if raw9.ndim != 3 or raw9.shape[0] != 9:
        raise ValueError(f"Expected raw shape (9,H,W), got {raw9.shape}")
    if mode == "none":
        return raw9
    if mode == "flip_lr":
        return np.flip(raw9, axis=-1).copy()
    if mode == "flip_ud":
        return np.flip(raw9, axis=-2).copy()
    if mode == "rot180":
        return np.flip(np.flip(raw9, axis=-1), axis=-2).copy()
    raise ValueError(f"Unknown raw9 transform: {mode}")


def robust_unit_image(image, valid_mask):
    image = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(image)
    if int(valid.sum()) < 32:
        return np.zeros_like(image, dtype=np.float32), valid

    values = image[valid]
    lo, hi = np.percentile(values, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi - lo) < 1e-6:
        lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or float(hi - lo) < 1e-6:
        return np.zeros_like(image, dtype=np.float32), valid

    unit = ((image - float(lo)) / float(hi - lo)).astype(np.float32)
    unit = np.clip(unit, 0.0, 1.0)
    fill = float(np.median(unit[valid]))
    unit[~valid] = fill
    unit[~np.isfinite(unit)] = fill
    return unit, valid


def edge_magnitude(image):
    image = np.asarray(image, dtype=np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)


def masked_corrcoef(a, b, mask):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 32:
        return 0.0
    av = np.asarray(a, dtype=np.float32)[valid].astype(np.float64)
    bv = np.asarray(b, dtype=np.float32)[valid].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(av * bv) / denom)


def estimate_raw9_transform(raw9, depth, reliable_mask, amplitude_mode):
    depth_unit, valid = robust_unit_image(depth, reliable_mask)
    corr_mask = valid
    if int(corr_mask.sum()) >= 128:
        eroded = cv2.erode(corr_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        if int(eroded.sum()) >= 128:
            corr_mask = eroded
    depth_edge = edge_magnitude(depth_unit)

    scores = {}
    for mode in RAW9_SPATIAL_TRANSFORMS:
        transformed = apply_raw9_spatial_transform(raw9, mode)
        _, amplitude_mean = raw9_to_amplitude(transformed, amplitude_mode)
        amp_unit, amp_valid = robust_unit_image(np.log1p(np.maximum(amplitude_mean, 0.0)), corr_mask)
        scores[mode] = masked_corrcoef(depth_edge, edge_magnitude(amp_unit), corr_mask & amp_valid)
    best_mode = max(scores, key=lambda key: (scores[key], key == "none"))
    return best_mode, scores


def effective_raw9_transform(args):
    transform = str(getattr(args, "raw9_transform_effective", getattr(args, "raw9_transform", "none")) or "none")
    if transform == "checkpoint":
        transform = str(getattr(args, "checkpoint_raw9_transform", "none") or "none")
    return transform


def align_raw9_to_depth(raw9, depth, args):
    requested = effective_raw9_transform(args)
    if requested == "auto":
        reliable = (
            np.isfinite(depth)
            & (depth > float(args.hole_depth_threshold))
            & (depth >= float(args.valid_min_depth))
            & (depth <= float(args.valid_max_depth))
        )
        best_mode, scores = estimate_raw9_transform(raw9, depth, reliable, args.amplitude_mode)
        info = {
            "raw9_transform": requested,
            "raw9_transform_estimated": best_mode,
        }
        for mode, score in scores.items():
            info[f"raw9_transform_score_{mode}"] = float(score)
        return apply_raw9_spatial_transform(raw9, best_mode), info

    return apply_raw9_spatial_transform(raw9, requested), {
        "raw9_transform": requested,
        "raw9_transform_estimated": requested,
    }


def make_condition(depth, raw9, ckpt_args, args):
    depth = np.asarray(depth, dtype=np.float32)
    raw9 = np.asarray(raw9, dtype=np.float32)
    raw9, raw9_align_info = align_raw9_to_depth(raw9, depth, args)
    amplitude, amplitude_mean = raw9_to_amplitude(raw9, args.amplitude_mode)

    if args.hole_mask_mode == "cleaned":
        hole, mask_diagnostics = build_cleaned_hole_mask(depth, args)
    elif args.hole_mask_mode == "speckle_cleaned":
        hole, mask_diagnostics = build_speckle_cleaned_hole_mask(depth, args)
    elif args.hole_mask_mode == "amp_speckle_cleaned":
        hole, mask_diagnostics = build_amp_speckle_cleaned_hole_mask(depth, amplitude_mean, args)
    else:
        hole = build_threshold_hole(depth, args)
        mask_diagnostics = {"added_pixels": 0, "outlier_pixels": 0}
    confidence = (~hole).astype(np.float32)
    reliable = (
        (~hole)
        & np.isfinite(depth)
        & (depth >= float(args.valid_min_depth))
        & (depth <= float(args.valid_max_depth))
    )

    ckpt_radius = ckpt_args.get("anchor_inpaint_radius", 15)
    if ckpt_radius is None:
        ckpt_radius = 15
    radius = int(args.anchor_inpaint_radius) if args.anchor_inpaint_radius is not None else int(ckpt_radius)
    anchor = opencv_depth_inpaint(depth, hole, method="ns", radius=radius).astype(np.float32)

    stat_mask = (
        (~hole)
        & np.isfinite(anchor)
        & (anchor > float(args.valid_min_depth))
        & (anchor < float(args.valid_max_depth))
    )
    if stat_mask.sum() == 0:
        stat_mask = reliable & np.isfinite(anchor)
    if stat_mask.sum() > 0:
        lo, hi = np.percentile(anchor[stat_mask], ckpt_args.get("norm_percentiles", [5.0, 95.0]))
        center = float(np.median(anchor[stat_mask]))
        scale = float(hi - lo)
    else:
        center = 0.0
        scale = 1.0
    scale = max(scale, float(ckpt_args.get("min_depth_scale", 0.25)))

    clip_norm_depth = float(ckpt_args.get("clip_norm_depth", 8.0))
    anchor_norm = normalize_depth(anchor, center, scale, clip_norm_depth)
    noisy_norm = normalize_depth(depth, center, scale, clip_norm_depth)

    amplitude = amplitude.copy()
    amplitude_mean = amplitude_mean.copy()
    hole_amplitude_mode = str(getattr(args, "hole_amplitude_mode", "zero"))
    if hole_amplitude_mode == "zero":
        amplitude[:, hole] = 0.0
        amplitude_mean[hole] = 0.0
    elif hole_amplitude_mode == "keep_all":
        pass
    else:
        raise ValueError(f"Unknown hole_amplitude_mode: {hole_amplitude_mode}")

    channels = [
        anchor_norm,
        noisy_norm,
        hole.astype(np.float32),
        confidence,
    ]
    if bool(ckpt_args.get("include_hole_distance", False)):
        dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        dist = np.clip(dist / max(float(radius), 1.0), 0.0, 1.0)
        channels.append(dist)
    channels.extend(
        robust_nonnegative_channels(
            amplitude,
            reliable,
            percentile=ckpt_args.get("feature_percentile", 99.0),
            clip=ckpt_args.get("feature_clip", 3.0),
        )
    )
    channels.extend(
        robust_nonnegative_channels(
            amplitude_mean,
            reliable,
            percentile=ckpt_args.get("feature_percentile", 99.0),
            clip=ckpt_args.get("feature_clip", 3.0),
        )
    )

    x = np.stack(channels, axis=0).astype(np.float32)
    return {
        "x": torch.from_numpy(x[None]),
        "anchor_norm": torch.from_numpy(anchor_norm[None, None]),
        "center": torch.tensor([center], dtype=torch.float32),
        "scale": torch.tensor([scale], dtype=torch.float32),
        "depth": depth,
        "anchor": anchor,
        "hole": hole,
        "mask_diagnostics": mask_diagnostics,
        "reliable": reliable,
        "amplitude": amplitude,
        "amplitude_mean": amplitude_mean,
        "raw9_transform": raw9_align_info.get("raw9_transform", effective_raw9_transform(args)),
        "raw9_transform_estimated": raw9_align_info.get(
            "raw9_transform_estimated",
            raw9_align_info.get("raw9_transform", effective_raw9_transform(args)),
        ),
        "raw9_transform_scores": {
            key.replace("raw9_transform_score_", ""): value
            for key, value in raw9_align_info.items()
            if key.startswith("raw9_transform_score_")
        },
        "center_value": center,
        "scale_value": scale,
        "radius": radius,
    }


def summarize_rows(rows):
    if not rows:
        return {}

    def collect(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return np.asarray(values, dtype=np.float64)

    summary = {"num_samples": len(rows)}
    for key in [
        "hole_ratio",
        "repair_hole_ratio",
        "preserved_hole_ratio",
        "valid_ratio",
        "threshold_hole_ratio",
        "cleaned_added_ratio",
        "mean_abs_model_anchor_hole",
        "mean_abs_model_anchor_valid",
        "mean_abs_model_raw_valid",
        "mean_abs_hole_only_anchor_hole",
        "mean_abs_split_anchor_hole",
        "mean_abs_plane_anchor_hole",
        "mean_abs_aligned_anchor_hole",
        "hybrid_model_hole_ratio",
        "mean_abs_hybrid_anchor_hole",
        "mean_abs_gated_anchor_hole",
        "gated_mean_weight_hole",
        "gated_model_pixel_ratio",
    ]:
        values = collect(key)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
    return summary


def main():
    args = parse_args()
    pairs = filter_pairs_by_samples(collect_pairs(args.raw_dir, args.depth_dir), args.samples)
    if not pairs:
        raise FileNotFoundError(f"No paired .npy files found under {args.raw_dir} and {args.depth_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("input_mode") != "noisy_amp":
        raise ValueError(
            f"Real raw9 inference expects input_mode='noisy_amp', got {ckpt_args.get('input_mode')!r}."
        )
    checkpoint_raw9_transform = str(ckpt_args.get("raw9_transform", "none") or "none")
    args.checkpoint_raw9_transform = checkpoint_raw9_transform
    args.raw9_transform_effective = (
        checkpoint_raw9_transform
        if str(args.raw9_transform) == "checkpoint"
        else str(args.raw9_transform)
    )
    model = build_model(ckpt, ckpt_args, device)

    sampling_mode = args.sampling_mode or ckpt_args.get("eval_sampling_mode", "endpoint")
    sample_steps = int(args.sample_steps or ckpt_args.get("sample_steps", 8))

    for subdir in [
        "restored",
        "restored_raw",
        "anchor",
        "hole_only",
        "hole_mask",
        "repair_hole_mask",
        "preserved_hole_mask",
        "threshold_hole_mask",
        "split_hole_only",
        "plane_hole_only",
        "aligned_hole_only",
        "hybrid_hole_only",
        "hybrid_use_model_mask",
        "gated_hole_only",
        "gated_weight",
        "visualizations",
    ]:
        ensure_dir(os.path.join(args.output_dir, subdir))

    rows = []
    for index, (stem, raw_path, depth_path) in enumerate(pairs):
        raw9 = np.load(raw_path).astype(np.float32)
        depth = depth_to_meters(np.load(depth_path), args.depth_unit).astype(np.float32)
        if raw9.shape != (9,) + depth.shape:
            raise ValueError(f"Shape mismatch for {stem}: raw {raw9.shape}, depth {depth.shape}")

        condition_cpu = make_condition(depth, raw9, ckpt_args, args)
        condition = move_condition_to_device(condition_cpu, device)
        pred = predict_depth(model, condition, ckpt_args, sampling_mode, sample_steps)
        pred_raw_np = pred.detach().cpu().numpy()[0, 0].astype(np.float32)

        hole = condition_cpu["hole"]
        repair_hole, preserved_hole, repair_components = build_repair_hole_mask(hole, args)
        threshold_hole = build_threshold_hole(depth, args)
        threshold_repair_hole = threshold_hole & repair_hole
        reliable = condition_cpu["reliable"]
        anchor = condition_cpu["anchor"]
        pred_np, clip_bounds = clip_prediction(pred_raw_np, depth, reliable, args)
        hole_only = np.where(repair_hole, pred_np, depth).astype(np.float32)
        hole_only = apply_preserved_holes(hole_only, preserved_hole, args)
        plane_hole_only = None
        plane_components = []
        aligned_hole_only = None
        aligned_components = []
        split_hole_only = None
        split_added_components = []
        if args.split_added_fill:
            split_hole_only, split_added_components = fill_threshold_then_added(
                depth,
                threshold_repair_hole,
                repair_hole,
                pred_np,
                anchor,
                args,
            )
            split_hole_only = apply_preserved_holes(split_hole_only, preserved_hole, args)
        if args.plane_fill:
            plane_hole_only, plane_components = plane_or_median_fill(depth, repair_hole, args, model_pred=pred_np)
            plane_hole_only = apply_preserved_holes(plane_hole_only, preserved_hole, args)
        if args.aligned_fill:
            aligned_hole_only, aligned_components = locally_aligned_model_fill(depth, repair_hole, pred_np, args)
            aligned_hole_only = apply_preserved_holes(aligned_hole_only, preserved_hole, args)
        hybrid_hole_only, hybrid_use_model_mask, hybrid_components = build_hybrid_hole_only(
            depth,
            repair_hole,
            anchor,
            pred_np,
            args,
        )
        if hybrid_hole_only is not None:
            hybrid_hole_only = apply_preserved_holes(hybrid_hole_only, preserved_hole, args)
        gated_hole_only, gated_weight, gated_components = build_anchor_gated_hole_only(
            depth,
            repair_hole,
            anchor,
            pred_np,
            args,
        )
        if gated_hole_only is not None:
            gated_hole_only = apply_preserved_holes(gated_hole_only, preserved_hole, args)

        restored_path = os.path.join(args.output_dir, "restored", f"{stem}_restored.npy")
        restored_raw_path = os.path.join(args.output_dir, "restored_raw", f"{stem}_restored_raw.npy")
        anchor_path = os.path.join(args.output_dir, "anchor", f"{stem}_anchor.npy")
        hole_only_path = os.path.join(args.output_dir, "hole_only", f"{stem}_hole_only.npy")
        hole_mask_path = os.path.join(args.output_dir, "hole_mask", f"{stem}_hole_mask.npy")
        repair_hole_mask_path = os.path.join(args.output_dir, "repair_hole_mask", f"{stem}_repair_hole_mask.npy")
        preserved_hole_mask_path = os.path.join(
            args.output_dir, "preserved_hole_mask", f"{stem}_preserved_hole_mask.npy"
        )
        threshold_hole_mask_path = os.path.join(
            args.output_dir, "threshold_hole_mask", f"{stem}_threshold_hole_mask.npy"
        )
        plane_hole_only_path = os.path.join(args.output_dir, "plane_hole_only", f"{stem}_plane_hole_only.npy")
        aligned_hole_only_path = os.path.join(
            args.output_dir, "aligned_hole_only", f"{stem}_aligned_hole_only.npy"
        )
        split_hole_only_path = os.path.join(args.output_dir, "split_hole_only", f"{stem}_split_hole_only.npy")
        hybrid_hole_only_path = os.path.join(args.output_dir, "hybrid_hole_only", f"{stem}_hybrid_hole_only.npy")
        hybrid_use_model_mask_path = os.path.join(
            args.output_dir, "hybrid_use_model_mask", f"{stem}_hybrid_use_model_mask.npy"
        )
        gated_hole_only_path = os.path.join(args.output_dir, "gated_hole_only", f"{stem}_gated_hole_only.npy")
        gated_weight_path = os.path.join(args.output_dir, "gated_weight", f"{stem}_gated_weight.npy")
        np.save(restored_path, pred_np)
        np.save(restored_raw_path, pred_raw_np)
        np.save(anchor_path, anchor)
        np.save(hole_only_path, hole_only)
        np.save(hole_mask_path, hole.astype(np.uint8))
        np.save(repair_hole_mask_path, repair_hole.astype(np.uint8))
        np.save(preserved_hole_mask_path, preserved_hole.astype(np.uint8))
        np.save(threshold_hole_mask_path, threshold_hole.astype(np.uint8))
        if split_hole_only is not None:
            np.save(split_hole_only_path, split_hole_only)
        if plane_hole_only is not None:
            np.save(plane_hole_only_path, plane_hole_only)
        if aligned_hole_only is not None:
            np.save(aligned_hole_only_path, aligned_hole_only)
        if hybrid_hole_only is not None and hybrid_use_model_mask is not None:
            np.save(hybrid_hole_only_path, hybrid_hole_only)
            np.save(hybrid_use_model_mask_path, hybrid_use_model_mask.astype(np.uint8))
        if gated_hole_only is not None and gated_weight is not None:
            np.save(gated_hole_only_path, gated_hole_only)
            np.save(gated_weight_path, gated_weight)

        diff_model_anchor = pred_np - anchor
        diff_model_raw = pred_np - depth
        diff_split_anchor = None if split_hole_only is None else split_hole_only - anchor
        diff_hybrid_anchor = None if hybrid_hole_only is None else hybrid_hole_only - anchor
        diff_gated_anchor = None if gated_hole_only is None else gated_hole_only - anchor
        diff_plane_anchor = None if plane_hole_only is None else plane_hole_only - anchor
        diff_aligned_anchor = None if aligned_hole_only is None else aligned_hole_only - anchor
        hybrid_model_hole_ratio = None
        if hybrid_use_model_mask is not None and hole.any():
            hybrid_model_hole_ratio = float(hybrid_use_model_mask[repair_hole].mean()) if repair_hole.any() else 0.0
        gated_mean_weight_hole = None
        gated_model_pixel_ratio = None
        if gated_weight is not None and repair_hole.any():
            gated_mean_weight_hole = float(np.mean(gated_weight[repair_hole]))
            gated_model_pixel_ratio = float(np.mean(gated_weight[repair_hole] > 0.5))
        threshold_count = int(threshold_hole.sum())
        cleaned_count = int(hole.sum())
        repair_count = int(repair_hole.sum())
        preserved_count = int(preserved_hole.sum())
        added_count = max(0, cleaned_count - threshold_count)
        row = {
            "name": stem,
            "raw_path": raw_path,
            "depth_path": depth_path,
            "restored_path": restored_path,
            "restored_raw_path": restored_raw_path,
            "anchor_path": anchor_path,
            "hole_only_path": hole_only_path,
            "hole_mask_path": hole_mask_path,
            "repair_hole_mask_path": repair_hole_mask_path,
            "preserved_hole_mask_path": preserved_hole_mask_path,
            "threshold_hole_mask_path": threshold_hole_mask_path,
            "split_hole_only_path": split_hole_only_path if split_hole_only is not None else None,
            "plane_hole_only_path": plane_hole_only_path if plane_hole_only is not None else None,
            "aligned_hole_only_path": aligned_hole_only_path if aligned_hole_only is not None else None,
            "hybrid_hole_only_path": hybrid_hole_only_path if hybrid_hole_only is not None else None,
            "hybrid_use_model_mask_path": (
                hybrid_use_model_mask_path if hybrid_use_model_mask is not None else None
            ),
            "gated_hole_only_path": gated_hole_only_path if gated_hole_only is not None else None,
            "gated_weight_path": gated_weight_path if gated_weight is not None else None,
            "shape": list(depth.shape),
            "raw_shape": list(raw9.shape),
            "amplitude_mode": args.amplitude_mode,
            "depth_unit": args.depth_unit,
            "raw9_transform": condition_cpu.get("raw9_transform", args.raw9_transform_effective),
            "raw9_transform_estimated": condition_cpu.get(
                "raw9_transform_estimated",
                condition_cpu.get("raw9_transform", args.raw9_transform_effective),
            ),
            "raw9_transform_scores": condition_cpu.get("raw9_transform_scores"),
            "hole_ratio": float(hole.mean()),
            "repair_hole_ratio": float(repair_hole.mean()),
            "preserved_hole_ratio": float(preserved_hole.mean()),
            "threshold_hole_ratio": float(threshold_hole.mean()),
            "cleaned_added_ratio": float(added_count / max(1, hole.size)),
            "repair_mask_mode": str(args.repair_mask_mode),
            "repair_component_count": len(repair_components),
            "repair_component_preserved_count": int(
                sum(comp["action"] == "preserve" for comp in repair_components)
            ),
            "repair_component_repaired_count": int(
                sum(comp["action"] == "repair" for comp in repair_components)
            ),
            "repair_components": repair_components if repair_components else None,
            "mask_diagnostics": condition_cpu["mask_diagnostics"],
            "valid_ratio": float(reliable.mean()),
            "post_clip_mode": args.post_clip_mode,
            "post_clip_bounds": clip_bounds,
            "norm_center": condition_cpu["center_value"],
            "norm_scale": condition_cpu["scale_value"],
            "anchor_inpaint_radius": condition_cpu["radius"],
            "raw_valid_stats": finite_stats(depth[reliable]),
            "anchor_stats": finite_stats(anchor),
            "model_raw_stats": finite_stats(pred_raw_np),
            "model_stats": finite_stats(pred_np),
            "hole_only_stats": finite_stats(hole_only),
            "split_hole_only_stats": finite_stats(split_hole_only) if split_hole_only is not None else None,
            "plane_hole_only_stats": finite_stats(plane_hole_only) if plane_hole_only is not None else None,
            "aligned_hole_only_stats": finite_stats(aligned_hole_only) if aligned_hole_only is not None else None,
            "hybrid_hole_only_stats": finite_stats(hybrid_hole_only) if hybrid_hole_only is not None else None,
            "gated_hole_only_stats": finite_stats(gated_hole_only) if gated_hole_only is not None else None,
            "mean_abs_model_anchor_hole": safe_mean_abs(diff_model_anchor, hole),
            "mean_abs_model_anchor_valid": safe_mean_abs(diff_model_anchor, reliable),
            "mean_abs_model_raw_valid": safe_mean_abs(diff_model_raw, reliable),
            "mean_abs_hole_only_anchor_hole": safe_mean_abs(hole_only - anchor, hole),
            "mean_abs_split_anchor_hole": (
                safe_mean_abs(diff_split_anchor, hole) if diff_split_anchor is not None else None
            ),
            "mean_abs_plane_anchor_hole": (
                safe_mean_abs(diff_plane_anchor, hole) if diff_plane_anchor is not None else None
            ),
            "mean_abs_aligned_anchor_hole": (
                safe_mean_abs(diff_aligned_anchor, hole) if diff_aligned_anchor is not None else None
            ),
            "mean_abs_hybrid_anchor_hole": (
                safe_mean_abs(diff_hybrid_anchor, hole) if diff_hybrid_anchor is not None else None
            ),
            "mean_abs_gated_anchor_hole": (
                safe_mean_abs(diff_gated_anchor, hole) if diff_gated_anchor is not None else None
            ),
            "hybrid_model_hole_ratio": hybrid_model_hole_ratio,
            "gated_mean_weight_hole": gated_mean_weight_hole,
            "gated_model_pixel_ratio": gated_model_pixel_ratio,
            "gated_component_count": len(gated_components),
            "gated_force_anchor_component_count": int(
                sum(comp["force_anchor"] for comp in gated_components)
            ),
            "gated_components": gated_components if gated_components else None,
            "hole_component_count": len(hybrid_components),
            "hybrid_model_component_count": int(sum(comp["use_model"] for comp in hybrid_components)),
            "hybrid_anchor_component_count": int(sum(not comp["use_model"] for comp in hybrid_components)),
            "hybrid_components": hybrid_components if hybrid_components else None,
            "plane_component_count": len(plane_components),
            "plane_components": plane_components if plane_components else None,
            "aligned_component_count": len(aligned_components),
            "aligned_components": aligned_components if aligned_components else None,
            "split_added_component_count": len(split_added_components),
            "split_added_components": split_added_components if split_added_components else None,
        }
        rows.append(row)

        if not args.no_visualize and index < int(args.vis_max_samples):
            vis_path = os.path.join(args.output_dir, "visualizations", f"{stem}.png")
            save_visualization(
                vis_path,
                stem,
                depth,
                hole,
                anchor,
                pred_np,
                hole_only,
                split_hole_only=split_hole_only,
                hybrid_hole_only=hybrid_hole_only,
                hybrid_use_model_mask=hybrid_use_model_mask,
                plane_hole_only=plane_hole_only,
                aligned_hole_only=aligned_hole_only,
                gated_hole_only=gated_hole_only,
                gated_weight=gated_weight,
                repair_hole=repair_hole,
                preserved_hole=preserved_hole,
            )

        print(
            f"[{index + 1:03d}/{len(pairs):03d}] {stem} "
            f"hole={row['hole_ratio']:.3f} "
            f"added={row['cleaned_added_ratio']:.3f} "
            f"|model-anchor|_hole={row['mean_abs_model_anchor_hole'] or 0.0:.4f} "
            f"|model-raw|_valid={row['mean_abs_model_raw_valid'] or 0.0:.4f}"
        )

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_args": ckpt_args,
        "raw_dir": args.raw_dir,
        "depth_dir": args.depth_dir,
        "output_dir": args.output_dir,
        "amplitude_mode": args.amplitude_mode,
        "depth_unit": args.depth_unit,
        "raw9_transform": args.raw9_transform_effective,
        "checkpoint_raw9_transform": checkpoint_raw9_transform,
        "sampling_mode": sampling_mode,
        "sample_steps": sample_steps,
        "aggregate": summarize_rows(rows),
        "per_sample": rows,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved real raw9 inference results to {args.output_dir}")


if __name__ == "__main__":
    main()
