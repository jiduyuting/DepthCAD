import _bootstrap
import argparse
import json
import os
import re
from glob import glob

import cv2
import numpy as np
import torch

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from infer_real_raw9_flow import build_amp_speckle_cleaned_hole_mask, build_anchor_gated_hole_only
from infer_real_depth_flow import (
    clip_prediction,
    ensure_dir,
    move_condition_to_device,
    natural_key,
    normalize_depth,
    predict_depth,
)
from inference_depth_postprocess import opencv_depth_inpaint
from real_depth_masked_self_test import aggregate, mae, make_block_mask, save_visualization
from train_depth_flow_restoration import flow_model_in_channels
from train_depth_restoration import robust_nonnegative_channels


RAW9_SPATIAL_TRANSFORMS = ("none", "flip_lr", "flip_ud", "rot180")
RAW9_TRANSFORM_CHOICES = RAW9_SPATIAL_TRANSFORMS + ("auto",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Masked self-test on real 9-channel ToF raw data plus paired depth."
    )
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/real_raw9_masked_self_test_ratio10_thr1m_iq6",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="Optional train/val split JSON produced by real-data fine-tuning.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="all",
        choices=["all", "train", "val"],
        help="Subset to evaluate when --split_json is provided.",
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
        default="none",
        choices=RAW9_TRANSFORM_CHOICES,
        help=(
            "Spatial transform applied to raw9 before amplitude/mask/condition construction. "
            "flip_lr aligns most current Real raw9/depth pairs; auto estimates per sample from depth/amplitude edges."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="block",
        choices=["block", "real_hole_shapes", "real_hole_speckle_shapes", "threshold_amp_depth"],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.10)
    parser.add_argument("--num_masks_per_sample", type=int, default=1)
    parser.add_argument("--min_block_size", type=int, default=12)
    parser.add_argument("--max_block_size", type=int, default=72)
    parser.add_argument("--real_hole_min_area", type=int, default=24)
    parser.add_argument("--real_hole_max_area", type=int, default=0)
    parser.add_argument("--real_hole_min_overlap", type=float, default=0.6)
    parser.add_argument("--real_hole_max_components", type=int, default=8)
    parser.add_argument("--real_hole_max_attempts", type=int, default=512)
    parser.add_argument("--real_hole_exclude_self", action="store_true")
    parser.add_argument(
        "--component_raw_dir",
        type=str,
        default=None,
        help="Optional raw9 directory used only to build real-hole-shaped mask components.",
    )
    parser.add_argument(
        "--component_depth_dir",
        type=str,
        default=None,
        help="Optional depth directory used only to build real-hole-shaped mask components.",
    )
    parser.add_argument("--clean_outlier_mad_scale", type=float, default=6.0)
    parser.add_argument("--clean_outlier_abs", type=float, default=0.35)
    parser.add_argument("--clean_median_ksize", type=int, default=7)
    parser.add_argument("--clean_dilate", type=int, default=0)
    parser.add_argument("--clean_min_component_area", type=int, default=4)
    parser.add_argument("--speckle_window", type=int, default=11)
    parser.add_argument("--speckle_density_threshold", type=float, default=0.10)
    parser.add_argument("--speckle_residual_abs", type=float, default=0.18)
    parser.add_argument("--speckle_link_radius", type=int, default=2)
    parser.add_argument("--speckle_min_component_area", type=int, default=4)
    parser.add_argument("--speckle_max_component_area", type=int, default=30000)
    parser.add_argument("--speckle_max_bbox_side", type=int, default=260)
    parser.add_argument("--speckle_amp_ring_radius", type=int, default=7)
    parser.add_argument("--speckle_amp_ratio_min", type=float, default=2.5)
    parser.add_argument("--speckle_amp_delta_min", type=float, default=4000.0)
    parser.add_argument("--speckle_amp_abs_min", type=float, default=8000.0)
    parser.add_argument("--real_speckle_train_min_area", type=int, default=6)
    parser.add_argument("--real_speckle_train_max_area", type=int, default=0)
    parser.add_argument("--real_speckle_component_ratio", type=float, default=0.6)
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
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
        "--threshold_depth_min",
        type=float,
        default=None,
        help="Depth below this value is treated as an observed sensor hole in threshold_amp_depth mode.",
    )
    parser.add_argument(
        "--threshold_depth_max",
        type=float,
        default=None,
        help="Depth above this value is treated as an observed sensor hole in threshold_amp_depth mode.",
    )
    parser.add_argument(
        "--threshold_amp_threshold",
        type=float,
        default=None,
        help="Fixed amplitude threshold for threshold_amp_depth mode. Overrides percentile threshold.",
    )
    parser.add_argument(
        "--threshold_amp_percentile",
        type=float,
        default=8.0,
        help="Per-sample amplitude percentile used when --threshold_amp_threshold is omitted.",
    )
    parser.add_argument("--threshold_mask_open", type=int, default=0)
    parser.add_argument("--threshold_mask_close", type=int, default=0)
    parser.add_argument("--threshold_mask_dilate", type=int, default=0)
    parser.add_argument("--threshold_mask_min_component_area", type=int, default=1)
    parser.add_argument(
        "--hole_amplitude_mode",
        type=str,
        default="zero",
        choices=["zero", "keep_artificial", "keep_all"],
        help="Controls amplitude zeroing in depth holes. keep_all preserves raw9 amplitude cues in threshold masks.",
    )
    parser.add_argument(
        "--post_clip_mode",
        type=str,
        default="valid_range",
        choices=["none", "valid_range", "valid_percentile"],
    )
    parser.add_argument(
        "--post_clip_percentiles",
        type=float,
        nargs=2,
        default=[0.5, 99.5],
    )
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["endpoint", "euler"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument(
        "--gated_fill",
        action="store_true",
        default=False,
        help=(
            "Evaluate an anchor-gated hole-only output: anchor + gate * (model-anchor) "
            "inside holes, corrupted depth outside holes."
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
            "If >0, force a masked component to anchor when mean |model-anchor| exceeds "
            "this value in meters."
        ),
    )
    parser.add_argument(
        "--gate_component_max_p95_abs_diff",
        type=float,
        default=0.0,
        help=(
            "If >0, force a masked component to anchor when p95 |model-anchor| exceeds "
            "this value in meters."
        ),
    )
    parser.add_argument(
        "--gate_keep_border_anchor",
        action="store_true",
        default=False,
        help="If set, masked components touching the image border always keep anchor in gated fill.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_max_samples", type=int, default=24)
    return parser.parse_args()


def stem_sort_key(value):
    stem = os.path.splitext(os.path.basename(str(value)))[0]
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", stem)]


def collect_pairs(raw_dir, depth_dir):
    raw_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob(os.path.join(raw_dir, "*.npy"))
    }
    depth_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob(os.path.join(depth_dir, "*.npy"))
    }
    stems = sorted(set(raw_paths) & set(depth_paths), key=stem_sort_key)
    return [(stem, raw_paths[stem], depth_paths[stem]) for stem in stems]


def pair_dir_diagnostics(raw_dir, depth_dir, max_examples=8):
    raw_files = sorted(glob(os.path.join(raw_dir, "*.npy")), key=stem_sort_key)
    depth_files = sorted(glob(os.path.join(depth_dir, "*.npy")), key=stem_sort_key)
    raw_stems = {os.path.splitext(os.path.basename(path))[0] for path in raw_files}
    depth_stems = {os.path.splitext(os.path.basename(path))[0] for path in depth_files}
    common = sorted(raw_stems & depth_stems, key=stem_sort_key)
    raw_only = sorted(raw_stems - depth_stems, key=stem_sort_key)
    depth_only = sorted(depth_stems - raw_stems, key=stem_sort_key)
    max_examples = int(max_examples)
    return {
        "raw_dir": raw_dir,
        "depth_dir": depth_dir,
        "raw_dir_exists": os.path.isdir(raw_dir),
        "depth_dir_exists": os.path.isdir(depth_dir),
        "raw_npy_count": len(raw_files),
        "depth_npy_count": len(depth_files),
        "paired_count": len(common),
        "raw_examples": [os.path.basename(path) for path in raw_files[:max_examples]],
        "depth_examples": [os.path.basename(path) for path in depth_files[:max_examples]],
        "common_examples": common[:max_examples],
        "raw_only_examples": raw_only[:max_examples],
        "depth_only_examples": depth_only[:max_examples],
    }


def filter_pairs_by_split(pairs, split_json, eval_split):
    if eval_split == "all":
        return list(pairs)
    if not split_json:
        raise ValueError("--eval_split train/val requires --split_json")

    with open(split_json, "r") as f:
        split = json.load(f)
    if eval_split not in split or not split[eval_split]:
        raise ValueError(f"{split_json} does not contain a non-empty {eval_split!r} split.")

    pair_by_stem = {pair[0]: pair for pair in pairs}
    wanted = list(split[eval_split])
    missing = [stem for stem in wanted if stem not in pair_by_stem]
    if missing:
        raise ValueError(
            f"{split_json} contains {eval_split!r} stems not found under raw/depth: "
            f"{sorted(set(missing), key=stem_sort_key)}"
        )
    wanted_set = set(wanted)
    return [pair for pair in pairs if pair[0] in wanted_set]


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


def observed_hole_mask(depth, hole_depth_threshold):
    depth = np.asarray(depth, dtype=np.float32)
    return (~np.isfinite(depth)) | (depth <= float(hole_depth_threshold))


def selected_raw_channels(raw9, channels):
    raw9 = np.asarray(raw9)
    out = []
    for channel in channels:
        idx = int(channel)
        if idx < 0 or idx >= raw9.shape[0]:
            raise ValueError(f"Raw channel index {idx} is out of range for raw shape {raw9.shape}")
        out.append(idx)
    if not out:
        raise ValueError("At least one raw channel must be selected.")
    return out


def raw9_saturation_mask(raw9, channels, clip_value=65535.0, clip_margin=1.0, dilate=0):
    raw9 = np.asarray(raw9, dtype=np.float32)
    channel_ids = selected_raw_channels(raw9, channels)
    threshold = float(clip_value) - float(clip_margin)
    mask = np.any(raw9[channel_ids] >= threshold, axis=0)
    if int(dilate) > 0:
        kernel = np.ones((2 * int(dilate) + 1, 2 * int(dilate) + 1), dtype=np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask.astype(bool)


def append_connected_components(library, mask, source, kind, min_area=24, max_area=0):
    mask = np.asarray(mask, dtype=bool)
    max_area = None if int(max_area) <= 0 else int(max_area)
    if not mask.any():
        return
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        if max_area is not None and area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        crop = labels[y : y + h, x : x + w] == label
        ys, xs = np.nonzero(crop)
        if ys.size == 0:
            continue
        library.append(
            {
                "source": str(source),
                "kind": str(kind),
                "mask": crop.astype(np.bool_),
                "area": area,
                "center_y": float(ys.mean()),
                "center_x": float(xs.mean()),
            }
        )


def build_real_hole_component_library(
    pairs,
    hole_depth_threshold,
    min_area=24,
    max_area=0,
    source_mode="real_hole_shapes",
    amplitude_mode="iq6",
    args=None,
):
    library = []
    max_area = None if int(max_area) <= 0 else int(max_area)
    for stem, raw_path, depth_path in pairs:
        depth = np.load(depth_path).astype(np.float32)
        hole = observed_hole_mask(depth, hole_depth_threshold)
        if source_mode in {"real_hole_shapes", "real_hole_speckle_shapes"} and hole.any():
            append_connected_components(
                library,
                hole,
                stem,
                "threshold_hole",
                min_area=min_area,
                max_area=0 if max_area is None else max_area,
            )

        include_saturation = bool(getattr(args, "include_saturation_components", False))
        needs_raw9 = source_mode == "real_hole_speckle_shapes" or include_saturation
        raw9 = None
        if needs_raw9:
            raw9 = np.load(raw_path).astype(np.float32)
            if args is not None:
                depth_m = depth_to_meters(depth, getattr(args, "depth_unit", "auto"))
                if raw9.shape == (9,) + depth_m.shape:
                    align_reliable = (
                        np.isfinite(depth_m)
                        & (depth_m > float(getattr(args, "hole_depth_threshold", 0.0)))
                        & (depth_m >= float(getattr(args, "valid_min_depth", 0.0)))
                        & (depth_m <= float(getattr(args, "valid_max_depth", np.inf)))
                    )
                    raw9, _ = align_raw9_to_depth(raw9, depth_m, align_reliable, args)

        if include_saturation:
            sat_mask = raw9_saturation_mask(
                raw9,
                getattr(args, "sat_component_channels", [2, 5, 8]),
                clip_value=getattr(args, "sat_component_clip_value", 65535.0),
                clip_margin=getattr(args, "sat_component_clip_margin", 1.0),
                dilate=getattr(args, "sat_component_dilate", 0),
            )
            append_connected_components(
                library,
                sat_mask,
                stem,
                "raw_saturation",
                min_area=getattr(args, "sat_component_min_area", min_area),
                max_area=getattr(args, "sat_component_max_area", 0),
            )

        if source_mode != "real_hole_speckle_shapes":
            continue

        if args is None:
            raise ValueError("source_mode='real_hole_speckle_shapes' requires args with speckle settings.")
        _, amplitude_mean = raw9_to_amplitude(raw9, amplitude_mode)
        artifact_mask, _ = build_amp_speckle_cleaned_hole_mask(depth, amplitude_mean, args)
        artifact_mask &= ~hole
        if not artifact_mask.any():
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            artifact_mask.astype(np.uint8), connectivity=8
        )
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(args.speckle_min_component_area):
                continue
            if int(args.speckle_max_component_area) > 0 and area > int(args.speckle_max_component_area):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            crop = (labels[y : y + h, x : x + w] == label)
            ys, xs = np.nonzero(crop)
            if ys.size == 0:
                continue
            library.append(
                {
                    "source": stem,
                    "kind": "amp_speckle",
                    "mask": crop.astype(np.bool_),
                    "area": area,
                    "center_y": float(ys.mean()),
                    "center_x": float(xs.mean()),
                }
            )
    return library


def summarize_component_library(component_library):
    summary = {
        "total": int(len(component_library)),
        "threshold_hole": 0,
        "amp_speckle": 0,
        "raw_saturation": 0,
    }
    areas_by_kind = {}
    for component in component_library:
        kind = str(component.get("kind", "threshold_hole"))
        summary[kind] = int(summary.get(kind, 0)) + 1
        area = int(component.get("area", int(np.asarray(component["mask"], dtype=np.uint8).sum())))
        areas_by_kind.setdefault(kind, []).append(area)
    for kind, areas in areas_by_kind.items():
        areas_arr = np.asarray(areas, dtype=np.float32)
        summary[f"{kind}_area_mean"] = float(np.mean(areas_arr))
        summary[f"{kind}_area_p50"] = float(np.median(areas_arr))
        summary[f"{kind}_area_p90"] = float(np.percentile(areas_arr, 90.0))
        summary[f"{kind}_area_max"] = int(np.max(areas_arr))
    return summary


def subsample_components(components, keep_count, rng):
    keep_count = max(0, min(int(keep_count), len(components)))
    if keep_count >= len(components):
        return list(components)
    if keep_count == 0:
        return []
    indices = np.sort(rng.choice(len(components), size=keep_count, replace=False))
    return [components[int(idx)] for idx in indices]


def filter_and_rebalance_component_library(component_library, args):
    shape_components = []
    speckle_components = []
    speckle_dropped_small = 0
    speckle_dropped_large = 0
    speckle_min_area = int(getattr(args, "real_speckle_train_min_area", 0))
    speckle_max_area = int(getattr(args, "real_speckle_train_max_area", 0))

    for component in component_library:
        kind = str(component.get("kind", "threshold_hole"))
        area = int(component.get("area", int(np.asarray(component["mask"], dtype=np.uint8).sum())))
        if kind == "amp_speckle":
            if area < speckle_min_area:
                speckle_dropped_small += 1
                continue
            if speckle_max_area > 0 and area > speckle_max_area:
                speckle_dropped_large += 1
                continue
            speckle_components.append(component)
        else:
            shape_components.append(component)

    rng = np.random.default_rng(int(getattr(args, "seed", 123)))
    target_ratio = float(getattr(args, "real_speckle_component_ratio", 0.6))
    target_ratio = min(max(target_ratio, 0.0), 1.0)
    mask_mode = str(getattr(args, "mask_mode", "real_hole_shapes"))

    if mask_mode == "real_hole_speckle_shapes" and shape_components and speckle_components:
        if target_ratio <= 0.0:
            speckle_components = []
        elif target_ratio >= 1.0:
            shape_components = []
        else:
            max_speckles_for_threshold = int(
                round(len(shape_components) * target_ratio / max(1.0 - target_ratio, 1e-6))
            )
            max_thresholds_for_speckle = int(
                round(len(speckle_components) * (1.0 - target_ratio) / max(target_ratio, 1e-6))
            )
            if len(speckle_components) > max_speckles_for_threshold:
                speckle_components = subsample_components(speckle_components, max_speckles_for_threshold, rng)
            elif len(shape_components) > max_thresholds_for_speckle:
                shape_components = subsample_components(
                    shape_components,
                    max_thresholds_for_speckle,
                    rng,
                )

    filtered = shape_components + speckle_components
    actual_ratio = float(len(speckle_components) / max(len(filtered), 1))
    kept_by_kind = {}
    for component in filtered:
        kind = str(component.get("kind", "threshold_hole"))
        kept_by_kind[kind] = int(kept_by_kind.get(kind, 0)) + 1
    filter_summary = {
        "shape_kept": int(len(shape_components)),
        "threshold_kept": int(kept_by_kind.get("threshold_hole", 0)),
        "raw_saturation_kept": int(kept_by_kind.get("raw_saturation", 0)),
        "speckle_kept": int(len(speckle_components)),
        "speckle_dropped_small": int(speckle_dropped_small),
        "speckle_dropped_large": int(speckle_dropped_large),
        "target_speckle_ratio": float(target_ratio),
        "actual_speckle_ratio": actual_ratio,
    }
    return filtered, filter_summary


def place_component_mask(image_shape, component, target_center_y, target_center_x):
    height, width = image_shape
    comp = component["mask"]
    comp_h, comp_w = comp.shape
    y0 = int(round(float(target_center_y) - float(component["center_y"])))
    x0 = int(round(float(target_center_x) - float(component["center_x"])))
    y1 = y0 + comp_h
    x1 = x0 + comp_w

    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(height, y1)
    dst_x1 = min(width, x1)
    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        return np.zeros((height, width), dtype=bool)

    src_y0 = dst_y0 - y0
    src_x0 = dst_x0 - x0
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    out = np.zeros((height, width), dtype=bool)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = comp[src_y0:src_y1, src_x0:src_x1]
    return out


def make_real_hole_shape_mask(
    valid_mask,
    component_library,
    rng,
    target_ratio,
    min_overlap_ratio=0.6,
    max_components=8,
    max_attempts=512,
):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    mask = np.zeros_like(valid_mask, dtype=bool)
    target_pixels = max(1, int(round(float(target_ratio) * int(valid_mask.sum()))))
    if target_pixels <= 0 or not component_library:
        return mask

    valid_yx = np.argwhere(valid_mask)
    if valid_yx.size == 0:
        return mask

    placed = 0
    attempts = 0
    current = 0
    while current < target_pixels and placed < int(max_components) and attempts < int(max_attempts):
        attempts += 1
        comp = component_library[int(rng.integers(0, len(component_library)))]
        center_y, center_x = valid_yx[int(rng.integers(0, len(valid_yx)))]
        candidate = place_component_mask(valid_mask.shape, comp, center_y, center_x)
        candidate &= valid_mask
        candidate &= ~mask
        overlap = int(candidate.sum())
        if overlap <= 0:
            continue
        required = max(1, int(round(float(min_overlap_ratio) * int(comp["area"]))))
        if overlap < required:
            continue
        remaining = target_pixels - current
        if current > 0 and overlap > max(int(remaining * 1.75), int(target_pixels * 0.75)):
            continue
        mask |= candidate
        current = int(mask.sum())
        placed += 1
    return mask & valid_mask


def make_artificial_mask(valid_mask, rng, args, component_library=None, source_stem=None):
    if getattr(args, "mask_mode", "block") == "threshold_amp_depth":
        raise ValueError("threshold_amp_depth masks require depth/raw9; call make_threshold_amp_depth_mask directly.")
    if getattr(args, "mask_mode", "block") == "block":
        return make_block_mask(
            valid_mask,
            rng,
            args.mask_ratio,
            args.min_block_size,
            args.max_block_size,
        )

    active_library = component_library or []
    if getattr(args, "real_hole_exclude_self", False) and source_stem is not None:
        active_library = [comp for comp in active_library if comp["source"] != source_stem]
    max_components = int(getattr(args, "real_hole_max_components", 8))
    max_attempts = int(getattr(args, "real_hole_max_attempts", 512))
    if getattr(args, "mask_mode", "block") == "real_hole_speckle_shapes":
        max_components = max(max_components, 64)
        max_attempts = max(max_attempts, 2048)
    return make_real_hole_shape_mask(
        valid_mask,
        active_library,
        rng,
        args.mask_ratio,
        min_overlap_ratio=args.real_hole_min_overlap,
        max_components=max_components,
        max_attempts=max_attempts,
    )


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


def apply_spatial_transform(image, mode):
    mode = str(mode or "none")
    if mode == "none":
        return np.asarray(image)
    if mode == "flip_lr":
        return np.flip(image, axis=-1).copy()
    if mode == "flip_ud":
        return np.flip(image, axis=-2).copy()
    if mode == "rot180":
        return np.flip(np.flip(image, axis=-1), axis=-2).copy()
    raise ValueError(f"Unknown spatial transform: {mode}")


def transform_raw9(raw9, mode):
    mode = str(mode or "none")
    if mode == "auto":
        raise ValueError("raw9_transform='auto' requires depth; call align_raw9_to_depth().")
    raw9 = np.asarray(raw9, dtype=np.float32)
    if raw9.ndim != 3 or raw9.shape[0] != 9:
        raise ValueError(f"Expected raw shape (9,H,W), got {raw9.shape}")
    return apply_spatial_transform(raw9, mode).astype(np.float32, copy=False)


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


def estimate_raw9_transform(raw9, depth_m, reliable_mask, amplitude_mode="iq6"):
    raw9 = np.asarray(raw9, dtype=np.float32)
    depth = np.asarray(depth_m, dtype=np.float32)
    reliable = np.asarray(reliable_mask, dtype=bool)
    if raw9.shape != (9,) + depth.shape:
        raise ValueError(f"Shape mismatch for raw9/depth alignment: raw {raw9.shape}, depth {depth.shape}")

    depth_unit, valid = robust_unit_image(depth, reliable)
    corr_mask = valid
    if int(corr_mask.sum()) >= 128:
        eroded = cv2.erode(corr_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
        if int(eroded.sum()) >= 128:
            corr_mask = eroded
    depth_edge = edge_magnitude(depth_unit)

    scores = {}
    for mode in RAW9_SPATIAL_TRANSFORMS:
        transformed = transform_raw9(raw9, mode)
        _, amplitude_mean = raw9_to_amplitude(transformed, amplitude_mode)
        amp_unit, amp_valid = robust_unit_image(np.log1p(np.maximum(amplitude_mean, 0.0)), corr_mask)
        score_mask = corr_mask & amp_valid
        scores[mode] = masked_corrcoef(depth_edge, edge_magnitude(amp_unit), score_mask)

    best_mode = max(scores, key=lambda key: (scores[key], key == "none"))
    return best_mode, scores


def align_raw9_to_depth(raw9, depth_m, reliable_mask, args):
    requested = str(getattr(args, "raw9_transform", "none") or "none")
    if requested == "auto":
        best_mode, scores = estimate_raw9_transform(
            raw9,
            depth_m,
            reliable_mask,
            amplitude_mode=getattr(args, "amplitude_mode", "iq6"),
        )
        info = {
            "raw9_transform": requested,
            "raw9_transform_estimated": best_mode,
        }
        for mode, score in scores.items():
            info[f"raw9_transform_score_{mode}"] = float(score)
        return transform_raw9(raw9, best_mode), info

    return transform_raw9(raw9, requested), {
        "raw9_transform": requested,
        "raw9_transform_estimated": requested,
    }


def filter_small_components(mask, min_area):
    mask = np.asarray(mask, dtype=bool)
    min_area = int(min_area)
    if min_area <= 1 or not mask.any():
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out |= labels == label
    return out


def morph_mask(mask, open_radius=0, close_radius=0, dilate_radius=0):
    out = np.asarray(mask, dtype=bool).astype(np.uint8)
    close_radius = int(close_radius)
    open_radius = int(open_radius)
    dilate_radius = int(dilate_radius)
    if close_radius > 0:
        kernel = np.ones((2 * close_radius + 1, 2 * close_radius + 1), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if open_radius > 0:
        kernel = np.ones((2 * open_radius + 1, 2 * open_radius + 1), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if dilate_radius > 0:
        kernel = np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), dtype=np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)
    return out.astype(bool)


def make_threshold_amp_depth_mask(depth_m, raw9, reliable_mask, args):
    """Build deterministic sensor-style holes from depth range and low amplitude."""
    depth = np.asarray(depth_m, dtype=np.float32)
    reliable = np.asarray(reliable_mask, dtype=bool)
    _, amplitude_mean = raw9_to_amplitude(raw9, args.amplitude_mode)

    min_depth = (
        float(args.valid_min_depth)
        if args.threshold_depth_min is None
        else float(args.threshold_depth_min)
    )
    max_depth = (
        float(args.valid_max_depth)
        if args.threshold_depth_max is None
        else float(args.threshold_depth_max)
    )
    depth_range_hole = (~np.isfinite(depth)) | (depth < min_depth) | (depth > max_depth)

    amp_valid = reliable & np.isfinite(amplitude_mean)
    if amp_valid.sum() == 0:
        amp_threshold = 0.0 if args.threshold_amp_threshold is None else float(args.threshold_amp_threshold)
        amp_low = np.zeros_like(reliable, dtype=bool)
    else:
        if args.threshold_amp_threshold is None:
            amp_threshold = float(np.percentile(amplitude_mean[amp_valid], float(args.threshold_amp_percentile)))
        else:
            amp_threshold = float(args.threshold_amp_threshold)
        amp_low = amp_valid & (amplitude_mean <= amp_threshold)

    repair_mask = morph_mask(
        amp_low,
        open_radius=args.threshold_mask_open,
        close_radius=args.threshold_mask_close,
        dilate_radius=args.threshold_mask_dilate,
    )
    repair_mask = filter_small_components(repair_mask, args.threshold_mask_min_component_area)
    repair_mask &= reliable

    observed_hole = morph_mask(
        depth_range_hole,
        open_radius=args.threshold_mask_open,
        close_radius=args.threshold_mask_close,
        dilate_radius=0,
    )
    full_hole = observed_hole | repair_mask
    diagnostics = {
        "threshold_depth_min": float(min_depth),
        "threshold_depth_max": float(max_depth),
        "threshold_amp_threshold": float(amp_threshold),
        "threshold_amp_percentile": None
        if args.threshold_amp_threshold is not None
        else float(args.threshold_amp_percentile),
        "depth_range_hole_ratio": float(observed_hole.mean()),
        "amp_repair_mask_ratio": float(repair_mask.mean()),
        "full_threshold_hole_ratio": float(full_hole.mean()),
        "repair_pixels": int(repair_mask.sum()),
        "observed_hole_pixels": int(observed_hole.sum()),
    }
    return full_hole.astype(bool), repair_mask.astype(bool), diagnostics


def make_condition(
    corrupted_depth,
    raw9,
    artificial_mask,
    reliable_mask,
    ckpt_args,
    args,
    preserve_amplitude_mask=None,
):
    depth = np.asarray(corrupted_depth, dtype=np.float32)
    artificial_mask = np.asarray(artificial_mask, dtype=bool)
    hole = ((~np.isfinite(depth)) | (depth <= float(args.hole_depth_threshold))) | artificial_mask
    confidence = (~hole).astype(np.float32)

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
        stat_mask = reliable_mask & np.isfinite(anchor)
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

    amplitude, amplitude_mean = raw9_to_amplitude(raw9, args.amplitude_mode)
    amp_hole = hole | artificial_mask
    hole_amplitude_mode = str(getattr(args, "hole_amplitude_mode", "zero"))
    if hole_amplitude_mode == "keep_artificial":
        amp_hole = hole & (~artificial_mask)
    elif hole_amplitude_mode == "keep_all":
        amp_hole = np.zeros_like(hole, dtype=bool)
    elif hole_amplitude_mode != "zero":
        raise ValueError(f"Unknown hole_amplitude_mode: {hole_amplitude_mode}")
    if preserve_amplitude_mask is not None:
        amp_hole &= ~np.asarray(preserve_amplitude_mask, dtype=bool)
    amplitude = amplitude.copy()
    amplitude_mean = amplitude_mean.copy()
    amplitude[:, amp_hole] = 0.0
    amplitude_mean[amp_hole] = 0.0

    channels = [
        anchor_norm,
        noisy_norm,
        hole.astype(np.float32),
        confidence,
    ]
    channels.extend(
        robust_nonnegative_channels(
            amplitude,
            reliable_mask,
            percentile=ckpt_args.get("feature_percentile", 99.0),
            clip=ckpt_args.get("feature_clip", 3.0),
        )
    )
    channels.extend(
        robust_nonnegative_channels(
            amplitude_mean,
            reliable_mask,
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
        "anchor": anchor,
        "hole": hole,
        "amplitude": amplitude,
        "amplitude_mean": amplitude_mean,
        "center_value": center,
        "scale_value": scale,
        "radius": radius,
    }


def main():
    args = parse_args()
    pairs = collect_pairs(args.raw_dir, args.depth_dir)
    if not pairs:
        raise FileNotFoundError(f"No paired .npy files found in {args.raw_dir} and {args.depth_dir}")
    paired_case_count = len(pairs)
    pairs = filter_pairs_by_split(pairs, args.split_json, args.eval_split)
    if not pairs:
        raise ValueError(f"No pairs left after applying eval_split={args.eval_split!r}.")

    hole_component_library = []
    component_filter_summary = {}
    component_library_summary = {"total": 0, "threshold_hole": 0, "amp_speckle": 0}
    if args.mask_mode in {"real_hole_shapes", "real_hole_speckle_shapes"}:
        component_pairs = pairs
        component_raw_dir = args.component_raw_dir or args.raw_dir
        component_depth_dir = args.component_depth_dir or args.depth_dir
        if args.component_raw_dir or args.component_depth_dir:
            if not (args.component_raw_dir and args.component_depth_dir):
                raise ValueError("--component_raw_dir and --component_depth_dir must be provided together.")
            component_pairs = collect_pairs(component_raw_dir, component_depth_dir)
            if not component_pairs:
                raise FileNotFoundError(
                    f"No paired .npy files found in component dirs: {component_raw_dir}, {component_depth_dir}"
                )
        hole_component_library = build_real_hole_component_library(
            component_pairs,
            args.hole_depth_threshold,
            min_area=args.real_hole_min_area,
            max_area=args.real_hole_max_area,
            source_mode=args.mask_mode,
            amplitude_mode=args.amplitude_mode,
            args=args,
        )
        hole_component_library, component_filter_summary = filter_and_rebalance_component_library(
            hole_component_library,
            args,
        )
        component_library_summary = summarize_component_library(hole_component_library)
        if not hole_component_library:
            raise ValueError(f"No eligible real-hole components found for mask_mode={args.mask_mode!r}.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("input_mode") != "noisy_amp":
        raise ValueError(f"Expected input_mode='noisy_amp', got {ckpt_args.get('input_mode')!r}")
    model = build_model(ckpt, ckpt_args, device)

    sampling_mode = args.sampling_mode or ckpt_args.get("eval_sampling_mode", "endpoint")
    sample_steps = int(args.sample_steps or ckpt_args.get("sample_steps", 8))

    for subdir in [
        "restored",
        "hole_only",
        "gated_hole_only",
        "gated_weight",
        "anchor",
        "corrupted",
        "mask",
        "condition_mask",
        "visualizations",
    ]:
        ensure_dir(os.path.join(args.output_dir, subdir))

    rows = []
    vis_saved = 0
    for pair_index, (stem, raw_path, depth_path) in enumerate(pairs):
        raw9 = np.load(raw_path).astype(np.float32)
        clean = depth_to_meters(np.load(depth_path), args.depth_unit)
        if raw9.shape != (9,) + clean.shape:
            raise ValueError(f"Shape mismatch for {stem}: raw {raw9.shape}, depth {clean.shape}")

        reliable = (
            np.isfinite(clean)
            & (clean > float(args.hole_depth_threshold))
            & (clean >= float(args.valid_min_depth))
            & (clean <= float(args.valid_max_depth))
        )
        if reliable.sum() == 0:
            continue
        raw9, raw9_align_info = align_raw9_to_depth(raw9, clean, reliable, args)

        for repeat in range(int(args.num_masks_per_sample)):
            rng = np.random.default_rng(int(args.seed) + pair_index * 1009 + repeat)
            corrupted = clean.copy()
            threshold_hole = np.zeros_like(reliable, dtype=bool)
            threshold_diagnostics = {}
            if args.mask_mode == "threshold_amp_depth":
                threshold_hole, artificial_mask, threshold_diagnostics = make_threshold_amp_depth_mask(
                    clean,
                    raw9,
                    reliable,
                    args,
                )
                corrupted[threshold_hole] = 0.0
            else:
                artificial_mask = make_artificial_mask(
                    reliable,
                    rng,
                    args,
                    component_library=hole_component_library,
                    source_stem=stem,
                )
                corrupted[artificial_mask] = 0.0
            if artificial_mask.sum() == 0:
                continue

            condition_cpu = make_condition(corrupted, raw9, artificial_mask, reliable, ckpt_args, args)
            condition = move_condition_to_device(condition_cpu, device)
            pred = predict_depth(model, condition, ckpt_args, sampling_mode, sample_steps)
            pred_raw = pred.detach().cpu().numpy()[0, 0].astype(np.float32)
            pred, clip_bounds = clip_prediction(pred_raw, clean, reliable, args)

            condition_hole = condition_cpu["hole"]
            anchor = condition_cpu["anchor"]
            hole_only = np.where(condition_hole, pred, corrupted).astype(np.float32)
            gated_hole_only, gated_weight, gated_components = build_anchor_gated_hole_only(
                corrupted,
                condition_hole,
                anchor,
                pred,
                args,
            )
            unmasked_reliable = reliable & (~artificial_mask)

            row = {
                "name": stem,
                "repeat": repeat,
                "raw_path": raw_path,
                "depth_path": depth_path,
                "shape": list(clean.shape),
                "raw_shape": list(raw9.shape),
                "amplitude_mode": args.amplitude_mode,
                "raw9_transform": raw9_align_info.get("raw9_transform", args.raw9_transform),
                "raw9_transform_estimated": raw9_align_info.get(
                    "raw9_transform_estimated",
                    raw9_align_info.get("raw9_transform", args.raw9_transform),
                ),
                "mask_mode": args.mask_mode,
                "mask_ratio_target": float(args.mask_ratio),
                "mask_ratio_actual": float(artificial_mask.sum() / max(reliable.sum(), 1)),
                "mask_pixel_count": int(artificial_mask.sum()),
                "condition_hole_ratio": float(condition_cpu["hole"].mean()),
                "threshold_hole_pixel_count": int(threshold_hole.sum()),
                "reliable_pixel_count": int(reliable.sum()),
                "post_clip_bounds": clip_bounds,
            }
            row.update(
                {
                    key: value
                    for key, value in raw9_align_info.items()
                    if key.startswith("raw9_transform_score_")
                }
            )
            row.update(threshold_diagnostics)
            metric_specs = [
                ("anchor_mask_mae", anchor, artificial_mask),
                ("model_mask_mae", pred, artificial_mask),
                ("hole_only_mask_mae", hole_only, artificial_mask),
                ("model_unmasked_mae", pred, unmasked_reliable),
                ("hole_only_unmasked_mae", hole_only, unmasked_reliable),
                ("anchor_global_mae", anchor, reliable),
                ("model_global_mae", pred, reliable),
                ("hole_only_global_mae", hole_only, reliable),
            ]
            if gated_hole_only is not None:
                metric_specs.extend(
                    [
                        ("gated_mask_mae", gated_hole_only, artificial_mask),
                        ("gated_unmasked_mae", gated_hole_only, unmasked_reliable),
                        ("gated_global_mae", gated_hole_only, reliable),
                    ]
                )
            for key, prediction, mask in metric_specs:
                value, count = mae(prediction, clean, mask)
                row[key] = value
                row[f"{key}_count"] = count
            if row["anchor_mask_mae"] is not None and row["model_mask_mae"] is not None:
                row["mask_improve_vs_anchor"] = (
                    row["anchor_mask_mae"] - row["model_mask_mae"]
                ) / max(row["anchor_mask_mae"], 1e-12)
            if row["anchor_mask_mae"] is not None and row.get("gated_mask_mae") is not None:
                row["gated_mask_improve_vs_anchor"] = (
                    row["anchor_mask_mae"] - row["gated_mask_mae"]
                ) / max(row["anchor_mask_mae"], 1e-12)
            if gated_weight is not None and artificial_mask.any():
                row["gated_mean_weight_mask"] = float(np.mean(gated_weight[artificial_mask]))
                row["gated_model_pixel_ratio_mask"] = float(np.mean(gated_weight[artificial_mask] > 0.5))
                row["gated_component_count"] = int(len(gated_components))
                row["gated_force_anchor_component_count"] = int(
                    sum(comp["force_anchor"] for comp in gated_components)
                )
            rows.append(row)

            case_name = f"{stem}_r{repeat:02d}"
            np.save(os.path.join(args.output_dir, "restored", f"{case_name}_restored.npy"), pred)
            np.save(os.path.join(args.output_dir, "hole_only", f"{case_name}_hole_only.npy"), hole_only)
            if gated_hole_only is not None and gated_weight is not None:
                np.save(
                    os.path.join(args.output_dir, "gated_hole_only", f"{case_name}_gated_hole_only.npy"),
                    gated_hole_only,
                )
                np.save(
                    os.path.join(args.output_dir, "gated_weight", f"{case_name}_gated_weight.npy"),
                    gated_weight,
                )
            np.save(os.path.join(args.output_dir, "anchor", f"{case_name}_anchor.npy"), anchor)
            np.save(os.path.join(args.output_dir, "corrupted", f"{case_name}_corrupted.npy"), corrupted)
            np.save(os.path.join(args.output_dir, "mask", f"{case_name}_mask.npy"), artificial_mask.astype(np.uint8))
            np.save(
                os.path.join(args.output_dir, "condition_mask", f"{case_name}_condition_mask.npy"),
                condition_hole.astype(np.uint8),
            )

            if args.visualize and vis_saved < int(args.vis_max_samples):
                title = (
                    f"{case_name} {args.amplitude_mode} | "
                    f"anchor_mask={row['anchor_mask_mae']:.4f} "
                    f"model_mask={row['model_mask_mae']:.4f} "
                    f"improve={row.get('mask_improve_vs_anchor', 0.0):.1%}"
                )
                save_visualization(
                    os.path.join(args.output_dir, "visualizations", f"{case_name}.png"),
                    title,
                    clean,
                    corrupted,
                    artificial_mask,
                    anchor,
                    pred,
                    hole_only,
                    gated_hole_only=gated_hole_only,
                    gated_weight=gated_weight,
                )
                vis_saved += 1

            gated_text = ""
            if row.get("gated_mask_mae") is not None:
                gated_text = (
                    f" gated={row['gated_mask_mae']:.4f} "
                    f"gated_improve={row.get('gated_mask_improve_vs_anchor', 0.0):.1%}"
                )
            print(
                f"[{len(rows):03d}] {case_name} "
                f"amp={args.amplitude_mode} "
                f"mask_mode={args.mask_mode} "
                f"mask={row['mask_ratio_actual']:.3f} "
                f"anchor={row['anchor_mask_mae']:.4f} "
                f"model={row['model_mask_mae']:.4f} "
                f"improve={row.get('mask_improve_vs_anchor', 0.0):.1%}"
                f"{gated_text}"
            )

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_args": ckpt_args,
        "raw_dir": args.raw_dir,
        "depth_dir": args.depth_dir,
        "output_dir": args.output_dir,
        "split_json": args.split_json,
        "eval_split": args.eval_split,
        "amplitude_mode": args.amplitude_mode,
        "raw9_transform": args.raw9_transform,
        "paired_cases": paired_case_count,
        "evaluated_cases": len(pairs),
        "mask_mode": args.mask_mode,
        "hole_depth_threshold": args.hole_depth_threshold,
        "depth_unit": args.depth_unit,
        "valid_min_depth": args.valid_min_depth,
        "valid_max_depth": args.valid_max_depth,
        "threshold_depth_min": args.threshold_depth_min,
        "threshold_depth_max": args.threshold_depth_max,
        "threshold_amp_threshold": args.threshold_amp_threshold,
        "threshold_amp_percentile": args.threshold_amp_percentile,
        "threshold_mask_open": args.threshold_mask_open,
        "threshold_mask_close": args.threshold_mask_close,
        "threshold_mask_dilate": args.threshold_mask_dilate,
        "threshold_mask_min_component_area": args.threshold_mask_min_component_area,
        "hole_amplitude_mode": args.hole_amplitude_mode,
        "mask_ratio": args.mask_ratio,
        "num_masks_per_sample": args.num_masks_per_sample,
        "hole_component_count": len(hole_component_library),
        "hole_component_summary": component_library_summary,
        "hole_component_filter_summary": component_filter_summary,
        "component_raw_dir": args.component_raw_dir,
        "component_depth_dir": args.component_depth_dir,
        "sampling_mode": sampling_mode,
        "sample_steps": sample_steps,
        "gated_fill": bool(args.gated_fill),
        "gate_diff_soft": float(args.gate_diff_soft),
        "gate_diff_hard": float(args.gate_diff_hard),
        "gate_component_max_mean_abs_diff": float(args.gate_component_max_mean_abs_diff),
        "gate_component_max_p95_abs_diff": float(args.gate_component_max_p95_abs_diff),
        "gate_keep_border_anchor": bool(args.gate_keep_border_anchor),
        "aggregate": aggregate(rows),
        "per_sample": rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved real raw9 masked self-test results to {args.output_dir}")


if __name__ == "__main__":
    main()
