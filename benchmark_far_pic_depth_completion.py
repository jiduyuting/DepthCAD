import argparse
import json
import os
from glob import glob
from types import SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from infer_real_depth_flow import (
    clip_prediction,
    make_depth_condition,
    move_condition_to_device,
    predict_depth,
)
from inference_depth_postprocess import opencv_depth_inpaint
from real_depth_masked_self_test import make_block_mask
from train_depth_flow_restoration import flow_model_in_channels


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def natural_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def finite_percentiles(values, percentiles):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None for _ in percentiles]
    return [float(np.percentile(values, p)) for p in percentiles]


def save_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_model(checkpoint_path, device):
    ckpt = load_checkpoint(checkpoint_path, device)
    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("input_mode", "noisy") != "noisy":
        raise ValueError(
            f"Depth-only benchmark requires input_mode='noisy', got {ckpt_args.get('input_mode')!r}"
        )

    condition_channels = 4 + int(bool(ckpt_args.get("include_hole_distance", False)))
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
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args


def zero_mask(depth, args):
    depth = np.asarray(depth, dtype=np.float32)
    return (~np.isfinite(depth)) | (depth <= float(args.hole_depth_threshold))


def reliable_mask(depth, args):
    depth = np.asarray(depth, dtype=np.float32)
    return (
        np.isfinite(depth)
        & (depth > float(args.valid_min_depth))
        & (depth <= float(args.valid_max_depth))
    )


def filter_components(mask, max_area, min_area=1):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    max_area = int(max_area)
    min_area = int(min_area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        out[labels == label] = True
    return out


def median_filter_float(image, ksize):
    ksize = int(ksize)
    if ksize <= 1:
        return image.astype(np.float32, copy=True)
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    padded = np.pad(np.asarray(image, dtype=np.float32), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (ksize, ksize))
    return np.median(windows, axis=(-2, -1)).astype(np.float32)


def build_bad_depth_mask(depth, args):
    depth = np.asarray(depth, dtype=np.float32)
    zmask = zero_mask(depth, args)
    reliable = reliable_mask(depth, args) & (~zmask)
    if reliable.sum() == 0:
        return zmask, {
            "zero_ratio": float(zmask.mean()),
            "bad_extra_ratio": 0.0,
            "reason": "no reliable pixels",
        }

    anchor = opencv_depth_inpaint(depth, zmask, method="ns", radius=5)
    local = median_filter_float(anchor, int(args.bad_median_ksize))
    residual = np.abs(depth - local)

    far = reliable & (depth > float(args.bad_far_depth))
    near = np.isfinite(depth) & (depth > float(args.hole_depth_threshold)) & (
        depth < float(args.valid_min_depth)
    )
    residual_seed = reliable & (residual > float(args.bad_residual_abs))
    residual_seed = filter_components(
        residual_seed,
        max_area=int(args.bad_max_component_area),
        min_area=int(args.bad_min_component_area),
    )

    extra = far | near | residual_seed
    if int(args.bad_dilate) > 0 and extra.any():
        k = 2 * int(args.bad_dilate) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        extra = cv2.dilate(extra.astype(np.uint8), kernel, iterations=1).astype(bool)
        extra &= np.isfinite(depth)

    mask = zmask | extra
    diagnostics = {
        "zero_ratio": float(zmask.mean()),
        "far_ratio": float(far.mean()),
        "near_ratio": float(near.mean()),
        "residual_seed_ratio": float(residual_seed.mean()),
        "bad_extra_ratio": float((mask & (~zmask)).mean()),
        "bad_total_ratio": float(mask.mean()),
        "bad_far_depth": float(args.bad_far_depth),
        "bad_residual_abs": float(args.bad_residual_abs),
    }
    return mask, diagnostics


def extract_component_templates(mask, args, source):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    templates = []
    min_area = int(args.realhole_template_min_area)
    max_area = int(args.realhole_template_max_area)
    pad = int(args.realhole_template_pad)
    h, w = mask.shape
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        y0 = max(0, y - pad)
        y1 = min(h, y + bh + pad)
        x0 = max(0, x - pad)
        x1 = min(w, x + bw + pad)
        local = labels[y0:y1, x0:x1] == label
        if local.any():
            templates.append(
                {
                    "mask": local.astype(bool),
                    "area": area,
                    "source": source,
                    "bbox": [int(y0), int(x0), int(y1), int(x1)],
                }
            )
    return templates


def load_realhole_templates(args, raw_by_stem=None):
    templates = []
    for template_dir in args.realhole_template_dirs:
        for path in sorted(glob(os.path.join(template_dir, "*.npy"))):
            mask = np.load(path).astype(bool)
            templates.extend(extract_component_templates(mask, args, os.path.abspath(path)))

    if not templates and raw_by_stem is not None:
        for stem, raw in raw_by_stem.items():
            zmask = zero_mask(raw, args)
            templates.extend(extract_component_templates(zmask, args, f"{stem}:zero_mask"))
            bad_mask, _ = build_bad_depth_mask(raw, args)
            templates.extend(extract_component_templates(bad_mask, args, f"{stem}:bad_depth_mask"))

    return templates


def transform_realhole_template(template_mask, rng, args, out_shape):
    mask = np.asarray(template_mask, dtype=np.uint8)
    if rng.random() < 0.5:
        mask = np.flip(mask, axis=0)
    if rng.random() < 0.5:
        mask = np.flip(mask, axis=1)
    scale = float(rng.uniform(float(args.realhole_min_scale), float(args.realhole_max_scale)))
    if abs(scale - 1.0) > 1e-3:
        new_h = max(1, int(round(mask.shape[0] * scale)))
        new_w = max(1, int(round(mask.shape[1] * scale)))
        mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    h, w = out_shape
    if mask.shape[0] > h or mask.shape[1] > w:
        scale = min(h / max(mask.shape[0], 1), w / max(mask.shape[1], 1), 1.0)
        new_h = max(1, int(round(mask.shape[0] * scale)))
        new_w = max(1, int(round(mask.shape[1] * scale)))
        mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def make_realhole_mask(valid_mask, templates, rng, target_ratio, args):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    h, w = valid_mask.shape
    target = max(1, int(round(float(target_ratio) * int(valid_mask.sum()))))
    mask = np.zeros((h, w), dtype=bool)
    valid_yx = np.argwhere(valid_mask)
    if valid_yx.size == 0 or not templates:
        return mask, {
            "realhole_target_pixels": target,
            "realhole_pixels": 0,
            "realhole_template_count": len(templates),
            "realhole_reason": "no valid pixels or no templates",
        }

    sources = []
    attempts = 0
    max_attempts = int(args.realhole_max_attempts)
    min_overlap = float(args.realhole_min_valid_overlap)
    while int((mask & valid_mask).sum()) < target and attempts < max_attempts:
        attempts += 1
        template = templates[int(rng.integers(0, len(templates)))]
        local = transform_realhole_template(template["mask"], rng, args, valid_mask.shape)
        if not local.any():
            continue
        th, tw = local.shape
        cy, cx = valid_yx[int(rng.integers(0, len(valid_yx)))]
        y0 = int(np.clip(cy - th // 2, 0, max(h - th, 0)))
        x0 = int(np.clip(cx - tw // 2, 0, max(w - tw, 0)))
        y1 = min(h, y0 + th)
        x1 = min(w, x0 + tw)
        local = local[: y1 - y0, : x1 - x0]
        if not local.any():
            continue

        valid_patch = valid_mask[y0:y1, x0:x1]
        candidate = local & valid_patch & (~mask[y0:y1, x0:x1])
        overlap = candidate.sum() / max(int(local.sum()), 1)
        if candidate.sum() == 0 or overlap < min_overlap:
            continue
        mask[y0:y1, x0:x1] |= candidate
        if len(sources) < 20:
            sources.append(template["source"])

    mask &= valid_mask
    pixels = int(mask.sum())
    return mask, {
        "realhole_target_pixels": int(target),
        "realhole_pixels": pixels,
        "realhole_ratio_over_reliable": float(pixels / max(int(valid_mask.sum()), 1)),
        "realhole_attempts": int(attempts),
        "realhole_template_count": int(len(templates)),
        "realhole_sampled_sources": sources,
    }


def boundary_jump(depth, reference_depth, hole):
    hole = np.asarray(hole, dtype=bool)
    valid = (~hole) & np.isfinite(reference_depth)
    if hole.sum() == 0 or valid.sum() == 0:
        return None
    jumps = []
    pairs = [
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ]
    for hole_slice, valid_slice in pairs:
        edge = hole[hole_slice] & valid[valid_slice]
        if edge.any():
            diff = np.abs(depth[hole_slice][edge] - reference_depth[valid_slice][edge])
            diff = diff[np.isfinite(diff)]
            if diff.size:
                jumps.append(diff)
    if not jumps:
        return None
    jumps = np.concatenate(jumps)
    return {
        "mean": float(np.mean(jumps)),
        "median": float(np.median(jumps)),
        "p95": float(np.percentile(jumps, 95.0)),
    }


def masked_tv(depth, mask):
    mask = np.asarray(mask, dtype=bool)
    vals = []
    dx = mask[:, 1:] & mask[:, :-1]
    if dx.any():
        vals.append(np.abs(depth[:, 1:][dx] - depth[:, :-1][dx]))
    dy = mask[1:, :] & mask[:-1, :]
    if dy.any():
        vals.append(np.abs(depth[1:, :][dy] - depth[:-1, :][dy]))
    if not vals:
        return None
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95.0)),
    }


def mae(pred, target, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    if valid.sum() == 0:
        return None, 0
    return float(np.mean(np.abs(pred[valid] - target[valid]))), int(valid.sum())


def summarize_method(output, raw, repair_mask, eval_mask=None):
    outside = (~repair_mask) & np.isfinite(raw)
    diff = output[outside] - raw[outside]
    diff = diff[np.isfinite(diff)]
    out = {
        "zero_ratio_in_repair_mask": float(np.mean(output[repair_mask] == 0.0))
        if repair_mask.any()
        else 0.0,
        "outside_mean_abs_change": float(np.mean(np.abs(diff))) if diff.size else None,
        "outside_max_abs_change": float(np.max(np.abs(diff))) if diff.size else None,
        "hole_min_p50_p95_p99_max": finite_percentiles(
            output[repair_mask & np.isfinite(output)], [0, 50, 95, 99, 100]
        ),
        "boundary_jump_m": boundary_jump(output, raw, repair_mask),
        "hole_total_variation_m": masked_tv(output, repair_mask),
    }
    if eval_mask is not None:
        value, count = mae(output, raw, eval_mask)
        out["eval_mask_mae"] = value
        out["eval_mask_mae_count"] = count
    return out


def aggregate_rows(rows):
    out = {"num_rows": len(rows)}
    methods = sorted({name for row in rows for name in row["methods"].keys()})
    for method in methods:
        method_rows = [row["methods"][method] for row in rows if method in row["methods"]]
        for key in [
            "zero_ratio_in_repair_mask",
            "outside_mean_abs_change",
            "outside_max_abs_change",
            "eval_mask_mae",
        ]:
            weighted_total = 0.0
            weighted_count = 0
            vals = []
            for r in method_rows:
                value = r.get(key)
                if value is None:
                    continue
                vals.append(float(value))
                if key == "eval_mask_mae":
                    count = int(r.get("eval_mask_mae_count", 0))
                    weighted_total += float(value) * count
                    weighted_count += count
            if vals:
                out[f"{method}_{key}_mean"] = float(np.mean(vals))
                out[f"{method}_{key}_max"] = float(np.max(vals))
            if key == "eval_mask_mae" and weighted_count > 0:
                out[f"{method}_eval_mask_mae_weighted"] = float(weighted_total / weighted_count)
                out[f"{method}_eval_mask_mae_count"] = int(weighted_count)
        for nested in ["boundary_jump_m", "hole_total_variation_m"]:
            for stat in ["mean", "median", "p95"]:
                vals = [
                    r[nested][stat]
                    for r in method_rows
                    if r.get(nested) is not None and r[nested].get(stat) is not None
                ]
                if vals:
                    out[f"{method}_{nested}_{stat}_mean"] = float(np.mean(vals))
    return out


def image_limits(arrays, mask):
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        v = arr[mask & np.isfinite(arr)]
        if v.size:
            vals.append(v)
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def save_visualization(path, title, raw, repair_mask, methods, eval_mask=None):
    valid = (~zero_mask(raw, SimpleNamespace(hole_depth_threshold=0.0))) & np.isfinite(raw)
    vmin, vmax = image_limits([raw] + list(methods.values()), valid)
    raw_vis = raw.copy()
    raw_vis[repair_mask] = np.nan
    panels = [
        ("raw depth\n(mask hidden)", raw_vis, "viridis", vmin, vmax),
        ("repair mask", repair_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("OpenCV NS", methods.get("opencv_ns"), "viridis", vmin, vmax),
        ("OpenCV Telea", methods.get("opencv_telea"), "viridis", vmin, vmax),
        ("ours hole_only", methods.get("ours_hole_only"), "viridis", vmin, vmax),
        ("ours restored", methods.get("ours_restored"), "viridis", vmin, vmax),
    ]
    if "propainter" in methods:
        panels.append(("ProPainter", methods["propainter"], "viridis", vmin, vmax))
    if eval_mask is not None and eval_mask.any():
        err_values = []
        for key in ["opencv_ns", "ours_hole_only", "propainter"]:
            if key in methods:
                err = np.abs(methods[key] - raw)
                err_values.append(err[eval_mask & np.isfinite(err)])
        err_values = np.concatenate([x for x in err_values if x.size]) if err_values else np.array([])
        emax = float(np.percentile(err_values, 98.0)) if err_values.size else 1.0
        emax = max(emax, 1e-6)
        panels.extend(
            [
                ("eval mask", eval_mask.astype(np.float32), "gray", 0.0, 1.0),
                (
                    "|ours-raw|\neval mask",
                    np.where(eval_mask, np.abs(methods["ours_hole_only"] - raw), np.nan),
                    "magma",
                    0.0,
                    emax,
                ),
                (
                    "|NS-raw|\neval mask",
                    np.where(eval_mask, np.abs(methods["opencv_ns"] - raw), np.nan),
                    "magma",
                    0.0,
                    emax,
                ),
            ]
        )

    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax, (name, image, cmap, lo, hi) in zip(axes, panels):
        if image is None:
            ax.axis("off")
            continue
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def depth_to_uint8(depth, mask, lo, hi):
    depth = np.asarray(depth, dtype=np.float32)
    out = (depth - float(lo)) / max(float(hi - lo), 1e-6)
    out = np.nan_to_num(out, nan=0.0, neginf=0.0, posinf=1.0)
    out = np.clip(out, 0.0, 1.0)
    out[mask] = 0.0
    return np.round(out * 255.0).astype(np.uint8)


def export_external_inputs(case_dir, case_name, entries, raw_by_stem, mask_by_stem, args):
    depth_dir = os.path.join(case_dir, "depth_npy")
    mask_dir = os.path.join(case_dir, "mask_npy")
    frames_dir = os.path.join(case_dir, "export", "frames")
    masks_dir = os.path.join(case_dir, "export", "masks")
    for d in [depth_dir, mask_dir, frames_dir, masks_dir]:
        ensure_dir(d)

    valid_values = []
    for stem in entries:
        raw = raw_by_stem[stem]
        mask = mask_by_stem[stem]
        valid = (~mask) & np.isfinite(raw) & (raw > float(args.valid_min_depth)) & (
            raw <= float(args.valid_max_depth)
        )
        if valid.any():
            valid_values.append(raw[valid])
    if valid_values:
        valid_values = np.concatenate(valid_values)
        lo, hi = np.percentile(valid_values, [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    lo = float(lo)
    hi = float(hi if hi > lo else lo + 1.0)

    frame_mapping = []
    for idx, stem in enumerate(entries):
        raw = raw_by_stem[stem].astype(np.float32)
        mask = mask_by_stem[stem].astype(bool)
        corrupted = raw.copy()
        corrupted[mask] = 0.0
        np.save(os.path.join(depth_dir, f"{idx:04d}.npy"), corrupted)
        np.save(os.path.join(mask_dir, f"{idx:04d}.npy"), mask.astype(np.uint8))
        cv2.imwrite(os.path.join(frames_dir, f"{idx:04d}.png"), depth_to_uint8(corrupted, mask, lo, hi))
        cv2.imwrite(os.path.join(masks_dir, f"{idx:04d}.png"), (mask.astype(np.uint8) * 255))
        frame_mapping.append(
            {
                "frame_index": idx,
                "source_stem": stem,
                "source_path_240x320_m": os.path.abspath(
                    os.path.join(args.input_dir, f"{stem}.npy")
                ),
                "mask_ratio": float(mask.mean()),
            }
        )

    save_json(
        os.path.join(case_dir, "source_mapping.json"),
        {
            "source_dir_240x320_m": os.path.abspath(args.input_dir),
            "case_dir": os.path.abspath(case_dir),
            "case_name": case_name,
            "frame_count": len(entries),
            "mask_semantics": "nonzero/white means invalid region to inpaint",
            "frame_mapping": frame_mapping,
        },
    )
    save_json(
        os.path.join(case_dir, "export", "depth_meta.json"),
        {
            "source_depth_npy": os.path.abspath(depth_dir),
            "resolved_layout": "frame_dir_hw",
            "original_shape": [len(entries), 240, 320],
            "canonical_shape": [len(entries), 240, 320],
            "depth_min": lo,
            "depth_max": hi,
            "mask_source": os.path.abspath(mask_dir),
            "mask_semantics": "invalid_is_nonzero",
            "percentile_min": 1.0,
            "percentile_max": 99.0,
            "frame_prefix": "",
            "frames_dir_name": "frames",
            "masks_dir_name": "masks",
            "notes": "Frames are grayscale normalized depth with masked pixels set to zero.",
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="far_pic/noise_depth_240x320_m")
    parser.add_argument("--output_dir", default="output/far_pic_benchmark")
    parser.add_argument(
        "--checkpoint",
        default="output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hole_depth_threshold", type=float, default=0.0)
    parser.add_argument("--valid_min_depth", type=float, default=0.1)
    parser.add_argument("--valid_max_depth", type=float, default=9.9)
    parser.add_argument("--inpaint_radius", type=int, default=15)
    parser.add_argument("--synthetic_ratios", type=float, nargs="*", default=[0.05, 0.10, 0.20])
    parser.add_argument("--min_block_size", type=int, default=12)
    parser.add_argument("--max_block_size", type=int, default=72)
    parser.add_argument("--realhole_ratios", type=float, nargs="*", default=[0.05, 0.10, 0.20])
    parser.add_argument(
        "--realhole_template_dirs",
        nargs="*",
        default=[
            "output/far_pic_benchmark/bad_depth_mask_v1/masks",
            "output/far_pic_benchmark/zero_mask/masks",
        ],
    )
    parser.add_argument("--realhole_template_min_area", type=int, default=8)
    parser.add_argument("--realhole_template_max_area", type=int, default=12000)
    parser.add_argument("--realhole_template_pad", type=int, default=1)
    parser.add_argument("--realhole_min_scale", type=float, default=0.75)
    parser.add_argument("--realhole_max_scale", type=float, default=1.35)
    parser.add_argument("--realhole_min_valid_overlap", type=float, default=0.30)
    parser.add_argument("--realhole_max_attempts", type=int, default=1000)
    parser.add_argument("--bad_far_depth", type=float, default=6.0)
    parser.add_argument("--bad_residual_abs", type=float, default=0.45)
    parser.add_argument("--bad_median_ksize", type=int, default=7)
    parser.add_argument("--bad_min_component_area", type=int, default=2)
    parser.add_argument("--bad_max_component_area", type=int, default=2500)
    parser.add_argument("--bad_dilate", type=int, default=1)
    parser.add_argument("--sampling_mode", choices=["endpoint", "euler"], default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--vis_max_per_case", type=int, default=8)
    parser.add_argument(
        "--only_cases",
        nargs="*",
        default=None,
        help="Optional case names or kinds to run, e.g. synthetic_realhole synthetic_realhole_05.",
    )
    parser.add_argument(
        "--propainter_zero_case",
        default="propainter_depth_test/far_pic_noise_depth_240x320_m_zero_mask/restored_by_stem",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob(os.path.join(args.input_dir, "*.npy")), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No .npy files found under {args.input_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt_args = build_model(args.checkpoint, device)
    sampling_mode = args.sampling_mode or ckpt_args.get("eval_sampling_mode", "endpoint")
    sample_steps = int(args.sample_steps or ckpt_args.get("sample_steps", 8))
    infer_args = SimpleNamespace(
        hole_depth_threshold=args.hole_depth_threshold,
        valid_min_depth=args.valid_min_depth,
        valid_max_depth=args.valid_max_depth,
        anchor_inpaint_radius=None,
        post_clip_mode="valid_range",
        post_clip_percentiles=[0.5, 99.5],
    )

    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    raw_by_stem = {stem: np.load(path).astype(np.float32) for stem, path in zip(stems, paths)}
    realhole_templates = load_realhole_templates(args, raw_by_stem=raw_by_stem)
    if args.realhole_ratios and not realhole_templates:
        raise RuntimeError(
            "No real-hole templates found. Check --realhole_template_dirs or set realhole template filters less strictly."
        )

    case_defs = [{"name": "zero_mask", "kind": "zero", "ratio": None}]
    for ratio in args.synthetic_ratios:
        case_defs.append({"name": f"synthetic_{int(round(ratio * 100)):02d}", "kind": "synthetic", "ratio": ratio})
    for ratio in args.realhole_ratios:
        case_defs.append(
            {
                "name": f"synthetic_realhole_{int(round(ratio * 100)):02d}",
                "kind": "synthetic_realhole",
                "ratio": ratio,
            }
        )
    case_defs.append({"name": "bad_depth_mask_v1", "kind": "bad", "ratio": None})
    if args.only_cases:
        requested = set(args.only_cases)
        case_defs = [case for case in case_defs if case["name"] in requested or case["kind"] in requested]
        if not case_defs:
            raise ValueError(f"--only_cases matched no cases: {sorted(requested)}")

    benchmark_summary = {
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "checkpoint": args.checkpoint,
        "sampling_mode": sampling_mode,
        "sample_steps": sample_steps,
        "realhole_template_count": len(realhole_templates),
        "realhole_template_dirs": [os.path.abspath(p) for p in args.realhole_template_dirs],
        "cases": {},
    }

    for case in case_defs:
        case_name = case["name"]
        case_dir = os.path.join(args.output_dir, case_name)
        for sub in ["masks", "corrupted", "outputs/opencv_ns", "outputs/opencv_telea", "outputs/ours_hole_only", "outputs/ours_restored", "visualizations"]:
            ensure_dir(os.path.join(case_dir, sub))

        mask_by_stem = {}
        eval_mask_by_stem = {}
        mask_diagnostics = {}
        for idx, stem in enumerate(stems):
            raw = raw_by_stem[stem]
            zmask = zero_mask(raw, args)
            if case["kind"] == "zero":
                repair_mask = zmask
                eval_mask = None
                diagnostics = {"zero_ratio": float(zmask.mean())}
            elif case["kind"] == "synthetic":
                reliable = reliable_mask(raw, args) & (~zmask)
                rng = np.random.default_rng(int(args.seed) + idx * 1009 + int(round(case["ratio"] * 10000)))
                artificial = make_block_mask(
                    reliable,
                    rng,
                    float(case["ratio"]),
                    int(args.min_block_size),
                    int(args.max_block_size),
                )
                repair_mask = zmask | artificial
                eval_mask = artificial
                diagnostics = {
                    "zero_ratio": float(zmask.mean()),
                    "synthetic_ratio_over_reliable": float(artificial.sum() / max(int(reliable.sum()), 1)),
                    "synthetic_pixel_count": int(artificial.sum()),
                }
            elif case["kind"] == "synthetic_realhole":
                reliable = reliable_mask(raw, args) & (~zmask)
                rng = np.random.default_rng(
                    int(args.seed) + idx * 1009 + int(round(case["ratio"] * 10000)) + 777001
                )
                artificial, realhole_info = make_realhole_mask(
                    reliable,
                    realhole_templates,
                    rng,
                    float(case["ratio"]),
                    args,
                )
                repair_mask = zmask | artificial
                eval_mask = artificial
                diagnostics = {
                    "zero_ratio": float(zmask.mean()),
                    "synthetic_kind": "realhole_template",
                    **realhole_info,
                }
            else:
                repair_mask, diagnostics = build_bad_depth_mask(raw, args)
                eval_mask = None

            mask_by_stem[stem] = repair_mask.astype(bool)
            eval_mask_by_stem[stem] = eval_mask.astype(bool) if eval_mask is not None else None
            mask_diagnostics[stem] = diagnostics
            corrupted = raw.copy()
            corrupted[repair_mask] = 0.0
            np.save(os.path.join(case_dir, "masks", f"{stem}_mask.npy"), repair_mask.astype(np.uint8))
            np.save(os.path.join(case_dir, "corrupted", f"{stem}_corrupted.npy"), corrupted.astype(np.float32))

        export_external_inputs(
            os.path.join(case_dir, "external_inputs"),
            case_name,
            stems,
            raw_by_stem,
            mask_by_stem,
            args,
        )

        rows = []
        vis_saved = 0
        for idx, stem in enumerate(stems):
            raw = raw_by_stem[stem]
            repair_mask = mask_by_stem[stem]
            eval_mask = eval_mask_by_stem[stem]
            corrupted = raw.copy()
            corrupted[repair_mask] = 0.0

            ns = opencv_depth_inpaint(corrupted, repair_mask, method="ns", radius=args.inpaint_radius)
            telea = opencv_depth_inpaint(corrupted, repair_mask, method="telea", radius=args.inpaint_radius)

            condition_cpu = make_depth_condition(corrupted, ckpt_args, infer_args)
            condition = move_condition_to_device(condition_cpu, device)
            pred = predict_depth(model, condition, ckpt_args, sampling_mode, sample_steps)
            pred_raw = pred.detach().cpu().numpy()[0, 0].astype(np.float32)
            valid_for_clip = reliable_mask(raw, args) & (~repair_mask)
            pred_np, _ = clip_prediction(pred_raw, raw, valid_for_clip, infer_args)
            condition_hole = condition_cpu["hole"]
            ours_hole_only = np.where(condition_hole, pred_np, corrupted).astype(np.float32)

            methods = {
                "opencv_ns": ns,
                "opencv_telea": telea,
                "ours_hole_only": ours_hole_only,
                "ours_restored": pred_np,
            }
            prop_path = os.path.join(args.propainter_zero_case, f"{stem}_propainter_restored.npy")
            if case_name == "zero_mask" and os.path.exists(prop_path):
                methods["propainter"] = np.load(prop_path).astype(np.float32)
                ensure_dir(os.path.join(case_dir, "outputs", "propainter"))

            for name, arr in methods.items():
                ensure_dir(os.path.join(case_dir, "outputs", name))
                np.save(os.path.join(case_dir, "outputs", name, f"{stem}_{name}.npy"), arr.astype(np.float32))

            method_summary = {
                name: summarize_method(arr, raw, repair_mask, eval_mask=eval_mask)
                for name, arr in methods.items()
            }
            row = {
                "source_stem": stem,
                "repair_mask_ratio": float(repair_mask.mean()),
                "eval_mask_ratio": float(eval_mask.mean()) if eval_mask is not None else None,
                "mask_diagnostics": mask_diagnostics[stem],
                "raw_valid_min_p50_p95_p99_max": finite_percentiles(
                    raw[reliable_mask(raw, args)], [0, 50, 95, 99, 100]
                ),
                "methods": method_summary,
            }
            rows.append(row)

            if vis_saved < int(args.vis_max_per_case):
                title = f"{case_name} | {stem}"
                if eval_mask is not None:
                    ours_mae = method_summary["ours_hole_only"].get("eval_mask_mae")
                    ns_mae = method_summary["opencv_ns"].get("eval_mask_mae")
                    title += f" | NS={ns_mae:.3f} ours={ours_mae:.3f}"
                save_visualization(
                    os.path.join(case_dir, "visualizations", f"{idx:02d}_{stem}.png"),
                    title,
                    raw,
                    repair_mask,
                    methods,
                    eval_mask=eval_mask,
                )
                vis_saved += 1

            print(
                f"{case_name} [{idx + 1:02d}/{len(stems):02d}] {stem} "
                f"mask={repair_mask.mean():.3f}"
            )

        case_summary = {
            "case_name": case_name,
            "kind": case["kind"],
            "ratio": case.get("ratio"),
            "realhole_template_count": len(realhole_templates) if case["kind"] == "synthetic_realhole" else None,
            "aggregate": aggregate_rows(rows),
            "per_sample": rows,
            "external_inputs": os.path.abspath(os.path.join(case_dir, "external_inputs")),
        }
        save_json(os.path.join(case_dir, "summary.json"), case_summary)
        benchmark_summary["cases"][case_name] = {
            "summary_path": os.path.abspath(os.path.join(case_dir, "summary.json")),
            "aggregate": case_summary["aggregate"],
            "external_inputs": case_summary["external_inputs"],
        }

    save_json(os.path.join(args.output_dir, "summary.json"), benchmark_summary)
    print(json.dumps(benchmark_summary["cases"], indent=2))
    print(f"Saved benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
