import argparse
import json
import os
from glob import glob

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from inference_depth_postprocess import opencv_depth_inpaint
from train_depth_flow_restoration import flow_model_in_channels, predict_endpoint_norm, sample_flow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run conditional-flow restoration on real depth-only .npy files."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/real_depth_flow_noisy_ns_endpoint",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["endpoint", "euler"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--hole_depth_threshold", type=float, default=0.0)
    parser.add_argument("--valid_min_depth", type=float, default=0.1)
    parser.add_argument("--valid_max_depth", type=float, default=9.9)
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


def make_depth_condition(depth, ckpt_args, args):
    depth = np.asarray(depth, dtype=np.float32)
    hole = (~np.isfinite(depth)) | (depth <= float(args.hole_depth_threshold))
    confidence = (~hole).astype(np.float32)

    radius = (
        int(args.anchor_inpaint_radius)
        if args.anchor_inpaint_radius is not None
        else int(ckpt_args.get("anchor_inpaint_radius", 15))
    )
    anchor = opencv_depth_inpaint(depth, hole, method="ns", radius=radius).astype(np.float32)

    stat_mask = (
        (~hole)
        & np.isfinite(anchor)
        & (anchor > float(args.valid_min_depth))
        & (anchor < float(args.valid_max_depth))
    )
    if stat_mask.sum() == 0:
        stat_mask = np.isfinite(anchor) & (anchor > float(args.valid_min_depth)) & (
            anchor < float(args.valid_max_depth)
        )
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

    channels = [anchor_norm, noisy_norm, hole.astype(np.float32), confidence]
    if bool(ckpt_args.get("include_hole_distance", False)):
        dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        dist = np.clip(dist / max(float(radius), 1.0), 0.0, 1.0)
        channels.append(dist)

    input_mode = ckpt_args.get("input_mode", "noisy")
    if input_mode != "noisy":
        raise ValueError(
            f"Checkpoint input_mode={input_mode!r} requires features not present in depth-only real data. "
            "Use a depth-only checkpoint such as output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt."
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
        "confidence": confidence,
        "center_value": center,
        "scale_value": scale,
        "radius": radius,
    }


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


def save_visualization(path, name, depth, hole, anchor, pred, hole_only):
    ensure_dir(os.path.dirname(path))
    vmin, vmax = image_limits(depth[~hole], anchor, pred, hole_only)
    delta = np.abs(pred - anchor)
    dmax = float(np.percentile(delta[np.isfinite(delta)], 98.0)) if np.isfinite(delta).any() else 1.0
    dmax = max(dmax, 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("raw depth", depth, "viridis", vmin, vmax),
        ("hole mask", hole.astype(np.float32), "gray", 0.0, 1.0),
        ("NS anchor", anchor, "viridis", vmin, vmax),
        ("flow restored", pred, "viridis", vmin, vmax),
        ("hole-only blend", hole_only, "viridis", vmin, vmax),
        ("|flow-anchor|", delta, "magma", 0.0, dmax),
    ]

    for ax, (title, image, cmap_name, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap_name, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(name)
    fig.savefig(path, dpi=140)
    plt.close(fig)


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


def summarize_rows(rows):
    if not rows:
        return {}

    def collect(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return np.asarray(values, dtype=np.float64)

    summary = {"num_samples": len(rows)}
    for key in [
        "hole_ratio",
        "mean_abs_model_anchor_hole",
        "mean_abs_model_raw_valid",
        "mean_abs_hole_only_anchor_hole",
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
    paths = sorted(glob(os.path.join(args.input_dir, "*.npy")), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No .npy depth files found under {args.input_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})

    input_mode = ckpt_args.get("input_mode", "noisy")
    if input_mode != "noisy":
        raise ValueError(
            f"Depth-only real inference expects input_mode='noisy', got {input_mode!r}. "
            "Use the no-amplitude endpoint checkpoint first."
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

    sampling_mode = args.sampling_mode or ckpt_args.get("eval_sampling_mode", "endpoint")
    sample_steps = int(args.sample_steps or ckpt_args.get("sample_steps", 8))

    for subdir in ["restored", "restored_raw", "anchor", "hole_only", "hole_mask", "visualizations"]:
        ensure_dir(os.path.join(args.output_dir, subdir))

    rows = []
    for index, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        depth = np.load(path).astype(np.float32)
        if depth.ndim != 2:
            raise ValueError(f"{path} must be a 2D depth map, got shape {depth.shape}")

        condition_cpu = make_depth_condition(depth, ckpt_args, args)
        condition = move_condition_to_device(condition_cpu, device)
        pred = predict_depth(model, condition, ckpt_args, sampling_mode, sample_steps)
        pred_raw_np = pred.detach().cpu().numpy()[0, 0].astype(np.float32)

        hole = condition_cpu["hole"]
        anchor = condition_cpu["anchor"]
        valid = (~hole) & np.isfinite(depth)
        pred_np, clip_bounds = clip_prediction(pred_raw_np, depth, valid, args)
        hole_only = np.where(hole, pred_np, depth).astype(np.float32)

        restored_path = os.path.join(args.output_dir, "restored", f"{name}_restored.npy")
        restored_raw_path = os.path.join(args.output_dir, "restored_raw", f"{name}_restored_raw.npy")
        anchor_path = os.path.join(args.output_dir, "anchor", f"{name}_anchor.npy")
        hole_only_path = os.path.join(args.output_dir, "hole_only", f"{name}_hole_only.npy")
        hole_mask_path = os.path.join(args.output_dir, "hole_mask", f"{name}_hole_mask.npy")
        np.save(restored_path, pred_np)
        np.save(restored_raw_path, pred_raw_np)
        np.save(anchor_path, anchor)
        np.save(hole_only_path, hole_only)
        np.save(hole_mask_path, hole.astype(np.uint8))

        diff_model_anchor = pred_np - anchor
        diff_model_raw = pred_np - depth
        row = {
            "name": name,
            "input_path": path,
            "restored_path": restored_path,
            "restored_raw_path": restored_raw_path,
            "anchor_path": anchor_path,
            "hole_only_path": hole_only_path,
            "shape": list(depth.shape),
            "hole_ratio": float(hole.mean()),
            "valid_ratio": float(valid.mean()),
            "post_clip_mode": args.post_clip_mode,
            "post_clip_bounds": clip_bounds,
            "norm_center": condition_cpu["center_value"],
            "norm_scale": condition_cpu["scale_value"],
            "anchor_inpaint_radius": condition_cpu["radius"],
            "raw_valid_stats": finite_stats(depth[valid]),
            "anchor_stats": finite_stats(anchor),
            "model_raw_stats": finite_stats(pred_raw_np),
            "model_stats": finite_stats(pred_np),
            "hole_only_stats": finite_stats(hole_only),
            "mean_abs_model_anchor_hole": safe_mean_abs(diff_model_anchor, hole),
            "mean_abs_model_anchor_valid": safe_mean_abs(diff_model_anchor, valid),
            "mean_abs_model_raw_valid": safe_mean_abs(diff_model_raw, valid),
            "mean_abs_hole_only_anchor_hole": safe_mean_abs(hole_only - anchor, hole),
        }
        rows.append(row)

        if not args.no_visualize and index < int(args.vis_max_samples):
            vis_path = os.path.join(args.output_dir, "visualizations", f"{name}.png")
            save_visualization(vis_path, name, depth, hole, anchor, pred_np, hole_only)

        print(
            f"[{index + 1:03d}/{len(paths):03d}] {name} "
            f"hole={row['hole_ratio']:.3f} "
            f"|model-anchor|_hole={row['mean_abs_model_anchor_hole'] or 0.0:.4f} "
            f"|model-raw|_valid={row['mean_abs_model_raw_valid'] or 0.0:.4f}"
        )

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_args": ckpt_args,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "sampling_mode": sampling_mode,
        "sample_steps": sample_steps,
        "aggregate": summarize_rows(rows),
        "per_sample": rows,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved real-depth inference results to {args.output_dir}")


if __name__ == "__main__":
    main()
