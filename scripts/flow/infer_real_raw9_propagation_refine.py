import argparse
import json
import os

import numpy as np
import torch

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from infer_real_raw9_flow import (
    RAW9_TRANSFORM_CHOICES,
    build_threshold_hole,
    collect_pairs,
    depth_to_meters,
    ensure_dir,
    filter_pairs_by_samples,
    fill_threshold_then_added,
    finite_stats,
    make_condition,
    safe_mean_abs,
    save_visualization,
    summarize_rows,
)
from train_depth_flow_restoration import predict_endpoint_norm, sample_flow
from train_real_raw9_flow_finetune import (
    add_checkpoint_args as add_flow_checkpoint_args,
    build_model as build_flow_model,
    move_condition_to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sensor-only propagation-refinement restoration on paired real raw9/depth files."
    )
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--samples", type=str, nargs="+", default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--amplitude_mode", type=str, default="iq6", choices=["iq6", "raw_258"])
    parser.add_argument(
        "--raw9_transform",
        type=str,
        default="checkpoint",
        choices=RAW9_TRANSFORM_CHOICES,
        help="Spatial transform applied to raw9 before amplitude features.",
    )
    parser.add_argument("--device", type=str, default=None)
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
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
    parser.add_argument("--post_clip_mode", type=str, default="valid_range",
                        choices=["none", "valid_range", "valid_percentile"])
    parser.add_argument("--post_clip_percentiles", type=float, nargs=2, default=[0.5, 99.5])

    parser.add_argument("--hole_mask_mode", type=str, default="amp_speckle_cleaned",
                        choices=["threshold", "cleaned", "speckle_cleaned", "amp_speckle_cleaned"])
    parser.add_argument("--clean_outlier_abs", type=float, default=0.35)
    parser.add_argument("--clean_outlier_mad_scale", type=float, default=6.0)
    parser.add_argument("--clean_median_ksize", type=int, default=7)
    parser.add_argument("--clean_dilate", type=int, default=1)
    parser.add_argument("--clean_min_component_area", type=int, default=6)
    parser.add_argument("--speckle_window", type=int, default=11)
    parser.add_argument("--speckle_density_threshold", type=float, default=0.10)
    parser.add_argument("--speckle_residual_abs", type=float, default=0.18)
    parser.add_argument("--speckle_link_radius", type=int, default=2)
    parser.add_argument("--speckle_min_component_area", type=int, default=4)
    parser.add_argument("--speckle_max_component_area", type=int, default=9000)
    parser.add_argument("--speckle_max_bbox_side", type=int, default=140)
    parser.add_argument("--speckle_amp_ring_radius", type=int, default=7)
    parser.add_argument("--speckle_amp_ratio_min", type=float, default=2.5)
    parser.add_argument("--speckle_amp_delta_min", type=float, default=4000.0)
    parser.add_argument("--speckle_amp_abs_min", type=float, default=8000.0)

    parser.add_argument("--split_added_fill", action="store_true", default=False)
    parser.add_argument("--split_added_mode", type=str, default="anchor_ns",
                        choices=["ns", "plane", "anchor_ns"])
    parser.add_argument("--split_added_inpaint_radius", type=int, default=5)
    parser.add_argument("--vis_max_samples", type=int, default=1000000)
    parser.add_argument("--no_visualize", action="store_true")
    return parser.parse_args()


def build_model(ckpt_args, device):
    in_channels = 4 + int(bool(ckpt_args.get("include_hole_distance", False))) + 4
    model = build_depth_backbone(
        ckpt_args.get("backbone", "propagation_refine"),
        in_channels=in_channels,
        base_channels=int(ckpt_args.get("base_channels", 32)),
        out_channels=1,
        res_blocks=int(ckpt_args.get("res_blocks", 1)),
        propagation_steps=int(ckpt_args.get("propagation_steps", 6)),
        propagation_hidden_scale=float(ckpt_args.get("propagation_hidden_scale", 1.0)),
        refine_dilate_radius=int(ckpt_args.get("refine_dilate_radius", 3)),
        residual_scale=float(ckpt_args.get("residual_scale", 1.5)),
    ).to(device)
    return model


def build_frozen_flow_anchor(pretrained_checkpoint, device):
    flow_ckpt = load_checkpoint(pretrained_checkpoint, device)
    flow_ckpt_args = flow_ckpt.get("args", {})
    flow_args = argparse.Namespace()
    flow_args.pretrained_checkpoint = pretrained_checkpoint
    add_flow_checkpoint_args(flow_args, flow_ckpt_args)
    flow_model = build_flow_model(flow_ckpt, flow_ckpt_args, device)
    flow_model.eval()
    for param in flow_model.parameters():
        param.requires_grad_(False)
    return flow_model, flow_args


@torch.no_grad()
def predict_flow_anchor_norm(flow_model, flow_args, condition):
    if flow_args.eval_sampling_mode == "endpoint":
        return predict_endpoint_norm(
            flow_model,
            condition,
            flow_args.time_channels,
            flow_args.max_velocity_norm,
            flow_args.clip_norm_depth,
            flow_args.velocity_scale,
        )
    return sample_flow(
        flow_model,
        condition,
        flow_args.time_channels,
        flow_args.max_velocity_norm,
        flow_args.sample_steps,
        flow_args.clip_norm_depth,
        flow_args.velocity_scale,
    )


@torch.no_grad()
def prepare_refine_condition(condition, flow_model, flow_args):
    flow_anchor_norm = predict_flow_anchor_norm(flow_model, flow_args, condition).detach()
    condition = dict(condition)
    condition["x"] = condition["x"].clone()
    condition["x"][:, 0:1] = flow_anchor_norm
    condition["anchor_norm"] = flow_anchor_norm
    scale = condition["scale"].view(-1, 1, 1, 1)
    center = condition["center"].view(-1, 1, 1, 1)
    condition["depth_anchor"] = flow_anchor_norm * scale + center
    return condition


@torch.no_grad()
def predict_depth(model, condition):
    out = model(condition["x"])
    pred_norm = out["refined"]
    coarse_norm = out["coarse"]
    scale = condition["scale"].view(-1, 1, 1, 1)
    center = condition["center"].view(-1, 1, 1, 1)
    pred = pred_norm * scale + center
    coarse = coarse_norm * scale + center
    return pred, coarse


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


def main():
    args = parse_args()
    pairs = filter_pairs_by_samples(collect_pairs(args.raw_dir, args.depth_dir), args.samples)
    if not pairs:
        raise FileNotFoundError(f"No paired .npy files found under {args.raw_dir} and {args.depth_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    checkpoint_raw9_transform = str(ckpt_args.get("raw9_transform", "none") or "none")
    args.checkpoint_raw9_transform = checkpoint_raw9_transform
    args.raw9_transform_effective = (
        checkpoint_raw9_transform
        if str(args.raw9_transform) == "checkpoint"
        else str(args.raw9_transform)
    )
    model = build_model(ckpt_args, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    flow_anchor_model, flow_anchor_args = build_frozen_flow_anchor(ckpt_args["pretrained_checkpoint"], device)

    for subdir in [
        "restored",
        "coarse",
        "anchor",
        "hole_only",
        "split_hole_only",
        "hole_mask",
        "threshold_hole_mask",
        "visualizations",
    ]:
        ensure_dir(os.path.join(args.output_dir, subdir))

    rows = []
    for index, (stem, raw_path, depth_path) in enumerate(pairs):
        raw9 = np.load(raw_path).astype(np.float32)
        depth = depth_to_meters(np.load(depth_path), args.depth_unit).astype(np.float32)
        condition_cpu = make_condition(depth, raw9, ckpt_args, args)
        condition = move_condition_to_device(condition_cpu, device)
        condition = prepare_refine_condition(condition, flow_anchor_model, flow_anchor_args)
        pred, coarse = predict_depth(model, condition)
        pred_raw_np = pred.detach().cpu().numpy()[0, 0].astype(np.float32)
        coarse_raw_np = coarse.detach().cpu().numpy()[0, 0].astype(np.float32)

        hole = condition_cpu["hole"]
        threshold_hole = build_threshold_hole(depth, args)
        reliable = condition_cpu["reliable"]
        anchor = condition["depth_anchor"].detach().cpu().numpy()[0, 0].astype(np.float32)
        ns_anchor = condition_cpu["anchor"]
        pred_np, clip_bounds = clip_prediction(pred_raw_np, depth, reliable, args)
        coarse_np, _ = clip_prediction(coarse_raw_np, depth, reliable, args)
        hole_only = np.where(hole, pred_np, depth).astype(np.float32)
        split_hole_only = None
        split_added_components = []
        if args.split_added_fill:
            split_hole_only, split_added_components = fill_threshold_then_added(
                depth,
                threshold_hole,
                hole,
                pred_np,
                anchor,
                args,
            )

        restored_path = os.path.join(args.output_dir, "restored", f"{stem}_restored.npy")
        coarse_path = os.path.join(args.output_dir, "coarse", f"{stem}_coarse.npy")
        anchor_path = os.path.join(args.output_dir, "anchor", f"{stem}_anchor.npy")
        hole_only_path = os.path.join(args.output_dir, "hole_only", f"{stem}_hole_only.npy")
        split_hole_only_path = os.path.join(args.output_dir, "split_hole_only", f"{stem}_split_hole_only.npy")
        hole_mask_path = os.path.join(args.output_dir, "hole_mask", f"{stem}_hole_mask.npy")
        threshold_hole_mask_path = os.path.join(
            args.output_dir, "threshold_hole_mask", f"{stem}_threshold_hole_mask.npy"
        )
        np.save(restored_path, pred_np)
        np.save(coarse_path, coarse_np)
        np.save(anchor_path, anchor)
        np.save(hole_only_path, hole_only)
        np.save(hole_mask_path, hole.astype(np.uint8))
        np.save(threshold_hole_mask_path, threshold_hole.astype(np.uint8))
        if split_hole_only is not None:
            np.save(split_hole_only_path, split_hole_only)

        diff_model_anchor = pred_np - anchor
        diff_model_raw = pred_np - depth
        diff_split_anchor = None if split_hole_only is None else split_hole_only - anchor
        threshold_count = int(threshold_hole.sum())
        cleaned_count = int(hole.sum())
        added_count = max(0, cleaned_count - threshold_count)
        row = {
            "name": stem,
            "raw_path": raw_path,
            "depth_path": depth_path,
            "restored_path": restored_path,
            "coarse_path": coarse_path,
            "anchor_path": anchor_path,
            "hole_only_path": hole_only_path,
            "split_hole_only_path": split_hole_only_path if split_hole_only is not None else None,
            "hole_mask_path": hole_mask_path,
            "threshold_hole_mask_path": threshold_hole_mask_path,
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
            "threshold_hole_ratio": float(threshold_hole.mean()),
            "cleaned_added_ratio": float(added_count / max(1, hole.size)),
            "mask_diagnostics": condition_cpu["mask_diagnostics"],
            "valid_ratio": float(reliable.mean()),
            "post_clip_mode": args.post_clip_mode,
            "post_clip_bounds": clip_bounds,
            "anchor_source": f"frozen_flow_{flow_anchor_args.eval_sampling_mode}",
            "norm_center": condition_cpu["center_value"],
            "norm_scale": condition_cpu["scale_value"],
            "anchor_inpaint_radius": condition_cpu["radius"],
            "raw_valid_stats": finite_stats(depth[reliable]),
            "ns_anchor_stats": finite_stats(ns_anchor),
            "anchor_stats": finite_stats(anchor),
            "model_raw_stats": finite_stats(pred_raw_np),
            "model_stats": finite_stats(pred_np),
            "coarse_stats": finite_stats(coarse_np),
            "hole_only_stats": finite_stats(hole_only),
            "split_hole_only_stats": finite_stats(split_hole_only) if split_hole_only is not None else None,
            "mean_abs_model_anchor_hole": safe_mean_abs(diff_model_anchor, hole),
            "mean_abs_model_anchor_valid": safe_mean_abs(diff_model_anchor, reliable),
            "mean_abs_model_raw_valid": safe_mean_abs(diff_model_raw, reliable),
            "mean_abs_hole_only_anchor_hole": safe_mean_abs(hole_only - anchor, hole),
            "mean_abs_split_anchor_hole": (
                safe_mean_abs(diff_split_anchor, hole) if diff_split_anchor is not None else None
            ),
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
        "aggregate": summarize_rows(rows),
        "per_sample": rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"Saved real raw9 propagation refine results to {args.output_dir}")


if __name__ == "__main__":
    main()
