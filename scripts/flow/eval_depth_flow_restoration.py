import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from eval_depth_restoration import (
    aggregate_metrics,
    collect_eval_paths,
    load_checkpoint,
    metric_delta,
    sample_metrics,
    save_visualization,
    select_ranked_visualization_rows,
    tensor_to_numpy,
)
from depth_restoration_backbones import build_depth_backbone
from train_depth_completion import move_batch_to_device
from train_depth_flow_restoration import flow_model_in_channels, predict_endpoint_norm, sample_flow
from train_depth_restoration import DepthRestorationCacheDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a conditional-flow depth restoration model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--sample_list", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample_steps", type=int, default=None,
                        help="Override checkpoint sample_steps for Euler sampling.")
    parser.add_argument("--sampling_mode", type=str, default=None,
                        choices=["euler", "endpoint"],
                        help=(
                            "Override checkpoint eval_sampling_mode. euler integrates the flow; "
                            "endpoint directly predicts anchor -> restored depth at t=0."
                        ))
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--vis_max_samples", type=int, default=20)
    parser.add_argument("--vis_error_percentile", type=float, default=99.0)
    parser.add_argument("--vis_rank", type=str, default="first",
                        choices=[
                            "first",
                            "best_hole",
                            "worst_hole",
                            "best_worst_hole",
                            "best_global",
                            "worst_global",
                            "best_worst_global",
                        ])
    parser.add_argument("--vis_rank_baseline", type=str, default="anchor",
                        choices=["anchor", "base", "noisy"])
    parser.add_argument(
        "--preserve_observed",
        action="store_true",
        default=False,
        help="Replace predictions on observed pixels with the input observed depth before scoring.",
    )
    return parser.parse_args()


@torch.no_grad()
def predict_batch(model, batch, ckpt_args, sample_steps, sampling_mode):
    if sampling_mode == "endpoint":
        pred_norm = predict_endpoint_norm(
            model,
            batch,
            int(ckpt_args.get("time_channels", 16)),
            float(ckpt_args.get("max_velocity_norm", 4.0)),
            float(ckpt_args.get("clip_norm_depth", 8.0)),
            float(ckpt_args.get("velocity_scale", 1.0)),
        )
    else:
        pred_norm = sample_flow(
            model,
            batch,
            int(ckpt_args.get("time_channels", 16)),
            float(ckpt_args.get("max_velocity_norm", 4.0)),
            sample_steps,
            float(ckpt_args.get("clip_norm_depth", 8.0)),
            float(ckpt_args.get("velocity_scale", 1.0)),
        )
    scale = batch["scale"].view(-1, 1, 1, 1)
    center = batch["center"].view(-1, 1, 1, 1)
    return pred_norm * scale + center


def apply_observed_constraint(pred, batch, preserve_observed):
    if not preserve_observed:
        return pred
    observed = ~batch["hole_mask"].bool()
    return torch.where(observed, batch["depth_noisy"], pred)


@torch.no_grad()
def save_ranked_visualizations(rows, dataset_kwargs, model, device, ckpt_args, sample_steps, sampling_mode, out_dir, args):
    if not rows:
        return 0

    dataset = DepthRestorationCacheDataset([row["path"] for row in rows], **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    rows_by_path = {row["path"]: (index, row) for index, row in enumerate(rows)}
    saved = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = predict_batch(model, batch, ckpt_args, sample_steps, sampling_mode)
        pred = apply_observed_constraint(pred, batch, args.preserve_observed)
        batch_size = pred.shape[0]

        for i in range(batch_size):
            path = batch["path"][i]
            rank_index, row = rows_by_path[path]
            sample_name = batch["sample_name"][i]
            label = row.get("_vis_label", args.vis_rank)
            region = "hole" if "hole" in label else "global"
            baseline = args.vis_rank_baseline
            delta = metric_delta(row, region, baseline)
            title = (
                f"{label} {sample_name} | "
                f"{baseline}_{region}={row[f'{baseline}_{region}_mae']:.4f} "
                f"model_{region}={row[f'model_{region}_mae']:.4f} "
                f"delta={delta:.4f}"
            )

            depths = {
                "model": tensor_to_numpy(pred, i),
                "anchor": tensor_to_numpy(batch["depth_anchor"], i),
                "noisy": tensor_to_numpy(batch["depth_noisy"], i),
                "base": tensor_to_numpy(batch["depth_base"], i),
            }
            gt = tensor_to_numpy(batch["gt_depth"], i)
            hole = tensor_to_numpy(batch["hole_mask"].float(), i) > 0.5

            safe_label = label.replace("/", "_")
            safe_name = sample_name.replace("/", "_")
            out_png = os.path.join(
                out_dir,
                "visualizations",
                f"vis_{rank_index:03d}_{safe_label}_{safe_name}.png",
            )
            save_visualization(out_png, title, depths, gt, hole, args)
            saved += 1

    return saved


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    paths, path_source = collect_eval_paths(args, ckpt_args)
    if not paths:
        raise FileNotFoundError("No cache samples found for evaluation.")

    out_dir = args.output_dir
    if out_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    dataset_kwargs = {
        "input_mode": ckpt_args.get("input_mode", "noisy"),
        "include_hole_distance": ckpt_args.get("include_hole_distance", False),
        "anchor_mode": ckpt_args.get("anchor_mode", "noisy_ns"),
        "anchor_inpaint_radius": ckpt_args.get("anchor_inpaint_radius") or 15,
        "norm_percentiles": ckpt_args.get("norm_percentiles", [5.0, 95.0]),
        "min_depth_scale": ckpt_args.get("min_depth_scale", 0.25),
        "clip_norm_depth": ckpt_args.get("clip_norm_depth", 8.0),
        "feature_percentile": ckpt_args.get("feature_percentile", 99.0),
        "feature_clip": ckpt_args.get("feature_clip", 3.0),
        "iq_clip": ckpt_args.get("iq_clip", 3.0),
    }
    dataset = DepthRestorationCacheDataset(paths, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    time_channels = int(ckpt_args.get("time_channels", 16))
    in_channels = flow_model_in_channels(dataset.input_channels, time_channels)
    backbone = ckpt_args.get("backbone", "resunet")
    model = build_depth_backbone(
        backbone,
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

    sample_steps = int(args.sample_steps or ckpt_args.get("sample_steps", 8))
    sampling_mode = args.sampling_mode or ckpt_args.get("eval_sampling_mode", "euler")

    per_sample = []
    vis_saved = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred = predict_batch(model, batch, ckpt_args, sample_steps, sampling_mode)
        pred = apply_observed_constraint(pred, batch, args.preserve_observed)
        batch_size = pred.shape[0]

        for i in range(batch_size):
            sample_name = batch["sample_name"][i]
            depths = {
                "model": tensor_to_numpy(pred, i),
                "anchor": tensor_to_numpy(batch["depth_anchor"], i),
                "noisy": tensor_to_numpy(batch["depth_noisy"], i),
                "base": tensor_to_numpy(batch["depth_base"], i),
            }
            gt = tensor_to_numpy(batch["gt_depth"], i)
            valid = tensor_to_numpy(batch["valid_mask"].float(), i) > 0.5
            hole = tensor_to_numpy(batch["hole_mask"].float(), i) > 0.5
            row = {
                "sample_name": sample_name,
                "path": batch["path"][i],
            }
            row.update(sample_metrics(depths, gt, valid, hole))
            per_sample.append(row)

            if args.visualize and args.vis_rank == "first" and vis_saved < args.vis_max_samples:
                safe_name = sample_name.replace("/", "_")
                out_png = os.path.join(out_dir, "visualizations", f"vis_{safe_name}.png")
                save_visualization(out_png, sample_name, depths, gt, hole, args)
                vis_saved += 1

    aggregate = aggregate_metrics(per_sample)
    if args.visualize and args.vis_rank != "first":
        ranked_rows = select_ranked_visualization_rows(per_sample, args)
        vis_saved = save_ranked_visualizations(
            ranked_rows,
            dataset_kwargs,
            model,
            device,
            ckpt_args,
            sample_steps,
            sampling_mode,
            out_dir,
            args,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "cache_dir": args.cache_dir or ckpt_args.get("cache_dir"),
        "sample_list": args.sample_list,
        "path_source": path_source,
        "split": args.split,
        "num_samples": len(per_sample),
        "visualized_samples": vis_saved,
        "preserve_observed": args.preserve_observed,
        "vis_rank": args.vis_rank if args.visualize else None,
        "vis_rank_baseline": args.vis_rank_baseline if args.visualize and args.vis_rank != "first" else None,
        "method": "conditional_rectified_flow",
        "backbone": backbone,
        "input_mode": dataset_kwargs["input_mode"],
        "anchor_mode": dataset_kwargs["anchor_mode"],
        "include_hole_distance": dataset_kwargs["include_hole_distance"],
        "base_channels": int(ckpt_args.get("base_channels", 32)),
        "res_blocks": int(ckpt_args.get("res_blocks", 2)),
        "transformer_layers": int(ckpt_args.get("transformer_layers", 0 if backbone != "transformer_bottleneck" else 2)),
        "transformer_heads": int(ckpt_args.get("transformer_heads", 8)),
        "transformer_pool": int(ckpt_args.get("transformer_pool", 2)),
        "time_channels": time_channels,
        "bridge_noise": float(ckpt_args.get("bridge_noise", 0.0)),
        "endpoint_weight": float(ckpt_args.get("endpoint_weight", 0.0)),
        "sampling_mode": sampling_mode,
        "sample_steps": sample_steps,
        "max_velocity_norm": float(ckpt_args.get("max_velocity_norm", 4.0)),
        "velocity_scale": float(ckpt_args.get("velocity_scale", 1.0)),
        "hole_definition": "hole_mask & valid_mask",
        "gt_invalid_policy": "excluded_from_loss_and_metrics",
        "aggregate": aggregate,
    }

    with open(os.path.join(out_dir, "per_sample_results.json"), "w") as f:
        json.dump(per_sample, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved eval results to {out_dir}")


if __name__ == "__main__":
    main()
