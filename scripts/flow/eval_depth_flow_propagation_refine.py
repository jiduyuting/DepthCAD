import _bootstrap
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from depth_restoration_backbones import build_depth_backbone
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
from train_depth_completion import move_batch_to_device
from train_depth_flow_propagation_refine import (
    PropagationRefineCacheDataset,
    build_flow_model_from_checkpoint,
    build_refine_model,
    flow_dataset_kwargs,
    prepare_propagation_batch,
    predict_refined_norm,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Flow-anchor propagation-refinement checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--anchor_cache_dir",
        type=str,
        default=None,
        help="Optional directory containing precomputed normalized Flow anchors.",
    )
    parser.add_argument("--sample_list", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default=None,
        help="Override the frozen Flow checkpoint recorded in the refine checkpoint.",
    )
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--vis_max_samples", type=int, default=20)
    parser.add_argument("--vis_error_percentile", type=float, default=99.0)
    parser.add_argument(
        "--vis_rank",
        type=str,
        default="first",
        choices=[
            "first",
            "best_hole",
            "worst_hole",
            "best_worst_hole",
            "best_global",
            "worst_global",
            "best_worst_global",
        ],
    )
    parser.add_argument("--vis_rank_baseline", type=str, default="anchor", choices=["anchor", "base", "noisy"])
    parser.add_argument(
        "--preserve_observed",
        action="store_true",
        default=False,
        help="Replace predictions on observed pixels with the input observed depth before scoring.",
    )
    return parser.parse_args()


def build_refine_from_checkpoint(ckpt, ckpt_args, condition_channels, device):
    args = argparse.Namespace(
        base_channels=int(ckpt_args.get("base_channels", 32)),
        res_blocks=int(ckpt_args.get("res_blocks", 1)),
        propagation_steps=int(ckpt_args.get("propagation_steps", 6)),
        propagation_hidden_scale=float(ckpt_args.get("propagation_hidden_scale", 1.0)),
        refine_dilate_radius=int(ckpt_args.get("refine_dilate_radius", 3)),
        residual_scale=float(ckpt_args.get("residual_scale", 1.5)),
        global_refine=bool(ckpt_args.get("global_refine", False)),
    )
    model = build_refine_model(condition_channels, args, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, batch):
    pred_norm, coarse_norm, _ = predict_refined_norm(model, batch)
    scale = batch["scale"].view(-1, 1, 1, 1)
    center = batch["center"].view(-1, 1, 1, 1)
    pred = pred_norm * scale + center
    coarse = coarse_norm * scale + center
    return pred, coarse


def apply_observed_constraint(pred, batch, preserve_observed):
    if not preserve_observed:
        return pred
    observed = ~batch["hole_mask"].bool()
    return torch.where(observed, batch["depth_noisy"], pred)


def aggregate_metrics_with_coarse(per_sample):
    aggregate = aggregate_metrics(per_sample)
    abs_totals = {}
    sq_totals = {}
    counts = {}
    for row in per_sample:
        for region in ["global", "hole", "valid"]:
            mae = row.get(f"coarse_{region}_mae")
            rmse = row.get(f"coarse_{region}_rmse")
            count = row.get(f"coarse_{region}_count", 0)
            if mae is None or count == 0:
                continue
            key = f"coarse_{region}"
            abs_totals[key] = abs_totals.get(key, 0.0) + float(mae) * int(count)
            if rmse is not None:
                sq_totals[key] = sq_totals.get(key, 0.0) + (float(rmse) * float(rmse)) * int(count)
            counts[key] = counts.get(key, 0) + int(count)

    for key, total in abs_totals.items():
        aggregate[f"{key}_mae"] = total / max(counts[key], 1)
        if key in sq_totals:
            aggregate[f"{key}_rmse"] = math.sqrt(sq_totals[key] / max(counts[key], 1))
        aggregate[f"{key}_count"] = counts[key]
    return aggregate


@torch.no_grad()
def save_ranked_visualizations(
    rows,
    dataset_kwargs,
    flow_model,
    flow_args,
    model,
    device,
    out_dir,
    args,
):
    if not rows:
        return 0

    dataset = PropagationRefineCacheDataset([row["path"] for row in rows], **dataset_kwargs)
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
        batch = prepare_propagation_batch(batch, flow_model, flow_args)
        pred, _ = predict_batch(model, batch)
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
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    flow_checkpoint = args.pretrained_checkpoint or ckpt_args.get("pretrained_checkpoint")
    if not flow_checkpoint:
        raise ValueError(
            "No frozen Flow checkpoint found. Provide --pretrained_checkpoint or evaluate a "
            "checkpoint trained by train_depth_flow_propagation_refine.py."
        )
    flow_ckpt = load_checkpoint(flow_checkpoint, device)
    flow_args = flow_ckpt.get("args", {})
    dataset_kwargs = flow_dataset_kwargs(flow_args)

    paths, path_source = collect_eval_paths(args, ckpt_args)
    if not paths:
        raise FileNotFoundError("No cache samples found for evaluation.")

    out_dir = args.output_dir
    if out_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    dataset_kwargs["anchor_cache_dir"] = args.anchor_cache_dir
    dataset = PropagationRefineCacheDataset(paths, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    condition_channels = dataset.input_channels
    flow_model = None
    if not args.anchor_cache_dir:
        flow_model = build_flow_model_from_checkpoint(flow_ckpt, flow_args, condition_channels, device)
    model = build_refine_from_checkpoint(ckpt, ckpt_args, condition_channels, device)

    per_sample = []
    vis_saved = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        batch = prepare_propagation_batch(batch, flow_model, flow_args)
        pred, coarse = predict_batch(model, batch)
        pred = apply_observed_constraint(pred, batch, args.preserve_observed)
        batch_size = pred.shape[0]

        for i in range(batch_size):
            sample_name = batch["sample_name"][i]
            depths = {
                "model": tensor_to_numpy(pred, i),
                "coarse": tensor_to_numpy(coarse, i),
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
                save_visualization(
                    out_png,
                    sample_name,
                    {
                        "model": depths["model"],
                        "anchor": depths["anchor"],
                        "noisy": depths["noisy"],
                        "base": depths["base"],
                    },
                    gt,
                    hole,
                    args,
                )
                vis_saved += 1

    aggregate = aggregate_metrics_with_coarse(per_sample)
    if args.visualize and args.vis_rank != "first":
        ranked_rows = select_ranked_visualization_rows(per_sample, args)
        vis_saved = save_ranked_visualizations(
            ranked_rows,
            dataset_kwargs,
            flow_model,
            flow_args,
            model,
            device,
            out_dir,
            args,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "pretrained_checkpoint": flow_checkpoint,
        "cache_dir": args.cache_dir or ckpt_args.get("cache_dir"),
        "sample_list": args.sample_list,
        "path_source": path_source,
        "split": args.split,
        "num_samples": len(per_sample),
        "visualized_samples": vis_saved,
        "preserve_observed": args.preserve_observed,
        "vis_rank": args.vis_rank if args.visualize else None,
        "vis_rank_baseline": args.vis_rank_baseline if args.visualize and args.vis_rank != "first" else None,
        "method": "flow_anchor_propagation_refine",
        "backbone": "propagation_refine",
        "flow_backbone": flow_args.get("backbone", "resunet"),
        "input_mode": dataset_kwargs["input_mode"],
        "anchor_mode": dataset_kwargs["anchor_mode"],
        "include_hole_distance": dataset_kwargs["include_hole_distance"],
        "base_channels": int(ckpt_args.get("base_channels", 32)),
        "res_blocks": int(ckpt_args.get("res_blocks", 1)),
        "propagation_steps": int(ckpt_args.get("propagation_steps", 6)),
        "refine_dilate_radius": int(ckpt_args.get("refine_dilate_radius", 3)),
        "residual_scale": float(ckpt_args.get("residual_scale", 1.5)),
        "global_refine": bool(ckpt_args.get("global_refine", False)),
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
