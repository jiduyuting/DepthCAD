#!/usr/bin/env python
"""Calibrate a region-wise soft blend of Flow output and its anchor.

The blend coefficients are selected on validation only, then evaluated once on
the fixed test split. This is intended to test whether residual over-correction
on observed pixels or severe holes is responsible for the remaining MAE.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eval_depth_flow_restoration import load_checkpoint, predict_batch
from depth_restoration_backbones import build_depth_backbone
from train_depth_completion import read_list
from train_depth_flow_restoration import flow_model_in_channels
from train_depth_completion import move_batch_to_device
from train_depth_restoration import DepthRestorationCacheDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--val_list", required=True)
    p.add_argument("--test_list", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def dataset_kwargs(ckpt_args):
    return {
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


def score(pred, batch):
    gt = batch["gt_depth"]
    valid = batch["valid_mask"].bool() & torch.isfinite(gt)
    hole = valid & batch["hole_mask"].bool()
    regions = {"global": valid, "hole": hole, "valid": valid & (~batch["hole_mask"].bool())}
    out = {}
    for name, mask in regions.items():
        err = torch.abs(pred - gt)
        out[name] = (float(err[mask].sum().cpu()), int(mask.sum().cpu()))
    return out


@torch.no_grad()
def collect(model, loader, ckpt_args, device, alpha_hole, alpha_valid):
    totals = {name: [0.0, 0] for name in ("global", "hole", "valid")}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        flow = predict_batch(model, batch, ckpt_args, int(ckpt_args.get("sample_steps", 8)), "endpoint")
        anchor = batch["depth_anchor"]
        hole = batch["hole_mask"].bool()
        alpha = torch.where(hole, torch.as_tensor(alpha_hole, device=device), torch.as_tensor(alpha_valid, device=device))
        pred = anchor + alpha * (flow - anchor)
        batch_scores = score(pred, batch)
        for name, (err_sum, count) in batch_scores.items():
            totals[name][0] += err_sum
            totals[name][1] += count
    return {name: (err / count if count else None) for name, (err, count) in totals.items()}


@torch.no_grad()
def collect_grid(model, loader, ckpt_args, device, candidates):
    totals = [{name: [0.0, 0] for name in ("global", "hole", "valid")} for _ in candidates]
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        flow = predict_batch(model, batch, ckpt_args, int(ckpt_args.get("sample_steps", 8)), "endpoint")
        anchor = batch["depth_anchor"]
        hole = batch["hole_mask"].bool()
        for index, row in enumerate(candidates):
            alpha = torch.where(
                hole,
                torch.as_tensor(row["alpha_hole"], device=device),
                torch.as_tensor(row["alpha_valid"], device=device),
            )
            pred = anchor + alpha * (flow - anchor)
            batch_scores = score(pred, batch)
            for name, (err_sum, count) in batch_scores.items():
                totals[index][name][0] += err_sum
                totals[index][name][1] += count
    for index, row in enumerate(candidates):
        for name, (err, count) in totals[index].items():
            row[name] = err / count if count else None
    return candidates


def make_loader(paths, kwargs, args):
    dataset = DepthRestorationCacheDataset(paths, **kwargs)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    kwargs = dataset_kwargs(ckpt_args)

    val_paths = read_list(args.val_list)
    test_paths = read_list(args.test_list)
    if not val_paths or not test_paths:
        raise FileNotFoundError("Validation or test list is empty.")

    val_loader = make_loader(val_paths, kwargs, args)
    test_loader = make_loader(test_paths, kwargs, args)
    condition_channels = val_loader.dataset.input_channels
    time_channels = int(ckpt_args.get("time_channels", 16))
    model = build_depth_backbone(
        ckpt_args.get("backbone", "resunet"),
        in_channels=flow_model_in_channels(condition_channels, time_channels),
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
    candidates = []
    for alpha_hole in (0.50, 0.75, 1.00, 1.25):
        for alpha_valid in (0.00, 0.25, 0.50, 0.75, 1.00, 1.25):
            candidates.append({
                "alpha_hole": alpha_hole,
                "alpha_valid": alpha_valid,
            })
    collect_grid(model, val_loader, ckpt_args, device, candidates)

    best_global = min(candidates, key=lambda row: row["global"])
    best_hole = min(candidates, key=lambda row: row["hole"])
    best_valid = min(candidates, key=lambda row: row["valid"])
    selected = {"global": best_global, "hole": best_hole, "valid": best_valid}
    test_metrics = {}
    for key, row in selected.items():
        test_metrics[key] = {
            "alpha_hole": row["alpha_hole"],
            "alpha_valid": row["alpha_valid"],
            **collect(model, test_loader, ckpt_args, device, row["alpha_hole"], row["alpha_valid"]),
        }

    payload = {
        "checkpoint": args.checkpoint,
        "selection_split": "validation",
        "val_size": len(val_paths),
        "test_size": len(test_paths),
        "all_validation_candidates": candidates,
        "selected_validation": selected,
        "selected_test": test_metrics,
    }
    Path(args.output_dir, "soft_blend_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
