import _bootstrap
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from infer_real_depth_flow import clip_prediction, normalize_depth
from real_depth_masked_self_test import mae
from real_raw9_masked_self_test import (
    build_real_hole_component_library,
    collect_pairs,
    filter_and_rebalance_component_library,
    make_artificial_mask,
    make_condition,
    summarize_component_library,
)
from train_depth_completion import gradient_l1, masked_mean, seed_everything
from train_depth_flow_restoration import predict_endpoint_norm, sample_flow
from train_depth_restoration import save_checkpoint
from train_real_raw9_flow_finetune import (
    RealRaw9MaskedDataset,
    aggregate_training_metrics,
    add_checkpoint_args as add_flow_checkpoint_args,
    autocast_cuda,
    build_model as build_flow_model,
    component_area_masks,
    dilate_mask,
    dilate_mask_np,
    make_grad_scaler,
    move_condition_to_device,
    split_real_pairs,
    weighted_masked_mean,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sensor-only propagation-refinement model on paired real raw9/depth."
    )
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="output/real_raw9_flow_finetune_after_synth_realhole_e20_lr5e6/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/real_raw9_propagation_refine_iq6_pilot",
    )
    parser.add_argument("--amplitude_mode", type=str, default="iq6", choices=["iq6", "raw_258"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--val_count", type=int, default=8)
    parser.add_argument("--split_json", type=str, default=None)
    parser.add_argument("--masks_per_sample", type=int, default=8)
    parser.add_argument("--val_masks_per_sample", type=int, default=5)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="real_hole_speckle_shapes",
        choices=["block", "real_hole_shapes", "real_hole_speckle_shapes"],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.08)
    parser.add_argument("--min_block_size", type=int, default=12)
    parser.add_argument("--max_block_size", type=int, default=72)
    parser.add_argument("--real_hole_min_area", type=int, default=24)
    parser.add_argument("--real_hole_max_area", type=int, default=0)
    parser.add_argument("--real_hole_min_overlap", type=float, default=0.6)
    parser.add_argument("--real_hole_max_components", type=int, default=8)
    parser.add_argument("--real_hole_exclude_self", action="store_true")
    parser.add_argument("--max_mask_retries", type=int, default=8)
    parser.add_argument("--clean_outlier_abs", type=float, default=0.35)
    parser.add_argument("--clean_outlier_mad_scale", type=float, default=6.0)
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
    parser.add_argument("--real_speckle_component_ratio", type=float, default=0.3)
    parser.add_argument("--hole_depth_threshold", type=float, default=1.0)
    parser.add_argument("--valid_min_depth", type=float, default=1.0)
    parser.add_argument("--valid_max_depth", type=float, default=9.9)
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
    parser.add_argument("--post_clip_mode", type=str, default="valid_range",
                        choices=["none", "valid_range", "valid_percentile"])
    parser.add_argument("--post_clip_percentiles", type=float, nargs=2, default=[0.5, 99.5])

    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--res_blocks", type=int, default=1)
    parser.add_argument("--propagation_steps", type=int, default=6)
    parser.add_argument("--propagation_hidden_scale", type=float, default=1.0)
    parser.add_argument("--refine_dilate_radius", type=int, default=3)
    parser.add_argument("--residual_scale", type=float, default=1.5)

    parser.add_argument("--mask_loss_weight", type=float, default=4.0)
    parser.add_argument("--mask_center_weight", type=float, default=2.0)
    parser.add_argument("--valid_loss_weight", type=float, default=0.1)
    parser.add_argument("--coarse_loss_weight", type=float, default=0.5)
    parser.add_argument("--grad_loss_weight", type=float, default=0.0)
    parser.add_argument("--hole_grad_loss_weight", type=float, default=0.25)
    parser.add_argument("--boundary_grad_loss_weight", type=float, default=0.5)
    parser.add_argument("--boundary_l1_loss_weight", type=float, default=0.05)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--refine_consistency_weight", type=float, default=0.05)
    parser.add_argument("--eval_component_area_threshold", type=int, default=700)
    parser.add_argument("--selection_metric", type=str, default="model_mask_mae")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=20)
    return parser.parse_args()


def build_model(ckpt_args, args, device):
    in_channels = 4 + int(bool(args.include_hole_distance)) + 4
    model = build_depth_backbone(
        "propagation_refine",
        in_channels=in_channels,
        base_channels=int(args.base_channels),
        out_channels=1,
        res_blocks=int(args.res_blocks),
        propagation_steps=int(args.propagation_steps),
        propagation_hidden_scale=float(args.propagation_hidden_scale),
        refine_dilate_radius=int(args.refine_dilate_radius),
        residual_scale=float(args.residual_scale),
    ).to(device)
    return model


def add_checkpoint_args(args, ckpt_args):
    args.checkpoint = args.pretrained_checkpoint
    args.input_mode = "noisy_amp"
    args.method = "real_raw9_propagation_refinement"
    args.eval_sampling_mode = "endpoint"
    args.sample_steps = 1
    args.clip_norm_depth = float(ckpt_args.get("clip_norm_depth", 8.0))
    args.backbone = "propagation_refine"
    args.feature_percentile = float(ckpt_args.get("feature_percentile", 99.0))
    args.feature_clip = float(ckpt_args.get("feature_clip", 3.0))
    args.include_hole_distance = bool(ckpt_args.get("include_hole_distance", False))


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
def predict_flow_anchor_norm(flow_model, flow_args, batch):
    if flow_args.eval_sampling_mode == "endpoint":
        return predict_endpoint_norm(
            flow_model,
            batch,
            flow_args.time_channels,
            flow_args.max_velocity_norm,
            flow_args.clip_norm_depth,
            flow_args.velocity_scale,
        )
    return sample_flow(
        flow_model,
        batch,
        flow_args.time_channels,
        flow_args.max_velocity_norm,
        flow_args.sample_steps,
        flow_args.clip_norm_depth,
        flow_args.velocity_scale,
    )


@torch.no_grad()
def prepare_propagation_batch(batch, flow_model, flow_args):
    flow_anchor_norm = predict_flow_anchor_norm(flow_model, flow_args, batch).detach()
    batch = dict(batch)
    batch["x"] = batch["x"].clone()
    batch["x"][:, 0:1] = flow_anchor_norm
    batch["anchor_norm"] = flow_anchor_norm
    scale = batch["scale"].view(-1, 1, 1, 1)
    center = batch["center"].view(-1, 1, 1, 1)
    batch["depth_anchor"] = flow_anchor_norm * scale + center
    return batch


def predict_refined_depth(model, batch, args):
    out = model(batch["x"])
    pred_norm = out["refined"]
    coarse_norm = out["coarse"]
    scale = batch["scale"].view(-1, 1, 1, 1)
    center = batch["center"].view(-1, 1, 1, 1)
    pred = pred_norm * scale + center
    coarse = coarse_norm * scale + center
    return pred, coarse, out


def compute_loss(model, batch, args):
    pred, coarse, out = predict_refined_depth(model, batch, args)
    target = batch["gt_depth"]
    valid = batch["valid_mask"] & torch.isfinite(target) & torch.isfinite(pred)
    mask = batch["artificial_mask"] & valid
    unmasked = valid & (~batch["artificial_mask"])
    err = torch.abs(pred - target)
    coarse_err = torch.abs(coarse - target)

    if float(args.mask_center_weight) > 0:
        mask_weight = 1.0 + float(args.mask_center_weight) * batch["mask_distance"].to(device=pred.device)
        mask_loss = weighted_masked_mean(err, mask, mask_weight)
        coarse_mask_loss = weighted_masked_mean(coarse_err, mask, mask_weight)
    else:
        mask_loss = masked_mean(err, mask)
        coarse_mask_loss = masked_mean(coarse_err, mask)

    valid_loss = masked_mean(err, unmasked)
    grad_loss = gradient_l1(pred, target, valid, valid)
    hole_grad_loss = gradient_l1(pred, target, valid, mask)

    boundary = dilate_mask(mask, int(args.boundary_width)) & valid
    boundary_ring = boundary & (~mask)
    boundary_grad_loss = gradient_l1(pred, target, valid, boundary)
    boundary_l1_loss = masked_mean(err, boundary_ring)

    refine_consistency = masked_mean(torch.abs(pred - coarse), mask)

    total = (
        float(args.mask_loss_weight) * mask_loss
        + float(args.valid_loss_weight) * valid_loss
        + float(args.coarse_loss_weight) * coarse_mask_loss
        + float(args.grad_loss_weight) * grad_loss
        + float(args.hole_grad_loss_weight) * hole_grad_loss
        + float(args.boundary_grad_loss_weight) * boundary_grad_loss
        + float(args.boundary_l1_loss_weight) * boundary_l1_loss
        + float(args.refine_consistency_weight) * refine_consistency
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "mask": float(mask_loss.detach().cpu()),
        "valid": float(valid_loss.detach().cpu()),
        "coarse_mask": float(coarse_mask_loss.detach().cpu()),
        "grad": float(grad_loss.detach().cpu()),
        "hole_grad": float(hole_grad_loss.detach().cpu()),
        "boundary_grad": float(boundary_grad_loss.detach().cpu()),
        "boundary_l1": float(boundary_l1_loss.detach().cpu()),
        "refine_consistency": float(refine_consistency.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, dataloader, args, device, flow_anchor_model, flow_anchor_args):
    model.eval()
    rows = []
    for batch in dataloader:
        batch = move_condition_to_device(batch, device)
        batch = prepare_propagation_batch(batch, flow_anchor_model, flow_anchor_args)
        pred, coarse, _ = predict_refined_depth(model, batch, args)
        anchor = batch["depth_anchor"]
        clean = batch["gt_depth"]
        corrupted = batch["depth_noisy"]
        artificial = batch["artificial_mask"]
        valid = batch["valid_mask"]
        condition_hole = batch["hole_mask"]

        bs = pred.shape[0]
        for i in range(bs):
            pred_np = pred[i, 0].detach().cpu().numpy().astype(np.float32)
            coarse_np = coarse[i, 0].detach().cpu().numpy().astype(np.float32)
            clean_np = clean[i, 0].detach().cpu().numpy().astype(np.float32)
            valid_np = valid[i, 0].detach().cpu().numpy().astype(bool)
            pred_np, _ = clip_prediction(pred_np, clean_np, valid_np, args)
            coarse_np, _ = clip_prediction(coarse_np, clean_np, valid_np, args)
            anchor_np = anchor[i, 0].detach().cpu().numpy().astype(np.float32)
            corrupted_np = corrupted[i, 0].detach().cpu().numpy().astype(np.float32)
            artificial_np = artificial[i, 0].detach().cpu().numpy().astype(bool)
            hole_np = condition_hole[i, 0].detach().cpu().numpy().astype(bool)
            hole_only = np.where(hole_np, pred_np, corrupted_np).astype(np.float32)
            unmasked = valid_np & (~artificial_np)
            small_mask, large_mask = component_area_masks(artificial_np, args.eval_component_area_threshold)
            boundary_mask = dilate_mask_np(artificial_np, args.boundary_width) & valid_np & (~artificial_np)

            row = {
                "sample_name": batch["sample_name"][i],
                "mask_pixel_count": int(artificial_np.sum()),
                "small_mask_pixel_count": int(small_mask.sum()),
                "large_mask_pixel_count": int(large_mask.sum()),
                "boundary_pixel_count": int(boundary_mask.sum()),
            }
            for key, prediction, mask_np in [
                ("anchor_mask_mae", anchor_np, artificial_np),
                ("coarse_mask_mae", coarse_np, artificial_np),
                ("model_mask_mae", pred_np, artificial_np),
                ("hole_only_mask_mae", hole_only, artificial_np),
                ("anchor_small_mask_mae", anchor_np, small_mask),
                ("model_small_mask_mae", pred_np, small_mask),
                ("anchor_large_mask_mae", anchor_np, large_mask),
                ("model_large_mask_mae", pred_np, large_mask),
                ("anchor_boundary_mae", anchor_np, boundary_mask),
                ("model_boundary_mae", pred_np, boundary_mask),
                ("model_unmasked_mae", pred_np, unmasked),
                ("anchor_global_mae", anchor_np, valid_np),
                ("model_global_mae", pred_np, valid_np),
            ]:
                value, count = mae(prediction, clean_np, mask_np)
                row[key] = value
                row[f"{key}_count"] = count
            if row["anchor_mask_mae"] is not None and row["model_mask_mae"] is not None:
                row["mask_improve_vs_anchor"] = (
                    row["anchor_mask_mae"] - row["model_mask_mae"]
                ) / max(row["anchor_mask_mae"], 1e-12)
            rows.append(row)
    return aggregate_training_metrics(rows), rows


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.pretrained_checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("input_mode") != "noisy_amp":
        raise ValueError(f"Expected pretrained input_mode noisy_amp, got {ckpt_args.get('input_mode')!r}")
    add_checkpoint_args(args, ckpt_args)

    pairs = collect_pairs(args.raw_dir, args.depth_dir)
    train_pairs, val_pairs = split_real_pairs(pairs, args)

    hole_component_library = []
    component_filter_summary = {}
    component_library_summary = {"total": 0, "threshold_hole": 0, "amp_speckle": 0}
    if args.mask_mode in {"real_hole_shapes", "real_hole_speckle_shapes"}:
        hole_component_library = build_real_hole_component_library(
            pairs,
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

    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump(
            {
                "train": [p[0] for p in train_pairs],
                "val": [p[0] for p in val_pairs],
                "split_json": args.split_json,
                "component_library_summary": component_library_summary,
                "component_filter_summary": component_filter_summary,
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_dataset = RealRaw9MaskedDataset(
        train_pairs,
        ckpt_args,
        args,
        masks_per_sample=args.masks_per_sample,
        fixed_masks=False,
        hole_component_library=hole_component_library,
    )
    val_dataset = RealRaw9MaskedDataset(
        val_pairs,
        ckpt_args,
        args,
        masks_per_sample=args.val_masks_per_sample,
        fixed_masks=True,
        hole_component_library=hole_component_library,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_model(ckpt_args, args, device)
    flow_anchor_model, flow_anchor_args = build_frozen_flow_anchor(args.pretrained_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(enabled=(args.amp and device.type == "cuda"))

    print(f"Pairs: {len(pairs)} train={len(train_pairs)} val={len(val_pairs)}")
    print(
        "Real component library="
        f"{component_library_summary.get('total', 0)} "
        f"(threshold={component_library_summary.get('threshold_hole', 0)}, "
        f"amp_speckle={component_library_summary.get('amp_speckle', 0)})"
    )
    if component_filter_summary:
        print(f"Component filtering={json.dumps(component_filter_summary, sort_keys=True)}")
    print(f"Device: {device}")
    print(f"Frozen flow anchor: {args.pretrained_checkpoint}")
    print(f"Output: {args.output_dir}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    best_score = float("inf")
    best_epoch = -1

    initial_metrics, _ = evaluate(model, val_loader, args, device, flow_anchor_model, flow_anchor_args)
    print(f"Initial val: {json.dumps(initial_metrics, sort_keys=True)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, start=1):
            batch = move_condition_to_device(batch, device)
            batch = prepare_propagation_batch(batch, flow_anchor_model, flow_anchor_args)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(enabled=(args.amp and device.type == "cuda")):
                loss, parts = compute_loss(model, batch, args)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(parts["loss"])
            if args.log_every and (step == 1 or step % args.log_every == 0):
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(train_loader)} "
                    f"loss={parts['loss']:.5f} mask={parts['mask']:.5f} "
                    f"coarse_mask={parts['coarse_mask']:.5f} valid={parts['valid']:.5f} "
                    f"hole_grad={parts['hole_grad']:.5f} "
                    f"boundary_grad={parts['boundary_grad']:.5f} "
                    f"refine_consistency={parts['refine_consistency']:.5f}"
                )

        metrics, _ = evaluate(model, val_loader, args, device, flow_anchor_model, flow_anchor_args)
        train_loss = float(np.mean(losses)) if losses else math.nan
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": time.time() - t0,
        }
        record.update(metrics)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"anchor_mask={metrics.get('anchor_mask_mae', math.nan):.6f} "
            f"coarse_mask={metrics.get('coarse_mask_mae', math.nan):.6f} "
            f"model_mask={metrics.get('model_mask_mae', math.nan):.6f} "
            f"large_mask={metrics.get('model_large_mask_mae', math.nan):.6f} "
            f"boundary={metrics.get('model_boundary_mae', math.nan):.6f} "
            f"improve={metrics.get('mask_improve_vs_anchor', math.nan):.2%} "
            f"model_unmasked={metrics.get('model_unmasked_mae', math.nan):.6f}"
        )

        score = metrics.get(args.selection_metric, float("inf"))
        if score < best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(os.path.join(args.output_dir, "best.pt"), model, optimizer, epoch, args, metrics)
        save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch, args, metrics)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"), model, optimizer, epoch, args, metrics)

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "output_dir": args.output_dir,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "split_json": args.split_json,
        "amplitude_mode": args.amplitude_mode,
        "mask_mode": args.mask_mode,
        "num_threshold_hole_components": int(component_library_summary.get("threshold_hole", 0)),
        "num_amp_speckle_components": int(component_library_summary.get("amp_speckle", 0)),
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "backbone": "propagation_refine",
        "refine_dilate_radius": int(args.refine_dilate_radius),
        "residual_scale": float(args.residual_scale),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Done.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
