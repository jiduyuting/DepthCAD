import argparse
import contextlib
import hashlib
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from train_depth_completion import (
    edge_aware_smoothness,
    finite_depth_mask,
    gradient_l1,
    masked_mean,
    move_batch_to_device,
    seed_everything,
)
from train_depth_flow_restoration import (
    flow_model_in_channels,
    predict_endpoint_norm,
    sample_flow,
)
from train_depth_restoration import (
    DepthRestorationCacheDataset,
    collect_cache_paths,
    mae_sum_and_count,
    metric_for_selection,
    print_eval_line,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a propagation-refinement head on PBRT cache samples using a frozen "
            "Flow checkpoint as the dense anchor."
        )
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt",
        help="Frozen Flow checkpoint that provides the anchor before local propagation refinement.",
    )
    parser.add_argument(
        "--anchor_cache_dir",
        type=str,
        default=None,
        help="Optional directory containing precomputed normalized Flow anchors.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/depth_flow_full_pbrt_iq_propagation_refine",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/last.pt.")
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--res_blocks", type=int, default=1)
    parser.add_argument("--propagation_steps", type=int, default=6)
    parser.add_argument("--propagation_hidden_scale", type=float, default=1.0)
    parser.add_argument("--refine_dilate_radius", type=int, default=3)
    parser.add_argument("--residual_scale", type=float, default=1.5)
    parser.add_argument(
        "--global_refine",
        action="store_true",
        default=False,
        help="Allow the refinement head to denoise the full image instead of only a dilated hole band.",
    )

    parser.add_argument("--mask_weight", type=float, default=5.0)
    parser.add_argument("--mask_center_weight", type=float, default=2.0)
    parser.add_argument("--valid_weight", type=float, default=0.2)
    parser.add_argument("--coarse_weight", type=float, default=0.5)
    parser.add_argument("--grad_weight", type=float, default=0.1)
    parser.add_argument("--hole_grad_weight", type=float, default=0.25)
    parser.add_argument("--boundary_grad_weight", type=float, default=0.5)
    parser.add_argument("--boundary_l1_weight", type=float, default=0.05)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--anchor_weight", type=float, default=0.02)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="hole",
        choices=["global", "hole", "composite"],
    )
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def flow_dataset_kwargs(ckpt_args):
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


class PropagationRefineCacheDataset(DepthRestorationCacheDataset):
    def __init__(self, paths, anchor_cache_dir=None, **kwargs):
        super().__init__(paths, **kwargs)
        self.anchor_cache_dir = anchor_cache_dir

    def __getitem__(self, index):
        item = super().__getitem__(index)
        mask = item["hole_mask"].numpy()[0].astype(np.uint8)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)
        max_distance = float(distance.max())
        if max_distance > 0:
            distance = distance / max_distance
        item["mask_distance"] = torch.from_numpy(distance[None])
        if self.anchor_cache_dir:
            anchor_path = flow_anchor_cache_path(self.anchor_cache_dir, item["path"])
            if not os.path.exists(anchor_path):
                raise FileNotFoundError(
                    f"Missing cached Flow anchor for {item['path']}: {anchor_path}. "
                    "Run cache_flow_anchors.py first."
                )
            cached = np.load(anchor_path).astype(np.float32)
            if cached.shape == item["anchor_norm"].shape[1:]:
                cached = cached[None]
            if tuple(cached.shape) != tuple(item["anchor_norm"].shape):
                raise ValueError(
                    f"Cached Flow anchor shape {cached.shape} does not match "
                    f"{tuple(item['anchor_norm'].shape)} for {item['path']}"
                )
            item["flow_anchor_norm"] = torch.from_numpy(cached)
        return item


def flow_anchor_cache_path(cache_dir, source_path):
    key = hashlib.sha256(str(os.path.abspath(source_path)).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def build_flow_model_from_checkpoint(ckpt, ckpt_args, condition_channels, device):
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
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_refine_model(condition_channels, args, device):
    return build_depth_backbone(
        "propagation_refine",
        in_channels=condition_channels,
        base_channels=int(args.base_channels),
        out_channels=1,
        res_blocks=int(args.res_blocks),
        propagation_steps=int(args.propagation_steps),
        propagation_hidden_scale=float(args.propagation_hidden_scale),
        refine_dilate_radius=int(args.refine_dilate_radius),
        residual_scale=float(args.residual_scale),
        global_refine=bool(getattr(args, "global_refine", False)),
    ).to(device)


@torch.no_grad()
def predict_flow_anchor_norm(flow_model, flow_args, batch):
    if flow_args.get("eval_sampling_mode", "euler") == "endpoint":
        return predict_endpoint_norm(
            flow_model,
            batch,
            int(flow_args.get("time_channels", 16)),
            float(flow_args.get("max_velocity_norm", 4.0)),
            float(flow_args.get("clip_norm_depth", 8.0)),
            float(flow_args.get("velocity_scale", 1.0)),
        )
    return sample_flow(
        flow_model,
        batch,
        int(flow_args.get("time_channels", 16)),
        float(flow_args.get("max_velocity_norm", 4.0)),
        int(flow_args.get("sample_steps", 8)),
        float(flow_args.get("clip_norm_depth", 8.0)),
        float(flow_args.get("velocity_scale", 1.0)),
    )


@torch.no_grad()
def prepare_propagation_batch(batch, flow_model, flow_args):
    flow_anchor_norm = batch.get("flow_anchor_norm")
    if flow_anchor_norm is None:
        flow_anchor_norm = predict_flow_anchor_norm(flow_model, flow_args, batch).detach()
    else:
        flow_anchor_norm = flow_anchor_norm.detach()
    batch = dict(batch)
    batch["x"] = batch["x"].clone()
    batch["x"][:, 0:1] = flow_anchor_norm
    batch["anchor_norm"] = flow_anchor_norm
    scale = batch["scale"].view(-1, 1, 1, 1)
    center = batch["center"].view(-1, 1, 1, 1)
    batch["depth_anchor"] = flow_anchor_norm * scale + center
    return batch


def predict_refined_norm(model, batch):
    out = model(batch["x"])
    return out["refined"], out["coarse"], out


def weighted_masked_mean(x, mask, weight, eps=1e-6):
    mask_weight = mask.to(dtype=x.dtype) * weight.to(dtype=x.dtype)
    denom = mask_weight.sum().clamp_min(eps)
    return (x * mask_weight).sum() / denom


def compute_loss(model, batch, args):
    pred_norm, coarse_norm, _ = predict_refined_norm(model, batch)
    target_norm = batch["target_norm"]
    anchor_norm = batch["anchor_norm"]

    valid = batch["valid_mask"] & finite_depth_mask(target_norm) & finite_depth_mask(pred_norm)
    hole = batch["hole_mask"] & valid
    observed = valid & (~batch["hole_mask"])

    err = torch.abs(pred_norm - target_norm)
    coarse_err = torch.abs(coarse_norm - target_norm)
    if float(args.mask_center_weight) > 0:
        weight = 1.0 + float(args.mask_center_weight) * batch["mask_distance"].to(device=pred_norm.device)
        mask_loss = weighted_masked_mean(err, hole, weight)
        coarse_loss = weighted_masked_mean(coarse_err, hole, weight)
    else:
        mask_loss = masked_mean(err, hole)
        coarse_loss = masked_mean(coarse_err, hole)

    valid_loss = masked_mean(err, observed)
    grad_loss = gradient_l1(pred_norm, target_norm, valid, valid)
    hole_grad_loss = gradient_l1(pred_norm, target_norm, valid, hole)

    boundary = batch["hole_mask"]
    if int(args.boundary_width) > 0:
        kernel_size = 2 * int(args.boundary_width) + 1
        boundary = (
            torch.nn.functional.max_pool2d(
                batch["hole_mask"].to(dtype=torch.float32),
                kernel_size=kernel_size,
                stride=1,
                padding=int(args.boundary_width),
            )
            > 0
        )
    boundary = boundary & valid
    boundary_ring = boundary & (~batch["hole_mask"])
    boundary_grad_loss = gradient_l1(pred_norm, target_norm, valid, boundary)
    boundary_l1_loss = masked_mean(err, boundary_ring)
    smooth_loss = edge_aware_smoothness(pred_norm, anchor_norm, valid, valid)
    anchor_loss = masked_mean(torch.abs(pred_norm - anchor_norm), observed)

    total = (
        float(args.mask_weight) * mask_loss
        + float(args.valid_weight) * valid_loss
        + float(args.coarse_weight) * coarse_loss
        + float(args.grad_weight) * grad_loss
        + float(args.hole_grad_weight) * hole_grad_loss
        + float(args.boundary_grad_weight) * boundary_grad_loss
        + float(args.boundary_l1_weight) * boundary_l1_loss
        + float(args.smooth_weight) * smooth_loss
        + float(args.anchor_weight) * anchor_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "mask": float(mask_loss.detach().cpu()),
        "valid": float(valid_loss.detach().cpu()),
        "coarse": float(coarse_loss.detach().cpu()),
        "grad": float(grad_loss.detach().cpu()),
        "hole_grad": float(hole_grad_loss.detach().cpu()),
        "boundary_grad": float(boundary_grad_loss.detach().cpu()),
        "boundary_l1": float(boundary_l1_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, dataloader, args, device, flow_model, flow_args):
    model.eval()
    totals = {}
    for prefix in ["model", "coarse", "anchor", "noisy", "base"]:
        for region in ["global", "hole", "valid"]:
            totals[f"{prefix}_{region}"] = [0.0, 0]

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch = prepare_propagation_batch(batch, flow_model, flow_args)
        pred_norm, coarse_norm, _ = predict_refined_norm(model, batch)
        scale = batch["scale"].view(-1, 1, 1, 1)
        center = batch["center"].view(-1, 1, 1, 1)
        pred = pred_norm * scale + center
        coarse = coarse_norm * scale + center

        target = batch["gt_depth"]
        depth_by_prefix = {
            "model": pred,
            "coarse": coarse,
            "anchor": batch["depth_anchor"],
            "noisy": batch["depth_noisy"],
            "base": batch["depth_base"],
        }
        valid = batch["valid_mask"]
        hole = batch["hole_mask"]
        region_masks = {
            "global": valid,
            "hole": valid & hole,
            "valid": valid & (~hole),
        }
        for prefix, depth in depth_by_prefix.items():
            for region_name, region_mask in region_masks.items():
                total, count = mae_sum_and_count(depth, target, region_mask)
                totals[f"{prefix}_{region_name}"][0] += total
                totals[f"{prefix}_{region_name}"][1] += count

    metrics = {}
    for key, (total, count) in totals.items():
        metrics[f"{key}_mae"] = total / count if count > 0 else math.nan
        metrics[f"{key}_count"] = count
    return metrics


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_cuda(enabled):
    if not enabled:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def normalize_optimizer_state(optimizer):
    """Keep AdamW step counters on CPU for PyTorch 1.12 compatibility."""
    for state in optimizer.state.values():
        step = state.get("step")
        if torch.is_tensor(step) and step.is_cuda and step.numel() == 1:
            state["step"] = step.cpu()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    flow_ckpt = load_checkpoint(args.pretrained_checkpoint, device)
    flow_args = flow_ckpt.get("args", {})
    dataset_kwargs = flow_dataset_kwargs(flow_args)

    train_paths, val_paths = collect_cache_paths(args)
    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump({"train": train_paths, "val": val_paths}, f, indent=2)

    args.method = "flow_anchor_propagation_refine"
    args.flow_input_mode = dataset_kwargs["input_mode"]
    args.flow_backbone = flow_args.get("backbone", "resunet")
    args.flow_eval_sampling_mode = flow_args.get("eval_sampling_mode", "euler")
    args.flow_sample_steps = int(flow_args.get("sample_steps", 8))
    args.input_mode = dataset_kwargs["input_mode"]
    args.include_hole_distance = bool(dataset_kwargs["include_hole_distance"])
    args.anchor_mode = dataset_kwargs["anchor_mode"]
    args.anchor_inpaint_radius = int(dataset_kwargs["anchor_inpaint_radius"])
    args.norm_percentiles = list(dataset_kwargs["norm_percentiles"])
    args.min_depth_scale = float(dataset_kwargs["min_depth_scale"])
    args.clip_norm_depth = float(dataset_kwargs["clip_norm_depth"])
    args.feature_percentile = float(dataset_kwargs["feature_percentile"])
    args.feature_clip = float(dataset_kwargs["feature_clip"])
    args.iq_clip = float(dataset_kwargs["iq_clip"])
    args.backbone = "propagation_refine"

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    dataset_kwargs["anchor_cache_dir"] = args.anchor_cache_dir
    train_dataset = PropagationRefineCacheDataset(train_paths, **dataset_kwargs)
    val_dataset = PropagationRefineCacheDataset(val_paths, **dataset_kwargs) if val_paths else None
    condition_channels = train_dataset.input_channels

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    flow_model = None
    if not args.anchor_cache_dir:
        flow_model = build_flow_model_from_checkpoint(flow_ckpt, flow_args, condition_channels, device)
    model = build_refine_model(condition_channels, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(enabled=(args.amp and device.type == "cuda"))

    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples:   {len(val_paths)}")
    print(f"Device:        {device}")
    print(f"Frozen Flow:   {args.pretrained_checkpoint}")
    print(f"Flow input:    {dataset_kwargs['input_mode']}")
    print(f"Condition ch:  {condition_channels}")
    print(f"Backbone:      propagation_refine")
    print(f"Output:        {args.output_dir}")

    best_score = float("inf")
    best_epoch = -1
    start_epoch = 1
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    if args.resume:
        resume_path = os.path.join(args.output_dir, "last.pt")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"No resumable checkpoint found: {resume_path}")
        state = load_checkpoint(resume_path, device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        normalize_optimizer_state(optimizer)
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state.get("metrics", {}).get("best_score", best_score))
        best_epoch = int(state.get("metrics", {}).get("best_epoch", best_epoch))
        print(f"Resuming propagation refine from epoch {state['epoch']}; target epoch {args.epochs}")

    if val_loader is not None:
        initial_metrics = evaluate(model, val_loader, args, device, flow_model, flow_args)
        print_eval_line("Initial val", initial_metrics)

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            batch = prepare_propagation_batch(batch, flow_model, flow_args)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(enabled=(args.amp and device.type == "cuda")):
                loss, loss_parts = compute_loss(model, batch, args)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss_parts["loss"])
            global_step += 1
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(train_loader)} "
                    f"loss={loss_parts['loss']:.5f} mask={loss_parts['mask']:.5f} "
                    f"coarse={loss_parts['coarse']:.5f} valid={loss_parts['valid']:.5f} "
                    f"boundary={loss_parts['boundary_grad']:.5f}"
                )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "seconds": time.time() - t0,
        }

        if val_loader is not None:
            metrics = evaluate(model, val_loader, args, device, flow_model, flow_args)
            record.update(metrics)
            print_eval_line(f"[epoch {epoch:03d}] train_loss={train_loss:.6f}", metrics)
            score = metric_for_selection(metrics, args)
            if not math.isnan(score) and score < best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(os.path.join(args.output_dir, "best.pt"), model, optimizer, epoch, args, metrics)
            hole_score = metrics["model_hole_mae"]
            global_score = metrics["model_global_mae"]
            if not math.isnan(hole_score):
                previous = getattr(main, "_best_hole", float("inf"))
                if hole_score < previous:
                    setattr(main, "_best_hole", hole_score)
                    save_checkpoint(os.path.join(args.output_dir, "best_hole.pt"), model, optimizer, epoch, args, metrics)
            if not math.isnan(global_score):
                previous = getattr(main, "_best_global", float("inf"))
                if global_score < previous:
                    setattr(main, "_best_global", global_score)
                    save_checkpoint(os.path.join(args.output_dir, "best_global.pt"), model, optimizer, epoch, args, metrics)
        else:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f}")

        append_jsonl(metrics_path, record)
        last_record = dict(record)
        last_record["best_score"] = best_score
        last_record["best_epoch"] = best_epoch
        save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch, args, last_record)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"), model, optimizer, epoch, args, record)

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selection_metric": args.selection_metric,
        "output_dir": args.output_dir,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "method": "flow_anchor_propagation_refine",
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "flow_input_mode": dataset_kwargs["input_mode"],
        "backbone": "propagation_refine",
        "base_channels": args.base_channels,
        "res_blocks": args.res_blocks,
        "propagation_steps": args.propagation_steps,
        "refine_dilate_radius": args.refine_dilate_radius,
        "residual_scale": args.residual_scale,
        "global_refine": args.global_refine,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Done.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
