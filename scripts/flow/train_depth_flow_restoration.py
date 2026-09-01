import _bootstrap
import argparse
import inspect
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from depth_restoration_backbones import build_depth_backbone
from train_depth_completion import (
    edge_aware_smoothness,
    finite_depth_mask,
    gradient_l1,
    masked_mean,
    move_batch_to_device,
    seed_everything,
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
            "Train a conditional rectified-flow depth restoration model. The model "
            "starts from a deterministic anchor and learns a time-conditioned flow "
            "toward clean dense depth."
        )
    )
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/depth_flow_restoration")
    parser.add_argument("--resume", action="store_true", help="Resume from output_dir/last.pt.")
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="resunet",
                        choices=["resunet", "large_resunet", "transformer_bottleneck"],
                        help="Model backbone. resunet reproduces the previous architecture.")
    parser.add_argument("--res_blocks", type=int, default=2,
                        help="Residual blocks per stage for large_resunet/transformer_bottleneck.")
    parser.add_argument("--transformer_layers", type=int, default=2,
                        help="Transformer layers at the UNet bottleneck.")
    parser.add_argument("--transformer_heads", type=int, default=8,
                        help="Attention heads for transformer_bottleneck.")
    parser.add_argument("--transformer_mlp_ratio", type=float, default=4.0,
                        help="MLP expansion ratio for transformer_bottleneck.")
    parser.add_argument("--transformer_pool", type=int, default=2,
                        help="Average-pool stride before bottleneck attention to control memory.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--input_mode", type=str, default="noisy",
                        choices=["noisy", "noisy_amp", "noisy_iq", "noisy_iq_amp"],
                        help=(
                            "Condition channels reused from train_depth_restoration.py. "
                            "IQ modes require caches generated with --depth_cache_save_iq."
                        ))
    parser.add_argument("--include_hole_distance", action="store_true", default=False)
    parser.add_argument("--anchor_mode", type=str, default="noisy_ns",
                        choices=["noisy_zero", "noisy_ns", "noisy_telea", "cache_base"])
    parser.add_argument("--anchor_inpaint_radius", type=int, default=15)

    parser.add_argument("--norm_percentiles", type=float, nargs=2, default=[5.0, 95.0])
    parser.add_argument("--min_depth_scale", type=float, default=0.25)
    parser.add_argument("--clip_norm_depth", type=float, default=8.0)
    parser.add_argument("--feature_percentile", type=float, default=99.0)
    parser.add_argument("--feature_clip", type=float, default=3.0)
    parser.add_argument("--iq_clip", type=float, default=3.0,
                        help="Clamp robust-normalized signed IQ channels to +/- this value.")
    parser.add_argument("--mask_augment", action="store_true", default=False,
                        help="Regenerate coarse Kinect-style training holes from raw IQ and clean noisy depth.")
    parser.add_argument("--val_mask_augment", action="store_true", default=False,
                        help="Validate on both cached masks and a fixed coarse-mask stress protocol.")
    parser.add_argument("--mask_augment_probability", type=float, default=0.50)
    parser.add_argument("--mask_augment_block_sizes", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--mask_augment_hole_ratios", type=float, nargs=2, default=[0.15, 0.20])
    parser.add_argument(
        "--mask_augment_noise_depth_root",
        type=str,
        default="/data/pre_student/hcy/pbrt/noise_depth",
    )

    parser.add_argument("--time_channels", type=int, default=16,
                        help="Number of constant per-pixel time embedding channels.")
    parser.add_argument("--time_min", type=float, default=0.0,
                        help="Lower bound for random training time.")
    parser.add_argument("--time_max", type=float, default=1.0,
                        help="Upper bound for random training time.")
    parser.add_argument("--t0_sample_probability", type=float, default=0.0,
                        help="Probability of forcing a training item to t=0 to match endpoint inference.")
    parser.add_argument("--bridge_noise", type=float, default=0.0,
                        help=(
                            "Optional normalized stochastic bridge noise. 0.0 is pure "
                            "rectified flow; small values such as 0.02-0.05 add a "
                            "diffusion-like denoising component."
                        ))
    parser.add_argument("--sample_steps", type=int, default=8,
                        help="Euler steps used for validation sampling.")
    parser.add_argument("--eval_sampling_mode", type=str, default="euler",
                        choices=["euler", "endpoint"],
                        help=(
                            "Validation sampler. euler integrates the learned flow; endpoint "
                            "uses a direct t=0 anchor -> target prediction from the same model."
                        ))
    parser.add_argument("--max_velocity_norm", type=float, default=4.0,
                        help="Clamp normalized velocity prediction to +/- this value.")
    parser.add_argument("--velocity_scale", type=float, default=1.0,
                        help="Multiplier applied to predicted velocity during training and sampling.")

    parser.add_argument("--hole_weight", type=float, default=5.0)
    parser.add_argument("--valid_weight", type=float, default=1.0)
    parser.add_argument("--velocity_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0,
                        help="One-step x_t -> x_1 reconstruction loss weight.")
    parser.add_argument("--grad_weight", type=float, default=0.5)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--anchor_weight", type=float, default=0.0,
                        help="Optional valid-region regularization toward the anchor.")
    parser.add_argument("--endpoint_weight", type=float, default=0.0,
                        help=(
                            "Optional auxiliary direct restoration loss at t=0. "
                            "This turns the model into flow + endpoint restoration."
                        ))
    parser.add_argument("--endpoint_grad_weight", type=float, default=0.5,
                        help="Gradient loss weight inside the endpoint auxiliary loss.")
    parser.add_argument("--endpoint_smooth_weight", type=float, default=0.02,
                        help="Smoothness loss weight inside the endpoint auxiliary loss.")

    parser.add_argument("--selection_metric", type=str, default="global",
                        choices=["global", "hole", "composite"])
    parser.add_argument("--hard_sampling", action="store_true", default=False,
                        help="Oversample training samples with larger hole areas.")
    parser.add_argument("--hard_sampling_gamma", type=float, default=2.0,
                        help="Strength of large-hole oversampling relative to uniform sampling.")
    parser.add_argument("--hard_loss_gamma", type=float, default=2.0,
                        help="Extra hole loss weight proportional to the sample hole area.")
    parser.add_argument("--hard_loss_area_scale", type=float, default=0.10,
                        help="Hole-area fraction that maps to one unit of hard loss weight.")
    parser.add_argument("--hard_loss_max_weight", type=float, default=4.0,
                        help="Maximum per-sample hard loss multiplier.")
    parser.add_argument("--boundary_weight", type=float, default=0.0,
                        help="Additional reconstruction loss weight near hole boundaries.")
    parser.add_argument("--boundary_px", type=float, default=3.0,
                        help="Width in pixels of the hole-boundary loss band.")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def flow_model_in_channels(condition_channels, time_channels):
    return int(condition_channels) + 1 + int(time_channels)


def make_time_channels(t, num_channels, spatial_shape):
    if num_channels <= 0:
        b = t.shape[0]
        h, w = spatial_shape
        return t.new_empty((b, 0, h, w))

    h, w = spatial_shape
    maps = [t]
    remaining = num_channels - 1
    if remaining > 0:
        half = int(math.ceil(remaining / 2))
        freqs = 2.0 ** torch.arange(half, device=t.device, dtype=t.dtype)
        angles = t * freqs.view(1, half, 1, 1) * (2.0 * math.pi)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        maps.append(emb[:, :remaining])
    out = torch.cat(maps, dim=1)
    return out.expand(-1, -1, h, w)


def build_flow_input(batch, x_t, t, time_channels):
    _, _, h, w = x_t.shape
    time_maps = make_time_channels(t, time_channels, (h, w))
    return torch.cat([batch["x"], x_t, time_maps], dim=1)


def sample_train_times(batch, args):
    b = batch["target_norm"].shape[0]
    t_min = float(args.time_min)
    t_max = float(args.time_max)
    if not (0.0 <= t_min < t_max <= 1.0):
        raise ValueError("--time_min and --time_max must satisfy 0 <= min < max <= 1.")
    t = torch.rand((b, 1, 1, 1), device=batch["target_norm"].device, dtype=batch["target_norm"].dtype)
    t = t_min + (t_max - t_min) * t
    probability = float(getattr(args, "t0_sample_probability", 0.0))
    if not (0.0 <= probability <= 1.0):
        raise ValueError("--t0_sample_probability must be in [0, 1].")
    if probability > 0:
        force_t0 = torch.rand((b, 1, 1, 1), device=t.device) < probability
        t = torch.where(force_t0, torch.zeros_like(t), t)
    return t


def make_flow_pair(batch, t, bridge_noise):
    anchor = batch["anchor_norm"]
    target = batch["target_norm"]
    velocity = target - anchor
    if bridge_noise > 0:
        eps = torch.randn_like(anchor)
        sigma = float(bridge_noise)
        bridge = sigma * t * (1.0 - t) * eps
        x_t = (1.0 - t) * anchor + t * target + bridge
        velocity = velocity + sigma * (1.0 - 2.0 * t) * eps
    else:
        x_t = (1.0 - t) * anchor + t * target
    return x_t, velocity


def predict_velocity(
    model,
    batch,
    x_t,
    t,
    time_channels,
    max_velocity_norm,
    velocity_scale=1.0,
):
    model_input = build_flow_input(batch, x_t, t, time_channels)
    velocity = model(model_input)
    if max_velocity_norm > 0:
        velocity = torch.clamp(velocity, -float(max_velocity_norm), float(max_velocity_norm))
    return velocity * float(velocity_scale)


@torch.no_grad()
def sample_flow(
    model,
    batch,
    time_channels,
    max_velocity_norm,
    sample_steps,
    clip_norm_depth,
    velocity_scale=1.0,
):
    steps = max(1, int(sample_steps))
    x_t = batch["anchor_norm"].clone()
    dt = 1.0 / float(steps)
    b = x_t.shape[0]
    dtype = x_t.dtype
    device = x_t.device
    for step in range(steps):
        t_value = (step + 0.5) / float(steps)
        t = torch.full((b, 1, 1, 1), t_value, device=device, dtype=dtype)
        velocity = predict_velocity(
            model,
            batch,
            x_t,
            t,
            time_channels,
            max_velocity_norm,
            velocity_scale,
        )
        x_t = x_t + dt * velocity
        if clip_norm_depth > 0:
            x_t = torch.clamp(x_t, -float(clip_norm_depth), float(clip_norm_depth))
    return x_t


@torch.no_grad()
def predict_endpoint_norm(
    model,
    batch,
    time_channels,
    max_velocity_norm,
    clip_norm_depth,
    velocity_scale=1.0,
):
    x0 = batch["anchor_norm"]
    b = x0.shape[0]
    t0 = torch.zeros((b, 1, 1, 1), device=x0.device, dtype=x0.dtype)
    velocity = predict_velocity(
        model,
        batch,
        x0,
        t0,
        time_channels,
        max_velocity_norm,
        velocity_scale,
    )
    pred = x0 + velocity
    if clip_norm_depth > 0:
        pred = torch.clamp(pred, -float(clip_norm_depth), float(clip_norm_depth))
    return pred


def predict_endpoint_norm_train(
    model,
    batch,
    args,
):
    x0 = batch["anchor_norm"]
    b = x0.shape[0]
    t0 = torch.zeros((b, 1, 1, 1), device=x0.device, dtype=x0.dtype)
    velocity = predict_velocity(
        model,
        batch,
        x0,
        t0,
        args.time_channels,
        args.max_velocity_norm,
        args.velocity_scale,
    )
    pred = x0 + velocity
    if args.clip_norm_depth > 0:
        pred = torch.clamp(pred, -float(args.clip_norm_depth), float(args.clip_norm_depth))
    return pred


def compute_loss(model, batch, args):
    t = sample_train_times(batch, args)
    x_t, target_velocity = make_flow_pair(batch, t, args.bridge_noise)
    pred_velocity = predict_velocity(
        model,
        batch,
        x_t,
        t,
        args.time_channels,
        args.max_velocity_norm,
        args.velocity_scale,
    )
    target = batch["target_norm"]
    anchor = batch["anchor_norm"]
    pred_x1 = x_t + (1.0 - t) * pred_velocity

    valid = batch["valid_mask"] & finite_depth_mask(target) & finite_depth_mask(pred_x1)
    hole = batch["hole_mask"]
    valid_region = valid & (~hole)
    hole_region = valid & hole
    hole_distance = batch.get("hole_distance")
    if hole_distance is None:
        hole_distance = torch.zeros_like(target)
    boundary_region = hole_region & (hole_distance <= float(args.boundary_px))

    hole_area_fraction = batch.get("hole_area_fraction")
    if hole_area_fraction is None:
        hole_area_fraction = torch.zeros((target.shape[0],), device=target.device, dtype=target.dtype)
    hard_weight = 1.0 + float(args.hard_loss_gamma) * torch.clamp(
        hole_area_fraction.to(dtype=target.dtype) / max(float(args.hard_loss_area_scale), 1e-6),
        min=0.0,
        max=max(float(args.hard_loss_max_weight) - 1.0, 0.0),
    )
    hard_weight = hard_weight.view(-1, 1, 1, 1).to(dtype=target.dtype)

    velocity_err = torch.abs(pred_velocity - target_velocity)
    recon_err = torch.abs(pred_x1 - target)

    vel_hole_loss = masked_mean(velocity_err * hard_weight, hole_region)
    vel_valid_loss = masked_mean(velocity_err, valid_region)
    recon_hole_loss = masked_mean(recon_err * hard_weight, hole_region)
    recon_valid_loss = masked_mean(recon_err, valid_region)
    grad_loss = gradient_l1(pred_x1, target, valid, valid)
    smooth_loss = edge_aware_smoothness(pred_x1, anchor, valid, valid)
    anchor_loss = masked_mean(torch.abs(pred_x1 - anchor), valid_region)
    boundary_loss = masked_mean(recon_err * hard_weight, boundary_region)

    velocity_loss = args.hole_weight * vel_hole_loss + args.valid_weight * vel_valid_loss
    recon_loss = args.hole_weight * recon_hole_loss + args.valid_weight * recon_valid_loss
    total = (
        args.velocity_weight * velocity_loss
        + args.recon_weight * recon_loss
        + args.grad_weight * grad_loss
        + args.smooth_weight * smooth_loss
        + args.anchor_weight * anchor_loss
        + args.boundary_weight * boundary_loss
    )
    endpoint_loss = pred_x1.new_tensor(0.0)
    endpoint_hole_loss = pred_x1.new_tensor(0.0)
    endpoint_valid_loss = pred_x1.new_tensor(0.0)
    endpoint_grad_loss = pred_x1.new_tensor(0.0)
    endpoint_smooth_loss = pred_x1.new_tensor(0.0)
    if args.endpoint_weight > 0:
        endpoint_pred = predict_endpoint_norm_train(model, batch, args)
        endpoint_valid = batch["valid_mask"] & finite_depth_mask(target) & finite_depth_mask(endpoint_pred)
        endpoint_hole = endpoint_valid & batch["hole_mask"]
        endpoint_valid_region = endpoint_valid & (~batch["hole_mask"])
        endpoint_err = torch.abs(endpoint_pred - target)
        endpoint_hole_loss = masked_mean(endpoint_err, endpoint_hole)
        endpoint_valid_loss = masked_mean(endpoint_err, endpoint_valid_region)
        endpoint_grad_loss = gradient_l1(endpoint_pred, target, endpoint_valid, endpoint_valid)
        endpoint_smooth_loss = edge_aware_smoothness(endpoint_pred, anchor, endpoint_valid, endpoint_valid)
        endpoint_loss = (
            args.hole_weight * endpoint_hole_loss
            + args.valid_weight * endpoint_valid_loss
            + args.endpoint_grad_weight * endpoint_grad_loss
            + args.endpoint_smooth_weight * endpoint_smooth_loss
        )
        total = total + args.endpoint_weight * endpoint_loss

    return total, {
        "loss": float(total.detach().cpu()),
        "vel_hole": float(vel_hole_loss.detach().cpu()),
        "vel_valid": float(vel_valid_loss.detach().cpu()),
        "recon_hole": float(recon_hole_loss.detach().cpu()),
        "recon_valid": float(recon_valid_loss.detach().cpu()),
        "grad": float(grad_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()),
        "boundary": float(boundary_loss.detach().cpu()),
        "endpoint": float(endpoint_loss.detach().cpu()),
        "endpoint_hole": float(endpoint_hole_loss.detach().cpu()),
        "endpoint_valid": float(endpoint_valid_loss.detach().cpu()),
        "endpoint_grad": float(endpoint_grad_loss.detach().cpu()),
        "endpoint_smooth": float(endpoint_smooth_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    model.eval()
    totals = {}
    for prefix in ["model", "anchor", "noisy", "base"]:
        for region in ["global", "hole", "valid"]:
            totals[f"{prefix}_{region}"] = [0.0, 0]

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if args.eval_sampling_mode == "endpoint":
            pred_norm = predict_endpoint_norm(
                model,
                batch,
                args.time_channels,
                args.max_velocity_norm,
                args.clip_norm_depth,
                args.velocity_scale,
            )
        else:
            pred_norm = sample_flow(
                model,
                batch,
                args.time_channels,
                args.max_velocity_norm,
                args.sample_steps,
                args.clip_norm_depth,
                args.velocity_scale,
            )
        scale = batch["scale"].view(-1, 1, 1, 1)
        center = batch["center"].view(-1, 1, 1, 1)
        pred = pred_norm * scale + center

        target = batch["gt_depth"]
        depth_by_prefix = {
            "model": pred,
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
                key = f"{prefix}_{region_name}"
                totals[key][0] += total
                totals[key][1] += count

    metrics = {}
    for key, (total, count) in totals.items():
        metrics[f"{key}_mae"] = total / count if count > 0 else math.nan
        metrics[f"{key}_count"] = count
    return metrics


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def hard_sampling_weights(paths, gamma, area_scale, max_weight):
    weights = []
    area_scale = max(float(area_scale), 1e-6)
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            hole_area = float((data["hole_mask"] > 0.5).mean())
        weight = 1.0 + float(gamma) * min(hole_area / area_scale, float(max_weight) - 1.0)
        weights.append(max(weight, 1e-6))
    return torch.as_tensor(weights, dtype=torch.double)


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    train_paths, val_paths = collect_cache_paths(args)
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples:   {len(val_paths)}")
    print(f"Device:        {device}")
    print(f"Input mode:    {args.input_mode}")
    print(f"Anchor mode:   {args.anchor_mode}")
    print(f"Backbone:      {args.backbone}")
    print(f"Base channels: {args.base_channels}")
    print(f"Flow steps:    {args.sample_steps}")
    print(f"Eval sampler:  {args.eval_sampling_mode}")
    print(f"Bridge noise:  {args.bridge_noise}")
    print(f"Endpoint aux:  {args.endpoint_weight}")
    print(f"Selection:     {args.selection_metric}")

    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump({"train": train_paths, "val": val_paths}, f, indent=2)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    dataset_kwargs = {
        "input_mode": args.input_mode,
        "include_hole_distance": args.include_hole_distance,
        "anchor_mode": args.anchor_mode,
        "anchor_inpaint_radius": args.anchor_inpaint_radius,
        "norm_percentiles": args.norm_percentiles,
        "min_depth_scale": args.min_depth_scale,
        "clip_norm_depth": args.clip_norm_depth,
        "feature_percentile": args.feature_percentile,
        "feature_clip": args.feature_clip,
        "iq_clip": args.iq_clip,
    }
    train_dataset_kwargs = dict(dataset_kwargs)
    train_dataset_kwargs.update({
        "mask_augment": args.mask_augment,
        "mask_augment_probability": args.mask_augment_probability,
        "mask_augment_block_sizes": args.mask_augment_block_sizes,
        "mask_augment_hole_ratios": args.mask_augment_hole_ratios,
        "mask_augment_noise_depth_root": args.mask_augment_noise_depth_root,
        "mask_augment_seed": args.seed,
    })
    train_dataset = DepthRestorationCacheDataset(train_paths, **train_dataset_kwargs)
    val_dataset = DepthRestorationCacheDataset(val_paths, **dataset_kwargs) if val_paths else None
    if args.val_mask_augment and val_paths:
        stress_dataset_kwargs = dict(dataset_kwargs)
        stress_dataset_kwargs.update({
            "mask_augment": True,
            "mask_augment_probability": 1.0,
            "mask_augment_block_sizes": args.mask_augment_block_sizes,
            "mask_augment_hole_ratios": args.mask_augment_hole_ratios,
            "mask_augment_noise_depth_root": args.mask_augment_noise_depth_root,
            "mask_augment_seed": args.seed + 1000003,
            "mask_augment_deterministic": True,
        })
        stress_val_dataset = DepthRestorationCacheDataset(val_paths, **stress_dataset_kwargs)
        val_dataset = ConcatDataset([val_dataset, stress_val_dataset])
    condition_channels = train_dataset.input_channels
    in_channels = flow_model_in_channels(condition_channels, args.time_channels)
    print(f"Condition channels: {condition_channels}")
    print(f"Model input chans:  {in_channels}")

    train_sampler = None
    if args.hard_sampling:
        train_sampler = WeightedRandomSampler(
            hard_sampling_weights(
                train_paths,
                args.hard_sampling_gamma,
                args.hard_loss_area_scale,
                args.hard_loss_max_weight,
            ),
            num_samples=len(train_dataset),
            replacement=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    model = build_depth_backbone(
        args.backbone,
        in_channels=in_channels,
        base_channels=args.base_channels,
        out_channels=1,
        res_blocks=args.res_blocks,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        transformer_pool=args.transformer_pool,
    ).to(device)
    supports_capturable = "capturable" in inspect.signature(torch.optim.AdamW).parameters
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda" and supports_capturable:
        optimizer_kwargs["capturable"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    amp_enabled = args.amp and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_score = float("inf")
    best_hole_score = float("inf")
    best_global_score = float("inf")
    best_epoch = -1
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    start_epoch = 1
    if args.resume:
        resume_path = os.path.join(args.output_dir, "last.pt")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"No resumable checkpoint found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.weight_decay
            if supports_capturable:
                param_group["capturable"] = device.type == "cuda"
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state.get("metrics", {}).get("best_score", best_score))
        best_hole_score = float(state.get("metrics", {}).get("best_hole_score", best_hole_score))
        best_global_score = float(state.get("metrics", {}).get("best_global_score", best_global_score))
        for checkpoint_name, score_key in (
            ("best_hole.pt", "model_hole_mae"),
            ("best_global.pt", "model_global_mae"),
        ):
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                checkpoint_metrics = torch.load(checkpoint_path, map_location="cpu").get("metrics", {})
                checkpoint_score = checkpoint_metrics.get(score_key)
                if checkpoint_score is not None and not math.isnan(float(checkpoint_score)):
                    if score_key == "model_hole_mae":
                        best_hole_score = min(best_hole_score, float(checkpoint_score))
                    else:
                        best_global_score = min(best_global_score, float(checkpoint_score))
        print(
            f"Resuming Flow from epoch {state['epoch']}; target epoch {args.epochs}; "
            f"lr reset to {args.lr:g}"
        )

    if val_loader is not None:
        initial_metrics = evaluate(model, val_loader, args, device)
        print_eval_line("Initial val", initial_metrics)

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_context = torch.amp.autocast("cuda", enabled=amp_enabled)
            else:
                autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled)
            with autocast_context:
                loss, loss_parts = compute_loss(model, batch, args)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss_parts["loss"])
            global_step += 1
            if args.log_every > 0 and (step % args.log_every == 0 or step == 1):
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(train_loader)} "
                    f"loss={loss_parts['loss']:.5f} "
                    f"vel_hole={loss_parts['vel_hole']:.5f} "
                    f"vel_valid={loss_parts['vel_valid']:.5f} "
                    f"rec_hole={loss_parts['recon_hole']:.5f} "
                    f"rec_valid={loss_parts['recon_valid']:.5f} "
                    f"grad={loss_parts['grad']:.5f} "
                    f"boundary={loss_parts['boundary']:.5f} "
                    f"endpoint={loss_parts['endpoint']:.5f}"
                )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "seconds": time.time() - t0,
        }

        if val_loader is not None:
            metrics = evaluate(model, val_loader, args, device)
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
                if hole_score < best_hole_score:
                    best_hole_score = hole_score
                    save_checkpoint(os.path.join(args.output_dir, "best_hole.pt"), model, optimizer, epoch, args, metrics)
            if not math.isnan(global_score):
                if global_score < best_global_score:
                    best_global_score = global_score
                    save_checkpoint(os.path.join(args.output_dir, "best_global.pt"), model, optimizer, epoch, args, metrics)
        else:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f}")

        append_jsonl(metrics_path, record)
        last_record = dict(record)
        last_record["best_score"] = best_score
        last_record["best_epoch"] = best_epoch
        last_record["best_hole_score"] = best_hole_score
        last_record["best_global_score"] = best_global_score
        save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch, args, last_record)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"), model, optimizer, epoch, args, record)

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_hole_score": best_hole_score,
        "best_global_score": best_global_score,
        "selection_metric": args.selection_metric,
        "output_dir": args.output_dir,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "method": "conditional_rectified_flow",
        "sample_steps": args.sample_steps,
        "eval_sampling_mode": args.eval_sampling_mode,
        "bridge_noise": args.bridge_noise,
        "endpoint_weight": args.endpoint_weight,
        "backbone": args.backbone,
        "base_channels": args.base_channels,
        "res_blocks": args.res_blocks,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_pool": args.transformer_pool,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Done.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
