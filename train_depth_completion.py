import argparse
import json
import math
import os
import random
import time
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a residual depth completion U-Net from cached tensors."
    )
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory created by apply_kinect_holes_and_eval.py --save_depth_completion_cache.")
    parser.add_argument("--output_dir", type=str, default="./output/depth_completion_unet",
                        help="Directory for checkpoints and metrics.")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of cache samples used for validation when no split files are provided.")
    parser.add_argument("--train_list", type=str, default=None,
                        help="Optional text file with train .npz paths, one per line.")
    parser.add_argument("--val_list", type=str, default=None,
                        help="Optional text file with validation .npz paths, one per line.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Use torch autocast/mixed precision on CUDA.")
    parser.add_argument("--device", type=str, default=None,
                        help="Default: cuda if available else cpu.")
    parser.add_argument("--input_mode", type=str, default="depth",
                        choices=["depth", "depth_amp", "depth_iq", "depth_iq_amp"],
                        help=(
                            "Input channels. depth=base/depthcad/noisy/mask/conf; "
                            "depth_amp adds noisy/denoised amplitude features; "
                            "depth_iq adds noisy+denoised raw 6-channel IQ; "
                            "depth_iq_amp adds both."
                        ))
    parser.add_argument("--iq_clip", type=float, default=3.0,
                        help="Clamp robust-normalized IQ channels to +/- this value.")
    parser.add_argument("--feature_clip", type=float, default=3.0,
                        help="Clamp robust-normalized nonnegative feature channels to [0, this value].")
    parser.add_argument("--feature_percentile", type=float, default=99.0,
                        help="Robust percentile for amplitude/IQ feature normalization.")

    parser.add_argument("--norm_percentiles", type=float, nargs=2, default=[5.0, 95.0],
                        help="Per-sample robust normalization percentiles computed from depth_base.")
    parser.add_argument("--min_depth_scale", type=float, default=0.25,
                        help="Lower bound for per-sample depth normalization scale in meters.")
    parser.add_argument("--clip_norm_depth", type=float, default=8.0,
                        help="Clamp normalized depth inputs to +/- this value.")

    parser.add_argument("--hole_weight", type=float, default=5.0)
    parser.add_argument("--near_weight", type=float, default=1.0)
    parser.add_argument("--valid_weight", type=float, default=0.25)
    parser.add_argument("--grad_weight", type=float, default=0.5)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--max_residual_norm", type=float, default=4.0,
                        help="Clamp normalized residual prediction to +/- this value.")
    parser.add_argument("--residual_scale", type=float, default=1.0,
                        help="Multiplier applied to the predicted residual before blending with the baseline.")
    parser.add_argument("--residual_apply_mask", type=str, default="refine",
                        choices=["refine", "hole"],
                        help=(
                            "Where to apply the predicted residual. refine updates the hole plus "
                            "its refinement band; hole keeps the baseline unchanged outside holes."
                        ))
    parser.add_argument("--residual_gate", type=str, default="binary",
                        choices=["binary", "soft_hole_distance"],
                        help=(
                            "Residual blending gate. binary uses the selected apply mask directly. "
                            "soft_hole_distance fades residuals near hole boundaries."
                        ))
    parser.add_argument("--gate_boundary_px", type=float, default=1.0,
                        help="For soft_hole_distance, pixels inside the hole boundary with zero residual.")
    parser.add_argument("--gate_ramp_px", type=float, default=4.0,
                        help="For soft_hole_distance, distance over which residual ramps from 0 to 1.")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"List file not found: {path}\n"
            "If this is a train/val split, create it first with make_depth_completion_splits.py, "
            "or omit --train_list/--val_list to let train_depth_completion.py split --cache_dir directly."
        )
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def collect_cache_paths(args):
    if args.train_list or args.val_list:
        if not args.train_list or not args.val_list:
            raise ValueError("--train_list and --val_list must be provided together.")
        train_paths = read_list(args.train_list)
        val_paths = read_list(args.val_list)
        return train_paths, val_paths

    paths = sorted(glob(os.path.join(args.cache_dir, "**", "*.npz"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .npz cache files found under {args.cache_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(paths)
    val_count = max(1, int(round(len(paths) * args.val_ratio))) if len(paths) > 1 else 0
    val_paths = paths[:val_count]
    train_paths = paths[val_count:]
    if not train_paths and val_paths:
        train_paths, val_paths = val_paths, []
    return train_paths, val_paths


def masked_mean(x, mask, eps=1e-6):
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


def finite_depth_mask(depth):
    return torch.isfinite(depth)


def gradient_l1(pred, target, valid_mask, region_mask):
    mask = (valid_mask & region_mask).to(dtype=pred.dtype)

    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    dx_loss = masked_mean(torch.abs(dx_pred - dx_tgt), dx_mask)
    dy_loss = masked_mean(torch.abs(dy_pred - dy_tgt), dy_mask)
    return dx_loss + dy_loss


def edge_aware_smoothness(pred, guide, valid_mask, region_mask):
    mask = (valid_mask & region_mask).to(dtype=pred.dtype)

    dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx_guide = torch.abs(guide[:, :, :, 1:] - guide[:, :, :, :-1])
    dy_guide = torch.abs(guide[:, :, 1:, :] - guide[:, :, :-1, :])

    dx_weight = torch.exp(-torch.clamp(dx_guide, 0.0, 10.0))
    dy_weight = torch.exp(-torch.clamp(dy_guide, 0.0, 10.0))
    dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    return masked_mean(dx_pred * dx_weight, dx_mask) + masked_mean(dy_pred * dy_weight, dy_mask)


class DepthCompletionCacheDataset(Dataset):
    """
    Depth completion cache dataset.

    Inputs are normalized per sample using depth_base statistics. No GT-derived
    normalization is used, so the same transform is available at inference time.
    """

    def __init__(
        self,
        paths,
        norm_percentiles=(5.0, 95.0),
        min_depth_scale=0.25,
        clip_norm_depth=8.0,
        input_mode="depth",
        feature_percentile=99.0,
        iq_clip=3.0,
        feature_clip=3.0,
    ):
        self.paths = list(paths)
        self.norm_percentiles = tuple(norm_percentiles)
        self.min_depth_scale = float(min_depth_scale)
        self.clip_norm_depth = float(clip_norm_depth)
        self.input_mode = input_mode
        self.feature_percentile = float(feature_percentile)
        self.iq_clip = float(iq_clip)
        self.feature_clip = float(feature_clip)
        self.input_channels = self._compute_input_channels()

    def _compute_input_channels(self):
        channels = 5
        if self.input_mode in ["depth_amp", "depth_iq_amp"]:
            channels += 8  # noisy amp(3), noisy mean(1), denoised amp(3), denoised mean(1)
        if self.input_mode in ["depth_iq", "depth_iq_amp"]:
            channels += 12  # noisy IQ(6), denoised IQ(6)
        return channels

    def __len__(self):
        return len(self.paths)

    def _normalize_depth(self, depth, center, scale):
        depth = (depth - center) / scale
        depth = np.nan_to_num(depth, nan=0.0, neginf=-self.clip_norm_depth, posinf=self.clip_norm_depth)
        return np.clip(depth, -self.clip_norm_depth, self.clip_norm_depth).astype(np.float32)

    def _robust_signed_channels(self, values, valid_mask):
        values = values.astype(np.float32)
        out = np.zeros_like(values, dtype=np.float32)
        for i in range(values.shape[0]):
            ch = values[i]
            valid = valid_mask & np.isfinite(ch)
            if valid.sum() > 0:
                scale = np.percentile(np.abs(ch[valid]), self.feature_percentile)
            else:
                scale = np.percentile(np.abs(ch[np.isfinite(ch)]), self.feature_percentile) if np.isfinite(ch).any() else 1.0
            scale = max(float(scale), 1e-6)
            out[i] = np.clip(np.nan_to_num(ch / scale, nan=0.0), -self.iq_clip, self.iq_clip)
        return out.astype(np.float32)

    def _robust_nonnegative_channels(self, values, valid_mask):
        values = values.astype(np.float32)
        if values.ndim == 2:
            values = values[None]
        out = np.zeros_like(values, dtype=np.float32)
        for i in range(values.shape[0]):
            ch = values[i]
            valid = valid_mask & np.isfinite(ch)
            if valid.sum() > 0:
                scale = np.percentile(ch[valid], self.feature_percentile)
            else:
                scale = np.percentile(ch[np.isfinite(ch)], self.feature_percentile) if np.isfinite(ch).any() else 1.0
            scale = max(float(scale), 1e-6)
            out[i] = np.clip(np.nan_to_num(ch / scale, nan=0.0), 0.0, self.feature_clip)
        return out.astype(np.float32)

    def _require_keys(self, data, keys, path):
        missing = [key for key in keys if key not in data.files]
        if missing:
            raise KeyError(
                f"{path} is missing keys {missing}. Regenerate cache with "
                "--save_depth_completion_cache --depth_cache_save_iq for depth_iq modes."
            )

    def __getitem__(self, index):
        path = self.paths[index]
        data = np.load(path)

        depth_base = data["depth_base"].astype(np.float32)
        depth_depthcad = data["depth_depthcad"].astype(np.float32)
        depth_noisy = data["depth_noisy"].astype(np.float32)
        gt_depth = data["gt_depth"].astype(np.float32)
        hole_mask = (data["hole_mask"] > 0.5).astype(np.float32)
        refine_mask = (data["refine_mask"] > 0.5).astype(np.float32)
        hole_distance = cv2.distanceTransform(hole_mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        confidence = data["confidence"].astype(np.float32)
        valid_mask = (data["valid_mask"] > 0.5) & np.isfinite(gt_depth) & np.isfinite(depth_base)

        base_valid = depth_base[valid_mask & np.isfinite(depth_base)]
        if base_valid.size > 0:
            lo, hi = np.percentile(base_valid, self.norm_percentiles)
            center = float(np.median(base_valid))
            scale = float(hi - lo)
        else:
            center = 0.0
            scale = 1.0
        scale = max(scale, self.min_depth_scale)

        base_norm = self._normalize_depth(depth_base, center, scale)
        depthcad_norm = self._normalize_depth(depth_depthcad, center, scale)
        noisy_norm = self._normalize_depth(depth_noisy, center, scale)
        gt_norm = self._normalize_depth(gt_depth, center, scale)

        channels = [
            base_norm,
            depthcad_norm,
            noisy_norm,
            hole_mask,
            confidence,
        ]

        if self.input_mode in ["depth_amp", "depth_iq_amp"]:
            self._require_keys(
                data,
                ["noisy_amplitude", "noisy_amplitude_mean", "denoised_amplitude", "denoised_amplitude_mean"],
                path,
            )
            channels.extend(self._robust_nonnegative_channels(data["noisy_amplitude"], valid_mask))
            channels.extend(self._robust_nonnegative_channels(data["noisy_amplitude_mean"], valid_mask))
            channels.extend(self._robust_nonnegative_channels(data["denoised_amplitude"], valid_mask))
            channels.extend(self._robust_nonnegative_channels(data["denoised_amplitude_mean"], valid_mask))

        if self.input_mode in ["depth_iq", "depth_iq_amp"]:
            self._require_keys(data, ["noisy_iq", "denoised_iq"], path)
            channels.extend(self._robust_signed_channels(data["noisy_iq"], valid_mask))
            channels.extend(self._robust_signed_channels(data["denoised_iq"], valid_mask))

        x = np.stack(channels, axis=0).astype(np.float32)

        sample_name = str(data["sample_name"]) if "sample_name" in data.files else os.path.basename(path)
        return {
            "x": torch.from_numpy(x),
            "base_norm": torch.from_numpy(base_norm[None]),
            "target_norm": torch.from_numpy(gt_norm[None]),
            "depth_base": torch.from_numpy(depth_base[None]),
            "gt_depth": torch.from_numpy(gt_depth[None]),
            "hole_mask": torch.from_numpy(hole_mask[None].astype(np.bool_)),
            "refine_mask": torch.from_numpy(refine_mask[None].astype(np.bool_)),
            "hole_distance": torch.from_numpy(hole_distance[None]),
            "valid_mask": torch.from_numpy(valid_mask[None].astype(np.bool_)),
            "center": torch.tensor(center, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "path": path,
            "sample_name": sample_name,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualUNet(nn.Module):
    def __init__(self, in_channels=5, base_channels=32, out_channels=1):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 8, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, kernel_size=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(e4)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)


def prediction_apply_mask(batch, apply_mask):
    if apply_mask == "refine":
        return batch["refine_mask"]
    if apply_mask == "hole":
        return batch["hole_mask"]
    raise ValueError(f"Unknown residual apply mask: {apply_mask}")


def prediction_apply_weight(
    batch,
    apply_mask,
    residual_gate="binary",
    gate_boundary_px=1.0,
    gate_ramp_px=4.0,
):
    mask = prediction_apply_mask(batch, apply_mask)
    weight = mask.to(dtype=batch["base_norm"].dtype)

    if residual_gate == "binary":
        return weight

    if residual_gate == "soft_hole_distance":
        distance = batch["hole_distance"].to(dtype=weight.dtype)
        ramp = max(float(gate_ramp_px), 1e-6)
        gate = torch.clamp((distance - float(gate_boundary_px)) / ramp, 0.0, 1.0)
        return weight * gate

    raise ValueError(f"Unknown residual gate: {residual_gate}")


def predict_depth_norm(
    model,
    batch,
    max_residual_norm,
    apply_mask="refine",
    residual_gate="binary",
    gate_boundary_px=1.0,
    gate_ramp_px=4.0,
    residual_scale=1.0,
):
    residual = model(batch["x"])
    residual = torch.clamp(residual, -max_residual_norm, max_residual_norm) * float(residual_scale)
    apply = prediction_apply_weight(
        batch,
        apply_mask,
        residual_gate=residual_gate,
        gate_boundary_px=gate_boundary_px,
        gate_ramp_px=gate_ramp_px,
    ).to(dtype=residual.dtype)
    return batch["base_norm"] + residual * apply


def compute_loss(model, batch, args):
    pred = predict_depth_norm(
        model,
        batch,
        args.max_residual_norm,
        args.residual_apply_mask,
        args.residual_gate,
        args.gate_boundary_px,
        args.gate_ramp_px,
        args.residual_scale,
    )
    target = batch["target_norm"]
    base = batch["base_norm"]

    valid = batch["valid_mask"] & finite_depth_mask(target) & finite_depth_mask(pred)
    hole = batch["hole_mask"]
    refine = batch["refine_mask"]
    near_valid = refine & (~hole)

    abs_err = torch.abs(pred - target)
    hole_loss = masked_mean(abs_err, valid & hole)
    near_loss = masked_mean(abs_err, valid & near_valid)
    valid_loss = masked_mean(abs_err, valid & (~hole))
    grad_loss = gradient_l1(pred, target, valid, refine)
    smooth_loss = edge_aware_smoothness(pred, base, valid, refine)

    total = (
        args.hole_weight * hole_loss
        + args.near_weight * near_loss
        + args.valid_weight * valid_loss
        + args.grad_weight * grad_loss
        + args.smooth_weight * smooth_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "hole_l1": float(hole_loss.detach().cpu()),
        "near_l1": float(near_loss.detach().cpu()),
        "valid_l1": float(valid_loss.detach().cpu()),
        "grad": float(grad_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
    }


def mae_sum_and_count(pred, target, mask):
    valid = mask & torch.isfinite(pred) & torch.isfinite(target)
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0
    total = torch.abs(pred[valid] - target[valid]).sum().item()
    return float(total), count


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    model.eval()
    totals = {
        "model_global": [0.0, 0],
        "model_hole": [0.0, 0],
        "model_valid": [0.0, 0],
        "base_global": [0.0, 0],
        "base_hole": [0.0, 0],
        "base_valid": [0.0, 0],
    }

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        pred_norm = predict_depth_norm(
            model,
            batch,
            args.max_residual_norm,
            args.residual_apply_mask,
            args.residual_gate,
            args.gate_boundary_px,
            args.gate_ramp_px,
            args.residual_scale,
        )
        scale = batch["scale"].view(-1, 1, 1, 1)
        center = batch["center"].view(-1, 1, 1, 1)
        pred = pred_norm * scale + center
        target = batch["gt_depth"]
        base = batch["depth_base"]

        valid = batch["valid_mask"]
        hole = batch["hole_mask"]
        valid_region = valid & (~hole)

        for prefix, depth in [("model", pred), ("base", base)]:
            for region_name, region_mask in [
                ("global", valid),
                ("hole", valid & hole),
                ("valid", valid_region),
            ]:
                total, count = mae_sum_and_count(depth, target, region_mask)
                key = f"{prefix}_{region_name}"
                totals[key][0] += total
                totals[key][1] += count

    metrics = {}
    for key, (total, count) in totals.items():
        metrics[f"{key}_mae"] = total / count if count > 0 else math.nan
        metrics[f"{key}_count"] = count
    return metrics


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }, path)


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


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

    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump({"train": train_paths, "val": val_paths}, f, indent=2)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_dataset = DepthCompletionCacheDataset(
        train_paths,
        norm_percentiles=args.norm_percentiles,
        min_depth_scale=args.min_depth_scale,
        clip_norm_depth=args.clip_norm_depth,
        input_mode=args.input_mode,
        feature_percentile=args.feature_percentile,
        iq_clip=args.iq_clip,
        feature_clip=args.feature_clip,
    )
    val_dataset = DepthCompletionCacheDataset(
        val_paths,
        norm_percentiles=args.norm_percentiles,
        min_depth_scale=args.min_depth_scale,
        clip_norm_depth=args.clip_norm_depth,
        input_mode=args.input_mode,
        feature_percentile=args.feature_percentile,
        iq_clip=args.iq_clip,
        feature_clip=args.feature_clip,
    ) if val_paths else None
    print(f"Input mode:    {args.input_mode}")
    print(f"Input channels:{train_dataset.input_channels}")
    print(f"Residual mask: {args.residual_apply_mask}")
    print(f"Residual gate: {args.residual_gate}")
    print(f"Residual scale:{args.residual_scale}")

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
    ) if val_dataset is not None else None

    model = ResidualUNet(in_channels=train_dataset.input_channels, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    best_hole_mae = float("inf")
    best_epoch = -1

    if val_loader is not None:
        initial_metrics = evaluate(model, val_loader, args, device)
        print(
            "Initial val: "
            f"model_global={initial_metrics['model_global_mae']:.6f}, "
            f"model_hole={initial_metrics['model_hole_mae']:.6f}, "
            f"base_global={initial_metrics['base_global_mae']:.6f}, "
            f"base_hole={initial_metrics['base_hole_mae']:.6f}"
        )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
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
                    f"hole={loss_parts['hole_l1']:.5f} "
                    f"near={loss_parts['near_l1']:.5f} "
                    f"valid={loss_parts['valid_l1']:.5f} "
                    f"grad={loss_parts['grad']:.5f}"
                )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "seconds": time.time() - t0,
        }

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, args, device)
            record.update(val_metrics)
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"model_global={val_metrics['model_global_mae']:.6f} "
                f"model_hole={val_metrics['model_hole_mae']:.6f} "
                f"model_valid={val_metrics['model_valid_mae']:.6f} | "
                f"base_global={val_metrics['base_global_mae']:.6f} "
                f"base_hole={val_metrics['base_hole_mae']:.6f}"
            )

            current_hole = val_metrics["model_hole_mae"]
            if not math.isnan(current_hole) and current_hole < best_hole_mae:
                best_hole_mae = current_hole
                best_epoch = epoch
                save_checkpoint(
                    os.path.join(args.output_dir, "best.pt"),
                    model,
                    optimizer,
                    epoch,
                    args,
                    val_metrics,
                )
        else:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f}")

        append_jsonl(metrics_path, record)

        save_checkpoint(
            os.path.join(args.output_dir, "last.pt"),
            model,
            optimizer,
            epoch,
            args,
            record,
        )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"),
                model,
                optimizer,
                epoch,
                args,
                record,
            )

    summary = {
        "best_epoch": best_epoch,
        "best_hole_mae": best_hole_mae,
        "output_dir": args.output_dir,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Done.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
