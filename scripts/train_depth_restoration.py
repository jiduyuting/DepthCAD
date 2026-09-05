import argparse
import hashlib
import json
import math
import os
import random
import time
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from inference_depth_postprocess import opencv_depth_inpaint
from train_depth_completion import (
    ResidualUNet,
    edge_aware_smoothness,
    finite_depth_mask,
    gradient_l1,
    masked_mean,
    move_batch_to_device,
    read_list,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a single depth restoration model that denoises valid regions and "
            "completes holes from noisy depth, mask, and confidence."
        )
    )
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/depth_restoration_unet")
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
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--input_mode", type=str, default="noisy",
                        choices=["noisy", "noisy_amp", "noisy_iq", "noisy_iq_amp"],
                        help=(
                            "noisy uses anchor/noisy/mask/conf. noisy_amp adds raw noisy amplitude features. "
                            "noisy_iq adds raw 6-channel noisy IQ; noisy_iq_amp adds both."
                        ))
    parser.add_argument("--include_hole_distance", action="store_true", default=False,
                        help="Add normalized distance-to-hole-boundary as an input channel.")
    parser.add_argument("--anchor_mode", type=str, default="noisy_ns",
                        choices=["noisy_zero", "noisy_ns", "noisy_telea", "cache_base"],
                        help=(
                            "Initial residual anchor. noisy_ns/noisy_telea use only noisy depth and hole mask. "
                            "cache_base uses DepthCAD+plane cache and is only an upper-bound ablation."
                        ))
    parser.add_argument("--anchor_inpaint_radius", type=int, default=15)
    parser.add_argument("--prediction_mode", type=str, default="residual",
                        choices=["residual", "gated_residual"],
                        help=(
                            "residual predicts pred=anchor+residual. gated_residual predicts "
                            "pred=anchor+sigmoid(gate)*residual."
                        ))
    parser.add_argument("--max_residual_norm", type=float, default=4.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--gate_bias_init", type=float, default=2.0,
                        help="Initial bias for the gated_residual gate logit head.")
    parser.add_argument("--gate_prior_weight", type=float, default=0.0,
                        help="Optional L1 prior for gated_residual gate values.")
    parser.add_argument("--gate_hole_target", type=float, default=1.0,
                        help="Gate prior target inside holes when --gate_prior_weight > 0.")
    parser.add_argument("--gate_valid_target", type=float, default=0.25,
                        help="Gate prior target outside holes when --gate_prior_weight > 0.")

    parser.add_argument("--norm_percentiles", type=float, nargs=2, default=[5.0, 95.0])
    parser.add_argument("--min_depth_scale", type=float, default=0.25)
    parser.add_argument("--clip_norm_depth", type=float, default=8.0)
    parser.add_argument("--feature_percentile", type=float, default=99.0)
    parser.add_argument("--feature_clip", type=float, default=3.0)
    parser.add_argument("--iq_clip", type=float, default=3.0,
                        help="Clamp robust-normalized signed IQ channels to +/- this value.")

    parser.add_argument("--hole_weight", type=float, default=5.0)
    parser.add_argument("--valid_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=0.5)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--anchor_weight", type=float, default=0.0,
                        help="Optional regularization to keep valid pixels near the anchor.")
    parser.add_argument("--base_teacher_weight", type=float, default=0.0,
                        help=(
                            "Optional hole-region regularization toward cached depth_base. "
                            "Use only as a teacher/upper-bound ablation."
                        ))

    parser.add_argument("--selection_metric", type=str, default="global",
                        choices=["global", "hole", "composite"],
                        help="Metric used for best.pt. composite = model_hole_mae + model_valid_mae.")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


def collect_cache_paths(args):
    if args.train_list or args.val_list:
        if not args.train_list or not args.val_list:
            raise ValueError("--train_list and --val_list must be provided together.")
        return read_list(args.train_list), read_list(args.val_list)

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


def robust_nonnegative_channels(values, valid_mask, percentile=99.0, clip=3.0):
    values = values.astype(np.float32)
    if values.ndim == 2:
        values = values[None]
    out = np.zeros_like(values, dtype=np.float32)
    for i in range(values.shape[0]):
        ch = values[i]
        valid = valid_mask & np.isfinite(ch)
        if valid.sum() > 0:
            scale = np.percentile(ch[valid], percentile)
        elif np.isfinite(ch).any():
            scale = np.percentile(ch[np.isfinite(ch)], percentile)
        else:
            scale = 1.0
        scale = max(float(scale), 1e-6)
        out[i] = np.clip(np.nan_to_num(ch / scale, nan=0.0), 0.0, clip)
    return out.astype(np.float32)


def make_anchor(depth_noisy, hole_mask, depth_base, mode, radius):
    if mode == "cache_base":
        return depth_base.astype(np.float32)
    if mode == "noisy_zero":
        return depth_noisy.astype(np.float32)
    if mode == "noisy_ns":
        return opencv_depth_inpaint(depth_noisy, hole_mask, method="ns", radius=radius)
    if mode == "noisy_telea":
        return opencv_depth_inpaint(depth_noisy, hole_mask, method="telea", radius=radius)
    raise ValueError(f"Unknown anchor_mode: {mode}")


class DepthRestorationCacheDataset(Dataset):
    def __init__(
        self,
        paths,
        input_mode="noisy",
        include_hole_distance=False,
        anchor_mode="noisy_ns",
        anchor_inpaint_radius=15,
        norm_percentiles=(5.0, 95.0),
        min_depth_scale=0.25,
        clip_norm_depth=8.0,
        feature_percentile=99.0,
        feature_clip=3.0,
        iq_clip=3.0,
        iq_normalization="channel",
        effective_hole_only=True,
        mask_augment=False,
        mask_augment_probability=0.0,
        mask_augment_block_sizes=(4, 8, 12),
        mask_augment_hole_ratios=(0.15, 0.20),
        mask_augment_noise_depth_root=None,
        mask_augment_seed=42,
        mask_augment_deterministic=False,
    ):
        self.paths = list(paths)
        self.input_mode = input_mode
        self.include_hole_distance = bool(include_hole_distance)
        self.anchor_mode = anchor_mode
        self.anchor_inpaint_radius = int(anchor_inpaint_radius)
        self.norm_percentiles = tuple(norm_percentiles)
        self.min_depth_scale = float(min_depth_scale)
        self.clip_norm_depth = float(clip_norm_depth)
        self.feature_percentile = float(feature_percentile)
        self.feature_clip = float(feature_clip)
        self.iq_clip = float(iq_clip)
        self.iq_normalization = str(iq_normalization)
        if self.iq_normalization not in {"channel", "pairwise"}:
            raise ValueError("iq_normalization must be channel or pairwise")
        self.effective_hole_only = bool(effective_hole_only)
        self.mask_augment = bool(mask_augment)
        self.mask_augment_probability = float(mask_augment_probability)
        self.mask_augment_block_sizes = tuple(int(value) for value in mask_augment_block_sizes)
        self.mask_augment_hole_ratios = tuple(float(value) for value in mask_augment_hole_ratios)
        self.mask_augment_noise_depth_root = mask_augment_noise_depth_root
        self.mask_augment_seed = int(mask_augment_seed)
        self.mask_augment_deterministic = bool(mask_augment_deterministic)
        if self.mask_augment and not self.mask_augment_noise_depth_root:
            raise ValueError("mask_augment requires mask_augment_noise_depth_root")
        if not self.mask_augment_block_sizes:
            raise ValueError("mask_augment_block_sizes cannot be empty")
        if len(self.mask_augment_hole_ratios) != 2:
            raise ValueError("mask_augment_hole_ratios must contain min and max")
        self.input_channels = 4
        if self.include_hole_distance:
            self.input_channels += 1
        if self.input_mode == "noisy_amp":
            self.input_channels += 4
        if self.input_mode == "noisy_iq":
            self.input_channels += 6
        if self.input_mode == "noisy_iq_amp":
            self.input_channels += 10

    def __len__(self):
        return len(self.paths)

    def _normalize_depth(self, depth, center, scale):
        depth = (depth - center) / scale
        depth = np.nan_to_num(depth, nan=0.0, neginf=-self.clip_norm_depth, posinf=self.clip_norm_depth)
        return np.clip(depth, -self.clip_norm_depth, self.clip_norm_depth).astype(np.float32)

    def _require_keys(self, data, keys, path):
        missing = [key for key in keys if key not in data.files]
        if missing:
            raise KeyError(
                f"{path} is missing keys {missing}. "
                "Regenerate the cache with --save_depth_completion_cache --depth_cache_save_iq "
                "for noisy_iq/noisy_iq_amp modes."
            )

    def _robust_signed_channels(self, values, valid_mask):
        values = values.astype(np.float32)
        if values.ndim == 2:
            values = values[None]
        out = np.zeros_like(values, dtype=np.float32)
        for i in range(values.shape[0]):
            ch = values[i]
            valid = valid_mask & np.isfinite(ch)
            if valid.sum() > 0:
                scale = np.percentile(np.abs(ch[valid]), self.feature_percentile)
            elif np.isfinite(ch).any():
                scale = np.percentile(np.abs(ch[np.isfinite(ch)]), self.feature_percentile)
            else:
                scale = 1.0
            scale = max(float(scale), 1e-6)
            out[i] = np.clip(np.nan_to_num(ch / scale, nan=0.0), -self.iq_clip, self.iq_clip)
        return out.astype(np.float32)

    def _robust_iq_channels(self, values, valid_mask):
        """Normalize each I/Q pair with one common scale to preserve phase."""
        values = values.astype(np.float32)
        if values.ndim != 3 or values.shape[0] < 2 or values.shape[0] % 2:
            return self._robust_signed_channels(values, valid_mask)
        out = np.zeros_like(values, dtype=np.float32)
        for i in range(0, values.shape[0], 2):
            pair = values[i : i + 2]
            amplitude = np.sqrt(pair[0] ** 2 + pair[1] ** 2)
            finite = valid_mask & np.isfinite(amplitude)
            if finite.sum() > 0:
                scale = np.percentile(amplitude[finite], self.feature_percentile)
            elif np.isfinite(amplitude).any():
                scale = np.percentile(amplitude[np.isfinite(amplitude)], self.feature_percentile)
            else:
                scale = 1.0
            scale = max(float(scale), 1e-6)
            out[i : i + 2] = np.clip(np.nan_to_num(pair / scale, nan=0.0), -self.iq_clip, self.iq_clip)
        return out.astype(np.float32)

    def _sample_name(self, data, path):
        if "sample_name" in data.files:
            value = data["sample_name"]
            return str(value.item()) if np.ndim(value) == 0 else str(value)
        return os.path.splitext(os.path.basename(path))[0]

    def _load_clean_noisy_depth(self, sample_name, shape):
        scene, view, frame = sample_name.split("/")
        path = os.path.join(self.mask_augment_noise_depth_root, scene, view, f"{frame}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing clean noisy depth for mask augmentation: {path}")
        depth = np.load(path).astype(np.float32)
        if depth.shape != shape:
            depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return depth

    @staticmethod
    def _block_hole_mask(valid_mask, confidence, block_size, target_ratio, rng):
        height, width = valid_mask.shape
        target_count = min(int(round(float(target_ratio) * valid_mask.size)), int(valid_mask.sum()))
        if target_count <= 0:
            return np.zeros_like(valid_mask, dtype=bool)

        blocks = []
        offset_y = int(rng.randint(0, max(block_size, 1)))
        offset_x = int(rng.randint(0, max(block_size, 1)))
        for y0 in range(-offset_y, height, block_size):
            for x0 in range(-offset_x, width, block_size):
                y1, y2 = max(0, y0), min(height, y0 + block_size)
                x1, x2 = max(0, x0), min(width, x0 + block_size)
                if y1 >= y2 or x1 >= x2:
                    continue
                block_valid = valid_mask[y1:y2, x1:x2]
                if not block_valid.any():
                    continue
                score = float(np.median(confidence[y1:y2, x1:x2][block_valid]))
                blocks.append((score + float(rng.uniform(0.0, 1e-6)), y1, y2, x1, x2))
        blocks.sort(key=lambda row: row[0])

        hole = np.zeros_like(valid_mask, dtype=bool)
        for _, y1, y2, x1, x2 in blocks:
            candidate = valid_mask[y1:y2, x1:x2] & (~hole[y1:y2, x1:x2])
            remaining = target_count - int(hole.sum())
            if remaining <= 0:
                break
            if int(candidate.sum()) <= remaining:
                hole[y1:y2, x1:x2] |= candidate
            else:
                yy, xx = np.nonzero(candidate)
                values = confidence[y1:y2, x1:x2][yy, xx]
                keep = np.argsort(values, kind="stable")[:remaining]
                hole[y1 + yy[keep], x1 + xx[keep]] = True

        kernel = np.ones((3, 3), dtype=np.uint8)
        hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        hole &= valid_mask
        if int(hole.sum()) > target_count:
            indices = np.flatnonzero(hole)
            order = np.argsort(confidence.reshape(-1)[indices], kind="stable")[:target_count]
            trimmed = np.zeros_like(hole, dtype=bool)
            trimmed.reshape(-1)[indices[order]] = True
            hole = trimmed
        return hole

    def _augment_mask(self, data, sample_name, gt_depth, valid_mask, rng):
        if "noisy_iq" not in data.files:
            raise KeyError("mask augmentation requires raw noisy_iq in the cache")
        noisy_iq = data["noisy_iq"].astype(np.float32)
        amplitude = np.sqrt(noisy_iq[0::2] ** 2 + noisy_iq[1::2] ** 2).mean(axis=0)
        amp_values = amplitude[valid_mask & np.isfinite(amplitude)]
        if amp_values.size == 0:
            return None
        low = float(np.percentile(amp_values, 5.0))
        high = float(np.percentile(amp_values, 95.0))
        confidence_amp = np.clip((amplitude - low) / (high - low + 1e-8), 0.0, 1.0)
        grad_x = cv2.Sobel(gt_depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gt_depth, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        finite_gradient = gradient[valid_mask & np.isfinite(gradient)]
        grad_scale = float(np.percentile(finite_gradient, 95.0)) if finite_gradient.size else 1.0
        confidence_edge = np.exp(-np.nan_to_num(gradient / max(grad_scale, 1e-6), nan=1.0))
        confidence = (confidence_amp * confidence_edge * valid_mask).astype(np.float32)
        block_size = int(self.mask_augment_block_sizes[int(rng.randint(len(self.mask_augment_block_sizes)))])
        ratio_min, ratio_max = self.mask_augment_hole_ratios
        target_ratio = float(rng.uniform(min(ratio_min, ratio_max), max(ratio_min, ratio_max)))
        hole = self._block_hole_mask(valid_mask, confidence, block_size, target_ratio, rng)
        clean_depth = self._load_clean_noisy_depth(sample_name, gt_depth.shape)
        return hole, confidence, clean_depth, noisy_iq

    def __getitem__(self, index):
        path = self.paths[index]
        data = np.load(path)

        depth_noisy = data["depth_noisy"].astype(np.float32)
        if "depth_base" in data.files:
            depth_base = data["depth_base"].astype(np.float32)
        else:
            if self.anchor_mode == "cache_base":
                raise KeyError(f"{path} is missing depth_base required by anchor_mode='cache_base'")
            depth_base = depth_noisy.copy()
        gt_depth = data["gt_depth"].astype(np.float32)
        raw_hole_mask = data["hole_mask"] > 0.5
        confidence = data["confidence"].astype(np.float32)
        valid_mask = (
            (data["valid_mask"] > 0.5)
            & np.isfinite(gt_depth)
            & (gt_depth > 0.1)
            & (gt_depth < 9.9)
        )
        sample_name = self._sample_name(data, path)
        if self.mask_augment_deterministic:
            digest = hashlib.sha256(f"{self.mask_augment_seed}:{sample_name}".encode("utf-8")).digest()
            rng_seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
        else:
            rng_seed = (int(np.random.randint(0, 2**31 - 1)) + self.mask_augment_seed) % (2**32 - 1)
        rng = np.random.RandomState(rng_seed)
        augmented = None
        if self.mask_augment and rng.rand() < self.mask_augment_probability:
            augmented = self._augment_mask(data, sample_name, gt_depth, valid_mask, rng)
        if augmented is not None:
            augmented_hole, confidence, depth_noisy, raw_noisy_iq = augmented
            raw_hole_mask = augmented_hole
            depth_noisy = depth_noisy.copy()
            depth_noisy[raw_hole_mask] = 0.0
        else:
            raw_noisy_iq = data["noisy_iq"].astype(np.float32) if "noisy_iq" in data.files else None

        hole_bool = raw_hole_mask & valid_mask if self.effective_hole_only else raw_hole_mask
        hole_mask = hole_bool.astype(np.float32)
        confidence = confidence.copy()
        confidence[hole_bool | (~valid_mask)] = 0.0
        if np.any(hole_bool & (~valid_mask)):
            raise AssertionError(f"effective hole overlaps invalid GT: {path}")
        hole_distance = cv2.distanceTransform(
            hole_bool.astype(np.uint8), cv2.DIST_L2, 3
        ).astype(np.float32)

        anchor = make_anchor(
            depth_noisy,
            hole_mask,
            depth_base,
            self.anchor_mode,
            self.anchor_inpaint_radius,
        ).astype(np.float32)

        stat_mask = (~hole_bool) & np.isfinite(anchor) & (anchor > 0.1) & (anchor < 9.9)
        if stat_mask.sum() == 0:
            stat_mask = np.isfinite(anchor) & (anchor > 0.1) & (anchor < 9.9)
        if stat_mask.sum() > 0:
            lo, hi = np.percentile(anchor[stat_mask], self.norm_percentiles)
            center = float(np.median(anchor[stat_mask]))
            scale = float(hi - lo)
        else:
            center = 0.0
            scale = 1.0
        scale = max(scale, self.min_depth_scale)

        anchor_norm = self._normalize_depth(anchor, center, scale)
        noisy_norm = self._normalize_depth(depth_noisy, center, scale)
        target_norm = self._normalize_depth(gt_depth, center, scale)
        base_norm = self._normalize_depth(depth_base, center, scale)

        channels = [
            anchor_norm,
            noisy_norm,
            hole_mask,
            confidence,
        ]

        if self.include_hole_distance:
            dist = cv2.distanceTransform(hole_bool.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
            dist = np.clip(dist / max(float(self.anchor_inpaint_radius), 1.0), 0.0, 1.0)
            channels.append(dist)

        if self.input_mode in ["noisy_amp", "noisy_iq_amp"]:
            for key in ["noisy_amplitude", "noisy_amplitude_mean"]:
                if key not in data.files:
                    raise KeyError(f"{path} is missing {key}; regenerate cache with amplitude features.")
            noisy_amplitude = data["noisy_amplitude"].astype(np.float32).copy()
            noisy_amplitude_mean = data["noisy_amplitude_mean"].astype(np.float32).copy()
            if augmented is not None:
                noisy_amplitude = np.sqrt(
                    raw_noisy_iq[0::2] ** 2 + raw_noisy_iq[1::2] ** 2
                ).astype(np.float32)
                noisy_amplitude_mean = noisy_amplitude.mean(axis=0).astype(np.float32)
                noisy_amplitude[:, hole_bool] = 0.0
                noisy_amplitude_mean[hole_bool] = 0.0
            channels.extend(
                robust_nonnegative_channels(
                    noisy_amplitude,
                    valid_mask,
                    percentile=self.feature_percentile,
                    clip=self.feature_clip,
                )
            )
            channels.extend(
                robust_nonnegative_channels(
                    noisy_amplitude_mean,
                    valid_mask,
                    percentile=self.feature_percentile,
                    clip=self.feature_clip,
                )
            )

        if self.input_mode in ["noisy_iq", "noisy_iq_amp"]:
            self._require_keys(data, ["noisy_iq"], path)
            channels.extend(
                self._robust_iq_channels(raw_noisy_iq, valid_mask)
                if self.iq_normalization == "pairwise"
                else self._robust_signed_channels(raw_noisy_iq, valid_mask)
            )

        x = np.stack(channels, axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "anchor_norm": torch.from_numpy(anchor_norm[None]),
            "target_norm": torch.from_numpy(target_norm[None]),
            "base_norm": torch.from_numpy(base_norm[None]),
            "depth_anchor": torch.from_numpy(anchor[None]),
            "depth_noisy": torch.from_numpy(depth_noisy[None]),
            "depth_base": torch.from_numpy(depth_base[None]),
            "gt_depth": torch.from_numpy(gt_depth[None]),
            "hole_mask": torch.from_numpy(hole_mask[None].astype(np.bool_)),
            "raw_hole_mask": torch.from_numpy(raw_hole_mask[None].astype(np.bool_)),
            "gt_invalid_mask": torch.from_numpy((~valid_mask)[None].astype(np.bool_)),
            "hole_distance": torch.from_numpy(hole_distance[None]),
            "hole_area_fraction": torch.tensor(float(hole_bool.mean()), dtype=torch.float32),
            "valid_mask": torch.from_numpy(valid_mask[None].astype(np.bool_)),
            "center": torch.tensor(center, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "path": path,
            "sample_name": sample_name,
        }


def model_output_channels(prediction_mode):
    if prediction_mode == "gated_residual":
        return 2
    if prediction_mode == "residual":
        return 1
    raise ValueError(f"Unknown prediction_mode: {prediction_mode}")


def initialize_prediction_head(model, args):
    if args.prediction_mode != "gated_residual":
        return
    if not hasattr(model, "out") or model.out.bias is None or model.out.bias.numel() < 2:
        return
    with torch.no_grad():
        model.out.bias[1].fill_(float(args.gate_bias_init))


def predict_depth_norm(
    model,
    batch,
    max_residual_norm,
    residual_scale=1.0,
    prediction_mode="residual",
    return_aux=False,
):
    output = model(batch["x"])
    residual = output[:, :1]
    residual = torch.clamp(residual, -max_residual_norm, max_residual_norm)

    aux = {"residual": residual}
    if prediction_mode == "gated_residual":
        if output.shape[1] < 2:
            raise ValueError("gated_residual requires a model with 2 output channels.")
        gate = torch.sigmoid(output[:, 1:2])
        pred = batch["anchor_norm"] + gate * residual * float(residual_scale)
        aux["gate"] = gate
    elif prediction_mode == "residual":
        pred = batch["anchor_norm"] + residual * float(residual_scale)
        aux["gate"] = None
    else:
        raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

    if return_aux:
        return pred, aux
    return pred


def compute_loss(model, batch, args):
    pred, aux = predict_depth_norm(
        model,
        batch,
        args.max_residual_norm,
        args.residual_scale,
        args.prediction_mode,
        return_aux=True,
    )
    target = batch["target_norm"]
    anchor = batch["anchor_norm"]
    base = batch["base_norm"]

    valid = batch["valid_mask"] & finite_depth_mask(target) & finite_depth_mask(pred)
    hole = batch["hole_mask"]
    valid_region = valid & (~hole)
    hole_region = valid & hole

    abs_err = torch.abs(pred - target)
    hole_loss = masked_mean(abs_err, hole_region)
    valid_loss = masked_mean(abs_err, valid_region)
    grad_loss = gradient_l1(pred, target, valid, valid)
    smooth_loss = edge_aware_smoothness(pred, anchor, valid, valid)
    anchor_loss = masked_mean(torch.abs(pred - anchor), valid_region)
    base_teacher_loss = masked_mean(torch.abs(pred - base), hole_region)
    gate_prior_loss = pred.new_tensor(0.0)
    if args.prediction_mode == "gated_residual" and args.gate_prior_weight > 0:
        gate = aux["gate"]
        gate_target = torch.where(
            hole,
            torch.full_like(gate, float(args.gate_hole_target)),
            torch.full_like(gate, float(args.gate_valid_target)),
        )
        gate_prior_loss = masked_mean(torch.abs(gate - gate_target), valid)

    total = (
        args.hole_weight * hole_loss
        + args.valid_weight * valid_loss
        + args.grad_weight * grad_loss
        + args.smooth_weight * smooth_loss
        + args.anchor_weight * anchor_loss
        + args.base_teacher_weight * base_teacher_loss
        + args.gate_prior_weight * gate_prior_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "hole_l1": float(hole_loss.detach().cpu()),
        "valid_l1": float(valid_loss.detach().cpu()),
        "grad": float(grad_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()),
        "base_teacher": float(base_teacher_loss.detach().cpu()),
        "gate_prior": float(gate_prior_loss.detach().cpu()),
    }


def mae_sum_and_count(pred, target, mask):
    valid = mask & torch.isfinite(pred) & torch.isfinite(target)
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0
    return float(torch.abs(pred[valid] - target[valid]).sum().item()), count


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    model.eval()
    totals = {}
    for prefix in ["model", "anchor", "noisy", "base"]:
        for region in ["global", "hole", "valid"]:
            totals[f"{prefix}_{region}"] = [0.0, 0]

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        pred_norm = predict_depth_norm(
            model,
            batch,
            args.max_residual_norm,
            args.residual_scale,
            args.prediction_mode,
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


def metric_for_selection(metrics, args):
    if args.selection_metric == "global":
        return metrics["model_global_mae"]
    if args.selection_metric == "hole":
        return metrics["model_hole_mae"]
    if args.selection_metric == "composite":
        return metrics["model_hole_mae"] + metrics["model_valid_mae"]
    raise ValueError(args.selection_metric)


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


def print_eval_line(prefix, metrics):
    print(
        f"{prefix}: "
        f"model_global={metrics['model_global_mae']:.6f} "
        f"model_hole={metrics['model_hole_mae']:.6f} "
        f"model_valid={metrics['model_valid_mae']:.6f} | "
        f"anchor_global={metrics['anchor_global_mae']:.6f} "
        f"anchor_hole={metrics['anchor_hole_mae']:.6f} "
        f"anchor_valid={metrics['anchor_valid_mae']:.6f} | "
        f"noisy_global={metrics['noisy_global_mae']:.6f} "
        f"noisy_hole={metrics['noisy_hole_mae']:.6f} "
        f"noisy_valid={metrics['noisy_valid_mae']:.6f} | "
        f"base_global={metrics['base_global_mae']:.6f} "
        f"base_hole={metrics['base_hole_mae']:.6f} "
        f"base_valid={metrics['base_valid_mae']:.6f}"
    )


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
    print(f"Prediction:    {args.prediction_mode}")
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
    train_dataset = DepthRestorationCacheDataset(train_paths, **dataset_kwargs)
    val_dataset = DepthRestorationCacheDataset(val_paths, **dataset_kwargs) if val_paths else None
    print(f"Input channels:{train_dataset.input_channels}")

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

    model = ResidualUNet(
        in_channels=train_dataset.input_channels,
        base_channels=args.base_channels,
        out_channels=model_output_channels(args.prediction_mode),
    ).to(device)
    initialize_prediction_head(model, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_score = float("inf")
    best_epoch = -1
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    if val_loader is not None:
        initial_metrics = evaluate(model, val_loader, args, device)
        print_eval_line("Initial val", initial_metrics)

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
                    f"valid={loss_parts['valid_l1']:.5f} "
                    f"grad={loss_parts['grad']:.5f} "
                    f"smooth={loss_parts['smooth']:.5f}"
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
        save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, epoch, args, record)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"), model, optimizer, epoch, args, record)

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selection_metric": args.selection_metric,
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
