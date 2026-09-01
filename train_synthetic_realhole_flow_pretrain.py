import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from eval_depth_restoration import load_checkpoint
from infer_real_depth_flow import move_condition_to_device, normalize_depth
from inference_depth_postprocess import opencv_depth_inpaint
from real_depth_masked_self_test import make_block_mask
from real_raw9_masked_self_test import (
    build_real_hole_component_library,
    collect_pairs,
    filter_and_rebalance_component_library,
    make_artificial_mask,
    summarize_component_library,
)
from train_depth_completion import seed_everything
from train_depth_restoration import collect_cache_paths, robust_nonnegative_channels, save_checkpoint
from train_real_raw9_flow_finetune import (
    add_checkpoint_args,
    autocast_cuda,
    build_model,
    compute_loss,
    evaluate,
    make_grad_scaler,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Supervised pre-train a noisy_amp flow model on synthetic clean depth with "
            "real-hole-shaped masks sampled from paired real depth files."
        )
    )
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--train_list", type=str, default=None)
    parser.add_argument("--val_list", type=str, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--real_raw_dir", type=str, required=True)
    parser.add_argument("--real_depth_dir", type=str, required=True)
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="output/real_raw9_flow_finetune_holefocus_iq6_realholes_continue_e20_lr5e6/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/synthetic_realhole_flow_pretrain_iq6_holefocus",
    )

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--masks_per_sample", type=int, default=4)
    parser.add_argument("--val_masks_per_sample", type=int, default=2)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="real_hole_shapes",
        choices=["block", "real_hole_shapes", "real_hole_speckle_shapes"],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.10)
    parser.add_argument("--min_block_size", type=int, default=12)
    parser.add_argument("--max_block_size", type=int, default=72)
    parser.add_argument("--real_hole_min_area", type=int, default=24)
    parser.add_argument("--real_hole_max_area", type=int, default=0)
    parser.add_argument("--real_hole_min_overlap", type=float, default=0.6)
    parser.add_argument("--real_hole_max_components", type=int, default=24)
    parser.add_argument("--max_mask_retries", type=int, default=8)
    parser.add_argument(
        "--component_val_ratio",
        type=float,
        default=0.0,
        help=(
            "Hold out this fraction of real-hole component source stems for validation masks. "
            "0 keeps the historical behavior where train/val sample masks from the same library."
        ),
    )
    parser.add_argument(
        "--amplitude_mode",
        type=str,
        default="iq6",
        choices=["iq6", "raw_258"],
        help="Amplitude convention used when mining real speckle components from raw9 inputs.",
    )
    parser.add_argument(
        "--real_speckle_train_min_area",
        type=int,
        default=6,
        help="Drop mined amp-speckle components smaller than this before synthetic training.",
    )
    parser.add_argument(
        "--real_speckle_train_max_area",
        type=int,
        default=0,
        help="Optional max area for mined amp-speckle components used in synthetic training.",
    )
    parser.add_argument(
        "--real_speckle_component_ratio",
        type=float,
        default=0.6,
        help=(
            "Target fraction of amp-speckle components inside the mixed real-hole library. "
            "Used only with --mask_mode=real_hole_speckle_shapes."
        ),
    )
    parser.add_argument("--clean_outlier_mad_scale", type=float, default=6.0)
    parser.add_argument("--clean_outlier_abs", type=float, default=0.35)
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

    parser.add_argument("--source_depth", type=str, default="clean", choices=["clean", "noisy"])
    parser.add_argument("--amplitude_source", type=str, default="noisy", choices=["noisy", "denoised"])
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
    parser.add_argument("--hole_depth_threshold", type=float, default=1.0)
    parser.add_argument("--valid_min_depth", type=float, default=1.0)
    parser.add_argument("--valid_max_depth", type=float, default=9.9)
    parser.add_argument("--post_clip_mode", type=str, default="valid_range",
                        choices=["none", "valid_range", "valid_percentile"])
    parser.add_argument("--post_clip_percentiles", type=float, nargs=2, default=[0.5, 99.5])

    parser.add_argument("--mask_loss_weight", type=float, default=4.0)
    parser.add_argument("--mask_center_weight", type=float, default=2.0)
    parser.add_argument("--valid_loss_weight", type=float, default=0.02)
    parser.add_argument("--grad_loss_weight", type=float, default=0.0)
    parser.add_argument("--hole_grad_loss_weight", type=float, default=0.25)
    parser.add_argument("--boundary_grad_loss_weight", type=float, default=0.5)
    parser.add_argument("--boundary_l1_loss_weight", type=float, default=0.05)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--eval_component_area_threshold", type=int, default=700)
    parser.add_argument("--selection_metric", type=str, default="model_mask_mae")
    parser.add_argument("--log_every", type=int, default=40)
    parser.add_argument("--save_every", type=int, default=10)
    return parser.parse_args()


class SyntheticRealHoleFlowDataset(Dataset):
    def __init__(
        self,
        paths,
        ckpt_args,
        args,
        masks_per_sample,
        fixed_masks=False,
        hole_component_library=None,
    ):
        self.paths = list(paths)
        self.ckpt_args = ckpt_args
        self.args = args
        self.masks_per_sample = int(masks_per_sample)
        self.fixed_masks = bool(fixed_masks)
        self.hole_component_library = hole_component_library or []
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.paths) * self.masks_per_sample

    def _make_mask(self, valid, rng):
        for _ in range(max(1, int(self.args.max_mask_retries))):
            mask = make_artificial_mask(
                valid,
                rng,
                self.args,
                component_library=self.hole_component_library,
                source_stem=None,
            )
            if mask.any():
                return mask
        return make_block_mask(
            valid,
            rng,
            self.args.mask_ratio,
            self.args.min_block_size,
            self.args.max_block_size,
        )

    def _normalize_depth(self, depth, center, scale):
        return normalize_depth(
            depth,
            center,
            scale,
            float(self.ckpt_args.get("clip_norm_depth", 8.0)),
        )

    def __getitem__(self, index):
        path_index = index // self.masks_per_sample
        repeat = index % self.masks_per_sample
        path = self.paths[path_index]
        data = np.load(path)

        gt = data["gt_depth"].astype(np.float32)
        cache_valid = (data["valid_mask"] > 0.5) if "valid_mask" in data.files else np.ones_like(gt, dtype=bool)
        valid = (
            cache_valid
            & np.isfinite(gt)
            & (gt >= float(self.args.valid_min_depth))
            & (gt <= float(self.args.valid_max_depth))
        )
        if valid.sum() == 0:
            raise ValueError(f"No valid synthetic GT pixels in {path}")

        if self.args.source_depth == "noisy":
            observed = data["depth_noisy"].astype(np.float32).copy()
            observed[~valid] = 0.0
        else:
            observed = gt.copy()
            observed[~valid] = 0.0

        epoch_offset = 0 if self.fixed_masks else self.epoch * 1000003
        rng = np.random.default_rng(int(self.args.seed) + epoch_offset + path_index * 1009 + repeat)
        artificial_mask = self._make_mask(valid, rng)
        artificial_mask &= valid

        mask_distance = cv2.distanceTransform(artificial_mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        max_distance = float(mask_distance.max())
        if max_distance > 0:
            mask_distance /= max_distance

        corrupted = observed.copy()
        corrupted[artificial_mask] = 0.0
        hole = artificial_mask | (~np.isfinite(corrupted)) | (corrupted <= float(self.args.hole_depth_threshold))
        confidence = (~hole).astype(np.float32)

        ckpt_radius = self.ckpt_args.get("anchor_inpaint_radius", 15)
        if ckpt_radius is None:
            ckpt_radius = 15
        radius = int(self.args.anchor_inpaint_radius) if self.args.anchor_inpaint_radius is not None else int(ckpt_radius)
        anchor = opencv_depth_inpaint(corrupted, hole, method="ns", radius=radius).astype(np.float32)

        stat_mask = (
            (~hole)
            & valid
            & np.isfinite(anchor)
            & (anchor > float(self.args.valid_min_depth))
            & (anchor < float(self.args.valid_max_depth))
        )
        if stat_mask.sum() == 0:
            stat_mask = valid & np.isfinite(anchor)
        if stat_mask.sum() > 0:
            lo, hi = np.percentile(anchor[stat_mask], self.ckpt_args.get("norm_percentiles", [5.0, 95.0]))
            center = float(np.median(anchor[stat_mask]))
            scale = float(hi - lo)
        else:
            center = 0.0
            scale = 1.0
        scale = max(scale, float(self.ckpt_args.get("min_depth_scale", 0.25)))

        anchor_norm = self._normalize_depth(anchor, center, scale)
        noisy_norm = self._normalize_depth(corrupted, center, scale)
        target_norm = self._normalize_depth(gt, center, scale)

        amp_key = f"{self.args.amplitude_source}_amplitude"
        amp_mean_key = f"{self.args.amplitude_source}_amplitude_mean"
        if amp_key not in data.files:
            raise KeyError(f"{path} is missing {amp_key}")
        amplitude = data[amp_key].astype(np.float32).copy()
        if amp_mean_key in data.files:
            amplitude_mean = data[amp_mean_key].astype(np.float32).copy()
        else:
            amplitude_mean = amplitude.mean(axis=0).astype(np.float32)
        amplitude[:, hole] = 0.0
        amplitude_mean[hole] = 0.0

        channels = [
            anchor_norm,
            noisy_norm,
            hole.astype(np.float32),
            confidence,
        ]
        channels.extend(
            robust_nonnegative_channels(
                amplitude,
                valid,
                percentile=self.ckpt_args.get("feature_percentile", 99.0),
                clip=self.ckpt_args.get("feature_clip", 3.0),
            )
        )
        channels.extend(
            robust_nonnegative_channels(
                amplitude_mean,
                valid,
                percentile=self.ckpt_args.get("feature_percentile", 99.0),
                clip=self.ckpt_args.get("feature_clip", 3.0),
            )
        )
        x = np.stack(channels, axis=0).astype(np.float32)

        sample_name = str(data["sample_name"]) if "sample_name" in data.files else os.path.basename(path)
        return {
            "x": torch.from_numpy(x),
            "anchor_norm": torch.from_numpy(anchor_norm[None]),
            "target_norm": torch.from_numpy(target_norm[None]),
            "hole_mask": torch.from_numpy(hole[None].astype(np.bool_)),
            "valid_mask": torch.from_numpy(valid[None].astype(np.bool_)),
            "artificial_mask": torch.from_numpy(artificial_mask[None].astype(np.bool_)),
            "mask_distance": torch.from_numpy(mask_distance[None].astype(np.float32)),
            "depth_anchor": torch.from_numpy(anchor[None].astype(np.float32)),
            "depth_noisy": torch.from_numpy(corrupted[None].astype(np.float32)),
            "gt_depth": torch.from_numpy(gt[None].astype(np.float32)),
            "center": torch.tensor(center, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "sample_name": f"{sample_name}_r{repeat:02d}",
            "path": path,
        }


def split_component_library_by_source(component_library, val_ratio, seed):
    if not component_library:
        return [], [], {
            "component_val_ratio": float(val_ratio),
            "reason": "empty_library",
        }
    val_ratio = float(val_ratio)
    if val_ratio <= 0.0:
        return list(component_library), list(component_library), {
            "component_val_ratio": val_ratio,
            "split_enabled": False,
            "train_components": int(len(component_library)),
            "val_components": int(len(component_library)),
            "note": "train and val share the same component library",
        }

    sources = sorted({str(component.get("source", "")) for component in component_library})
    if len(sources) < 2:
        return list(component_library), list(component_library), {
            "component_val_ratio": val_ratio,
            "split_enabled": False,
            "source_count": int(len(sources)),
            "reason": "not_enough_sources",
        }

    rng = np.random.default_rng(int(seed))
    shuffled = list(sources)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_sources = set(shuffled[:val_count])
    train_sources = set(shuffled[val_count:])
    train_components = [c for c in component_library if str(c.get("source", "")) in train_sources]
    val_components = [c for c in component_library if str(c.get("source", "")) in val_sources]

    if not train_components or not val_components:
        return list(component_library), list(component_library), {
            "component_val_ratio": val_ratio,
            "split_enabled": False,
            "source_count": int(len(sources)),
            "reason": "empty_component_side_after_split",
        }

    return train_components, val_components, {
        "component_val_ratio": val_ratio,
        "split_enabled": True,
        "source_count": int(len(sources)),
        "train_source_count": int(len(train_sources)),
        "val_source_count": int(len(val_sources)),
        "train_components": int(len(train_components)),
        "val_components": int(len(val_components)),
        "val_sources": sorted(val_sources),
    }


def synthetic_valid_pixel_count(path, args):
    data = np.load(path)
    gt = data["gt_depth"].astype(np.float32)
    cache_valid = (data["valid_mask"] > 0.5) if "valid_mask" in data.files else np.ones_like(gt, dtype=bool)
    valid = (
        cache_valid
        & np.isfinite(gt)
        & (gt >= float(args.valid_min_depth))
        & (gt <= float(args.valid_max_depth))
    )
    return int(valid.sum())


def filter_valid_synthetic_paths(paths, args, split_name):
    valid_paths = []
    skipped = []
    for path in paths:
        count = synthetic_valid_pixel_count(path, args)
        if count > 0:
            valid_paths.append(path)
        else:
            skipped.append(path)
    if skipped:
        print(
            f"Skipped {len(skipped)} {split_name} synthetic samples with no GT pixels in "
            f"[{args.valid_min_depth}, {args.valid_max_depth}]."
        )
        print(f"First skipped {split_name} sample: {skipped[0]}")
    return valid_paths, skipped


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


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
    if args.mask_mode == "real_hole_speckle_shapes":
        args.method = "synthetic_real_hole_speckle_supervised_pretrain"
    else:
        args.method = "synthetic_real_hole_supervised_pretrain"

    real_pairs = collect_pairs(args.real_raw_dir, args.real_depth_dir)
    if not real_pairs:
        raise FileNotFoundError(f"No paired real raw/depth files found in {args.real_raw_dir} and {args.real_depth_dir}")
    hole_component_library = []
    component_filter_summary = {}
    component_library_summary = {"total": 0, "threshold_hole": 0, "amp_speckle": 0}
    if args.mask_mode in {"real_hole_shapes", "real_hole_speckle_shapes"}:
        hole_component_library = build_real_hole_component_library(
            real_pairs,
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
            raise ValueError("No eligible real-hole components found.")
    train_component_library, val_component_library, component_split_summary = split_component_library_by_source(
        hole_component_library,
        args.component_val_ratio,
        args.seed,
    )
    train_component_library_summary = summarize_component_library(train_component_library)
    val_component_library_summary = summarize_component_library(val_component_library)

    train_paths, val_paths = collect_cache_paths(args)
    train_paths, skipped_train_paths = filter_valid_synthetic_paths(train_paths, args, "train")
    val_paths, skipped_val_paths = filter_valid_synthetic_paths(val_paths, args, "val")
    if not train_paths:
        raise ValueError("No training cache paths found.")
    if not val_paths:
        raise ValueError("No validation cache paths found.")

    with open(os.path.join(args.output_dir, "split.json"), "w") as f:
        json.dump(
            {
                "train": train_paths,
                "val": val_paths,
                "skipped_train": skipped_train_paths,
                "skipped_val": skipped_val_paths,
                "component_library_summary": component_library_summary,
                "component_filter_summary": component_filter_summary,
                "component_split_summary": component_split_summary,
                "train_component_library_summary": train_component_library_summary,
                "val_component_library_summary": val_component_library_summary,
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    train_dataset = SyntheticRealHoleFlowDataset(
        train_paths,
        ckpt_args,
        args,
        masks_per_sample=args.masks_per_sample,
        fixed_masks=False,
        hole_component_library=train_component_library,
    )
    val_dataset = SyntheticRealHoleFlowDataset(
        val_paths,
        ckpt_args,
        args,
        masks_per_sample=args.val_masks_per_sample,
        fixed_masks=True,
        hole_component_library=val_component_library,
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

    model = build_model(ckpt, ckpt_args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(enabled=(args.amp and device.type == "cuda"))

    print(f"Synthetic train={len(train_paths)} val={len(val_paths)}")
    print(
        "Real component library="
        f"{component_library_summary.get('total', 0)} "
        f"(threshold={component_library_summary.get('threshold_hole', 0)}, "
        f"amp_speckle={component_library_summary.get('amp_speckle', 0)})"
    )
    print(
        "Component split="
        f"{json.dumps(component_split_summary, sort_keys=True)}"
    )
    if component_filter_summary:
        print(
            "Component filtering="
            f"{json.dumps(component_filter_summary, sort_keys=True)}"
        )
    print(f"Device: {device}")
    print(f"Pretrained: {args.pretrained_checkpoint}")
    print(f"Output: {args.output_dir}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    best_score = float("inf")
    best_epoch = -1

    initial_metrics, _ = evaluate(model, val_loader, args, device)
    print(f"Initial val: {json.dumps(initial_metrics, sort_keys=True)}")

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        t0 = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = move_condition_to_device(batch, device)
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
                    f"valid={parts['valid']:.5f} grad={parts['grad']:.5f} "
                    f"hole_grad={parts['hole_grad']:.5f} "
                    f"boundary_grad={parts['boundary_grad']:.5f}"
                )

        metrics, _ = evaluate(model, val_loader, args, device)
        train_loss = float(np.mean(losses)) if losses else math.nan
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": time.time() - t0,
        }
        record.update(metrics)
        append_jsonl(metrics_path, record)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"anchor_mask={metrics.get('anchor_mask_mae', math.nan):.6f} "
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
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "num_real_hole_components": len(hole_component_library),
        "num_train_real_hole_components": len(train_component_library),
        "num_val_real_hole_components": len(val_component_library),
        "num_threshold_hole_components": int(component_library_summary.get("threshold_hole", 0)),
        "num_amp_speckle_components": int(component_library_summary.get("amp_speckle", 0)),
        "component_filter_summary": component_filter_summary,
        "component_split_summary": component_split_summary,
        "train_component_library_summary": train_component_library_summary,
        "val_component_library_summary": val_component_library_summary,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "method": args.method,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Done.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
