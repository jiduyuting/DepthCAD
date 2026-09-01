import argparse
import json
import math
import os
import time
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from infer_real_depth_flow import clip_prediction, move_condition_to_device, normalize_depth, predict_depth
from real_depth_masked_self_test import mae, make_block_mask
from real_raw9_masked_self_test import (
    RAW9_TRANSFORM_CHOICES,
    align_raw9_to_depth,
    build_real_hole_component_library,
    collect_pairs,
    depth_to_meters,
    filter_and_rebalance_component_library,
    make_artificial_mask,
    make_threshold_amp_depth_mask,
    make_condition,
    pair_dir_diagnostics,
    selected_raw_channels,
    stem_sort_key,
    summarize_component_library,
)
from train_depth_completion import gradient_l1, masked_mean, seed_everything
from train_depth_flow_restoration import flow_model_in_channels, predict_endpoint_norm_train
from train_depth_restoration import DepthRestorationCacheDataset, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune noisy_amp flow on paired real raw9/depth using masked self-supervision."
    )
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/real_raw9_flow_finetune_iq6_pilot",
    )
    parser.add_argument("--amplitude_mode", type=str, default="iq6", choices=["iq6", "raw_258"])
    parser.add_argument(
        "--raw9_transform",
        type=str,
        default="none",
        choices=RAW9_TRANSFORM_CHOICES,
        help=(
            "Spatial transform applied to raw9 before amplitude/mask/condition construction. "
            "Use flip_lr for the current Real raw9/depth alignment issue, or auto to estimate per sample."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--val_count", type=int, default=8)
    parser.add_argument(
        "--shuffle_split",
        action="store_true",
        default=False,
        help="Use a deterministic seed shuffle before taking --val_count validation pairs.",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file with {'train': [...], 'val': [...]} real sample stems. "
            "If provided, overrides --val_count so real-data finetuning can reuse an exact split."
        ),
    )
    parser.add_argument("--masks_per_sample", type=int, default=8)
    parser.add_argument("--val_masks_per_sample", type=int, default=3)
    parser.add_argument(
        "--mask_mode",
        type=str,
        default="block",
        choices=["block", "real_hole_shapes", "real_hole_speckle_shapes", "threshold_amp_depth"],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.10)
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
    parser.add_argument("--real_speckle_component_ratio", type=float, default=0.6)
    parser.add_argument(
        "--include_saturation_components",
        action="store_true",
        default=False,
        help="Add connected components from clipped raw9 channels to the real-hole/speckle shape library.",
    )
    parser.add_argument("--sat_component_channels", type=int, nargs="+", default=[2, 5, 8])
    parser.add_argument("--sat_component_clip_value", type=float, default=65535.0)
    parser.add_argument("--sat_component_clip_margin", type=float, default=1.0)
    parser.add_argument("--sat_component_min_area", type=int, default=16)
    parser.add_argument("--sat_component_max_area", type=int, default=30000)
    parser.add_argument("--sat_component_dilate", type=int, default=1)
    parser.add_argument(
        "--saturation_aug_prob",
        type=float,
        default=0.0,
        help="Probability of turning each artificial repair mask into a synthetic raw9 saturation/clipping case.",
    )
    parser.add_argument(
        "--val_saturation_aug_prob",
        type=float,
        default=None,
        help="Validation saturation augmentation probability. Defaults to --saturation_aug_prob.",
    )
    parser.add_argument("--saturation_aug_channels", type=int, nargs="+", default=[2, 5, 8])
    parser.add_argument("--saturation_aug_clip_value", type=float, default=65535.0)
    parser.add_argument(
        "--saturation_aug_jitter",
        type=float,
        default=0.0,
        help="Uniform downward jitter from clip value, in raw units, applied inside synthetic saturation masks.",
    )
    parser.add_argument("--saturation_aug_dilate", type=int, default=0)
    parser.add_argument(
        "--saturation_aug_depth_mode",
        type=str,
        default="zero",
        choices=["zero", "near", "far", "keep"],
        help="How synthetic saturation corrupts the depth observation inside the repair mask.",
    )
    parser.add_argument(
        "--saturation_aug_keep_amplitude",
        action="store_true",
        default=False,
        help="Do not zero raw/amplitude features inside synthetic saturation repair pixels.",
    )
    parser.add_argument(
        "--hole_amplitude_mode",
        type=str,
        default="zero",
        choices=["zero", "keep_artificial", "keep_all"],
        help="Controls amplitude zeroing in depth holes. Keep default for old checkpoints.",
    )
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
    parser.add_argument(
        "--threshold_depth_min",
        type=float,
        default=None,
        help="Depth below this value is treated as an observed sensor hole in threshold_amp_depth mode.",
    )
    parser.add_argument(
        "--threshold_depth_max",
        type=float,
        default=None,
        help="Depth above this value is treated as an observed sensor hole in threshold_amp_depth mode.",
    )
    parser.add_argument(
        "--threshold_amp_threshold",
        type=float,
        default=None,
        help="Fixed amplitude threshold for threshold_amp_depth mode. Overrides percentile threshold.",
    )
    parser.add_argument(
        "--threshold_amp_percentile",
        type=float,
        default=8.0,
        help="Per-sample amplitude percentile used when --threshold_amp_threshold is omitted.",
    )
    parser.add_argument("--threshold_mask_open", type=int, default=0)
    parser.add_argument("--threshold_mask_close", type=int, default=0)
    parser.add_argument("--threshold_mask_dilate", type=int, default=0)
    parser.add_argument("--threshold_mask_min_component_area", type=int, default=1)
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
    parser.add_argument("--post_clip_mode", type=str, default="valid_range",
                        choices=["none", "valid_range", "valid_percentile"])
    parser.add_argument("--post_clip_percentiles", type=float, nargs=2, default=[0.5, 99.5])

    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--mask_center_weight",
        type=float,
        default=0.0,
        help="Extra L1 weight for pixels near the center of artificial holes.",
    )
    parser.add_argument("--valid_loss_weight", type=float, default=0.05)
    parser.add_argument("--grad_loss_weight", type=float, default=0.05)
    parser.add_argument("--hole_grad_loss_weight", type=float, default=0.0)
    parser.add_argument("--boundary_grad_loss_weight", type=float, default=0.0)
    parser.add_argument("--boundary_l1_loss_weight", type=float, default=0.0)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--eval_component_area_threshold", type=int, default=700)
    parser.add_argument("--selection_metric", type=str, default="model_mask_mae")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=20)

    parser.add_argument(
        "--replay_weight",
        type=float,
        default=0.0,
        help="If >0, mix PBRT synthetic cache batches into real fine-tuning to reduce forgetting.",
    )
    parser.add_argument(
        "--replay_cache_dir",
        type=str,
        default=None,
        help="PBRT cache directory for replay. Defaults to the pretrained checkpoint cache_dir.",
    )
    parser.add_argument(
        "--replay_split_json",
        type=str,
        default=None,
        help="PBRT split.json for replay. Defaults to split.json next to the pretrained checkpoint when present.",
    )
    parser.add_argument(
        "--replay_sample_list",
        type=str,
        default=None,
        help="Optional text file of PBRT cache paths. Overrides replay split/cache discovery.",
    )
    parser.add_argument(
        "--replay_split",
        type=str,
        default="train",
        choices=["train", "val", "all"],
        help="Subset from --replay_split_json to use for replay.",
    )
    parser.add_argument("--replay_batch_size", type=int, default=0)
    parser.add_argument("--replay_num_workers", type=int, default=-1)
    parser.add_argument(
        "--replay_max_samples",
        type=int,
        default=0,
        help="If >0, deterministically subsample this many PBRT replay samples.",
    )
    return parser.parse_args()


class RealRaw9MaskedDataset(Dataset):
    def __init__(
        self,
        pairs,
        ckpt_args,
        args,
        masks_per_sample,
        fixed_masks=False,
        hole_component_library=None,
        saturation_aug_prob=None,
    ):
        self.pairs = list(pairs)
        self.ckpt_args = ckpt_args
        self.args = args
        self.masks_per_sample = int(masks_per_sample)
        self.fixed_masks = bool(fixed_masks)
        self.hole_component_library = hole_component_library or []
        self.saturation_aug_prob = (
            float(args.saturation_aug_prob)
            if saturation_aug_prob is None
            else float(saturation_aug_prob)
        )
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.pairs) * self.masks_per_sample

    def _make_mask(self, reliable, rng, source_stem):
        for _ in range(max(1, int(self.args.max_mask_retries))):
            mask = make_artificial_mask(
                reliable,
                rng,
                self.args,
                component_library=self.hole_component_library,
                source_stem=source_stem,
            )
            if mask.any():
                return mask
        return make_block_mask(
            reliable,
            rng,
            self.args.mask_ratio,
            self.args.min_block_size,
            self.args.max_block_size,
        )

    def _apply_saturation_augmentation(self, raw9, corrupted, repair_mask, reliable, rng):
        sat_mask = np.zeros_like(repair_mask, dtype=bool)
        if float(self.saturation_aug_prob) <= 0.0:
            return raw9, corrupted, repair_mask, sat_mask, False
        if float(rng.random()) >= float(self.saturation_aug_prob):
            return raw9, corrupted, repair_mask, sat_mask, False

        sat_mask = repair_mask.copy()
        if int(self.args.saturation_aug_dilate) > 0:
            sat_mask = dilate_mask_np(sat_mask, int(self.args.saturation_aug_dilate))
        sat_mask &= reliable
        if not sat_mask.any():
            return raw9, corrupted, repair_mask, sat_mask, False

        repair_mask = repair_mask | sat_mask
        raw9 = raw9.copy()
        channel_ids = selected_raw_channels(raw9, self.args.saturation_aug_channels)
        clip_value = float(self.args.saturation_aug_clip_value)
        jitter = max(0.0, float(self.args.saturation_aug_jitter))
        for channel in channel_ids:
            values = np.full(int(sat_mask.sum()), clip_value, dtype=np.float32)
            if jitter > 0.0:
                values -= rng.uniform(0.0, jitter, size=values.shape).astype(np.float32)
            raw9[channel, sat_mask] = values

        corrupted = corrupted.copy()
        mode = str(self.args.saturation_aug_depth_mode)
        if mode == "zero":
            corrupted[sat_mask] = 0.0
        elif mode == "near":
            corrupted[sat_mask] = min(float(self.args.valid_min_depth) * 0.5, float(self.args.hole_depth_threshold) * 0.5)
        elif mode == "far":
            corrupted[sat_mask] = float(self.args.valid_max_depth) + 1.0
        elif mode == "keep":
            pass
        else:
            raise ValueError(f"Unknown saturation_aug_depth_mode: {mode}")
        return raw9, corrupted, repair_mask, sat_mask, True

    def __getitem__(self, index):
        pair_index = index // self.masks_per_sample
        repeat = index % self.masks_per_sample
        stem, raw_path, depth_path = self.pairs[pair_index]
        raw9 = np.load(raw_path).astype(np.float32)
        clean = depth_to_meters(np.load(depth_path), self.args.depth_unit)
        if raw9.shape != (9,) + clean.shape:
            raise ValueError(f"Shape mismatch for {stem}: raw {raw9.shape}, depth {clean.shape}")
        reliable = (
            np.isfinite(clean)
            & (clean > float(self.args.hole_depth_threshold))
            & (clean >= float(self.args.valid_min_depth))
            & (clean <= float(self.args.valid_max_depth))
        )
        if reliable.sum() == 0:
            raise ValueError(f"No reliable pixels for {stem}")
        raw9, _ = align_raw9_to_depth(raw9, clean, reliable, self.args)

        epoch_offset = 0 if self.fixed_masks else self.epoch * 1000003
        rng = np.random.default_rng(int(self.args.seed) + epoch_offset + pair_index * 1009 + repeat)
        corrupted = clean.copy()
        threshold_hole = np.zeros_like(reliable, dtype=bool)
        if self.args.mask_mode == "threshold_amp_depth":
            threshold_hole, artificial_mask, _ = make_threshold_amp_depth_mask(
                clean,
                raw9,
                reliable,
                self.args,
            )
            corrupted[threshold_hole] = 0.0
        else:
            artificial_mask = self._make_mask(reliable, rng, stem)
            corrupted[artificial_mask] = 0.0
        raw9, corrupted, artificial_mask, saturation_mask, saturation_applied = self._apply_saturation_augmentation(
            raw9,
            corrupted,
            artificial_mask,
            reliable,
            rng,
        )
        mask_distance = cv2.distanceTransform(artificial_mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        max_distance = float(mask_distance.max())
        if max_distance > 0:
            mask_distance = mask_distance / max_distance

        preserve_amplitude_mask = None
        if bool(self.args.saturation_aug_keep_amplitude) and saturation_applied:
            preserve_amplitude_mask = saturation_mask
        condition = make_condition(
            corrupted,
            raw9,
            artificial_mask,
            reliable,
            self.ckpt_args,
            self.args,
            preserve_amplitude_mask=preserve_amplitude_mask,
        )
        target_norm = normalize_depth(
            clean,
            condition["center_value"],
            condition["scale_value"],
            float(self.ckpt_args.get("clip_norm_depth", 8.0)),
        )
        return {
            "x": condition["x"][0],
            "anchor_norm": condition["anchor_norm"][0],
            "target_norm": torch.from_numpy(target_norm[None]),
            "hole_mask": torch.from_numpy(condition["hole"][None].astype(np.bool_)),
            "valid_mask": torch.from_numpy(reliable[None].astype(np.bool_)),
            "artificial_mask": torch.from_numpy(artificial_mask[None].astype(np.bool_)),
            "mask_distance": torch.from_numpy(mask_distance[None].astype(np.float32)),
            "saturation_mask": torch.from_numpy(saturation_mask[None].astype(np.bool_)),
            "threshold_hole_mask": torch.from_numpy(threshold_hole[None].astype(np.bool_)),
            "saturation_aug_applied": torch.tensor(bool(saturation_applied)),
            "depth_anchor": torch.from_numpy(condition["anchor"][None].astype(np.float32)),
            "depth_noisy": torch.from_numpy(corrupted[None].astype(np.float32)),
            "gt_depth": torch.from_numpy(clean[None].astype(np.float32)),
            "center": torch.tensor(condition["center_value"], dtype=torch.float32),
            "scale": torch.tensor(condition["scale_value"], dtype=torch.float32),
            "sample_name": f"{stem}_r{repeat:02d}",
            "path": depth_path,
        }


def build_model(ckpt, ckpt_args, device):
    condition_channels = 4 + int(bool(ckpt_args.get("include_hole_distance", False))) + 4
    in_channels = flow_model_in_channels(condition_channels, int(ckpt_args.get("time_channels", 16)))
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
    return model


def add_checkpoint_args(args, ckpt_args):
    for key, value in ckpt_args.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args.checkpoint = args.pretrained_checkpoint
    args.input_mode = "noisy_amp"
    args.method = "real_raw9_masked_self_supervised_finetune"
    args.eval_sampling_mode = ckpt_args.get("eval_sampling_mode", "endpoint")
    args.sample_steps = int(ckpt_args.get("sample_steps", 8))
    args.time_channels = int(ckpt_args.get("time_channels", 16))
    args.max_velocity_norm = float(ckpt_args.get("max_velocity_norm", 4.0))
    args.clip_norm_depth = float(ckpt_args.get("clip_norm_depth", 8.0))
    args.velocity_scale = float(ckpt_args.get("velocity_scale", 1.0))
    args.backbone = ckpt_args.get("backbone", "resunet")
    args.base_channels = int(ckpt_args.get("base_channels", 32))
    args.res_blocks = int(ckpt_args.get("res_blocks", 2))
    args.transformer_layers = int(ckpt_args.get("transformer_layers", 2))
    args.transformer_heads = int(ckpt_args.get("transformer_heads", 8))
    args.transformer_mlp_ratio = float(ckpt_args.get("transformer_mlp_ratio", 4.0))
    args.transformer_pool = int(ckpt_args.get("transformer_pool", 2))


def split_real_pairs(pairs, args):
    if args.split_json:
        with open(args.split_json, "r") as f:
            split = json.load(f)
        train_stems = [str(stem) for stem in split.get("train", [])]
        val_stems = [str(stem) for stem in split.get("val", [])]
        if not train_stems or not val_stems:
            raise ValueError(f"{args.split_json} must contain non-empty 'train' and 'val' lists.")

        pair_by_stem = {pair[0]: pair for pair in pairs}

        missing = [stem for stem in train_stems + val_stems if stem not in pair_by_stem]
        if missing:
            raise ValueError(
                f"{args.split_json} contains stems not found under raw/depth: {sorted(set(missing), key=stem_sort_key)}"
            )

        overlap = sorted(set(train_stems) & set(val_stems), key=stem_sort_key)
        if overlap:
            raise ValueError(f"{args.split_json} has overlapping train/val stems: {overlap}")

        train_pairs = [pair_by_stem[stem] for stem in train_stems]
        val_pairs = [pair_by_stem[stem] for stem in val_stems]
        return train_pairs, val_pairs

    if len(pairs) <= args.val_count:
        raise ValueError(f"Need more pairs than val_count={args.val_count}, got {len(pairs)}")
    if bool(getattr(args, "shuffle_split", False)):
        rng = np.random.default_rng(int(args.seed))
        indices = np.arange(len(pairs))
        rng.shuffle(indices)
        val_indices = set(int(idx) for idx in indices[: int(args.val_count)])
        train_pairs = [pair for idx, pair in enumerate(pairs) if idx not in val_indices]
        val_pairs = [pair for idx, pair in enumerate(pairs) if idx in val_indices]
        return train_pairs, val_pairs
    train_pairs = pairs[: -args.val_count]
    val_pairs = pairs[-args.val_count :]
    return train_pairs, val_pairs


def read_path_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def collect_replay_paths(args, ckpt_args):
    if float(args.replay_weight) <= 0.0:
        return [], {"enabled": False}

    source = None
    paths = []
    split_json = args.replay_split_json
    if args.replay_sample_list:
        paths = read_path_list(args.replay_sample_list)
        source = "sample_list"
    else:
        if not split_json:
            candidate = os.path.join(os.path.dirname(os.path.abspath(args.pretrained_checkpoint)), "split.json")
            if os.path.exists(candidate):
                split_json = candidate
        if split_json:
            with open(split_json, "r") as f:
                split = json.load(f)
            if args.replay_split == "all":
                paths = list(split.get("train", [])) + list(split.get("val", []))
            else:
                paths = list(split.get(args.replay_split, []))
            source = "split_json"
        else:
            cache_dir = args.replay_cache_dir or ckpt_args.get("cache_dir")
            if not cache_dir:
                raise ValueError(
                    "--replay_weight > 0 requires --replay_cache_dir, --replay_sample_list, "
                    "--replay_split_json, or a pretrained checkpoint with cache_dir."
                )
            paths = sorted(glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True))
            source = "cache_dir"

    paths = [str(path) for path in paths]
    missing = [path for path in paths if not os.path.exists(path)]
    existing = [path for path in paths if os.path.exists(path)]
    if not existing:
        raise FileNotFoundError(
            "No existing PBRT replay cache samples found. "
            f"source={source} split_json={split_json!r} sample_list={args.replay_sample_list!r} "
            f"cache_dir={(args.replay_cache_dir or ckpt_args.get('cache_dir'))!r}"
        )

    max_samples = int(args.replay_max_samples)
    if max_samples > 0 and len(existing) > max_samples:
        rng = np.random.default_rng(int(args.seed) + 9409)
        selected = sorted(rng.choice(len(existing), size=max_samples, replace=False).tolist())
        existing = [existing[index] for index in selected]

    return existing, {
        "enabled": True,
        "source": source,
        "split_json": split_json,
        "sample_list": args.replay_sample_list,
        "cache_dir": args.replay_cache_dir or ckpt_args.get("cache_dir"),
        "split": args.replay_split,
        "requested_count": len(paths),
        "missing_count": len(missing),
        "missing_examples": missing[:8],
        "used_count": len(existing),
        "max_samples": max_samples,
        "weight": float(args.replay_weight),
    }


def replay_dataset_kwargs(ckpt_args):
    return {
        "input_mode": ckpt_args.get("input_mode", "noisy_amp"),
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


def next_loader_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def weighted_masked_mean(x, mask, weight, eps=1e-6):
    weight = weight.to(dtype=x.dtype)
    mask_weight = mask.to(dtype=x.dtype) * weight
    denom = mask_weight.sum().clamp_min(eps)
    return (x * mask_weight).sum() / denom


def dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    dilated = F.max_pool2d(mask.to(dtype=torch.float32), kernel_size, stride=1, padding=radius)
    return dilated > 0


def compute_loss(model, batch, args, mask_key="artificial_mask"):
    pred = predict_endpoint_norm_train(model, batch, args)
    target = batch["target_norm"]
    valid = batch["valid_mask"] & torch.isfinite(target) & torch.isfinite(pred)
    repair_source = batch[mask_key] if mask_key in batch else batch["hole_mask"]
    repair_source = repair_source.to(dtype=torch.bool)
    mask = repair_source & valid
    unmasked = valid & (~repair_source)
    err = torch.abs(pred - target)

    if float(args.mask_center_weight) > 0 and "mask_distance" in batch:
        mask_weight = 1.0 + float(args.mask_center_weight) * batch["mask_distance"].to(device=pred.device)
        mask_loss = weighted_masked_mean(err, mask, mask_weight)
    else:
        mask_loss = masked_mean(err, mask)
    valid_loss = masked_mean(err, unmasked)
    grad_loss = gradient_l1(pred, target, valid, valid)
    hole_grad_loss = gradient_l1(pred, target, valid, mask)

    boundary = dilate_mask(mask, int(args.boundary_width)) & valid
    boundary_ring = boundary & (~mask)
    boundary_grad_loss = gradient_l1(pred, target, valid, boundary)
    boundary_l1_loss = masked_mean(err, boundary_ring)
    total = (
        float(args.mask_loss_weight) * mask_loss
        + float(args.valid_loss_weight) * valid_loss
        + float(args.grad_loss_weight) * grad_loss
        + float(args.hole_grad_loss_weight) * hole_grad_loss
        + float(args.boundary_grad_loss_weight) * boundary_grad_loss
        + float(args.boundary_l1_loss_weight) * boundary_l1_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "mask": float(mask_loss.detach().cpu()),
        "valid": float(valid_loss.detach().cpu()),
        "grad": float(grad_loss.detach().cpu()),
        "hole_grad": float(hole_grad_loss.detach().cpu()),
        "boundary_grad": float(boundary_grad_loss.detach().cpu()),
        "boundary_l1": float(boundary_l1_loss.detach().cpu()),
    }


def dilate_mask_np(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def component_area_masks(mask, area_threshold):
    mask = np.asarray(mask, dtype=bool)
    small = np.zeros_like(mask, dtype=bool)
    large = np.zeros_like(mask, dtype=bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        if area <= int(area_threshold):
            small |= component
        else:
            large |= component
    return small, large


def aggregate_training_metrics(rows):
    sums = {}
    counts = {}
    for row in rows:
        for key, value in row.items():
            if not key.endswith("_mae"):
                continue
            count = row.get(f"{key}_count", 0)
            if value is None or count == 0:
                continue
            sums[key] = sums.get(key, 0.0) + float(value) * int(count)
            counts[key] = counts.get(key, 0) + int(count)

    out = {"num_cases": len(rows)}
    for key, total in sums.items():
        out[key] = total / max(counts[key], 1)
        out[f"{key}_count"] = counts[key]

    for region in ["mask", "global", "small_mask", "large_mask", "boundary"]:
        anchor_key = f"anchor_{region}_mae"
        model_key = f"model_{region}_mae"
        if (
            out.get(anchor_key) is not None
            and out.get(model_key) is not None
            and abs(float(out[anchor_key])) > 1e-12
        ):
            out[f"{region}_improve_vs_anchor"] = (
                out[anchor_key] - out[model_key]
            ) / max(out[anchor_key], 1e-12)
    return out


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_cuda(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


@torch.no_grad()
def evaluate(model, dataloader, args, device):
    model.eval()
    rows = []
    for batch in dataloader:
        batch = move_condition_to_device(batch, device)
        pred = predict_depth(model, batch, vars(args), args.eval_sampling_mode, args.sample_steps)
        scale = batch["scale"].view(-1, 1, 1, 1)
        center = batch["center"].view(-1, 1, 1, 1)
        anchor = batch["depth_anchor"]
        clean = batch["gt_depth"]
        corrupted = batch["depth_noisy"]
        artificial = batch["artificial_mask"]
        valid = batch["valid_mask"]
        condition_hole = batch["hole_mask"]

        bs = pred.shape[0]
        for i in range(bs):
            pred_np = pred[i, 0].detach().cpu().numpy().astype(np.float32)
            clean_np = clean[i, 0].detach().cpu().numpy().astype(np.float32)
            valid_np = valid[i, 0].detach().cpu().numpy().astype(bool)
            pred_np, _ = clip_prediction(pred_np, clean_np, valid_np, args)
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
            for key, prediction, mask in [
                ("anchor_mask_mae", anchor_np, artificial_np),
                ("model_mask_mae", pred_np, artificial_np),
                ("hole_only_mask_mae", hole_only, artificial_np),
                ("anchor_small_mask_mae", anchor_np, small_mask),
                ("model_small_mask_mae", pred_np, small_mask),
                ("anchor_large_mask_mae", anchor_np, large_mask),
                ("model_large_mask_mae", pred_np, large_mask),
                ("anchor_boundary_mae", anchor_np, boundary_mask),
                ("model_boundary_mae", pred_np, boundary_mask),
                ("model_unmasked_mae", pred_np, unmasked),
                ("hole_only_unmasked_mae", hole_only, unmasked),
                ("anchor_global_mae", anchor_np, valid_np),
                ("model_global_mae", pred_np, valid_np),
                ("hole_only_global_mae", hole_only, valid_np),
            ]:
                value, count = mae(prediction, clean_np, mask)
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
    if not pairs:
        diagnostics = pair_dir_diagnostics(args.raw_dir, args.depth_dir)
        raise FileNotFoundError(
            "No paired raw/depth .npy files found. "
            f"Diagnostics: {json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)}"
        )
    train_pairs, val_pairs = split_real_pairs(pairs, args)
    replay_paths, replay_info = collect_replay_paths(args, ckpt_args)

    hole_component_library = []
    component_filter_summary = {}
    component_library_summary = {"total": 0, "threshold_hole": 0, "amp_speckle": 0, "raw_saturation": 0}
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
                "shuffle_split": bool(args.shuffle_split),
                "seed": int(args.seed),
                "raw9_transform": args.raw9_transform,
                "component_library_summary": component_library_summary,
                "component_filter_summary": component_filter_summary,
                "replay": replay_info,
                "replay_paths": replay_paths,
            },
            f,
            indent=2,
        )
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    val_saturation_aug_prob = (
        float(args.saturation_aug_prob)
        if args.val_saturation_aug_prob is None
        else float(args.val_saturation_aug_prob)
    )
    train_dataset = RealRaw9MaskedDataset(
        train_pairs,
        ckpt_args,
        args,
        masks_per_sample=args.masks_per_sample,
        fixed_masks=False,
        hole_component_library=hole_component_library,
        saturation_aug_prob=float(args.saturation_aug_prob),
    )
    val_dataset = RealRaw9MaskedDataset(
        val_pairs,
        ckpt_args,
        args,
        masks_per_sample=args.val_masks_per_sample,
        fixed_masks=True,
        hole_component_library=hole_component_library,
        saturation_aug_prob=val_saturation_aug_prob,
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
    replay_loader = None
    if replay_paths:
        replay_dataset = DepthRestorationCacheDataset(
            replay_paths,
            **replay_dataset_kwargs(ckpt_args),
        )
        replay_batch_size = int(args.replay_batch_size) if int(args.replay_batch_size) > 0 else int(args.batch_size)
        replay_num_workers = int(args.num_workers) if int(args.replay_num_workers) < 0 else int(args.replay_num_workers)
        replay_loader = DataLoader(
            replay_dataset,
            batch_size=replay_batch_size,
            shuffle=True,
            num_workers=replay_num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = build_model(ckpt, ckpt_args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(enabled=(args.amp and device.type == "cuda"))

    print(f"Pairs: {len(pairs)} train={len(train_pairs)} val={len(val_pairs)}")
    if replay_loader is not None:
        print(
            "PBRT replay: "
            f"count={len(replay_paths)} "
            f"split={replay_info.get('split')} "
            f"source={replay_info.get('source')} "
            f"weight={float(args.replay_weight):.3f} "
            f"batch_size={replay_loader.batch_size}"
        )
    else:
        print("PBRT replay: disabled")
    print(
        "Real component library="
        f"{component_library_summary.get('total', 0)} "
        f"(threshold={component_library_summary.get('threshold_hole', 0)}, "
        f"amp_speckle={component_library_summary.get('amp_speckle', 0)}, "
        f"raw_saturation={component_library_summary.get('raw_saturation', 0)})"
    )
    if component_filter_summary:
        print(f"Component filtering={json.dumps(component_filter_summary, sort_keys=True)}")
    print(f"Device: {device}")
    print(f"Amplitude mode: {args.amplitude_mode}")
    print(f"Raw9 transform: {args.raw9_transform}")
    print(f"Depth unit: {args.depth_unit}")
    print(
        "Saturation aug: "
        f"train_prob={float(args.saturation_aug_prob):.3f} "
        f"val_prob={val_saturation_aug_prob:.3f} "
        f"channels={args.saturation_aug_channels} "
        f"keep_amp={bool(args.saturation_aug_keep_amplitude)}"
    )
    print(f"Pretrained: {args.pretrained_checkpoint}")
    print(f"Output: {args.output_dir}")

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    best_score = float("inf")
    best_epoch = -1

    initial_metrics, _ = evaluate(model, val_loader, args, device)
    print(f"Initial val: {json.dumps(initial_metrics, sort_keys=True)}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        losses = []
        real_losses = []
        replay_losses = []
        replay_iter = iter(replay_loader) if replay_loader is not None else None
        for step, batch in enumerate(train_loader, start=1):
            batch = move_condition_to_device(batch, device)
            replay_batch = None
            if replay_loader is not None:
                replay_batch, replay_iter = next_loader_batch(replay_loader, replay_iter)
                replay_batch = move_condition_to_device(replay_batch, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(enabled=(args.amp and device.type == "cuda")):
                real_loss, parts = compute_loss(model, batch, args, mask_key="artificial_mask")
                loss = real_loss
                replay_parts = None
                if replay_batch is not None:
                    replay_loss, replay_parts = compute_loss(model, replay_batch, args, mask_key="hole_mask")
                    loss = real_loss + float(args.replay_weight) * replay_loss
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            combined_loss = float(loss.detach().cpu())
            losses.append(combined_loss)
            real_losses.append(parts["loss"])
            if replay_parts is not None:
                replay_losses.append(replay_parts["loss"])
            if args.log_every and (step == 1 or step % args.log_every == 0):
                replay_text = ""
                if replay_parts is not None:
                    replay_text = (
                        f" replay={replay_parts['loss']:.5f}"
                        f" replay_mask={replay_parts['mask']:.5f}"
                    )
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(train_loader)} "
                    f"loss={combined_loss:.5f} real={parts['loss']:.5f}{replay_text} "
                    f"mask={parts['mask']:.5f} "
                    f"valid={parts['valid']:.5f} grad={parts['grad']:.5f} "
                    f"hole_grad={parts['hole_grad']:.5f} "
                    f"boundary_grad={parts['boundary_grad']:.5f}"
                )

        metrics, _ = evaluate(model, val_loader, args, device)
        train_loss = float(np.mean(losses)) if losses else math.nan
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_real_loss": float(np.mean(real_losses)) if real_losses else math.nan,
            "train_replay_loss": float(np.mean(replay_losses)) if replay_losses else math.nan,
            "replay_weight": float(args.replay_weight),
            "seconds": time.time() - t0,
        }
        record.update(metrics)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

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
        "output_dir": args.output_dir,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "split_json": args.split_json,
        "shuffle_split": bool(args.shuffle_split),
        "amplitude_mode": args.amplitude_mode,
        "raw9_transform": args.raw9_transform,
        "mask_mode": args.mask_mode,
        "depth_unit": args.depth_unit,
        "valid_min_depth": float(args.valid_min_depth),
        "valid_max_depth": float(args.valid_max_depth),
        "threshold_depth_min": args.threshold_depth_min,
        "threshold_depth_max": args.threshold_depth_max,
        "threshold_amp_threshold": args.threshold_amp_threshold,
        "threshold_amp_percentile": float(args.threshold_amp_percentile),
        "threshold_mask_open": int(args.threshold_mask_open),
        "threshold_mask_close": int(args.threshold_mask_close),
        "threshold_mask_dilate": int(args.threshold_mask_dilate),
        "threshold_mask_min_component_area": int(args.threshold_mask_min_component_area),
        "mask_loss_weight": float(args.mask_loss_weight),
        "mask_center_weight": float(args.mask_center_weight),
        "valid_loss_weight": float(args.valid_loss_weight),
        "grad_loss_weight": float(args.grad_loss_weight),
        "hole_grad_loss_weight": float(args.hole_grad_loss_weight),
        "boundary_grad_loss_weight": float(args.boundary_grad_loss_weight),
        "boundary_l1_loss_weight": float(args.boundary_l1_loss_weight),
        "boundary_width": int(args.boundary_width),
        "num_threshold_hole_components": int(component_library_summary.get("threshold_hole", 0)),
        "num_amp_speckle_components": int(component_library_summary.get("amp_speckle", 0)),
        "num_raw_saturation_components": int(component_library_summary.get("raw_saturation", 0)),
        "include_saturation_components": bool(args.include_saturation_components),
        "saturation_aug_prob": float(args.saturation_aug_prob),
        "val_saturation_aug_prob": float(val_saturation_aug_prob),
        "saturation_aug_channels": [int(channel) for channel in args.saturation_aug_channels],
        "saturation_aug_keep_amplitude": bool(args.saturation_aug_keep_amplitude),
        "hole_amplitude_mode": args.hole_amplitude_mode,
        "replay_enabled": bool(replay_paths),
        "replay_weight": float(args.replay_weight),
        "replay_count": int(len(replay_paths)),
        "replay_split": replay_info.get("split"),
        "replay_source": replay_info.get("source"),
        "replay_split_json": replay_info.get("split_json"),
        "replay_cache_dir": replay_info.get("cache_dir"),
        "replay_max_samples": int(args.replay_max_samples),
        "pretrained_checkpoint": args.pretrained_checkpoint,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("Done.")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
