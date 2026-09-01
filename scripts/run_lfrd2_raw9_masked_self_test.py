import argparse
import json
import os
import sys
import types
from glob import glob
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from inference_depth_postprocess import opencv_depth_inpaint
from real_raw9_masked_self_test import (
    build_real_hole_component_library,
    filter_and_rebalance_component_library,
    make_real_hole_shape_mask,
)


LFRD2_ROOT = "/data/pre_student/GJ/LFRD2"


def install_mmcv_stub():
    """Allow importing LFRD2 on environments without mmcv.

    The adapter patches the only code paths that would call the MMCV operator.
    This stub exists only so that LFRD2's module import succeeds.
    """
    if "mmcv.ops.modulated_deform_conv" in sys.modules:
        return

    mmcv_module = types.ModuleType("mmcv")
    ops_module = types.ModuleType("mmcv.ops")
    deform_module = types.ModuleType("mmcv.ops.modulated_deform_conv")

    class _UnavailableModulatedDeformConv2dFunction:
        @staticmethod
        def apply(*_args, **_kwargs):
            raise RuntimeError(
                "MMCV modulated deform conv was called. The LFRD2 adapter should "
                "patch this path with a torch grid_sample fallback."
            )

    deform_module.ModulatedDeformConv2dFunction = _UnavailableModulatedDeformConv2dFunction
    ops_module.modulated_deform_conv = deform_module
    mmcv_module.ops = ops_module
    sys.modules.setdefault("mmcv", mmcv_module)
    sys.modules.setdefault("mmcv.ops", ops_module)
    sys.modules.setdefault("mmcv.ops.modulated_deform_conv", deform_module)


def import_lfrd2():
    install_mmcv_stub()
    if LFRD2_ROOT not in sys.path:
        sys.path.insert(0, LFRD2_ROOT)
    from model.cplx import FracDiff, ImpInt
    from model.ImplicitNeural import SVI

    patch_lfrd2_cpu_fallback(SVI, ImpInt)
    return FracDiff


def sample_1x1(feature, offset):
    b, _c, h, w = feature.shape
    oy = offset[:, 0]
    ox = offset[:, 1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=feature.device),
        torch.arange(w, device=feature.device),
        indexing="ij",
    )
    yy = yy.float().unsqueeze(0) + oy
    xx = xx.float().unsqueeze(0) + ox
    grid = torch.stack(
        [
            2.0 * xx / max(w - 1, 1) - 1.0,
            2.0 * yy / max(h - 1, 1) - 1.0,
        ],
        dim=-1,
    )
    return F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def patch_lfrd2_cpu_fallback(SVI, ImpInt):
    def get_subpixel_values(self, image, offset):
        b, _c, h, w = image.size()
        offset = offset.view(b, 9, 2, h, w)
        values = [sample_1x1(image, offset[:, k]).squeeze(1) for k in range(9)]
        return torch.stack(values, dim=1)

    def get_offset_affinity(self, guidance, confidence=None, rgb=None):
        del rgb
        b, _channels, h, w = guidance.shape
        offset_aff = self.conv_offset_aff(guidance)
        o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

        offset = torch.cat((o1, o2), dim=1).view(b, self.num, 2, h, w)
        offset_parts = list(torch.chunk(offset, self.num, dim=1))
        offset_parts.insert(
            self.idx_ref,
            torch.zeros((b, 1, 2, h, w), dtype=offset.dtype, device=offset.device),
        )
        offset = torch.cat(offset_parts, dim=1).view(b, -1, h, w)

        if self.affinity == "TC":
            aff = torch.tanh(aff) / self.aff_scale_const
        elif self.affinity == "TGASS":
            aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
        elif self.affinity in ["AS", "ASS"]:
            pass
        else:
            raise NotImplementedError(self.affinity)

        if self.args.conf_prop:
            if confidence is None:
                raise ValueError("confidence is required when conf_prop=True")
            sampled_conf = []
            for off in torch.chunk(offset, self.num + 1, dim=1):
                sampled_conf.append(sample_1x1(confidence, off).squeeze(1))
            sampled_conf.pop(self.idx_ref)
            aff = aff * torch.stack(sampled_conf, dim=1).contiguous()

        aff_abs_sum = torch.sum(torch.abs(aff), dim=1, keepdim=True) + 1e-4
        if self.affinity in ["ASS", "TGASS"]:
            aff_abs_sum = torch.where(
                aff_abs_sum < 1.0,
                torch.ones_like(aff_abs_sum),
                aff_abs_sum,
            )
        if self.affinity in ["AS", "ASS", "TGASS"]:
            aff = aff / aff_abs_sum

        aff_ref = -torch.sum(aff, dim=1, keepdim=True)
        aff_parts = list(torch.chunk(aff, self.num, dim=1))
        aff_parts.insert(self.idx_ref, aff_ref)
        aff = torch.cat(aff_parts, dim=1)
        return offset, aff

    SVI.get_subpixel_values = get_subpixel_values
    ImpInt._get_offset_affinity = get_offset_affinity


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an LFRD2 FracDiff proxy baseline on DepthCAD real raw9/depth "
            "samples for either masked self-test or observed-hole inference."
        )
    )
    parser.add_argument("--raw_dir", default="raw")
    parser.add_argument("--depth_dir", default="depth")
    parser.add_argument("--output_dir", default="output/lfrd2_raw9_masked_self_test")
    parser.add_argument("--lfrd2_root", default=LFRD2_ROOT)
    parser.add_argument("--checkpoint_net", default=None)
    parser.add_argument("--checkpoint_domain", choices=["real", "synthetic"], default="real")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=1,
        help="Limit PyTorch CPU threads. This keeps CPU fallback runs predictable.",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["masked_self_test", "observed_holes"],
        default="masked_self_test",
        help="Masked self-test reports same-mask MAE. Observed-holes fills existing depth holes without GT metrics.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--mask_ratio", type=float, default=0.10)
    parser.add_argument("--num_masks_per_sample", type=int, default=1)
    parser.add_argument("--min_block_size", type=int, default=12)
    parser.add_argument("--max_block_size", type=int, default=72)
    parser.add_argument(
        "--mask_mode",
        choices=["block", "real_hole_shapes"],
        default="block",
        help="Use random blocks or real-hole-shaped components from observed depth holes.",
    )
    parser.add_argument("--real_hole_min_area", type=int, default=24)
    parser.add_argument("--real_hole_max_area", type=int, default=0)
    parser.add_argument("--real_hole_min_overlap", type=float, default=0.6)
    parser.add_argument("--real_hole_max_components", type=int, default=8)
    parser.add_argument("--real_hole_max_attempts", type=int, default=512)
    parser.add_argument("--hole_depth_threshold", type=float, default=1.0)
    parser.add_argument("--valid_min_depth", type=float, default=1.0)
    parser.add_argument("--valid_max_depth", type=float, default=9.9)
    parser.add_argument("--anchor_inpaint_radius", type=int, default=15)
    parser.add_argument("--input_mode", choices=["masked", "anchor"], default="masked")
    parser.add_argument("--raw9_transform", choices=["none", "flip_lr", "flip_ud", "rot180"], default="none")
    parser.add_argument(
        "--crop_mode",
        choices=["fixed", "center", "max_hole_window"],
        default="fixed",
        help="Crop mode before sending data into the 180x240 LFRD2 proxy model.",
    )
    parser.add_argument("--crop_y", type=int, default=30)
    parser.add_argument("--crop_x", type=int, default=40)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--clip_output", action="store_true", default=True)
    parser.add_argument("--no_clip_output", action="store_false", dest="clip_output")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_max_samples", type=int, default=24)
    return parser.parse_args()


def natural_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def paired_paths(raw_dir, depth_dir):
    raw = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(raw_dir, "*.npy"))}
    depth = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(depth_dir, "*.npy"))}
    stems = sorted(set(raw) & set(depth), key=lambda s: natural_key(s))
    return [(stem, raw[stem], depth[stem]) for stem in stems]


def crop_hw(array, y, x, h, w):
    if array.ndim == 2:
        return array[y : y + h, x : x + w]
    if array.ndim == 3:
        return array[:, y : y + h, x : x + w]
    raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


def resolve_crop_bounds(shape_hw, args):
    height_in, width_in = int(shape_hw[0]), int(shape_hw[1])
    crop_h = min(int(args.height), height_in)
    crop_w = min(int(args.width), width_in)
    if args.crop_mode == "center":
        y = max(0, (height_in - crop_h) // 2)
        x = max(0, (width_in - crop_w) // 2)
    elif args.crop_mode == "max_hole_window":
        y = max(0, (height_in - crop_h) // 2)
        x = max(0, (width_in - crop_w) // 2)
    else:
        y = max(0, min(int(args.crop_y), max(0, height_in - crop_h)))
        x = max(0, min(int(args.crop_x), max(0, width_in - crop_w)))
    return y, x, crop_h, crop_w


def best_hole_window(hole_mask, crop_h, crop_w):
    hole_mask = np.asarray(hole_mask, dtype=np.uint8)
    height_in, width_in = hole_mask.shape
    crop_h = min(int(crop_h), height_in)
    crop_w = min(int(crop_w), width_in)
    if crop_h <= 0 or crop_w <= 0:
        return 0, 0, crop_h, crop_w
    if not hole_mask.any():
        y = max(0, (height_in - crop_h) // 2)
        x = max(0, (width_in - crop_w) // 2)
        return y, x, crop_h, crop_w

    integral = np.pad(hole_mask, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    best_sum = -1
    best_y = 0
    best_x = 0
    for y in range(height_in - crop_h + 1):
        y2 = y + crop_h
        x = np.arange(width_in - crop_w + 1, dtype=np.int32)
        x2 = x + crop_w
        sums = integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
        idx = int(np.argmax(sums))
        value = int(sums[idx])
        if value > best_sum:
            best_sum = value
            best_y = y
            best_x = idx
    return best_y, best_x, crop_h, crop_w


def ensure_raw9_chw(raw9, depth_shape):
    raw9 = np.asarray(raw9, dtype=np.float32)
    depth_shape = tuple(int(v) for v in depth_shape)
    if raw9.shape == (9,) + depth_shape:
        return raw9
    if raw9.shape == depth_shape + (9,):
        return np.transpose(raw9, (2, 0, 1)).copy()
    raise ValueError(f"Expected raw9 to match depth shape {depth_shape}, got {raw9.shape}")


def paste_crop(full_array, crop_array, y, x):
    out = np.asarray(full_array).copy()
    h, w = crop_array.shape[-2:]
    if out.ndim == 2:
        out[y : y + h, x : x + w] = crop_array
    elif out.ndim == 3:
        out[..., y : y + h, x : x + w] = crop_array
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {out.shape}")
    return out


def transform_raw9(raw9, mode):
    raw9 = np.asarray(raw9, dtype=np.float32)
    if mode == "none":
        return raw9
    if mode == "flip_lr":
        return np.flip(raw9, axis=-1).copy()
    if mode == "flip_ud":
        return np.flip(raw9, axis=-2).copy()
    if mode == "rot180":
        return np.flip(np.flip(raw9, axis=-1), axis=-2).copy()
    raise ValueError(mode)


def load_depth_m(path):
    depth = np.load(path).astype(np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size and np.nanpercentile(finite, 95.0) > 100.0:
        depth = depth / 1000.0
    return depth.astype(np.float32)


def lfrd2_real_amplitude_and_confidence(raw8, depth_input_m):
    raw8 = np.asarray(raw8, dtype=np.float32)
    real100 = raw8[4] - raw8[6]
    imag100 = raw8[7] - raw8[5]
    amplitude = np.sqrt(real100 * real100 + imag100 * imag100) / 2.0

    # This mirrors LFRD2's real-test confidence expression, but uses the
    # available captured raw instead of unavailable paired label raw.
    i100 = raw8[3] - raw8[1]
    q100 = raw8[0] - raw8[2]
    confidence = np.abs(i100) + np.abs(q100)
    h, w = depth_input_m.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy, cx, cy = 190.6186, 200.7526, 116.8406, 101.4767
    intrix = ((xx - cx) / fx) ** 2 + ((yy - cy) / fy) ** 2 + 1.0
    confidence_aug = (confidence * intrix) ** 2 * np.maximum(depth_input_m, 0.0)
    return amplitude.astype(np.float32), confidence_aug.astype(np.float32)


def make_block_mask(valid_mask, rng, ratio, min_size, max_size):
    valid_mask = np.asarray(valid_mask, dtype=bool)
    target = max(1, int(round(float(ratio) * int(valid_mask.sum()))))
    mask = np.zeros_like(valid_mask, dtype=bool)
    valid_yx = np.argwhere(valid_mask)
    if valid_yx.size == 0:
        return mask

    attempts = 0
    while mask.sum() < target and attempts < 2000:
        attempts += 1
        cy, cx = valid_yx[rng.integers(0, len(valid_yx))]
        bh = int(rng.integers(min_size, max_size + 1))
        bw = int(rng.integers(min_size, max_size + 1))
        y0 = max(0, int(cy - bh // 2))
        x0 = max(0, int(cx - bw // 2))
        y1 = min(mask.shape[0], y0 + bh)
        x1 = min(mask.shape[1], x0 + bw)
        candidate = np.zeros_like(mask, dtype=bool)
        candidate[y0:y1, x0:x1] = True
        candidate &= valid_mask
        mask |= candidate
    return mask & valid_mask


def build_model(args, device):
    global LFRD2_ROOT
    LFRD2_ROOT = args.lfrd2_root
    FracDiff = import_lfrd2()
    lfrd2_args = SimpleNamespace(
        prop_kernel=3,
        preserve_input=False,
        from_scratch=True,
        prop_time=6,
        affinity="TGASS",
        affinity_gamma=0.5,
        conf_prop=True,
        legacy=False,
    )
    model = FracDiff(lfrd2_args).to(device)
    checkpoint = args.checkpoint_net
    if checkpoint is None:
        checkpoint = os.path.join(
            args.lfrd2_root,
            "checkpoint",
            args.checkpoint_domain,
            "net_parameter_x.pth",
        )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def predict_lfrd2(model, depth_input_m, amplitude, confidence_aug, device):
    depth_t = torch.from_numpy(depth_input_m[None, None] * 1000.0).float().to(device)
    amp_t = torch.from_numpy(amplitude[None, None]).float().to(device)
    conf_t = torch.from_numpy(confidence_aug[None, None]).float().to(device)
    with torch.no_grad():
        pred_mm = model(depth_t, amp_t, conf_t)["y_pred"][0]
    return (pred_mm[0, 0].detach().cpu().numpy() / 1000.0).astype(np.float32)


def mae(pred, target, mask):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(target)
    if not valid.any():
        return None, 0
    return float(np.abs(pred[valid] - target[valid]).mean()), int(valid.sum())


def clip_to_valid_range(pred, reference, valid_mask):
    values = reference[valid_mask & np.isfinite(reference)]
    if values.size == 0:
        return pred
    lo = float(np.percentile(values, 0.5))
    hi = float(np.percentile(values, 99.5))
    if hi <= lo:
        return pred
    return np.clip(pred, lo, hi).astype(np.float32)


def save_visualization_masked(path, title, gt, corrupted, anchor, lfrd2, mask):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = np.isfinite(gt) & (gt > 0)
    if valid.any():
        vmin, vmax = np.percentile(gt[valid], [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    err = np.abs(lfrd2 - gt)
    err_vals = err[mask & np.isfinite(err)]
    err_max = float(np.percentile(err_vals, 98)) if err_vals.size else 1.0
    err_max = max(err_max, 1e-6)

    panels = [
        ("gt", gt, "turbo", vmin, vmax),
        ("corrupted", np.where(mask, np.nan, corrupted), "turbo", vmin, vmax),
        ("mask", mask.astype(np.float32), "gray", 0.0, 1.0),
        ("ns anchor", anchor, "turbo", vmin, vmax),
        ("lfrd2", lfrd2, "turbo", vmin, vmax),
        ("|lfrd2-gt| in mask", np.where(mask, err, np.nan), "magma", 0.0, err_max),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (name, image, cmap, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_visualization_observed(path, title, depth, corrupted, anchor, lfrd2, mask):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        vmin, vmax = np.percentile(depth[valid], [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    diff = np.abs(lfrd2 - anchor)
    diff_vals = diff[mask & np.isfinite(diff)]
    diff_max = float(np.percentile(diff_vals, 98)) if diff_vals.size else 1.0
    diff_max = max(diff_max, 1e-6)

    panels = [
        ("raw depth", depth, "turbo", vmin, vmax),
        ("corrupted", np.where(mask, np.nan, corrupted), "turbo", vmin, vmax),
        ("mask", mask.astype(np.float32), "gray", 0.0, 1.0),
        ("ns anchor", anchor, "turbo", vmin, vmax),
        ("lfrd2", lfrd2, "turbo", vmin, vmax),
        ("|lfrd2-anchor| in mask", np.where(mask, diff, np.nan), "magma", 0.0, diff_max),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (name, image, cmap, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def aggregate(rows):
    out = {"num_cases": len(rows)}
    metrics = [
        "anchor_mask_mae",
        "lfrd2_mask_mae",
        "lfrd2_hole_only_global_mae",
        "lfrd2_full_global_mae",
        "lfrd2_unmasked_mae",
        "lfrd2_outside_mean_abs_change",
        "observed_hole_ratio_full",
        "observed_hole_ratio_crop",
        "anchor_observed_fill_ratio",
        "lfrd2_observed_fill_ratio",
        "anchor_observed_median_m",
        "lfrd2_observed_median_m",
        "lfrd2_vs_anchor_observed_mean_abs_diff",
    ]
    for key in metrics:
        vals = [row[key] for row in rows if row.get(key) is not None and np.isfinite(row[key])]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_median"] = float(np.median(arr))
            out[f"{key}_min"] = float(arr.min())
            out[f"{key}_max"] = float(arr.max())
    improved = [row for row in rows if row.get("lfrd2_mask_mae") is not None and row.get("anchor_mask_mae") is not None]
    if improved:
        out["lfrd2_better_than_anchor_cases"] = int(
            sum(row["lfrd2_mask_mae"] < row["anchor_mask_mae"] for row in improved)
        )
        out["compared_cases"] = len(improved)
        anchor = float(np.mean([row["anchor_mask_mae"] for row in improved]))
        lfrd2 = float(np.mean([row["lfrd2_mask_mae"] for row in improved]))
        out["mask_improvement_vs_anchor"] = (anchor - lfrd2) / anchor if anchor > 0 else None
    return out


def main():
    args = parse_args()
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ["pred", "hole_only", "anchor", "mask", "visualizations"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    model, checkpoint = build_model(args, device)
    pairs = paired_paths(args.raw_dir, args.depth_dir)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise FileNotFoundError("No paired raw/depth .npy files found.")

    real_hole_library = []
    if args.eval_mode == "masked_self_test" and args.mask_mode == "real_hole_shapes":
        real_hole_library = build_real_hole_component_library(
            pairs,
            args.hole_depth_threshold,
            min_area=int(args.real_hole_min_area),
            max_area=int(args.real_hole_max_area),
            source_mode="real_hole_shapes",
            amplitude_mode="iq6",
            args=None,
        )
        real_hole_library, _ = filter_and_rebalance_component_library(
            real_hole_library,
            SimpleNamespace(
                seed=int(args.seed),
                mask_mode="real_hole_shapes",
                real_speckle_component_ratio=0.6,
                real_speckle_train_min_area=6,
                real_speckle_train_max_area=0,
            ),
        )
        if not real_hole_library:
            raise ValueError("No eligible real-hole components found for mask_mode='real_hole_shapes'.")

    rows = []
    rng = np.random.default_rng(args.seed)
    vis_saved = 0
    for stem, raw_path, depth_path in pairs:
        raw9 = transform_raw9(np.load(raw_path).astype(np.float32), args.raw9_transform)
        depth = load_depth_m(depth_path)
        raw9 = ensure_raw9_chw(raw9, depth.shape)
        observed_hole_full = (~np.isfinite(depth)) | (depth <= args.hole_depth_threshold)
        if args.crop_mode == "max_hole_window":
            crop_y, crop_x, crop_h, crop_w = best_hole_window(observed_hole_full, args.height, args.width)
        else:
            crop_y, crop_x, crop_h, crop_w = resolve_crop_bounds(depth.shape, args)

        raw8 = crop_hw(raw9[:8], crop_y, crop_x, crop_h, crop_w)
        gt = crop_hw(depth, crop_y, crop_x, crop_h, crop_w).astype(np.float32)
        observed_hole = (~np.isfinite(gt)) | (gt <= args.hole_depth_threshold)
        reliable = (
            np.isfinite(gt)
            & (gt > args.valid_min_depth)
            & (gt <= args.valid_max_depth)
            & (~observed_hole)
        )

        num_passes = max(1, int(args.num_masks_per_sample)) if args.eval_mode == "masked_self_test" else 1
        for mask_index in range(num_passes):
            if args.eval_mode == "observed_holes":
                artificial = np.zeros_like(observed_hole, dtype=bool)
            elif args.mask_mode == "real_hole_shapes":
                artificial = make_real_hole_shape_mask(
                    reliable,
                    real_hole_library,
                    rng,
                    args.mask_ratio,
                    min_overlap_ratio=float(args.real_hole_min_overlap),
                    max_components=int(args.real_hole_max_components),
                    max_attempts=int(args.real_hole_max_attempts),
                )
            else:
                artificial = make_block_mask(
                    reliable,
                    rng,
                    args.mask_ratio,
                    args.min_block_size,
                    args.max_block_size,
                )
            total_hole = observed_hole | artificial
            if not total_hole.any():
                continue
            if args.eval_mode == "masked_self_test" and not artificial.any():
                continue
            corrupted = gt.copy()
            corrupted[total_hole] = 0.0
            anchor = opencv_depth_inpaint(
                corrupted,
                total_hole,
                method="ns",
                radius=args.anchor_inpaint_radius,
            ).astype(np.float32)

            if args.input_mode == "anchor":
                model_input = anchor
            else:
                model_input = corrupted
            amplitude, confidence_aug = lfrd2_real_amplitude_and_confidence(raw8, model_input)
            pred = predict_lfrd2(model, model_input, amplitude, confidence_aug, device)
            if args.clip_output:
                pred = clip_to_valid_range(pred, gt, reliable)
            out_stem = stem if num_passes == 1 else f"{stem}_m{mask_index:02d}"
            if args.eval_mode == "observed_holes":
                hole_only = gt.copy()
                hole_only[observed_hole] = pred[observed_hole]

                pred_full = paste_crop(depth, pred, crop_y, crop_x)
                hole_only_full = paste_crop(depth, hole_only, crop_y, crop_x)
                anchor_insert = gt.copy()
                anchor_insert[observed_hole] = anchor[observed_hole]
                anchor_full = paste_crop(depth, anchor_insert, crop_y, crop_x)
                mask_full = np.zeros_like(depth, dtype=np.uint8)
                mask_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = observed_hole.astype(np.uint8)

                anchor_fill_ratio = float(np.isfinite(anchor[observed_hole]).mean()) if observed_hole.any() else None
                pred_fill_ratio = float(np.isfinite(pred[observed_hole]).mean()) if observed_hole.any() else None
                anchor_vals = anchor[observed_hole & np.isfinite(anchor)]
                pred_vals = pred[observed_hole & np.isfinite(pred)]
                anchor_hole_median = float(np.median(anchor_vals)) if anchor_vals.size else None
                pred_hole_median = float(np.median(pred_vals)) if pred_vals.size else None
                anchor_pred_diff, _ = mae(pred, anchor, observed_hole)
                outside_change, _ = mae(pred, corrupted, reliable)

                np.save(os.path.join(args.output_dir, "pred", f"{out_stem}_lfrd2.npy"), pred_full)
                np.save(os.path.join(args.output_dir, "hole_only", f"{out_stem}_lfrd2_hole_only.npy"), hole_only_full)
                np.save(os.path.join(args.output_dir, "anchor", f"{out_stem}_anchor.npy"), anchor_full)
                np.save(os.path.join(args.output_dir, "mask", f"{out_stem}_mask.npy"), mask_full)

                row = {
                    "stem": stem,
                    "raw_path": raw_path,
                    "depth_path": depth_path,
                    "crop_yxhw": [crop_y, crop_x, crop_h, crop_w],
                    "observed_hole_ratio_full": float((~np.isfinite(depth) | (depth <= args.hole_depth_threshold)).mean()),
                    "observed_hole_ratio_crop": float(observed_hole.mean()),
                    "observed_hole_count_crop": int(observed_hole.sum()),
                    "reliable_count_crop": int(reliable.sum()),
                    "anchor_observed_fill_ratio": anchor_fill_ratio,
                    "lfrd2_observed_fill_ratio": pred_fill_ratio,
                    "anchor_observed_median_m": anchor_hole_median,
                    "lfrd2_observed_median_m": pred_hole_median,
                    "lfrd2_vs_anchor_observed_mean_abs_diff": anchor_pred_diff,
                    "lfrd2_outside_mean_abs_change": outside_change,
                }
            else:
                hole_only = gt.copy()
                hole_only[total_hole] = pred[total_hole]

                anchor_mask_mae, mask_count = mae(anchor, gt, artificial)
                lfrd2_mask_mae, _ = mae(pred, gt, artificial)
                lfrd2_hole_only_global_mae, global_count = mae(hole_only, gt, reliable | artificial)
                lfrd2_full_global_mae, _ = mae(pred, gt, reliable | artificial)
                lfrd2_unmasked_mae, _ = mae(pred, gt, reliable & (~artificial))
                outside_change, _ = mae(pred, corrupted, reliable & (~artificial))

                np.save(os.path.join(args.output_dir, "pred", f"{out_stem}_lfrd2.npy"), pred)
                np.save(os.path.join(args.output_dir, "hole_only", f"{out_stem}_lfrd2_hole_only.npy"), hole_only)
                np.save(os.path.join(args.output_dir, "anchor", f"{out_stem}_anchor.npy"), anchor)
                np.save(os.path.join(args.output_dir, "mask", f"{out_stem}_mask.npy"), artificial.astype(np.uint8))

                row = {
                    "stem": stem,
                    "mask_index": mask_index,
                    "raw_path": raw_path,
                    "depth_path": depth_path,
                    "crop_yxhw": [crop_y, crop_x, crop_h, crop_w],
                    "mask_ratio": float(artificial.mean()),
                    "observed_hole_ratio": float(observed_hole.mean()),
                    "mask_count": mask_count,
                    "global_count": global_count,
                    "anchor_mask_mae": anchor_mask_mae,
                    "lfrd2_mask_mae": lfrd2_mask_mae,
                    "lfrd2_hole_only_global_mae": lfrd2_hole_only_global_mae,
                    "lfrd2_full_global_mae": lfrd2_full_global_mae,
                    "lfrd2_unmasked_mae": lfrd2_unmasked_mae,
                    "lfrd2_outside_mean_abs_change": outside_change,
                }
                if anchor_mask_mae is not None and lfrd2_mask_mae is not None and anchor_mask_mae > 0:
                    row["mask_improvement_vs_anchor"] = (anchor_mask_mae - lfrd2_mask_mae) / anchor_mask_mae
            rows.append(row)

            if args.visualize and vis_saved < args.vis_max_samples:
                if args.eval_mode == "observed_holes":
                    hole_diff = row.get("lfrd2_vs_anchor_observed_mean_abs_diff")
                    title = f"{out_stem} | mean|lfrd2-anchor|={hole_diff:.4f}" if hole_diff is not None else out_stem
                    save_visualization_observed(
                        os.path.join(args.output_dir, "visualizations", f"{out_stem}.png"),
                        title,
                        gt,
                        corrupted,
                        anchor,
                        pred,
                        observed_hole,
                    )
                else:
                    title = (
                        f"{out_stem} | anchor={anchor_mask_mae:.4f} "
                        f"lfrd2={lfrd2_mask_mae:.4f}"
                    )
                    save_visualization_masked(
                        os.path.join(args.output_dir, "visualizations", f"{out_stem}.png"),
                        title,
                        gt,
                        corrupted,
                        anchor,
                        pred,
                        artificial,
                    )
                vis_saved += 1

    if args.eval_mode == "observed_holes":
        method_name = "LFRD2 FracDiff proxy on DepthCAD raw9 observed-hole inference"
        note = (
            "This is not the official LFRD2 UDC raw pipeline. DepthCAD raw9 is "
            "adapted by selecting the first 8 channels and cropping to the model's "
            "180x240 input window, then pasting hole-only predictions back into the "
            "original frame. This mode has no ground-truth hole MAE."
        )
    else:
        method_name = "LFRD2 FracDiff proxy on DepthCAD raw9 masked self-test"
        note = (
            "This is not the official LFRD2 UDC raw pipeline. DepthCAD raw9 is "
            "9x240x320 while LFRD2 real checkpoints expect 8x180x240 UDC raw. "
            "The adapter uses the first 8 raw channels, a cropped input window, and LFRD2's "
            "FracDiff depth refinement checkpoint for a same-mask proxy baseline."
        )

    summary = {
        "method": method_name,
        "note": note,
        "lfrd2_root": os.path.abspath(args.lfrd2_root),
        "checkpoint_net": os.path.abspath(checkpoint),
        "device": str(device),
        "eval_mode": args.eval_mode,
        "input_mode": args.input_mode,
        "raw9_transform": args.raw9_transform,
        "crop_mode": args.crop_mode,
        "crop_yxhw": [args.crop_y, args.crop_x, args.height, args.width],
        "mask_mode": args.mask_mode,
        "mask_ratio": args.mask_ratio,
        "num_masks_per_sample": args.num_masks_per_sample,
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved LFRD2 proxy masked self-test to {args.output_dir}")


if __name__ == "__main__":
    main()
