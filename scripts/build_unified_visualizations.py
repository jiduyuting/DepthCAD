#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from inference_depth_postprocess import opencv_depth_inpaint
from train_depth_completion import move_batch_to_device
from train_depth_flow_restoration import flow_model_in_channels, predict_endpoint_norm, sample_flow
from train_depth_restoration import DepthRestorationCacheDataset


PBRT_FLOW_DIR = Path("output/pbrt_real_threshold_amp_depth_finetune_replay_e30_p8_keepamp_rw030")
PBRT_FLOW_CKPT = PBRT_FLOW_DIR / "best.pt"
PBRT_DEPTHCAD_VIS = Path("output/depthcad_holeaware_kinect_n1000_ft_pilot/eval_seed123/visualizations")
PBRT_PROPAINTER_CASE = Path("output/pbrt_propainter_seed123")
PBRT_LFRD2_ROOT = Path("/data/pre_student/hcy/LFRD2/results/pbrt/depth")

REAL_SELFTEST_ROOT = Path("output/real_raw9_masked_self_test_ratio10_thr1m_iq6")
REAL_SELFTEST_LFRD2_ROOT = Path("output/lfrd2_raw9_masked_self_test_anchor_fliplr")
REAL_SELFTEST_DEPTH_DIR = Path("depth")

REAL_REALHOLE_SELFTEST_METHODS = [
    (
        "Strong e100",
        Path("output/real_raw9_masked_self_test_realholes_ratio10_thr1m_iq6_finetuned_e100_m8_best_c24"),
    ),
    (
        "HoleFocus cont",
        Path("output/real_raw9_masked_self_test_realholes_ratio10_thr1m_iq6_holefocus_continue_e20_lr5e6_best"),
    ),
    (
        "HoleFocus e30",
        Path("output/real_raw9_masked_self_test_realholes_ratio10_thr1m_iq6_holefocus_e30_m8_best"),
    ),
    (
        "RealHoles e40",
        Path("output/real_raw9_masked_self_test_realholes_ratio10_thr1m_iq6_realholes_e40_m8_best"),
    ),
    (
        "Best real-hole",
        Path("output/real_raw9_masked_self_test_after_synth_realhole_e20_lr5e6"),
    ),
]
REAL_REALHOLE_SELFTEST_VIZ_METHODS = [
    "Strong e100",
    "RealHoles e40",
    "HoleFocus e30",
    "HoleFocus cont",
    "Best real-hole",
]

REAL_DATASET_ROOT = Path("/data/pre_student/hcy/datasets/pbrt/Real")
REAL_SELECTED_COMPARE_ROOT = Path("output/pbrt_real_new_selection/oneclick_compare")
REAL_SELECTED_LFRD2_ROOT = REAL_SELECTED_COMPARE_ROOT / "methods" / "lfrd2_holecrop"
REAL_SELECTED_OURS_ROOT = Path("output/pbrt_real_new_selection/raw9_flow_pbrt_real_ft_replay_selected_v99")


def parse_args():
    parser = argparse.ArgumentParser(description="Build unified visualization sheets from existing outputs.")
    parser.add_argument("--output_dir", type=Path, default=Path("output/unified_visualizations"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pbrt_limit", type=int, default=8)
    parser.add_argument("--pbrt_batch_size", type=int, default=1)
    parser.add_argument(
        "--pbrt_ours_checkpoint",
        type=Path,
        default=PBRT_FLOW_CKPT,
        help="Checkpoint used for the Ours panel in PBRT seed123 visualizations.",
    )
    parser.add_argument(
        "--real_dataset_root",
        type=Path,
        default=REAL_DATASET_ROOT,
        help="Root of the PBRT Real dataset used for aligned real-sample visualizations.",
    )
    parser.add_argument(
        "--real_selected_compare_root",
        type=Path,
        default=REAL_SELECTED_COMPARE_ROOT,
        help="Root of the selected PBRT Real one-click comparison outputs.",
    )
    parser.add_argument(
        "--real_selected_lfrd2_root",
        type=Path,
        default=REAL_SELECTED_LFRD2_ROOT,
        help="Optional root containing hole_only/*_lfrd2_hole_only.npy for selected PBRT Real visualizations.",
    )
    parser.add_argument(
        "--real_selected_ours_root",
        type=Path,
        default=REAL_SELECTED_OURS_ROOT,
        help=(
            "Root containing selected PBRT Real inference from the same Ours checkpoint. "
            "The restored/*_restored.npy output is blended only inside the observed hole mask."
        ),
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def natural_key(text):
    parts = []
    chunk = ""
    is_digit = None
    for ch in str(text):
        ch_is_digit = ch.isdigit()
        if is_digit is None or ch_is_digit == is_digit:
            chunk += ch
        else:
            parts.append(int(chunk) if is_digit else chunk)
            chunk = ch
        is_digit = ch_is_digit
    if chunk:
        parts.append(int(chunk) if is_digit else chunk)
    return parts


def stem_map(paths, suffix):
    out = {}
    for path in paths:
        stem = path.stem
        if suffix and not stem.endswith(suffix):
            continue
        key = stem[: -len(suffix)] if suffix else stem
        out[key] = path
    return out


def load_lfrd2_hole_only_map(root):
    root = Path(root)
    hole_dir = root / "hole_only"
    if not hole_dir.is_dir():
        return {}
    return stem_map(hole_dir.glob("*_lfrd2_hole_only.npy"), "_lfrd2_hole_only")


def load_restored_or_hole_only_map(root):
    root = Path(root)
    restored_dir = root / "restored"
    if restored_dir.is_dir():
        restored = stem_map(restored_dir.glob("*_restored.npy"), "_restored")
        if restored:
            return restored, "restored"
    hole_dir = root / "hole_only"
    if hole_dir.is_dir():
        hole_only = stem_map(hole_dir.glob("*_hole_only.npy"), "_hole_only")
        if hole_only:
            return hole_only, "hole_only"
    return {}, None


def blend_observed_holes(raw_depth, observed_hole, pred):
    if pred is None:
        return None
    pred = np.asarray(pred, dtype=np.float32)
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    observed_hole = np.asarray(observed_hole, dtype=bool)
    if pred.shape != raw_depth.shape:
        pred = cv2.resize(pred, (raw_depth.shape[1], raw_depth.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.where(observed_hole, pred, raw_depth).astype(np.float32)


def load_rgb_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def depth_to_meters(depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return depth
    if float(np.percentile(finite, 95.0)) > 30.0:
        return depth / 1000.0
    return depth


def depth_limits(*arrays):
    values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr) & (arr > 0)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    values = np.concatenate(values)
    lo, hi = np.percentile(values, [2.0, 98.0])
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def render_depth_rgb(depth, vmin, vmax):
    depth = np.asarray(depth, dtype=np.float32)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = (plt.colormaps["turbo"](norm)[..., :3] * 255.0).astype(np.uint8)
    invalid = ~np.isfinite(depth)
    rgb[invalid] = 255
    return rgb


def render_mask_rgb(mask):
    mask = np.asarray(mask).astype(bool)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask] = 255
    return rgb


def render_masked_error_rgb(pred, target, mask, vmax):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if pred.shape != target.shape:
        pred = cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if vmax <= 0:
        vmax = 1.0
    error = np.abs(pred - target)
    norm = np.clip(error / float(vmax), 0.0, 1.0)
    rgb = (plt.colormaps["magma"](norm)[..., :3] * 255.0).astype(np.uint8)
    rgb[~mask] = 230
    invalid = ~np.isfinite(error)
    rgb[invalid] = 255
    return rgb


def render_text_panel(text, size):
    width, height = size
    image = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(18, min(width, height) // 12))
    except Exception:
        font = ImageFont.load_default()
    lines = text.split("\n")
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [box[3] - box[1] for box in line_boxes]
    total_height = sum(line_heights) + max(0, len(lines) - 1) * 8
    y = (height - total_height) // 2
    for line, box, line_h in zip(lines, line_boxes, line_heights):
        line_w = box[2] - box[0]
        x = (width - line_w) // 2
        draw.text((x, y), line, fill=(35, 35, 35), font=font)
        y += line_h + 8
    return np.asarray(image)


def crop_grid_panel(path, grid_rows, grid_cols, row, col, pad_left=0.04, pad_right=0.04, pad_top=0.14, pad_bottom=0.04):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    cell_w = width / float(grid_cols)
    cell_h = height / float(grid_rows)
    x0 = int(round(col * cell_w + pad_left * cell_w))
    x1 = int(round((col + 1) * cell_w - pad_right * cell_w))
    y0 = int(round(row * cell_h + pad_top * cell_h))
    y1 = int(round((row + 1) * cell_h - pad_bottom * cell_h))
    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))
    return np.asarray(image.crop((x0, y0, x1, y1)))


def save_panel_figure(path, title, panels, cols=4):
    rows = int(math.ceil(len(panels) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax, (panel_title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    ensure_dir(Path(path).parent)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def masked_selftest_metric_map(summary):
    metrics = {}
    for row in summary.get("per_sample", []):
        repeat = int(row.get("repeat", 0))
        key = f"{row['name']}_r{repeat:02d}"
        metrics[key] = row
    return metrics


def format_metric_title(name, metric):
    if metric is None:
        return name
    return f"{name}\nmask MAE {float(metric):.4f}m"


def save_case_montage(path, title, case_rows, image_key, thumb_width=720, cols=2):
    images = []
    label_height = 34
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    for row in case_rows:
        image = Image.open(row[image_key]).convert("RGB")
        scale = float(thumb_width) / float(image.width)
        thumb_height = max(1, int(round(image.height * scale)))
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_width, thumb_height + label_height), (255, 255, 255))
        draw = ImageDraw.Draw(tile)
        label = (
            f"{row['sample']}  anchor {row['anchor_mask_mae']:.4f}m  "
            f"best {row['best_method']} {row['best_model_mask_mae']:.4f}m"
        )
        draw.text((10, 6), label, fill=(20, 20, 20), font=font)
        tile.paste(image, (0, label_height))
        images.append(tile)
    if not images:
        return

    rows = int(math.ceil(len(images) / float(cols)))
    title_height = 44
    gap = 16
    tile_w = max(image.width for image in images)
    tile_h = max(image.height for image in images)
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas_h = title_height + rows * tile_h + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), title, fill=(20, 20, 20), font=title_font)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = title_height + gap + row * (tile_h + gap)
        canvas.paste(image, (x, y))
    ensure_dir(Path(path).parent)
    canvas.save(path)


def build_flow_model(ckpt, ckpt_args, dataset, device):
    time_channels = int(ckpt_args.get("time_channels", 16))
    in_channels = flow_model_in_channels(dataset.input_channels, time_channels)
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
    return model


@torch.no_grad()
def predict_flow_samples(checkpoint_path, sample_paths, device_name, batch_size):
    device = torch.device(device_name)
    ckpt = load_checkpoint(str(checkpoint_path), device)
    ckpt_args = ckpt.get("args", {})
    dataset_kwargs = {
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
    dataset = DepthRestorationCacheDataset(sample_paths, **dataset_kwargs)
    model = build_flow_model(ckpt, ckpt_args, dataset, device)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    time_channels = int(ckpt_args.get("time_channels", 16))
    max_velocity_norm = float(ckpt_args.get("max_velocity_norm", 4.0))
    clip_norm_depth = float(ckpt_args.get("clip_norm_depth", 8.0))
    velocity_scale = float(ckpt_args.get("velocity_scale", 1.0))
    sample_steps = int(ckpt_args.get("sample_steps", 8))
    sampling_mode = ckpt_args.get("eval_sampling_mode", "euler")
    outputs = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if sampling_mode == "endpoint":
            pred_norm = predict_endpoint_norm(
                model,
                batch,
                time_channels,
                max_velocity_norm,
                clip_norm_depth,
                velocity_scale,
            )
        else:
            pred_norm = sample_flow(
                model,
                batch,
                time_channels,
                max_velocity_norm,
                sample_steps,
                clip_norm_depth,
                velocity_scale,
            )
        scale = batch["scale"].view(-1, 1, 1, 1)
        center = batch["center"].view(-1, 1, 1, 1)
        pred = pred_norm * scale + center
        for i in range(pred.shape[0]):
            sample_name = batch["sample_name"][i]
            outputs[sample_name] = pred[i, 0].detach().cpu().numpy().astype(np.float32)
    return outputs


def parse_depthcad_vis_sample(path):
    stem = Path(path).stem
    if not stem.startswith("vis_"):
        return None
    parts = stem[4:].split("_")
    if len(parts) != 3:
        return None
    return f"{parts[0]}/{parts[1]}/{parts[2]}"


def build_pbrt_figures(args):
    mapping_rows = load_json(PBRT_PROPAINTER_CASE / "external_inputs" / "source_mapping.json")["frame_mapping"]
    mapping_by_sample = {row["sample_name"]: row for row in mapping_rows}
    depthcad_vis_by_sample = {}
    for path in PBRT_DEPTHCAD_VIS.glob("vis_*.png"):
        sample = parse_depthcad_vis_sample(path)
        if sample is not None:
            depthcad_vis_by_sample[sample] = path
    lfrd2_samples = {
        str(path.relative_to(PBRT_LFRD2_ROOT)).replace(".npy", ""): path
        for path in PBRT_LFRD2_ROOT.rglob("*.npy")
    }
    common_all = sorted(
        set(mapping_by_sample) & set(depthcad_vis_by_sample) & set(lfrd2_samples),
        key=natural_key,
    )
    common = list(common_all)
    if args.pbrt_limit > 0:
        common = common[: int(args.pbrt_limit)]
    sample_paths = [mapping_by_sample[s]["cache_path"] for s in common]
    flow_preds = predict_flow_samples(args.pbrt_ours_checkpoint, sample_paths, args.device, args.pbrt_batch_size)

    out_dir = args.output_dir / "pbrt_seed123_aligned"
    ensure_dir(out_dir / "figures")
    summary_rows = []
    for sample in common:
        meta = mapping_by_sample[sample]
        cache_path = Path(meta["cache_path"])
        with np.load(cache_path, allow_pickle=False) as data:
            gt = data["gt_depth"].astype(np.float32)
            noisy = data["depth_noisy"].astype(np.float32)
            hole = data["hole_mask"] > 0.5
        anchor = opencv_depth_inpaint(noisy, hole, method="ns", radius=15).astype(np.float32)
        ours = flow_preds[sample]
        prop = np.load(PBRT_PROPAINTER_CASE / "propainter_run" / "restored_by_stem" / f"{meta['source_stem']}_propainter_restored.npy").astype(np.float32)
        lfrd2 = np.load(lfrd2_samples[sample]).astype(np.float32)
        if lfrd2.shape != gt.shape:
            lfrd2 = cv2.resize(lfrd2, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        depthcad_rgb = crop_grid_panel(depthcad_vis_by_sample[sample], 2, 4, 0, 3)

        vmin, vmax = depth_limits(gt, noisy, anchor, ours, prop, lfrd2)
        panels = [
            ("GT", render_depth_rgb(gt, vmin, vmax)),
            ("Noisy", render_depth_rgb(noisy, vmin, vmax)),
            ("Hole Mask", render_mask_rgb(hole)),
            ("Anchor", render_depth_rgb(anchor, vmin, vmax)),
            ("Ours", render_depth_rgb(ours, vmin, vmax)),
            ("DepthCAD", depthcad_rgb),
            ("ProPainter", render_depth_rgb(prop, vmin, vmax)),
            ("LFRD2", render_depth_rgb(lfrd2, vmin, vmax)),
        ]
        out_path = out_dir / "figures" / f"{sample.replace('/', '_')}.png"
        save_panel_figure(out_path, sample, panels)
        summary_rows.append(
            {
                "sample": sample,
                "cache_path": str(cache_path),
                "figure": str(out_path),
                "depthcad_visualization": str(depthcad_vis_by_sample[sample]),
                "note": "DepthCAD panel cropped from existing eval visualization; LFRD2 resized from 240x320 to 256x256.",
            }
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "benchmark": "pbrt_seed123",
                "num_figures": len(summary_rows),
                "samples": common,
                "ours_checkpoint": str(args.pbrt_ours_checkpoint),
                "coverage": {
                    "propainter_mapping": len(mapping_by_sample),
                    "depthcad_visualizations": len(depthcad_vis_by_sample),
                    "lfrd2_predictions": len(lfrd2_samples),
                    "common_before_limit": len(common_all),
                    "common_after_limit": len(common),
                },
                "rows": summary_rows,
            },
            f,
            indent=2,
        )


def build_real_masked_selftest_figures(args):
    out_dir = args.output_dir / "real_masked_selftest_partial"
    ensure_dir(out_dir / "figures")
    ours_stems = {
        path.name.split("_")[0]
        for path in (REAL_SELFTEST_ROOT / "mask").glob("*_r00_mask.npy")
    }
    lfrd2_stems = {
        path.name.split("_")[0]
        for path in (REAL_SELFTEST_LFRD2_ROOT / "hole_only").glob("*_lfrd2_hole_only.npy")
    }
    stems = sorted(ours_stems & lfrd2_stems, key=natural_key)
    rows = []
    for stem in stems:
        clean = depth_to_meters(np.load(REAL_SELFTEST_DEPTH_DIR / f"{stem}.npy").astype(np.float32))
        corrupted = np.load(REAL_SELFTEST_ROOT / "corrupted" / f"{stem}_r00_corrupted.npy").astype(np.float32)
        mask = np.load(REAL_SELFTEST_ROOT / "mask" / f"{stem}_r00_mask.npy").astype(bool)
        anchor = np.load(REAL_SELFTEST_ROOT / "anchor" / f"{stem}_r00_anchor.npy").astype(np.float32)
        ours = np.load(REAL_SELFTEST_ROOT / "hole_only" / f"{stem}_r00_hole_only.npy").astype(np.float32)
        lfrd2 = np.load(REAL_SELFTEST_LFRD2_ROOT / "hole_only" / f"{stem}_lfrd2_hole_only.npy").astype(np.float32)

        vmin, vmax = depth_limits(clean, corrupted, anchor, ours, lfrd2)
        panel_size = render_depth_rgb(clean, vmin, vmax).shape[1], render_depth_rgb(clean, vmin, vmax).shape[0]
        panels = [
            ("Pseudo-GT", render_depth_rgb(clean, vmin, vmax)),
            ("Corrupted", render_depth_rgb(corrupted, vmin, vmax)),
            ("Artificial Mask", render_mask_rgb(mask)),
            ("Anchor", render_depth_rgb(anchor, vmin, vmax)),
            ("Ours", render_depth_rgb(ours, vmin, vmax)),
            ("DepthCAD", render_text_panel("Unavailable\nfor this 41-sample\nmasked self-test", panel_size)),
            ("ProPainter", render_text_panel("Unavailable\nfor this 41-sample\nmasked self-test", panel_size)),
            ("LFRD2", render_depth_rgb(lfrd2, vmin, vmax)),
        ]
        out_path = out_dir / "figures" / f"{stem}.png"
        save_panel_figure(out_path, stem, panels)
        rows.append(
            {
                "sample": stem,
                "figure": str(out_path),
                "note": "DepthCAD and ProPainter were not available on this masked self-test split; placeholders were inserted.",
            }
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "benchmark": "real_masked_selftest_partial",
                "num_figures": len(rows),
                "samples": stems,
                "coverage": {
                    "ours": len(ours_stems),
                    "lfrd2": len(lfrd2_stems),
                    "common": len(stems),
                },
                "rows": rows,
            },
            f,
            indent=2,
        )


def build_real_raw9_realhole_selftest_figures(args):
    out_dir = args.output_dir / "real_raw9_masked_selftest_realholes_aligned"
    ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "error_figures")

    candidate_data = []
    for name, root in REAL_REALHOLE_SELFTEST_METHODS:
        summary_path = Path(root) / "summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        candidate_data.append(
            {
                "name": name,
                "root": Path(root),
                "summary": summary,
                "summary_path": summary_path,
                "metric_map": masked_selftest_metric_map(summary),
            }
        )

    candidate_by_name = {item["name"]: item for item in candidate_data}
    missing = [name for name in REAL_REALHOLE_SELFTEST_VIZ_METHODS if name not in candidate_by_name]
    if missing:
        raise FileNotFoundError(f"Missing real-hole self-test outputs for: {', '.join(missing)}")

    base_item = candidate_by_name["Best real-hole"]
    common_keys = [
        key
        for key in base_item["metric_map"].keys()
        if all(key in item["metric_map"] for item in candidate_data)
    ]
    common_keys = sorted(common_keys, key=natural_key)

    rows = []
    for key in common_keys:
        base_row = base_item["metric_map"][key]
        sample_name = str(base_row["name"])
        repeat = int(base_row.get("repeat", 0))
        sample_key = f"{sample_name}_r{repeat:02d}"

        depth_path = Path(base_row["depth_path"])
        raw_path = Path(base_row["raw_path"])
        clean = depth_to_meters(np.load(depth_path).astype(np.float32))
        corrupted = depth_to_meters(np.load(base_item["root"] / "corrupted" / f"{sample_key}_corrupted.npy").astype(np.float32))
        mask = np.load(base_item["root"] / "mask" / f"{sample_key}_mask.npy").astype(bool)
        anchor = depth_to_meters(np.load(base_item["root"] / "anchor" / f"{sample_key}_anchor.npy").astype(np.float32))

        preds = {}
        method_rows = {}
        for method_name in REAL_REALHOLE_SELFTEST_VIZ_METHODS:
            item = candidate_by_name[method_name]
            method_mask_path = item["root"] / "mask" / f"{sample_key}_mask.npy"
            method_mask = np.load(method_mask_path).astype(bool)
            if method_mask.shape != mask.shape or not np.array_equal(method_mask, mask):
                raise ValueError(
                    "Real-hole self-test masks are not aligned: "
                    f"{base_item['root']} vs {item['root']} sample={sample_key}"
                )
            pred_path = item["root"] / "hole_only" / f"{sample_key}_hole_only.npy"
            pred = depth_to_meters(np.load(pred_path).astype(np.float32))
            preds[method_name] = pred
            method_rows[method_name] = item["metric_map"][key]

        vmin, vmax = depth_limits(clean, corrupted, anchor, *preds.values())
        error_values = [np.abs(anchor - clean)[mask]]
        for pred in preds.values():
            error_values.append(np.abs(pred - clean)[mask])
        error_values = [arr.ravel() for arr in error_values if arr.size]
        error_vmax = float(np.percentile(np.concatenate(error_values), 98.0)) if error_values else 1.0
        error_vmax = max(error_vmax, 1e-6)

        panels = [
            ("Pseudo-GT", render_depth_rgb(clean, vmin, vmax)),
            ("Masked input", render_depth_rgb(corrupted, vmin, vmax)),
            ("Artificial mask", render_mask_rgb(mask)),
            ("NS anchor", render_depth_rgb(anchor, vmin, vmax)),
        ]
        for method_name in REAL_REALHOLE_SELFTEST_VIZ_METHODS:
            panels.append(
                (
                    format_metric_title(method_name, method_rows[method_name].get("model_mask_mae")),
                    render_depth_rgb(preds[method_name], vmin, vmax),
                )
            )
        figure_path = out_dir / "figures" / f"{sample_name}.png"
        save_panel_figure(figure_path, sample_name, panels)

        error_panels = [
            ("Artificial mask", render_mask_rgb(mask)),
            (
                format_metric_title("NS anchor", base_row.get("anchor_mask_mae")),
                render_masked_error_rgb(anchor, clean, mask, error_vmax),
            ),
        ]
        for method_name in REAL_REALHOLE_SELFTEST_VIZ_METHODS:
            error_panels.append(
                (
                    format_metric_title(method_name, method_rows[method_name].get("model_mask_mae")),
                    render_masked_error_rgb(preds[method_name], clean, mask, error_vmax),
                )
            )
        error_figure_path = out_dir / "error_figures" / f"{sample_name}.png"
        save_panel_figure(error_figure_path, f"{sample_name} abs error", error_panels)

        rows.append(
            {
                "sample": sample_name,
                "sample_key": sample_key,
                "figure": str(figure_path),
                "error_figure": str(error_figure_path),
                "raw_path": str(raw_path),
                "depth_path": str(depth_path),
                "mask_path": str(base_item["root"] / "mask" / f"{sample_key}_mask.npy"),
                "anchor_path": str(base_item["root"] / "anchor" / f"{sample_key}_anchor.npy"),
                "corrupted_path": str(base_item["root"] / "corrupted" / f"{sample_key}_corrupted.npy"),
                "methods": {
                    method_name: {
                        "root": str(candidate_by_name[method_name]["root"]),
                        "checkpoint": candidate_by_name[method_name]["summary"].get("checkpoint"),
                        "hole_only_path": str(candidate_by_name[method_name]["root"] / "hole_only" / f"{sample_key}_hole_only.npy"),
                        "model_mask_mae": method_rows[method_name].get("model_mask_mae"),
                        "model_global_mae": method_rows[method_name].get("model_global_mae"),
                        "model_unmasked_mae": method_rows[method_name].get("model_unmasked_mae"),
                    }
                    for method_name in REAL_REALHOLE_SELFTEST_VIZ_METHODS
                },
                "note": "Pseudo-GT is the paired real depth. Masks are real-hole-shaped artificial masks on valid depth regions.",
            }
        )

    summary_payload = {
        "benchmark": "real_raw9_masked_selftest_realholes_aligned",
        "num_figures": len(rows),
        "samples": [row["sample"] for row in rows],
        "visualized_methods": [
            {
                "name": name,
                "root": str(candidate_by_name[name]["root"]),
                "checkpoint": candidate_by_name[name]["summary"].get("checkpoint"),
                "aggregate": candidate_by_name[name]["summary"].get("aggregate", {}),
            }
            for name in REAL_REALHOLE_SELFTEST_VIZ_METHODS
        ],
        "candidate_methods": [
            {
                "name": item["name"],
                "root": str(item["root"]),
                "checkpoint": item["summary"].get("checkpoint"),
                "aggregate": item["summary"].get("aggregate", {}),
            }
            for item in candidate_data
        ],
        "coverage": {
            "common": len(common_keys),
            "candidate_methods": len(candidate_data),
            "visualized_methods": len(REAL_REALHOLE_SELFTEST_VIZ_METHODS),
        },
        "rows": rows,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    aggregate_rows = []
    for item in sorted(candidate_data, key=lambda entry: entry["summary"].get("aggregate", {}).get("model_mask_mae", float("inf"))):
        agg = item["summary"].get("aggregate", {})
        aggregate_rows.append(
            {
                "method": item["name"],
                "checkpoint": item["summary"].get("checkpoint"),
                "root": str(item["root"]),
                "anchor_mask_mae": agg.get("anchor_mask_mae"),
                "model_mask_mae": agg.get("model_mask_mae"),
                "mask_improve_vs_anchor": agg.get("mask_improve_vs_anchor"),
                "hole_only_global_mae": agg.get("hole_only_global_mae"),
                "model_unmasked_mae": agg.get("model_unmasked_mae"),
            }
        )
    with open(out_dir / "aggregate_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    per_sample_rows = []
    for row in rows:
        anchor_mae = float(base_item["metric_map"][row["sample_key"]].get("anchor_mask_mae"))
        method_maes = {
            method_name: float(row["methods"][method_name]["model_mask_mae"])
            for method_name in REAL_REALHOLE_SELFTEST_VIZ_METHODS
        }
        best_method, best_mae = min(method_maes.items(), key=lambda item: item[1])
        out_row = {
            "sample": row["sample"],
            "sample_key": row["sample_key"],
            "anchor_mask_mae": anchor_mae,
            "best_method": best_method,
            "best_model_mask_mae": best_mae,
            "best_improve_vs_anchor": (anchor_mae - best_mae) / anchor_mae if anchor_mae > 0 else math.nan,
            "figure": row["figure"],
            "error_figure": row["error_figure"],
        }
        for method_name, mae in method_maes.items():
            out_row[f"{method_name}_mask_mae"] = mae
        per_sample_rows.append(out_row)
    per_sample_rows = sorted(per_sample_rows, key=lambda row: natural_key(row["sample"]))
    per_sample_fields = [
        "sample",
        "sample_key",
        "anchor_mask_mae",
        "best_method",
        "best_model_mask_mae",
        "best_improve_vs_anchor",
        *[f"{method_name}_mask_mae" for method_name in REAL_REALHOLE_SELFTEST_VIZ_METHODS],
        "figure",
        "error_figure",
    ]
    with open(out_dir / "per_sample_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_sample_fields)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    best_cases = sorted(per_sample_rows, key=lambda row: row["best_improve_vs_anchor"], reverse=True)[:6]
    worst_cases = sorted(per_sample_rows, key=lambda row: row["best_improve_vs_anchor"])[:6]
    hardest_cases = sorted(per_sample_rows, key=lambda row: row["anchor_mask_mae"], reverse=True)[:6]
    with open(out_dir / "README.md", "w") as f:
        f.write("# Real Raw9 Real-Hole-Shape Masked Self-Test\n\n")
        f.write("Pseudo-GT is paired real depth. Artificial masks are real observed hole shapes pasted onto valid depth regions.\n\n")
        f.write("## Aggregate Ranking\n\n")
        f.write("| Method | Mask MAE (m) | Improve vs NS | Hole-only global MAE (m) | Checkpoint |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for row in aggregate_rows:
            improve = float(row["mask_improve_vs_anchor"]) * 100.0
            f.write(
                f"| {row['method']} | {float(row['model_mask_mae']):.6f} | "
                f"{improve:.2f}% | {float(row['hole_only_global_mae']):.6f} | `{row['checkpoint']}` |\n"
            )
        f.write("\n## Representative Cases\n\n")
        for title, case_rows in [
            ("Best Improvements", best_cases),
            ("Weakest Improvements", worst_cases),
            ("Hardest Anchor Cases", hardest_cases),
        ]:
            f.write(f"### {title}\n\n")
            f.write("| Sample | Anchor MAE | Best Method | Best MAE | Improve | Figure | Error Figure |\n")
            f.write("|---|---:|---|---:|---:|---|---|\n")
            for row in case_rows:
                f.write(
                    f"| {row['sample']} | {row['anchor_mask_mae']:.6f} | {row['best_method']} | "
                    f"{row['best_model_mask_mae']:.6f} | {row['best_improve_vs_anchor'] * 100.0:.2f}% | "
                    f"`{row['figure']}` | `{row['error_figure']}` |\n"
                )
            f.write("\n")

    montage_dir = out_dir / "representative_montages"
    for slug, title, case_rows in [
        ("best_improvements", "Best Improvements", best_cases),
        ("weakest_improvements", "Weakest Improvements", worst_cases),
        ("hardest_anchor_cases", "Hardest Anchor Cases", hardest_cases),
    ]:
        save_case_montage(montage_dir / f"{slug}.png", title, case_rows, "figure")
        save_case_montage(montage_dir / f"{slug}_errors.png", f"{title} - Mask Error", case_rows, "error_figure")


def build_real_pbrt_selected_figures(args):
    out_dir = args.output_dir / "real_pbrt_selected_aligned"
    ensure_dir(out_dir / "figures")

    real_root = Path(args.real_dataset_root)
    compare_root = Path(args.real_selected_compare_root)
    depth_only_selected_root = Path("output/pbrt_real_new_selection/depth_only_flow_selected")

    depth_dir = real_root / "depth"
    raw_vis_dir = real_root / "noise_visualization"
    mask_dir = real_root / "noise_masks"
    external_root = compare_root / "external_inpaint" / "propainter_run" / "restored_by_stem"
    lfrd2_root = Path(args.real_selected_lfrd2_root)
    ours_summary_path = Path(args.real_selected_ours_root) / "summary.json"
    ours_summary = load_json(ours_summary_path) if ours_summary_path.exists() else {}

    depth_map = stem_map(depth_dir.glob("*.npy"), "")
    raw_vis_map = stem_map(raw_vis_dir.glob("*_vis.png"), "_vis")
    mask_map = stem_map(mask_dir.glob("*_overall_missing_mask.npy"), "_overall_missing_mask")
    ours_map, ours_source_kind = load_restored_or_hole_only_map(args.real_selected_ours_root)
    depth_only_map = stem_map((depth_only_selected_root / "hole_only").glob("*_hole_only.npy"), "_hole_only")
    prop_map = stem_map(external_root.glob("*_propainter_restored.npy"), "_propainter_restored")
    lfrd2_map = load_lfrd2_hole_only_map(lfrd2_root)

    stems = sorted(
        set(depth_map) & set(raw_vis_map) & set(mask_map) & set(ours_map) & set(depth_only_map),
        key=natural_key,
    )

    rows = []
    for stem in stems:
        raw_depth = depth_to_meters(np.load(depth_map[stem]).astype(np.float32))
        raw_vis = load_rgb_image(raw_vis_map[stem])
        mask = np.load(mask_map[stem]).astype(bool)
        anchor = opencv_depth_inpaint(raw_depth, mask, method="ns", radius=15).astype(np.float32)
        anchor = blend_observed_holes(raw_depth, mask, anchor)
        ours = blend_observed_holes(raw_depth, mask, np.load(ours_map[stem]).astype(np.float32))
        depth_only = blend_observed_holes(raw_depth, mask, np.load(depth_only_map[stem]).astype(np.float32))
        prop = blend_observed_holes(raw_depth, mask, np.load(prop_map[stem]).astype(np.float32)) if stem in prop_map else None
        lfrd2 = blend_observed_holes(raw_depth, mask, np.load(lfrd2_map[stem]).astype(np.float32)) if stem in lfrd2_map else None

        vmin, vmax = depth_limits(raw_depth, anchor, ours, depth_only, prop, lfrd2)
        panel_size = raw_vis.shape[1], raw_vis.shape[0]
        panels = [
            ("Real Raw Vis", raw_vis),
            ("Raw Depth", render_depth_rgb(raw_depth, vmin, vmax)),
            ("Observed Hole Mask", render_mask_rgb(mask)),
            ("Anchor", render_depth_rgb(anchor, vmin, vmax)),
            ("Ours", render_depth_rgb(ours, vmin, vmax)),
            ("Depth-only", render_depth_rgb(depth_only, vmin, vmax)),
            (
                "ProPainter",
                render_depth_rgb(prop, vmin, vmax)
                if prop is not None
                else render_text_panel("Unavailable\nfor this\nsample", panel_size),
            ),
            (
                "LFRD2",
                render_depth_rgb(lfrd2, vmin, vmax)
                if lfrd2 is not None
                else render_text_panel("Unavailable\nfor this\nsample", panel_size),
            ),
        ]
        out_path = out_dir / "figures" / f"{stem}.png"
        save_panel_figure(out_path, stem, panels)
        rows.append(
            {
                "sample": stem,
                "figure": str(out_path),
                "raw_vis_path": str(raw_vis_map[stem]),
                "depth_path": str(depth_map[stem]),
                "mask_path": str(mask_map[stem]),
                "ours_path": str(ours_map[stem]),
                "propainter_available": stem in prop_map,
                "lfrd2_available": stem in lfrd2_map,
                "note": (
                    "Aligned PBRT Real selected sample. Method outputs are blended only inside the observed hole mask."
                ),
            }
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "benchmark": "real_pbrt_selected_aligned",
                "num_figures": len(rows),
                "samples": stems,
                "ours_root": str(args.real_selected_ours_root),
                "ours_checkpoint": ours_summary.get("checkpoint"),
                "ours_source_kind": ours_source_kind,
                "coverage": {
                    "depth": len(depth_map),
                    "raw_vis": len(raw_vis_map),
                    "mask": len(mask_map),
                    "ours": len(ours_map),
                    "depth_only": len(depth_only_map),
                    "propainter": len(prop_map),
                    "lfrd2": len(lfrd2_map),
                    "common": len(stems),
                    "common_with_propainter": len(set(stems) & set(prop_map)),
                    "common_with_lfrd2": len(set(stems) & set(lfrd2_map)),
                },
                "rows": rows,
            },
            f,
            indent=2,
        )


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    build_pbrt_figures(args)
    build_real_masked_selftest_figures(args)
    build_real_raw9_realhole_selftest_figures(args)
    build_real_pbrt_selected_figures(args)
    print(f"Saved unified visualizations under {args.output_dir}")


if __name__ == "__main__":
    main()
