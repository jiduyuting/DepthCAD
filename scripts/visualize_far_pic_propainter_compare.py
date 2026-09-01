#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def finite_values(arr, mask=None):
    arr = np.asarray(arr)
    if mask is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[mask & np.isfinite(arr)]
    return vals


def image_limits(arrays, valid_mask):
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        v = finite_values(arr, valid_mask)
        if v.size:
            vals.append(v)
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    lo, hi = np.percentile(vals, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def delta_limit(deltas, mask):
    vals = []
    for arr in deltas:
        v = finite_values(arr, mask)
        if v.size:
            vals.append(v)
    if not vals:
        return 1.0
    vals = np.concatenate(vals)
    hi = float(np.percentile(vals, 98.0))
    return max(hi, 1e-6)


def colorize_depth(depth, vmin, vmax, cmap_name="viridis"):
    depth = np.asarray(depth, dtype=np.float32)
    norm = (depth - vmin) / max(vmax - vmin, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    invalid = ~np.isfinite(depth)
    rgb[invalid] = 0
    return rgb


def draw_mask_boundary(rgb, mask, color=(255, 0, 0)):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.max() == 0:
        return rgb
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = (dilated != eroded)
    out = rgb.copy()
    out[boundary] = color
    return out


def save_panel_grid(path, title, panels, cols=5, dpi=130):
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax, panel in zip(axes, panels):
        label, image, cmap, vmin, vmax = panel
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        if cmap is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_contact_sheet(path, rows, thumb_w=224):
    labels = ["raw", "mask", "OpenCV NS", "ours", "ProPainter"]
    label_h = 26
    row_gap = 12
    col_gap = 8
    text_w = 220
    if not rows:
        return
    h, w = rows[0]["raw"].shape[:2]
    scale = thumb_w / float(w)
    thumb_h = int(round(h * scale))
    canvas_w = text_w + len(labels) * thumb_w + (len(labels) - 1) * col_gap
    canvas_h = label_h + len(rows) * thumb_h + max(0, len(rows) - 1) * row_gap
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, label in enumerate(labels):
        x = text_w + i * (thumb_w + col_gap)
        cv2.putText(canvas, label, (x + 4, 18), font, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    y = label_h
    for row in rows:
        cv2.putText(canvas, row["stem"], (4, y + 18), font, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        images = [row["raw"], row["mask"], row["opencv_ns"], row["ours"], row["propainter"]]
        for i, image in enumerate(images):
            thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            x = text_w + i * (thumb_w + col_gap)
            canvas[y : y + thumb_h, x : x + thumb_w] = thumb
        y += thumb_h + row_gap

    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def load_method(case_dir, method, stem):
    path = case_dir / "outputs" / method / f"{stem}_{method}.npy"
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", default="output/far_pic_benchmark/bad_depth_mask_v1")
    parser.add_argument("--propainter_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_frames", type=int, default=1000000)
    return parser.parse_args()


def main():
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    external_dir = case_dir / "external_inputs"
    mapping = load_json(external_dir / "source_mapping.json")["frame_mapping"]
    propainter_dir = (
        Path(args.propainter_dir).resolve()
        if args.propainter_dir
        else case_dir / "propainter_run" / "restored_by_stem"
    )
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else case_dir / "propainter_run" / "visualizations_compare"
    )
    detail_dir = out_dir / "per_frame"
    quick_dir = out_dir / "quick_png"
    mkdir(detail_dir)
    mkdir(quick_dir)

    contact_rows = []
    summary = []
    for item in mapping[: int(args.max_frames)]:
        idx = int(item["frame_index"])
        stem = item["source_stem"]
        raw = np.load(item["source_path_240x320_m"]).astype(np.float32)
        mask = np.load(external_dir / "mask_npy" / f"{idx:04d}.npy").astype(bool)
        corrupted = np.load(external_dir / "depth_npy" / f"{idx:04d}.npy").astype(np.float32)
        opencv_ns = load_method(case_dir, "opencv_ns", stem)
        opencv_telea = load_method(case_dir, "opencv_telea", stem)
        ours_hole = load_method(case_dir, "ours_hole_only", stem)
        ours_restored = load_method(case_dir, "ours_restored", stem)
        propainter_path = propainter_dir / f"{stem}_propainter_restored.npy"
        if not propainter_path.exists():
            raise FileNotFoundError(propainter_path)
        propainter = np.load(propainter_path).astype(np.float32)

        valid = np.isfinite(raw) & (raw > 0.0) & (~mask)
        vmin, vmax = image_limits([raw, opencv_ns, ours_hole, propainter], valid)
        raw_hidden = raw.copy()
        raw_hidden[mask] = np.nan
        corrupted_hidden = corrupted.copy()
        corrupted_hidden[mask] = np.nan

        diff_prop_ns = np.abs(propainter - opencv_ns) if opencv_ns is not None else None
        diff_prop_ours = np.abs(propainter - ours_hole) if ours_hole is not None else None
        diff_ours_ns = np.abs(ours_hole - opencv_ns) if ours_hole is not None and opencv_ns is not None else None
        dmax = delta_limit([d for d in [diff_prop_ns, diff_prop_ours, diff_ours_ns] if d is not None], mask)

        panels = [
            ("raw depth\n(mask hidden)", raw_hidden, "viridis", vmin, vmax),
            ("corrupted input\n(mask hidden)", corrupted_hidden, "viridis", vmin, vmax),
            ("repair mask", mask.astype(np.float32), "gray", 0.0, 1.0),
            ("OpenCV NS", opencv_ns, "viridis", vmin, vmax),
            ("OpenCV Telea", opencv_telea, "viridis", vmin, vmax),
            ("ours hole_only", ours_hole, "viridis", vmin, vmax),
            ("ours restored", ours_restored, "viridis", vmin, vmax),
            ("ProPainter", propainter, "viridis", vmin, vmax),
            ("|ProPainter-NS|\nin mask", np.where(mask, diff_prop_ns, np.nan), "magma", 0.0, dmax),
            ("|ProPainter-ours|\nin mask", np.where(mask, diff_prop_ours, np.nan), "magma", 0.0, dmax),
        ]
        save_panel_grid(
            detail_dir / f"{idx:02d}_{stem}.png",
            f"{idx:02d} {stem} | mask={mask.mean():.3f} | depth scale {vmin:.2f}-{vmax:.2f} m",
            panels,
            cols=5,
        )

        raw_rgb = draw_mask_boundary(colorize_depth(raw, vmin, vmax), mask)
        mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        ns_rgb = draw_mask_boundary(colorize_depth(opencv_ns, vmin, vmax), mask)
        ours_rgb = draw_mask_boundary(colorize_depth(ours_hole, vmin, vmax), mask)
        prop_rgb = draw_mask_boundary(colorize_depth(propainter, vmin, vmax), mask)
        cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_raw.png"), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_opencv_ns.png"), cv2.cvtColor(ns_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_ours_hole_only.png"), cv2.cvtColor(ours_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(quick_dir / f"{idx:02d}_{stem}_propainter.png"), cv2.cvtColor(prop_rgb, cv2.COLOR_RGB2BGR))

        contact_rows.append(
            {
                "stem": f"{idx:02d}_{stem}",
                "raw": raw_rgb,
                "mask": mask_rgb,
                "opencv_ns": ns_rgb,
                "ours": ours_rgb,
                "propainter": prop_rgb,
            }
        )
        summary.append(
            {
                "frame_index": idx,
                "source_stem": stem,
                "detail_png": str(detail_dir / f"{idx:02d}_{stem}.png"),
                "mask_ratio": float(mask.mean()),
                "depth_vmin": float(vmin),
                "depth_vmax": float(vmax),
                "propainter_path": str(propainter_path),
            }
        )

    save_contact_sheet(out_dir / "contact_sheet.png", contact_rows)
    with open(out_dir / "visualization_summary.json", "w") as f:
        json.dump(
            {
                "case_dir": str(case_dir),
                "propainter_dir": str(propainter_dir),
                "output_dir": str(out_dir),
                "detail_dir": str(detail_dir),
                "quick_png_dir": str(quick_dir),
                "contact_sheet": str(out_dir / "contact_sheet.png"),
                "frames": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved per-frame visualizations to {detail_dir}")
    print(f"Saved quick PNGs to {quick_dir}")
    print(f"Saved contact sheet to {out_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
