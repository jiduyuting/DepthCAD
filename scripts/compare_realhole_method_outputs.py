#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_depth_postprocess import opencv_depth_inpaint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build side-by-side comparisons for real-hole depth restoration outputs."
    )
    parser.add_argument(
        "--depth_dir",
        type=Path,
        default=Path("/data/pre_student/hcy/datasets/pbrt/Real/depth"),
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=Path("/data/pre_student/hcy/datasets/pbrt/Real/noise_masks"),
    )
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("output/realhole_method_comparison"))
    parser.add_argument("--hole_depth_threshold", type=float, default=0.0)
    parser.add_argument("--depth_unit", choices=["auto", "m", "mm"], default="auto")
    parser.add_argument("--anchor_radius", type=int, default=15)
    parser.add_argument("--depth_vis_min", type=float, default=0.5)
    parser.add_argument("--depth_vis_max", type=float, default=4.5)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method spec in the form name=path. Repeatable.",
    )
    return parser.parse_args()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def natural_key(text):
    stem = Path(text).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (0, int(digits)) if digits else (1, stem)


def depth_to_meters(depth, unit="auto"):
    depth = np.asarray(depth, dtype=np.float32)
    unit = str(unit)
    if unit == "m":
        return depth
    if unit == "mm":
        return depth / 1000.0
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return depth
    median = float(np.median(finite))
    p95 = float(np.percentile(finite, 95.0))
    if median > 20.0 or p95 > 100.0:
        return depth / 1000.0
    return depth


def load_depths(depth_dir):
    paths = sorted(Path(depth_dir).glob("*.npy"), key=lambda p: natural_key(p.name))
    if not paths:
        raise FileNotFoundError(f"No depth .npy files under {depth_dir}")
    return {p.stem: p for p in paths}


def load_mask(mask_dir, sample, depth):
    if mask_dir is not None and Path(mask_dir).is_dir():
        candidates = [
            Path(mask_dir) / f"{sample}_overall_missing_mask.npy",
            Path(mask_dir) / f"{sample}_hole_mask.npy",
            Path(mask_dir) / f"{sample}_mask.npy",
            Path(mask_dir) / f"{sample}.npy",
        ]
        for path in candidates:
            if path.exists():
                mask = np.load(path).astype(bool)
                if mask.shape == depth.shape:
                    return mask, str(path)
    mask = (~np.isfinite(depth)) | (depth <= float(args.hole_depth_threshold))
    return mask.astype(bool), "depth_threshold"


def find_method_file(method_root, sample):
    root = Path(method_root)
    if not root.exists():
        return None

    direct_names = [
        f"{sample}_restored.npy",
        f"{sample}_hole_only.npy",
        f"{sample}_pred.npy",
        f"{sample}_depthcad_depth_pred.npy",
        f"{sample}_depthcad_depth_hole_only.npy",
        f"{sample}_lfrd2.npy",
        f"{sample}_lfrd2_hole_only.npy",
        f"{sample}_propainter_restored.npy",
    ]
    subdirs = [
        root,
        root / "hole_only",
        root / "restored",
        root / "pred",
        root / "pred_depth",
        root / "restored_by_stem",
    ]
    for subdir in subdirs:
        if not subdir.is_dir():
            continue
        for name in direct_names:
            path = subdir / name
            if path.exists():
                return path
        matches = sorted(subdir.glob(f"{sample}*.npy"), key=lambda p: natural_key(p.name))
        if matches:
            return matches[0]
    matches = sorted(root.rglob(f"{sample}*.npy"), key=lambda p: natural_key(p.name))
    return matches[0] if matches else None


def image_limits(arrays, hole):
    vals = []
    keep = ~hole
    for arr in arrays:
        if arr is None:
            continue
        v = np.asarray(arr, dtype=np.float32)[keep & np.isfinite(arr)]
        if v.size:
            vals.append(v)
    if not vals:
        return 0.0, 1.0
    values = np.concatenate(vals)
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_panel(ax, title, image, cmap, vmin, vmax):
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if cmap != "gray":
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def save_figure(path, sample, raw, hole, panels, anchor_radius, depth_vis_min, depth_vis_max):
    anchor = opencv_depth_inpaint(raw, hole, method="ns", radius=int(anchor_radius)).astype(np.float32)
    depth_panels = [raw, anchor] + [image for _name, image in panels if image is not None]
    vmin, vmax = image_limits(depth_panels, hole)
    vmin = float(depth_vis_min if depth_vis_min is not None else vmin)
    vmax = float(depth_vis_max if depth_vis_max is not None else vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0

    figure_panels = [
        ("raw depth", np.where(hole, np.nan, raw), "viridis", vmin, vmax),
        ("hole mask", hole.astype(np.float32), "gray", 0.0, 1.0),
        (f"NS anchor r{int(anchor_radius)}", anchor, "viridis", vmin, vmax),
    ]
    figure_panels.extend((name, image, "viridis", vmin, vmax) for name, image in panels if image is not None)

    cols = 4
    rows = int(math.ceil(len(figure_panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[len(figure_panels) :]:
        ax.axis("off")
    for ax, (title, image, cmap, lo, hi) in zip(axes, figure_panels):
        add_panel(ax, title, image, cmap, lo, hi)
    fig.suptitle(sample, fontsize=12)
    mkdir(Path(path).parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    global args
    args = parse_args()
    mkdir(args.output_dir)

    depth_map = load_depths(args.depth_dir)
    samples = list(depth_map.keys())
    if args.samples:
        wanted = {str(s) for s in args.samples}
        samples = [s for s in samples if s in wanted]
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise FileNotFoundError("No matching samples found.")

    methods = []
    for spec in args.method:
        if "=" not in spec:
            raise ValueError("--method must be formatted as name=path")
        name, path = spec.split("=", 1)
        methods.append((name, Path(path)))

    rows = []
    figure_paths = []
    for sample in samples:
        raw = depth_to_meters(np.load(depth_map[sample]).astype(np.float32), args.depth_unit)
        hole, hole_source = load_mask(args.mask_dir, sample, raw)
        method_panels = []
        for name, root in methods:
            path = find_method_file(root, sample)
            if path is None:
                continue
            pred = np.load(path).astype(np.float32)
            if pred.shape != raw.shape:
                continue
            method_panels.append((name, np.where(hole, pred, raw).astype(np.float32), str(path)))

        save_figure(
            args.output_dir / "figures" / f"{sample}.png",
            sample,
            raw,
            hole,
            [(name, image) for name, image, _path in method_panels],
            args.anchor_radius,
            args.depth_vis_min,
            args.depth_vis_max,
        )
        figure_paths.append(str(args.output_dir / "figures" / f"{sample}.png"))
        rows.append(
            {
                "sample": sample,
                "depth_path": str(depth_map[sample]),
                "hole_source": hole_source,
                "hole_ratio": float(hole.mean()),
                "methods": [
                    {"name": name, "path": path}
                    for name, _image, path in method_panels
                ],
            }
        )

    summary = {
        "depth_dir": str(Path(args.depth_dir).resolve()),
        "mask_dir": str(Path(args.mask_dir).resolve()) if args.mask_dir else None,
        "samples": samples,
        "methods": [{"name": name, "path": str(path.resolve())} for name, path in methods],
        "figures_dir": str((args.output_dir / "figures").resolve()),
        "rows": rows,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "figures.txt").open("w") as f:
        for path in figure_paths:
            f.write(path + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
