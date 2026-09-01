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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_depth_postprocess import opencv_depth_inpaint


DEFAULT_SOURCE_DIR = Path("output/unified_visualizations/pbrt_seed123_aligned")
DEFAULT_BAYESTOF_ROOT = Path("output/approx_bayestof_cache_n100_nt5000")
DEFAULT_PROPAINTER_ROOT = Path("output/pbrt_propainter_seed123/propainter_run/restored_by_stem")
DEFAULT_LFRD2_ROOT = Path("/data/pre_student/hcy/LFRD2/results/pbrt/depth")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the pbrt_seed123_aligned unified figures and add BayesToF "
            "without requiring torch."
        )
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--bayestof_root", type=Path, default=DEFAULT_BAYESTOF_ROOT)
    parser.add_argument("--propainter_root", type=Path, default=DEFAULT_PROPAINTER_ROOT)
    parser.add_argument("--lfrd2_root", type=Path, default=DEFAULT_LFRD2_ROOT)
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def render_depth_rgb(depth, vmin, vmax):
    depth = np.asarray(depth, dtype=np.float32)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((depth - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = plt.get_cmap("turbo")
    rgb = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    invalid = ~np.isfinite(depth)
    rgb[invalid] = 255
    return rgb


def render_mask_rgb(mask):
    mask = np.asarray(mask).astype(bool)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[mask] = 255
    return rgb


def depth_limits(*arrays):
    values = []
    for arr in arrays:
        if arr is None:
            continue
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


def save_panel_figure(path, title, panels, cols=3):
    rows = int(math.ceil(len(panels) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), constrained_layout=True)
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


def load_bayestof_map(root):
    root = Path(root)
    summary_path = root / "summary.json"
    mapping = {}
    if summary_path.exists():
        summary = load_json(summary_path)
        for row in summary.get("rows", []):
            sample = row.get("sample_name")
            path = row.get("depth_path")
            if sample and path:
                mapping[sample] = Path(path)
    if mapping:
        return mapping, summary_path

    depth_dir = root / "depth"
    for path in depth_dir.rglob("*_approx_bayestof.npy"):
        rel = path.relative_to(depth_dir)
        parts = list(rel.parts)
        parts[-1] = parts[-1].replace("_approx_bayestof.npy", "")
        mapping["/".join(parts)] = path
    return mapping, summary_path if summary_path.exists() else None


def load_propainter_path(root, sample):
    sample_stem = sample.replace("/", "_")
    path = Path(root) / f"{sample_stem}_propainter_restored.npy"
    if path.exists():
        return path
    return None


def load_lfrd2_path(root, sample):
    scene, idx, stem = sample.split("/")
    path = Path(root) / scene / idx / f"{stem}.npy"
    if path.exists():
        return path
    return None


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "figures")

    source_summary_path = source_dir / "summary.json"
    if not source_summary_path.exists():
        raise FileNotFoundError(source_summary_path)
    source_summary = load_json(source_summary_path)
    rows = source_summary.get("rows", [])

    bayestof_map, bayestof_summary_path = load_bayestof_map(args.bayestof_root)
    updated_rows = []

    for row in rows:
        sample = row["sample"]
        figure_path = Path(row["figure"])
        cache_path = Path(row["cache_path"])
        if not figure_path.exists():
            raise FileNotFoundError(figure_path)
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)

        with np.load(cache_path, allow_pickle=False) as data:
            gt = data["gt_depth"].astype(np.float32)
            noisy = data["depth_noisy"].astype(np.float32)
            hole = data["hole_mask"] > 0.5
        anchor = opencv_depth_inpaint(noisy, hole, method="ns", radius=15).astype(np.float32)

        bayestof_path = bayestof_map.get(sample)
        bayestof = np.load(bayestof_path).astype(np.float32) if bayestof_path and Path(bayestof_path).exists() else None
        prop_path = load_propainter_path(args.propainter_root, sample)
        prop = np.load(prop_path).astype(np.float32) if prop_path else None
        lfrd2_path = load_lfrd2_path(args.lfrd2_root, sample)
        lfrd2 = np.load(lfrd2_path).astype(np.float32) if lfrd2_path else None
        if lfrd2 is not None and lfrd2.shape != gt.shape:
            lfrd2 = cv2.resize(lfrd2, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        vmin, vmax = depth_limits(gt, noisy, anchor, prop, lfrd2, bayestof)

        panels = [
            ("GT", crop_grid_panel(figure_path, 2, 4, 0, 0)),
            ("Noisy", crop_grid_panel(figure_path, 2, 4, 0, 1)),
            ("Hole Mask", crop_grid_panel(figure_path, 2, 4, 0, 2)),
            ("Anchor", crop_grid_panel(figure_path, 2, 4, 0, 3)),
            ("Ours", crop_grid_panel(figure_path, 2, 4, 1, 0)),
            (
                "BayesToF (approx)",
                render_depth_rgb(bayestof, vmin, vmax) if bayestof is not None else crop_grid_panel(figure_path, 2, 4, 1, 1),
            ),
            ("DepthCAD", crop_grid_panel(figure_path, 2, 4, 1, 1)),
            ("ProPainter", crop_grid_panel(figure_path, 2, 4, 1, 2)),
            ("LFRD2", crop_grid_panel(figure_path, 2, 4, 1, 3)),
        ]

        save_panel_figure(figure_path, sample, panels, cols=3)
        row = dict(row)
        row["bayestof_depth_path"] = str(bayestof_path) if bayestof_path else None
        row["bayestof_included"] = bayestof is not None
        updated_rows.append(row)
        print(f"[ok] {sample} -> {figure_path}")

    updated_summary = dict(source_summary)
    updated_summary["bayestof_root"] = str(Path(args.bayestof_root).resolve())
    updated_summary["bayestof_summary"] = str(bayestof_summary_path.resolve()) if bayestof_summary_path else None
    updated_summary["bayestof_coverage"] = {
        "available": len(bayestof_map),
        "included": sum(1 for row in updated_rows if row.get("bayestof_included")),
    }
    updated_summary["rows"] = updated_rows
    save_json(source_summary_path, updated_summary)

    print(f"Saved BayesToF-augmented unified figures to {output_dir}")


if __name__ == "__main__":
    main()
