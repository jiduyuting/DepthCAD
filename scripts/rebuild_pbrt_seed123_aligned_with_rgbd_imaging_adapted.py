#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_SOURCE_DIR = Path("output/unified_visualizations/pbrt_seed123_aligned")
DEFAULT_BACKUP_FIGURES = DEFAULT_SOURCE_DIR / "figures_before_rgbd_adapted"
DEFAULT_RGBD_EVAL = Path("output/rgbd_imaging_adapted_depth_iq_amp_seed123/eval_pbrt_seed123_aligned8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add the local IQ+Depth+Amp proxy baseline panels to pbrt_seed123_aligned unified figures."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--source_figures_dir", type=Path, default=DEFAULT_BACKUP_FIGURES)
    parser.add_argument("--rgbd_eval_dir", type=Path, default=DEFAULT_RGBD_EVAL)
    parser.add_argument("--cols", type=int, default=4)
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


def save_panel_figure(path, title, panels, cols):
    rows = int(math.ceil(len(panels) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.9 * rows), constrained_layout=True)
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


def rgbd_vis_path(rgbd_eval_dir, sample):
    return Path(rgbd_eval_dir) / "visualizations" / f"vis_{sample.replace('/', '_')}.png"


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    summary_path = source_dir / "summary.json"
    summary = load_json(summary_path)
    rgbd_summary_path = Path(args.rgbd_eval_dir) / "summary.json"
    rgbd_summary = load_json(rgbd_summary_path)

    updated_rows = []
    included = 0
    for row in summary.get("rows", []):
        sample = row["sample"]
        figure_path = Path(row["figure"])
        source_figure = Path(args.source_figures_dir) / figure_path.name
        if not source_figure.exists():
            source_figure = figure_path
        if not source_figure.exists():
            raise FileNotFoundError(source_figure)

        rgbd_path = rgbd_vis_path(args.rgbd_eval_dir, sample)
        if not rgbd_path.exists():
            raise FileNotFoundError(rgbd_path)

        panels = [
            ("GT", crop_grid_panel(source_figure, 3, 3, 0, 0)),
            ("Noisy", crop_grid_panel(source_figure, 3, 3, 0, 1)),
            ("Hole Mask", crop_grid_panel(source_figure, 3, 3, 0, 2)),
            ("Anchor", crop_grid_panel(source_figure, 3, 3, 1, 0)),
            ("Ours", crop_grid_panel(source_figure, 3, 3, 1, 1)),
            ("BayesToF (approx)", crop_grid_panel(source_figure, 3, 3, 1, 2)),
            ("IQ+Depth+Amp U-Net proxy", crop_grid_panel(rgbd_path, 3, 4, 1, 0)),
            ("DepthCAD", crop_grid_panel(source_figure, 3, 3, 2, 0)),
            ("ProPainter", crop_grid_panel(source_figure, 3, 3, 2, 1)),
            ("LFRD2", crop_grid_panel(source_figure, 3, 3, 2, 2)),
        ]
        save_panel_figure(figure_path, sample, panels, cols=int(args.cols))

        updated = dict(row)
        updated["rgbd_imaging_adapted_visualization"] = str(rgbd_path)
        updated["rgbd_imaging_adapted_included"] = True
        updated_rows.append(updated)
        included += 1
        print(f"[ok] {sample} -> {figure_path}")

    updated_summary = dict(summary)
    updated_summary["rows"] = updated_rows
    updated_summary["rgbd_imaging_adapted"] = {
        "label": "IQ+Depth+Amp U-Net proxy",
        "note": (
            "Proxy baseline using the available DepthCAD IQ/depth/amplitude cache and "
            "the local depth_iq_amp checkpoint. This is not an output from the "
            "/data/pre_student/GJ/RGBD_imaging repository, not the original paper checkpoint, "
            "and not the RGB-D discriminator training setup from the IJCAI paper."
        ),
        "eval_dir": str(Path(args.rgbd_eval_dir).resolve()),
        "summary": str(rgbd_summary_path.resolve()),
        "checkpoint": rgbd_summary.get("checkpoint"),
        "input_mode": rgbd_summary.get("input_mode"),
        "aggregate": rgbd_summary.get("aggregate"),
        "included": included,
    }
    save_json(summary_path, updated_summary)
    print(f"Saved RGBD-Imaging adapted unified figures under {source_dir / 'figures'}")


if __name__ == "__main__":
    main()
