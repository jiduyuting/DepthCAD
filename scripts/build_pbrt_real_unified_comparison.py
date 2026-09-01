#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("ours anti-forgetting", "hole_only", "ours_antiforget"),
    ("depth-only flow", "methods/depth_only_flow/hole_only", "depth_only"),
    ("PBRT e40", "methods/pbrt_e40/hole_only", "pbrt_e40"),
    ("PBRT replay e30", "methods/pbrt_replay_e30/hole_only", "pbrt_replay_e30"),
    ("PBRT flipLR e30", "methods/pbrt_replay_fliplr_e30/hole_only", "pbrt_fliplr_e30"),
    ("PBRT boundary v2", "methods/pbrt_boundary_v2_e15/hole_only", "pbrt_boundary_v2"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a unified PBRT Real multi-checkpoint comparison.")
    parser.add_argument(
        "--case_dir",
        type=Path,
        default=Path("output/pbrt_real_new_selection/masked_realholes_antiforget_v2b_fullshape_c64"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/unified_visualizations/pbrt_real_fullshape_c64_aligned"),
    )
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def load_array(path):
    path = Path(path)
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def depth_limits(arrays, mask):
    values = []
    for array in arrays:
        if array is None:
            continue
        valid = np.isfinite(array) & (array > 0) & mask
        if valid.any():
            values.append(array[valid])
    if not values:
        return 0.0, 1.0
    values = np.concatenate(values)
    low, high = np.percentile(values, [2.0, 98.0])
    return float(low), max(float(high), float(low) + 1.0)


def finite_mae(prediction, target, mask):
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        return None
    return float(np.mean(np.abs(prediction[valid] - target[valid])))


def save_figure(path, title, panels, cols):
    rows = int(math.ceil(len(panels) / float(cols)))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.4 * cols, 3.5 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    depth_image = None
    for axis, (panel_title, image, cmap, vmin, vmax) in zip(axes, panels):
        depth_image = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title, fontsize=9)
        axis.axis("off")
        if cmap == "gray":
            depth_image = None
    for axis in axes[len(panels) :]:
        axis.axis("off")
    if depth_image is not None:
        fig.colorbar(depth_image, ax=axes.tolist(), fraction=0.025, pad=0.01, label="depth (m)")
    fig.suptitle(title, fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    case_dir = args.case_dir
    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((case_dir / "summary.json").read_text())
    rows = []

    samples = [f"{row['name']}_r{int(row.get('repeat', 0)):02d}" for row in summary["per_sample"]]
    for sample in samples:
        gt = load_array(case_dir / "per_sample_missing" / f"{sample}_gt.npy")
        if gt is None:
            row = next(item for item in summary["per_sample"] if f"{item['name']}_r{int(item.get('repeat', 0)):02d}" == sample)
            gt = load_array(row["depth_path"])
        corrupted = load_array(case_dir / "corrupted" / f"{sample}_corrupted.npy")
        artificial_mask = load_array(case_dir / "mask" / f"{sample}_mask.npy").astype(bool)
        condition_mask = load_array(case_dir / "condition_mask" / f"{sample}_condition_mask.npy").astype(bool)
        anchor = load_array(case_dir / "anchor" / f"{sample}_anchor.npy")
        method_arrays = {}
        for label, relative_dir, key in METHODS:
            suffix = "_corrupted_hole_only.npy" if key == "depth_only" else "_hole_only.npy"
            method_arrays[key] = load_array(case_dir / relative_dir / f"{sample}{suffix}")

        vmin, vmax = depth_limits(
            [gt, corrupted, anchor] + list(method_arrays.values()),
            ~condition_mask,
        )
        panels = [
            ("pseudo-GT", gt, "viridis", vmin, vmax),
            ("masked input", corrupted, "viridis", vmin, vmax),
            ("artificial mask", artificial_mask.astype(np.float32), "gray", 0.0, 1.0),
            ("condition holes", condition_mask.astype(np.float32), "gray", 0.0, 1.0),
            ("NS anchor", anchor, "viridis", vmin, vmax),
        ]
        for label, _relative_dir, key in METHODS:
            if method_arrays[key] is not None:
                panels.append((label, method_arrays[key], "viridis", vmin, vmax))
        figure_path = figure_dir / f"{sample}.png"
        save_figure(figure_path, sample, panels, args.cols)

        metrics = {key: finite_mae(value, gt, artificial_mask) for key, value in method_arrays.items()}
        metrics["anchor"] = finite_mae(anchor, gt, artificial_mask)
        rows.append(
            {
                "sample": sample,
                "figure": str(figure_path),
                "mask_pixel_count": int(artificial_mask.sum()),
                "condition_hole_ratio": float(condition_mask.mean()),
                **{f"{key}_mask_mae": value for key, value in metrics.items()},
            }
        )

    with (output_dir / "per_sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_keys = ["anchor"] + [key for _label, _dir, key in METHODS]
    aggregate = {}
    for key in metric_keys:
        weighted = [(row[f"{key}_mask_mae"], row["mask_pixel_count"]) for row in rows if row[f"{key}_mask_mae"] is not None]
        count = sum(weight for _value, weight in weighted)
        aggregate[f"{key}_mask_mae"] = sum(value * weight for value, weight in weighted) / count if count else None
        aggregate[f"{key}_mask_mae_count"] = count
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "benchmark": "pbrt_real_fullshape_c64_unified_comparison",
                "case_dir": str(case_dir),
                "num_figures": len(rows),
                "samples": samples,
                "methods": [{"label": label, "directory": str(case_dir / relative_dir)} for label, relative_dir, _key in METHODS],
                "aggregate": aggregate,
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"Saved {len(rows)} unified figures to {figure_dir}")


if __name__ == "__main__":
    main()
