#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build aligned figures for PBRT Real artificial real-hole-shape masked self-test outputs."
    )
    parser.add_argument(
        "--case_dir",
        type=Path,
        default=Path("output/pbrt_real_new_selection/masked_realholes_antiforget_v2b_c64"),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--depth_only_dir", type=Path, default=None)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    return parser.parse_args()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def depth_to_meters(depth):
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return depth
    if float(np.percentile(finite, 95.0)) > 30.0:
        return depth / 1000.0
    return depth


def finite_mae(pred, target, mask):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(target)
    if int(valid.sum()) == 0:
        return None, 0
    return float(np.mean(np.abs(pred[valid] - target[valid]))), int(valid.sum())


def depth_limits(arrays, mask=None):
    values = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        valid = np.isfinite(arr) & (arr > 0)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        vals = arr[valid]
        if vals.size:
            values.append(vals)
    if not values:
        return 0.0, 1.0
    values = np.concatenate(values)
    lo, hi = np.percentile(values, [2.0, 98.0])
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def add_panel(ax, title, image, cmap, vmin, vmax):
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def save_figure(path, title, panels, cols=4):
    rows = int(math.ceil(len(panels) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.5 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    last_depth = None
    for ax, panel in zip(axes, panels):
        im = add_panel(ax, *panel)
        if panel[2] != "gray":
            last_depth = im
    for ax in axes[len(panels) :]:
        ax.axis("off")
    if last_depth is not None:
        fig.colorbar(last_depth, ax=axes.tolist(), fraction=0.025, pad=0.01, label="depth (m)")
    fig.suptitle(title, fontsize=12)
    mkdir(Path(path).parent)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def maybe_load(path):
    path = Path(path)
    if not path.exists():
        return None
    return np.load(path).astype(np.float32)


def main():
    args = parse_args()
    case_dir = Path(args.case_dir)
    out_dir = Path(args.output_dir) if args.output_dir else case_dir / "aligned_comparison"
    depth_only_dir = Path(args.depth_only_dir) if args.depth_only_dir else case_dir / "methods" / "depth_only_flow"
    mkdir(out_dir / "figures")
    mkdir(out_dir / "error_figures")

    summary = load_json(case_dir / "summary.json")
    rows = []
    for row in summary.get("per_sample", []):
        sample = str(row["name"])
        repeat = int(row.get("repeat", 0))
        key = f"{sample}_r{repeat:02d}"
        depth_only_key = f"{key}_corrupted"

        gt = depth_to_meters(np.load(row["depth_path"]).astype(np.float32))
        corrupted = maybe_load(case_dir / "corrupted" / f"{key}_corrupted.npy")
        mask = np.load(case_dir / "mask" / f"{key}_mask.npy").astype(bool)
        condition_mask = np.load(case_dir / "condition_mask" / f"{key}_condition_mask.npy").astype(bool)
        anchor = maybe_load(case_dir / "anchor" / f"{key}_anchor.npy")
        ours = maybe_load(case_dir / "hole_only" / f"{key}_hole_only.npy")
        depth_only = maybe_load(depth_only_dir / "hole_only" / f"{depth_only_key}_hole_only.npy")

        vmin, vmax = depth_limits([gt, corrupted, anchor, ours, depth_only], mask=~condition_mask)
        if args.vmin is not None:
            vmin = float(args.vmin)
        if args.vmax is not None:
            vmax = float(args.vmax)
        if vmax <= vmin:
            vmax = vmin + 1.0

        panels = [
            ("pseudo-GT", gt, "viridis", vmin, vmax),
            ("masked input", corrupted, "viridis", vmin, vmax),
            ("artificial mask", mask.astype(np.float32), "gray", 0.0, 1.0),
            ("condition holes", condition_mask.astype(np.float32), "gray", 0.0, 1.0),
            ("NS anchor", anchor, "viridis", vmin, vmax),
            ("ours anti-forgetting", ours, "viridis", vmin, vmax),
        ]
        if depth_only is not None:
            panels.append(("depth-only flow", depth_only, "viridis", vmin, vmax))
        figure = out_dir / "figures" / f"{key}.png"
        save_figure(figure, key, panels)

        err_values = []
        for arr in [anchor, ours, depth_only]:
            if arr is not None:
                err_values.append(np.abs(arr - gt)[mask & np.isfinite(arr) & np.isfinite(gt)])
        err_values = [v.ravel() for v in err_values if v.size]
        err_max = float(np.percentile(np.concatenate(err_values), 98.0)) if err_values else 1.0
        err_max = max(err_max, 1e-6)
        err_panels = [
            ("artificial mask", mask.astype(np.float32), "gray", 0.0, 1.0),
            ("NS anchor error", np.where(mask, np.abs(anchor - gt), np.nan), "magma", 0.0, err_max),
            ("ours error", np.where(mask, np.abs(ours - gt), np.nan), "magma", 0.0, err_max),
        ]
        if depth_only is not None:
            err_panels.append(
                ("depth-only error", np.where(mask, np.abs(depth_only - gt), np.nan), "magma", 0.0, err_max)
            )
        error_figure = out_dir / "error_figures" / f"{key}.png"
        save_figure(error_figure, f"{key} mask error", err_panels)

        anchor_mae, mask_count = finite_mae(anchor, gt, mask)
        ours_mae, _ = finite_mae(ours, gt, mask)
        depth_only_mae, _ = finite_mae(depth_only, gt, mask) if depth_only is not None else (None, 0)
        rows.append(
            {
                "sample": sample,
                "sample_key": key,
                "mask_ratio_actual": float(mask.sum() / max(int(row.get("reliable_pixel_count", 0)), 1)),
                "mask_pixel_count": int(mask.sum()),
                "condition_hole_ratio": float(condition_mask.mean()),
                "anchor_mask_mae": anchor_mae,
                "ours_antiforget_mask_mae": ours_mae,
                "depth_only_mask_mae": depth_only_mae,
                "mask_count": mask_count,
                "figure": str(figure),
                "error_figure": str(error_figure),
            }
        )

    metric_keys = ["anchor_mask_mae", "ours_antiforget_mask_mae", "depth_only_mask_mae"]
    aggregate = {}
    for key in metric_keys:
        weighted = [(r[key], r["mask_count"]) for r in rows if r.get(key) is not None and r["mask_count"] > 0]
        if weighted:
            total_count = sum(count for _value, count in weighted)
            aggregate[key] = float(sum(value * count for value, count in weighted) / total_count)
            aggregate[f"{key}_count"] = int(total_count)
        else:
            aggregate[key] = None
            aggregate[f"{key}_count"] = 0
    if aggregate["anchor_mask_mae"] and aggregate["ours_antiforget_mask_mae"] is not None:
        aggregate["ours_improve_vs_anchor"] = (
            aggregate["anchor_mask_mae"] - aggregate["ours_antiforget_mask_mae"]
        ) / aggregate["anchor_mask_mae"]
    if aggregate["anchor_mask_mae"] and aggregate["depth_only_mask_mae"] is not None:
        aggregate["depth_only_improve_vs_anchor"] = (
            aggregate["anchor_mask_mae"] - aggregate["depth_only_mask_mae"]
        ) / aggregate["anchor_mask_mae"]

    with (out_dir / "per_sample_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "benchmark": "pbrt_real_selected_artificial_realhole_masks",
                "case_dir": str(case_dir),
                "ours_checkpoint": summary.get("checkpoint"),
                "depth_only_dir": str(depth_only_dir),
                "num_samples": len(rows),
                "aggregate": aggregate,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(json.dumps({"output_dir": str(out_dir), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
