#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize PBRT gt_depth and raw 9-channel gt arrays directly from source files."
    )
    parser.add_argument("--gt_depth_root", type=Path, default=Path("/data/pre_student/hcy/pbrt/gt_depth"))
    parser.add_argument("--gt_root", type=Path, default=Path("/data/pre_student/hcy/pbrt/gt"))
    parser.add_argument("--split", type=Path, default=Path("output/pbrt100_depth_completion/split.json"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/pbrt_gt_source_visualization"),
    )
    return parser.parse_args()


def load_split(path):
    payload = json.loads(path.read_text())
    samples = payload.get("test")
    if samples is None and isinstance(payload.get("samples"), dict):
        samples = payload["samples"].get("test")
    if not samples:
        raise ValueError(f"No test samples found in {path}")
    return list(samples)


def depth_stats(depth):
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.1) & (depth < 9.9)
    values = depth[finite]
    valid_values = depth[valid]
    return {
        "zero_ratio": float(np.mean(depth == 0.0)),
        "invalid_ratio": float(np.mean(~valid)),
        "finite_min_m": float(values.min()) if values.size else None,
        "finite_max_m": float(values.max()) if values.size else None,
        "valid_p02_m": float(np.percentile(valid_values, 2)) if valid_values.size else None,
        "valid_p98_m": float(np.percentile(valid_values, 98)) if valid_values.size else None,
    }, valid


def select_samples(samples, gt_depth_root):
    grouped = {}
    stats_by_sample = {}
    for sample in samples:
        depth = np.load(gt_depth_root / f"{sample}.npy").astype(np.float32)
        stats, _ = depth_stats(depth)
        stats_by_sample[sample] = stats
        grouped.setdefault(sample.split("/", 1)[0], []).append(sample)

    selected = []
    for scene in sorted(grouped):
        ordered = sorted(grouped[scene], key=lambda sample: (stats_by_sample[sample]["invalid_ratio"], sample))
        selected.append(ordered[0])
        if ordered[-1] != ordered[0]:
            selected.append(ordered[-1])
    return selected, stats_by_sample


def depth_limits(depth, valid):
    values = depth[valid]
    if not values.size:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def show_raw_channel(axis, channel, title):
    finite = channel[np.isfinite(channel)]
    scale = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    scale = max(scale, 1e-6)
    axis.imshow(channel, cmap="RdBu_r", vmin=-scale, vmax=scale)
    axis.set_title(f"{title} | +/-{scale:.2f}", fontsize=9)
    axis.axis("off")


def save_detail(path, sample, depth, raw_gt, valid, stats):
    vmin, vmax = depth_limits(depth, valid)
    fig, axes = plt.subplots(4, 3, figsize=(12, 15), constrained_layout=True)
    axes = axes.ravel()

    depth_image = axes[0].imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"gt_depth raw | scale {vmin:.2f}-{vmax:.2f} m")
    axes[0].axis("off")
    fig.colorbar(depth_image, ax=axes[0], fraction=0.046, pad=0.04, label="Depth (m)")

    axes[1].imshow(~valid, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Invalid mask | {stats['invalid_ratio']:.2%}")
    axes[1].axis("off")

    amplitude = np.sqrt(raw_gt[[1, 4, 7]] ** 2 + raw_gt[[0, 3, 6]] ** 2).mean(axis=0)
    amp_hi = max(float(np.percentile(amplitude[np.isfinite(amplitude)], 99)), 1e-6)
    axes[2].imshow(amplitude, cmap="magma", vmin=0, vmax=amp_hi)
    axes[2].set_title(f"Raw gt mean amplitude | p99={amp_hi:.2f}")
    axes[2].axis("off")

    for index in range(9):
        show_raw_channel(axes[index + 3], raw_gt[index], f"gt channel {index}")

    fig.suptitle(
        f"{sample} | gt_depth zero={stats['zero_ratio']:.2%} "
        f"| invalid={stats['invalid_ratio']:.2%} | raw gt shape={tuple(raw_gt.shape)}"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_overview(path, rows):
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.2 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None]
    for row_axes, row in zip(axes, rows):
        depth = row["depth"]
        raw_gt = row["raw_gt"]
        valid = row["valid"]
        stats = row["stats"]
        vmin, vmax = depth_limits(depth, valid)

        row_axes[0].imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
        row_axes[0].set_title(
            f"{row['sample']}\ngt_depth raw | zero={stats['zero_ratio']:.1%} | invalid={stats['invalid_ratio']:.1%}",
            fontsize=9,
        )
        row_axes[1].imshow(~valid, cmap="gray", vmin=0, vmax=1)
        row_axes[1].set_title("Invalid depth mask", fontsize=9)
        show_raw_channel(row_axes[2], raw_gt[0], "gt channel 0")

        amplitude = np.sqrt(raw_gt[[1, 4, 7]] ** 2 + raw_gt[[0, 3, 6]] ** 2).mean(axis=0)
        amp_hi = max(float(np.percentile(amplitude[np.isfinite(amplitude)], 99)), 1e-6)
        row_axes[3].imshow(amplitude, cmap="magma", vmin=0, vmax=amp_hi)
        row_axes[3].set_title(f"Raw gt mean amplitude | p99={amp_hi:.1f}", fontsize=9)
        row_axes[3].axis("off")
        for axis in row_axes[:2]:
            axis.axis("off")
    fig.suptitle("PBRT source arrays: direct gt_depth and gt visualization")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = args.output_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    samples = load_split(args.split)
    selected, stats_by_sample = select_samples(samples, args.gt_depth_root)
    rows = []
    csv_rows = []
    for sample in selected:
        depth_path = args.gt_depth_root / f"{sample}.npy"
        gt_path = args.gt_root / f"{sample}.npy"
        depth = np.load(depth_path).astype(np.float32)
        raw_gt = np.load(gt_path).astype(np.float32)
        if raw_gt.shape[0] != 9 or raw_gt.shape[1:] != depth.shape:
            raise ValueError(f"Shape mismatch for {sample}: depth={depth.shape}, gt={raw_gt.shape}")
        stats, valid = depth_stats(depth)
        row = {"sample": sample, "depth": depth, "raw_gt": raw_gt, "valid": valid, "stats": stats}
        rows.append(row)

        detail_path = detail_dir / f"{sample.replace('/', '_')}.png"
        save_detail(detail_path, sample, depth, raw_gt, valid, stats)
        csv_rows.append(
            {
                "sample": sample,
                **stats,
                "gt_depth_path": str(depth_path),
                "gt_path": str(gt_path),
                "detail_figure": str(detail_path),
            }
        )

    save_overview(args.output_dir / "overview.png", rows)
    with (args.output_dir / "stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "gt_depth_root": str(args.gt_depth_root),
                "gt_root": str(args.gt_root),
                "split": str(args.split),
                "selection": "lowest and highest invalid-depth ratio per scene",
                "samples": csv_rows,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(rows)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
