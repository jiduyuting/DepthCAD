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
        description="Build method comparison figures for the newly captured depth/IQ data."
    )
    parser.add_argument("--depth_dir", type=Path, default=Path("data/prepared_new_capture/all/depth_m"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/new_capture_method_comparison"))
    parser.add_argument("--vmin", type=float, default=0.5)
    parser.add_argument("--vmax", type=float, default=4.5)
    parser.add_argument("--valid_min", type=float, default=0.1)
    parser.add_argument("--valid_max", type=float, default=4.5)
    return parser.parse_args()


METHODS = [
    {
        "key": "ns_anchor",
        "title": "NS anchor",
        "path": Path("output/new_capture_raw9_flow_satclip_all/anchor/{s}_anchor.npy"),
    },
    {
        "key": "depth_only_flow",
        "title": "depth-only flow",
        "path": Path("output/new_capture_depth_only_flow_all/hole_only/{s}_hole_only.npy"),
    },
    {
        "key": "raw9_satclip",
        "title": "raw9 satclip",
        "path": Path("output/new_capture_raw9_flow_satclip_all/hole_only/{s}_hole_only.npy"),
    },
    {
        "key": "raw9_realholes",
        "title": "raw9 realholes",
        "path": Path("output/new_capture_raw9_flow_realholes_all/hole_only/{s}_hole_only.npy"),
    },
    {
        "key": "after_synth_split",
        "title": "after-synth split",
        "path": Path("output/new_capture_raw9_flow_after_synth_realhole_all/split_hole_only/{s}_split_hole_only.npy"),
    },
    {
        "key": "prop_refine_split",
        "title": "propagation split",
        "path": Path("output/new_capture_raw9_propagation_refine_all/split_hole_only/{s}_split_hole_only.npy"),
    },
    {
        "key": "propainter",
        "title": "ProPainter",
        "path": Path("output/new_capture_external_inpaint/propainter_run/restored_by_stem/{s}_propainter_restored.npy"),
    },
    {
        "key": "depthcad_depth_gray",
        "title": "DepthCAD depth-gray",
        "path": Path("output/new_capture_depthcad_depth_hole_all_s5/hole_only/{s}_depthcad_depth_hole_only.npy"),
    },
]


def load_array(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def method_path(template, sample):
    return Path(str(template).format(s=sample))


def finite_valid(arr, valid_min, valid_max):
    return np.isfinite(arr) & (arr >= valid_min) & (arr <= valid_max)


def add_panel(ax, image, title, cmap, vmin, vmax):
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def save_figure(args, sample, raw, threshold_mask, cleaned_mask, outputs):
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12), constrained_layout=True)
    axes = axes.ravel()
    panels = [
        ("raw depth", raw, "viridis", args.vmin, args.vmax),
        ("threshold hole", threshold_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("cleaned hole", cleaned_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("NS anchor", outputs["ns_anchor"], "viridis", args.vmin, args.vmax),
        ("depth-only flow", outputs["depth_only_flow"], "viridis", args.vmin, args.vmax),
        ("raw9 satclip", outputs["raw9_satclip"], "viridis", args.vmin, args.vmax),
        ("raw9 realholes", outputs["raw9_realholes"], "viridis", args.vmin, args.vmax),
        ("after-synth split", outputs["after_synth_split"], "viridis", args.vmin, args.vmax),
        ("propagation split", outputs["prop_refine_split"], "viridis", args.vmin, args.vmax),
        ("ProPainter", outputs["propainter"], "viridis", args.vmin, args.vmax),
        ("DepthCAD depth-gray", outputs["depthcad_depth_gray"], "viridis", args.vmin, args.vmax),
    ]
    last_depth_im = None
    for ax, (title, image, cmap, vmin, vmax) in zip(axes, panels):
        im = add_panel(ax, image, title, cmap, vmin, vmax)
        if cmap != "gray":
            last_depth_im = im
    for ax in axes[len(panels):]:
        ax.axis("off")
    if last_depth_im is not None:
        fig.colorbar(last_depth_im, ax=axes.tolist(), fraction=0.025, pad=0.01, label="depth (m)")
    fig.suptitle(f"new capture method comparison: {sample}", fontsize=14)
    out_path = args.output_dir / "figures" / f"{sample}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def sample_rows(args, sample, raw, threshold_mask, cleaned_mask, outputs):
    rows = []
    valid_raw = finite_valid(raw, args.valid_min, args.valid_max)
    for method in METHODS:
        key = method["key"]
        arr = outputs[key]
        valid_out = finite_valid(arr, args.valid_min, args.valid_max)
        threshold_fill_ratio = float(valid_out[threshold_mask].mean()) if threshold_mask.any() else None
        cleaned_fill_ratio = float(valid_out[cleaned_mask].mean()) if cleaned_mask.any() else None
        valid_change = np.abs(arr[valid_raw] - raw[valid_raw])
        threshold_values = arr[threshold_mask & valid_out]
        rows.append(
            {
                "sample": sample,
                "method": key,
                "title": method["title"],
                "raw_valid_ratio": float(valid_raw.mean()),
                "threshold_hole_ratio": float(threshold_mask.mean()),
                "cleaned_hole_ratio": float(cleaned_mask.mean()),
                "threshold_fill_ratio": threshold_fill_ratio,
                "cleaned_fill_ratio": cleaned_fill_ratio,
                "mean_abs_change_on_raw_valid_m": float(valid_change.mean()) if valid_change.size else None,
                "filled_threshold_median_m": float(np.median(threshold_values)) if threshold_values.size else None,
                "filled_threshold_p05_m": float(np.quantile(threshold_values, 0.05)) if threshold_values.size else None,
                "filled_threshold_p95_m": float(np.quantile(threshold_values, 0.95)) if threshold_values.size else None,
            }
        )
    return rows


def summarize(rows):
    summary = {}
    for method in METHODS:
        key = method["key"]
        subset = [row for row in rows if row["method"] == key]
        good_subset = [row for row in subset if row["raw_valid_ratio"] >= 0.10]
        target = good_subset if good_subset else subset
        summary[key] = {
            "title": method["title"],
            "num_samples": len(target),
            "mean_threshold_fill_ratio": float(np.mean([row["threshold_fill_ratio"] for row in target])),
            "mean_cleaned_fill_ratio": float(np.mean([row["cleaned_fill_ratio"] for row in target])),
            "mean_abs_change_on_raw_valid_m": float(
                np.mean([row["mean_abs_change_on_raw_valid_m"] for row in target])
            ),
            "mean_filled_threshold_median_m": float(
                np.mean([row["filled_threshold_median_m"] for row in target])
            ),
        }
    return summary


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted(path.stem for path in args.depth_dir.glob("*.npy"))
    if not samples:
        raise FileNotFoundError(f"No .npy files found under {args.depth_dir}")

    all_rows = []
    figure_paths = []
    for sample in samples:
        raw = load_array(args.depth_dir / f"{sample}.npy")
        threshold_mask = load_array(
            Path("output/new_capture_raw9_flow_satclip_all/hole_mask") / f"{sample}_hole_mask.npy"
        ).astype(bool)
        cleaned_mask = load_array(
            Path("output/new_capture_raw9_flow_after_synth_realhole_all/hole_mask") / f"{sample}_hole_mask.npy"
        ).astype(bool)
        outputs = {
            method["key"]: load_array(method_path(method["path"], sample))
            for method in METHODS
        }
        figure_paths.append(str(save_figure(args, sample, raw, threshold_mask, cleaned_mask, outputs)))
        all_rows.extend(sample_rows(args, sample, raw, threshold_mask, cleaned_mask, outputs))

    summary = {
        "samples": samples,
        "num_samples": len(samples),
        "figures_dir": str(args.output_dir / "figures"),
        "good_sample_rule": "raw_valid_ratio >= 0.10",
        "method_summary_on_good_samples": summarize(all_rows),
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_csv(args.output_dir / "per_sample_metrics.csv", all_rows)
    with (args.output_dir / "figures.txt").open("w") as f:
        for path in figure_paths:
            f.write(path + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
