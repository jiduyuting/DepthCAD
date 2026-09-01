import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build key-sample comparison figures for real raw9 flow outputs.")
    parser.add_argument("--output_dir", type=str, default="output/real_raw9_flow_comparison_key_samples")
    parser.add_argument("--samples", type=str, nargs="+", default=["33", "34", "35", "41", "42"])
    parser.add_argument("--depth_dir", type=str, default="depth")
    parser.add_argument(
        "--old_flow_dir",
        type=str,
        default="output/real_raw9_flow_infer_holefocus_continue_e20_lr5e6_best",
    )
    parser.add_argument(
        "--new_flow_dir",
        type=str,
        default="output/real_raw9_flow_infer_after_synth_realhole_e20_lr5e6",
    )
    parser.add_argument(
        "--recommended_dir",
        type=str,
        default="output/real_raw9_flow_infer_cleaned_plane_recommended",
    )
    parser.add_argument(
        "--strong_dir",
        type=str,
        default="output/real_raw9_flow_infer_cleaned_plane_strong_after_synth",
    )
    return parser.parse_args()


def load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def image_limits(arrays):
    values = []
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    values = np.concatenate(values)
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def add_panel(fig, ax, title, image, cmap, vmin, vmax):
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def save_comparison(args, sample):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = load_npy(os.path.join(args.depth_dir, f"{sample}.npy"))
    old_hole_only = load_npy(os.path.join(args.old_flow_dir, "hole_only", f"{sample}_hole_only.npy"))
    new_flow = load_npy(os.path.join(args.new_flow_dir, "restored", f"{sample}_restored.npy"))
    new_hole_only = load_npy(os.path.join(args.new_flow_dir, "hole_only", f"{sample}_hole_only.npy"))
    threshold_mask = load_npy(
        os.path.join(args.recommended_dir, "threshold_hole_mask", f"{sample}_threshold_hole_mask.npy")
    ).astype(bool)
    rec_mask = load_npy(os.path.join(args.recommended_dir, "hole_mask", f"{sample}_hole_mask.npy")).astype(bool)
    rec_plane = load_npy(os.path.join(args.recommended_dir, "plane_hole_only", f"{sample}_plane_hole_only.npy"))
    strong_mask = load_npy(os.path.join(args.strong_dir, "hole_mask", f"{sample}_hole_mask.npy")).astype(bool)
    strong_plane = load_npy(os.path.join(args.strong_dir, "plane_hole_only", f"{sample}_plane_hole_only.npy"))

    vmin, vmax = image_limits([raw[~threshold_mask], old_hole_only, new_hole_only, rec_plane, strong_plane])
    mask_delta = strong_mask.astype(np.float32) - threshold_mask.astype(np.float32)

    fig, axes = plt.subplots(2, 5, figsize=(22, 8), constrained_layout=True)
    panels = [
        ("raw depth", raw, "viridis", vmin, vmax),
        ("threshold mask", threshold_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("old flow hole-only", old_hole_only, "viridis", vmin, vmax),
        ("new flow restored", new_flow, "viridis", vmin, vmax),
        ("new flow hole-only", new_hole_only, "viridis", vmin, vmax),
        ("recommended mask", rec_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("recommended plane", rec_plane, "viridis", vmin, vmax),
        ("strong mask", strong_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("strong plane", strong_plane, "viridis", vmin, vmax),
        ("strong added mask", mask_delta, "magma", 0.0, 1.0),
    ]
    for ax, panel in zip(axes.ravel(), panels):
        add_panel(fig, ax, *panel)
    fig.suptitle(f"real raw9 key comparison: {sample}")
    out_path = os.path.join(args.output_dir, f"{sample}_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    saved = []
    for sample in args.samples:
        saved.append(save_comparison(args, sample))
    print("Saved comparison figures:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
