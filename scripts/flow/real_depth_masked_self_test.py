import _bootstrap
import argparse
import json
import os
from glob import glob

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from depth_restoration_backbones import build_depth_backbone
from eval_depth_restoration import load_checkpoint
from infer_real_depth_flow import (
    clip_prediction,
    ensure_dir,
    make_depth_condition,
    move_condition_to_device,
    natural_key,
    predict_depth,
)
from train_depth_flow_restoration import flow_model_in_channels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Masked self-test for real depth maps using valid pixels as pseudo-GT."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/depth_flow_restoration_noisy_ns_n1000_endpoint/best.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/real_depth_masked_self_test",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mask_ratio", type=float, default=0.10)
    parser.add_argument("--num_masks_per_sample", type=int, default=1)
    parser.add_argument("--min_block_size", type=int, default=12)
    parser.add_argument("--max_block_size", type=int, default=72)
    parser.add_argument("--anchor_inpaint_radius", type=int, default=None)
    parser.add_argument("--hole_depth_threshold", type=float, default=1.0)
    parser.add_argument("--valid_min_depth", type=float, default=1.0)
    parser.add_argument("--valid_max_depth", type=float, default=9.9)
    parser.add_argument(
        "--post_clip_mode",
        type=str,
        default="valid_range",
        choices=["none", "valid_range", "valid_percentile"],
    )
    parser.add_argument(
        "--post_clip_percentiles",
        type=float,
        nargs=2,
        default=[0.5, 99.5],
    )
    parser.add_argument("--sampling_mode", type=str, default=None, choices=["endpoint", "euler"])
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_max_samples", type=int, default=24)
    return parser.parse_args()


def build_model(checkpoint, ckpt_args, device):
    condition_channels = 4 + int(bool(ckpt_args.get("include_hole_distance", False)))
    time_channels = int(ckpt_args.get("time_channels", 16))
    in_channels = flow_model_in_channels(condition_channels, time_channels)
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
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def make_block_mask(valid_mask, rng, target_ratio, min_size, max_size):
    h, w = valid_mask.shape
    target = max(1, int(round(float(target_ratio) * int(valid_mask.sum()))))
    mask = np.zeros((h, w), dtype=bool)
    valid_yx = np.argwhere(valid_mask)
    if valid_yx.size == 0:
        return mask

    attempts = 0
    max_attempts = 600
    while int((mask & valid_mask).sum()) < target and attempts < max_attempts:
        attempts += 1
        cy, cx = valid_yx[rng.integers(0, len(valid_yx))]
        bh = int(rng.integers(min_size, max_size + 1))
        bw = int(rng.integers(min_size, max_size + 1))
        y0 = max(0, int(cy - bh // 2))
        y1 = min(h, int(y0 + bh))
        x0 = max(0, int(cx - bw // 2))
        x1 = min(w, int(x0 + bw))
        y0 = max(0, y1 - bh)
        x0 = max(0, x1 - bw)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        if rng.random() < 0.65:
            ry = max((y1 - y0) / 2.0, 1.0)
            rx = max((x1 - x0) / 2.0, 1.0)
            local = ((yy - (y0 + y1 - 1) / 2.0) / ry) ** 2 + (
                (xx - (x0 + x1 - 1) / 2.0) / rx
            ) ** 2 <= 1.0
        else:
            local = np.ones((y1 - y0, x1 - x0), dtype=bool)

        mask[y0:y1, x0:x1] |= local
        mask &= valid_mask

    return mask & valid_mask


def mae(pred, target, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    if valid.sum() == 0:
        return None, 0
    return float(np.mean(np.abs(pred[valid] - target[valid]))), int(valid.sum())


def aggregate(rows):
    sums = {}
    counts = {}
    for row in rows:
        metric_keys = sorted(
            key
            for key in row
            if key.endswith("_mae")
            and isinstance(row.get(key), (int, float))
            and f"{key}_count" in row
        )
        for key in metric_keys:
            value = row.get(key)
            count = row.get(f"{key}_count", 0)
            if value is None or count == 0:
                continue
            sums[key] = sums.get(key, 0.0) + float(value) * int(count)
            counts[key] = counts.get(key, 0) + int(count)

    out = {"num_cases": len(rows)}
    for key, total in sums.items():
        out[key] = total / max(counts[key], 1)
        out[f"{key}_count"] = counts[key]

    if out.get("anchor_mask_mae") and out.get("model_mask_mae") is not None:
        out["mask_improve_vs_anchor"] = (
            out["anchor_mask_mae"] - out["model_mask_mae"]
        ) / max(out["anchor_mask_mae"], 1e-12)
    if out.get("anchor_global_mae") and out.get("model_global_mae") is not None:
        out["global_improve_vs_anchor"] = (
            out["anchor_global_mae"] - out["model_global_mae"]
        ) / max(out["anchor_global_mae"], 1e-12)
    for prefix in ["hole_only", "gated"]:
        mask_key = f"{prefix}_mask_mae"
        global_key = f"{prefix}_global_mae"
        if out.get("anchor_mask_mae") and out.get(mask_key) is not None:
            out[f"{prefix}_mask_improve_vs_anchor"] = (
                out["anchor_mask_mae"] - out[mask_key]
            ) / max(out["anchor_mask_mae"], 1e-12)
        if out.get("anchor_global_mae") and out.get(global_key) is not None:
            out[f"{prefix}_global_improve_vs_anchor"] = (
                out["anchor_global_mae"] - out[global_key]
            ) / max(out["anchor_global_mae"], 1e-12)
    return out


def image_limits(*arrays):
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


def save_visualization(
    path,
    title,
    clean,
    corrupted,
    artificial_mask,
    anchor,
    model,
    hole_only,
    gated_hole_only=None,
    gated_weight=None,
):
    ensure_dir(os.path.dirname(path))
    limit_arrays = [clean, corrupted, anchor, model, hole_only]
    if gated_hole_only is not None:
        limit_arrays.append(gated_hole_only)
    vmin, vmax = image_limits(*limit_arrays)
    anchor_err = np.abs(anchor - clean)
    model_err = np.abs(model - clean)
    err_arrays = [
        anchor_err[artificial_mask & np.isfinite(anchor_err)],
        model_err[artificial_mask & np.isfinite(model_err)],
    ]
    gated_err = None
    if gated_hole_only is not None:
        gated_err = np.abs(gated_hole_only - clean)
        err_arrays.append(gated_err[artificial_mask & np.isfinite(gated_err)])
    nonempty_err_arrays = [arr for arr in err_arrays if arr.size]
    err_values = (
        np.concatenate(nonempty_err_arrays)
        if nonempty_err_arrays
        else np.asarray([], dtype=np.float32)
    )
    err_max = float(np.percentile(err_values, 98.0)) if err_values.size else 1.0
    err_max = max(err_max, 1e-6)

    panels = [
        ("pseudo-GT clean", clean, "viridis", vmin, vmax),
        ("corrupted input", corrupted, "viridis", vmin, vmax),
        ("artificial mask", artificial_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("NS anchor", anchor, "viridis", vmin, vmax),
        ("flow model", model, "viridis", vmin, vmax),
        ("hole-only model", hole_only, "viridis", vmin, vmax),
        ("anchor error", np.where(artificial_mask, anchor_err, 0.0), "magma", 0.0, err_max),
        ("model error", np.where(artificial_mask, model_err, 0.0), "magma", 0.0, err_max),
    ]
    if gated_hole_only is not None:
        panels.insert(6, ("gated hole-only", gated_hole_only, "viridis", vmin, vmax))
        if gated_weight is not None:
            panels.insert(7, ("gate weight", gated_weight.astype(np.float32), "gray", 0.0, 1.0))
        panels.append(("gated error", np.where(artificial_mask, gated_err, 0.0), "magma", 0.0, err_max))
    cols = 4
    rows_fig = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows_fig, cols, figsize=(4.5 * cols, 4.0 * rows_fig), constrained_layout=True)
    for ax, (name, image, cmap, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    paths = sorted(glob(os.path.join(args.input_dir, "*.npy")), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No .npy files found under {args.input_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    if ckpt_args.get("input_mode", "noisy") != "noisy":
        raise ValueError("This real depth-only self-test requires a checkpoint with input_mode='noisy'.")
    model = build_model(ckpt, ckpt_args, device)

    sampling_mode = args.sampling_mode or ckpt_args.get("eval_sampling_mode", "endpoint")
    sample_steps = int(args.sample_steps or ckpt_args.get("sample_steps", 8))

    for subdir in ["restored", "hole_only", "anchor", "corrupted", "mask", "visualizations"]:
        ensure_dir(os.path.join(args.output_dir, subdir))

    rows = []
    vis_saved = 0
    for path_index, path in enumerate(paths):
        clean = np.load(path).astype(np.float32)
        if clean.ndim != 2:
            raise ValueError(f"{path} must be 2D, got {clean.shape}")
        base_name = os.path.splitext(os.path.basename(path))[0]

        reliable = (
            np.isfinite(clean)
            & (clean > float(args.hole_depth_threshold))
            & (clean >= float(args.valid_min_depth))
            & (clean <= float(args.valid_max_depth))
        )
        if reliable.sum() == 0:
            continue

        for repeat in range(int(args.num_masks_per_sample)):
            rng = np.random.default_rng(int(args.seed) + path_index * 1009 + repeat)
            artificial_mask = make_block_mask(
                reliable,
                rng,
                args.mask_ratio,
                args.min_block_size,
                args.max_block_size,
            )
            if artificial_mask.sum() == 0:
                continue

            corrupted = clean.copy()
            corrupted[artificial_mask] = 0.0

            condition_cpu = make_depth_condition(corrupted, ckpt_args, args)
            condition = move_condition_to_device(condition_cpu, device)
            pred = predict_depth(model, condition, ckpt_args, sampling_mode, sample_steps)
            pred_raw = pred.detach().cpu().numpy()[0, 0].astype(np.float32)
            pred, clip_bounds = clip_prediction(pred_raw, clean, reliable, args)

            condition_hole = condition_cpu["hole"]
            anchor = condition_cpu["anchor"]
            hole_only = np.where(condition_hole, pred, corrupted).astype(np.float32)
            unmasked_reliable = reliable & (~artificial_mask)

            row = {
                "name": base_name,
                "repeat": repeat,
                "input_path": path,
                "shape": list(clean.shape),
                "mask_ratio_target": float(args.mask_ratio),
                "mask_ratio_actual": float(artificial_mask.sum() / max(reliable.sum(), 1)),
                "mask_pixel_count": int(artificial_mask.sum()),
                "reliable_pixel_count": int(reliable.sum()),
                "post_clip_bounds": clip_bounds,
            }
            metric_specs = [
                ("anchor_mask_mae", anchor, artificial_mask),
                ("model_mask_mae", pred, artificial_mask),
                ("hole_only_mask_mae", hole_only, artificial_mask),
                ("model_unmasked_mae", pred, unmasked_reliable),
                ("hole_only_unmasked_mae", hole_only, unmasked_reliable),
                ("anchor_global_mae", anchor, reliable),
                ("model_global_mae", pred, reliable),
                ("hole_only_global_mae", hole_only, reliable),
            ]
            for key, prediction, mask in metric_specs:
                value, count = mae(prediction, clean, mask)
                row[key] = value
                row[f"{key}_count"] = count
            if row["anchor_mask_mae"] is not None and row["model_mask_mae"] is not None:
                row["mask_improve_vs_anchor"] = (
                    row["anchor_mask_mae"] - row["model_mask_mae"]
                ) / max(row["anchor_mask_mae"], 1e-12)
            rows.append(row)

            stem = f"{base_name}_r{repeat:02d}"
            np.save(os.path.join(args.output_dir, "restored", f"{stem}_restored.npy"), pred)
            np.save(os.path.join(args.output_dir, "hole_only", f"{stem}_hole_only.npy"), hole_only)
            np.save(os.path.join(args.output_dir, "anchor", f"{stem}_anchor.npy"), anchor)
            np.save(os.path.join(args.output_dir, "corrupted", f"{stem}_corrupted.npy"), corrupted)
            np.save(os.path.join(args.output_dir, "mask", f"{stem}_mask.npy"), artificial_mask.astype(np.uint8))

            if args.visualize and vis_saved < int(args.vis_max_samples):
                title = (
                    f"{stem} | anchor_mask={row['anchor_mask_mae']:.4f} "
                    f"model_mask={row['model_mask_mae']:.4f} "
                    f"improve={row.get('mask_improve_vs_anchor', 0.0):.1%}"
                )
                save_visualization(
                    os.path.join(args.output_dir, "visualizations", f"{stem}.png"),
                    title,
                    clean,
                    corrupted,
                    artificial_mask,
                    anchor,
                    pred,
                    hole_only,
                )
                vis_saved += 1

            print(
                f"[{len(rows):03d}] {stem} "
                f"mask={row['mask_ratio_actual']:.3f} "
                f"anchor={row['anchor_mask_mae']:.4f} "
                f"model={row['model_mask_mae']:.4f} "
                f"improve={row.get('mask_improve_vs_anchor', 0.0):.1%}"
            )

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_args": ckpt_args,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "hole_depth_threshold": args.hole_depth_threshold,
        "mask_ratio": args.mask_ratio,
        "num_masks_per_sample": args.num_masks_per_sample,
        "sampling_mode": sampling_mode,
        "sample_steps": sample_steps,
        "aggregate": aggregate(rows),
        "per_sample": rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved masked self-test results to {args.output_dir}")


if __name__ == "__main__":
    main()
