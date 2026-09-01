import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from kinect_init import base_cor
from utils import processPixelStage2


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert full-resolution Kinect triplet raw9 tensors "
            "([a0,b0,amp0]x3, 424x512) into candidate sensor depth maps."
        )
    )
    parser.add_argument("--input_dir", type=str, default="far_pic/kinect_raw9_424x512_sign_extend")
    parser.add_argument("--output_dir", type=str, default="far_pic/kinect_depth_240x320_sign_extend")
    parser.add_argument("--preview_dir", type=str, default="output/far_pic_kinect_depth_previews_sign_extend")
    parser.add_argument("--target_h", type=int, default=240)
    parser.add_argument("--target_w", type=int, default=320)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--min_resize_valid_ratio", type=float, default=0.25)
    parser.add_argument("--no_baseline_correction", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def triplet_raw9_to_stage_input(raw9):
    if raw9.shape != (9, 424, 512):
        raise ValueError(f"Expected raw9 shape (9,424,512), got {raw9.shape}")
    return np.stack(
        [
            raw9[0],
            raw9[3],
            raw9[6],
            raw9[1],
            raw9[4],
            raw9[7],
            raw9[2],
            raw9[5],
            raw9[8],
        ],
        axis=2,
    ).astype(np.float32)


def compute_depth_m(raw9, min_depth, max_depth, apply_baseline_correction):
    stage_input = triplet_raw9_to_stage_input(raw9)
    depth_mm = processPixelStage2(tf.convert_to_tensor(stage_input, dtype=tf.float32)).numpy()
    depth = np.nan_to_num(depth_mm / 1000.0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if apply_baseline_correction:
        depth = depth * base_cor["k"].astype(np.float32) + base_cor["b"].astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    valid = (depth >= min_depth) & (depth <= max_depth)
    return np.where(valid, depth, 0.0).astype(np.float32)


def resize_depth_preserve_holes(depth, target_h, target_w, min_valid_ratio):
    valid = (depth > 0).astype(np.float32)
    weighted = depth * valid
    weighted_small = cv2.resize(weighted, (target_w, target_h), interpolation=cv2.INTER_AREA)
    valid_small = cv2.resize(valid, (target_w, target_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((target_h, target_w), dtype=np.float32)
    keep = valid_small >= float(min_valid_ratio)
    out[keep] = weighted_small[keep] / np.maximum(valid_small[keep], 1e-6)
    return out


def save_depth_preview(depth, path):
    valid = depth > 0
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    depth_u8 = np.clip((depth - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    depth_u8[~valid] = 0
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), color)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    preview_dir = Path(args.preview_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    for path in paths:
        out_path = output_dir / path.name
        preview_path = preview_dir / f"{path.stem}.png"
        if out_path.exists() and preview_path.exists() and not args.overwrite:
            depth = np.load(out_path)
        else:
            raw9 = np.load(path).astype(np.float32)
            depth_full = compute_depth_m(
                raw9,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                apply_baseline_correction=not args.no_baseline_correction,
            )
            depth = resize_depth_preserve_holes(
                depth_full,
                target_h=args.target_h,
                target_w=args.target_w,
                min_valid_ratio=args.min_resize_valid_ratio,
            )
            np.save(out_path, depth.astype(np.float32))
            save_depth_preview(depth, preview_path)

        valid = depth > 0
        valid_mean = float(depth[valid].mean()) if np.any(valid) else 0.0
        print(
            f"[ok] {path.name} -> {out_path.name} shape={depth.shape} "
            f"valid={float(valid.mean()):.3f} min={float(depth.min()):.3f} "
            f"max={float(depth.max()):.3f} valid_mean={valid_mean:.3f}"
        )

    print(f"Saved candidate depth maps to {output_dir}")
    print(f"Saved previews to {preview_dir}")


if __name__ == "__main__":
    main()
