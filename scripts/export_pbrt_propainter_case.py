#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a PBRT depth-completion cache to the external_inputs layout "
            "used by run_external_inpainting_far_pic.py, so ProPainter can be "
            "run as a depth-as-grayscale inpainting baseline."
        )
    )
    parser.add_argument(
        "--cache_dir",
        default="depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123",
        help="PBRT cache containing .npz files with depth_noisy/gt_depth/hole_mask.",
    )
    parser.add_argument(
        "--sample_list",
        default=None,
        help="Optional text file listing .npz samples. Overrides recursive cache scan.",
    )
    parser.add_argument(
        "--output_dir",
        default="output/pbrt_propainter_seed123",
        help="Output case directory.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--range_source",
        default="gt",
        choices=["gt", "observed", "combined"],
        help="Depth values used to compute the grayscale decode range.",
    )
    parser.add_argument(
        "--range_percentiles",
        type=float,
        nargs=2,
        default=[0.0, 100.0],
        help="Percentiles for depth_min/depth_max over the selected range source.",
    )
    parser.add_argument(
        "--clip_frames",
        action="store_true",
        default=True,
        help="Clip encoded grayscale frames to depth_min/depth_max.",
    )
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def collect_paths(args):
    if args.sample_list:
        paths = read_list(args.sample_list)
    else:
        paths = sorted(glob(os.path.join(args.cache_dir, "**", "*.npz"), recursive=True))
    if args.max_samples is not None:
        paths = paths[: max(0, int(args.max_samples))]
    if not paths:
        raise FileNotFoundError(f"No .npz files found for {args.cache_dir!r}")
    return paths


def scalar_str(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def safe_stem(sample_name):
    return sample_name.replace("/", "_").replace("\\", "_")


def finite_values(arr, mask):
    vals = np.asarray(arr, dtype=np.float32)[mask]
    return vals[np.isfinite(vals)]


def compute_depth_range(paths, args):
    values = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            gt = data["gt_depth"].astype(np.float32)
            noisy = data["depth_noisy"].astype(np.float32)
            hole = data["hole_mask"] > 0.5
            valid = (data["valid_mask"] > 0.5) if "valid_mask" in data.files else np.isfinite(gt)

            if args.range_source == "gt":
                vals = finite_values(gt, valid)
            elif args.range_source == "observed":
                vals = finite_values(noisy, valid & (~hole) & (noisy > 0.0))
            else:
                vals = np.concatenate(
                    [
                        finite_values(gt, valid),
                        finite_values(noisy, valid & (~hole) & (noisy > 0.0)),
                    ]
                )
            if vals.size:
                values.append(vals)

    if not values:
        raise ValueError("Could not find finite depth values to compute grayscale range.")

    all_values = np.concatenate(values)
    lo, hi = np.percentile(all_values, args.range_percentiles)
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid depth range: {lo}..{hi}")
    return lo, hi


def encode_depth_png(depth, hole, lo, hi, clip=True):
    depth = np.asarray(depth, dtype=np.float32)
    encoded = depth.copy()
    encoded[hole] = lo
    if clip:
        encoded = np.clip(encoded, lo, hi)
    gray = (encoded - lo) / max(hi - lo, 1e-6)
    gray = np.nan_to_num(gray, nan=0.0, neginf=0.0, posinf=1.0)
    gray = np.clip(gray, 0.0, 1.0)
    gray_u8 = np.round(gray * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def save_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main():
    args = parse_args()
    paths = collect_paths(args)
    depth_min, depth_max = compute_depth_range(paths, args)

    case_dir = Path(args.output_dir)
    external_dir = case_dir / "external_inputs"
    export_dir = external_dir / "export"
    frames_dir = export_dir / "frames"
    masks_png_dir = export_dir / "masks"
    depth_dir = external_dir / "depth_npy"
    gt_dir = external_dir / "gt_npy"
    mask_npy_dir = external_dir / "mask_npy"
    valid_dir = external_dir / "valid_npy"
    for directory in [frames_dir, masks_png_dir, depth_dir, gt_dir, mask_npy_dir, valid_dir]:
        ensure_dir(directory)

    mapping = []
    depth_stats = []
    seen_stems = set()
    for idx, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            sample_name = (
                scalar_str(data["sample_name"]) if "sample_name" in data.files else Path(path).stem
            )
            stem = safe_stem(sample_name)
            if stem in seen_stems:
                stem = f"{stem}_{idx:04d}"
            seen_stems.add(stem)

            noisy = data["depth_noisy"].astype(np.float32)
            gt = data["gt_depth"].astype(np.float32)
            hole = (data["hole_mask"] > 0.5)
            valid = (data["valid_mask"] > 0.5) if "valid_mask" in data.files else np.isfinite(gt)

        frame = encode_depth_png(noisy, hole, depth_min, depth_max, clip=args.clip_frames)
        mask_png = hole.astype(np.uint8) * 255

        frame_path = frames_dir / f"{idx:04d}.png"
        mask_png_path = masks_png_dir / f"{idx:04d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise IOError(f"Failed to write {frame_path}")
        if not cv2.imwrite(str(mask_png_path), mask_png):
            raise IOError(f"Failed to write {mask_png_path}")

        np.save(depth_dir / f"{idx:04d}.npy", noisy.astype(np.float32))
        np.save(gt_dir / f"{idx:04d}.npy", gt.astype(np.float32))
        np.save(mask_npy_dir / f"{idx:04d}.npy", hole.astype(bool))
        np.save(valid_dir / f"{idx:04d}.npy", valid.astype(bool))

        valid_hole = valid & hole
        valid_obs = valid & (~hole)
        mapping.append(
            {
                "frame_index": idx,
                "source_stem": stem,
                "sample_name": sample_name,
                "cache_path": os.path.abspath(path),
                "frame_png": str(frame_path),
                "mask_png": str(mask_png_path),
                "depth_npy": str(depth_dir / f"{idx:04d}.npy"),
                "gt_npy": str(gt_dir / f"{idx:04d}.npy"),
                "mask_npy": str(mask_npy_dir / f"{idx:04d}.npy"),
                "valid_npy": str(valid_dir / f"{idx:04d}.npy"),
                "hole_ratio": float(hole.mean()),
                "valid_hole_count": int(valid_hole.sum()),
                "valid_observed_count": int(valid_obs.sum()),
            }
        )
        depth_stats.append(gt[valid])

    save_json(
        external_dir / "source_mapping.json",
        {
            "cache_dir": os.path.abspath(args.cache_dir),
            "sample_list": os.path.abspath(args.sample_list) if args.sample_list else None,
            "frame_mapping": mapping,
        },
    )
    save_json(
        export_dir / "depth_meta.json",
        {
            "depth_min": depth_min,
            "depth_max": depth_max,
            "range_source": args.range_source,
            "range_percentiles": args.range_percentiles,
            "source_depth_npy": str(depth_dir.resolve()),
            "gt_source": str(gt_dir.resolve()),
            "mask_source": str(mask_npy_dir.resolve()),
            "valid_source": str(valid_dir.resolve()),
            "decode_rule": "gray/255*(depth_max-depth_min)+depth_min, merged inside hole mask",
            "frame_encoding": "depth_noisy encoded as grayscale with holes set to depth_min",
        },
    )
    summary = {
        "case_dir": str(case_dir.resolve()),
        "external_inputs": str(external_dir.resolve()),
        "num_samples": len(mapping),
        "depth_min": depth_min,
        "depth_max": depth_max,
        "frames_dir": str(frames_dir.resolve()),
        "masks_dir": str(masks_png_dir.resolve()),
        "mapping": str((external_dir / "source_mapping.json").resolve()),
        "depth_meta": str((export_dir / "depth_meta.json").resolve()),
        "mean_hole_ratio": float(np.mean([row["hole_ratio"] for row in mapping])),
    }
    save_json(case_dir / "export_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
