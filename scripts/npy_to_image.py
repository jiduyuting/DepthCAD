#!/usr/bin/env python3
"""
Batch convert .npy arrays to image files (PNG). Designed for the DepthCAD preprocessed outputs.

Usage examples:
  # Convert single-channel confidence maps to PNGs (per-file min/max normalization)
  python scripts/npy_to_image.py --input data/confidence --output out/confidence_png

  # Convert per-channel IQ files stored as single .npy (512x512) under data/ideal_IQ to PNGs
  python scripts/npy_to_image.py --input data/ideal_IQ --output out/ideal_png --suffix _A _B _C _D _E _F

  # Convert multi-channel .npy (6,512,512) -> save each channel and also a 3-channel RGB preview
  python scripts/npy_to_image.py --input some_folder --output out --as_rgb_preview

Options:
  --normalize per|none     : per-file min-max normalize (default: per) or no normalization
  --colormap none|jet      : apply matplotlib colormap for visualization (only for single-channel), default none
  --ext png|jpg            : output extension (default png)

The script preserves directory structure under the output root.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import sys


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def to_uint8(arr, normalize="per", clip_percentiles=None):
    # arr: numpy array float or int
    a = np.array(arr, copy=True)
    if a.dtype == np.uint8:
        return a
    if normalize == "none":
        # clip into 0..255
        a = np.clip(a, 0, 255)
        return a.astype(np.uint8)

    # per-file min-max normalization
    if clip_percentiles is not None:
        lo, hi = np.percentile(a, [clip_percentiles[0], clip_percentiles[1]])
        a = np.clip(a, lo, hi)

    amin = a.min()
    amax = a.max()
    if amax - amin < 1e-8:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - amin) / (amax - amin)
    a = (a * 255.0).astype(np.uint8)
    return a


def apply_colormap_uint8(gray_u8, cmap="jet"):
    if cmap == "none":
        return gray_u8
    # OpenCV expects BGR
    if cmap == "jet":
        return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    else:
        return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)


def process_file(src_path: Path, dst_root: Path, args):
    rel = src_path.relative_to(args.input)
    dst_dir = dst_root.joinpath(rel.parent)
    ensure_dir(dst_dir)

    try:
        arr = np.load(str(src_path))
    except Exception as e:
        print(f"[ERROR] load {src_path}: {e}")
        return

    # Handle different shapes
    if arr.ndim == 2:
        # single-channel HxW
        u8 = to_uint8(arr, normalize=args.normalize, clip_percentiles=args.clip_percentiles)
        if args.colormap != "none":
            colored = apply_colormap_uint8(u8, cmap=args.colormap)
            out_path = dst_dir.joinpath(src_path.stem + "." + args.ext)
            cv2.imwrite(str(out_path), colored)
        else:
            out_path = dst_dir.joinpath(src_path.stem + "." + args.ext)
            cv2.imwrite(str(out_path), u8)

    elif arr.ndim == 3:
        # Possible layouts: (C, H, W) or (H, W, C)
        C = arr.shape[0]
        if C in (1, 3, 6):
            # assume (C, H, W)
            chw = True
        elif arr.shape[2] in (1, 3, 6):
            chw = False
        else:
            chw = True

        if not chw:
            arr = np.transpose(arr, (2, 0, 1))

        C, H, W = arr.shape

        # If it's 3-channel, save as RGB
        if C == 3:
            rgb = np.stack([to_uint8(arr[i], normalize=args.normalize, clip_percentiles=args.clip_percentiles) for i in range(3)], axis=-1)
            out_path = dst_dir.joinpath(src_path.stem + "." + args.ext)
            # cv2 wants BGR
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return

        # If it's 6-channel (DepthCAD IQ), save each channel separately with suffixes A..F
        if C == 6:
            suffixes = ["_A", "_B", "_C", "_D", "_E", "_F"]
            for i in range(6):
                u8 = to_uint8(arr[i], normalize=args.normalize, clip_percentiles=args.clip_percentiles)
                out_path = dst_dir.joinpath(src_path.stem + f"{suffixes[i]}." + args.ext)
                cv2.imwrite(str(out_path), u8)

            # optional RGB preview from channels (choose 0,2,4 as example)
            if args.as_rgb_preview:
                preview = np.stack([
                    to_uint8(arr[0], normalize=args.normalize, clip_percentiles=args.clip_percentiles),
                    to_uint8(arr[2], normalize=args.normalize, clip_percentiles=args.clip_percentiles),
                    to_uint8(arr[4], normalize=args.normalize, clip_percentiles=args.clip_percentiles),
                ], axis=-1)
                preview_path = dst_dir.joinpath(src_path.stem + "_preview." + args.ext)
                cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            return

        # Generic: save each channel
        for i in range(C):
            u8 = to_uint8(arr[i], normalize=args.normalize, clip_percentiles=args.clip_percentiles)
            out_path = dst_dir.joinpath(f"{src_path.stem}_ch{i}." + args.ext)
            cv2.imwrite(str(out_path), u8)

    else:
        print(f"[WARN] Unsupported array ndim {arr.ndim} for {src_path}")


def iter_files(root: Path, pattern="*.npy"):
    for p in root.rglob(pattern):
        yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory or file (.npy)")
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument("--pattern", default="*.npy", help="Glob pattern to find .npy files")
    parser.add_argument("--normalize", choices=["per", "none"], default="per", help="Normalization mode")
    parser.add_argument("--colormap", choices=["none", "jet"], default="none", help="Apply colormap to single-channel images")
    parser.add_argument("--ext", choices=["png", "jpg"], default="png")
    parser.add_argument("--as-rgb-preview", dest="as_rgb_preview", action="store_true", help="Save an RGB preview for 6-channel inputs")
    parser.add_argument("--clip-percentiles", nargs=2, type=float, metavar=("LOW", "HIGH"), help="Optional clip percentiles before normalization e.g. 1 99")
    args = parser.parse_args()

    args.input = Path(args.input).resolve()
    args.output = Path(args.output).resolve()

    if not args.input.exists():
        print(f"Input {args.input} does not exist")
        sys.exit(1)

    ensure_dir(args.output)

    args.clip_percentiles = tuple(args.clip_percentiles) if args.clip_percentiles is not None else None

    if args.input.is_file():
        process_file(args.input, args.output, args)
    else:
        for p in iter_files(args.input, args.pattern):
            process_file(p, args.output, args)


if __name__ == "__main__":
    main()
