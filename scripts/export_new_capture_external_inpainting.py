import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export prepared new-capture depth holes to the external_inputs layout."
    )
    parser.add_argument("--depth_dir", type=Path, default=Path("data/prepared_new_capture/all/depth_m"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/new_capture_external_inpaint/external_inputs"),
    )
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--hole_depth_threshold", type=float, default=0.0)
    parser.add_argument("--valid_min_depth", type=float, default=0.1)
    parser.add_argument("--valid_max_depth", type=float, default=4.5)
    parser.add_argument("--percentile_min", type=float, default=1.0)
    parser.add_argument("--percentile_max", type=float, default=99.0)
    return parser.parse_args()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    mkdir(Path(path).parent)
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


def natural_key(stem):
    digits = "".join(ch for ch in str(stem) if ch.isdigit())
    return (0, int(digits)) if digits else (1, str(stem))


def depth_to_uint8(depth, mask, lo, hi):
    depth = np.asarray(depth, dtype=np.float32)
    out = (depth - float(lo)) / max(float(hi - lo), 1e-6)
    out = np.nan_to_num(out, nan=0.0, neginf=0.0, posinf=1.0)
    out = np.clip(out, 0.0, 1.0)
    out[np.asarray(mask, dtype=bool)] = 0.0
    return np.round(out * 255.0).astype(np.uint8)


def load_samples(depth_dir, requested):
    paths = sorted(Path(depth_dir).glob("*.npy"), key=lambda p: natural_key(p.stem))
    if requested:
        wanted = {str(s) for s in requested}
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"No matching .npy files found under {depth_dir}")
    rows = []
    for path in paths:
        depth = np.load(path).astype(np.float32)
        rows.append((path.stem, path, depth))
    return rows


def main():
    args = parse_args()
    rows = load_samples(args.depth_dir, args.samples)
    output_dir = Path(args.output_dir)
    depth_out = output_dir / "depth_npy"
    mask_out = output_dir / "mask_npy"
    frames_out = output_dir / "export" / "frames"
    masks_png_out = output_dir / "export" / "masks"
    for path in [depth_out, mask_out, frames_out, masks_png_out]:
        mkdir(path)

    masks = {}
    valid_values = []
    for stem, _path, depth in rows:
        mask = (~np.isfinite(depth)) | (depth <= float(args.hole_depth_threshold))
        masks[stem] = mask
        valid = (
            (~mask)
            & np.isfinite(depth)
            & (depth >= float(args.valid_min_depth))
            & (depth <= float(args.valid_max_depth))
        )
        if valid.any():
            valid_values.append(depth[valid])

    if valid_values:
        values = np.concatenate(valid_values)
        lo, hi = np.percentile(values, [float(args.percentile_min), float(args.percentile_max)])
    else:
        lo, hi = args.valid_min_depth, args.valid_max_depth
    lo = float(lo)
    hi = float(hi if hi > lo else lo + 1.0)

    frame_mapping = []
    for idx, (stem, path, depth) in enumerate(rows):
        mask = masks[stem].astype(bool)
        corrupted = depth.copy()
        corrupted[mask] = 0.0
        np.save(depth_out / f"{idx:04d}.npy", corrupted.astype(np.float32))
        np.save(mask_out / f"{idx:04d}.npy", mask.astype(np.uint8))
        frame = depth_to_uint8(corrupted, mask, lo, hi)
        if not cv2.imwrite(str(frames_out / f"{idx:04d}.png"), frame):
            raise RuntimeError(f"Failed to write {frames_out / f'{idx:04d}.png'}")
        if not cv2.imwrite(str(masks_png_out / f"{idx:04d}.png"), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to write {masks_png_out / f'{idx:04d}.png'}")
        frame_mapping.append(
            {
                "frame_index": idx,
                "source_stem": stem,
                "source_path_m": str(path.resolve()),
                "mask_ratio": float(mask.mean()),
                "shape": list(depth.shape),
            }
        )

    shape0 = list(rows[0][2].shape)
    save_json(
        output_dir / "source_mapping.json",
        {
            "source_dir_m": str(Path(args.depth_dir).resolve()),
            "case_dir": str(output_dir.parent.resolve()),
            "case_name": output_dir.parent.name,
            "frame_count": len(rows),
            "mask_semantics": "nonzero/white means invalid region to inpaint",
            "frame_mapping": frame_mapping,
        },
    )
    save_json(
        output_dir / "export" / "depth_meta.json",
        {
            "source_depth_npy": str(depth_out.resolve()),
            "resolved_layout": "frame_dir_hw",
            "original_shape": [len(rows)] + shape0,
            "canonical_shape": [len(rows)] + shape0,
            "depth_min": lo,
            "depth_max": hi,
            "mask_source": str(mask_out.resolve()),
            "mask_semantics": "invalid_is_nonzero",
            "percentile_min": float(args.percentile_min),
            "percentile_max": float(args.percentile_max),
            "hole_depth_threshold": float(args.hole_depth_threshold),
            "valid_min_depth": float(args.valid_min_depth),
            "valid_max_depth": float(args.valid_max_depth),
            "frame_prefix": "",
            "frames_dir_name": "frames",
            "masks_dir_name": "masks",
            "notes": "Frames are grayscale normalized depth with masked pixels set to zero.",
        },
    )
    save_json(
        output_dir / "export_summary.json",
        {
            "output_dir": str(output_dir.resolve()),
            "frames_dir": str(frames_out.resolve()),
            "masks_dir": str(masks_png_out.resolve()),
            "depth_min": lo,
            "depth_max": hi,
            "num_samples": len(rows),
            "samples": frame_mapping,
        },
    )
    print(f"Exported {len(rows)} samples to {output_dir}")
    print(f"  frames: {frames_out}")
    print(f"  masks:  {masks_png_out}")
    print(f"  depth range: [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
