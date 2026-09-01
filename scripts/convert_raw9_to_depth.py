import argparse
from pathlib import Path

import numpy as np

from depth_estimator import DepthEstimator


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw9 tensors into depth maps using DepthCAD's fixed DepthEstimator. "
            "The first 6 channels must follow [I30, Q30, I40, Q40, I58, Q58]."
        )
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw9 .npy files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save depth .npy files.")
    parser.add_argument("--maxd", type=float, default=10.0, help="Maximum depth range in meters.")
    parser.add_argument("--nt", type=int, default=5000, help="Number of depth candidates in DepthEstimator.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_one(raw9_path: Path, estimator: DepthEstimator) -> np.ndarray:
    raw9 = np.load(raw9_path).astype(np.float32)
    if raw9.ndim != 3 or raw9.shape[0] < 6:
        raise ValueError(f"{raw9_path} has invalid shape {raw9.shape}; expected at least (6,H,W)")

    iq6 = raw9[:6]
    depth = estimator.process(iq6).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    estimator = DepthEstimator(maxd=args.maxd, nt=args.nt)

    for path in paths:
        out_path = output_dir / path.name
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue

        depth = convert_one(path, estimator)
        np.save(out_path, depth)
        print(
            f"[ok] {path.name} -> {out_path.name} "
            f"shape={depth.shape} min={depth.min():.4f} max={depth.max():.4f} mean={depth.mean():.4f}"
        )


if __name__ == "__main__":
    main()
