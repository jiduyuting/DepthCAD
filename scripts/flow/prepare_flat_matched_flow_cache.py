#!/usr/bin/env python3
"""Build FLAT cache with the same amplitude-driven holes as the PBRT cache."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow direct execution from the repository root as well as invocation from the parallel runner, whose PYTHONPATH already includes ``scripts``.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from prepare_full_pbrt_cache import generate_holes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_data_root", type=Path, default=Path("flat_dataset/data"))
    parser.add_argument("--depth_root", type=Path, default=Path("/data/pre_student/hcy/ControlNet/data"))
    parser.add_argument("--output_root", type=Path, default=Path("depth_completion_cache/flat_flow_matched_pbrt"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hole_ratio", type=float, default=0.15)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--amp_percentile", type=float, default=5.0)
    parser.add_argument("--low_amp_ratio", type=float, default=0.4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    flat = args.flat_data_root
    depth = args.depth_root
    ids = sorted(
        p.name[:-6]
        for p in (flat / "noise_IQ").glob("*_A.npy")
        if (flat / "ideal_IQ" / p.name).is_file()
        and (depth / "ideal_depth" / f"{p.name[:-6]}.npy").is_file()
        and (depth / "noise_depth" / f"{p.name[:-6]}.npy").is_file()
    )
    if not ids:
        raise SystemExit("No paired FLAT samples found.")
    out = args.output_root
    cache_dir = out / "flat" / "0"
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = argparse.Namespace(
        amp_percentile=args.amp_percentile,
        block_size=args.block_size,
        low_amp_ratio=args.low_amp_ratio,
        hole_ratio=args.hole_ratio,
    )
    channels = "ABCDEF"
    written = 0
    for sample_id in ids:
        target = cache_dir / f"{sample_id}.npz"
        if target.exists() and not args.overwrite:
            continue
        ideal_depth = np.load(depth / "ideal_depth" / f"{sample_id}.npy", allow_pickle=False).astype(np.float32)
        noise_depth = np.load(depth / "noise_depth" / f"{sample_id}.npy", allow_pickle=False).astype(np.float32)
        ideal_iq = np.stack([np.load(flat / "ideal_IQ" / f"{sample_id}_{c}.npy", allow_pickle=False) for c in channels]).astype(np.float32)
        noisy_iq = np.stack([np.load(flat / "noise_IQ" / f"{sample_id}_{c}.npy", allow_pickle=False) for c in channels]).astype(np.float32)
        if ideal_depth.shape != (args.resolution, args.resolution):
            import cv2
            ideal_depth = cv2.resize(ideal_depth, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
            noise_depth = cv2.resize(noise_depth, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR)
            ideal_iq = np.stack([cv2.resize(x, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR) for x in ideal_iq])
            noisy_iq = np.stack([cv2.resize(x, (args.resolution, args.resolution), interpolation=cv2.INTER_LINEAR) for x in noisy_iq])
        ideal_iq = np.nan_to_num(ideal_iq, nan=0.0, posinf=0.0, neginf=0.0)
        noisy_iq = np.nan_to_num(noisy_iq, nan=0.0, posinf=0.0, neginf=0.0)
        amplitude = np.sqrt(noisy_iq[0::2] ** 2 + noisy_iq[1::2] ** 2).astype(np.float32)
        valid = np.isfinite(ideal_depth) & (ideal_depth > 0.1) & (ideal_depth < 9.9)
        # PBRT is nearly fully valid, while FLAT has invalid background. Keep
        # the same 15% missing-point rate within valid depth pixels.
        sample_cfg = argparse.Namespace(**vars(base_cfg))
        sample_cfg.hole_ratio = args.hole_ratio * float(valid.mean())
        hole, confidence = generate_holes(ideal_depth, amplitude.mean(axis=0), sample_cfg)
        hole &= valid
        noisy_with_holes = noisy_iq.copy()
        noisy_with_holes[:, hole] = 0.0
        depth_noisy = noise_depth.copy()
        depth_noisy[hole] = 0.0
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            sample_name=np.array(f"flat/0/{sample_id}"),
            depth_noisy=depth_noisy,
            gt_depth=ideal_depth,
            hole_mask=hole.astype(np.uint8),
            confidence=confidence.astype(np.float32),
            valid_mask=valid.astype(np.uint8),
            noisy_amplitude=np.sqrt(noisy_with_holes[0::2] ** 2 + noisy_with_holes[1::2] ** 2).astype(np.float32),
            noisy_amplitude_mean=np.sqrt(noisy_with_holes[0::2] ** 2 + noisy_with_holes[1::2] ** 2).mean(axis=0).astype(np.float32),
            noisy_iq=noisy_with_holes,
            ideal_iq=ideal_iq,
        )
        written += 1

    paths = sorted(cache_dir.glob("*.npz"))
    n = len(paths)
    train_end = int(round(n * args.train_ratio))
    val_end = int(round(n * (args.train_ratio + args.val_ratio)))
    splits = {"train": paths[:train_end], "val": paths[train_end:val_end], "test": paths[val_end:]}
    for name, values in splits.items():
        (out / f"{name}.txt").write_text("\n".join(str(p.resolve()) for p in values) + "\n")
    summary = {
        "protocol": "PBRT-matched amplitude-driven holes",
        "hole_ratio_scope": "valid_depth_pixels",
        "pseudo_gt": True,
        "flat_data_root": str(flat.resolve()),
        "depth_root": str(depth.resolve()),
        "output_root": str(out.resolve()),
        "samples": n,
        "written": written,
        "splits": {k: len(v) for k, v in splits.items()},
        "hole_ratio_target": args.hole_ratio,
        "hole_ratio_target_scope": "valid_depth_pixels",
        "block_size": args.block_size,
        "amp_percentile": args.amp_percentile,
        "low_amp_ratio": args.low_amp_ratio,
        "split_policy": "time-sorted contiguous blocks",
        "seed": args.seed,
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
