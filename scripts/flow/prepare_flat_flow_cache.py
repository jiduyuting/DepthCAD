#!/usr/bin/env python3
"""Convert paired FLAT data into the cache format used by depth Flow."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

CHANNELS = "ABCDEF"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_data_root", type=Path, default=Path("flat_dataset/data"))
    parser.add_argument("--depth_root", type=Path, default=Path("/data/pre_student/hcy/ControlNet/data"))
    parser.add_argument("--output_root", type=Path, default=Path("depth_completion_cache/flat_flow_pseudo_gt"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize(array, resolution, interpolation=cv2.INTER_LINEAR):
    if array.shape == (resolution, resolution):
        return array.astype(np.float32, copy=False)
    return cv2.resize(array.astype(np.float32), (resolution, resolution), interpolation=interpolation)


def load_iq(root, sample_id, resolution):
    channels = []
    for channel in CHANNELS:
        path = root / f"{sample_id}_{channel}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        channels.append(resize(np.load(path, allow_pickle=False), resolution))
    return np.stack(channels, axis=0).astype(np.float32)


def stable_bucket(sample_id, seed):
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def main():
    args = parse_args()
    ideal_iq_root = args.flat_data_root / "ideal_IQ"
    noise_iq_root = args.flat_data_root / "noise_IQ"
    confidence_root = args.flat_data_root / "confidence"
    ideal_depth_root = args.depth_root / "ideal_depth"
    noise_depth_root = args.depth_root / "noise_depth"
    ids = sorted(
        path.name[:-6]
        for path in noise_iq_root.glob("*_A.npy")
        if (ideal_iq_root / path.name).is_file()
        and (ideal_depth_root / f"{path.name[:-6]}.npy").is_file()
        and (noise_depth_root / f"{path.name[:-6]}.npy").is_file()
    )
    if not ids:
        raise SystemExit("No paired FLAT IQ/depth samples found.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = []
    for sample_id in ids:
        output = output_root / "flat" / "0" / f"{sample_id}.npz"
        if output.exists() and not args.overwrite:
            continue
        try:
            ideal_depth = resize(np.load(ideal_depth_root / f"{sample_id}.npy", allow_pickle=False), args.resolution)
            noise_depth = resize(np.load(noise_depth_root / f"{sample_id}.npy", allow_pickle=False), args.resolution)
            noisy_iq = load_iq(noise_iq_root, sample_id, args.resolution)
            ideal_iq = load_iq(ideal_iq_root, sample_id, args.resolution)
            confidence_path = confidence_root / f"{sample_id}.npy"
            confidence = resize(np.load(confidence_path, allow_pickle=False), args.resolution) if confidence_path.is_file() else np.isfinite(noise_depth).astype(np.float32)
            valid = np.isfinite(ideal_depth) & (ideal_depth > 0.1) & (ideal_depth < 9.9)
            hole = ((~np.isfinite(noise_depth)) | (noise_depth <= 0.1)) & valid
            noisy_iq = np.nan_to_num(noisy_iq, nan=0.0, posinf=0.0, neginf=0.0)
            ideal_iq = np.nan_to_num(ideal_iq, nan=0.0, posinf=0.0, neginf=0.0)
            amplitude = np.sqrt(noisy_iq[0::2] ** 2 + noisy_iq[1::2] ** 2).astype(np.float32)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                sample_name=np.array(f"flat/0/{sample_id}"),
                depth_noisy=noise_depth.astype(np.float32),
                gt_depth=ideal_depth.astype(np.float32),
                hole_mask=hole.astype(np.uint8),
                confidence=np.clip(confidence, 0.0, 1.0).astype(np.float32),
                valid_mask=valid.astype(np.uint8),
                noisy_amplitude=amplitude,
                noisy_amplitude_mean=amplitude.mean(axis=0).astype(np.float32),
                noisy_iq=noisy_iq,
                ideal_iq=ideal_iq,
            )
            written += 1
        except (OSError, ValueError, FileNotFoundError) as exc:
            skipped.append({"sample_id": sample_id, "reason": str(exc)})

    paths = sorted((output_root / "flat" / "0").glob("*.npz"))
    splits = {"train": [], "val": [], "test": []}
    for path in paths:
        bucket = stable_bucket(path.stem, args.seed)
        split = "test" if bucket < args.test_ratio else "val" if bucket < args.test_ratio + args.val_ratio else "train"
        splits[split].append(str(path.resolve()))
    for split, split_paths in splits.items():
        (output_root / f"{split}.txt").write_text("\n".join(split_paths) + ("\n" if split_paths else ""))
    summary = {
        "data_root": str(args.flat_data_root.resolve()),
        "depth_root": str(args.depth_root.resolve()),
        "output_root": str(output_root.resolve()),
        "pseudo_gt": True,
        "resolution": args.resolution,
        "paired_ids": len(ids),
        "cached": len(paths),
        "written": written,
        "skipped": skipped,
        "splits": {key: len(value) for key, value in splits.items()},
        "seed": args.seed,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
