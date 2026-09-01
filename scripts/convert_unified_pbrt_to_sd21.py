#!/usr/bin/env python3
"""Export unified PBRT cache samples in the original SD2.1 dataset layout."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


CHANNELS = "ABCDEF"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("output/full_pbrt_manifest_seed123.json"))
    parser.add_argument("--ideal_iq_root", type=Path, default=Path("pbrt_dataset/data_256/ideal_IQ"))
    parser.add_argument("--noise_iq_root", type=Path, default=Path("pbrt_dataset/data_256/noise_IQ"))
    parser.add_argument("--train_cache", type=Path, default=Path("depth_completion_cache/depth_cache_full_pbrt_plane_r12"))
    parser.add_argument("--holdout_cache", type=Path, default=Path("depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq"))
    parser.add_argument("--output_root", type=Path, default=Path("pbrt_dataset/data"))
    parser.add_argument("--prefix", default="sd21_full_pbrt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(source, target, overwrite):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()
    target.symlink_to(Path(os.path.relpath(source, target.parent)))


def export_split(name, samples, cache_root, args):
    ideal_dir = args.output_root / f"ideal_IQ_{args.prefix}_{name}"
    noise_dir = args.output_root / f"noise_IQ_{args.prefix}_{name}"
    confidence_dir = args.output_root / f"confidence_{args.prefix}_{name}"
    frame_count = 0
    channel_count = 0
    for sample in samples:
        scene, view, frame = sample.split("/")
        cache_path = cache_root / scene / view / f"{frame}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        ideal_sample = args.ideal_iq_root / scene / view
        noise_sample = args.noise_iq_root / scene / view
        with np.load(cache_path, allow_pickle=False) as data:
            confidence = data["confidence"].astype(np.float32)
        confidence_path = confidence_dir / scene / view / f"{frame}.npy"
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not confidence_path.exists():
            temporary = confidence_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, confidence)
            temporary.replace(confidence_path)
        for channel in CHANNELS:
            ideal_source = ideal_sample / f"{frame}_{channel}.npy"
            noise_source = noise_sample / f"{frame}_{channel}.npy"
            if not ideal_source.is_file():
                raise FileNotFoundError(ideal_source)
            if not noise_source.is_file():
                raise FileNotFoundError(noise_source)
            link_or_copy(ideal_source, ideal_dir / scene / view / ideal_source.name, args.overwrite)
            link_or_copy(noise_source, noise_dir / scene / view / noise_source.name, args.overwrite)
            channel_count += 1
        frame_count += 1
    return {
        "frames": frame_count,
        "channels": channel_count,
        "ideal_dir": str(ideal_dir),
        "noise_dir": str(noise_dir),
        "confidence_dir": str(confidence_dir),
    }


def main():
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = payload["samples"]
    train_cache = args.train_cache
    holdout_cache = args.holdout_cache
    outputs = {
        "train": export_split("train", samples["train"], train_cache, args),
        "val": export_split("val", samples["val"], train_cache, args),
        "test": export_split("test", samples["test"], holdout_cache, args),
    }
    metadata = {
        "source_manifest": str(args.manifest.resolve()),
        "counts": {name: len(values) for name, values in samples.items()},
        "outputs": outputs,
        "layout": "ideal_IQ / noise_IQ per channel A-F, confidence per frame",
    }
    metadata_path = args.output_root / f"{args.prefix}_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
