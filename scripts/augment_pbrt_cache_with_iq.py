#!/usr/bin/env python3
"""Copy PBRT depth caches and attach aligned noisy/ideal IQ tensors."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def sample_name(data, path):
    value = data.get("sample_name")
    if value is not None:
        return str(value.item() if value.shape == () else value)
    return "/".join(path.parts[-3:]).replace(".npz", "")


def load_iq(root, sample, shape):
    scene, view, frame = sample.split("/")
    channels = []
    for channel in "ABCDEF":
        path = Path(root) / scene / view / f"{frame}_{channel}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.load(path).astype(np.float32)
        if array.shape != shape:
            array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        channels.append(array)
    return np.stack(channels, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_cache", type=Path, required=True)
    parser.add_argument("--output_cache", type=Path, required=True)
    parser.add_argument("--noise_iq_root", type=Path, default=Path("pbrt_dataset/data/noise_IQ"))
    parser.add_argument("--ideal_iq_root", type=Path, default=Path("pbrt_dataset/data/ideal_IQ"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_paths = sorted(args.source_cache.rglob("*.npz"))
    if not source_paths:
        raise FileNotFoundError(f"No cache files under {args.source_cache}")
    processed = 0
    for source_path in source_paths:
        target_path = args.output_cache / source_path.relative_to(args.source_cache)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not args.overwrite:
            processed += 1
            continue
        with np.load(source_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        sample = sample_name(arrays, source_path)
        shape = tuple(arrays["gt_depth"].shape)
        arrays["noisy_iq"] = load_iq(args.noise_iq_root, sample, shape)
        arrays["ideal_iq"] = load_iq(args.ideal_iq_root, sample, shape)
        temporary = target_path.with_suffix(".npz.tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(target_path)
        processed += 1
    print(f"Augmented {processed} cache files: {args.output_cache}")


if __name__ == "__main__":
    main()
