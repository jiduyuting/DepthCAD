#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


def sample_from_cache(data, path):
    if "sample_name" in data.files:
        value = data["sample_name"]
        if np.ndim(value) == 0:
            return str(value.item())
        return str(value)
    parts = path.parts
    return "/".join(parts[-3:]).replace(".npz", "")


def augment(cache_root, iq_root, output_root, overwrite=False):
    cache_root = Path(cache_root)
    iq_root = Path(iq_root)
    output_root = Path(output_root)
    paths = sorted(cache_root.rglob("*.npz"))
    missing = []
    processed = 0
    for path in tqdm(paths, desc=f"Augmenting {cache_root}"):
        relative = path.relative_to(cache_root)
        out_path = output_root / relative
        if out_path.exists() and not overwrite:
            processed += 1
            continue
        with np.load(path, allow_pickle=False) as data:
            sample = sample_from_cache(data, path)
            scene, view, frame = sample.split("/")
            iq_files = [iq_root / scene / view / f"{frame}_{channel}.npy" for channel in "ABCDEF"]
            if not all(p.exists() for p in iq_files):
                missing.append({"sample": sample, "cache": str(path)})
                continue
            arrays = [np.load(p).astype(np.float32) for p in iq_files]
            shape = data["gt_depth"].shape
            arrays = [
                array if array.shape == shape else cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                for array in arrays
            ]
            payload = {key: data[key] for key in data.files}
            payload["noisy_iq"] = np.stack(arrays, axis=0).astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **payload)
        tmp.replace(out_path)
        processed += 1
    return {"cache_root": str(cache_root), "output_root": str(output_root), "processed": processed, "missing": missing}


def main(args):
    result = {
        "train": augment(args.train_cache, args.iq_root, args.train_output, args.overwrite),
        "test": augment(args.test_cache, args.iq_root, args.test_output, args.overwrite),
    }
    Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {"processed": v["processed"], "missing_count": len(v["missing"])} for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12")
    parser.add_argument("--test_cache", default="depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123")
    parser.add_argument("--train_output", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq")
    parser.add_argument("--test_output", default="depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq")
    parser.add_argument("--iq_root", default="pbrt_dataset/data/noise_IQ")
    parser.add_argument("--report", default="output/full_pbrt_iq_cache_augmentation.json")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
