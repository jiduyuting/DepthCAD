#!/usr/bin/env python3
"""Build one canonical PBRT train/validation/test manifest for all baselines."""

import argparse
import json
from pathlib import Path


def canonical_sample(value):
    path = Path(value)
    if path.suffix == ".npz":
        path = path.with_suffix("")
    parts = path.parts
    if len(parts) < 3:
        raise ValueError(f"Cannot parse PBRT sample from {value!r}")
    return "/".join(parts[-3:])


def read_split_json(path):
    payload = json.loads(Path(path).read_text())
    return {
        "train": sorted({canonical_sample(value) for value in payload["train"]}),
        "val": sorted({canonical_sample(value) for value in payload["val"]}),
    }


def read_holdout(cache_dir):
    return sorted(
        {
            canonical_sample(path.relative_to(cache_dir))
            for path in Path(cache_dir).rglob("*.npz")
        }
    )


def validate_disjoint(splits):
    names = list(splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = sorted(set(splits[left_name]) & set(splits[right_name]))
            if overlap:
                raise ValueError(f"{left_name}/{right_name} overlap: {overlap[:5]}")


def validate_sources(splits, train_cache, holdout_cache, iq_root):
    missing_cache = []
    missing_iq = []
    cache_roots = {
        **{sample: train_cache for sample in splits["train"]},
        **{sample: train_cache for sample in splits["val"]},
        **{sample: holdout_cache for sample in splits["test"]},
    }
    samples = [sample for values in splits.values() for sample in values]
    for sample in samples:
        scene, view, frame = sample.split("/")
        cache_path = Path(cache_roots[sample]) / scene / view / f"{frame}.npz"
        if not cache_path.exists():
            missing_cache.append(str(cache_path))
        for channel in "ABCDEF":
            iq_path = Path(iq_root) / scene / view / f"{frame}_{channel}.npy"
            if not iq_path.exists():
                missing_iq.append(str(iq_path))
    if missing_cache:
        raise FileNotFoundError(f"Missing cache files, first: {missing_cache[:3]}")
    if missing_iq:
        raise FileNotFoundError(f"Missing IQ files, first: {missing_iq[:3]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_json",
        type=Path,
        default=Path(
            "output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/split.json"
        ),
    )
    parser.add_argument(
        "--holdout_cache",
        type=Path,
        default=Path("depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123"),
    )
    parser.add_argument(
        "--train_cache",
        type=Path,
        default=Path("depth_completion_cache/depth_cache_0515_n1000_plane_r12"),
    )
    parser.add_argument(
        "--iq_root",
        type=Path,
        default=Path("pbrt_dataset/data/noise_IQ"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/unified_pbrt_manifest_seed123.json"),
    )
    args = parser.parse_args()

    splits = read_split_json(args.split_json)
    splits["test"] = read_holdout(args.holdout_cache)
    validate_disjoint(splits)
    validate_sources(splits, args.train_cache, args.holdout_cache, args.iq_root)

    payload = {
        "protocol": "unified_pbrt_seed123",
        "train_cache": str(args.train_cache),
        "holdout_cache": str(args.holdout_cache),
        "iq_root": str(args.iq_root),
        "split_json": str(args.split_json),
        "counts": {name: len(values) for name, values in splits.items()},
        "scenes": {
            name: sorted({sample.split("/")[0] for sample in values})
            for name, values in splits.items()
        },
        "samples": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
