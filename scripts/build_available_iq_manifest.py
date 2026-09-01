#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def samples(root):
    return sorted(path.relative_to(root).with_suffix("").as_posix() for path in root.rglob("*.npz"))


def main():
    parser = argparse.ArgumentParser(description="Build a manifest from the PBRT IQ caches available locally.")
    parser.add_argument("--train_cache", type=Path, default=Path("depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq"))
    parser.add_argument("--holdout_cache", type=Path, default=Path("depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq"))
    parser.add_argument("--iq_root", type=Path, default=Path("pbrt_dataset/data_256/noise_IQ"))
    parser.add_argument("--output", type=Path, default=Path("output/full_pbrt_manifest_available_iq.json"))
    parser.add_argument("--source_manifest", type=Path, default=Path("output/full_pbrt_manifest_seed123.json"))
    parser.add_argument("--expected_train_count", type=int, default=9900)
    args = parser.parse_args()

    train_all = samples(args.train_cache)
    test = samples(args.holdout_cache)
    if args.expected_train_count is not None and len(train_all) != args.expected_train_count:
        raise ValueError(
            f"Expected {args.expected_train_count} full-cache samples, found {len(train_all)}. "
            "Finish IQ cache augmentation before training."
        )
    overlap = set(train_all) & set(test)
    if overlap:
        raise ValueError(f"Train/holdout overlap: {sorted(overlap)[:3]}")
    if args.source_manifest.is_file():
        source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        source_samples = source.get("samples", {})
        train = list(source_samples.get("train", []))
        val = list(source_samples.get("val", []))
        source_test = list(source_samples.get("test", []))
        if len(train) + len(val) != len(train_all) or len(source_test) != len(test):
            raise ValueError("Source manifest counts do not match the available cache counts")
        if set(train + val) != set(train_all) or set(source_test) != set(test):
            raise ValueError("Source manifest samples do not match the available cache samples")
    else:
        split_at = max(1, int(len(train_all) * 0.9))
        train, val, source_test = train_all[:split_at], train_all[split_at:], test
    payload = {
        "protocol": "unified_pbrt_available_iq",
        "train_cache": str(args.train_cache),
        "holdout_cache": str(args.holdout_cache),
        "iq_root": str(args.iq_root),
        "counts": {"train": len(train), "val": len(val), "test": len(source_test)},
        "samples": {"train": train, "val": val, "test": source_test},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
