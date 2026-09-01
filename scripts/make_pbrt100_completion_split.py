#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a PBRT depth-completion split JSON for the seed123 100-sample holdout."
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/full_pbrt_manifest_seed123_iq.json"),
        help="Optional manifest with samples.test. Falls back to scanning cache_root when absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pbrt100_depth_completion/split.json"),
    )
    parser.add_argument("--expected_count", type=int, default=100)
    return parser.parse_args()


def sample_digest(samples):
    return hashlib.sha256("\n".join(samples).encode("utf-8")).hexdigest()


def samples_from_manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", {}).get("test")
    if samples is None:
        raise ValueError(f"{path} does not contain samples.test")
    return [str(sample) for sample in samples]


def samples_from_cache(cache_root):
    return sorted(path.with_suffix("").relative_to(cache_root).as_posix() for path in cache_root.rglob("*.npz"))


def main():
    args = parse_args()
    cache_root = args.cache_root.resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)

    if args.manifest.is_file():
        samples = samples_from_manifest(args.manifest)
        source = str(args.manifest.resolve())
    else:
        samples = samples_from_cache(cache_root)
        source = str(cache_root)

    missing = [sample for sample in samples if not (cache_root / f"{sample}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cache files, first: {missing[0]}")
    if args.expected_count is not None and len(samples) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} samples, found {len(samples)}")

    result = {
        "train": [],
        "val": samples,
        "test": samples,
        "metadata": {
            "protocol": "pbrt100_seed123_depth_completion",
            "cache_root": str(cache_root),
            "source": source,
            "sample_count": len(samples),
            "sha256": sample_digest(samples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
