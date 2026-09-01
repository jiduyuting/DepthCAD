import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a CompletionFormer split for the full PBRT cache."
    )
    parser.add_argument(
        "--source_split",
        type=Path,
        default=Path("output/depth_flow_full_pbrt_iq/split.json"),
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/completionformer_full_pbrt/split.json"),
    )
    return parser.parse_args()


def normalize_sample(item, cache_root):
    path = Path(item)
    if path.suffix == ".npz":
        if path.is_absolute():
            path = path.relative_to(cache_root)
        else:
            parts = path.parts
            if cache_root.name in parts:
                path = Path(*parts[parts.index(cache_root.name) + 1 :])
        path = path.with_suffix("")
    return path.as_posix()


def split_digest(train, test):
    payload = "\n".join(["[train]", *train, "[test]", *test])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    args = parse_args()
    cache_root = args.cache_root.resolve()
    with args.source_split.open("r", encoding="utf-8") as handle:
        source = json.load(handle)

    source_samples = source.get("samples", source)
    train = [normalize_sample(item, cache_root) for item in source_samples["train"]]
    if "samples" in source:
        # The unified manifest keeps the 100-sample holdout in a separate cache.
        # CompletionFormer training must validate against the full training cache,
        # so use the 990-sample validation split here; PBRT100 evaluation uses a
        # separate holdout split later.
        test_source = source_samples.get("val")
    else:
        test_source = source_samples.get("test", source_samples.get("val"))
    if test_source is None:
        raise ValueError("Source split needs a 'test' or 'val' list")
    test = [normalize_sample(item, cache_root) for item in test_source]

    if set(train) & set(test):
        raise ValueError("Train and test samples overlap")

    missing = [
        sample
        for sample in train + test
        if not (cache_root / f"{sample}.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} cache files, first: {missing[0]}")

    result = {
        "train": train,
        "test": test,
        "metadata": {
            "cache_root": str(cache_root),
            "source_split": str(args.source_split.resolve()),
            "train_count": len(train),
            "test_count": len(test),
            "sha256": split_digest(train, test),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
