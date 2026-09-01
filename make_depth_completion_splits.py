import argparse
import json
import os
import random
from glob import glob

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/val lists for depth completion caches while excluding holdout samples."
    )
    parser.add_argument("--cache_dir", type=str, action="append", required=True,
                        help="Candidate cache directory used for train/val. Can be provided multiple times.")
    parser.add_argument("--dedupe", action="store_true", default=False,
                        help="Keep only the first occurrence when multiple cache dirs contain the same sample_name.")
    parser.add_argument("--holdout_cache_dir", type=str, action="append", default=[],
                        help="Cache directory to exclude from train/val. Can be provided multiple times.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for train.txt, val.txt, holdout.txt, and summary.json.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation ratio after holdout exclusion.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_npz_paths(cache_dir):
    return sorted(glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True))


def fallback_sample_name(path, cache_dir):
    rel = os.path.relpath(path, cache_dir)
    return os.path.splitext(rel)[0].replace(os.sep, "/")


def read_sample_name(path, cache_dir):
    try:
        with np.load(path, allow_pickle=False) as data:
            if "sample_name" in data:
                value = data["sample_name"]
                if value.shape == ():
                    return str(value.item())
                return str(value)
    except Exception:
        pass
    return fallback_sample_name(path, cache_dir)


def write_list(path, values):
    with open(path, "w") as f:
        for value in values:
            f.write(value + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    candidate_records = []
    duplicate_records = []
    seen_candidate_names = set()
    per_cache_counts = {}
    for cache_dir in args.cache_dir:
        paths = collect_npz_paths(cache_dir)
        per_cache_counts[cache_dir] = len(paths)
        for path in paths:
            sample_name = read_sample_name(path, cache_dir)
            is_duplicate = args.dedupe and sample_name in seen_candidate_names
            record = {
                "path": path,
                "sample_name": sample_name,
                "cache_dir": cache_dir,
                "is_duplicate": is_duplicate,
            }
            candidate_records.append(record)
            if is_duplicate:
                duplicate_records.append(record)
            else:
                seen_candidate_names.add(sample_name)

    if not candidate_records:
        raise FileNotFoundError(f"No .npz files found under {args.cache_dir}")

    holdout_paths = []
    holdout_names = set()
    for holdout_dir in args.holdout_cache_dir:
        paths = collect_npz_paths(holdout_dir)
        holdout_paths.extend(paths)
        for path in paths:
            holdout_names.add(read_sample_name(path, holdout_dir))

    kept_paths = []
    excluded_paths = []
    for record in candidate_records:
        if record["is_duplicate"]:
            continue
        if record["sample_name"] in holdout_names:
            excluded_paths.append(record["path"])
        else:
            kept_paths.append(record["path"])

    if not kept_paths:
        raise ValueError("All candidate samples were excluded by the holdout set.")

    rng = random.Random(args.seed)
    rng.shuffle(kept_paths)
    val_count = max(1, int(round(len(kept_paths) * args.val_ratio))) if len(kept_paths) > 1 else 0
    val_paths = kept_paths[:val_count]
    train_paths = kept_paths[val_count:]
    if not train_paths and val_paths:
        train_paths, val_paths = val_paths, []

    outputs = {
        "train_list": os.path.join(args.output_dir, "train.txt"),
        "val_list": os.path.join(args.output_dir, "val.txt"),
        "holdout_list": os.path.join(args.output_dir, "holdout.txt"),
        "excluded_overlap_list": os.path.join(args.output_dir, "excluded_holdout_overlap.txt"),
        "summary": os.path.join(args.output_dir, "summary.json"),
    }
    write_list(outputs["train_list"], train_paths)
    write_list(outputs["val_list"], val_paths)
    write_list(outputs["holdout_list"], holdout_paths)
    write_list(outputs["excluded_overlap_list"], excluded_paths)

    summary = {
        "cache_dir": args.cache_dir,
        "per_cache_counts": per_cache_counts,
        "dedupe": args.dedupe,
        "holdout_cache_dir": args.holdout_cache_dir,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "candidate_count": len(candidate_records),
        "duplicate_count": len(duplicate_records),
        "candidate_after_dedupe_count": len(candidate_records) - len(duplicate_records),
        "holdout_count": len(holdout_paths),
        "excluded_overlap_count": len(excluded_paths),
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "outputs": outputs,
    }
    with open(outputs["summary"], "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
