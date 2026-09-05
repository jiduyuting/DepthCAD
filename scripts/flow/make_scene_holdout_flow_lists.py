#!/usr/bin/env python3
"""Create scene-disjoint Flow lists from existing PBRT cache lists."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--holdout_scene", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def scene_for(path, cache_dir):
    try:
        return Path(path).resolve().relative_to(cache_dir.resolve()).parts[0]
    except ValueError:
        return ""


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = [p for p in read(args.source_dir / "train.txt") if scene_for(p, args.cache_dir) != args.holdout_scene]
    val = [p for p in read(args.source_dir / "val.txt") if scene_for(p, args.cache_dir) != args.holdout_scene]
    test = [p for p in read(args.source_dir / "test.txt") if scene_for(p, args.cache_dir) == args.holdout_scene]
    if not test:
        raise SystemExit(f"No test samples found for holdout scene {args.holdout_scene!r}.")
    for name, values in (("train", train), ("val", val), ("test", test)):
        (args.output_dir / f"{name}.txt").write_text("\n".join(values) + "\n")
    print(f"holdout_scene={args.holdout_scene} train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
