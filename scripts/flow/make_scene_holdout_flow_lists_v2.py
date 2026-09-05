#!/usr/bin/env python3
"""Create scene-disjoint Flow lists from existing PBRT cache lists."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--holdout_scene", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    def scene_for(path):
        try:
            return Path(path).resolve().relative_to(args.cache_dir.resolve()).parts[0]
        except ValueError:
            return ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = [p for p in read(args.source_dir / "train.txt") if scene_for(p) != args.holdout_scene]
    val = [p for p in read(args.source_dir / "val.txt") if scene_for(p) != args.holdout_scene]
    # The standard seed123 test list is mixed and may omit an entire scene.
    test = [
        str(path.resolve())
        for path in sorted(args.cache_dir.rglob("*.npz"))
        if scene_for(path) == args.holdout_scene
    ]
    if not test:
        raise SystemExit(f"No cached samples found for holdout scene {args.holdout_scene!r}.")
    for name, values in (("train", train), ("val", val), ("test", test)):
        (args.output_dir / f"{name}.txt").write_text("\n".join(values) + "\n")
    print(f"holdout_scene={args.holdout_scene} train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
