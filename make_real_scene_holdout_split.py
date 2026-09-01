import argparse
import json
import os
import re
from glob import glob


def scene_name(stem):
    return str(stem).split("_", 1)[0]


def stem_sort_key(value):
    stem = os.path.splitext(os.path.basename(str(value)))[0]
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", stem)]


def collect_pairs(raw_dir, depth_dir):
    raw_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob(os.path.join(raw_dir, "*.npy"))
    }
    depth_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob(os.path.join(depth_dir, "*.npy"))
    }
    stems = sorted(set(raw_paths) & set(depth_paths), key=stem_sort_key)
    return [(stem, raw_paths[stem], depth_paths[stem]) for stem in stems]


def main():
    parser = argparse.ArgumentParser(description="Create a Real split with selected scene prefixes held out for val.")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument(
        "--train_scenes",
        type=str,
        nargs="+",
        default=None,
        help="Optional scene prefixes allowed in train. Defaults to all scenes not in val_scenes.",
    )
    parser.add_argument("--val_scenes", type=str, nargs="+", required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    val_scenes = {str(scene) for scene in args.val_scenes}
    train_scenes = None if args.train_scenes is None else {str(scene) for scene in args.train_scenes}
    if train_scenes is not None and train_scenes & val_scenes:
        raise ValueError(f"train_scenes and val_scenes overlap: {sorted(train_scenes & val_scenes)}")
    pairs = collect_pairs(args.raw_dir, args.depth_dir)
    stems = [pair[0] for pair in pairs]
    train = sorted(
        [
            stem
            for stem in stems
            if scene_name(stem) not in val_scenes
            and (train_scenes is None or scene_name(stem) in train_scenes)
        ],
        key=stem_sort_key,
    )
    val = sorted([stem for stem in stems if scene_name(stem) in val_scenes], key=stem_sort_key)
    if not train:
        raise ValueError("Train split is empty.")
    if not val:
        raise ValueError(f"Val split is empty for scenes {sorted(val_scenes)}.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(
            {
                "train": train,
                "val": val,
                "train_scenes": None if train_scenes is None else sorted(train_scenes),
                "val_scenes": sorted(val_scenes),
                "raw_dir": args.raw_dir,
                "depth_dir": args.depth_dir,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(
        json.dumps(
            {
                "train": len(train),
                "val": len(val),
                "train_scenes": None if train_scenes is None else sorted(train_scenes),
                "val_scenes": sorted(val_scenes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
