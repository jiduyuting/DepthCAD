import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from depth_completion_baselines.common import PBRTCompletionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export full PBRT to OMNI-DC/DMD3C uniformat npy files.")
    parser.add_argument("--cache_root", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq")
    parser.add_argument("--split_json", default="output/completionformer_full_pbrt/split.json")
    parser.add_argument("--split", choices=("train", "test", "val"), default="test")
    parser.add_argument("--output_dir", type=Path, default=Path("output/depth_completion_baselines/uniformat_full_pbrt_val"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = PBRTCompletionDataset(args.cache_root, args.split_json, split=args.split, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for sample in dataset:
        output_path = args.output_dir / f"{len(index):06d}.npy"
        if args.overwrite or not output_path.exists():
            payload = {
                "rgb": (sample["image"].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8),
                "dep": sample["sparse_depth"].numpy().astype(np.float32),
                "gt": sample["target"].numpy().astype(np.float32),
                "K": sample["intrinsics"].numpy().astype(np.float32),
                "hole_mask": sample["hole_mask"].numpy().astype(np.uint8),
                "valid_mask": sample["valid_mask"].numpy().astype(np.uint8),
                "sample_id": sample["sample_id"],
            }
            np.save(output_path, payload, allow_pickle=True)
        index.append({"file": output_path.name, "sample_id": sample["sample_id"]})
    with (args.output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    print(f"Exported {len(index)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
