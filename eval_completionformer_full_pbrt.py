import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CompletionFormer on the unified full PBRT split."
    )
    parser.add_argument("--completionformer_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq"),
    )
    parser.add_argument(
        "--split_json",
        type=Path,
        default=Path("output/completionformer_full_pbrt/split.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/completionformer_full_pbrt/eval"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def add_region(stats, name, prediction, target, mask):
    mask = mask & torch.isfinite(prediction) & torch.isfinite(target)
    count = int(mask.sum().item())
    if count == 0:
        return
    error = prediction[mask] - target[mask]
    stats[name]["count"] += count
    stats[name]["abs_sum"] += float(error.abs().sum().item())
    stats[name]["sq_sum"] += float(error.square().sum().item())


def summarize(stats):
    result = {}
    for name, values in stats.items():
        count = values["count"]
        result[name] = {
            "count": count,
            "mae_m": values["abs_sum"] / count,
            "rmse_m": (values["sq_sum"] / count) ** 0.5,
        }
    return result


def main():
    args = parse_args()
    completionformer_src = (args.completionformer_root / "src").resolve()
    sys.path.insert(0, str(REPO_ROOT / "integrations" / "completionformer" / "compat"))
    sys.path.insert(0, str(completionformer_src))
    sys.path.insert(0, str(REPO_ROOT / "integrations" / "completionformer"))

    from model.completionformer import CompletionFormer
    from pbrtfull import PBRTFull

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["args"]
    model_args.dir_data = str(args.cache_root.resolve())
    model_args.split_json = str(args.split_json.resolve())
    model_args.augment = False

    previous_cwd = Path.cwd()
    os.chdir(completionformer_src)
    try:
        model = CompletionFormer(model_args)
    finally:
        os.chdir(previous_cwd)
    model.load_state_dict(checkpoint["net"], strict=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = PBRTFull(model_args, "test")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    stats = defaultdict(lambda: {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0})
    prediction_dir = args.output_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for sample in loader:
            tensors = {
                key: value.to(device, non_blocking=True)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            prediction = model(tensors)["pred"]
            target = tensors["gt"]
            valid = tensors["valid_mask"].bool()
            hole = tensors["hole_mask"].bool()

            add_region(stats, "global", prediction, target, valid)
            add_region(stats, "hole", prediction, target, valid & hole)
            add_region(stats, "observed", prediction, target, valid & ~hole)

            if args.save_predictions:
                sample_id = sample["idx"][0]
                output_path = prediction_dir / f"{sample_id}.npy"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, prediction[0, 0].cpu().numpy().astype(np.float32))

    result = {
        "method": "CompletionFormer",
        "checkpoint": str(args.checkpoint.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "split_json": str(args.split_json.resolve()),
        "samples": len(dataset),
        "metrics": summarize(stats),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
