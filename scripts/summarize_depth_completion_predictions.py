import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from depth_completion_baselines.common import (
    MetricAccumulator,
    PBRTCompletionDataset,
    evaluate_prediction,
    save_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Score externally generated PBRT depth predictions.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--prediction_root", type=Path, required=True)
    parser.add_argument(
        "--index_json",
        type=Path,
        help="Uniformat index.json for sequential predictions such as 0000000000.png.",
    )
    parser.add_argument(
        "--prediction_format",
        choices=("sample_npy", "indexed_npy", "indexed_png"),
        default="sample_npy",
    )
    parser.add_argument(
        "--index_digits",
        type=int,
        default=10,
        help="Zero-padding width used by indexed prediction filenames.",
    )
    parser.add_argument("--cache_root", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq")
    parser.add_argument("--split_json", default="output/completionformer_full_pbrt/split.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction_scale", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_index(index_json):
    if index_json is None:
        raise ValueError("--index_json is required for indexed prediction formats")
    with index_json.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    return {entry["sample_id"]: index for index, entry in enumerate(entries)}


def prediction_path(args, sample_id, sample_index):
    if args.prediction_format == "sample_npy":
        return args.prediction_root / f"{sample_id}.npy"
    suffix = ".png" if args.prediction_format == "indexed_png" else ".npy"
    return args.prediction_root / f"{sample_index:0{args.index_digits}d}{suffix}"


def load_prediction(path):
    if path.suffix.lower() == ".png":
        return np.asarray(Image.open(path), dtype=np.float32)
    return np.load(path).squeeze().astype(np.float32)


def main():
    args = parse_args()
    dataset = PBRTCompletionDataset(args.cache_root, args.split_json, split=args.split, limit=args.limit)
    accumulator = MetricAccumulator()
    missing = []
    sample_indices = None
    if args.prediction_format != "sample_npy":
        sample_indices = load_index(args.index_json)
    for sample in dataset:
        sample_id = sample["sample_id"]
        sample_index = sample_indices.get(sample_id) if sample_indices is not None else None
        if sample_indices is not None and sample_index is None:
            missing.append(f"index entry for {sample_id}")
            continue
        path = prediction_path(args, sample_id, sample_index)
        if not path.exists():
            missing.append(str(path))
            continue
        prediction = load_prediction(path) * args.prediction_scale
        evaluate_prediction(accumulator, prediction, sample)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} predictions, first: {missing[0]}")
    save_summary(
        args.output,
        args.method,
        dataset,
        accumulator.summary(),
        prediction_root=str(args.prediction_root.resolve()),
        prediction_format=args.prediction_format,
        prediction_scale=args.prediction_scale,
    )


if __name__ == "__main__":
    main()
