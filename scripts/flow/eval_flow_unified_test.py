#!/usr/bin/env python3
import _bootstrap
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from depth_restoration_backbones import build_depth_backbone
from train_depth_flow_restoration import evaluate
from train_depth_restoration import DepthRestorationCacheDataset


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    saved = checkpoint.get("args", {})
    manifest = json.loads(Path(args.manifest).read_text())
    cache_root = Path(manifest["holdout_cache"])
    paths = []
    for sample in manifest["samples"]["test"]:
        scene, view, frame = sample.split("/")
        paths.append(str(cache_root / scene / view / f"{frame}.npz"))

    dataset_kwargs = {
        "input_mode": saved.get("input_mode", "noisy_amp"),
        "include_hole_distance": saved.get("include_hole_distance", False),
        "anchor_mode": saved.get("anchor_mode", "noisy_ns"),
        "anchor_inpaint_radius": saved.get("anchor_inpaint_radius", 15),
        "norm_percentiles": saved.get("norm_percentiles", [5.0, 95.0]),
        "min_depth_scale": saved.get("min_depth_scale", 0.25),
        "clip_norm_depth": saved.get("clip_norm_depth", 8.0),
        "feature_percentile": saved.get("feature_percentile", 99.0),
        "feature_clip": saved.get("feature_clip", 3.0),
        "iq_clip": saved.get("iq_clip", 3.0),
    }
    dataset = DepthRestorationCacheDataset(paths, **dataset_kwargs)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.workers)
    condition_channels = dataset.input_channels
    model = build_depth_backbone(
        saved.get("backbone", "transformer_bottleneck"),
        in_channels=condition_channels + 1 + saved.get("time_channels", 16),
        base_channels=saved.get("base_channels", 32),
        out_channels=1,
        res_blocks=saved.get("res_blocks", 2),
        transformer_layers=saved.get("transformer_layers", 2),
        transformer_heads=saved.get("transformer_heads", 8),
        transformer_mlp_ratio=saved.get("transformer_mlp_ratio", 4.0),
        transformer_pool=saved.get("transformer_pool", 2),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    eval_args = argparse.Namespace(**saved)
    metrics = evaluate(model, loader, eval_args, torch.device(args.device))
    result = {"split": "test", "num_samples": len(dataset), "checkpoint": args.checkpoint, "metrics": metrics}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="output/full_pbrt_manifest_seed123.json")
    parser.add_argument("--checkpoint", default="output/depth_flow_full_pbrt_transformer/best.pt")
    parser.add_argument("--output", default="output/flow_unified_test_full.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    main(parser.parse_args())
