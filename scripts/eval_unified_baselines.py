#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unified_pbrt_dataset import UnifiedPbrtDataset


def add_metrics(total, prediction, target, hole, valid):
    finite = torch.isfinite(prediction) & torch.isfinite(target)
    masks = {
        "global": finite & valid,
        "hole": finite & valid & hole,
        "valid": finite & valid & (~hole),
    }
    error = torch.abs(prediction - target)
    for name, mask in masks.items():
        total[name + "_sum"] += float(error[mask].sum())
        total[name + "_sq_sum"] += float(error[mask].square().sum())
        total[name + "_count"] += int(mask.sum())


def finish_metrics(total):
    result = {}
    for name in ("global", "hole", "valid"):
        count = total[name + "_count"]
        result[name + "_mae_m"] = total[name + "_sum"] / count if count else None
        result[name + "_rmse_m"] = (total[name + "_sq_sum"] / count) ** 0.5 if count else None
        result[name + "_count"] = count
    return result


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def evaluate_lfrd2(args, loader, device):
    repo = Path("/data/pre_student/GJ/LFRD2")
    sys.path.insert(0, str(repo))
    import Loss
    from model.cplx import FracDiff

    model = FracDiff(args).to(device)
    state = torch.load(args.lfrd2_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    total = {
        k: 0.0
        for k in ("global_sum", "hole_sum", "valid_sum", "global_sq_sum", "hole_sq_sum", "valid_sq_sum")
    }
    total.update({k: 0 for k in ("global_count", "hole_count", "valid_count")})
    with torch.no_grad():
        for batch in loader:
            depth = batch["depth"].to(device) / 10.0
            amplitude = batch["amplitude"].to(device)
            confidence = batch["confidence"].to(device)
            output = model(depth, amplitude, confidence)["y_pred"][0] * 10.0
            add_metrics(total, output, batch["target"].to(device), batch["hole_mask"].to(device), batch["valid_mask"].to(device))
    return finish_metrics(total)


def evaluate_rgbd(args, loader, device):
    repo = Path("/data/pre_student/hcy/RGBD_imaging")
    sys.path.insert(0, str(repo))
    from srresnet_unet3 import _NetG

    model = _NetG().to(device)
    state = torch.load(args.rgbd_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    total = {
        k: 0.0
        for k in ("global_sum", "hole_sum", "valid_sum", "global_sq_sum", "hole_sq_sum", "valid_sq_sum")
    }
    total.update({k: 0 for k in ("global_count", "hole_count", "valid_count")})
    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"][:, :4].to(device)
            depth = batch["depth"].to(device)
            amplitude = batch["amplitude"].to(device)
            prediction = model(torch.cat((iq, depth / 10.0, amplitude), dim=1)) * 10.0
            add_metrics(total, prediction, batch["target"].to(device), batch["hole_mask"].to(device), batch["valid_mask"].to(device))
    return finish_metrics(total)


def main(args):
    device = resolve_device(args.device)
    dataset = UnifiedPbrtDataset(args.manifest, "test", args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    results = {
        "split": "test",
        "num_samples": len(dataset),
    }
    if args.model in ("lfrd2", "both"):
        results["lfrd2"] = evaluate_lfrd2(args, loader, device)
    if args.model in ("rgbd", "both"):
        results["rgbd"] = evaluate_rgbd(args, loader, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="output/full_pbrt_manifest_seed123.json")
    parser.add_argument("--lfrd2_checkpoint", default="output/lfrd2_full_pbrt/checkpoint_best_net.pth")
    parser.add_argument("--rgbd_checkpoint", default="output/rgbd_imaging_full_pbrt/checkpoint_best.pth")
    parser.add_argument("--output", default="output/pbrt100_depth_completion/rgbd_lfrd2/summary.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--model", choices=("lfrd2", "rgbd", "both"), default="both")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--prop_kernel", type=int, default=3)
    parser.add_argument("--conf_prop", action="store_true", default=True)
    parser.add_argument("--prop_time", type=int, default=6)
    parser.add_argument("--affinity", default="TGASS")
    parser.add_argument("--affinity_gamma", type=float, default=0.5)
    main(parser.parse_args())
