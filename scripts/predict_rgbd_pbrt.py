"""Run the RGBD-imaging PBRT checkpoint on the PBRT test split."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

RGBD_REPO = Path("/data/pre_student/hcy/RGBD_imaging")
sys.path.insert(0, str(RGBD_REPO))
from srresnet_unet3 import _NetG  # noqa: E402
from train_rgbd_pbrt import load_input  # noqa: E402


def main(args):
    root = Path(args.dataset)
    output = Path(args.output)
    depth_out = output / "depth"
    png_out = output / "png"
    depth_out.mkdir(parents=True, exist_ok=True)
    png_out.mkdir(parents=True, exist_ok=True)
    scenes = [line.strip() for line in (root / "list" / args.split).read_text().splitlines() if line.strip()]
    if args.max_scenes is not None:
        scenes = scenes[:args.max_scenes]
    device = torch.device(args.device)
    model = _NetG()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for scene in tqdm(scenes, desc="PBRT inference"):
            (depth_out / scene).mkdir(parents=True, exist_ok=True)
            (png_out / scene).mkdir(parents=True, exist_ok=True)
            for frame in range(1, args.max_frames + 1):
                relative = Path(scene) / f"{frame}.npy"
                corr, amplitude = load_input(root / "noise" / relative)
                depth = np.load(root / "noise_depth" / relative).astype(np.float32)[None] / 10.0
                target = np.load(root / "gt_depth" / relative).astype(np.float32)
                tensor = torch.from_numpy(np.concatenate((corr, np.nan_to_num(depth), amplitude), axis=0))[None].to(device)
                prediction = model(tensor)[0, 0].cpu().numpy() * 10.0
                prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
                np.save(depth_out / relative, prediction.astype(np.float32))
                normalized = np.clip(prediction / max(float(args.png_scale), 1e-6), 0.0, 1.0)
                cv2.imwrite(str(png_out / relative.with_suffix(".png")), (normalized * 255).astype(np.uint8))
                valid = target > 0
                rows.append({
                    "sample": str(relative.with_suffix("")),
                    "mae": float(np.abs(prediction[valid] - target[valid]).mean()) if valid.any() else 0.0,
                    "rmse": float(np.sqrt(np.square(prediction[valid] - target[valid]).mean())) if valid.any() else 0.0,
                })
    summary = {"checkpoint": str(args.checkpoint), "split": args.split, "count": len(rows), "mae": float(np.mean([r["mae"] for r in rows])), "rmse": float(np.mean([r["rmse"] for r in rows]))}
    (output / "per_sample_metrics.json").write_text(json.dumps(rows, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/data/pre_student/hcy/datasets/pbrt")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="output/rgbd_imaging_pbrt_inference")
    parser.add_argument("--split", default="test.txt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--png-scale", type=float, default=10.0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-frames", type=int, default=250)
    main(parser.parse_args())
