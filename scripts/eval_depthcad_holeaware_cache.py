import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from tqdm import tqdm

from IQToDepth import IQ_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DepthCAD-HoleAware model on cache files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="DepthCAD/ControlNet checkpoint path.")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--list", type=str, default=None, help="Optional txt file of .npz cache paths.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--infer_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--vis_max_samples", type=int, default=20)
    parser.add_argument("--vis_error_percentile", type=float, default=99.0)
    return parser.parse_args()


def read_cache_paths(cache_dir, list_path):
    cache_dir = Path(cache_dir)
    if list_path:
        paths = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = Path.cwd() / p
                paths.append(p)
        return paths
    return sorted(cache_dir.rglob("*.npz"))


def tensor_image_to_iq(image, scale):
    arr = np.array(image).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr / 255.0
    arr = 2.0 * arr - 1.0
    return arr * float(scale)


@torch.no_grad()
def infer_iq(pipe, noisy_iq, confidence, args, device, dtype):
    h, w = noisy_iq.shape[-2:]
    infer_size = int(args.infer_size)
    conf_resized = cv2.resize(confidence.astype(np.float32), (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    scale = max(float(np.nanmax(noisy_iq)), abs(float(np.nanmin(noisy_iq))), 1e-8)
    norm_iq = noisy_iq.astype(np.float32) / scale
    pred_iq = np.zeros((6, infer_size, infer_size), dtype=np.float32)

    for ch in range(6):
        noise_resized = cv2.resize(norm_iq[ch], (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        cond = np.stack([noise_resized, conf_resized], axis=0).astype(np.float32)
        cond = torch.from_numpy(cond).unsqueeze(0).to(device=device, dtype=dtype)
        generator = torch.Generator(device=device).manual_seed(int(args.seed) + ch)
        image = pipe(
            "",
            image=cond,
            height=infer_size,
            width=infer_size,
            num_inference_steps=int(args.num_inference_steps),
            generator=generator,
        ).images[0]
        pred_iq[ch] = tensor_image_to_iq(image, scale)

    out = np.zeros((6, h, w), dtype=np.float32)
    for ch in range(6):
        out[ch] = cv2.resize(pred_iq[ch], (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def mae_count(pred, target, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return float("nan"), 0
    return float(np.mean(np.abs(pred[valid] - target[valid]))), count


def region_metrics(pred, gt, hole_mask):
    valid_depth = (gt > 0.1) & (gt < 9.9) & np.isfinite(gt)
    hole = hole_mask > 0.5
    out = {}
    for name, mask in [
        ("global", valid_depth),
        ("hole", valid_depth & hole),
        ("valid", valid_depth & (~hole)),
    ]:
        value, count = mae_count(pred, gt, mask)
        out[f"{name}_mae"] = value
        out[f"{name}_count"] = count
    return out


def add_to_aggregate(agg, prefix, metrics):
    for region in ["global", "hole", "valid"]:
        mae = metrics[f"{region}_mae"]
        count = metrics[f"{region}_count"]
        key = f"{prefix}_{region}"
        if not np.isfinite(mae) or count == 0:
            return
        agg[key][0] += mae * count
        agg[key][1] += count


def finalize_aggregate(agg):
    out = {}
    for key, (total, count) in agg.items():
        out[f"{key}_count"] = int(count)
        out[f"{key}_mae"] = float(total / count) if count else float("nan")
    return out


def save_visualization(out_path, sample_name, gt, noisy, base, pred, hole_mask, args):
    import matplotlib.pyplot as plt

    valid = (gt > 0.1) & (gt < 9.9) & np.isfinite(gt)
    vmin = float(gt[valid].min()) if valid.any() else float(np.nanmin(gt))
    vmax = float(gt[valid].max()) if valid.any() else float(np.nanmax(gt))
    depth_kwargs = {"cmap": "turbo", "vmin": vmin, "vmax": vmax}
    err_maps = [np.abs(x - gt) for x in [noisy, base, pred]]
    finite = np.concatenate([e[np.isfinite(e)].reshape(-1) for e in err_maps if np.isfinite(e).any()])
    err_vmax = max(float(np.percentile(finite, args.vis_error_percentile)) if finite.size else 1.0, 1e-6)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    items = [
        (gt, "GT", depth_kwargs),
        (noisy, "Noisy", depth_kwargs),
        (base, "DepthCAD+Fill Base", depth_kwargs),
        (pred, "DepthCAD-HoleAware", depth_kwargs),
        (hole_mask, "Hole Mask", {"cmap": "gray", "vmin": 0, "vmax": 1}),
        (np.abs(noisy - gt), "|Noisy-GT|", {"cmap": "hot", "vmin": 0, "vmax": err_vmax}),
        (np.abs(base - gt), "|Base-GT|", {"cmap": "hot", "vmin": 0, "vmax": err_vmax}),
        (np.abs(pred - gt), "|Model-GT|", {"cmap": "hot", "vmin": 0, "vmax": err_vmax}),
    ]
    for ax, (image, title, kwargs) in zip(axes.reshape(-1), items):
        ax.imshow(image, **kwargs)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(sample_name)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = read_cache_paths(args.cache_dir, args.list)
    if args.num_samples is not None:
        paths = paths[: args.num_samples]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    depthcad = ControlNetModel.from_pretrained(args.checkpoint, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=depthcad,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device.startswith("cuda"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    aggregate = {
        f"{prefix}_{region}": [0.0, 0]
        for prefix in ["model", "noisy", "base"]
        for region in ["global", "hole", "valid"]
    }
    per_sample = []
    vis_saved = 0

    for path in tqdm(paths, desc="Evaluating DepthCAD-HoleAware"):
        with np.load(path) as data:
            if "noisy_iq" not in data.files:
                raise KeyError(f"{path} is missing noisy_iq; regenerate cache with --depth_cache_save_iq")
            sample_name = str(data["sample_name"].item() if data["sample_name"].shape == () else data["sample_name"])
            noisy_iq = data["noisy_iq"].astype(np.float32)
            confidence = data["confidence"].astype(np.float32)
            gt = data["gt_depth"].astype(np.float32)
            hole = data["hole_mask"].astype(np.float32)
            noisy_depth = data["depth_noisy"].astype(np.float32)
            base_depth = data["depth_base"].astype(np.float32)

        pred_iq = infer_iq(pipe, noisy_iq, confidence, args, device, dtype)
        pred_depth = IQ_to_depth(pred_iq, corr_save_path=None, depth_save_path=None).astype(np.float32)
        if pred_depth.shape != gt.shape:
            pred_depth = cv2.resize(pred_depth, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        row = {"sample_name": sample_name, "path": str(path)}
        for prefix, depth in [("model", pred_depth), ("noisy", noisy_depth), ("base", base_depth)]:
            metrics = region_metrics(depth, gt, hole)
            add_to_aggregate(aggregate, prefix, metrics)
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        per_sample.append(row)

        if args.visualize and vis_saved < args.vis_max_samples:
            safe = sample_name.replace("/", "_")
            save_visualization(
                str(out_dir / "visualizations" / f"vis_{safe}.png"),
                sample_name,
                gt,
                noisy_depth,
                base_depth,
                pred_depth,
                hole,
                args,
            )
            vis_saved += 1

    summary = {
        "aggregate": finalize_aggregate(aggregate),
        "checkpoint": args.checkpoint,
        "cache_dir": args.cache_dir,
        "path_source": "list" if args.list else "cache_dir",
        "num_samples": len(per_sample),
        "num_inference_steps": args.num_inference_steps,
        "infer_size": args.infer_size,
        "visualized_samples": vis_saved,
    }
    with open(out_dir / "per_sample_results.json", "w") as f:
        json.dump(per_sample, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved eval results to {out_dir}")


if __name__ == "__main__":
    main()
