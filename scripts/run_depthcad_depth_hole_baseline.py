import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a depth-as-gray DepthCAD ControlNet baseline on depth=0 holes. "
            "This is a diagnostic baseline, not the original IQ-domain DepthCAD pipeline."
        )
    )
    parser.add_argument("--depth_dir", type=Path, default=Path("data/prepared_new_capture/all/depth_m"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/new_capture_depthcad_depth_hole_baseline"))
    parser.add_argument("--checkpoint", type=str, default="output/depthcad")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--infer_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth_min", type=float, default=0.5)
    parser.add_argument("--depth_max", type=float, default=4.5)
    parser.add_argument("--hole_depth_threshold", type=float, default=0.0)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    return parser.parse_args()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    mkdir(Path(path).parent)
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


def natural_key(path):
    stem = Path(path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


def normalize_depth(depth, depth_min, depth_max):
    depth = np.asarray(depth, dtype=np.float32)
    norm = (depth - float(depth_min)) / max(float(depth_max - depth_min), 1e-6)
    norm = np.nan_to_num(norm, nan=0.0, neginf=0.0, posinf=1.0)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 2.0 - 1.0).astype(np.float32)


def denormalize_depth(norm, depth_min, depth_max):
    norm = np.asarray(norm, dtype=np.float32)
    unit = np.clip((norm + 1.0) * 0.5, 0.0, 1.0)
    return (unit * float(depth_max - depth_min) + float(depth_min)).astype(np.float32)


def tensor_image_to_norm(image):
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    arr = arr / 255.0
    return (2.0 * arr - 1.0).astype(np.float32)


def save_visualization(path, stem, raw, hole, pred, hole_only, depth_min, depth_max):
    panels = [
        ("raw depth", raw, "viridis", depth_min, depth_max),
        ("hole mask", hole.astype(np.float32), "gray", 0.0, 1.0),
        ("DepthCAD depth-gray pred", pred, "viridis", depth_min, depth_max),
        ("hole-only merge", hole_only, "viridis", depth_min, depth_max),
        ("|hole-only - raw|", np.where(hole, np.nan, np.abs(hole_only - raw)), "magma", 0.0, 0.2),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    for ax, (title, image, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(stem)
    mkdir(Path(path).parent)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_depth_paths(depth_dir, samples, max_samples):
    paths = sorted(Path(depth_dir).glob("*.npy"), key=natural_key)
    if samples:
        wanted = {str(s) for s in samples}
        paths = [p for p in paths if p.stem in wanted]
    if max_samples is not None:
        paths = paths[: int(max_samples)]
    if not paths:
        raise FileNotFoundError(f"No matching .npy files found under {depth_dir}")
    return paths


def main():
    args = parse_args()
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow_cpu for a slow diagnostic smoke test.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(args.checkpoint, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        local_files_only=bool(args.local_files_only),
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    out_dir = Path(args.output_dir)
    for sub in ["pred_depth", "hole_only", "hole_mask", "visualizations"]:
        mkdir(out_dir / sub)

    rows = []
    paths = load_depth_paths(args.depth_dir, args.samples, args.max_samples)
    infer_size = int(args.infer_size)
    for index, depth_path in enumerate(paths):
        stem = depth_path.stem
        raw = np.load(depth_path).astype(np.float32)
        hole = (~np.isfinite(raw)) | (raw <= float(args.hole_depth_threshold))
        confidence = (~hole).astype(np.float32)
        raw_for_condition = raw.copy()
        raw_for_condition[hole] = float(args.depth_min)
        depth_norm = normalize_depth(raw_for_condition, args.depth_min, args.depth_max)
        depth_resized = cv2.resize(depth_norm, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        conf_resized = cv2.resize(confidence, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
        condition = np.stack([depth_resized, conf_resized], axis=0).astype(np.float32)
        condition_t = torch.from_numpy(condition).unsqueeze(0).to(device=device, dtype=dtype)
        generator = torch.Generator(device=device).manual_seed(int(args.seed) + index)
        image = pipe(
            "",
            image=condition_t,
            height=infer_size,
            width=infer_size,
            num_inference_steps=int(args.num_inference_steps),
            generator=generator,
        ).images[0]
        pred_norm = tensor_image_to_norm(image)
        pred_norm = cv2.resize(pred_norm, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred = denormalize_depth(pred_norm, args.depth_min, args.depth_max)
        hole_only = raw.copy()
        hole_only[hole] = pred[hole]
        hole_only = np.nan_to_num(hole_only, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        np.save(out_dir / "pred_depth" / f"{stem}_depthcad_depth_pred.npy", pred.astype(np.float32))
        np.save(out_dir / "hole_only" / f"{stem}_depthcad_depth_hole_only.npy", hole_only.astype(np.float32))
        np.save(out_dir / "hole_mask" / f"{stem}_hole_mask.npy", hole.astype(np.uint8))
        save_visualization(
            out_dir / "visualizations" / f"{stem}.png",
            stem,
            raw,
            hole,
            pred,
            hole_only,
            float(args.depth_min),
            float(args.depth_max),
        )

        rows.append(
            {
                "sample": stem,
                "input_depth": str(depth_path),
                "hole_ratio": float(hole.mean()),
                "pred_depth": str((out_dir / "pred_depth" / f"{stem}_depthcad_depth_pred.npy").resolve()),
                "hole_only": str((out_dir / "hole_only" / f"{stem}_depthcad_depth_hole_only.npy").resolve()),
                "visualization": str((out_dir / "visualizations" / f"{stem}.png").resolve()),
            }
        )
        print(f"[{index + 1}/{len(paths)}] {stem} hole={hole.mean():.3f}")

    save_json(
        out_dir / "summary.json",
        {
            "note": (
                "Diagnostic depth-as-gray DepthCAD baseline. This is not the original "
                "IQ-domain DepthCAD method."
            ),
            "depth_dir": str(Path(args.depth_dir).resolve()),
            "checkpoint": args.checkpoint,
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "device": device,
            "infer_size": int(args.infer_size),
            "num_inference_steps": int(args.num_inference_steps),
            "depth_min": float(args.depth_min),
            "depth_max": float(args.depth_max),
            "num_samples": len(rows),
            "rows": rows,
        },
    )
    print(f"Saved DepthCAD depth-hole diagnostic baseline to {out_dir}")


if __name__ == "__main__":
    main()
