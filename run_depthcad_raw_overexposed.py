#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from depth_estimator import DepthEstimator
from pbrt_dataset.preprocess import compute_gradient_confidence, load_raw as load_raw_pbrt


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    mkdir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def raw_candidates(raw_dir):
    out = []
    for path in sorted(Path(raw_dir).glob("*.npy")):
        try:
            arr = np.load(path, mmap_mode="r")
            shape = tuple(arr.shape)
            dtype = str(arr.dtype)
        except Exception as exc:
            out.append({"path": path, "usable": False, "reason": f"load failed: {exc}"})
            continue
        usable = len(shape) == 3 and shape[0] == 9
        reason = "ok" if usable else f"skip non-(9,H,W) raw: shape={shape}, dtype={dtype}"
        out.append({"path": path, "usable": usable, "shape": shape, "dtype": dtype, "reason": reason})
    return out


def scale_iq(iq, mode="percentile", percentile=99.5):
    iq = np.asarray(iq, dtype=np.float32)
    abs_iq = np.abs(iq[np.isfinite(iq)])
    if abs_iq.size == 0:
        return iq, 1.0
    if mode == "max":
        scale = float(max(np.max(abs_iq), 1e-8))
    elif mode == "percentile":
        scale = float(max(np.percentile(abs_iq, float(percentile)), 1e-8))
    else:
        raise ValueError(f"unknown scale mode: {mode}")
    return (iq / scale).astype(np.float32), scale


def saturation_mask(raw9, saturation_value, target_size):
    raw9 = np.asarray(raw9)
    mask = np.any(raw9 >= float(saturation_value), axis=0)
    target_h, target_w = target_size
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def image_limits(arrays, mask=None):
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        v = arr[np.isfinite(arr)] if mask is None else arr[mask & np.isfinite(arr)]
        if v.size:
            vals.append(v)
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def save_prepare_visualization(path, stem, raw_depth, confidence, sat_mask):
    vmin, vmax = image_limits([raw_depth], np.isfinite(raw_depth) & (raw_depth > 0))
    panels = [
        ("raw IQ -> depth", raw_depth, "viridis", vmin, vmax),
        ("saturation mask\nany channel >= limit", sat_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("confidence", confidence, "gray", 0.0, 1.0),
        ("depth hidden at saturation", np.where(sat_mask, np.nan, raw_depth), "viridis", vmin, vmax),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, (title, image, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(stem)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_result_visualization(path, stem, raw_depth, pred_depth, final_depth, confidence, sat_mask):
    valid = np.isfinite(raw_depth) & (raw_depth > 0) & (~sat_mask)
    vmin, vmax = image_limits([raw_depth, pred_depth, final_depth], valid)
    delta = np.abs(pred_depth - raw_depth)
    dmax = image_limits([delta], sat_mask)[1]
    panels = [
        ("raw IQ -> depth", raw_depth, "viridis", vmin, vmax),
        ("saturation mask", sat_mask.astype(np.float32), "gray", 0.0, 1.0),
        ("confidence", confidence, "gray", 0.0, 1.0),
        ("DepthCAD pred", pred_depth, "viridis", vmin, vmax),
        ("hole-only merge", final_depth, "viridis", vmin, vmax),
        ("|pred-raw|\nat saturation", np.where(sat_mask, delta, np.nan), "magma", 0.0, dmax),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (title, image, cmap, lo, hi) in zip(axes.ravel(), panels):
        im = ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(stem)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def load_and_prepare(raw_path, target_size, args, estimator):
    raw9 = np.load(raw_path).astype(np.float32)
    iq = load_raw_pbrt(str(raw_path), target_size=target_size, sqrt_in=True)
    raw_depth = estimator.process(iq).astype(np.float32)
    conf = compute_gradient_confidence(raw_depth).astype(np.float32)
    sat = saturation_mask(raw9, args.saturation_value, target_size)
    if args.saturation_conf_zero:
        conf = conf.copy()
        conf[sat] = 0.0
    iq_norm, scale = scale_iq(iq, mode=args.scale_mode, percentile=args.scale_percentile)
    return raw9, iq, iq_norm, raw_depth, conf, sat, scale


def run_depthcad_model(prepared, args):
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Re-run on a GPU machine, or pass --allow_cpu for a very slow smoke test."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    depthcad = ControlNetModel.from_pretrained(args.depthcad_path, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=depthcad,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            print(f"WARNING: xformers not enabled: {exc}")
        pipe.to(device)
    else:
        pipe.to(device)

    estimator = DepthEstimator()
    results = []
    for item in prepared:
        stem = item["stem"]
        iq_norm = np.load(item["iq_norm_path"])
        confidence = np.load(item["confidence_path"])
        scale = float(item["scale"])
        pred_iqs = np.zeros_like(iq_norm, dtype=np.float32)

        infer_h = infer_w = int(args.infer_size)
        conf_resized = cv2.resize(confidence, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        generator = torch.Generator(device=device).manual_seed(int(args.seed))
        for ch in range(6):
            noise_resized = cv2.resize(iq_norm[ch], (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            guidance = np.stack([noise_resized, conf_resized], axis=0).astype(np.float32)
            guidance_t = torch.from_numpy(guidance).unsqueeze(0).to(device=device, dtype=dtype)
            image = pipe(
                "",
                num_inference_steps=int(args.num_inference_steps),
                generator=generator,
                image=guidance_t,
                height=infer_h,
                width=infer_w,
            ).images[0]
            pred = np.mean(np.asarray(image), axis=2).astype(np.float32) / 255.0
            pred = 2.0 * pred - 1.0
            pred = cv2.resize(pred, (args.target_size[1], args.target_size[0]), interpolation=cv2.INTER_LINEAR)
            pred_iqs[ch] = pred * scale

        pred_depth = estimator.process(pred_iqs).astype(np.float32)
        raw_depth = np.load(item["raw_depth_path"])
        sat = np.load(item["saturation_mask_path"]).astype(bool)
        final_depth = raw_depth.copy()
        final_depth[sat] = pred_depth[sat]

        pred_iq_path = Path(args.output_dir) / "pred_iq" / f"{stem}_depthcad_pred_iq.npy"
        pred_depth_path = Path(args.output_dir) / "outputs" / "depthcad_raw_overexposed" / f"{stem}_depthcad_raw_overexposed.npy"
        final_path = Path(args.output_dir) / "outputs" / "depthcad_raw_overexposed_hole_only" / f"{stem}_depthcad_raw_overexposed_hole_only.npy"
        vis_path = Path(args.output_dir) / "visualizations" / f"{stem}_depthcad_raw_overexposed.png"
        for p in [pred_iq_path, pred_depth_path, final_path, vis_path]:
            mkdir(p.parent)
        np.save(pred_iq_path, pred_iqs)
        np.save(pred_depth_path, pred_depth)
        np.save(final_path, final_depth)
        save_result_visualization(vis_path, stem, raw_depth, pred_depth, final_depth, confidence, sat)
        results.append(
            {
                **item,
                "pred_iq_path": str(pred_iq_path),
                "pred_depth_path": str(pred_depth_path),
                "final_depth_path": str(final_path),
                "visualization": str(vis_path),
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="raw")
    parser.add_argument("--output_dir", default="output/raw_overexposed_depthcad")
    parser.add_argument("--depthcad_path", default="output/depthcad")
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--target_size", type=int, nargs=2, default=[240, 320])
    parser.add_argument("--infer_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--stems", nargs="*", default=None)
    parser.add_argument("--saturation_value", type=float, default=65535.0)
    parser.add_argument("--saturation_conf_zero", action="store_true", default=True)
    parser.add_argument("--scale_mode", choices=["percentile", "max"], default="percentile")
    parser.add_argument("--scale_percentile", type=float, default=99.5)
    parser.add_argument("--run_model", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    for sub in ["prepared/iq", "prepared/iq_norm", "prepared/raw_depth", "prepared/confidence", "prepared/saturation_mask", "visualizations_prepare"]:
        mkdir(out_dir / sub)

    selected = []
    skipped = []
    wanted = set(args.stems or [])
    for item in raw_candidates(args.raw_dir):
        path = item["path"]
        stem = path.stem
        if wanted and stem not in wanted:
            continue
        if not item["usable"]:
            skipped.append({"stem": stem, **{k: str(v) for k, v in item.items() if k != "path"}, "path": str(path)})
            continue
        selected.append(path)
        if not wanted and len(selected) >= int(args.max_samples):
            break

    estimator = DepthEstimator()
    prepared = []
    for path in selected:
        stem = path.stem
        raw9, iq, iq_norm, raw_depth, confidence, sat, scale = load_and_prepare(
            path, tuple(args.target_size), args, estimator
        )
        iq_path = out_dir / "prepared" / "iq" / f"{stem}_iq6.npy"
        iq_norm_path = out_dir / "prepared" / "iq_norm" / f"{stem}_iq6_norm.npy"
        raw_depth_path = out_dir / "prepared" / "raw_depth" / f"{stem}_raw_depth.npy"
        conf_path = out_dir / "prepared" / "confidence" / f"{stem}_confidence.npy"
        sat_path = out_dir / "prepared" / "saturation_mask" / f"{stem}_saturation_mask.npy"
        vis_path = out_dir / "visualizations_prepare" / f"{stem}_prepare.png"
        np.save(iq_path, iq.astype(np.float32))
        np.save(iq_norm_path, iq_norm.astype(np.float32))
        np.save(raw_depth_path, raw_depth.astype(np.float32))
        np.save(conf_path, confidence.astype(np.float32))
        np.save(sat_path, sat.astype(np.uint8))
        save_prepare_visualization(vis_path, stem, raw_depth, confidence, sat)
        prepared.append(
            {
                "stem": stem,
                "raw_path": str(path.resolve()),
                "raw_shape": list(raw9.shape),
                "raw_min": float(np.min(raw9)),
                "raw_p50": float(np.percentile(raw9, 50)),
                "raw_p99": float(np.percentile(raw9, 99)),
                "raw_max": float(np.max(raw9)),
                "saturation_ratio_any_channel": float(np.mean(sat)),
                "iq_path": str(iq_path.resolve()),
                "iq_norm_path": str(iq_norm_path.resolve()),
                "raw_depth_path": str(raw_depth_path.resolve()),
                "confidence_path": str(conf_path.resolve()),
                "saturation_mask_path": str(sat_path.resolve()),
                "scale": float(scale),
                "prepare_visualization": str(vis_path.resolve()),
            }
        )
        print(
            f"prepared {stem}: sat={sat.mean():.4f}, scale={scale:.3f}, "
            f"depth=[{np.nanmin(raw_depth):.3f},{np.nanmax(raw_depth):.3f}]"
        )

    result = {
        "raw_dir": str(Path(args.raw_dir).resolve()),
        "output_dir": str(out_dir.resolve()),
        "depthcad_path": str(Path(args.depthcad_path).resolve()),
        "base_model": args.pretrained_model_name_or_path,
        "target_size": args.target_size,
        "scale_mode": args.scale_mode,
        "scale_percentile": args.scale_percentile,
        "saturation_value": args.saturation_value,
        "run_model": bool(args.run_model),
        "prepared": prepared,
        "skipped_preview": skipped[:20],
        "notes": (
            "Only (9,H,W) raw npy files are used. Flat uint8 packet files are skipped. "
            "DepthCAD receives 6-channel IQ plus confidence; saturated pixels are set to low confidence."
        ),
    }
    if args.run_model:
        result["results"] = run_depthcad_model(prepared, args)
    save_json(out_dir / "summary.json", result)
    print(f"Saved summary to {out_dir / 'summary.json'}")
    if not args.run_model:
        print("Prepared inputs only. Re-run with --run_model on a CUDA machine for DepthCAD inference.")


if __name__ == "__main__":
    main()
