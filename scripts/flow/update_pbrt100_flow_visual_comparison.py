#!/usr/bin/env python3
import _bootstrap
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_depth_restoration import load_checkpoint
from train_depth_completion import move_batch_to_device, read_list
from train_depth_flow_propagation_refine import (
    PropagationRefineCacheDataset,
    build_refine_model,
    flow_dataset_kwargs,
    prepare_propagation_batch,
    predict_refined_norm,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Update PBRT100 visual comparisons with the latest Flow refinement model.")
    parser.add_argument("--sample_list", required=True)
    parser.add_argument("--flow_checkpoint", required=True)
    parser.add_argument("--flow_anchor_checkpoint", required=True)
    parser.add_argument("--anchor_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_samples", type=int, default=12)
    return parser.parse_args()


def refine_args(checkpoint_args):
    return SimpleNamespace(
        base_channels=int(checkpoint_args.get("base_channels", 32)),
        res_blocks=int(checkpoint_args.get("res_blocks", 1)),
        propagation_steps=int(checkpoint_args.get("propagation_steps", 6)),
        propagation_hidden_scale=float(checkpoint_args.get("propagation_hidden_scale", 1.0)),
        refine_dilate_radius=int(checkpoint_args.get("refine_dilate_radius", 3)),
        residual_scale=float(checkpoint_args.get("residual_scale", 1.5)),
    )


def prediction_path(root, method, sample):
    return Path(root) / "predictions" / method / f"{sample}.npy"


def main():
    args = parse_args()
    device = torch.device(args.device)
    flow_checkpoint = load_checkpoint(args.flow_anchor_checkpoint, device)
    flow_args = flow_checkpoint.get("args", {})
    dataset_kwargs = flow_dataset_kwargs(flow_args)
    dataset_kwargs["anchor_cache_dir"] = args.anchor_cache_dir
    paths = read_list(args.sample_list)

    dataset = PropagationRefineCacheDataset(paths, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = load_checkpoint(args.flow_checkpoint, device)
    model = build_refine_model(dataset.input_channels, refine_args(checkpoint.get("args", {})), device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    output_root = Path(args.output_dir)
    prediction_root = output_root / "predictions" / "Ours-Flow"
    prediction_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batch = prepare_propagation_batch(batch, None, flow_args)
            pred_norm, _, _ = predict_refined_norm(model, batch)
            scale = batch["scale"].view(-1, 1, 1, 1)
            center = batch["center"].view(-1, 1, 1, 1)
            pred = (pred_norm * scale + center).cpu().numpy()[:, 0]
            for index, sample in enumerate(batch["sample_name"]):
                target = prediction_root / f"{sample}.npy"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, pred[index].astype(np.float32))

    cache_root = Path(args.cache_root)
    records = []
    for path in paths:
        sample = str(Path(path).resolve().relative_to(cache_root.resolve()).with_suffix(""))
        with np.load(path) as data:
            noisy = data["depth_noisy"].astype(np.float32)
            gt = data["gt_depth"].astype(np.float32)
            hole = data["hole_mask"] > 0.5
            valid_gt = data["valid_mask"] > 0.5
        records.append((float(hole.mean()), sample, noisy, gt, hole, valid_gt))
    records.sort(reverse=True)

    panels = [
        ("Input", "Input"), ("GT (raw)", "GT"),
        ("RGBD-Imaging", "RGBD-Imaging"), ("CompletionFormer", "CompletionFormer"),
        ("Ours-Flow", "Ours-Flow"), ("LFRD2", "LFRD2"), ("DMD3C", "DMD3C"),
        ("OMNI-DC", "OMNI-DC"), ("LingBot-Depth", "LingBot-Depth"),
        ("LDCM", "LDCM"), ("DEPTHOR", "DEPTHOR"), ("Hole mask", "Hole"),
    ]
    for rank, (_, sample, noisy, gt, hole, valid_gt) in enumerate(records[:args.max_samples]):
        arrays = {"Input": noisy, "GT": gt, "Hole": hole.astype(np.float32)}
        for _, method in panels[2:]:
            arrays[method] = np.load(prediction_path(output_root, method, sample))
        valid_values = gt[valid_gt & np.isfinite(gt)]
        vmin, vmax = np.nanpercentile(valid_values, [2, 98])
        cmap = plt.get_cmap("turbo")
        fig, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True)
        axes = axes.ravel()
        color_image = None
        for axis, (label, key) in zip(axes, panels):
            image = arrays[key]
            if image.shape != gt.shape:
                image = cv2.resize(image, (gt.shape[1], gt.shape[0]))
            if key == "Hole":
                axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                color_image = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(label)
            axis.axis("off")
        for axis in axes[len(panels):]:
            axis.axis("off")
        fig.colorbar(color_image, ax=axes[:len(panels) - 1], shrink=0.72, pad=0.02, label="Depth (m)")
        fig.suptitle(
            f"{sample} | hole={hole.mean():.3f} | gt_invalid={(~valid_gt).mean():.3f} "
            f"| color scale={vmin:.2f}-{vmax:.2f} m"
        )
        figure_path = output_root / "figures" / f"{rank:02d}_{sample.replace('/', '_')}.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

    metadata = {
        "flow_checkpoint": str(Path(args.flow_checkpoint).resolve()),
        "flow_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "flow_anchor_checkpoint": str(Path(args.flow_anchor_checkpoint).resolve()),
        "sample_list": str(Path(args.sample_list).resolve()),
        "num_predictions": len(paths),
        "num_figures": min(args.max_samples, len(records)),
        "gt_visualization": {
            "valid_definition": "finite(gt_depth) & (gt_depth > 0.1) & (gt_depth < 9.9)",
            "display": "raw gt_depth array without invalid-pixel masking",
            "hole_contour_panels": "none; a dedicated Hole mask panel is used",
            "note": "GT values are shown as stored; invalid values remain visible but are excluded from metrics.",
        },
    }
    (output_root / "flow_update.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
