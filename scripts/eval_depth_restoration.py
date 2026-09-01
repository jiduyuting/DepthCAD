import argparse
import json
import math
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_depth_completion import move_batch_to_device, read_list
from train_depth_restoration import (
    DepthRestorationCacheDataset,
    ResidualUNet,
    model_output_channels,
    predict_depth_norm,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single depth restoration model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--sample_list", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--vis_max_samples", type=int, default=20)
    parser.add_argument("--vis_error_percentile", type=float, default=99.0)
    parser.add_argument("--vis_rank", type=str, default="first",
                        choices=[
                            "first",
                            "best_hole",
                            "worst_hole",
                            "best_worst_hole",
                            "best_global",
                            "worst_global",
                            "best_worst_global",
                        ],
                        help="Which samples to visualize. Ranking uses model MAE minus baseline MAE.")
    parser.add_argument("--vis_rank_baseline", type=str, default="anchor",
                        choices=["anchor", "base", "noisy"],
                        help="Baseline used for ranked visualization deltas.")
    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def collect_eval_paths(args, ckpt_args):
    if args.sample_list:
        return read_list(args.sample_list), "sample_list"

    if args.cache_dir is not None:
        paths = sorted(glob(os.path.join(args.cache_dir, "**", "*.npz"), recursive=True))
        return paths, "cache_dir"

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    split_path = os.path.join(ckpt_dir, "split.json")
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        if args.split == "all":
            return split.get("train", []) + split.get("val", []), "checkpoint_split"
        return split.get(args.split, []), "checkpoint_split"

    cache_dir = ckpt_args.get("cache_dir")
    if cache_dir is None:
        raise ValueError("Provide --cache_dir or evaluate a checkpoint with cache_dir in args.")
    return sorted(glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True)), "checkpoint_cache_dir"


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def error_stats(pred, target, mask):
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    count = int(valid.sum())
    if count == 0:
        return math.nan, math.nan, 0
    error = pred[valid] - target[valid]
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))
    return mae, rmse, count


def sample_metrics(depths, target, valid_mask, hole_mask):
    valid_region = valid_mask & (~hole_mask)
    metrics = {}
    for prefix, depth in depths.items():
        for region_name, region_mask in [
            ("global", valid_mask),
            ("hole", valid_mask & hole_mask),
            ("valid", valid_region),
        ]:
            mae, rmse, count = error_stats(depth, target, region_mask)
            metrics[f"{prefix}_{region_name}_mae"] = safe_float(mae)
            metrics[f"{prefix}_{region_name}_rmse"] = safe_float(rmse)
            metrics[f"{prefix}_{region_name}_count"] = count
    return metrics


def aggregate_metrics(per_sample):
    abs_totals = {}
    sq_totals = {}
    counts = {}
    prefixes = ["model", "anchor", "noisy", "base"]
    for row in per_sample:
        for prefix in prefixes:
            for region in ["global", "hole", "valid"]:
                mae = row.get(f"{prefix}_{region}_mae")
                rmse = row.get(f"{prefix}_{region}_rmse")
                count = row.get(f"{prefix}_{region}_count", 0)
                if mae is None or count == 0:
                    continue
                key = f"{prefix}_{region}"
                abs_totals[key] = abs_totals.get(key, 0.0) + mae * count
                if rmse is not None:
                    sq_totals[key] = sq_totals.get(key, 0.0) + (rmse * rmse) * count
                counts[key] = counts.get(key, 0) + count

    out = {}
    for key in sorted(abs_totals):
        out[f"{key}_mae"] = abs_totals[key] / max(counts[key], 1)
        if key in sq_totals:
            out[f"{key}_rmse"] = math.sqrt(sq_totals[key] / max(counts[key], 1))
        out[f"{key}_count"] = counts[key]
    return out


def tensor_to_numpy(batch_value, batch_index):
    return batch_value[batch_index, 0].detach().cpu().numpy()


def save_visualization(out_path, sample_name, depths, target, hole_mask, args):
    import matplotlib.pyplot as plt

    valid = (target > 0.1) & (target < 9.9) & np.isfinite(target)
    if valid.sum() > 0:
        vmin = float(target[valid].min())
        vmax = float(target[valid].max())
    else:
        vmin = float(np.nanmin(target))
        vmax = float(np.nanmax(target))
    depth_kwargs = {"cmap": "turbo"}
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        depth_kwargs.update({"vmin": vmin, "vmax": vmax})

    err_maps = [np.abs(depth - target) for depth in depths.values()]
    finite_err = np.concatenate([e[np.isfinite(e)].reshape(-1) for e in err_maps if np.isfinite(e).any()])
    err_vmax = float(np.percentile(finite_err, args.vis_error_percentile)) if finite_err.size else 1.0
    err_vmax = max(err_vmax, 1e-6)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes[0, 0].imshow(target, **depth_kwargs)
    axes[0, 0].set_title("GT")
    axes[0, 1].imshow(depths["noisy"], **depth_kwargs)
    axes[0, 1].set_title("Noisy")
    axes[0, 2].imshow(depths["anchor"], **depth_kwargs)
    axes[0, 2].set_title("Anchor")
    axes[0, 3].imshow(depths["base"], **depth_kwargs)
    axes[0, 3].set_title("Teacher/Base")

    axes[1, 0].imshow(depths["model"], **depth_kwargs)
    axes[1, 0].set_title("Model")
    axes[1, 1].imshow(hole_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Hole Mask")
    axes[1, 2].imshow(depths["model"] - depths["anchor"], cmap="coolwarm")
    axes[1, 2].set_title("Model - Anchor")
    axes[1, 3].imshow(np.abs(depths["model"] - depths["anchor"]), cmap="hot")
    axes[1, 3].set_title("|Model - Anchor|")

    keys = ["noisy", "anchor", "base", "model"]
    for i, key in enumerate(keys):
        axes[2, i].imshow(np.abs(depths[key] - target), cmap="hot", vmin=0, vmax=err_vmax)
        axes[2, i].set_title(f"|{key} - GT|")

    for ax in axes.reshape(-1):
        ax.axis("off")
    fig.suptitle(sample_name)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def metric_delta(row, region, baseline):
    model_mae = row.get(f"model_{region}_mae")
    base_mae = row.get(f"{baseline}_{region}_mae")
    if model_mae is None or base_mae is None:
        return None
    delta = float(model_mae) - float(base_mae)
    if not np.isfinite(delta):
        return None
    return delta


def sorted_rows_by_delta(per_sample, region, baseline, reverse=False):
    scored = []
    for row in per_sample:
        delta = metric_delta(row, region, baseline)
        if delta is not None:
            scored.append((delta, row))
    scored.sort(key=lambda item: item[0], reverse=reverse)
    return [row for _, row in scored]


def select_ranked_visualization_rows(per_sample, args):
    max_samples = max(0, int(args.vis_max_samples))
    if max_samples == 0 or args.vis_rank == "first":
        return []

    selected = []
    seen_paths = set()
    baseline = args.vis_rank_baseline

    def add_rows(rows, label, limit):
        for row in rows:
            if len(selected) >= max_samples or limit <= 0:
                break
            path = row["path"]
            if path in seen_paths:
                continue
            item = dict(row)
            item["_vis_label"] = label
            selected.append(item)
            seen_paths.add(path)
            limit -= 1

    if args.vis_rank in ["best_hole", "worst_hole", "best_worst_hole"]:
        region = "hole"
    else:
        region = "global"

    if args.vis_rank.startswith("best_worst"):
        best_count = (max_samples + 1) // 2
        worst_count = max_samples - best_count
        add_rows(
            sorted_rows_by_delta(per_sample, region, baseline, reverse=False),
            f"best_{baseline}_{region}",
            best_count,
        )
        add_rows(
            sorted_rows_by_delta(per_sample, region, baseline, reverse=True),
            f"worst_{baseline}_{region}",
            worst_count,
        )
    elif args.vis_rank.startswith("best"):
        add_rows(
            sorted_rows_by_delta(per_sample, region, baseline, reverse=False),
            f"best_{baseline}_{region}",
            max_samples,
        )
    elif args.vis_rank.startswith("worst"):
        add_rows(
            sorted_rows_by_delta(per_sample, region, baseline, reverse=True),
            f"worst_{baseline}_{region}",
            max_samples,
        )

    return selected


@torch.no_grad()
def save_ranked_visualizations(
    rows,
    dataset_kwargs,
    model,
    device,
    max_residual_norm,
    residual_scale,
    prediction_mode,
    out_dir,
    args,
):
    if not rows:
        return 0

    dataset = DepthRestorationCacheDataset([row["path"] for row in rows], **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    rows_by_path = {row["path"]: (index, row) for index, row in enumerate(rows)}
    saved = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_norm = predict_depth_norm(
            model,
            batch,
            max_residual_norm,
            residual_scale,
            prediction_mode,
        )
        scale = batch["scale"].view(-1, 1, 1, 1)
        center = batch["center"].view(-1, 1, 1, 1)
        pred = pred_norm * scale + center

        batch_size = pred.shape[0]
        for i in range(batch_size):
            path = batch["path"][i]
            rank_index, row = rows_by_path[path]
            sample_name = batch["sample_name"][i]
            label = row.get("_vis_label", args.vis_rank)
            region = "hole" if "hole" in label else "global"
            baseline = args.vis_rank_baseline
            delta = metric_delta(row, region, baseline)
            title = (
                f"{label} {sample_name} | "
                f"{baseline}_{region}={row[f'{baseline}_{region}_mae']:.4f} "
                f"model_{region}={row[f'model_{region}_mae']:.4f} "
                f"delta={delta:.4f}"
            )

            depths = {
                "model": tensor_to_numpy(pred, i),
                "anchor": tensor_to_numpy(batch["depth_anchor"], i),
                "noisy": tensor_to_numpy(batch["depth_noisy"], i),
                "base": tensor_to_numpy(batch["depth_base"], i),
            }
            gt = tensor_to_numpy(batch["gt_depth"], i)
            hole = tensor_to_numpy(batch["hole_mask"].float(), i) > 0.5

            safe_label = label.replace("/", "_")
            safe_name = sample_name.replace("/", "_")
            out_png = os.path.join(
                out_dir,
                "visualizations",
                f"vis_{rank_index:03d}_{safe_label}_{safe_name}.png",
            )
            save_visualization(out_png, title, depths, gt, hole, args)
            saved += 1

    return saved


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    paths, path_source = collect_eval_paths(args, ckpt_args)
    if not paths:
        raise FileNotFoundError("No cache samples found for evaluation.")

    out_dir = args.output_dir
    if out_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        out_dir = os.path.join(ckpt_dir, f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    dataset_kwargs = {
        "input_mode": ckpt_args.get("input_mode", "noisy"),
        "include_hole_distance": ckpt_args.get("include_hole_distance", False),
        "anchor_mode": ckpt_args.get("anchor_mode", "noisy_ns"),
        "anchor_inpaint_radius": ckpt_args.get("anchor_inpaint_radius", 15),
        "norm_percentiles": ckpt_args.get("norm_percentiles", [5.0, 95.0]),
        "min_depth_scale": ckpt_args.get("min_depth_scale", 0.25),
        "clip_norm_depth": ckpt_args.get("clip_norm_depth", 8.0),
        "feature_percentile": ckpt_args.get("feature_percentile", 99.0),
        "feature_clip": ckpt_args.get("feature_clip", 3.0),
        "iq_clip": ckpt_args.get("iq_clip", 3.0),
    }
    dataset = DepthRestorationCacheDataset(paths, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ResidualUNet(
        in_channels=dataset.input_channels,
        base_channels=int(ckpt_args.get("base_channels", 32)),
        out_channels=model_output_channels(ckpt_args.get("prediction_mode", "residual")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    max_residual_norm = float(ckpt_args.get("max_residual_norm", 4.0))
    residual_scale = float(ckpt_args.get("residual_scale", 1.0))
    prediction_mode = ckpt_args.get("prediction_mode", "residual")

    per_sample = []
    vis_saved = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        pred_norm = predict_depth_norm(
            model,
            batch,
            max_residual_norm,
            residual_scale,
            prediction_mode,
        )
        scale = batch["scale"].view(-1, 1, 1, 1)
        center = batch["center"].view(-1, 1, 1, 1)
        pred = pred_norm * scale + center

        batch_size = pred.shape[0]
        for i in range(batch_size):
            sample_name = batch["sample_name"][i]
            depths = {
                "model": tensor_to_numpy(pred, i),
                "anchor": tensor_to_numpy(batch["depth_anchor"], i),
                "noisy": tensor_to_numpy(batch["depth_noisy"], i),
                "base": tensor_to_numpy(batch["depth_base"], i),
            }
            gt = tensor_to_numpy(batch["gt_depth"], i)
            valid = tensor_to_numpy(batch["valid_mask"].float(), i) > 0.5
            hole = tensor_to_numpy(batch["hole_mask"].float(), i) > 0.5
            row = {
                "sample_name": sample_name,
                "path": batch["path"][i],
            }
            row.update(sample_metrics(depths, gt, valid, hole))
            per_sample.append(row)

            if args.visualize and args.vis_rank == "first" and vis_saved < args.vis_max_samples:
                safe_name = sample_name.replace("/", "_")
                out_png = os.path.join(out_dir, "visualizations", f"vis_{safe_name}.png")
                save_visualization(out_png, sample_name, depths, gt, hole, args)
                vis_saved += 1

    aggregate = aggregate_metrics(per_sample)
    if args.visualize and args.vis_rank != "first":
        ranked_rows = select_ranked_visualization_rows(per_sample, args)
        vis_saved = save_ranked_visualizations(
            ranked_rows,
            dataset_kwargs,
            model,
            device,
            max_residual_norm,
            residual_scale,
            prediction_mode,
            out_dir,
            args,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "cache_dir": args.cache_dir or ckpt_args.get("cache_dir"),
        "path_source": path_source,
        "split": args.split,
        "num_samples": len(per_sample),
        "visualized_samples": vis_saved,
        "vis_rank": args.vis_rank if args.visualize else None,
        "vis_rank_baseline": args.vis_rank_baseline if args.visualize and args.vis_rank != "first" else None,
        "input_mode": dataset_kwargs["input_mode"],
        "anchor_mode": dataset_kwargs["anchor_mode"],
        "include_hole_distance": dataset_kwargs["include_hole_distance"],
        "base_channels": int(ckpt_args.get("base_channels", 32)),
        "prediction_mode": prediction_mode,
        "max_residual_norm": max_residual_norm,
        "residual_scale": residual_scale,
        "aggregate": aggregate,
    }

    with open(os.path.join(out_dir, "per_sample_results.json"), "w") as f:
        json.dump(per_sample, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved eval results to {out_dir}")


if __name__ == "__main__":
    main()
