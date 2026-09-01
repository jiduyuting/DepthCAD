import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from depth_completion_baselines.common import PBRTCompletionDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train DEPTHOR on the unified full PBRT split.")
    parser.add_argument("--depthor_root", type=Path, default=Path("/data/pre_student/GJ/Depthor"))
    parser.add_argument("--dav2_checkpoint", type=Path, default=Path("output/depth_completion_weights/depthor/depth_anything_v2_vits.pth"))
    parser.add_argument("--init_checkpoint", type=Path, default=Path("output/depth_completion_weights/depthor/depthor_zju_large.pt"))
    parser.add_argument("--cache_root", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq")
    parser.add_argument("--split_json", default="output/completionformer_full_pbrt/split.json")
    parser.add_argument("--output_dir", type=Path, default=Path("output/depth_completion_baselines/depthor_pbrt_train"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--min_depth", type=float, default=1e-3)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--hole_weight", type=float, default=4.0)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--train_limit", type=int)
    parser.add_argument("--val_limit", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def install_dav2_loader(depthor_root, checkpoint):
    sys.path.insert(0, str(depthor_root.resolve()))
    from src.models.depth_anything_v2.dpt import DepthAnythingV2
    from src.utils import set_mde

    def load_depth_anything(encoder="vits"):
        model_configs = {
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        }
        model = DepthAnythingV2(**model_configs[encoder])
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        return model

    set_mde.set_depthanything = load_depth_anything


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_flexible_weights(model, checkpoint):
    if not checkpoint or not Path(checkpoint).exists():
        return "none"
    payload = torch.load(checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    state = {}
    for key, value in payload.items():
        state[key.replace("module.", "", 1) if key.startswith("module.") else key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "path": str(Path(checkpoint).resolve()),
        "missing": missing,
        "unexpected": unexpected,
    }


def prepare_batch(batch, device, input_size):
    image = batch["image"].to(device, non_blocking=True).float()
    sparse = batch["sparse_depth"].to(device, non_blocking=True).float().unsqueeze(1)
    target = batch["target"].to(device, non_blocking=True).float().unsqueeze(1)
    valid = batch["valid_mask"].to(device, non_blocking=True).bool().unsqueeze(1)
    hole = batch["hole_mask"].to(device, non_blocking=True).bool().unsqueeze(1)

    image = F.interpolate(image, input_size, mode="bilinear", align_corners=False)
    sparse = F.interpolate(sparse, input_size, mode="nearest")
    target = F.interpolate(target, input_size, mode="bilinear", align_corners=False)
    valid = F.interpolate(valid.float(), input_size, mode="nearest").bool()
    hole = F.interpolate(hole.float(), input_size, mode="nearest").bool()
    return image, sparse, target, valid, hole


def weighted_charbonnier(prediction, target, valid, hole, min_depth, max_depth, hole_weight):
    mask = valid & torch.isfinite(target) & (target > min_depth) & (target < max_depth)
    if not mask.any():
        return prediction.sum() * 0.0
    diff = prediction - target
    loss = torch.sqrt(diff * diff + 1e-6)
    weights = torch.ones_like(loss)
    weights = torch.where(hole, weights * hole_weight, weights)
    weights = weights * mask.float()
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def add_region(stats, name, prediction, target, mask):
    mask = mask & torch.isfinite(prediction) & torch.isfinite(target)
    count = int(mask.sum().item())
    if count == 0:
        return
    err = prediction[mask] - target[mask]
    stats[name]["count"] += count
    stats[name]["abs_sum"] += float(err.abs().sum().item())
    stats[name]["sq_sum"] += float(err.square().sum().item())


def summarize_stats(stats):
    output = {}
    for name, values in stats.items():
        count = values["count"]
        output[name] = {
            "count": count,
            "mae_m": values["abs_sum"] / count,
            "rmse_m": math.sqrt(values["sq_sum"] / count),
        }
    return output


def jsonable_args(args):
    output = {}
    for key, value in vars(args).items():
        output[key] = str(value) if isinstance(value, Path) else value
    return output


@torch.no_grad()
def evaluate(model, loader, device, input_size, args, epoch):
    model.eval()
    stats = defaultdict(lambda: {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0})
    prediction_dir = args.output_dir / "predictions" / f"epoch_{epoch:03d}"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        image, sparse, _, _, _ = prepare_batch(batch, device, input_size)
        _, pred = model({"image": image, "sparse_depth": sparse})
        pred = F.interpolate(pred, batch["target"].shape[-2:], mode="bilinear", align_corners=False).cpu()

        target = batch["target"].float().unsqueeze(1)
        valid = batch["valid_mask"].bool().unsqueeze(1)
        hole = batch["hole_mask"].bool().unsqueeze(1)
        valid = valid & (target > args.min_depth) & (target < args.max_depth)
        add_region(stats, "global", pred, target, valid)
        add_region(stats, "hole", pred, target, valid & hole)
        add_region(stats, "observed", pred, target, valid & ~hole)

        if args.save_predictions:
            for item_idx, sample_id in enumerate(batch["sample_id"]):
                path = prediction_dir / f"{sample_id}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, pred[item_idx, 0].numpy().astype(np.float32))

    model.train()
    return summarize_stats(stats)


def save_training_state(path, model, optimizer, epoch, metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": jsonable_args(args),
        },
        path,
    )


def save_weights(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    input_size = (args.input_height, args.input_width)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.depthor_root.exists():
        raise FileNotFoundError(f"Missing DEPTHOR repo: {args.depthor_root}")
    if not args.dav2_checkpoint.exists():
        raise FileNotFoundError(f"Missing DAV2 checkpoint: {args.dav2_checkpoint}")

    install_dav2_loader(args.depthor_root, args.dav2_checkpoint)
    from src.models.depthor import Depthor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Depthor(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm="linear").to(device)
    model.set_extra_param(device=device)
    init_info = load_flexible_weights(model, args.init_checkpoint)
    model.train()

    train_dataset = PBRTCompletionDataset(args.cache_root, args.split_json, split="train", limit=args.train_limit)
    val_dataset = PBRTCompletionDataset(args.cache_root, args.split_json, split="test", limit=args.val_limit)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.get_lr_params(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = {
        "args": jsonable_args(args),
        "init": init_info,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "input_size": list(input_size),
        "guidance": "tof_amplitude_3freq",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    history = []
    best_hole_mae = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        running = []
        for step, batch in enumerate(train_loader, start=1):
            image, sparse, target, valid, hole = prepare_batch(batch, device, input_size)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                _, pred = model({"image": image, "sparse_depth": sparse})
                loss = weighted_charbonnier(
                    pred,
                    target,
                    valid,
                    hole,
                    args.min_depth,
                    args.max_depth,
                    args.hole_weight,
                )
                scaled_loss = loss / args.accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % args.accumulation_steps == 0 or step == len(train_loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_step = not (scaler.is_enabled() and scaler.get_scale() < old_scale)
                optimizer.zero_grad(set_to_none=True)
                if did_step:
                    scheduler.step()
                    global_step += 1
            running.append(float(loss.detach().cpu().item()))

        metrics = None
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate(model, val_loader, device, input_size, args, epoch)
            hole_mae = metrics.get("hole", {}).get("mae_m", float("inf"))
            if hole_mae < best_hole_mae:
                best_hole_mae = hole_mae
                save_training_state(args.output_dir / "best_checkpoint.pt", model, optimizer, epoch, metrics, args)
                save_weights(args.output_dir / "best_weights.pt", model)

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(running)) if running else float("nan"),
            "metrics": metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(json.dumps(record, indent=2))

        save_training_state(args.output_dir / "last_checkpoint.pt", model, optimizer, epoch, metrics, args)
        save_weights(args.output_dir / "last_weights.pt", model)

    summary = {
        "method": "DEPTHOR_PBRT_SUPERVISED",
        "samples": {"train": len(train_dataset), "val": len(val_dataset)},
        "best_hole_mae_m": best_hole_mae,
        "best_weights": str((args.output_dir / "best_weights.pt").resolve()),
        "last_weights": str((args.output_dir / "last_weights.pt").resolve()),
        "history": history,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
