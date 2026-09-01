#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unified_pbrt_dataset import UnifiedPbrtDataset

LFRD2_REPO = Path("/data/pre_student/GJ/LFRD2")
sys.path.insert(0, str(LFRD2_REPO))
import Loss  # noqa: E402
from model.cplx import FracDiff  # noqa: E402


def compute_loss(predictions, target, valid, depth_grad):
    count = valid.sum().clamp_min(1)
    depth_loss = torch.abs(predictions[0] - target)[valid].sum() / count
    for prediction in predictions[1:]:
        depth_loss = depth_loss + 0.1 * torch.abs(prediction - target)[valid].sum() / count
    grad_loss = (depth_grad(predictions[0], target) * valid.squeeze(1)).sum() / count
    return 0.1 * depth_loss + grad_loss, depth_loss


def run_epoch(model, loader, device, depth_grad, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_depth = 0.0
    steps = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            depth = batch["depth"].to(device) / 10.0
            amplitude = batch["amplitude"].to(device)
            confidence = batch["confidence"].to(device)
            target = batch["target"].to(device) / 10.0
            valid = batch["valid_mask"].to(device)
            output = model(depth, amplitude, confidence)
            loss, depth_loss = compute_loss(output["y_pred"], target, valid, depth_grad)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            total_depth += float(depth_loss.detach())
            steps += 1
    steps = max(steps, 1)
    return {"loss": total_loss / steps, "depth_loss": total_depth / steps}


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = UnifiedPbrtDataset(args.manifest, "train", args.max_train_samples)
    val = UnifiedPbrtDataset(args.manifest, "val", args.max_val_samples)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val, args.batch_size, shuffle=False, num_workers=args.workers)
    model = FracDiff(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    depth_grad = Loss.DepthGrad().to(device)
    best = float("inf")
    history = []
    start_epoch = 1
    if args.resume:
        resume_path = output / "last.pth"
        if not resume_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best = state["best"]
        history = state.get("history", [])
        start_epoch = state["epoch"] + 1
        print(f"Resuming LFRD2 from epoch {state['epoch']}; target epoch {args.epochs}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, depth_grad, optimizer)
        val_metrics = run_epoch(model, val_loader, device, depth_grad)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["depth_loss"] < best:
            best = val_metrics["depth_loss"]
            torch.save(model.state_dict(), output / "checkpoint_best_net.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2))
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best": best,
                "history": history,
                "args": vars(args),
            },
            output / "last.pth",
        )
    (output / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="output/full_pbrt_manifest_seed123.json")
    parser.add_argument("--output", default="output/lfrd2_full_pbrt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume from output/last.pth.")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--prop_kernel", type=int, default=3)
    parser.add_argument("--conf_prop", action="store_true", default=True)
    parser.add_argument("--prop_time", type=int, default=6)
    parser.add_argument("--affinity", default="TGASS", choices=("AS", "ASS", "TC", "TGASS"))
    parser.add_argument("--affinity_gamma", type=float, default=0.5)
    main(parser.parse_args())
