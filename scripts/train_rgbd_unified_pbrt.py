#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from unified_pbrt_dataset import UnifiedPbrtDataset

RGBD_REPO = Path("/data/pre_student/hcy/RGBD_imaging")
sys.path.insert(0, str(RGBD_REPO))
from srresnet_unet3 import _NetG  # noqa: E402


def gradients(image):
    dx = F.pad(image, (0, 1, 0, 0), mode="replicate")[:, :, :, :-1] - F.pad(
        image, (0, 1, 0, 0), mode="replicate"
    )[:, :, :, 1:]
    dy = F.pad(image, (0, 0, 0, 1), mode="replicate")[:, :, :-1, :] - F.pad(
        image, (0, 0, 0, 1), mode="replicate"
    )[:, :, 1:, :]
    return dx, dy


def compute_loss(prediction_m, target_m, amplitude, valid_mask):
    valid = valid_mask & torch.isfinite(prediction_m) & torch.isfinite(target_m)
    l1 = torch.abs(prediction_m - target_m)[valid].mean()
    pred_dx, pred_dy = gradients(prediction_m)
    amp_dx, amp_dy = gradients(amplitude)
    smooth = torch.abs(pred_dx) * torch.exp(-torch.abs(amp_dx))
    smooth = smooth + torch.abs(pred_dy) * torch.exp(-torch.abs(amp_dy))
    smooth = smooth[valid].mean()
    return 0.5 * l1 + 0.5 * smooth, l1


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "l1": 0.0, "steps": 0}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            iq = batch["iq"][:, :4].to(device)
            depth = batch["depth"].to(device)
            amplitude = batch["amplitude"].to(device)
            target = batch["target"].to(device)
            valid = batch["valid_mask"].to(device)
            network_input = torch.cat((iq, depth / 10.0, amplitude), dim=1)
            prediction = model(network_input) * 10.0
            loss, l1 = compute_loss(prediction, target, amplitude, valid)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["l1"] += float(l1.detach())
            totals["steps"] += 1
    steps = max(totals["steps"], 1)
    return {"loss": totals["loss"] / steps, "l1": totals["l1"] / steps}


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
    model = _NetG().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    history = []
    best = float("inf")
    start_epoch = 1
    if args.resume:
        resume_path = output / "last.pth"
        if not resume_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best = state["best"]
        history = state.get("history", [])
        start_epoch = state["epoch"] + 1
        print(f"Resuming RGBD from epoch {state['epoch']}; target epoch {args.epochs}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["l1"] < best:
            best = val_metrics["l1"]
            torch.save(model.state_dict(), output / "checkpoint_best.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2))
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
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
    parser.add_argument("--output", default="output/rgbd_imaging_full_pbrt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume from output/last.pth.")
    main(parser.parse_args())
