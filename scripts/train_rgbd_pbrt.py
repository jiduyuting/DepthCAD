"""Train the RGBD-imaging model on the PBRT train split."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RGBD_REPO = Path("/data/pre_student/hcy/RGBD_imaging")
sys.path.insert(0, str(RGBD_REPO))
from srresnet_unet3 import _NetG  # noqa: E402


def sqrt_ldr(correlations):
    confidence = np.abs(correlations[0]) + np.abs(correlations[1])
    confidence_ldr = 16 * np.sqrt(confidence + 36) - 96
    confidence = confidence.copy()
    confidence[confidence == 0] = 1
    return np.stack(
        (confidence_ldr * correlations[0] / confidence,
         confidence_ldr * correlations[1] / confidence), axis=0
    )


def load_input(path):
    raw = np.load(path).astype(np.float32)
    iq_40 = sqrt_ldr(np.stack((raw[0], raw[1]), axis=0))
    iq_30 = sqrt_ldr(np.stack((raw[3], raw[4]), axis=0))
    corr = np.concatenate((iq_30, iq_40), axis=0) / 500.0
    amplitude = raw[2:3] / 500.0
    return np.nan_to_num(corr), np.nan_to_num(amplitude)


class PbrtDataset(Dataset):
    def __init__(self, root, split, max_samples=None):
        self.root = Path(root)
        list_path = self.root / "list" / f"{split}.txt"
        scenes = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        self.samples = [
            (scene, frame)
            for scene in scenes
            for frame in range(1, 251)
        ]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene, frame = self.samples[index]
        relative = Path(scene) / f"{frame}.npy"
        corr, amplitude = load_input(self.root / "noise" / relative)
        depth = np.load(self.root / "noise_depth" / relative).astype(np.float32)[None] / 10.0
        target = np.load(self.root / "gt_depth" / relative).astype(np.float32)[None] / 10.0
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.from_numpy(corr),
            torch.from_numpy(depth),
            torch.from_numpy(amplitude),
            torch.from_numpy(target),
        )


def grad_x(image):
    padded = F.pad(image, (0, 1, 0, 0), mode="replicate")
    return padded[:, :, :, :-1] - padded[:, :, :, 1:]


def grad_y(image):
    padded = F.pad(image, (0, 0, 0, 1), mode="replicate")
    return padded[:, :, :-1, :] - padded[:, :, 1:, :]


def smooth_loss(depth, amplitude, mask):
    weights_x = torch.exp(-torch.abs(grad_x(amplitude)))
    weights_y = torch.exp(-torch.abs(grad_y(amplitude)))
    smoothness = torch.abs(grad_x(depth) * weights_x) + torch.abs(grad_y(depth) * weights_y)
    valid = mask & torch.isfinite(smoothness)
    return smoothness[valid].mean() if valid.any() else smoothness.mean() * 0.0


def loss_fn(prediction, amplitude, target):
    mask = target > 0
    valid = mask & torch.isfinite(prediction)
    l1 = torch.abs(prediction - target)[valid].mean() if valid.any() else prediction.mean() * 0.0
    return 0.5 * l1 + 0.5 * smooth_loss(prediction, amplitude, mask), l1


def run_epoch(model, loader, device, optimizer=None, description="epoch"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_l1 = 0.0
    steps = 0
    progress = tqdm(loader, desc=description)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for corr, depth, amplitude, target in progress:
            corr, depth = corr.to(device), depth.to(device)
            amplitude, target = amplitude.to(device), target.to(device)
            prediction = model(torch.cat((corr, depth, amplitude), dim=1))
            loss, l1 = loss_fn(prediction, amplitude, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_l1 += l1.item()
            steps += 1
            progress.set_postfix(loss=total_loss / steps, l1=total_l1 / steps)
    return total_loss / max(steps, 1), total_l1 / max(steps, 1)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "debug").mkdir(exist_ok=True)
    device = torch.device(args.device)
    train_loader = DataLoader(
        PbrtDataset(root, "train", args.max_train_samples), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        PbrtDataset(root, "test", args.max_val_samples), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = _NetG().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_loss, train_l1 = run_epoch(model, train_loader, device, optimizer, f"train {epoch + 1}/{args.epochs}")
        val_loss, val_l1 = run_epoch(model, val_loader, device, description=f"test {epoch + 1}/{args.epochs}")
        row = {"epoch": epoch + 1, "train_loss": train_loss, "train_l1": train_l1, "test_loss": val_loss, "test_l1": val_l1}
        history.append(row)
        print(json.dumps(row))
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), output / "checkpoint_best.pth")
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output / f"checkpoint_{epoch + 1}.pth")
        (output / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="/data/pre_student/hcy/datasets/pbrt")
    parser.add_argument("--output", default="output/rgbd_imaging_pbrt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    main(parser.parse_args())
