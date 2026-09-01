#!/usr/bin/env python3
"""Precompute frozen Flow outputs used by propagation-refine training."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_depth_restoration import load_checkpoint
from train_depth_completion import move_batch_to_device, read_list
from train_depth_flow_propagation_refine import (
    PropagationRefineCacheDataset,
    build_flow_model_from_checkpoint,
    flow_anchor_cache_path,
    flow_dataset_kwargs,
    predict_flow_anchor_norm,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_list", required=True)
    parser.add_argument("--val_list", required=True)
    parser.add_argument("--test_list", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "cache_protocol": "effective_hole_v2",
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = read_list(args.train_list) + read_list(args.val_list)
    if args.test_list:
        paths += read_list(args.test_list)
    paths = list(dict.fromkeys(paths))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = checkpoint_fingerprint(args.pretrained_checkpoint)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("checkpoint_fingerprint") != fingerprint:
            print("Flow checkpoint/protocol changed; rebuilding all cached anchors.")
            args.overwrite = True

    pending = [
        path for path in paths
        if args.overwrite or not os.path.exists(flow_anchor_cache_path(args.output_dir, path))
    ]
    print(f"Total samples: {len(paths)}")
    print(f"Pending anchors: {len(pending)}")
    if not pending:
        print("All Flow anchors already exist.")
        return

    checkpoint = load_checkpoint(args.pretrained_checkpoint, device)
    flow_args = checkpoint.get("args", {})
    dataset_kwargs = flow_dataset_kwargs(flow_args)
    dataset = PropagationRefineCacheDataset(pending, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_flow_model_from_checkpoint(checkpoint, flow_args, dataset.input_channels, device)
    model.eval()

    saved = 0
    amp_enabled = args.amp and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            try:
                if amp_enabled:
                    with torch.cuda.amp.autocast():
                        anchors = predict_flow_anchor_norm(model, flow_args, batch)
                else:
                    anchors = predict_flow_anchor_norm(model, flow_args, batch)
            except RuntimeError as error:
                if not amp_enabled or "Half" not in str(error):
                    raise
                print("AMP failed in Flow anchor inference; retrying this batch in FP32.")
                amp_enabled = False
                anchors = predict_flow_anchor_norm(model, flow_args, batch)
            anchors = anchors.detach().float().cpu().numpy()
            for index, source_path in enumerate(batch["path"]):
                target = flow_anchor_cache_path(args.output_dir, source_path)
                np.save(target, anchors[index, 0].astype(np.float32))
                saved += 1
            if saved % max(args.batch_size * 20, 1) == 0 or saved == len(pending):
                print(f"Saved {saved}/{len(pending)} anchors")

    metadata = {
        "pretrained_checkpoint": os.path.abspath(args.pretrained_checkpoint),
        "cache_dir": os.path.abspath(args.cache_dir),
        "num_samples": len(paths),
        "num_written": saved,
        "device": str(device),
        "eval_sampling_mode": flow_args.get("eval_sampling_mode", "euler"),
        "checkpoint_fingerprint": fingerprint,
        "hole_definition": "hole_mask & valid_mask",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
