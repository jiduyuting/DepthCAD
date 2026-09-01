import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from depth_completion_baselines.common import (
    MetricAccumulator,
    PBRTCompletionDataset,
    evaluate_prediction,
    save_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DEPTHOR evaluation on a PBRT completion split.")
    parser.add_argument("--depthor_root", type=Path, default=Path("/data/pre_student/GJ/Depthor"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dav2_checkpoint", type=Path, required=True)
    parser.add_argument("--cache_root", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq")
    parser.add_argument("--split_json", default="output/completionformer_full_pbrt/split.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/depth_completion_baselines/depthor"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input_height", type=int, default=480)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--preserve_observed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the original sparse depth at observed pixels after inference.",
    )
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def install_dav2_loader(checkpoint):
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


def load_depthor_weights(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported DEPTHOR checkpoint format: {checkpoint}")

    model_state = model.state_dict()
    loadable = {}
    skipped = {}
    for key, value in state.items():
        clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        if clean_key not in model_state:
            skipped[clean_key] = "unexpected"
            continue
        if tuple(value.shape) != tuple(model_state[clean_key].shape):
            skipped[clean_key] = f"shape {tuple(value.shape)} != {tuple(model_state[clean_key].shape)}"
            continue
        loadable[clean_key] = value

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    if skipped:
        print(f"Skipped DEPTHOR checkpoint keys: {sorted(skipped.items())}")
    if missing:
        print(f"Missing DEPTHOR checkpoint keys: {list(missing)}")
    if unexpected:
        print(f"Unexpected DEPTHOR checkpoint keys after filtering: {list(unexpected)}")
    return model


def main():
    args = parse_args()
    sys.path.insert(0, str(args.depthor_root.resolve()))
    install_dav2_loader(args.dav2_checkpoint)

    from src.models.depthor import Depthor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Depthor(n_bins=args.n_bins, min_val=1e-3, max_val=10.0, norm="linear").to(device)
    model.set_extra_param(device=device)
    model = load_depthor_weights(model, args.checkpoint)
    model.eval()

    dataset = PBRTCompletionDataset(args.cache_root, args.split_json, split=args.split, limit=args.limit)
    accumulator = MetricAccumulator()
    input_size = (args.input_height, args.input_width)

    for sample in dataset:
        image = F.interpolate(
            sample["image"].unsqueeze(0).to(device), input_size, mode="bilinear", align_corners=False
        )
        sparse = F.interpolate(
            sample["sparse_depth"].unsqueeze(0).unsqueeze(0).to(device), input_size, mode="nearest"
        )
        with torch.inference_mode():
            _, prediction = model({"image": image, "sparse_depth": sparse})
            prediction = F.interpolate(
                prediction, sample["target"].shape, mode="bilinear", align_corners=False
            )[0, 0]
        prediction = prediction.float().cpu().numpy()
        if args.preserve_observed:
            sparse_depth = sample["sparse_depth"].numpy()
            observed = (~sample["hole_mask"].numpy().astype(bool)) & np.isfinite(sparse_depth) & (sparse_depth > 0)
            prediction[observed] = sparse_depth[observed]
        evaluate_prediction(accumulator, prediction, sample)
        if args.save_predictions:
            path = args.output_dir / "predictions" / f"{sample['sample_id']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, prediction.astype(np.float32))

    save_summary(
        args.output_dir / "summary.json",
        "DEPTHOR",
        dataset,
        accumulator.summary(),
        checkpoint=str(args.checkpoint.resolve()),
        dav2_checkpoint=str(args.dav2_checkpoint.resolve()),
        n_bins=args.n_bins,
        model_input_size=list(input_size),
        guidance="tof_amplitude_3freq",
        preserve_observed=args.preserve_observed,
        split=args.split,
        cache_root=str(Path(args.cache_root).resolve()),
        split_json=str(Path(args.split_json).resolve()),
    )


if __name__ == "__main__":
    main()
