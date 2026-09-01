import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from depth_completion_baselines.common import (
    MetricAccumulator,
    PBRTCompletionDataset,
    evaluate_prediction,
    save_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot LingBot-Depth evaluation on a PBRT completion split.")
    parser.add_argument("--lingbot_root", type=Path, default=Path("/data/pre_student/GJ/lingbot-depth"))
    parser.add_argument("--model", default="robbyant/lingbot-depth-postrain-dc-vitl14")
    parser.add_argument("--cache_root", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq")
    parser.add_argument("--split_json", default="output/completionformer_full_pbrt/split.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/depth_completion_baselines/lingbot_dc_zero_shot"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution_level", type=int, default=9)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.lingbot_root.resolve()))
    from mdm.model.v2 import MDMModel

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MDMModel.from_pretrained(args.model).to(device).eval()
    dataset = PBRTCompletionDataset(args.cache_root, args.split_json, split=args.split, limit=args.limit)
    accumulator = MetricAccumulator()

    for sample in dataset:
        image = sample["image"].unsqueeze(0).to(device)
        sparse = sample["sparse_depth"].unsqueeze(0).to(device)
        height, width = sparse.shape[-2:]
        intrinsics = sample["intrinsics"].clone()
        intrinsics[0] /= width
        intrinsics[1] /= height
        intrinsics = intrinsics.unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model.infer(
                image,
                depth_in=sparse,
                intrinsics=intrinsics,
                resolution_level=args.resolution_level,
                apply_mask=False,
            )
            prediction = output["depth"][0].float().cpu().numpy()
        evaluate_prediction(accumulator, prediction, sample)
        if args.save_predictions:
            path = args.output_dir / "predictions" / f"{sample['sample_id']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, prediction.astype(np.float32))

    save_summary(
        args.output_dir / "summary.json",
        "LingBot-Depth-DC-zero-shot",
        dataset,
        accumulator.summary(),
        model=args.model,
        resolution_level=args.resolution_level,
        guidance="tof_amplitude_3freq",
        split=args.split,
        cache_root=str(Path(args.cache_root).resolve()),
        split_json=str(Path(args.split_json).resolve()),
    )


if __name__ == "__main__":
    main()
