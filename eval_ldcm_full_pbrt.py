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
    parser = argparse.ArgumentParser(description="Zero-shot LDCM evaluation on a PBRT completion split.")
    parser.add_argument("--ldcm_root", type=Path, default=Path("/data/pre_student/GJ/LDCM"))
    parser.add_argument("--model", default="pkqbajng/LDCM")
    parser.add_argument("--moge_model", default="Ruicheng/moge-2-vits-normal")
    parser.add_argument("--cache_root", default="depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq")
    parser.add_argument("--split_json", default="output/completionformer_full_pbrt/split.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/depth_completion_baselines/ldcm_zero_shot"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.ldcm_root.resolve()))
    from ldcm import LDCMModel

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = LDCMModel.from_pretrained(args.model, moge_path=args.moge_model).to(device).eval()
    dataset = PBRTCompletionDataset(args.cache_root, args.split_json, split=args.split, limit=args.limit)
    accumulator = MetricAccumulator()

    for sample in dataset:
        image = sample["image"].unsqueeze(0).to(device)
        sparse = sample["sparse_depth"].unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            prediction = model.infer(image, sparse)["depth_pred"][0, 0].float().cpu().numpy()
        evaluate_prediction(accumulator, prediction, sample)
        if args.save_predictions:
            path = args.output_dir / "predictions" / f"{sample['sample_id']}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, prediction.astype(np.float32))

    save_summary(
        args.output_dir / "summary.json",
        "LDCM-zero-shot",
        dataset,
        accumulator.summary(),
        model=args.model,
        moge_model=args.moge_model,
        guidance="tof_amplitude_3freq",
        split=args.split,
        cache_root=str(Path(args.cache_root).resolve()),
        split_json=str(Path(args.split_json).resolve()),
    )


if __name__ == "__main__":
    main()
