import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CHANNELS = "ABCDEF"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare split-specific PBRT IQ directories for DepthCAD-HoleAware "
            "from the unified depth-completion manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("pbrt_dataset/data"))
    parser.add_argument("--ideal_iq_root", type=Path, default=Path("pbrt_dataset/data/ideal_IQ"))
    parser.add_argument("--suffix", default="unified_pbrt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def resize(array, shape, interpolation=cv2.INTER_LINEAR):
    if array.shape == tuple(shape):
        return array.astype(np.float32, copy=False)
    return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=interpolation)


def load_iq(root, sample, shape):
    scene, view, frame = sample.split("/")
    channels = []
    for channel in CHANNELS:
        path = root / scene / view / f"{frame}_{channel}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing IQ channel: {path}")
        channels.append(resize(np.load(path), shape))
    return np.stack(channels, axis=0)


def load_cache(cache_root, sample):
    scene, view, frame = sample.split("/")
    path = cache_root / scene / view / f"{frame}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache sample: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"hole_mask", "confidence", "gt_depth"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")
        target_shape = data["gt_depth"].shape
        hole = resize(data["hole_mask"], target_shape, cv2.INTER_NEAREST) > 0.5
        confidence = resize(data["confidence"], target_shape)
        confidence = np.clip(confidence, 0.0, 1.0)
        target = data["gt_depth"].astype(np.float32)
    return path, hole, confidence, target


def save_channel_stack(directory, sample, stack, overwrite):
    scene, view, frame = sample.split("/")
    target_dir = directory / scene / view
    target_dir.mkdir(parents=True, exist_ok=True)
    for channel, array in zip(CHANNELS, stack):
        path = target_dir / f"{frame}_{channel}.npy"
        if path.exists() and not overwrite:
            continue
        np.save(path, array.astype(np.float32))


def save_map(directory, sample, array, overwrite):
    scene, view, frame = sample.split("/")
    target_dir = directory / scene / view
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{frame}.npy"
    if not path.exists() or overwrite:
        np.save(path, array.astype(np.float32))


def prepare_sample(sample, cache_root, ideal_iq_root, iq_root, directories, overwrite):
    try:
        cache_path, hole, confidence, _ = load_cache(cache_root, sample)
        ideal_iq = load_iq(ideal_iq_root, sample, hole.shape)
        noisy_iq = load_iq(iq_root, sample, hole.shape)
        noisy_iq[:, hole] = 0.0
        confidence[hole] = 0.0
        save_channel_stack(directories["ideal"], sample, ideal_iq, overwrite)
        save_channel_stack(directories["noise"], sample, noisy_iq, overwrite)
        save_map(directories["confidence"], sample, confidence, overwrite)
        save_map(directories["hole"], sample, hole.astype(np.float32), overwrite)
        return {"sample": sample, "cache": str(cache_path)}, None
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return None, {"sample": sample, "reason": str(exc)}


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    train_cache = Path(payload["train_cache"])
    holdout_cache = Path(payload["holdout_cache"])
    iq_root = Path(payload["iq_root"])

    summary = {
        "protocol": payload.get("protocol"),
        "source_manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "ideal_iq_root": str(args.ideal_iq_root),
        "suffix": args.suffix,
        "hole_policy": "noise_IQ channels are zeroed where hole_mask > 0.5",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        samples = list(payload["samples"][split])
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        cache_root = holdout_cache if split == "test" else train_cache
        prefix = f"{args.suffix}_{split}"
        ideal_dir = args.output_root / f"ideal_IQ_{prefix}"
        noise_dir = args.output_root / f"noise_IQ_{prefix}"
        conf_dir = args.output_root / f"confidence_{prefix}"
        hole_dir = args.output_root / f"hole_mask_{prefix}"
        processed = []
        failures = []

        directories = {
            "ideal": ideal_dir,
            "noise": noise_dir,
            "confidence": conf_dir,
            "hole": hole_dir,
        }
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            results = executor.map(
                lambda sample: prepare_sample(
                    sample,
                    cache_root,
                    args.ideal_iq_root,
                    iq_root,
                    directories,
                    args.overwrite,
                ),
                samples,
            )
            for result, failure in tqdm(results, total=len(samples), desc=f"Preparing DepthCAD {split}"):
                if result is not None:
                    processed.append(result)
                if failure is not None:
                    failures.append(failure)

        split_manifest = {
            "split": split,
            "cache_root": str(cache_root),
            "ideal_dir": str(ideal_dir),
            "noise_dir": str(noise_dir),
            "confidence_dir": str(conf_dir),
            "hole_mask_dir": str(hole_dir),
            "requested": len(samples),
            "processed": len(processed),
            "failures": failures,
            "samples": processed,
        }
        split_manifest_path = args.output_root / f"depthcad_{args.suffix}_{split}_manifest.json"
        split_manifest_path.write_text(json.dumps(split_manifest, indent=2))
        summary["splits"][split] = {
            "requested": len(samples),
            "processed": len(processed),
            "failures": len(failures),
            "manifest": str(split_manifest_path),
        }

    summary_path = args.output_root / f"depthcad_{args.suffix}_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
