import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


CHANNELS = ["A", "B", "C", "D", "E", "F"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-channel PBRT-style dataset for DepthCAD-HoleAware "
            "training from depth completion cache files."
        )
    )
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--list", type=str, default=None, help="Optional txt file of .npz cache paths.")
    parser.add_argument("--ideal_iq_dir", type=str, default="pbrt_dataset/data/ideal_IQ")
    parser.add_argument("--output_root", type=str, default="pbrt_dataset/data")
    parser.add_argument("--suffix", type=str, default="kinect_holeaware")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def read_cache_paths(cache_dir, list_path):
    cache_dir = Path(cache_dir)
    if list_path:
        paths = []
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if not p.is_absolute():
                    p = Path.cwd() / p
                paths.append(p)
        return paths
    return sorted(cache_dir.rglob("*.npz"))


def parse_sample_name(data, path):
    if "sample_name" in data.files:
        value = data["sample_name"]
        sample_name = str(value.item() if value.shape == () else value)
    else:
        parts = path.with_suffix("").parts[-3:]
        sample_name = "/".join(parts)
    parts = sample_name.split("/")
    if len(parts) != 3:
        raise ValueError(f"Expected sample_name scene/idx/name, got {sample_name!r} from {path}")
    return parts[0], parts[1], parts[2]


def load_ideal_iq(ideal_iq_dir, scene, idx, sample):
    ideal_iq_dir = Path(ideal_iq_dir)
    channels = []
    for ch in CHANNELS:
        path = ideal_iq_dir / scene / idx / f"{sample}_{ch}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing ideal IQ channel: {path}")
        channels.append(np.load(path).astype(np.float32))
    return np.stack(channels, axis=0)


def save_iq_stack(root, subdir, scene, idx, sample, iq, overwrite):
    out_dir = Path(root) / subdir / scene / idx
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, ch in enumerate(CHANNELS):
        out_path = out_dir / f"{sample}_{ch}.npy"
        if out_path.exists() and not overwrite:
            continue
        np.save(out_path, iq[i].astype(np.float32))


def save_conf(root, subdir, scene, idx, sample, confidence, overwrite):
    out_dir = Path(root) / subdir / scene / idx
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample}.npy"
    if out_path.exists() and not overwrite:
        return
    np.save(out_path, confidence.astype(np.float32))


def main():
    args = parse_args()
    paths = read_cache_paths(args.cache_dir, args.list)
    if args.max_samples is not None:
        paths = paths[: args.max_samples]

    ideal_subdir = f"ideal_IQ_{args.suffix}"
    noise_subdir = f"noise_IQ_{args.suffix}"
    conf_subdir = f"confidence_{args.suffix}"

    processed = 0
    skipped = []
    for path in tqdm(paths, desc="Preparing DepthCAD-HoleAware dataset"):
        if not path.exists():
            skipped.append({"path": str(path), "reason": "missing cache file"})
            continue
        with np.load(path) as data:
            missing = [key for key in ["noisy_iq", "confidence"] if key not in data.files]
            if missing:
                skipped.append(
                    {
                        "path": str(path),
                        "reason": (
                            f"missing {missing}; regenerate cache with "
                            "--save_depth_completion_cache --depth_cache_save_iq"
                        ),
                    }
                )
                continue
            scene, idx, sample = parse_sample_name(data, path)
            noisy_iq = data["noisy_iq"].astype(np.float32)
            confidence = data["confidence"].astype(np.float32)
            if "ideal_iq" in data.files:
                ideal_iq = data["ideal_iq"].astype(np.float32)
            else:
                ideal_iq = load_ideal_iq(args.ideal_iq_dir, scene, idx, sample)

        save_iq_stack(args.output_root, ideal_subdir, scene, idx, sample, ideal_iq, args.overwrite)
        save_iq_stack(args.output_root, noise_subdir, scene, idx, sample, noisy_iq, args.overwrite)
        save_conf(args.output_root, conf_subdir, scene, idx, sample, confidence, args.overwrite)
        processed += 1

    manifest = {
        "cache_dir": args.cache_dir,
        "list": args.list,
        "ideal_iq_dir": args.ideal_iq_dir,
        "output_root": args.output_root,
        "suffix": args.suffix,
        "ideal_subdir": ideal_subdir,
        "noise_subdir": noise_subdir,
        "conf_subdir": conf_subdir,
        "processed_samples": processed,
        "skipped_count": len(skipped),
        "skipped": skipped[:100],
    }
    manifest_path = Path(args.output_root) / f"depthcad_holeaware_{args.suffix}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
