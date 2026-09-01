import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CHANNELS = "ABCDEF"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a full PBRT depth-completion cache.")
    parser.add_argument("--ideal_iq_root", type=Path, default=Path("pbrt_dataset/data_256/ideal_IQ"))
    parser.add_argument("--noise_iq_root", type=Path, default=Path("pbrt_dataset/data_256/noise_IQ"))
    parser.add_argument("--gt_depth_root", type=Path, default=Path("/data/pre_student/hcy/pbrt/gt_depth"))
    parser.add_argument("--noise_depth_root", type=Path, default=Path("/data/pre_student/hcy/pbrt/noise_depth"))
    parser.add_argument(
        "--test_manifest",
        type=Path,
        default=Path("output/unified_pbrt_manifest_seed123.json"),
        help="Manifest with samples.test, or a split.json with a top-level test list.",
    )
    parser.add_argument(
        "--holdout_cache",
        type=Path,
        default=Path("depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq"),
        help="Used when --test_manifest is missing; its sample names become the fixed test set.",
    )
    parser.add_argument("--cache_root", type=Path, default=Path("depth_completion_cache/depth_cache_full_pbrt_plane_r12"))
    parser.add_argument(
        "--reuse_cache_root",
        type=Path,
        default=Path("depth_completion_cache/depth_cache_0515_n1000_plane_r12"),
    )
    parser.add_argument("--output_manifest", type=Path, default=Path("output/full_pbrt_manifest_seed123.json"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hole_ratio", type=float, default=0.15)
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--amp_percentile", type=float, default=5.0)
    parser.add_argument("--low_amp_ratio", type=float, default=0.4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_test_spec(args):
    if args.test_manifest.is_file():
        payload = json.loads(args.test_manifest.read_text())
        if isinstance(payload.get("samples"), dict) and payload["samples"].get("test") is not None:
            test_samples = payload["samples"]["test"]
        else:
            test_samples = payload.get("test")
        if test_samples is None:
            raise KeyError(
                f"{args.test_manifest} must contain samples.test or a top-level test list"
            )
        holdout_cache = payload.get("holdout_cache", str(args.holdout_cache))
        source = str(args.test_manifest)
    else:
        if not args.holdout_cache.is_dir():
            raise FileNotFoundError(
                f"Neither test manifest nor holdout cache found: {args.test_manifest} / {args.holdout_cache}"
            )
        test_samples = [
            path.relative_to(args.holdout_cache).with_suffix("").as_posix()
            for path in args.holdout_cache.rglob("*.npz")
        ]
        holdout_cache = str(args.holdout_cache)
        source = f"cache:{args.holdout_cache}"

    test_samples = sorted({str(sample) for sample in test_samples})
    if not test_samples:
        raise ValueError("No fixed test samples were found.")
    return test_samples, holdout_cache, source


def enumerate_samples(root):
    samples = []
    for path in root.rglob("*_A.npy"):
        scene, view, filename = path.parts[-3:]
        samples.append(f"{scene}/{view}/{filename[:-6]}")
    return sorted(samples)


def stable_rank(sample, seed):
    return hashlib.sha256(f"{seed}:{sample}".encode()).hexdigest()


def stratified_split(samples, val_ratio, seed):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.split("/", 1)[0]].append(sample)
    target_val = int(round(len(samples) * val_ratio))
    allocations = {scene: int(len(group) * val_ratio) for scene, group in grouped.items()}
    remainder_order = sorted(
        grouped,
        key=lambda scene: (len(grouped[scene]) * val_ratio - allocations[scene], scene),
        reverse=True,
    )
    for scene in remainder_order[: target_val - sum(allocations.values())]:
        allocations[scene] += 1
    train, val = [], []
    for scene in sorted(grouped):
        ordered = sorted(grouped[scene], key=lambda sample: stable_rank(sample, seed))
        val_count = allocations[scene]
        val.extend(ordered[:val_count])
        train.extend(ordered[val_count:])
    return sorted(train), sorted(val)


def load_iq(root, sample, resolution):
    scene, view, frame = sample.split("/")
    channels = []
    for channel in CHANNELS:
        path = root / scene / view / f"{frame}_{channel}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        array = np.load(path).astype(np.float32)
        if array.shape != (resolution, resolution):
            array = cv2.resize(array, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        channels.append(array)
    return np.stack(channels, axis=0)


def amplitude_features(iq):
    amplitude = np.sqrt(iq[0::2] ** 2 + iq[1::2] ** 2).astype(np.float32)
    return amplitude, amplitude.mean(axis=0).astype(np.float32)


def load_depth(root, sample, resolution):
    scene, view, frame = sample.split("/")
    path = root / scene / view / f"{frame}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    depth = np.load(path).astype(np.float32)
    if depth.shape != (resolution, resolution):
        depth = cv2.resize(depth, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    return depth


def select_mask_to_target(mask, confidence, valid, target_ratio):
    target_count = min(int(round(target_ratio * mask.size)), int(valid.sum()))
    flat_mask = mask.reshape(-1).astype(bool)
    flat_valid = valid.reshape(-1).astype(bool)
    flat_confidence = confidence.reshape(-1)
    flat_mask &= flat_valid
    current = np.flatnonzero(flat_mask & flat_valid)
    if len(current) > target_count:
        keep = current[np.argsort(flat_confidence[current], kind="stable")[:target_count]]
        flat_mask[:] = False
        flat_mask[keep] = True
    elif len(current) < target_count:
        available = np.flatnonzero(flat_valid & ~flat_mask)
        add = available[np.argsort(flat_confidence[available], kind="stable")[: target_count - len(current)]]
        flat_mask[add] = True
    return flat_mask.reshape(mask.shape)


def generate_holes(depth, amplitude, args):
    valid = np.isfinite(depth) & (depth > 0.1) & (depth < 9.9)
    values = amplitude[valid & np.isfinite(amplitude)]
    if values.size == 0:
        return np.zeros_like(depth, dtype=bool), np.zeros_like(depth, dtype=np.float32)
    threshold = float(np.percentile(values, args.amp_percentile))
    high = float(np.percentile(values, 95.0))
    confidence_amp = np.clip((amplitude - threshold) / (high - threshold + 1e-8), 0.0, 1.0)
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    confidence_edge = 1.0 - np.clip(gradient / (gradient.max() + 1e-8) * 3.0, 0.0, 1.0)
    confidence = (confidence_amp * confidence_edge * valid).astype(np.float32)
    hole = np.zeros_like(depth, dtype=np.uint8)
    block = args.block_size
    for row in range(0, depth.shape[0] - block + 1, block):
        for col in range(0, depth.shape[1] - block + 1, block):
            block_valid = valid[row : row + block, col : col + block]
            block_amplitude = amplitude[row : row + block, col : col + block][block_valid]
            if block_amplitude.size and np.median(block_amplitude) < threshold:
                if float((block_amplitude < threshold).mean()) > args.low_amp_ratio:
                    hole[row : row + block, col : col + block] = 1
    kernel = np.ones((3, 3), dtype=np.uint8)
    hole = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, kernel)
    hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, kernel)
    hole = select_mask_to_target(hole, confidence, valid, args.hole_ratio)
    confidence[hole] = 0.0
    return hole, confidence


def save_sample(cache_root, sample, noisy_iq, args, force=False):
    scene, view, frame = sample.split("/")
    output = cache_root / scene / view / f"{frame}.npz"
    if output.exists() and not (args.overwrite or force):
        return "existing"
    gt_depth = load_depth(args.gt_depth_root, sample, args.resolution)
    clean_noisy_depth = load_depth(args.noise_depth_root, sample, args.resolution)
    _, amplitude_mean = amplitude_features(noisy_iq)
    hole, confidence = generate_holes(gt_depth, amplitude_mean, args)
    noisy_with_holes = noisy_iq.copy()
    noisy_with_holes[:, hole] = 0.0
    depth_noisy = clean_noisy_depth.copy()
    depth_noisy[hole] = 0.0
    noisy_amplitude, noisy_amplitude_mean = amplitude_features(noisy_with_holes)
    valid = (np.isfinite(gt_depth) & (gt_depth > 0.1) & (gt_depth < 9.9)).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            sample_name=np.array(sample),
            depth_noisy=depth_noisy,
            gt_depth=gt_depth,
            hole_mask=hole.astype(np.uint8),
            confidence=confidence,
            valid_mask=valid,
            noisy_amplitude=noisy_amplitude,
            noisy_amplitude_mean=noisy_amplitude_mean,
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    return "written"


def reuse_existing_cache(samples, source_root, target_root):
    linked = 0
    for sample in samples:
        scene, view, frame = sample.split("/")
        source = source_root / scene / view / f"{frame}.npz"
        target = target_root / scene / view / f"{frame}.npz"
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source.resolve(), target)
        linked += 1
    return linked


def prepare_one(sample, args):
    try:
        scene, view, frame = sample.split("/")
        output = args.cache_root / scene / view / f"{frame}.npz"
        repair = False
        if output.exists() and not args.overwrite:
            try:
                with np.load(output) as data:
                    for key in ("depth_noisy", "gt_depth", "hole_mask", "confidence", "valid_mask", "noisy_amplitude_mean"):
                        _ = data[key].shape
                return "existing", None
            except Exception:
                repair = True
        noisy_iq = load_iq(args.noise_iq_root, sample, args.resolution)
        status = save_sample(args.cache_root, sample, noisy_iq, args, force=repair)
        return ("repaired" if repair else status), None
    except Exception as exc:
        return "failure", {"sample": sample, "reason": repr(exc)}


def main():
    args = parse_args()
    all_samples = enumerate_samples(args.ideal_iq_root)
    test_samples, holdout_cache, test_source = load_test_spec(args)
    missing_test = sorted(set(test_samples).difference(all_samples))
    if missing_test:
        raise ValueError(f"Fixed test samples missing from full PBRT IQ: {missing_test[:10]}")
    development = sorted(set(all_samples).difference(test_samples))
    train_samples, val_samples = stratified_split(development, args.val_ratio, args.seed)
    requested = train_samples + val_samples
    if args.max_samples is not None:
        requested = requested[: args.max_samples]
    stats = Counter()
    if args.reuse_cache_root:
        stats["linked"] = reuse_existing_cache(requested, args.reuse_cache_root, args.cache_root)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(lambda sample: prepare_one(sample, args), requested)
        for status, failure in tqdm(results, total=len(requested), desc="Preparing full PBRT cache"):
            stats[status] += 1
            if failure is not None:
                failures.append(failure)
    manifest = {
        "protocol": "full_pbrt_excluding_seed123_test",
        "train_cache": str(args.cache_root),
        "holdout_cache": holdout_cache,
        "iq_root": str(args.noise_iq_root),
        "ideal_iq_root": str(args.ideal_iq_root),
        "gt_depth_root": str(args.gt_depth_root),
        "noise_depth_root": str(args.noise_depth_root),
        "test_source_manifest": test_source,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "counts": {"all_pbrt": len(all_samples), "train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        "scenes": {split: sorted(set(sample.split("/")[0] for sample in samples)) for split, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples))},
        "samples": {"train": train_samples, "val": val_samples, "test": test_samples},
        "preparation": {
            "written": stats["written"],
            "linked": stats["linked"],
            "existing": stats["existing"],
            "repaired": stats["repaired"],
            "failures": failures,
        },
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"counts": manifest["counts"], "preparation": manifest["preparation"]}, indent=2))


if __name__ == "__main__":
    main()
