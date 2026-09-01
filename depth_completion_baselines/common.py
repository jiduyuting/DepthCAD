import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_sample_id(item, cache_root):
    path = Path(item)
    if path.suffix == ".npz":
        if path.is_absolute():
            path = path.relative_to(cache_root)
        else:
            parts = path.parts
            if cache_root.name in parts:
                path = Path(*parts[parts.index(cache_root.name) + 1 :])
        path = path.with_suffix("")
    return path.as_posix()


def amplitude_to_rgb(amplitude, valid_mask):
    if amplitude.ndim == 2:
        amplitude = np.repeat(amplitude[None], 3, axis=0)
    normalized = np.empty_like(amplitude, dtype=np.float32)
    for channel in range(3):
        values = amplitude[channel][valid_mask]
        scale = float(np.percentile(values, 99.0)) if values.size else 1.0
        normalized[channel] = np.clip(amplitude[channel] / max(scale, 1e-6), 0.0, 1.0)
    return normalized.transpose(1, 2, 0)


def default_intrinsics(height, width):
    focal = float(max(height, width))
    return np.asarray(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


class PBRTCompletionDataset(Dataset):
    def __init__(self, cache_root, split_json, split="test", limit=None):
        self.cache_root = Path(cache_root).expanduser().resolve()
        with Path(split_json).open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        if split not in split_data and split == "test":
            split = "val"
        self.sample_ids = [
            normalize_sample_id(item, self.cache_root) for item in split_data[split]
        ]
        if limit is not None:
            self.sample_ids = self.sample_ids[:limit]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        with np.load(self.cache_root / f"{sample_id}.npz", allow_pickle=False) as data:
            sparse_depth = data["depth_noisy"].astype(np.float32)
            target = data["gt_depth"].astype(np.float32)
            amplitude = data["noisy_amplitude"].astype(np.float32)
            hole_mask = data["hole_mask"].astype(bool)
            valid_mask = data["valid_mask"].astype(bool)

        sparse_depth = np.nan_to_num(sparse_depth, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        sparse_depth[hole_mask] = 0.0
        rgb = amplitude_to_rgb(amplitude, valid_mask)
        height, width = target.shape
        intrinsics = default_intrinsics(height, width)

        return {
            "sample_id": sample_id,
            "image": torch.from_numpy(rgb).permute(2, 0, 1),
            "sparse_depth": torch.from_numpy(sparse_depth),
            "target": torch.from_numpy(target),
            "hole_mask": torch.from_numpy(hole_mask),
            "valid_mask": torch.from_numpy(valid_mask),
            "intrinsics": torch.from_numpy(intrinsics),
        }


class MetricAccumulator:
    def __init__(self):
        self.stats = defaultdict(lambda: {"count": 0, "abs_sum": 0.0, "sq_sum": 0.0})

    def add(self, name, prediction, target, mask):
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        mask &= np.isfinite(prediction) & np.isfinite(target)
        if not mask.any():
            return
        error = prediction[mask] - target[mask]
        values = self.stats[name]
        values["count"] += int(mask.sum())
        values["abs_sum"] += float(np.abs(error).sum())
        values["sq_sum"] += float(np.square(error).sum())

    def summary(self):
        output = {}
        for name, values in self.stats.items():
            count = values["count"]
            output[name] = {
                "count": count,
                "mae_m": values["abs_sum"] / count,
                "rmse_m": (values["sq_sum"] / count) ** 0.5,
            }
        return output


def evaluate_prediction(accumulator, prediction, sample):
    target = sample["target"].detach().cpu().numpy()
    valid = sample["valid_mask"].detach().cpu().numpy().astype(bool)
    hole = sample["hole_mask"].detach().cpu().numpy().astype(bool)
    accumulator.add("global", prediction, target, valid)
    accumulator.add("hole", prediction, target, valid & hole)
    accumulator.add("observed", prediction, target, valid & ~hole)


def save_summary(path, method, dataset, metrics, **metadata):
    result = {
        "method": method,
        "samples": len(dataset),
        "metrics": metrics,
        **metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
