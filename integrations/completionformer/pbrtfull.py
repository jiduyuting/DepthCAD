import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def _load_split(path, mode):
    with Path(path).open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    if mode == "train":
        key = "train"
    else:
        key = "test" if "test" in split else "val"

    samples = split.get(key)
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Split {path} does not contain a non-empty '{key}' list")
    return samples


def _sample_id(item, cache_root):
    item_path = Path(item)
    if item_path.suffix == ".npz":
        if item_path.is_absolute():
            try:
                item_path = item_path.relative_to(cache_root)
            except ValueError:
                return item_path.with_suffix("")
        else:
            cache_name = cache_root.name
            parts = item_path.parts
            if cache_name in parts:
                item_path = Path(*parts[parts.index(cache_name) + 1 :])
        item_path = item_path.with_suffix("")
    return item_path


def _normalize_amplitude(amplitude, valid):
    guide = np.empty_like(amplitude, dtype=np.float32)
    for channel in range(3):
        scale_values = amplitude[channel][valid]
        scale = float(np.percentile(scale_values, 99.0)) if scale_values.size else 1.0
        guide[channel] = np.clip(amplitude[channel] / max(scale, 1e-6), 0.0, 1.0)
    return ((guide - _IMAGENET_MEAN) / _IMAGENET_STD).astype(np.float32)


class PBRTFull(Dataset):
    """CompletionFormer adapter for DepthCAD's full PBRT completion cache."""

    def __init__(self, args, mode):
        if mode not in {"train", "test", "debug"}:
            raise NotImplementedError(mode)

        self.cache_root = Path(args.dir_data).expanduser().resolve()
        self.samples = [
            _sample_id(item, self.cache_root)
            for item in _load_split(args.split_json, mode)
        ]
        self.augment = bool(args.augment and mode == "train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        cache_path = sample_id if sample_id.is_absolute() else self.cache_root / sample_id
        cache_path = cache_path.with_suffix(".npz")

        with np.load(cache_path, allow_pickle=False) as data:
            depth = data["depth_noisy"].astype(np.float32)
            target = data["gt_depth"].astype(np.float32)
            amplitude = data["noisy_amplitude"].astype(np.float32)
            hole_mask = data["hole_mask"].astype(bool)
            valid_mask = data["valid_mask"].astype(bool)

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        depth[hole_mask] = 0.0
        guide = _normalize_amplitude(amplitude, valid_mask)

        if self.augment and np.random.random() < 0.5:
            guide = guide[:, :, ::-1].copy()
            depth = depth[:, ::-1].copy()
            target = target[:, ::-1].copy()
            hole_mask = hole_mask[:, ::-1].copy()
            valid_mask = valid_mask[:, ::-1].copy()

        return {
            "rgb": torch.from_numpy(guide),
            "dep": torch.from_numpy(depth[None]),
            "gt": torch.from_numpy(target[None]),
            "hole_mask": torch.from_numpy(hole_mask[None]),
            "valid_mask": torch.from_numpy(valid_mask[None]),
            "idx": str(sample_id),
        }
