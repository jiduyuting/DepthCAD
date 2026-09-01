import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def robust_scale(array, valid, percentile=99.0):
    values = np.abs(array[:, valid]) if array.ndim == 3 else np.abs(array[valid])
    scale = float(np.percentile(values, percentile)) if values.size else 1.0
    return max(scale, 1e-6)


class UnifiedPbrtDataset(Dataset):
    def __init__(self, manifest, split, max_samples=None):
        payload = json.loads(Path(manifest).read_text())
        self.samples = list(payload["samples"][split])
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
        self.cache_root = Path(payload["holdout_cache"] if split == "test" else payload["train_cache"])
        self.iq_root = Path(payload["iq_root"])

    def __len__(self):
        return len(self.samples)

    def load_iq(self, sample, shape):
        scene, view, frame = sample.split("/")
        channels = []
        for channel in "ABCDEF":
            array = np.load(self.iq_root / scene / view / f"{frame}_{channel}.npy").astype(np.float32)
            if array.shape != shape:
                array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            channels.append(array)
        return np.stack(channels, axis=0)

    def __getitem__(self, index):
        sample = self.samples[index]
        scene, view, frame = sample.split("/")
        path = self.cache_root / scene / view / f"{frame}.npz"
        with np.load(path, allow_pickle=False) as data:
            noisy_depth = data["depth_noisy"].astype(np.float32)
            target = data["gt_depth"].astype(np.float32)
            hole = data["hole_mask"] > 0.5
            confidence = data["confidence"].astype(np.float32)
            amplitude = data["noisy_amplitude_mean"].astype(np.float32)
            valid = data["valid_mask"] > 0.5
        iq = self.load_iq(sample, target.shape)
        iq = np.clip(iq / robust_scale(iq, valid), -3.0, 3.0).astype(np.float32)
        amplitude = np.clip(amplitude / robust_scale(amplitude, valid), 0.0, 3.0).astype(np.float32)
        return {
            "sample_name": sample,
            "iq": torch.from_numpy(iq),
            "depth": torch.from_numpy(noisy_depth[None]),
            "amplitude": torch.from_numpy(amplitude[None]),
            "confidence": torch.from_numpy(confidence[None]),
            "target": torch.from_numpy(target[None]),
            "hole_mask": torch.from_numpy(hole[None]),
            "valid_mask": torch.from_numpy(valid[None]),
        }
