import os
import cv2
import numpy as np
import pandas as pd

import datasets

_VERSION = datasets.Version("0.0.1")

# RGB 图像基础路径
_DEFAULT_RGB_DIR = "/data/pre_student/hcy/LFRD2/results/pbrt/png"

_FEATURES = datasets.Features(
    {
        "ideal_IQ_path": datasets.Value("string"),
        "noise_IQ_path": datasets.Value("string"),
        "prompt": datasets.Value("string"),
        "conf_path": datasets.Value("string"),
        "rgb_path": datasets.Value("string"),  # 新增: RGB 图像路径
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


def bin_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError

    shape = (424, 512)
    target_size = (512, 512)
    data = np.fromfile(path, dtype=np.float32).reshape(shape)
    np_data = np.nan_to_num(data, 0)

    resized_data = cv2.resize(np_data, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized_data.astype(np.float32)


class FLATDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=_FEATURES,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,

                gen_kwargs={
                    "metadata_path": "/data/pre_student/GJ/DepthCAD/flat_dataset/data/train.jsonl",
                    "ideal_IQ_dir": "/data/pre_student/GJ/DepthCAD/flat_dataset/data/ideal_IQ",
                    "noise_IQ_dir": "/data/pre_student/GJ/DepthCAD/flat_dataset/data/noise_IQ",
                    "conf_dir": "/data/pre_student/GJ/DepthCAD/flat_dataset/data/confidence",
                    "rgb_dir": _DEFAULT_RGB_DIR,  # 新增: RGB 图像目录
                },
            ),
        ]

    def _generate_examples(self, metadata_path, ideal_IQ_dir, noise_IQ_dir, conf_dir, rgb_dir=None):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            prompt = row["text"]
            idx = row["idx"]

            # 解析 idx 获取场景和视图信息
            # idx 格式: scene_view_num_letter, e.g., "1499392477071669_A"
            # 需要从 ideal_IQ_dir 结构中推断场景和视图
            ideal_IQ_path = os.path.join(ideal_IQ_dir, f"{idx}.npy")
            noise_IQ_path = os.path.join(noise_IQ_dir, f"{idx}.npy")
            conf_path = os.path.join(conf_dir, f"{idx.split('_')[0]}.npy")

            # RGB 路径 (如果提供)
            rgb_path = ""
            if rgb_dir:
                # 从 IQ 路径推断场景和视图
                # 结构: ideal_IQ_dir/scene/view/idx.npy
                rel_path = os.path.relpath(ideal_IQ_path, ideal_IQ_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    scene = parts[0]   # e.g., bathroom
                    view = parts[1]    # e.g., 0
                    rgb_idx = idx.split('_')[0]  # e.g., 100
                    rgb_path = os.path.join(rgb_dir, scene, view, f"{rgb_idx}.png")

            yield idx, {
                "prompt": prompt,
                "ideal_IQ_path": ideal_IQ_path,
                "noise_IQ_path": noise_IQ_path,
                "conf_path": conf_path,
                "rgb_path": rgb_path,  # 新增
            }
