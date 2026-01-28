import os
from pathlib import Path

import datasets

_VERSION = datasets.Version("0.0.1")

_FEATURES = datasets.Features(
    {
        "ideal_IQ_path": datasets.Value("string"),
        "noise_IQ_path": datasets.Value("string"),
        "prompt": datasets.Value("string"),
        "conf_path": datasets.Value("string"),
    }
)


class PBRTDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=_VERSION, description="PBRT dataset without mask"),
        datasets.BuilderConfig(name="masked", version=_VERSION, description="PBRT dataset with amplitude mask"),
    ]

    def _info(self):
        return datasets.DatasetInfo(features=_FEATURES, supervised_keys=None)

    def _split_generators(self, dl_manager):
        # Expect repository-local layout under this module: pbrt_dataset/data
        # Note: when datasets library loads local modules it may copy files to a
        # temporary directory; try multiple fallbacks so we still locate the
        # original dataset directory in common cases.
        candidate_bases = []
        # 1) next to this file (usual when run directly)
        candidate_bases.append(Path(__file__).resolve().parent / "data")
        # 2) relative to current working directory (when datasets runs code from temp)
        candidate_bases.append(Path.cwd() / "pbrt_dataset" / "data")
        # 3) common absolute path used in this workspace
        candidate_bases.append(Path("/data/pre_student/GJ/DepthCAD/pbrt_dataset/data"))

        base = None
        for cb in candidate_bases:
            if cb.exists():
                base = cb
                break

        if base is None:
            # Let the datasets library raise a clear error if no data found
            base = Path(__file__).resolve().parent / "data"

        # Select data directory based on config (with or without mask)
        if self.config.name == "masked":
            ideal_dir = str(base / "ideal_IQ_masked")
            noise_dir = str(base / "noise_IQ_masked")
            conf_dir = str(base / "confidence_masked")
        else:  # default
            ideal_dir = str(base / "ideal_IQ")
            noise_dir = str(base / "noise_IQ")
            conf_dir = str(base / "confidence")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "ideal_IQ_dir": ideal_dir,
                    "noise_IQ_dir": noise_dir,
                    "conf_dir": conf_dir,
                },
            )
        ]

    def _generate_examples(self, ideal_IQ_dir, noise_IQ_dir, conf_dir):
        """Walk `ideal_IQ_dir` and yield entries for every .npy file found.

        The produced example dict matches the `flat_dataset` layout so training code can reuse
        the same preprocessing pipeline.
        """
        ideal_IQ_dir = Path(ideal_IQ_dir)
        noise_IQ_dir = Path(noise_IQ_dir)
        conf_dir = Path(conf_dir)

        idx = 0
        for root, _, files in os.walk(ideal_IQ_dir):
            # root example: /.../pbrt_dataset/data/ideal_IQ/breakfast/0
            rel_root = Path(root).relative_to(ideal_IQ_dir)
            for fname in sorted(files):
                if not fname.endswith(".npy"):
                    continue

                ideal_path = Path(root) / fname

                # map to noise path by replacing the top-level dir component
                noise_path = noise_IQ_dir / rel_root / fname

                # confidence files are saved per-base (without suffix _A.._F), e.g. 100.npy
                base_name = Path(fname).stem  # e.g. '100_A'
                conf_base = base_name.split("_")[0] + ".npy"
                conf_path = conf_dir / rel_root / conf_base

                # ensure corresponding noise and conf exist
                if not noise_path.exists() or not conf_path.exists():
                    # skip incomplete sample
                    continue

                # idx string uses relative dir plus base (no extension), e.g. 'breakfast/0/100_A'
                idx_str = str(rel_root / Path(fname).stem)

                yield idx_str, {
                    "ideal_IQ_path": str(ideal_path),
                    "noise_IQ_path": str(noise_path),
                    "prompt": "",
                    "conf_path": str(conf_path),
                }

                idx += 1
