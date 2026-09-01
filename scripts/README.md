# Utility Scripts

- `data_prep/`: raw data conversion, hole generation, and dataset preparation.
- `analysis/`: evaluation summaries, diagnostics, and result organization.
- `tests/`: standalone smoke tests and data checks.
- Root-level files in this directory: training, inference, and evaluation entry points.
- `runs/`: shell launchers for repeatable experiments.

Run Python entry points from the repository root, for example `python scripts/inference_pbrt.py ...`.
Run experiment launchers with `bash scripts/runs/<name>.sh`; each launcher sets the repository root and Python import paths before running.
