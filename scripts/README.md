# Utility Scripts

- `data_prep/`: raw data conversion, hole generation, and dataset preparation.
- `analysis/`: evaluation summaries, diagnostics, and result organization.
- `tests/`: standalone smoke tests and data checks.
- `flow/`: Flow-based depth restoration, propagation refinement, and real raw9 experiments.
- Root-level files in this directory: training, inference, and evaluation entry points.
- `runs/`: shell launchers for repeatable experiments; Flow launchers are in `runs/flow/`.

Run Python entry points from the repository root, for example `python scripts/inference_pbrt.py ...`.
Run experiment launchers with `bash scripts/runs/<name>.sh`; each launcher sets the repository root and Python import paths before running.
