# Experiment Launchers

Repeatable training, inference, and evaluation commands live in this directory.
Run them from the repository root, for example:

```bash
bash scripts/runs/run_pbrt100_all.sh
```

Each launcher resolves the repository root from its own location and adds both
`scripts/` and the repository root to `PYTHONPATH`, so imports remain stable
after the project layout cleanup.
