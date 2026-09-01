# Flow Pipeline

This directory contains the Flow-based depth restoration pipeline:

- `train_depth_flow_restoration.py` and `eval_depth_flow_restoration.py`: base conditional Flow model.
- `train_depth_flow_propagation_refine.py` and `eval_depth_flow_propagation_refine.py`: frozen-anchor propagation refinement.
- `train_real_raw9_flow_finetune.py` and `infer_real_raw9_flow.py`: real raw9 fine-tuning and inference.
- `train_real_raw9_propagation_refine.py` and `infer_real_raw9_propagation_refine.py`: real-scene refinement.
- `train_synthetic_realhole_flow_pretrain.py`: synthetic pretraining with real-hole masks.
- `cache_flow_anchors.py`, `audit_flow_protocol.py`, and the remaining utilities: cache, protocol, split, and result tooling.

Launchers for these experiments are in `scripts/runs/flow/`. Run them from the
repository root, for example:

```bash
bash scripts/runs/flow/run_flow_sota_experiments.sh
```

The shared restoration dataset and backbone helpers remain one level above in
`scripts/`, because non-Flow restoration and comparison tools reuse them.
