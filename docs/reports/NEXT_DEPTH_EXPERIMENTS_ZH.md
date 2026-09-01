# 下一轮 PBRT100 深度补全实验

## 实验目标

本轮只回答三个问题：

1. 实验 B 的验证集最优 checkpoint 在固定 PBRT100 test 上能否稳定复现；
2. 保留观测像素是否改善 Global/Observed 指标；
3. 保守续训（A）或 hole-distance 输入（C）能否把主指标 Hole MAE 降到 RGBD-Imaging 的 `0.100248` 以下。

训练只使用 `train.txt`，checkpoint 只由 990 张 `val.txt` 选择。100 张 `test.txt` 仅在训练完成后评估，不参与模型选择。A/C 的预注册主模型均为 `best_hole.pt`。

## 一键并行运行

默认使用三张 GPU：B checkpoint 评估在 GPU 0，A 在 GPU 1，C 在 GPU 2。

```bash
bash scripts/runs/run_next_depth_experiments_parallel.sh
```

指定 GPU：

```bash
GPU_B=0 GPU_A=2 GPU_C=3 bash scripts/runs/run_next_depth_experiments_parallel.sh
```

只有两张 GPU 时，先并行运行 B/A，再单独运行 C：

```bash
GPU_B=0 GPU_A=1 RUN_C=0 bash scripts/runs/run_next_depth_experiments_parallel.sh
GPU_C=0 RUN_B_EVAL=0 RUN_A=0 bash scripts/runs/run_next_depth_experiments_parallel.sh
```

快速短跑用于检查完整流程：

```bash
EPOCHS_A=110 EPOCHS_C=2 bash scripts/runs/run_next_depth_experiments_parallel.sh
```

A 从 epoch 108 继续训练，因此 `EPOCHS_A=110` 表示只训练两个 epoch。正式实验默认 A/C 的目标 epoch 都是 120。

## 产物

每次运行写入独立时间戳目录：

```text
output/next_depth_experiments/YYYYMMDD_HHMMSS/
├── a_conservative/
├── b_checkpoints/
├── c_hole_distance/
├── comparison/summary.md
└── logs/
```

脚本结束后，把终端打印的 `comparison/summary.md` 路径交给 Codex评估。判断顺序固定为：

1. Hole MAE（主指标）；
2. Hole RMSE；
3. Global MAE；
4. per-sample 胜率和最差样本可视化。

`preserve_observed` 是推理策略消融，不改变 hole 区域预测。它只有在 Global/Observed 指标改善且 Hole 指标完全不变时才有采用价值。
