# Flow 深度补全一键实验

## 目标与隔离原则

- 主目标：PBRT100 `Hole MAE < 0.102960`，超过 RGBD-Imaging。
- 联合目标：同时满足 `Global MAE < 0.038681`，超过 CompletionFormer。
- 8910 张 train 用于更新权重；990 张 val 用于选择 checkpoint。
- PBRT100 的 100 张 test 只在全部训练结束后评测，不参与选模型。
- GT-invalid 不作为零深度监督；所有洞指标使用 `hole_mask & valid_mask`。

## 实验矩阵

第一阶段并行微调现有 epoch-108 Flow：

加载旧 optimizer 状态后会显式把学习率重设为本轮配置，避免 checkpoint 中的 `1e-4` 覆盖微调学习率 `1e-5`。

| ID | Endpoint | t=0 比例 | 粗块洞增强 | 边界权重 | 目的 |
|---|---:|---:|:---:|---:|---|
| E0 | 2 | 0.25 | 否 | 0.5 | endpoint 控制组 |
| E1 | 2 | 0.25 | 是 | 0.5 | 主实验 |
| E2 | 2 | 0.50 | 是 | 1.0 | endpoint/边界强化 |
| E3 | 4 | 0.25 | 是 | 0.5 | endpoint 权重消融 |

粗块洞增强在 train 中以 50% 概率使用 4/8/12 像素块和 15%-20% 洞比例。val 同时包含原缓存洞和固定 seed 的 stress 洞；两者混合后的连通域数量和边缘占比更接近 PBRT100。

第二阶段只使用 val 最优 Flow，缓存其 anchor，然后并行训练：

| ID | Refine 区域 | 容量 | 目的 |
|---|---|---|---|
| R1 | hole + 3 px | 32 channels / 6 steps | 局部传播基线 |
| R2 | 全图 | 32 channels / 6 steps | 同时改善 observed |
| R3 | 全图 | 32 channels / 2 blocks / 8 steps | 强化版本 |

## 一键运行

自动使用所有可见 GPU：

```bash
bash scripts/runs/flow/run_flow_sota_experiments.sh
```

指定 GPU：

```bash
GPUS=0,1,2,3 bash scripts/runs/flow/run_flow_sota_experiments.sh
```

指定固定输出目录，重复运行时自动从各实验的 `last.pt` 续训：

```bash
RUN_ROOT=output/flow_sota_experiments/main_v1 \
GPUS=0,1,2,3 \
bash scripts/runs/flow/run_flow_sota_experiments.sh
```

如果续训后所选 Flow checkpoint 发生变化，启动器会自动使旧 anchor 缓存失效并重建。

短流程检查：

```bash
MODE=smoke GPUS=0 bash scripts/runs/flow/run_flow_sota_experiments.sh
```

没有 GPU 时只做 CPU 流程检查：

```bash
MODE=smoke GPUS=cpu bash scripts/runs/flow/run_flow_sota_experiments.sh
```

## 输出

```text
RUN_ROOT/
├── stage1/                  # E0-E3 Flow checkpoints
├── protocol_audit.json      # raw/effective hole 与几何统计
├── selected_flow.json      # 仅根据 val 选择
├── anchor_cache/           # 所选 Flow 的固定 anchors
├── stage2/                  # R1-R3 Refine checkpoints
├── test_eval/               # 训练完成后的 PBRT100 结果
├── logs/                    # 每个 GPU job 的独立日志
├── summary.json
└── summary.md
```

最终 `summary.md` 会明确标记每个模型是否同时超过 RGBD Hole 和 CompletionFormer Global。
现有 `depth_flow_full_pbrt_iq_propagation_refine/best.pt` 也会按新的 effective-hole 口径重评，作为 `current_refine_baseline` 出现在同一张表中。
