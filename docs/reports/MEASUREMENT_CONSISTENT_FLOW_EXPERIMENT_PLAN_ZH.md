# Measurement-Consistent Flow 泛化实验方案

## 目标

当前 Flow 在 PBRT100 的 hole 区域已经接近或超过 RGBD，但 global/valid 区域仍然落后。后续实验不再以调节单个 loss 权重为主，而是验证一个方法级假设：深度补全需要同时满足测量一致性、置信度控制和跨退化泛化。

## 方法候选

### M0: 当前 Flow baseline

使用 `transformer_bottleneck + noisy_iq_amp + noisy_ns anchor`，作为所有实验的固定基线。

### M1: Measurement projection

在每个 flow sampling step 后，把可靠观测点投影回输入测量；hole 区域保留模型预测。评测时同时报告 hole MAE 和 observed preservation error。

### M2: Confidence-gated residual

模型预测 residual 和像素级 gate，而不是直接预测完整深度：

```text
output = anchor + gate(confidence, amplitude, phase) * residual
```

高置信度观测点只允许小修正，低置信度点允许更大修正，hole 区域由 flow 完成。

### M3: 完整方法

M1 + M2，并在训练中混合 block、随机稀疏、边界、amplitude-derived、Kinect 风格和 real-hole mask。当前 runner 中的 `scene_mixed_mask` 是 M3 的数据增强 proxy；M1/M2 的网络改动需要后续在同一 cache/list 协议上实现。

## 数据协议

### PBRT in-domain

沿用现有 100 张 seed123 测试集，报告 Global/Hole/Valid MAE 和 RMSE。

### PBRT unseen-scene

按场景划分，训练和验证不包含 holdout 场景。例如训练 `bathroom, breakfast, pavilion, white-room`，测试整个 `contemporary-bathroom`。快速实验用一个 holdout 场景；最终实验轮换 5 个场景并报告均值和标准差。

### PBRT -> FLAT zero-shot

只在 PBRT 上训练，不使用任何 FLAT 训练样本，直接在 FLAT 上测试。`scripts/flow/prepare_flat_flow_cache.py` 将 FLAT 配对 IQ/深度转换为统一 `.npz` cache。

FLAT 的 `ideal_depth` 是由 ideal IQ 得到的 pseudo-GT，应明确标记为 pseudo-ground-truth，而不宣称是真实物理 GT。

### FLAT low-shot adaptation

按时间块划分 FLAT，使用 10% 做适配，90% 测试，不能随机打散相邻帧。报告 PBRT->FLAT zero-shot、10% adaptation 和 FLAT full-supervised 三种设置。

### Leave-one-corruption-out

训练时不使用测试 mask 类型，例如训练 block/random mask，测试 Kinect 或 real-hole mask，验证是否学到退化无关的恢复规律。

## 指标

每个协议固定报告 Global/Hole/Valid MAE 和 RMSE、observed preservation error、相对 PBRT in-domain 的跨域性能下降，以及至少 3 个随机种子的均值和标准差。

## 执行顺序

先运行 5 epoch pilot：

```bash
MODE=pilot HOLDOUT_SCENE=contemporary-bathroom GPU=0 \
  bash scripts/runs/flow/run_generalization_experiments.sh
```

pilot 只回答两个问题：混合退化是否改善未见 PBRT 场景，以及 PBRT 训练模型能否在 FLAT 上工作。只有 mixed-mask proxy 同时改善这两项，才进行完整训练：

```bash
MODE=full HOLDOUT_SCENE=contemporary-bathroom GPU=0 \
  bash scripts/runs/flow/run_generalization_experiments.sh
```

## 当前 DepthCAD/SD2.1 路线的定位

DepthCAD SD/ControlNet 的输入是 IQ，先生成 IQ，再通过深度估计器得到深度；当前 Flow 的输入是 noisy depth、hole/confidence 和 IQ 特征，直接在深度域恢复。两者不是同一个输入或输出空间，不能把 DepthCAD checkpoint 的结果直接和 Flow/RGBD 主表混合排名。

该路线可以保留为历史 baseline 或方法消融，但在当前 GPU 紧张、目标是泛化实验的情况下，不应继续优先训练。当前检查未发现它仍在运行，应优先完成 M0/M3 的场景独立和 PBRT->FLAT 评测。
