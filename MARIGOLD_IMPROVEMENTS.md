# Marigold-style Improvements for DepthCAD Training

本文档说明了对 train2.py 的 Marigold 微调协议改进。

## 参考资源

- Marigold GitHub: https://github.com/prs-eth/Marigold
- Marigold Paper: https://arxiv.org/abs/2505.09358

## 核心改进

### 1. 增强的 ControlNet 输入编码 (Enhanced Conditioning)

**原方法 (2 通道):**
```python
depthcad_image = torch.concatenate([batch["noises"], batch["confs"]], dim=1)
# Shape: [B, 2, H, W]
```

**改进后 (6 通道):**
```python
depthcad_image = compute_enhanced_features(batch["noises"], batch["confs"])
# Shape: [B, 6, H, W]
```

增强特征包括：
- `noise`: 原始噪声图
- `conf`: 原始置信度图
- `noise_dx`: 噪声 x 方向梯度
- `noise_dy`: 噪声 y 方向梯度
- `conf_dx`: 置信度 x 方向梯度
- `conf_dy`: 置信度 y 方向梯度

**优势:**
- 提供更丰富的边缘和结构信息
- 帮助模型更好地学习深度不连续性
- 类似于 Marigold 使用多尺度特征的思想

### 2. LoRA 参数高效微调 (LoRA Fine-tuning)

使用 LoRA (Low-Rank Adaptation) 进行微调，而非全参数训练。

**配置参数:**
```python
lora_config = LoraConfig(
    r=16,              # Rank (低秩矩阵维度)
    lora_alpha=32,     # Alpha (缩放因子)
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # 应用的模块
    lora_dropout=0.1,  # Dropout 率
    bias="none",
    task_type="CONTROLNET",
)
```

**优势:**
- 大幅减少可训练参数数量 (通常 < 1%)
- 降低过拟合风险
- 减少显存占用
- 允许使用更大的学习率

**参数说明:**
- `r` (rank): 控制适配器的表达能力。越大 = 越多参数 = 更强表达能力
  - 典型值: 8, 16, 32, 64
  - 建议: 从 16 开始，根据效果调整
- `lora_alpha`: 缩放因子，控制 LoRA 更新的幅度
  - 典型值: 16, 32, 64
  - 建议: 设置为 2*r
- `lora_dropout`: 防止过拟合
  - 典型值: 0.0, 0.05, 0.1
  - 建议: 小数据集使用 0.1，大数据集可以设为 0

## 使用方法

### 命令行参数

新增的命令行参数：

```bash
--use_enhanced_features     # 启用增强特征 (6通道)
--use_lora                  # 启用 LoRA 微调
--lora_rank 16              # LoRA rank (默认: 16)
--lora_alpha 32             # LoRA alpha (默认: 32)
--lora_dropout 0.1          # LoRA dropout (默认: 0.1)
```

### 训练配置示例

#### 方法 1: 基线 (原始方法)
```bash
python train2.py \
    --dataset_name=flat_dataset \
    --output_dir=model_baseline \
    --resolution=512 \
    --train_batch_size=4 \
    --learning_rate=5e-6 \
    --num_train_epochs=1
```

#### 方法 2: 仅增强特征
```bash
python train2.py \
    --dataset_name=flat_dataset \
    --output_dir=model_enhanced \
    --resolution=512 \
    --train_batch_size=4 \
    --learning_rate=5e-6 \
    --num_train_epochs=1 \
    --use_enhanced_features
```

#### 方法 3: 仅 LoRA
```bash
python train2.py \
    --dataset_name=flat_dataset \
    --output_dir=model_lora \
    --resolution=512 \
    --train_batch_size=4 \
    --learning_rate=1e-4 \
    --num_train_epochs=1 \
    --use_lora \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1
```

#### 方法 4: 完整 Marigold 风格 (推荐)
```bash
python train2.py \
    --dataset_name=flat_dataset \
    --output_dir=model_marigold \
    --resolution=512 \
    --train_batch_size=4 \
    --learning_rate=1e-4 \
    --num_train_epochs=1 \
    --use_enhanced_features \
    --use_lora \
    --lora_rank=16 \
    --lora_alpha=32 \
    --lora_dropout=0.1
```

## 模型保存

当使用 LoRA 时，模型会保存为两部分：

1. **LoRA 适配器**: `{output_dir}/lora/`
   - 仅包含 LoRA 权重
   - 文件小，易于分享
   - 可用于后续继续训练

2. **合并的完整模型**: `{output_dir}/`
   - LoRA 权重合并到基础模型
   - 可直接用于推理
   - 标准的 ControlNet 格式

## 超参数调优建议

### 学习率
- **不使用 LoRA**: `5e-6` (较小，因为训练所有参数)
- **使用 LoRA**: `1e-4` 到 `5e-4` (可以更大，因为只训练少量参数)

### LoRA Rank 选择
- **小数据集** (< 1000 样本): r=8 或 16
- **中等数据集** (1000-10000): r=16 或 32
- **大数据集** (> 10000): r=32 或 64

### 批次大小
- LoRA 允许使用更大的批次大小，因为显存占用更少
- 可以尝试从 batch_size=4 增加到 8 或 16

## 依赖安装

确保安装了 PEFT 库：

```bash
pip install peft
```

或使用 requirements:

```bash
pip install peft transformers diffusers accelerate torch torchvision
```

## 预期效果

根据 Marigold 的经验，这些改进应该能够：

1. **提高深度估计精度**: 梯度特征提供更好的边缘感知
2. **减少过拟合**: LoRA 限制可训练参数数量
3. **加快训练速度**: 更少的参数意味着更快的梯度计算
4. **更好的泛化**: LoRA 的正则化效果

## 与原始方法对比

| 特性 | 原始方法 | Marigold 改进 |
|------|---------|--------------|
| Conditioning 通道 | 2 | 6 (含梯度) |
| 可训练参数 | 100% | < 1% (使用 LoRA) |
| 显存占用 | 高 | 低 |
| 过拟合风险 | 高 | 低 |
| 学习率 | 5e-6 | 1e-4 |
| 边缘感知 | 基础 | 增强 |

## 进一步改进方向

1. **多尺度特征**: 添加不同尺度的特征图
2. **Attention 融合**: 在 attention 层中融合条件信息
3. **更复杂的 LoRA target**: 尝试不同的 target_modules 组合
4. **Adapter 模块**: 添加额外的适配器层
5. **数据增强**: 针对深度图的数据增强策略

## 参考资料

- LoRA Paper: https://arxiv.org/abs/2106.09685
- PEFT Documentation: https://huggingface.co/docs/peft
- Marigold: https://github.com/prs-eth/Marigold