# DepthCAD 掩码实验指南

## 概述

本指南介绍如何在 PBRT 数据集上使用振幅掩码来提升 DepthCAD 模型性能。

## 掩码原理

### 振幅掩码（Amplitude Masking）
基于原始数据的振幅通道（通道 2, 5, 8）来识别低质量区域：
- **低振幅区域**：信号弱，噪声主导，深度估计不可靠
- **高振幅异常**：可能的测量异常值

### 掩码策略
1. **下限阈值**：自适应 5% 分位数（可配置）
   - 掩盖掉振幅值最小的 5% 像素
   - 这些区域的信噪比太低

2. **上限阈值**：99.5% 分位数（可选）
   - 掩盖掉振幅值最大的 0.5% 像素
   - 去除可能的异常值

3. **掩码应用**：
   - IQ 数据：掩码区域置 0
   - 置信度图：掩码区域置信度置 0
   - 模型学习忽略这些区域

## 使用步骤

### 1. 生成带掩码的数据

```bash
bash generate_masked_data.sh
```

或直接运行：
```bash
python pbrt_dataset/process_mask.py
```

**输出位置**：
- `pbrt_dataset/data/ideal_IQ_masked/` - 带掩码的理想 IQ
- `pbrt_dataset/data/noise_IQ_masked/` - 带掩码的噪声 IQ
- `pbrt_dataset/data/confidence_masked/` - 带掩码的置信度图

### 2. 训练带掩码的模型

```bash
bash train_masked.sh
```

这会使用 `--dataset_config="masked"` 来加载带掩码的数据。

### 3. 评估性能差异

使用相同的评估脚本比较两种模型：
- 无掩码模型（`train.sh`）
- 带掩码模型（`train_masked.sh`）

## 调整掩码参数

编辑 `pbrt_dataset/process_mask.py` 中的参数：

```python
# 第 164-170 行
target_size = (512, 512)  # 分辨率

# 下限阈值：None = 自适应 5%，或设置固定值
amp_thresh = None  # 自适应
# amp_thresh = 0.01  # 固定阈值

# 上限阈值：99.5 = 掩盖前 0.5%，None = 禁用
upper_percent = 99.5  # 启用
# upper_percent = None  # 禁用
```

修改后需要重新运行 `generate_masked_data.sh`。

## 掩码代码分析

### 正确的部分 ✅
1. **振幅通道提取**：`correlations[[2, 5, 8]]` 正确
2. **掩码计算逻辑**：基于分位数的自适应阈值合理
3. **置信度处理**：掩码区域置信度置 0 正确
4. **IQ 数据处理**：掩码区域置 0 是合理的做法

### 已修复的问题
1. **分辨率统一**：从 256×256 改为 512×512，与其他模块一致
2. **数据加载支持**：`pbrt_dataset.py` 添加了 `masked` 配置
3. **训练脚本**：`train.py` 添加 `--dataset_config` 参数

## 对比实验

建议进行以下对比：

| 实验组 | 数据集 | 掩码配置 | 预期效果 |
|--------|--------|----------|----------|
| Baseline | 无掩码 | N/A | 当前性能 |
| Exp 1 | 5% 下限 | amp_thresh=None | 过滤低质量区域 |
| Exp 2 | 5% 下限 + 99.5% 上限 | + upper_percent=99.5 | 额外过滤异常值 |
| Exp 3 | 固定阈值 | amp_thresh=0.01 | 更激进的掩码 |

## 预期收益

掩码可能带来的好处：
1. **减少噪声影响**：模型不学习低质量区域的噪声模式
2. **提升泛化能力**：专注于高质量信号区域
3. **更好的深度估计**：在掩码区域的深度由模型从周围区域推断

可能的缺点：
1. **信息损失**：掩码区域完全依赖模型推断
2. **边缘效应**：掩码边界可能产生伪影
3. **训练难度增加**：模型需要学习推断缺失信息

## 技术细节

### 数据流

```
原始数据 (9通道)
    ↓
resize 到 512×512
    ↓
提取 IQ 对 + 振幅通道
    ↓
sqrt_ldr 变换
    ↓
计算振幅掩码 ← 基于振幅通道
    ↓
应用掩码到 IQ 数据（置0）
    ↓
归一化（除以 noise_max）
    ↓
保存为 6 个单独的 .npy 文件
```

### 掩码 vs 无掩码对比

| 方面 | 无掩码 | 有掩码 |
|------|--------|--------|
| IQ 数据 | 保留所有值 | 低振幅区域为 0 |
| 置信度图 | 梯度置信度 | 梯度置信度 × 掩码 |
| 模型输入 | 完整信息 | 部分缺失 |
| 模型任务 | 去噪 | 去噪 + 修复 |

## 文件结构

```
DepthCAD/
├── pbrt_dataset/
│   ├── data/
│   │   ├── ideal_IQ/           # 无掩码理想 IQ
│   │   ├── noise_IQ/           # 无掩码噪声 IQ
│   │   ├── confidence/         # 无掩码置信度
│   │   ├── ideal_IQ_masked/    # 带掩码理想 IQ
│   │   ├── noise_IQ_masked/    # 带掩码噪声 IQ
│   │   └── confidence_masked/  # 带掩码置信度
│   ├── pbrt_dataset.py         # 数据加载（已更新）
│   ├── preprocess.py           # 无掩码预处理
│   └── process_mask.py         # 带掩码预处理（已更新）
├── train.py                    # 训练脚本（已更新）
├── train.sh                    # 无掩码训练
├── train_masked.sh             # 带掩码训练
└── generate_masked_data.sh     # 生成掩码数据
```

## 故障排除

### 问题：掩码数据不存在
```bash
# 检查目录是否存在
ls pbrt_dataset/data/ideal_IQ_masked/

# 如果不存在，运行
bash generate_masked_data.sh
```

### 问题：训练时找不到掩码配置
确认 `pbrt_dataset/pbrt_dataset.py` 已更新，包含：
```python
BUILDER_CONFIGS = [
    datasets.BuilderConfig(name="default", ...),
    datasets.BuilderConfig(name="masked", ...),
]
```

### 问题：分辨率不匹配
确认 `process_mask.py` 中的 `target_size = (512, 512)`

## 引用

如果掩码提升了性能，在论文中可以这样描述：

> We employ amplitude-based masking to filter out low-quality measurements. Regions with amplitude values below the 5th percentile or above the 99.5th percentile are masked (set to zero in both IQ data and confidence maps). This prevents the model from learning noise patterns in unreliable regions while encouraging it to inpaint missing information from surrounding high-quality areas.
