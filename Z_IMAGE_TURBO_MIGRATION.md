# DepthCAD 适配 Z-Image-Turbo 重构方案

## 📋 架构对比分析

### 当前架构 (Stable Diffusion 2.1)
```python
Pipeline: StableDiffusionControlNetPipeline
Components:
  - text_encoder: CLIPTextModel
  - tokenizer: CLIPTokenizer
  - unet: UNet2DConditionModel
  - vae: AutoencoderKL (latent_channels=4)
  - scheduler: DDPMScheduler
  - controlnet: ControlNetModel (conditioning_channels=2)
```

### 目标架构 (Z-Image-Turbo)
```python
Pipeline: ZImagePipeline
Components:
  - text_encoder: Qwen3Model (⚠️ 不同)
  - tokenizer: Qwen2Tokenizer (⚠️ 不同)
  - transformer: ZImageTransformer2DModel (⚠️ 完全不同)
  - vae: AutoencoderKL (latent_channels=16) (⚠️ 通道数不同)
  - scheduler: FlowMatchEulerDiscreteScheduler (⚠️ 不同算法)
  - controlnet: QwenImageControlNetModel (⚠️ 需要验证)
```

## 🔧 关键修改点

### 1. 依赖升级

**问题**: Z-Image-Turbo 需要 diffusers >= 0.36.0，但当前是 0.31.0

**解决方案**:
```bash
pip install diffusers==0.36.0
pip install torch>=2.1.0  # 解决 torch.xpu 兼容性
```

### 2. train.py 修改

#### 2.1 text_encoder 加载 (train.py:73-85)
```python
# 原代码只支持 CLIP
def import_model_class_from_model_name_or_path(...):
    if model_class == "CLIPTextModel":
        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")

# 修改为支持 Qwen3
def import_model_class_from_model_name_or_path(...):
    text_encoder_config = PretrainedConfig.from_pretrained(...)
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "Qwen3ForCausalLM":  # ⚠️ 注意：可能是 Qwen3Model
        from transformers import Qwen3ForCausalLM, Qwen3Model
        return Qwen3Model  # 根据实际情况选择
    else:
        raise ValueError(f"{model_class} is not supported.")
```

#### 2.2 替换 UNet 加载为 Transformer (train.py:467-469)
```python
# 原代码
unet = UNet2DConditionModel.from_pretrained(...)

# 修改为
from diffusers.models.transformers import ZImageTransformer2DModel
transformer = ZImageTransformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="transformer",
    revision=args.revision,
    variant=args.variant
)
```

#### 2.3 ControlNet 创建 (train.py:471-476)
```python
# 原代码
if args.depthcad_path:
    depthcad = ControlNetModel.from_pretrained(args.depthcad_path)
else:
    depthcad = ControlNetModel.from_unet(unet, conditioning_channels=2)

# 修改为使用 QwenImageControlNetModel
from diffusers.models.controlnets import QwenImageControlNetModel

if args.depthcad_path:
    depthcad = QwenImageControlNetModel.from_pretrained(args.depthcad_path)
else:
    depthcad = QwenImageControlNetModel.from_original_model(
        transformer,
        conditioning_channels=2
    )
```

#### 2.4 训练循环修改 (train.py:700-718)

**核心问题**: ZImageTransformer 的 forward 签名可能与 UNet 不同

```python
# 原代码
down_block_res_samples, mid_block_res_sample = depthcad(
    noisy_latents,
    timesteps,
    encoder_hidden_states=encoder_hidden_states,
    controlnet_cond=depthcad_image,
    return_dict=False,
)

# 需要修改为 - 具体取决于 QwenImageControlNetModel 的 API
# 可能需要调整参数名称和返回值处理
```

#### 2.5 Scheduler 更换 (train.py:460)
```python
# 原代码
from diffusers import DDPMScheduler
noise_scheduler = DDPMScheduler.from_pretrained(...)

# 修改为
from diffusers import FlowMatchEulerDiscreteScheduler
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(...)
```

### 3. VAE 通道数问题 (train.py:681-682)

```python
# 原代码
latents = vae.encode(batch["ideals"].to(dtype=weight_dtype)).latent_dist.sample()
latents = latents * vae.config.scaling_factor

# Z-Image-Turbo 的 VAE 有 16 个通道而不是 4 个
# 可能需要调整后续处理逻辑
```

### 4. inference.py 修改

#### 4.1 Pipeline 更换 (inference.py:7, 296-298)
```python
# 原代码
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=depthcad, torch_dtype=torch.float16
)

# 修改为
from diffusers import ZImagePipeline
from diffusers.models.controlnets import QwenImageControlNetModel

pipe = ZImagePipeline.from_pretrained(
    base_model_path,
    controlnet=depthcad,  # ⚠️ 需要验证 ZImagePipeline 是否支持 controlnet 参数
    torch_dtype=torch.float16
)
```

## ⚠️ 风险和挑战

### 高风险项

1. **ControlNet 支持**
   - ❓ ZImagePipeline 可能不原生支持 ControlNet
   - 需要检查 QwenImageControlNetModel 是否可用
   - 可能需要自定义 Pipeline 类

2. **训练流程**
   - Flow-matching (新) vs DDPM (旧) 是完全不同的训练算法
   - Loss 计算、噪声添加、时间步采样都可能不同

3. **架构差异**
   - Transformer-based vs UNet-based
   - 参数维度、中间激活形状可能完全不同
   - ControlNet 的集成方式可能完全不同

### 中风险项

4. **VAE 通道数**
   - 4 通道 → 16 通道
   - 影响内存占用和计算量

5. **文本编码器**
   - CLIP → Qwen3
   - Tokenization 过程不同
   - Embedding 维度可能不同

## 🎯 实施策略

### 方案 A: 完全重构 (高风险，高收益)

1. 升级依赖到最新版本
2. 重写 train.py 和 inference.py
3. 适配 Z-Image-Turbo 架构
4. 测试并调试

**优点**: 可以使用最新的 Z-Image-Turbo 模型
**缺点**: 工作量大，风险高，可能需要几周时间

### 方案 B: 保持兼容 (推荐)

1. **保留现有的 stable-diffusion-2.1 代码**
2. 创建新的 `train_zimage.py` 和 `inference_zimage.py`
3. 逐步迁移和测试
4. 两个版本并存，根据需要选择

**优点**: 降低风险，可以逐步验证
**缺点**: 需要维护两套代码

### 方案 C: 混合方案

1. 仅替换 backbone (Transformer)
2. 保持其他组件 (VAE, 训练流程)
3. 适配器模式转换

**优点**: 平衡改动和兼容性
**缺点**: 可能性能不佳

## 📝 下一步行动

1. ✅ 分析完成 - 已识别所有关键差异
2. ⏳ 待决定 - 选择实施方案
3. ⏳ 待实施 - 代码修改
4. ⏳ 待测试 - 验证功能

## 💡 建议

基于当前分析，我的建议是：

1. **短期**: 继续使用 stable-diffusion-2.1，它已经验证可行
2. **中期**: 如果确实需要更好的模型，考虑其他 SD 架构的模型（如 SDXL）
3. **长期**: 如果必须使用 Z-Image-Turbo，采用**方案 B**（创建并行代码）

**关键决策点**:
- Z-Image-Turbo 是否支持 ControlNet？（需要进一步验证）
- Flow-matching 训练是否适用于 DepthCAD 的任务？
- 是否有足够的资源进行完整重构？
