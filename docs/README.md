# DepthCAD 运行指南

这份文档说明如何把仓库复制到另一台机器后，准备未上传的资源并运行模型。
命令默认从仓库根目录执行：

```bash
cd /path/to/DepthCAD
```

## 1. 仓库中没有的文件

GitHub 中只保存代码、配置、少量 manifest 和实验说明。下面这些内容被
`.gitignore` 排除，需要单独下载、复制或重新生成：

| 内容 | 典型位置 | 用途 |
| --- | --- | --- |
| PBRT/FLAT 原始数据 | `pbrt_dataset/data/`、`flat_dataset/data/` | 训练和评估输入 |
| 真实 raw9 与深度 | `data/`、`raw/`、`depth/` 或自定义目录 | 真实场景推理/微调 |
| 深度缓存 | `depth_completion_cache/` 下的 `.npz` | Flow 训练和评估 |
| 数据划分与运行 manifest | `output/` 下的 `.json`、`.txt` | 固定 train/val/test 划分 |
| Flow 权重 | `output/**/best.pt` 或 `last.pt` | Flow 推理、评估、refine |
| 原始 DepthCAD 权重 | Diffusers 格式目录，如 `checkpoint-*/depthcad/` | SD/ControlNet 路线 |
| Stable Diffusion 基础模型 | Hugging Face 本地缓存或自定义目录 | 原始 DepthCAD 训练/推理 |
| 外部 baseline 仓库和权重 | 仓库外目录 | CompletionFormer、DEPTHOR 等对比实验 |

权重和输出文件可能很大，因此没有提交到 Git。不要把个人数据、访问令牌或
本地绝对路径直接提交到仓库。

## 2. 安装环境

建议使用 Python 3.9 或更新版本，并安装与显卡驱动匹配的 PyTorch：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Flow 真实场景可视化和缓存预处理还会用到绘图库；如果当前环境没有这些包：

```bash
python -m pip install matplotlib tqdm
```

确认导入和 CUDA 状态：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

CPU 可以运行命令行检查和小规模测试，但完整训练通常应使用 CUDA。运行脚本
会把 `scripts/`、`scripts/flow/` 和仓库根目录加入 Python 搜索路径，因此应从
仓库根目录启动，或直接使用文档中的路径。

## 3. 推荐路线：Depth-domain Flow

当前仓库中最完整的实验路线位于：

- Python：`scripts/flow/`
- Shell 启动器：`scripts/runs/flow/`
- 通用缓存数据集和 backbone：`scripts/`

### 3.1 准备缓存

Flow 不直接读取原始 IQ 文件，而是读取每个样本一个 `.npz` 的深度缓存。每个
缓存至少应包含：

```text
depth_noisy, gt_depth, hole_mask, valid_mask, confidence
```

如果使用 `noisy_amp`、`noisy_iq` 或 `noisy_iq_amp`，还必须包含：

```text
noisy_amplitude, noisy_amplitude_mean, noisy_iq
```

最省事的方式是向项目维护者索取已经生成的缓存目录，例如：

```text
depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq/
  <scene>/<view>/<sample>.npz
```

如果只有 PBRT IQ 和 GT 深度，可以使用旧缓存生成器重新生成。它需要原始
`ideal_IQ`、`noise_IQ`、置信度、GT 深度，以及一个 DepthCAD checkpoint：

```bash
python scripts/apply_kinect_holes_and_eval.py \
  --ideal_iq_dir /data/pbrt/ideal_IQ \
  --noise_iq_dir /data/pbrt/noise_IQ \
  --gt_depth_dir /data/pbrt/gt_depth \
  --checkpoint_path /models/depthcad/checkpoint-15000/depthcad \
  --output_dir output/cache_generation \
  --save_depth_completion_cache \
  --depth_cache_save_iq \
  --depth_cache_dir depth_completion_cache/my_flow_cache
```

上面的目录只是示例，必须替换成当前机器上的真实路径。生成缓存后，先检查
文件数量和关键字段：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
root = Path("depth_completion_cache/my_flow_cache")
path = next(root.rglob("*.npz"))
with np.load(path) as item:
    print(path)
    print(sorted(item.files))
PY
```

### 3.2 从头训练基础 Flow

不要求固定划分时，可以让脚本按 `--val_ratio` 自动划分：

```bash
python scripts/flow/train_depth_flow_restoration.py \
  --cache_dir depth_completion_cache/my_flow_cache \
  --output_dir output/flow_baseline \
  --device cuda \
  --input_mode noisy_iq_amp \
  --anchor_mode noisy_ns \
  --backbone transformer_bottleneck \
  --epochs 120 \
  --batch_size 8 \
  --amp
```

如果缓存没有 `noisy_iq`，把 `--input_mode` 改成 `noisy` 或 `noisy_amp`。
训练结束后，通常使用 `output/flow_baseline/best.pt`。

如果你有包含 `train`、`val`、`test` 样本列表的 PBRT manifest，可以先生成固定
划分文件：

```bash
python scripts/flow/make_full_pbrt_flow_lists.py \
  --manifest /data/manifests/full_pbrt_manifest.json \
  --output_dir output/flow_lists
```

也可以使用可复现实验启动器：

```bash
bash scripts/runs/flow/run_flow_sota_experiments.sh
```

启动器中的默认数据和输出路径来自原实验机器；迁移到其他机器时，优先通过
环境变量覆盖 `CACHE_DIR`、`TRAIN_LIST`、`VAL_LIST`、`TEST_LIST`、`RUN_ROOT`
和 `PYTHON_BIN`，不要依赖脚本中的 `/data/pre_student/...` 默认值。

### 3.3 评估和推理

评估缓存中的验证集：

```bash
python scripts/flow/eval_depth_flow_restoration.py \
  --checkpoint output/flow_baseline/best.pt \
  --cache_dir depth_completion_cache/my_flow_cache \
  --split val \
  --output_dir output/flow_baseline/eval_val \
  --device cuda \
  --sampling_mode endpoint
```

真实深度图推理需要一批 `.npy` 深度文件：

```bash
python scripts/flow/infer_real_depth_flow.py \
  --input_dir /data/real/depth \
  --checkpoint output/flow_baseline/best.pt \
  --output_dir output/real_depth_flow \
  --device cuda
```

真实 raw9 推理需要 raw9 和 GT/参考深度按样本名配对。raw9 通常是
`(9,H,W)` 的 `.npy`：

```bash
python scripts/flow/infer_real_raw9_flow.py \
  --raw_dir /data/real/raw9 \
  --depth_dir /data/real/depth \
  --checkpoint output/flow_baseline/best.pt \
  --output_dir output/real_raw9_flow \
  --amplitude_mode iq6 \
  --device cuda
```

### 3.4 Propagation-refine（可选第二阶段）

第二阶段需要一个已经训练好的 Flow checkpoint。先缓存 Flow anchor：

```bash
python scripts/flow/cache_flow_anchors.py \
  --cache_dir depth_completion_cache/my_flow_cache \
  --pretrained_checkpoint output/flow_baseline/best.pt \
  --output_dir output/flow_anchor_cache \
  --train_list output/flow_lists/train.txt \
  --val_list output/flow_lists/val.txt \
  --device cuda
```

然后训练和评估 refinement：

```bash
python scripts/flow/train_depth_flow_propagation_refine.py \
  --cache_dir depth_completion_cache/my_flow_cache \
  --pretrained_checkpoint output/flow_baseline/best.pt \
  --anchor_cache_dir output/flow_anchor_cache \
  --output_dir output/flow_propagation_refine \
  --device cuda

python scripts/flow/eval_depth_flow_propagation_refine.py \
  --checkpoint output/flow_propagation_refine/best.pt \
  --cache_dir depth_completion_cache/my_flow_cache \
  --anchor_cache_dir output/flow_anchor_cache \
  --split val \
  --output_dir output/flow_propagation_refine/eval_val \
  --device cuda
```

## 4. 原始 DepthCAD SD/ControlNet 路线

这条路线对应 `scripts/train_pbrt.py`、`scripts/inference_pbrt.py` 和
`scripts/eval_pbrt.py`，需要额外准备：

1. Stable Diffusion 2.1 基础模型，例如 Hugging Face 的
   `stabilityai/stable-diffusion-2-1`，或一个已经下载好的本地目录。
2. DepthCAD ControlNet checkpoint，目录中应包含 Diffusers 模型配置和权重。
3. FLAT 或 PBRT 数据集，以及对应的 `train.jsonl`、`train.txt`、`test.txt`。

示例推理命令：

```bash
python scripts/inference_pbrt.py \
  --pretrained_model_name_or_path /models/stable-diffusion-2-1 \
  --depthcad_path /models/depthcad/checkpoint-15000/depthcad \
  --noise_IQ_file /data/pbrt/noise_IQ/sample.npy \
  --noise_depth_file /data/pbrt/noise_depth/sample.npy \
  --out_file output/depthcad_prediction.npy
```

如果使用 Hugging Face 模型名而不是本地目录，首次运行会下载模型；服务器
没有外网时，应先在有网络的机器下载后再复制模型缓存。

## 5. 外部 baseline

`scripts/runs/` 中部分总控脚本会尝试调用 CompletionFormer、DMD3C、OMNI-DC、
LDCM、LingBot-Depth 或 DEPTHOR。这些项目的源码、checkpoint、Python 环境和
第三方模型都不在本仓库中。只想运行 DepthCAD/Flow 时不需要安装它们；做完整
对比时请根据 `docs/reports/` 中对应报告单独准备，并通过环境变量指定外部仓库
和权重路径。

## 6. 迁移检查清单

在新机器上运行前，至少确认：

```text
[ ] requirements.txt 已安装，PyTorch 与 CUDA 可用
[ ] 原始数据或已生成的 .npz cache 已放到本地
[ ] Flow checkpoint 或 DepthCAD checkpoint 已复制到本地
[ ] 所有命令中的 /data/pre_student/... 路径已替换
[ ] train/val/test 划分文件存在，或明确使用 --val_ratio
[ ] 输出目录有足够空间（output/ 和 depth_completion_cache/ 不入 Git）
```

先运行各入口的 `--help`，再用少量样本做 smoke test，最后再启动完整训练。
