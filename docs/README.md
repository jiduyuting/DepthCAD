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

如果只运行 `scripts/flow/`，可以使用更小的依赖文件：

```bash
python -m pip install -r requirements-flow.txt
```

如果要在同一个环境中运行 DepthCAD 和 Flow，使用合并依赖文件：

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-depthcad-flow.txt
```

这套组合对应当前已验证的 `control` 环境。`xformers` 是全量 DepthCAD 启动器
显式启用的依赖；`bitsandbytes` 只有使用 `--use_8bit_adam` 时才需要安装。
如果目标机器的 CUDA 不是 12.1，只替换第一条命令中的 PyTorch wheel，其他版本
保持不变并先做导入 smoke test。

`requirements-flow.txt` 不锁定 PyTorch，因为 PyTorch 必须匹配目标机器的 CUDA。
例如当前 `control` 环境使用的是 PyTorch 2.4.0 + cu121、torchvision 0.19.0；可先
安装对应 CUDA wheel，再安装上面的文件：

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-flow.txt
```

如果 H200-4 已经准备好了 `control` 或 `SVDC` 环境，只需在对应环境中执行
`python -m pip install -r requirements-flow.txt`，不要重复安装另一套 PyTorch。

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

## 7. 从当前服务器迁移到新机器

### 7.1 最快的 Flow 迁移

如果目标是尽快继续 Flow 实验，不需要迁移完整 DepthCAD 训练输出。当前服务器
已有以下可复用资源：

```text
depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq/ 约 35G
output/full_pbrt_flow_lists_iq/                             约 1M
output/depth_flow_full_pbrt_iq_endpoint_w2/best.pt          约 96M
output/depth_flow_full_pbrt_iq_propagation_refine/best.pt   约 39M
output/flow_anchor_cache_epoch108/                          约 2.5G（可选）
```

可以用 `rsync` 复制，`SRC` 替换为当前服务器，`DST` 替换为新机器：

```bash
SRC=/data/pre_student/GJ/DepthCAD
DST=user@new-machine:/data/DepthCAD

rsync -a --info=progress2 "$SRC/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq" "$DST/depth_completion_cache/"
rsync -a "$SRC/output/full_pbrt_flow_lists_iq" "$DST/output/"
rsync -a "$SRC/output/depth_flow_full_pbrt_iq_endpoint_w2" "$DST/output/"
rsync -a "$SRC/output/depth_flow_full_pbrt_iq_propagation_refine" "$DST/output/"
```

Flow anchor cache 的文件名包含原始 cache 的绝对路径哈希。如果新机器路径不同，
不要直接复制 `flow_anchor_cache_epoch108`；在新机器用相同的 `best.pt` 重新运行
`scripts/flow/cache_flow_anchors.py` 即可。若坚持复用它，需要把 cache 挂载到和
原服务器完全相同的绝对路径。

### 7.2 迁移 DepthCAD 推理模型

只做推理或为已有 Flow cache 提供 DepthCAD 输出时，只需要复制：

```text
models/stable-diffusion-2-1/                                  约 4.9G
output/depthcad_sd21_full_pbrt/checkpoint-20000/depthcad/     约 1.4G
```

`depthcad/` 是 Diffusers ControlNet 模型目录，不能只复制其中一个权重文件。
完整的 SD/ControlNet pipeline 还需要 Hugging Face 的
`stable-diffusion-v1-5/stable-diffusion-inpainting`（旧版空洞填补流程使用）。

### 7.3 迁移全量 PBRT DepthCAD 训练

tmux 9 当前运行的是：

```text
accelerate launch --num_processes 1 train_pbrt.py
--pretrained_model_name_or_path /data/pre_student/GJ/DepthCAD/models/stable-diffusion-2-1
--dataset_name /data/pre_student/GJ/DepthCAD/pbrt_dataset
--dataset_config sd21_full_pbrt_train
--resolution 256 --train_batch_size 1 --gradient_accumulation_steps 8
--num_train_epochs 500 --checkpointing_steps 5000 --mixed_precision fp16
```

它当前约为 `495018/3341500` steps，输出目录已达到约 400G，其中绝大部分是历次
checkpoint 的 optimizer 状态。不要整体复制这个目录。若要在新机器续训，只复制
一个完整 checkpoint，例如：

```bash
rsync -a --info=progress2 \
  "$SRC/output/depthcad_sd21_full_pbrt/checkpoint-20000" \
  "$DST/output/depthcad_sd21_full_pbrt/"
```

完整 checkpoint 必须包含 `depthcad/`、`optimizer.bin`、`scheduler.bin`、
`scaler.pt` 和 `random_states_0.pkl`。新机器准备好数据后，用：

```bash
MODEL_DIR=/data/DepthCAD/models/stable-diffusion-2-1 \
DATASET_DIR=/data/DepthCAD/pbrt_dataset \
OUTPUT_DIR=/data/DepthCAD/output/depthcad_sd21_full_pbrt \
RESUME=1 GPU=0 \
bash scripts/runs/train_depthcad_sd21_full.sh
```

注意：tmux 9 的训练进度条已经显示 `loss=nan`。检查结果表明
`checkpoint-20000` 的 340 个权重张量全部有限，而从 `checkpoint-25000` 开始
已经出现非有限值，最新 `checkpoint-495000` 也已损坏。不要迁移最新 checkpoint
作为可靠模型；应以 `checkpoint-20000` 为安全基线，降低学习率并先做小规模 smoke
test，或者先定位 NaN 的数据/数值原因。

当前 SD2.1 训练目录中的 `ideal_IQ_sd21_full_pbrt_train` 和
`noise_IQ_sd21_full_pbrt_train` 是指向 `pbrt_dataset/data_256/{ideal_IQ,noise_IQ}`
的软链接。迁移这些目录时必须保留软链接及其目标，或者把目标数据实际复制到新
机器；另外还要复制 `confidence_sd21_full_pbrt_train`。只复制软链接目录本身会
得到断链数据集。

### 7.4 环境版本

当前服务器实际使用的是两个环境。Flow 代码本身不绑定某个环境；`control` 环境可以正常导入和运行 Flow。为了复现实验，仍应按启动器的默认值选择环境：完整 PBRT/SOTA Flow 多数使用 `SVDC`，real raw9 和部分微调脚本使用 `control`。

| 用途 | 关键版本 |
| --- | --- |
| Flow（完整 PBRT 默认 `SVDC`；`control` 也兼容） | PyTorch 1.12.0+cu113、torchvision 0.13.0、NumPy 1.20.3、OpenCV 4.6、matplotlib 3.6、SciPy 1.9、PyWavelets 1.4 |
| DepthCAD（`control`） | PyTorch 2.4.0+cu121、torchvision 0.19.0、diffusers 0.31.0、transformers 4.44.2、accelerate 0.34.0、datasets 2.21.0、xformers 0.0.27.post2、bitsandbytes 0.43.3 |

Flow 训练主要依赖 PyTorch、NumPy、OpenCV 和 matplotlib；DepthCAD 全量训练还
必须使用 `datasets<4`、xformers 和与显卡驱动匹配的 CUDA 版 PyTorch。迁移时建议
分别建立两个虚拟环境，不要把两个环境的 PyTorch/CUDA 版本混装。

如果要在 `control` 环境运行 Flow，可显式覆盖启动器的解释器，例如：

```bash
PYTHON_BIN=/home/lab507/anaconda3/envs/control/bin/python \
  bash scripts/runs/flow/run_real_raw9_flow_infer_split_added_ns.sh
```

## 8. 传输到 H200-4

H200-4 的 SSH 参数为：

```text
HostName 47.101.174.157
Port 31343
User root
```

先在本机 `~/.ssh/config` 中加入一个别名，并确认远端已经安装本机公钥（或准备好
远端接受的私钥）：

```sshconfig
Host H200-4
    HostName 47.101.174.157
    Port 31343
    User root
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
```

然后测试：

```bash
ssh H200-4 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

认证成功后，从当前服务器优先传 Flow 资源（约 35G 缓存加少量权重）：

```bash
SRC=/data/pre_student/GJ/DepthCAD
DEST=H200-4:/data/DepthCAD

ssh H200-4 'mkdir -p /data/DepthCAD/depth_completion_cache /data/DepthCAD/output'
rsync -a --info=progress2 --partial --append-verify \
  "$SRC/depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq" \
  "$DEST/depth_completion_cache/"
rsync -a --partial --append-verify "$SRC/output/full_pbrt_flow_lists_iq" "$DEST/output/"
rsync -a --partial --append-verify "$SRC/output/depth_flow_full_pbrt_iq_endpoint_w2" "$DEST/output/"
rsync -a --partial --append-verify "$SRC/output/depth_flow_full_pbrt_iq_propagation_refine" "$DEST/output/"
```

如果要迁移 DepthCAD 推理模型，再复制 SD2.1 基础模型和安全的
`checkpoint-20000/depthcad`。完整 PBRT 数据集约 167G，训练 checkpoint 输出约
400G，不建议一次性整体传输；应按实验需要分批复制。断线后重复执行上述
`rsync` 命令即可续传。
