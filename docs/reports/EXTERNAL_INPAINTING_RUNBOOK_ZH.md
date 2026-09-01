# ProPainter 和 RAD 运行记录

本记录针对远景 ToF/depth 补全对比，输入来自：

```text
output/far_pic_benchmark/<case>/external_inputs/export/frames
output/far_pic_benchmark/<case>/external_inputs/export/masks
```

其中 `frames` 是把米制 depth 按 `depth_meta.json` 归一化后的灰度 PNG，`masks` 是白色表示待补区域。外部方法输出 PNG 后需要再按 `depth_meta.json` 反归一化回米制 `.npy`，否则只能肉眼看，不能和当前 benchmark 对齐。

## 本地环境结论

### ProPainter

代码位置：

```text
/data/pre_student/cyx/ProPainter
```

入口：

```text
inference_propainter.py
```

README 要求：

```bash
conda create -n propainter python=3.8 -y
conda activate propainter
pip3 install -r requirements.txt
```

本地权重已存在：

```text
/data/pre_student/cyx/ProPainter/weights/ProPainter.pth
/data/pre_student/cyx/ProPainter/weights/raft-things.pth
/data/pre_student/cyx/ProPainter/weights/recurrent_flow_completion.pth
```

已测试可启动的现有环境：

```text
/home/lab507/anaconda3/envs/freeinpaint
```

`freeinpaint` 中 `inference_propainter.py --help` 可以正常运行。关键包版本：

```text
torch 2.0.1+cu117
torchvision 0.15.2+cu117
opencv 4.11.0
numpy 1.26.4
scipy 1.13.1
scikit-image 0.24.0
timm 0.6.13
```

注意：

```text
当前 shell 下 torch.cuda.is_available() = False
nvidia-smi 不能连接驱动
freeinpaint 缺 av 和 imageio_ffmpeg
```

ProPainter 可以读 split frames，不一定需要 `av`。但脚本最后会写 mp4，可能需要 `imageio_ffmpeg` 或系统 ffmpeg。系统 `ffmpeg` 存在。若最后写 mp4 失败，只要 `--save_frames` 已经保存 PNG 帧，就可以继续用解码脚本转回 depth `.npy`。

已实际遇到过的错误：

```text
TypeError: write() got an unexpected keyword argument 'fps'
```

原因是 `freeinpaint` 环境没有 `imageio_ffmpeg`，`imageio` 写 `.mp4` 时没有走 ffmpeg writer，误走了 tifffile writer。这个错误发生在 ProPainter 保存 PNG 帧之后，不代表补全帧失败。当前 wrapper 已处理这种情况：如果 ProPainter 返回非零退出码，但输出 PNG 帧齐全，并且命令带了 `--decode`，会继续解码。

不适合的环境：

```text
/home/lab507/anaconda3/envs/control
```

原因：

```text
ModuleNotFoundError: No module named 'einops'
```

### RAD

代码位置：

```text
/data/pre_student/GJ/RAD
```

入口：

```text
examples/unconditional_image_generation/inpaint.py
```

README 要求：

```text
python 3.8.16
pytorch 2.0.1
CUDA 11.8
pip install -e .[torch]
```

已测试可启动的现有环境：

```text
/home/lab507/anaconda3/envs/depthcad_zimage
```

运行 RAD 需要加：

```bash
PYTHONPATH=/data/pre_student/GJ/RAD/src
```

`depthcad_zimage` 中 `inpaint.py --help` 可以正常运行。关键包版本：

```text
torch 2.9.1+cu128
torchvision 0.24.1+cu128
opencv 4.13.0
numpy 2.2.6
diffusers 0.30.0.dev0
accelerate 1.12.0
datasets 2.19.2
transformers 4.57.6
peft 0.18.1
opencv_transforms present
```

本地 HuggingFace cache 里有 RAD README 使用的 ADM 基座：

```text
/home/lab507/.cache/huggingface/hub/models--xutongda--adm_ffhq_256x256
/home/lab507/.cache/huggingface/hub/models--xutongda--adm_lsun_bedroom_256x256
/home/lab507/.cache/huggingface/datasets/merkol___ffhq-256
```

但本地没有找到 RAD 官方 checkpoint。RAD README 指向 Google Drive checkpoint。DepthCAD 下有很多 `depthcad_pbrt_rad*` checkpoint，但结构是：

```text
checkpoint-xxxx/depthcad/diffusion_pytorch_model.safetensors
checkpoint-xxxx/inpaint_net/pytorch_model.bin
```

RAD 官方 `inpaint.py` 的 load hook 读取的是：

```text
checkpoint-xxxx/unet/diffusion_pytorch_model.safetensors
```

所以这些 DepthCAD checkpoint 不能直接作为 RAD 官方脚本的 `--resume_from_checkpoint`。

另外 RAD 代码里有 `mask.cuda()`，当前 shell 没有可用 GPU 时不能实际跑推理。

不适合的环境：

```text
/home/lab507/anaconda3/envs/control
```

原因：

```text
ModuleNotFoundError: No module named 'opencv_transforms'
```

## 已添加适配脚本

新增脚本：

```text
scripts/run_external_inpainting_far_pic.py
```

作用：

```text
1. 打印 ProPainter 运行命令
2. 可直接调用 ProPainter
3. 将 ProPainter 输出 PNG 反归一化并合并回米制 depth .npy
4. 准备 RAD 所需 val_data_path 目录
5. 打印 RAD 运行命令和 checkpoint 查找路径提示
```

脚本已通过：

```bash
python3 -m py_compile scripts/run_external_inpainting_far_pic.py
```

并已用旧 ProPainter PNG 结果测试了解码，输出 stack 形状为：

```text
(23, 240, 320), float32
```

## ProPainter 怎么跑

以 `bad_depth_mask_v1` 为例，先打印命令：

```bash
python3 scripts/run_external_inpainting_far_pic.py propainter-command \
  --case output/far_pic_benchmark/bad_depth_mask_v1
```

打印出的实际命令是：

```bash
cd /data/pre_student/cyx/ProPainter
MPLCONFIGDIR=/tmp/mpl_propainter /home/lab507/anaconda3/envs/freeinpaint/bin/python /data/pre_student/cyx/ProPainter/inference_propainter.py \
  --video /data/pre_student/GJ/DepthCAD/output/far_pic_benchmark/bad_depth_mask_v1/external_inputs/export/frames \
  --mask /data/pre_student/GJ/DepthCAD/output/far_pic_benchmark/bad_depth_mask_v1/external_inputs/export/masks \
  --output /data/pre_student/GJ/DepthCAD/output/far_pic_benchmark/bad_depth_mask_v1/propainter_run \
  --height 240 \
  --width 320 \
  --mask_dilation 0 \
  --save_frames
```

如果是在有 GPU 的 shell 中，建议直接用脚本跑并在结束后自动解码：

```bash
python3 scripts/run_external_inpainting_far_pic.py run-propainter \
  --case output/far_pic_benchmark/bad_depth_mask_v1 \
  --decode
```

如果 ProPainter 已经生成 PNG 帧，单独解码：

```bash
python3 scripts/run_external_inpainting_far_pic.py decode-propainter \
  --case output/far_pic_benchmark/bad_depth_mask_v1 \
  --output_dir output/far_pic_benchmark/bad_depth_mask_v1/propainter_run
```

这次 `bad_depth_mask_v1` 已经成功解码：

```text
output/far_pic_benchmark/bad_depth_mask_v1/propainter_run/restored_by_stem
output/far_pic_benchmark/bad_depth_mask_v1/propainter_run/restored_depth.npy
```

`restored_depth.npy` 形状：

```text
(23, 240, 320), float32
```

解码后会生成：

```text
output/far_pic_benchmark/bad_depth_mask_v1/propainter_run/restored_by_index/0000.npy
output/far_pic_benchmark/bad_depth_mask_v1/propainter_run/restored_by_stem/<stem>_propainter_restored.npy
output/far_pic_benchmark/bad_depth_mask_v1/propainter_run/restored_depth.npy
output/far_pic_benchmark/bad_depth_mask_v1/propainter_run/decode_summary.json
```

解码规则：

```text
gray / 255 * (depth_max - depth_min) + depth_min
```

只替换 mask 内像素，mask 外保留原始 corrupted depth，避免外部图像模型改动已观测深度。

## RAD 怎么准备和跑

RAD 的 `GivenDataset` 期望目录：

```text
val_data_path/
  thick/
    original/0000.png
    mask/0000.png
  box/
    original/0000.png
    mask/0000.png
  extreme/
    original/0000.png
    mask/0000.png
```

而 `inpaint.py` 里硬编码遍历：

```python
for mask in ['thick', 'box','extreme']:
```

所以我已经为 `bad_depth_mask_v1` 准备好目录：

```text
output/far_pic_benchmark/bad_depth_mask_v1/rad_val
```

检查结果：

```text
thick/original: 23 张
thick/mask:     23 张
图像尺寸:       256x256
mask 像素值:    0/255
```

重新准备命令：

```bash
python3 scripts/run_external_inpainting_far_pic.py prepare-rad \
  --case output/far_pic_benchmark/bad_depth_mask_v1
```

打印 RAD 命令：

```bash
python3 scripts/run_external_inpainting_far_pic.py rad-command \
  --case output/far_pic_benchmark/bad_depth_mask_v1 \
  --val_data_path output/far_pic_benchmark/bad_depth_mask_v1/rad_val \
  --resume_from_checkpoint checkpoint-300000
```

当前打印出的 RAD 命令是：

```bash
cd /data/pre_student/GJ/RAD
PYTHONPATH=/data/pre_student/GJ/RAD/src /home/lab507/anaconda3/envs/depthcad_zimage/bin/python /data/pre_student/GJ/RAD/examples/unconditional_image_generation/inpaint.py \
  --val_data_path /data/pre_student/GJ/DepthCAD/output/far_pic_benchmark/bad_depth_mask_v1/rad_val \
  --dataset_name merkol/ffhq-256 \
  --pretrained_model_name_or_path xutongda/adm_ffhq_256x256 \
  --resume_from_checkpoint checkpoint-300000 \
  --resolution 256 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --num_samples 23 \
  --ddpm_num_inference_steps 100 \
  --rank 16 \
  --exp_name ''
```

重要：RAD 的 `inpaint.py` 会忽略命令行 `--output_dir`，并把输出目录强制设为：

```text
ddpm-model-merkol/ffhq-256-256/_lora_rank_16
```

因此它会从这里找 checkpoint：

```text
/data/pre_student/GJ/RAD/ddpm-model-merkol/ffhq-256-256/_lora_rank_16/checkpoint-300000
```

要实际运行 RAD，有两条路：

```text
1. 下载 RAD 官方 Google Drive checkpoint，并放到上述目录。
2. 修改 /data/pre_student/GJ/RAD/examples/unconditional_image_generation/inpaint.py，让它接受绝对 checkpoint 路径。
```

在当前环境下即使 checkpoint 放对了，仍需要 GPU，因为代码里直接调用 `mask.cuda()`。

## 当前建议

短期先跑 ProPainter。它的权重和环境都已经在本机，输入也已经准备好。当前 shell 没有 GPU，所以不要在这个 shell 里直接启动完整推理，容易 CPU 跑很久。

RAD 适合作为后续对比，但需要先拿到官方 checkpoint，并在有 GPU 的环境里跑。不要把 DepthCAD 的 `depthcad_pbrt_rad*` checkpoint 当成 RAD 官方 checkpoint 直接塞进去，结构不匹配。
