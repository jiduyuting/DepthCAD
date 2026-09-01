# CompletionFormer 全量 PBRT 对比实验

## 对比目标

将 `/data/pre_student/hcy/CompletionFormer` 作为深度图补全基线，使用 DepthCAD 当前已经生成的全量 PBRT 缓存，不重新生成孔洞，也不改变训练/验证样本。

主实验固定为：

- 数据：`depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq`
- 划分：2160 张训练、240 张验证，seed 123，与 `output/depth_flow_full_pbrt_iq/split.json` 完全一致
- 分辨率：256×256，不做缩放或裁剪
- 深度输入：`depth_noisy`，并强制将 `hole_mask` 内像素设为 0
- 引导输入：`noisy_amplitude` 的三个频率幅度通道，逐样本、逐通道用有效区域 99 分位归一化到 `[0, 1]`，再使用 ImageNet mean/std 标准化
- GT：`gt_depth`，单位保持为米；其中 `0`/非有限/`<=0.1 m` 是无效参考像素，不作为 dense GT 使用，评测时由 `valid_mask` 排除。
- 增强：训练阶段仅做成对水平翻转，验证阶段无增强
- CompletionFormer 配置：原生 RGB-D 主干、6 次传播、L1+L2、72 epoch、seed 123

## 为什么不直接使用现有 `src/data/pbrt.py`

CompletionFormer 本地旧适配器硬编码到了 `/data/pre_student/hcy/datasets/pbrt`，使用 240×320 数据和另一套 train/test 文本；同时幅度输入没有按主干预期做 ImageNet 标准化。因此它不能保证与当前 DepthCAD 全量 PBRT 实验使用同一批样本、同一孔洞和同一评测口径。

新适配器位于 `integrations/completionformer/pbrtfull.py`，类名为 `PBRTFull`，不会覆盖旧的 `PBRT` 数据集。

## 安装适配器

CompletionFormer 位于当前仓库之外，因此先执行一次：

```bash
cp integrations/completionformer/pbrtfull.py \
  /data/pre_student/hcy/CompletionFormer/src/data/pbrtfull.py
```

## 训练

本机已验证可用的环境是 `/home/lab507/anaconda3/envs/cformer/bin/python`，对应 Python 3.8、Torch 1.10.1 和 CUDA 11.3。脚本会自动加入已有的 `DCN.cpython-38-*.so` 路径，并使用 `integrations/completionformer/compat` 提供模型实际需要的最小 MMCV/MMSeg checkpoint 接口。不要直接用默认 Python 3.9，否则会报 `ModuleNotFoundError: DCN`。

```bash
GPU=0 BATCH_SIZE=4 bash run_completionformer_full_pbrt.sh
```

显存不足时将 `BATCH_SIZE` 改为 2。若环境位置发生变化，可通过 `PYTHON_BIN=/path/to/python` 覆盖。脚本会先生成并校验 `output/completionformer_full_pbrt/split.json`，再启动训练。训练日志和每轮网络权重写入 `output/completionformer_full_pbrt/train_logs/`；默认不额外保存 optimizer/scheduler，以控制磁盘占用。

## 统一评测

找到训练目录中的最终 checkpoint 后执行：

```bash
PYTHONPATH=integrations/completionformer/compat:/data/pre_student/hcy/CompletionFormer/src/model/deformconv/build/lib.linux-x86_64-3.8 \
/home/lab507/anaconda3/envs/cformer/bin/python eval_completionformer_full_pbrt.py \
  --completionformer_root /data/pre_student/hcy/CompletionFormer \
  --checkpoint /path/to/model_00072.pt \
  --device cuda:0 \
  --save_predictions
```

评测输出为 `output/completionformer_full_pbrt/eval/summary.json`，包括：

- `global`：全部 GT 有效像素
- `hole`：人工孔洞区域，是深度补全主指标
- `observed`：非孔洞有效区域，用于检查模型是否破坏已有深度
- 每个区域统一报告 MAE（米）和 RMSE（米）

论文表格建议以 `hole.mae_m` 和 `hole.rmse_m` 为主，同时附上 `global` 与 `observed`。不要直接引用 CompletionFormer 自带的全图指标代替孔洞指标。

## 建议消融

主表只保留 `3-frequency amplitude + noisy depth`，因为 CompletionFormer 原生接口只接受 3 通道引导图和 1 通道深度。若需要补充消融，可增加：

1. `Depth only`：将引导图置零，判断提升是否来自 CompletionFormer 结构本身。
2. `preserve_input`：仅作为附录实验。当前观测深度包含噪声，强制保留输入可能降低全局效果，不建议作为主结果。
3. 官方预训练权重微调：必须与从头训练分开报告，避免混淆预训练收益与结构收益。

不建议直接把 6 通道 IQ 压缩到伪 RGB 后称为“同输入”比较；这会引入额外通道映射设计，偏离 CompletionFormer 原始结构。
