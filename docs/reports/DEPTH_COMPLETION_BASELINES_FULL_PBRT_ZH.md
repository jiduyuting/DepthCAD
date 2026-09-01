# 全量 PBRT 深度补全对比实验

## 1. 目标与结论

本对比只选择原生任务为深度补全、且已有本地代码的工作：CompletionFormer、DMD³C、OMNI-DC、LDCM、LingBot-Depth-DC 和 DEPTHOR。所有方法读取同一份 PBRT 缓存、同一划分和同一孔洞，不再使用单目深度估计或普通图像修复方法充当主 baseline。

由于六个方法的官方训练方式和预训练规模差异很大，结果必须分成两张表：

1. **官方预训练零样本迁移**：OMNI-DC、LDCM、LingBot-Depth-DC；DMD³C 和 DEPTHOR 仅在获得匹配的官方权重后加入。
2. **PBRT 监督重训**：CompletionFormer、DMD³C、DEPTHOR。三者使用相同的 2160/240 划分，不能与零样本结果混表。

OMNI-DC、LDCM 和 LingBot-Depth 属于大型预训练模型，当前阶段不建议为了形式上的“公平”从头重训。尤其 OMNI-DC 官方训练资源远高于本地常规单卡实验。

## 2. 统一数据协议

- 缓存：`depth_completion_cache/depth_cache_full_pbrt_plane_r12_iq`
- 原始划分：`output/depth_flow_full_pbrt_iq/split.json`
- 规范化划分：`output/completionformer_full_pbrt/split.json`
- 样本数：训练 2160，验证 240
- 分辨率：256×256
- 稀疏深度：`depth_noisy`，并强制把 `hole_mask` 内设为 0
- GT：`gt_depth`，单位为米；该 ToF 深度参考可能包含 `0`/非有限/`<=0.1 m` 无效像素，不视为 dense GT。可视化直接显示原始数组，指标通过 `valid_mask` 排除无效像素。
- 引导图：`noisy_amplitude` 的三个频率幅度通道
- 引导归一化：每个样本、每个通道在 `valid_mask` 内取 99 分位作为尺度，裁剪到 `[0,1]`
- 相机内参：PBRT 缓存未提供真实内参，暂用 `f=max(H,W)`、主点为图像中心的占位内参

PBRT 没有真实 RGB。这里的三频 ToF 幅度只是在保持所有方法输入一致的前提下适配 RGB-D 接口，论文和表格中应写成 **3-frequency ToF amplitude guidance**，不能写成真实 RGB。

占位内参对依赖投影几何的 DMD³C、OMNI-DC 和 LingBot-Depth 可能造成额外影响。报告中必须注明这一限制；如果后续找到 PBRT 渲染相机的真实 `K`，应统一重跑这三种方法。

## 3. 统一评测协议

主指标是孔洞区域，辅助报告全图和观测区域：

- `hole`：`valid_mask & hole_mask`
- `observed`：`valid_mask & ~hole_mask`
- `global`：`valid_mask`

其中 `valid_mask = finite(gt_depth) & (gt_depth > 0.1) & (gt_depth < 9.9)`。因此 `hole`、`observed` 和 `global` 的误差都只在 GT 有效像素上计算；GT 无效区域不进入损失或指标。非零但存在结构化条带的参考值不会被自动修复，报告中应将其视为源深度参考的质量限制。
- 每个区域报告 MAE 和 RMSE，单位均为米

统一实现位于：

- 数据和指标：`depth_completion_baselines/common.py`
- Uniformat 导出：`scripts/export_pbrt_depth_completion_uniformat.py`
- 外部预测评分：`scripts/summarize_depth_completion_predictions.py`

OMNI-DC 和 DMD³C 官方保存的 16-bit PNG 使用 `depth * 256`，评分时必须设置 `--prediction_scale 0.00390625`。编号文件通过导出目录中的 `index.json` 恢复到 PBRT `sample_id`。

## 4. 方法与运行方式

### CompletionFormer

- 分组：PBRT 监督重训
- 环境：`/home/lab507/anaconda3/envs/cformer/bin/python`
- 状态：数据适配和模型构建已验证；尚未完成 72 epoch 训练
- 启动：

```bash
GPU=0 BATCH_SIZE=4 bash scripts/runs/run_completionformer_full_pbrt.sh
```

训练完成后按 `COMPLETIONFORMER_FULL_PBRT_BASELINE_ZH.md` 使用 `scripts/eval_completionformer_full_pbrt.py` 统一评测。

### OMNI-DC v1.1

- 分组：官方预训练零样本
- 推荐环境：优先尝试 `cformer`，其已有 Torch 和 Apex
- 必需权重：`modelv1.1_best_72epochs.pt`、`depth_anything_v2_vitl.pth`
- 当前状态：两个权重均未在本地发现
- 启动：

```bash
CHECKPOINT=/path/to/modelv1.1_best_72epochs.pt \
DAV2_CHECKPOINT=/path/to/depth_anything_v2_vitl.pth \
bash scripts/runs/run_omnidc_full_pbrt.sh
```

脚本自动导出 Uniformat、调用官方 `main.py`、定位编号 PNG，并生成统一 `summary.json`。

### LDCM

- 分组：官方预训练零样本
- 推荐环境：`lingbot-world`
- 额外复用：`llava` 环境中的 `utils3d`
- 当前状态：代码可导入；模型权重和 MoGe 权重未缓存
- 启动：

```bash
bash scripts/runs/run_ldcm_full_pbrt.sh
```

可用 `MODEL=/local/LDCM MOGE_MODEL=/local/moge` 改为完全离线加载。

### LingBot-Depth-DC

- 分组：官方预训练零样本
- 推荐环境：`depthcad`，已有 Torch 2.8 和 xFormers
- 默认模型：`robbyant/lingbot-depth-postrain-dc-vitl14`
- 当前状态：代码可导入；模型权重未缓存
- 启动：

```bash
bash scripts/runs/run_lingbot_full_pbrt.sh
```

可用 `MODEL=/local/lingbot-depth` 指定本地模型目录。

### DMD³C

- 分组：优先 PBRT 监督重训；获得匹配官方 checkpoint 后可额外做零样本
- 输入：三频幅度、稀疏深度和占位 `K`
- 已验证环境：`llava`，Python 3.10、Torch 2.1.2+cu121、CUDA 12.1
- 已安装：Hydra 1.3.2、OmegaConf 2.3.0、DA3 API 所需最小依赖
- 已编译：RTX 3090 `sm_86` 对应的 `BpOps` CUDA 扩展
- 当前阻塞：只剩 `result_ema.pth` 和 `DA3METRIC-LARGE` 权重下载
- 启动：

```bash
bash scripts/download_dmd3c_weights.sh
LIMIT=1 GPU=2 bash scripts/runs/run_dmd3c_full_pbrt.sh
```

脚本使用仓库现有 `datasets/uni.py`，不修改官方模型。训练若采用官方蒸馏入口，应单独记录教师模型、初始化权重、训练轮数和 GPU 数量。

### DEPTHOR

- 分组：PBRT 监督重训；官方 ZJU 权重可作为附加迁移实验
- 当前阻塞：缺少 DEPTHOR checkpoint、DAV2-small checkpoint 和编译后的 `BpOps`
- 适配：`scripts/eval_depthor_full_pbrt.py` 在运行时注入 DAV2 checkpoint 路径，绕过外部仓库中的作者机器硬编码，不改其源码
- 模型内部尺寸：按官方结构将 256×256 输入插值到 480×640，输出再插值回 256×256 统一评分
- 启动：

```bash
CHECKPOINT=/path/to/depthor_weights.pt \
DAV2_CHECKPOINT=/path/to/depth_anything_v2_vits.pth \
bash scripts/runs/run_depthor_full_pbrt.sh
```

## 5. 环境复用建议

| 方法 | 首选环境 | 当前可复用部分 | 仍需处理 |
|---|---|---|---|
| CompletionFormer | `cformer` | Torch 1.10.1、Apex、DCN | 无，等待训练 |
| OMNI-DC | `cformer` | Torch、Apex、timm | 两个 checkpoint，其他导入需实跑确认 |
| LDCM | `lingbot-world` | Python 3.10、Torch 2.6 | 复用 `llava/utils3d`，下载或指定本地权重 |
| LingBot-Depth | `depthcad` | Torch 2.8、xFormers | 下载或指定本地权重 |
| DMD³C | `llava` | Torch 2.1.2+cu121、Hydra、OmegaConf、BpOps、DA3 API | checkpoint、DA3 权重 |
| DEPTHOR | `py310` 候选 | Torch 2.6、timm | BpOps、DEPTHOR 权重、DAV2-small 权重 |

不建议直接安装 DMD³C 提供的完整 `environment.yml`，其依赖过大且容易破坏现有环境。优先在一个现代 Torch 环境中补最小依赖并编译 `BpOps`。DMD³C 与 DEPTHOR 的扩展应分别在目标环境中编译，不能假设跨 Torch/CUDA 版本复用 `.so`。

## 6. 推荐执行顺序

1. 启动 CompletionFormer 72 epoch 监督训练。
2. 准备 LingBot-Depth 和 LDCM 的 Hugging Face 权重，先做 2 样本 smoke test，再跑 240 张验证集。
3. 准备 OMNI-DC v1.1 与 DAV2-L 权重，完成零样本表。
4. 为 DMD³C/DEPTHOR 建立最小环境并编译各自 `BpOps`。
5. 先用官方 checkpoint 做迁移测试，再决定是否承担 PBRT 重训或蒸馏成本。
6. 获取真实 PBRT 相机内参后，重跑所有使用 `K` 的方法。

所有 runner 都支持 `LIMIT=2`，应先用以下形式做冒烟测试：

```bash
LIMIT=2 bash scripts/runs/run_lingbot_full_pbrt.sh
```

## 7. 结果表模板

### 官方预训练零样本

| 方法 | 引导输入 | Hole MAE↓ | Hole RMSE↓ | Global MAE↓ | Global RMSE↓ | Observed MAE↓ | Observed RMSE↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| OMNI-DC v1.1 | 3-freq amplitude + sparse depth | - | - | - | - | - | - |
| LDCM | 3-freq amplitude + sparse depth | - | - | - | - | - | - |
| LingBot-Depth-DC | 3-freq amplitude + sparse depth | - | - | - | - | - | - |

### PBRT 监督重训

| 方法 | 初始化 | Epoch | Hole MAE↓ | Hole RMSE↓ | Global MAE↓ | Global RMSE↓ | Observed MAE↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| CompletionFormer | 记录实际设置 | 72 | - | - | - | - | - |
| DMD³C | 记录教师与初始化 | - | - | - | - | - | - |
| DEPTHOR | 记录 DAV2 与初始化 | - | - | - | - | - | - |

最终报告还应记录参数量、单张推理时间、峰值显存和是否使用测试时增强，但这些效率数据不能替代统一深度误差指标。
