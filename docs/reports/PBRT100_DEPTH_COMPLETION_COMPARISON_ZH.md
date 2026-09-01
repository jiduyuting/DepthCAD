# PBRT100 深度补全主对比协议

## 目标

这组实验的主表口径是：研究者自选一个主模型，然后在同一份 seed123 100 张 PBRT/ToF holdout 上，对比 CompletionFormer、DMD3C、OMNI-DC、LDCM、LingBot-Depth、DEPTHOR、RGBD-Imaging 和 LFRD2。

不要把已有 `output/depth_completion_baselines/*/summary.json` 的 240 张 full-PBRT val 结果直接写进这张表。100 张表只接受 `samples == 100` 的结果。

## GT 与指标口径（重要）

当前 `gt_depth` 来自 `/data/pre_student/hcy/pbrt/gt_depth` 的 ToF 深度参考图，不是保证每个像素都有值的渲染器 dense depth。GT 面板直接显示原始 `gt_depth` 数组，不对 `0`、非有限值或范围外像素做灰色遮罩、插值或修复。可视化不再把 `hole` 边界叠加到任何深度图，而是使用独立的 `Hole mask` 面板；因此深度图中的线状结构若仍存在，属于源数组本身，不能归因于绘图轮廓或模型。

所有数字只在 GT 有效像素上统计：

- `valid_mask = finite(gt_depth) & (gt_depth > 0.1) & (gt_depth < 9.9)`；
- `hole = valid_mask & hole_mask`；
- `observed = valid_mask & ~hole_mask`；
- `global = valid_mask`。

因此，主表中的 `Hole MAE/RMSE` 不包含 GT 本身无效的像素。若后续换成真正的 dense PBRT 渲染深度，必须重建 cache 并重新生成所有预测后才能与当前表直接比较。

当前 cache 的 QA 统计（`gt_depth` 无效比例）为：seed123 holdout 100 张平均 `4.04%`、P90 `12.90%`、最大 `53.92%`；训练 cache 1000 张平均 `5.69%`、最大 `78.73%`。可视化保留这些原始值，因此仍可看到参考深度自身的无效区域或结构化条带；它们属于源数据质量限制，不能归因于模型。指标仍通过 `valid_mask` 排除这些位置。

## 固定数据

- Holdout cache: `depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq`
- Manifest: `output/full_pbrt_manifest_seed123_iq.json`
- Split JSON: `output/pbrt100_depth_completion/split.json`
- Split name: `test`
- 主指标：`hole` 区域 MAE/RMSE，单位米
- 辅助指标：`global` 和 `observed`

先生成 split：

```bash
python scripts/make_pbrt100_completion_split.py
```

## 主表方法

| 方法 | 主表角色 | 结果路径 |
|---|---|---|
| Selected model | 自选主模型 | 通过 `--selected name:path` 指定 |
| CompletionFormer | PBRT supervised/fine-tuned baseline | `output/pbrt100_depth_completion/completionformer/summary.json` |
| DMD3C | 官方迁移或 PBRT supervised，需注明 | `output/pbrt100_depth_completion/dmd3c/summary.json` |
| OMNI-DC | 官方 zero-shot | `output/pbrt100_depth_completion/omnidc_zero_shot/summary.json` |
| LDCM | 官方 zero-shot | `output/pbrt100_depth_completion/ldcm_zero_shot/summary.json` |
| LingBot-Depth | 官方 zero-shot | `output/pbrt100_depth_completion/lingbot_dc_zero_shot/summary.json` |
| DEPTHOR | 官方迁移或 PBRT supervised，需注明 | `output/pbrt100_depth_completion/depthor/summary.json` |
| RGBD-Imaging | 本项目适配 baseline | `output/pbrt100_depth_completion/rgbd_lfrd2/summary.json` |
| LFRD2 | 本项目适配/proxy baseline | `output/pbrt100_depth_completion/rgbd_lfrd2/summary.json` |

## 运行命令

### 全部模型一键启动

现在推荐直接使用总 runner：

```bash
bash scripts/runs/run_pbrt100_all.sh
```

它按下面顺序执行：

1. 从 `seed123` holdout cache 生成严格 100 张 `test` split；
2. 训练/续训 RGBD-Imaging、LFRD2、CompletionFormer、DEPTHOR 和 Ours-Flow；
3. 对训练完成或已有 checkpoint 的全部方法执行 PBRT100 推理；
4. 最后生成 `output/pbrt100_depth_completion/comparison/summary.{md,csv,json}`。

外部仓库或权重缺失时不会让其他方法全部中断，会在 `output/pbrt100_depth_completion/run.log` 中记录并在最终汇总中标记为 `missing`。因此可以先直接启动，之后只补权重再重复运行。

常用控制参数：

```bash
# 只跑 RGBD/LFRD2 训练和评估
RUN_FLOW=0 RUN_COMPLETIONFORMER=0 RUN_DMD3C=0 RUN_OMNI=0 \
RUN_LDCM=0 RUN_LINGBOT=0 RUN_DEPTHOR=0 bash scripts/runs/run_pbrt100_all.sh

# 重新训练 Ours-Flow、CompletionFormer、DEPTHOR（默认就是 1）
TRAIN_FLOW=1 TRAIN_COMPLETIONFORMER=1 TRAIN_DEPTHOR=1 \
bash scripts/runs/run_pbrt100_all.sh

# 指定已有 Ours-Flow / DEPTHOR 权重
FLOW_CKPT=/path/to/flow/best.pt \
DEPTHOR_CHECKPOINT=/path/to/depthor.pt bash scripts/runs/run_pbrt100_all.sh

# 只做已有 checkpoint 推理，不重新训练任何模型
TRAIN_FLOW=0 TRAIN_COMPLETIONFORMER=0 TRAIN_DEPTHOR=0 \
RUN_UNIFIED_TRAIN=0 bash scripts/runs/run_pbrt100_all.sh
```

训练相关的默认环境是：RGBD 使用 `depthcad`，LFRD2/Ours-Flow 使用 `SVDC`，CompletionFormer 使用 `cformer`，DEPTHOR 使用 `py310`；总 runner 会自动检测 CUDA，不可用时使用 CPU。完整训练建议在有 NVIDIA 驱动的机器上运行。

如果有多张 GPU，可以使用实际存在的并行入口（不是示例占位文件）：

```bash
FLOW_GPU=1 COMPLETIONFORMER_GPU=2 DEPTHOR_GPU=3 \
bash scripts/runs/run_parallel_train.sh
```

三个训练日志分别写入：

```text
output/parallel_train_logs/flow.log
output/parallel_train_logs/completionformer.log
output/parallel_train_logs/depthor.log
```

注意：`CUDA_VISIBLE_DEVICES=2` 后，进程内部可见的那张卡会重新编号为 `cuda:0`，所以并行脚本内部统一使用 `GPU=0`/`DEVICE=cuda:0`，不要改成 `cuda:2`。如果某个模型依赖的初始化 checkpoint 或编译扩展缺失，它会在自己的日志中失败，不会覆盖其他模型的日志。

### RGBD-Imaging / LFRD2 重新训练

报告中已有的 RGBD-Imaging 和 LFRD2 数字是旧 checkpoint 的结果；当前仓库没有把这些 checkpoint 放在 `output/` 中，因此如果要在统一 PBRT 数据协议下复现实验，需要重新训练。两者训练入口已经适配到统一 manifest：

- RGBD-Imaging：`/home/lab507/anaconda3/envs/depthcad/bin/python`
- LFRD2：`/home/lab507/anaconda3/envs/SVDC/bin/python`（需要 `mmcv`）
- 当前机器若 CUDA 不可用，runner 会自动使用 CPU；这只影响速度，不改变数据划分

一键训练两个模型，并在训练结束后评估：

```bash
bash scripts/runs/run_train_rgbd_lfrd2.sh
```

runner 默认使用 `output/full_pbrt_manifest_seed123.json`、200 epochs，并在发现 `output/*/last.pth` 时自动续训。常用覆盖方式：

```bash
MODELS=rgbd EPOCHS=200 DEVICE=cuda:0 bash scripts/runs/run_train_rgbd_lfrd2.sh
MODELS=lfrd2 RESUME=0 RUN_EVAL=0 bash scripts/runs/run_train_rgbd_lfrd2.sh
MANIFEST=output/full_pbrt_manifest_seed123_iq.json bash scripts/runs/run_train_rgbd_lfrd2.sh
```

注意：`full_pbrt_manifest_seed123.json` 对应当前统一训练的 `train/val/test = 8910/990/100` 划分；`PBRT100` 主表只把最后的 100 张 `test` 结果用于比较。

公共变量：

```bash
export CACHE_ROOT=/data/pre_student/GJ/DepthCAD/depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq
export SPLIT_JSON=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/split.json
export SPLIT=test
```

LDCM：

```bash
OUTPUT_DIR=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/ldcm_zero_shot \
bash scripts/runs/run_ldcm_full_pbrt.sh
```

LingBot-Depth：

```bash
OUTPUT_DIR=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/lingbot_dc_zero_shot \
bash scripts/runs/run_lingbot_full_pbrt.sh
```

OMNI-DC：

```bash
UNIFORMAT_DIR=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/uniformat \
SUMMARY_OUTPUT=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/omnidc_zero_shot/summary.json \
LOG_DIR=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/omnidc_runs \
bash scripts/runs/run_omnidc_full_pbrt.sh
```

DMD3C：

```bash
UNIFORMAT_DIR=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/uniformat \
RUN_NAME=DMD3C_PBRT100 \
SUMMARY_OUTPUT=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/dmd3c/summary.json \
bash scripts/runs/run_dmd3c_full_pbrt.sh
```

DEPTHOR：

```bash
CHECKPOINT=/path/to/depthor_weights.pt \
DAV2_CHECKPOINT=/path/to/depth_anything_v2_vits.pth \
OUTPUT_DIR=/data/pre_student/GJ/DepthCAD/output/pbrt100_depth_completion/depthor \
bash scripts/runs/run_depthor_full_pbrt.sh
```

CompletionFormer：

```bash
/home/lab507/anaconda3/envs/cformer/bin/python scripts/eval_completionformer_full_pbrt.py \
  --completionformer_root /data/pre_student/hcy/CompletionFormer \
  --checkpoint /path/to/completionformer_checkpoint.pt \
  --cache_root "$CACHE_ROOT" \
  --split_json "$SPLIT_JSON" \
  --output_dir output/pbrt100_depth_completion/completionformer \
  --save_predictions
```

RGBD-Imaging 和 LFRD2：

```bash
python scripts/eval_unified_baselines.py \
  --manifest output/full_pbrt_manifest_seed123_iq.json \
  --model both \
  --workers 0 \
  --output output/pbrt100_depth_completion/rgbd_lfrd2/summary.json
```

## 汇总

```bash
python scripts/analysis/summarize_pbrt100_depth_completion_comparison.py \
  --selected "MyModel:output/my_model/eval_pbrt100/summary.json"
```

输出：

- `output/pbrt100_depth_completion/comparison/summary.md`
- `output/pbrt100_depth_completion/comparison/summary.csv`
- `output/pbrt100_depth_completion/comparison/summary.json`

汇总脚本会标注 `missing` 和 `non-100`，只有 `status == ok` 的行可以进入正式 100 张主表。

## 当前已跑结果

当前已经在 100 张 seed123 holdout 上得到 Ours-Flow-FullPBRT、CompletionFormer、DMD3C、OMNI-DC、LDCM、LingBot-Depth、DEPTHOR、RGBD-Imaging 和 LFRD2 的结果。统计口径为上面的 `valid_mask` 规则，正式主表中的 Ours-Flow 使用 full-PBRT 2160/240 训练版本。

| 方法 | Samples | Hole MAE | Hole RMSE | Global MAE | Global RMSE | Observed MAE | Observed RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours-Flow-FullPBRT | 100 | 0.112789 | 0.361051 | 0.050285 | 0.335690 | 0.038666 | 0.330761 |
| RGBD-Imaging | 100 | 0.168457 | 0.383984 | 0.051021 | 0.266265 | 0.029191 | 0.238050 |
| CompletionFormer | 100 | 0.177091 | 0.439940 | 0.047639 | 0.261484 | 0.023575 | 0.212380 |
| DMD3C | 100 | 0.298957 | 5.086503 | 0.344158 | 8.221503 | 0.352560 | 8.680361 |
| OMNI-DC | 100 | 0.350077 | 0.907430 | 0.162994 | 0.678295 | 0.128217 | 0.626530 |
| LingBot-Depth | 100 | 0.362810 | 0.874699 | 0.110015 | 0.487602 | 0.063023 | 0.373801 |
| LDCM | 100 | 0.432190 | 1.126487 | 0.131672 | 0.663345 | 0.075808 | 0.534725 |
| DEPTHOR | 100 | 0.564886 | 0.822986 | 0.436446 | 0.673192 | 0.412570 | 0.641503 |
| LFRD2 | 100 | 0.693878 | 0.962978 | 0.207129 | 0.532064 | 0.116646 | 0.404146 |

结果文件：

- `output/depth_flow_full_pbrt_iq_endpoint_w2/eval_test_endpoint_best/summary.json`
- `output/pbrt100_depth_completion/completionformer/summary.json`
- `output/pbrt100_depth_completion/dmd3c/summary.json`
- `output/pbrt100_depth_completion/omnidc_zero_shot/summary.json`
- `output/pbrt100_depth_completion/ldcm_zero_shot/summary.json`
- `output/pbrt100_depth_completion/lingbot_dc_zero_shot/summary.json`
- `output/pbrt100_depth_completion/depthor/summary.json`
- `output/pbrt100_depth_completion/rgbd_lfrd2/summary.json`
- `output/pbrt100_depth_completion/comparison/summary.md`
