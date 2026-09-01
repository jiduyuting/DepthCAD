# PBRT100 模型对比排名

## 评测口径

- 评测集：`depth_completion_cache/depth_cache_0514_n100_plane_r12_seed123_iq`
- 样本数：100
- 主指标：`valid_mask & hole_mask` 上的 Hole MAE，单位米，越低越好
- 辅助指标：Global/Observed MAE 和 RMSE，单位米
- 主表 Flow 结果使用 full-PBRT 监督训练版本：
  `output/depth_flow_full_pbrt_iq_endpoint_w2/eval_test_endpoint_best/summary.json`
- 该 Flow checkpoint 训练划分为 `2160` train / `240` val：
  `output/depth_flow_full_pbrt_iq_endpoint_w2/summary.json`

## 已完成模型排名

| Rank | Method | Train setting | Samples | Hole MAE↓ | Hole RMSE↓ | Global MAE↓ | Global RMSE↓ | Observed MAE↓ | 结论 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Ours-Flow-FullPBRT | full-PBRT supervised, 2160/240 | 100 | **0.112789** | **0.361051** | 0.050285 | 0.335690 | 0.038666 | 使用 full-PBRT 训练口径后仍是 Hole MAE 最优 |
| 2 | RGBD-Imaging | adapted baseline | 100 | 0.168457 | 0.383984 | 0.051021 | 0.266265 | 0.029191 | 第二名，Hole MAE 低于 CompletionFormer |
| 3 | CompletionFormer | supervised/fine-tuned baseline | 100 | 0.177091 | 0.439940 | **0.047639** | **0.261484** | **0.023575** | observed/global 区域更稳，但 hole 略弱 |
| 4 | DMD3C | zero-shot or PBRT-trained; record exact setting | 100 | 0.298957 | 5.086503 | 0.344158 | 8.221503 | 0.352560 | Hole MAE 排第四，但 RMSE 存在明显离群大误差 |
| 5 | OMNI-DC | official zero-shot | 100 | 0.350077 | 0.907430 | 0.162994 | 0.678295 | 0.128217 | 官方 full metric 较好，但统一 hole 口径弱于 DMD3C |
| 6 | LingBot-Depth | official zero-shot | 100 | 0.362810 | 0.874699 | 0.110015 | 0.487602 | 0.063023 | zero-shot 可用，但明显弱于监督/适配模型 |
| 7 | LDCM | official zero-shot | 100 | 0.432190 | 1.126487 | 0.131672 | 0.663345 | 0.075808 | zero-shot 表现偏弱 |
| 8 | DEPTHOR | PBRT-trained checkpoint | 100 | 0.564886 | 0.822986 | 0.436446 | 0.673192 | 0.412570 | 已完成 100 张评测，但当前 checkpoint 表现偏弱 |
| 9 | LFRD2 | adapted/proxy baseline | 100 | 0.693878 | 0.962978 | 0.207129 | 0.532064 | 0.116646 | 当前适配/proxy 口径下最弱 |

## 相对优势

以 Hole MAE 为准，full-PBRT Flow 相比各 baseline：

- 比 RGBD-Imaging 低约 33.0%。
- 比 CompletionFormer 低约 36.3%。
- 比 DMD3C 低约 62.3%。
- 比 OMNI-DC 低约 67.8%。
- 比 LingBot-Depth 低约 68.9%。
- 比 LDCM 低约 73.9%。
- 比 DEPTHOR 低约 80.0%。
- 比 LFRD2 低约 83.7%。

需要注意：CompletionFormer 的 Observed MAE 为 `0.023575`，低于 full-PBRT Flow 的 `0.038666`；但补洞任务主指标是 Hole MAE，full-PBRT Flow 在 hole 区域仍然最优。

## Ablation 说明

此前的 `Ours-Flow` 最优数值来自 n1000 训练版本：

`output/depth_flow_restoration_noisy_amp_ns_n1000_endpoint_w2_trans_b32_l2_p2/eval_seed123_endpoint/summary.json`

该模型在 PBRT100 上的 Hole MAE 为 `0.104364`，优于 full-PBRT 版本，但训练集不是统一 full-PBRT 2160/240 划分。因此它应作为 ablation 或历史最优补充，不作为正式主对比表的主结果。

## 完成状态

当前主对比方法均已有 100 张结果。OMNI-DC 的官方日志指标为全图 valid 口径；正式表采用保存的 100 个 indexed PNG 重新按 PBRT100 的 hole/global/observed 统一口径评分。
