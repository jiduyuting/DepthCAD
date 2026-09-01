
# import json
# from pathlib import Path

# summary = Path("/data/pre_student/GJ/DepthCAD/kinect_evaluation/clean_eval_0513_n30/ mae_results_summary.json")
# data = json.loads(summary.read_text())

# print("num_samples:", data["num_samples"])
# for r in data["per_sample_results"][:5]:
#     print("\n", r["sample_name"])
#     for name in ["noisy", "depthcad", "sdinpaint", "full", "depthfill"]:
#         print(
#             name,
#             "mae=", r[f"mae_{name}"],
#             "expected=", r[f"mae_{name}_expected_from_regions"],
#             "delta=", r[f"mae_{name}_consistency_delta"],
#         )
import torch
import mmcv

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("mmcv:", mmcv.__version__)

from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dFunction
print("MMCV deform conv OK")