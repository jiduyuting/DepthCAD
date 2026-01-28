# export MODEL_DIR="stabilityai/stable-diffusion-2-1"
# export OUTPUT_DIR="output/depthcad"
# export HF_ENDPOINT=https://hf-mirror.com

# accelerate launch train.py \
#  --pretrained_model_name_or_path=$MODEL_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --dataset_name="pbrt_dataset" \
#  --mixed_precision="fp16" \
#   --resolution=512 \
#   --learning_rate=1e-4 \
#   --train_batch_size=16 \
#   --gradient_accumulation_steps=4 \
#   --gradient_checkpointing \
#   --use_8bit_adam \
#   --enable_xformers_memory_efficient_attention \
#   --set_grads_to_none \
#   --num_train_epochs=500 \
#   --lr_scheduler="cosine" \
#   --checkpointing_steps 5000 \
#   --resume_from_checkpoint latest


# python inference.py \
#     --pretrained_model_name_or_path "$MODEL_DIR" \
#     --depthcad_path "/data/pre_student/GJ/DepthCAD/output/depthcad/checkpoint-80000/depthcad" \
#     --noise_IQ_file "/home/lab507/Documents/JishenLin/GLRUN/FLAT/noise/1499392477071669" \
#     --noise_depth_file "/data/pre_student/hcy/ControlNet/data/noise_depth/1499392477071669.npy" \
#     --out_file "/data/pre_student/GJ/DepthCAD/flat_dataset/data"

# python eval.py \
#     --test_list_path "/data/pre_student/GJ/DepthCAD/test.txt" \
#     --out_dir "/data/pre_student/GJ/DepthCAD/out" \
#     --pred_dir "/data/pre_student/GJ/DepthCAD/input"


