  # accelerate launch train.py \
  #   --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  #   --output_dir="output/depthcad_with_grad_loss" \
  #   --dataset_name="pbrt_dataset" \
  #   --gradient_loss_weight=0.1 \
  #   --mse_loss_weight=1.0 \
  #   --mixed_precision="fp16" \
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
  #   --checkpointing_steps 5000

export MODEL_DIR="stabilityai/stable-diffusion-2-1"
export OUTPUT_DIR="output/depthcad_with_grad_loss"
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="pbrt_dataset" \
 --mixed_precision="fp16" \
  --resolution=512 \
  --gradient_loss_weight=0.1 \
  --mse_loss_weight=1.0 \
  --learning_rate=1e-4 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --num_train_epochs=500 \
  --lr_scheduler="cosine" \
  --checkpointing_steps 5000 \
  --resume_from_checkpoint latest