#!/bin/bash

# ============================================================================
# HYPIR-Enhanced DepthCAD Inference Script
# ============================================================================
#
# This script runs batch inference using a model trained with train_hypir.py.
# Key features:
# - HYPIR LoRA weights are pre-merged into UNet during training
# - Uses simple 2-channel conditioning (noise + confidence)
# - Compatible with models trained using train_hypir.py
#
# ============================================================================

# ===== Configuration =====
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3  # train_hypir.py uses device '1'

# Model paths
MODEL_DIR="stabilityai/stable-diffusion-2-1"
DEPTHCAD_PATH="output/depthcad_pbrt_hypir/checkpoint-30000/depthcad"

# Data paths
TEST_LIST_PATH="pbrt_dataset/test.txt"
NOISE_IQ_DIR="/data/pre_student/hcy/pbrt/noise"
NOISE_DEPTH_DIR="/data/pre_student/hcy/pbrt/noise_depth"
OUT_DIR="pbrt/data_hypir_30000"

# Dataset configuration
DATASET_TYPE="pbrt"
TARGET_HEIGHT=240
TARGET_WIDTH=320

# Inference parameters
NUM_INFERENCE_STEPS=20
GUIDANCE_SCALE=1.0
SEED=42

# Create output directory
mkdir -p "$OUT_DIR"

# ===== Count total samples =====
total_samples=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    line=$(echo "$line" | xargs)
    [ -z "$line" ] && continue

    # Count slashes to determine format
    slash_count=$(echo "$line" | tr -cd '/' | wc -c)
    if [ "$slash_count" -eq 1 ]; then
        # Directory format: count files in directory
        category_dir="$NOISE_IQ_DIR/$line"
        if [ -d "$category_dir" ]; then
            file_count=$(find "$category_dir" -maxdepth 1 -type f -name "*.npy" | wc -l)
            total_samples=$((total_samples + file_count))
        fi
    else
        # Single file format
        total_samples=$((total_samples + 1))
    fi
done < "$TEST_LIST_PATH"

current_sample=0
success_count=0
fail_count=0
skip_count=0

# ===== Header =====
echo "=========================================="
echo "HYPIR-Enhanced DepthCAD Inference"
echo "=========================================="
echo "Model: $MODEL_DIR"
echo "ControlNet: $DEPTHCAD_PATH"
echo "Dataset type: $DATASET_TYPE"
echo "Target size: ${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo "Inference steps: $NUM_INFERENCE_STEPS"
echo "Guidance scale: $GUIDANCE_SCALE"
echo "Random seed: $SEED"
echo "Output directory: $OUT_DIR"
echo "Total samples: $total_samples"
echo "=========================================="
echo ""

# ===== Function to process a single sample =====
process_sample() {
    local sample_path=$1
    local local_current=$2
    local local_total=$3

    noise_iq_file="$NOISE_IQ_DIR/$sample_path.npy"
    noise_depth_file="$NOISE_DEPTH_DIR/$sample_path.npy"
    out_file="$OUT_DIR/$sample_path.npy"

    # Create output directory structure if needed
    out_dir_path=$(dirname "$out_file")
    mkdir -p "$out_dir_path"

    # Check if input files exist
    if [ ! -f "$noise_iq_file" ]; then
        echo "[WARN] [$local_current/$local_total] Noise IQ missing: $noise_iq_file" >&2
        return 1
    fi
    if [ ! -f "$noise_depth_file" ]; then
        echo "[WARN] [$local_current/$local_total] Noise depth missing: $noise_depth_file" >&2
        return 1
    fi

    # Skip if output already exists
    if [ -f "$out_file" ]; then
        echo "[SKIP] [$local_current/$local_total] Output exists: $sample_path"
        return 2
    fi

    echo "[INFO] [$local_current/$local_total] Processing: $sample_path"
    echo "  Input IQ: $noise_iq_file"
    echo "  Input Depth: $noise_depth_file"
    echo "  Output: $out_file"

    # Run inference
    if python inference_hypir.py \
        --pretrained_model_name_or_path "$MODEL_DIR" \
        --depthcad_path "$DEPTHCAD_PATH" \
        --noise_IQ_file "$noise_iq_file" \
        --noise_depth_file "$noise_depth_file" \
        --out_file "$out_file" \
        --dataset_type "$DATASET_TYPE" \
        --target_size "$TARGET_HEIGHT" "$TARGET_WIDTH" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --seed "$SEED"; then
        echo "[SUCCESS] [$local_current/$local_total] Completed: $sample_path"
        return 0
    else
        echo "[ERROR] [$local_current/$local_total] Failed: $sample_path" >&2
        return 1
    fi
}

# ===== Process each line in test.txt =====
while IFS= read -r line; do
    # Skip empty lines
    [ -z "$line" ] && continue

    # Remove leading/trailing whitespace
    line=$(echo "$line" | xargs)
    [ -z "$line" ] && continue

    # Check if line is a directory path (e.g., "bathroom/1")
    slash_count=$(echo "$line" | tr -cd '/' | wc -c)

    if [ "$slash_count" -eq 1 ]; then
        # Format: "category/subdir" - process all files in that directory
        category_subdir="$line"
        category_dir="$NOISE_IQ_DIR/$category_subdir"

        if [ ! -d "$category_dir" ]; then
            echo "[WARN] Directory not found: $category_dir" >&2
            fail_count=$((fail_count + 1))
            continue
        fi

        echo "[INFO] Processing directory: $category_subdir"

        # Create temporary file list
        temp_file_list=$(mktemp)
        find "$category_dir" -maxdepth 1 -type f -name "*.npy" | sort > "$temp_file_list"

        while IFS= read -r npy_file; do
            # Extract filename without extension
            filename=$(basename "$npy_file" .npy)
            sample_path="$category_subdir/$filename"

            current_sample=$((current_sample + 1))

            if process_sample "$sample_path" "$current_sample" "$total_samples"; then
                result=$?
                if [ $result -eq 0 ]; then
                    success_count=$((success_count + 1))
                elif [ $result -eq 2 ]; then
                    skip_count=$((skip_count + 1))
                else
                    fail_count=$((fail_count + 1))
                fi
            else
                fail_count=$((fail_count + 1))
            fi
            echo "---"
        done < "$temp_file_list"

        # Clean up temporary file
        rm -f "$temp_file_list"

    elif [ "$slash_count" -ge 2 ]; then
        # Format: "category/subdir/filename" - process single file
        current_sample=$((current_sample + 1))
        sample_path="${line%.npy}"  # Remove .npy extension if present

        if process_sample "$sample_path" "$current_sample" "$total_samples"; then
            result=$?
            if [ $result -eq 0 ]; then
                success_count=$((success_count + 1))
            elif [ $result -eq 2 ]; then
                skip_count=$((skip_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
        else
            fail_count=$((fail_count + 1))
        fi
        echo "---"
    else
        # Format: just a filename - not supported
        echo "[WARN] Unsupported format (single filename): $line" >&2
        echo "  Expected format: 'category/subdir' or 'category/subdir/filename'" >&2
        fail_count=$((fail_count + 1))
        continue
    fi
done < "$TEST_LIST_PATH"

# ===== Summary =====
echo ""
echo "=========================================="
echo "Inference completed!"
echo "=========================================="
echo "Total samples: $total_samples"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo "Skipped: $skip_count"
echo "Output directory: $OUT_DIR"
echo "=========================================="

# ===== Evaluation =====
# Uncomment to run evaluation after inference
# echo ""
# echo "Running evaluation..."
# python eval.py \
#     --test_list_path "$TEST_LIST_PATH" \
#     --out_dir "$OUT_DIR" \
#     --pred_dir "$OUT_DIR" \
#     --gt_dir "/data/pre_student/hcy/pbrt/gt_depth"
