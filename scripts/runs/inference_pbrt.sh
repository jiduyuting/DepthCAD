#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/scripts:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_DIR="/home/lab507/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3
DEPTHCAD_PATH="/data/pre_student/GJ/DepthCAD/output/depthcad_pbrt_sana_logital/checkpoint-10000/depthcad_ema"
TEST_LIST_PATH="/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt"
NOISE_IQ_DIR="/data/pre_student/hcy/pbrt/noise"
NOISE_DEPTH_DIR="/data/pre_student/hcy/pbrt/noise_depth"
OUT_DIR="/data/pre_student/GJ/DepthCAD/pbrt/data_ema_10000_1"

# Dataset configuration
DATASET_TYPE="pbrt"
TARGET_HEIGHT=512
TARGET_WIDTH=512

mkdir -p "$OUT_DIR"

# Count total samples for progress tracking
# For directory entries, count files in those directories
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

echo "=========================================="
echo "Starting inference for PBRT dataset"
echo "=========================================="
echo "Model: checkpoint-10000"
echo "Total samples: $total_samples"
echo "Dataset type: $DATASET_TYPE"
echo "Target size: ${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo "Output directory: $OUT_DIR"
echo "=========================================="

# Function to process a single sample
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

    if [ ! -f "$noise_iq_file" ]; then
        echo "[WARN] [$local_current/$local_total] noise IQ missing: $noise_iq_file" >&2
        return 1
    fi
    if [ ! -f "$noise_depth_file" ]; then
        echo "[WARN] [$local_current/$local_total] noise depth missing: $noise_depth_file" >&2
        return 1
    fi

    # Skip if output already exists
    if [ -f "$out_file" ]; then
        echo "[SKIP] [$local_current/$local_total] Output exists: $sample_path"
        return 0
    fi

    echo "[INFO] [$local_current/$local_total] Processing: $sample_path"
    echo "  Input IQ: $noise_iq_file"
    echo "  Input Depth: $noise_depth_file"
    echo "  Output: $out_file"

    if python scripts/inference_pbrt.py \
        --pretrained_model_name_or_path "$MODEL_DIR" \
        --depthcad_path "$DEPTHCAD_PATH" \
        --noise_IQ_file "$noise_iq_file" \
        --noise_depth_file "$noise_depth_file" \
        --out_file "$out_file" \
        --target_size "$TARGET_HEIGHT" "$TARGET_WIDTH"; then
        echo "[SUCCESS] [$local_current/$local_total] Completed: $sample_path"
        return 0
    else
        echo "[ERROR] [$local_current/$local_total] Failed: $sample_path" >&2
        return 1
    fi
}

# Process each line in test.txt
while IFS= read -r line; do
    # 跳过空行
    [ -z "$line" ] && continue

    # Remove leading/trailing whitespace
    line=$(echo "$line" | xargs)
    [ -z "$line" ] && continue

    # Check if line is a directory path (e.g., "bathroom/1")
    # Count slashes: directory format has exactly one slash
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
        # Find all .npy files in this directory and process them
        # Use process substitution with explicit bash, or use a temporary file approach
        # Create a temporary file list to avoid subshell issues
        temp_file_list=$(mktemp)
        find "$category_dir" -maxdepth 1 -type f -name "*.npy" | sort > "$temp_file_list"

        while IFS= read -r npy_file; do
            # Extract filename without extension
            filename=$(basename "$npy_file" .npy)
            sample_path="$category_subdir/$filename"

            current_sample=$((current_sample + 1))
            if process_sample "$sample_path" "$current_sample" "$total_samples"; then
                success_count=$((success_count + 1))
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
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
        echo "---"
    else
        # Format: just a filename - try to find it
        echo "[WARN] Unsupported format (single filename): $line" >&2
        echo "  Expected format: 'category/subdir' or 'category/subdir/filename'" >&2
        fail_count=$((fail_count + 1))
        continue
    fi
done < "$TEST_LIST_PATH"

echo "=========================================="
echo "Inference completed!"
echo "Total samples: $total_samples"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo "Skipped: $((total_samples - success_count - fail_count))"
echo "=========================================="
