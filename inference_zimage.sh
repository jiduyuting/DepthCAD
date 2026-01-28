#!/bin/bash

# DepthCAD Inference Script for Z-Image-Turbo Model
# Based on inference_zimage.py

# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# ========== 配置区域 ==========
# 模型配置
MODEL_DIR="Tongyi-MAI/Z-Image-Turbo"  # 使用 Z-Image-Turbo 作为基础模型
DEPTHCAD_PATH="/data/pre_student/GJ/DepthCAD/output/depthcad_pbrt_zimage/checkpoint-30000/depthcad"

# 数据路径
TEST_LIST_PATH="/data/pre_student/GJ/DepthCAD/pbrt_dataset/test.txt"
NOISE_IQ_DIR="/data/pre_student/hcy/pbrt/noise"
NOISE_DEPTH_DIR="/data/pre_student/hcy/pbrt/noise_depth"
OUT_DIR="/data/pre_student/GJ/DepthCAD/pbrt/data_zimage_30000"

# 数据集配置
DATASET_TYPE="pbrt"
TARGET_HEIGHT=240
TARGET_WIDTH=320

# GPU 设置（可选）
# export CUDA_VISIBLE_DEVICES=0

# ========== 检查配置 ==========
echo "=========================================="
echo "DepthCAD Inference Configuration"
echo "=========================================="
echo "Base Model: $MODEL_DIR"
echo "DepthCAD: $DEPTHCAD_PATH"
echo "Dataset: $DATASET_TYPE"
echo "Target Size: ${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo "Output: $OUT_DIR"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUT_DIR"

# 检查必要文件是否存在
if [ ! -f "$TEST_LIST_PATH" ]; then
    echo "[ERROR] Test list not found: $TEST_LIST_PATH"
    exit 1
fi

if [ ! -d "$NOISE_IQ_DIR" ]; then
    echo "[ERROR] Noise IQ directory not found: $NOISE_IQ_DIR"
    exit 1
fi

if [ ! -d "$NOISE_DEPTH_DIR" ]; then
    echo "[ERROR] Noise depth directory not found: $NOISE_DEPTH_DIR"
    exit 1
fi

# 统计总样本数
total_samples=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    line=$(echo "$line" | xargs)
    [ -z "$line" ] && continue

    # 计算斜杠数量以确定格式
    slash_count=$(echo "$line" | tr -cd '/' | wc -c)
    if [ "$slash_count" -eq 1 ]; then
        # 目录格式：统计目录中的文件数
        category_dir="$NOISE_IQ_DIR/$line"
        if [ -d "$category_dir" ]; then
            file_count=$(find "$category_dir" -maxdepth 1 -type f -name "*.npy" 2>/dev/null | wc -l)
            total_samples=$((total_samples + file_count))
        fi
    else
        # 单个文件格式
        total_samples=$((total_samples + 1))
    fi
done < "$TEST_LIST_PATH"

current_sample=0
success_count=0
fail_count=0
skip_count=0

echo "Total samples to process: $total_samples"
echo ""
echo "=========================================="
echo "Starting Inference..."
echo "=========================================="
echo ""

# 处理单个样本的函数
process_sample() {
    local sample_path=$1
    local local_current=$2
    local local_total=$3

    local noise_iq_file="$NOISE_IQ_DIR/$sample_path.npy"
    local noise_depth_file="$NOISE_DEPTH_DIR/$sample_path.npy"
    local out_file="$OUT_DIR/$sample_path.npy"

    # 创建输出目录结构
    local out_dir_path=$(dirname "$out_file")
    mkdir -p "$out_dir_path"

    # 检查输入文件是否存在
    if [ ! -f "$noise_iq_file" ]; then
        echo "[WARN] [$local_current/$local_total] Noise IQ missing: $sample_path" >&2
        return 1
    fi

    if [ ! -f "$noise_depth_file" ]; then
        echo "[WARN] [$local_current/$local_total] Noise depth missing: $sample_path" >&2
        return 1
    fi

    # 如果输出已存在则跳过
    if [ -f "$out_file" ]; then
        echo "[SKIP] [$local_current/$local_total] $sample_path (output exists)"
        return 2
    fi

    # 运行推理
    echo "[INFO] [$local_current/$local_total] Processing: $sample_path"

    if python inference_zimage.py \
        --pretrained_model_name_or_path "$MODEL_DIR" \
        --depthcad_path "$DEPTHCAD_PATH" \
        --noise_IQ_file "$noise_iq_file" \
        --noise_depth_file "$noise_depth_file" \
        --out_file "$out_file" \
        --dataset_type "$DATASET_TYPE" \
        --target_size "$TARGET_HEIGHT" "$TARGET_WIDTH" 2>&1; then
        echo "[SUCCESS] [$local_current/$local_total] $sample_path"
        return 0
    else
        local exit_code=$?
        echo "[ERROR] [$local_current/$local_total] Failed: $sample_path (exit code: $exit_code)" >&2
        return 1
    fi
}

# 处理测试列表中的每一行
while IFS= read -r line; do
    # 跳过空行
    [ -z "$line" ] && continue

    # 去除首尾空格
    line=$(echo "$line" | xargs)
    [ -z "$line" ] && continue

    # 计算斜杠数量以确定格式
    slash_count=$(echo "$line" | tr -cd '/' | wc -c)

    if [ "$slash_count" -eq 1 ]; then
        # 格式: "category/subdir" - 处理该目录下的所有文件
        category_subdir="$line"
        category_dir="$NOISE_IQ_DIR/$category_subdir"

        if [ ! -d "$category_dir" ]; then
            echo "[WARN] Directory not found: $category_dir" >&2
            fail_count=$((fail_count + 1))
            continue
        fi

        echo "[INFO] Processing directory: $category_subdir"

        # 创建临时文件列表以避免子shell问题
        temp_file_list=$(mktemp)
        find "$category_dir" -maxdepth 1 -type f -name "*.npy" | sort > "$temp_file_list"

        while IFS= read -r npy_file; do
            # 提取文件名（不含扩展名）
            filename=$(basename "$npy_file" .npy)
            sample_path="$category_subdir/$filename"

            current_sample=$((current_sample + 1))

            process_sample "$sample_path" "$current_sample" "$total_samples"
            result=$?

            if [ $result -eq 0 ]; then
                success_count=$((success_count + 1))
            elif [ $result -eq 2 ]; then
                skip_count=$((skip_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
            echo ""
        done < "$temp_file_list"

        # 清理临时文件
        rm -f "$temp_file_list"

    elif [ "$slash_count" -ge 2 ]; then
        # 格式: "category/subdir/filename" - 处理单个文件
        current_sample=$((current_sample + 1))
        local sample_path="${line%.npy}"  # 移除 .npy 扩展名（如果存在）

        process_sample "$sample_path" "$current_sample" "$total_samples"
        local result=$?

        if [ $result -eq 0 ]; then
            success_count=$((success_count + 1))
        elif [ $result -eq 2 ]; then
            skip_count=$((skip_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
        echo ""
    else
        # 格式: 只是文件名 - 尝试查找
        echo "[WARN] Unsupported format: $line" >&2
        echo "  Expected: 'category/subdir' or 'category/subdir/filename'" >&2
        fail_count=$((fail_count + 1))
        continue
    fi
done < "$TEST_LIST_PATH"

# 打印总结
echo ""
echo "=========================================="
echo "Inference Summary"
echo "=========================================="
echo "Total samples: $total_samples"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo "Skipped: $skip_count"
echo "Output directory: $OUT_DIR"
echo "=========================================="

if [ $fail_count -gt 0 ]; then
    echo "[WARNING] Some samples failed to process. Check logs above for details."
    exit 1
else
    echo "[SUCCESS] All samples processed successfully!"
    exit 0
fi
