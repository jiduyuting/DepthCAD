#!/bin/bash
# Master script to run multiple model checkpoints in parallel
# Each checkpoint will run in a separate terminal or background process

echo "=========================================="
echo "Running inference on multiple checkpoints"
echo "=========================================="

# Check available GPUs
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

echo ""
echo "Available checkpoints:"
echo "  - checkpoint-30000 -> output: pbrt/data_1_10_30000"
echo "  - checkpoint-40000 -> output: pbrt/data_1_10_40000"
echo "  - checkpoint-50000 -> output: pbrt/data_1_10_50000"
echo ""

# Function to run a single checkpoint inference
run_checkpoint() {
    local script=$1
    local checkpoint_name=$2

    echo "Starting $checkpoint_name..."

    # Option 1: Run in background (nohup)
    nohup bash "$script" > "logs/${checkpoint_name}.log" 2>&1 &
    local pid=$!
    echo "  $checkpoint_name started with PID: $pid"
    echo "  Log file: logs/${checkpoint_name}.log"
    echo $pid > "logs/${checkpoint_name}.pid"

    # Option 2: Uncomment below to run in new terminal instead
    # gnome-terminal -- bash -c "bash $script; exec bash" &
}

# Create logs directory
mkdir -p logs

# Run all checkpoints in parallel
run_checkpoint "inference_30k.sh" "checkpoint_30k"
sleep 2  # Small delay to avoid GPU initialization conflicts

run_checkpoint "inference_40k.sh" "checkpoint_40k"
sleep 2

run_checkpoint "inference_50k.sh" "checkpoint_50k"

echo ""
echo "=========================================="
echo "All checkpoints started!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/checkpoint_30k.log"
echo "  tail -f logs/checkpoint_40k.log"
echo "  tail -f logs/checkpoint_50k.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep inference"
echo ""
echo "Stop all processes:"
echo "  pkill -f inference.py"
echo "=========================================="
