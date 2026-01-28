#!/bin/bash
# Script to generate masked PBRT dataset
# This runs process_mask.py to create data with amplitude masking

echo "========================================================================"
echo "Generating Masked PBRT Dataset"
echo "========================================================================"
echo ""
echo "Mask Configuration:"
echo "  - Target size: 512x512"
echo "  - Lower threshold: Adaptive 5th percentile of amplitude"
echo "  - Upper threshold: 99.5th percentile (removes top 0.5%)"
echo "  - Masked regions: Set to 0 in IQ data and confidence map"
echo ""
echo "Output directories:"
echo "  - pbrt_dataset/data/ideal_IQ_masked/"
echo "  - pbrt_dataset/data/noise_IQ_masked/"
echo "  - pbrt_dataset/data/confidence_masked/"
echo ""
echo "========================================================================"
echo ""

# Run the preprocessing script
python pbrt_dataset/process_mask.py

echo ""
echo "========================================================================"
echo "Masked data generation completed!"
echo ""
echo "Next steps:"
echo "  1. Train with masked data: bash train_masked.sh"
echo "  2. Compare with baseline (train.sh) to evaluate mask benefit"
echo "========================================================================"
