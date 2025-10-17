#!/bin/bash
# Test GPU utilization with a single tile

# Create test directory for single tile
mkdir -p /mnt/d/ign/test_single_tile

# Copy one LAZ file for testing
cp "/mnt/d/ign/selected_tiles/asprs/tiles/LHD_FXX_0326_6829_PTS_C_LAMB93_IGN69.laz" /mnt/d/ign/test_single_tile/

# Activate conda environment and run test
conda activate ign_gpu

echo "🚀 Starting GPU test with single tile..."
echo "📊 Monitor GPU usage in another terminal with: watch -n 1 nvidia-smi"

# Run processing on single tile
ign-lidar-hd process \
    --config-file "ign_lidar/configs/presets/asprs_classification_gpu_optimized.yaml" \
    input_dir="/mnt/d/ign/test_single_tile" \
    output_dir="/mnt/d/ign/test_single_tile_output"

echo "✅ Test completed!"