#!/bin/bash
# ============================================================================
# IGN LiDAR HD - Training Patch Generation Script
# LOD3 Classification - Architecture Agnostic - Full Features - HYBRID MODEL
# ============================================================================
#
# This script generates training patches from raw LiDAR tiles with:
# - LOD3 (high detail building classification)
# - 32,768 points per patch (2x more than LOD2)
# - Geometric augmentation (5x per patch)
# - Full feature extraction (RGB, NIR, NDVI, geometric)
# - Aggressive preprocessing (noise removal)
# - Tile stitching with 20m buffer (boundary-aware processing)
# - Architecture-agnostic output (NPZ format for hybrid models)
#
# Usage:
#   bash generate_training_patches_lod2.sh
#
# Or with custom input/output:
#   bash generate_training_patches_lod2.sh /path/to/input /path/to/output
#
# ============================================================================

# Default paths
INPUT_DIR="${1:-/mnt/c/Users/Simon/ign/raw_tiles/urban_dense}"
OUTPUT_DIR="${2:-/mnt/c/Users/Simon/ign/training_patches_lod3_hybrid}"

echo "============================================================================"
echo "IGN LiDAR HD - Training Patch Generation"
echo "============================================================================"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "============================================================================"

# Main processing command - OPTIMIZED FOR LOD3 HYBRID MODEL
ign-lidar-hd process \
  input_dir="$INPUT_DIR" \
  output_dir="$OUTPUT_DIR" \
  processor.lod_level=LOD3 \
  processor.use_gpu=true \
  processor.num_workers=4 \
  processor.num_points=32768 \
  processor.patch_size=150.0 \
  processor.patch_overlap=0.15 \
  processor.augment=true \
  processor.num_augmentations=5 \
  features=full \
  features.mode=full \
  features.k_neighbors=30 \
  features.include_extra=true \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  features.sampling_method=fps \
  features.normalize_xyz=true \
  features.normalize_features=true \
  preprocess=aggressive \
  preprocess.enabled=true \
  stitching=enabled \
  stitching.enabled=true \
  stitching.buffer_size=20.0 \
  stitching.auto_detect_neighbors=true \
  stitching.auto_download_neighbors=true \
  stitching.cache_enabled=true \
  output.format=npz \
  output.save_enriched_laz=false \
  output.save_stats=true \
  output.save_metadata=true \
  log_level=INFO

# Capture exit status
EXIT_STATUS=$?

echo "============================================================================"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "✓ Processing completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
else
    echo "✗ Processing failed with exit code: $EXIT_STATUS"
    exit $EXIT_STATUS
fi
echo "============================================================================"
