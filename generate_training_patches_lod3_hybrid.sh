#!/bin/bash
# ============================================================================
# IGN LiDAR HD - Training Patch Generation Script
# LOD3 Classification - Hybrid Model - Architecture Agnostic
# ============================================================================
#
# 🎯 Optimized for Hybrid Deep Learning Models
#
# This script generates training patches specifically optimized for:
#   • PointNet++ (geometric features)
#   • Transformer (multi-modal attention)
#   • Octree-CNN (multi-scale hierarchy)
#   • Sparse Convolution (voxel-based)
#
# Features:
#   ✓ LOD3 (high detail building classification)
#   ✓ 32,768 points per patch (2x more than LOD2)
#   ✓ Geometric augmentation (5x per patch)
#   ✓ Full feature extraction (RGB, NIR, NDVI, geometric)
#   ✓ Aggressive preprocessing (noise removal)
#   ✓ Tile stitching with 20m buffer (boundary-aware)
#   ✓ Auto-download missing neighbors
#   ✓ NPZ format (architecture-agnostic)
#
# Usage:
#   bash generate_training_patches_lod3_hybrid.sh
#
# Or with custom input/output:
#   bash generate_training_patches_lod3_hybrid.sh /path/to/input /path/to/output
#
# ============================================================================

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default paths
INPUT_DIR="${1:-/mnt/c/Users/Simon/ign/raw_tiles}"
OUTPUT_DIR="${2:-/mnt/c/Users/Simon/ign/patch_1st_training}"

# Print banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           IGN LiDAR HD - LOD3 Hybrid Model Training Patches               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "   Input:  $INPUT_DIR"
echo "   Output: $OUTPUT_DIR"
echo ""
echo -e "${GREEN}Model Target:${NC} Hybrid Architecture (PointNet++ + Transformer + Octree + Sparse)"
echo -e "${GREEN}LOD Level:${NC} LOD3 (High Detail)"
echo -e "${GREEN}Points/Patch:${NC} 32,768"
echo -e "${GREEN}Features:${NC} RGB + NIR + NDVI + Geometric (30 neighbors)"
echo -e "${GREEN}Augmentation:${NC} 5x per patch"
echo -e "${GREEN}Preprocessing:${NC} Aggressive"
echo -e "${GREEN}Stitching:${NC} Enabled (20m buffer)"
echo -e "${GREEN}Output Format:${NC} NPZ (Architecture-agnostic)"
echo ""
echo "============================================================================"
echo -e "${YELLOW}Starting processing... This may take several hours.${NC}"
echo "============================================================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Main processing command
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

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================================"
if [ $EXIT_STATUS -eq 0 ]; then
    echo -e "${GREEN}✓ Processing completed successfully!${NC}"
    echo "   Output directory: $OUTPUT_DIR"
    echo "   Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "   1. Verify patches: ls -lh $OUTPUT_DIR"
    echo "   2. Check stats: cat $OUTPUT_DIR/stats.json"
    echo "   3. Train model: python train_hybrid_model.py --data $OUTPUT_DIR"
else
    echo -e "${RED}✗ Processing failed with exit code: $EXIT_STATUS${NC}"
    echo "   Check logs for details"
    exit $EXIT_STATUS
fi
echo "============================================================================"
echo ""
