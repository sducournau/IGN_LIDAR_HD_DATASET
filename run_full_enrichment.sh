#!/bin/bash
# Launch IGN LiDAR HD Pipeline with Full Enrichment Preset
# Date: October 16, 2025
# Input: D:\ign\selected_tiles\asprs\tiles
# Output: D:\ign\preprocessed\asprs

# Activate conda environment
echo "Activating ign_gpu conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

# Configuration
CONFIG="configs/enrichment_asprs_full.yaml"
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/mnt/d/ign/preprocessed/asprs"
CACHE_DIR="/mnt/d/ign/cache"

# Create output and cache directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Display configuration
echo "=================================================="
echo "IGN LiDAR HD - Full Enrichment Pipeline"
echo "=================================================="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Cache:  $CACHE_DIR"
echo "Config: $CONFIG"
echo "=================================================="
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Count input files
FILE_COUNT=$(find "$INPUT_DIR" -name "*.laz" -o -name "*.las" | wc -l)
echo "Found $FILE_COUNT LiDAR files to process"
echo ""

# Run the pipeline with overrides
ign-lidar-hd process \
    --config-file "$CONFIG" \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    output.format=laz \
    processor.processing_mode=enriched_only \
    processor.lod_level=ASPRS_classes \
    processor.num_workers=1 \
    processor.use_gpu=true \
    processor.use_ground_truth=true \
    processor.num_augmentations=0 \
    processor.patch_overlap=0.0 \
    output.enriched.save=true \
    output.patches.save=false \
    features.mode=full \
    features.include_extra=true \
    features.asprs_classes=true \
    data_sources.bd_topo.enabled=true \
    data_sources.bd_foret.enabled=true \
    data_sources.rpg.enabled=true \
    data_sources.cadastre.enabled=true \
    log_level=DEBUG

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Pipeline completed successfully!"
    echo "=================================================="
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    ls -lh "$OUTPUT_DIR"
else
    echo ""
    echo "=================================================="
    echo "Pipeline failed with errors"
    echo "=================================================="
    exit 1
fi
