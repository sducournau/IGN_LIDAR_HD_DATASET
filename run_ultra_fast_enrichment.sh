#!/bin/bash
# ULTRA-FAST IGN LiDAR HD Pipeline - Enriched LAZ Only
# Optimized for RTX 4080 Super - SPEED FOCUSED
# Date: October 17, 2025
# Target: <5 minutes per tile (vs 2+ hours)

# Function to log with timestamp
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Activate conda environment
log_info "Activating ign_gpu conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

# Check if environment activation was successful
if [ $? -ne 0 ]; then
    log_error "Failed to activate ign_gpu environment"
    exit 1
fi

# Configuration - Ultra-fast optimized config
CONFIG="configs/config_asprs_rtx4080.yaml"
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/mnt/d/ign/preprocessed/asprs/enriched_tiles_ultra_fast"
CACHE_DIR="/mnt/d/ign/cache"

# Create output and cache directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Display configuration
echo "======================================================================"
echo "IGN LiDAR HD - ULTRA-FAST Enriched LAZ Pipeline"
echo "======================================================================"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Config:       $CONFIG"
echo "Mode:         ENRICHED_ONLY (NO PATCHES!)"
echo "Speed:        ULTRA-FAST (minimal features, no NIR/NDVI)"
echo "======================================================================"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    log_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    log_error "Configuration file does not exist: $CONFIG"
    exit 1
fi

# Count input files
FILE_COUNT=$(find "$INPUT_DIR" -name "*.laz" -o -name "*.las" | wc -l)
log_info "Found $FILE_COUNT LiDAR files to process"

if [ $FILE_COUNT -eq 0 ]; then
    log_error "No LiDAR files found in input directory"
    exit 1
fi

# Record start time for performance tracking
START_TIME=$(date +%s)

# Run the ULTRA-FAST pipeline
log_info "Starting ULTRA-FAST IGN LiDAR HD processing..."
ign-lidar-hd process \
    --config-file "$CONFIG" \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    cache_dir="$CACHE_DIR" \
    processor.use_gpu=true \
    processor.architecture=direct \
    processor.generate_patches=false \
    processor.apply_reclassification_inline=false \
    processing.mode=enriched_only \
    processing.architecture=direct \
    processing.generate_patches=false \
    ground_truth.enabled=true \
    ground_truth.update_classification=true \
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    data_sources.bd_topo.enabled=true \
    data_sources.cadastre.enabled=false \
    features.use_nir=false \
    features.compute_ndvi=false \
    features.use_infrared=false \
    preprocess.enabled=false \
    stitching.enabled=false \
    log_level=INFO

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# Check exit status and provide detailed feedback
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ ULTRA-FAST Pipeline completed successfully!"
    echo "======================================================================"
    log_info "Processing completed in ${DURATION_MIN}m ${DURATION_SEC}s"
    log_info "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Output files (enriched LAZ only):"
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -name "*.laz" -not -name "*patch*" | head -10
        ENRICHED_COUNT=$(find "$OUTPUT_DIR" -name "*.laz" -not -name "*patch*" | wc -l)
        PATCH_COUNT=$(find "$OUTPUT_DIR" -name "*patch*.laz" | wc -l)
        echo ""
        echo "✅ Enriched LAZ files: $ENRICHED_COUNT"
        if [ $PATCH_COUNT -gt 0 ]; then
            echo "⚠️  Patches generated: $PATCH_COUNT (should be 0!)"
        else
            echo "✅ Patches: 0 (as expected)"
        fi
    fi
    echo ""
    echo "Performance summary:"
    echo "Files processed: $FILE_COUNT"
    if [ $DURATION -gt 0 ]; then
        echo "Processing rate: $(echo "scale=2; $FILE_COUNT * 60 / $DURATION" | bc 2>/dev/null || echo "N/A") files/hour"
        echo "Time per tile: $(echo "scale=1; $DURATION / $FILE_COUNT" | bc 2>/dev/null || echo "N/A") seconds"
    fi
else
    echo ""
    echo "======================================================================"
    echo "❌ ULTRA-FAST Pipeline failed with errors"
    echo "======================================================================"
    log_error "Processing failed after ${DURATION_MIN}m ${DURATION_SEC}s"
    log_error "Check the logs above for error details"
    exit 1
fi