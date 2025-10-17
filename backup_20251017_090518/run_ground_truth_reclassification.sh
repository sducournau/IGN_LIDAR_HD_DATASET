#!/bin/bash
# IGN LiDAR HD - Ground Truth Reclassification Pipeline
# ENRICHED LAZ with BD TOPO Ground Truth Classification
# NO Cadastre - FAST Processing
# Date: October 17, 2025

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

# Configuration paths
CONFIG="configs/config_asprs_rtx4080.yaml"
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/mnt/d/ign/preprocessed/asprs/enriched_reclassified"
CACHE_DIR="/mnt/d/ign/cache"

# Create output and cache directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Display configuration
echo "======================================================================"
echo "IGN LiDAR HD - Ground Truth Reclassification Pipeline"
echo "======================================================================"
echo "Input:              $INPUT_DIR"
echo "Output:             $OUTPUT_DIR"
echo "Config:             $CONFIG"
echo "Mode:               ENRICHED + GROUND TRUTH RECLASSIFICATION"
echo "Ground Truth:       BD TOPO (Buildings, Roads, Water)"
echo "Cadastre:           DISABLED (for speed)"
echo "NDVI:               DISABLED (for speed)" 
echo "Reclassification:   ENABLED (CPU mode for stability)"
echo "GPU Features:       ENABLED (16M batch size)"
echo "======================================================================"
echo ""

# Check directories
if [ ! -d "$INPUT_DIR" ]; then
    log_error "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

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

# Record start time
START_TIME=$(date +%s)

# Run with Ground Truth Reclassification ENABLED
log_info "Starting Ground Truth Reclassification pipeline..."

ign-lidar-hd process \
    --config-file "$CONFIG" \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    cache_dir="$CACHE_DIR" \
    \
    processor.use_gpu=true \
    processor.architecture=direct \
    processor.processing_mode=enriched_only \
    processor.batch_size=512 \
    processor.generate_patches=false \
    processor.patch_size=null \
    processor.patch_overlap=null \
    processor.num_points=null \
    processor.direct_enrichment=true \
    processor.use_stitching=false \
    processor.buffer_size=0.0 \
    processor.apply_reclassification_inline=true \
    processor.reclassification.enabled=true \
    processor.reclassification.acceleration_mode=cpu \
    processor.reclassification.chunk_size=2000000 \
    processor.reclassification.use_geometric_rules=true \
    \
    processing.mode=enriched_only \
    processing.architecture=direct \
    processing.generate_patches=false \
    processing.use_gpu=true \
    processing.num_workers=1 \
    \
    features.use_gpu=true \
    features.gpu_batch_size=16000000 \
    features.vram_utilization_target=0.9 \
    features.num_cuda_streams=8 \
    features.k_neighbors=8 \
    features.search_radius=0.6 \
    features.use_nir=false \
    features.use_infrared=false \
    features.compute_ndvi=false \
    features.compute_verticality=false \
    features.compute_horizontality=false \
    features.compute_sphericity=false \
    features.compute_curvature=false \
    features.compute_linearity=false \
    features.gpu_optimization.enable_mixed_precision=true \
    \
    preprocess.enabled=false \
    stitching.enabled=false \
    \
    ground_truth.enabled=true \
    ground_truth.update_classification=true \
    ground_truth.apply_reclassification=true \
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    ground_truth.optimization.force_method=strtree \
    ground_truth.optimization.enable_monitoring=false \
    ground_truth.optimization.enable_auto_tuning=false \
    \
    classification.enabled=true \
    classification.methods.ground_truth=true \
    classification.methods.geometric=true \
    classification.methods.ndvi=false \
    classification.methods.forest_types=false \
    classification.methods.crop_types=false \
    \
    data_sources.bd_topo_enabled=true \
    data_sources.bd_topo_buildings=true \
    data_sources.bd_topo_roads=true \
    data_sources.bd_topo_water=true \
    data_sources.bd_topo_vegetation=false \
    data_sources.cadastre_enabled=false \
    data_sources.bd_foret_enabled=false \
    data_sources.rpg_enabled=false \
    data_sources.bd_topo.enabled=true \
    data_sources.bd_topo.features.buildings=true \
    data_sources.bd_topo.features.roads=true \
    data_sources.bd_topo.features.water=true \
    data_sources.bd_topo.features.vegetation=false \
    data_sources.bd_topo.features.railways=false \
    data_sources.bd_topo.features.bridges=false \
    data_sources.bd_topo.features.parking=false \
    data_sources.bd_topo.features.cemeteries=false \
    data_sources.bd_topo.features.power_lines=false \
    data_sources.bd_topo.features.sports=false \
    data_sources.cadastre.enabled=false \
    \
    log_level=INFO \
    verbose=true

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Ground Truth Reclassification Pipeline completed successfully!"
    echo "======================================================================"
    log_info "Processing completed in ${DURATION_MIN}m ${DURATION_SEC}s"
    log_info "Output directory: $OUTPUT_DIR"
    echo ""
    
    if [ -d "$OUTPUT_DIR" ]; then
        ENRICHED_COUNT=$(find "$OUTPUT_DIR" -name "*enriched*.laz" | wc -l)
        RECLASSIFIED_COUNT=$(find "$OUTPUT_DIR" -name "*reclassified*.laz" | wc -l)
        PATCH_COUNT=$(find "$OUTPUT_DIR" -name "*patch*.laz" | wc -l 2>/dev/null || echo "0")
        
        echo "Results:"
        echo "✅ Enriched LAZ files: $ENRICHED_COUNT"
        echo "✅ Reclassified LAZ files: $RECLASSIFIED_COUNT"
        if [ "$PATCH_COUNT" -gt 0 ]; then
            echo "⚠️  WARNING: Patches still generated: $PATCH_COUNT (should be 0!)"
        else
            echo "✅ Patches: 0 (perfect!)"
        fi
        
        echo ""
        echo "Sample output files:"
        find "$OUTPUT_DIR" -name "*.laz" | head -5
        
        echo ""
        echo "File sizes:"
        find "$OUTPUT_DIR" -name "*.laz" -exec ls -lh {} \; | head -3
    fi
    
    echo ""
    echo "Ground Truth Classification Features Applied:"
    echo "✅ BD TOPO Buildings → ASPRS Class 6"
    echo "✅ BD TOPO Roads → ASPRS Class 11" 
    echo "✅ BD TOPO Water → ASPRS Class 9"
    echo "✅ Geometric rules for remaining points"
    echo "❌ Cadastre processing (disabled for speed)"
    echo "❌ NDVI classification (disabled for speed)"
    
    echo ""
    echo "Performance:"
    if [ $DURATION -gt 0 ]; then
        echo "Processing rate: $(echo "scale=1; $FILE_COUNT * 60 / $DURATION" | bc 2>/dev/null || echo "N/A") files/hour"
        echo "Time per tile: $(echo "scale=1; $DURATION / $FILE_COUNT" | bc 2>/dev/null || echo "N/A") seconds/tile"
    fi
else
    echo ""
    echo "======================================================================"
    echo "❌ Ground Truth Reclassification Pipeline failed!"
    echo "======================================================================"
    log_error "Processing failed after ${DURATION_MIN}m ${DURATION_SEC}s"
    log_error "Check the logs above for error details"
    
    echo ""
    echo "Common issues to check:"
    echo "1. GPU memory: nvidia-smi"
    echo "2. Disk space in output directory"
    echo "3. BD TOPO cache/network connectivity"
    echo "4. Input file validity"
    exit 1
fi