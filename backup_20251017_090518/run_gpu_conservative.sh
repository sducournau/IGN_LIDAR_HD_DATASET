#!/bin/bash
# IGN LiDAR HD - GPU Conservative Pipeline 
# Paramètres GPU ultra-conservatifs pour éviter tout fallback CPU
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
OUTPUT_DIR="/mnt/d/ign/preprocessed/asprs/enriched_gpu_conservative"
CACHE_DIR="/mnt/d/ign/cache"

# Create output and cache directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Display configuration
echo "======================================================================"
echo "IGN LiDAR HD - GPU Conservative Pipeline"
echo "======================================================================"
echo "Input:              $INPUT_DIR"
echo "Output:             $OUTPUT_DIR"
echo "Config:             $CONFIG"
echo "Mode:               ENRICHED_ONLY + GPU CONSERVATIVE"
echo "GPU Batch Size:     2M points (ultra-conservative)"
echo "VRAM Target:        70% (ultra-conservative)"
echo "Mixed Precision:    DISABLED"
echo "Streams:            2 (minimal)"
echo "Fallback CPU:       Enabled if GPU fails"
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

# Check GPU status
echo "GPU Status avant traitement:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
echo ""

# Record start time
START_TIME=$(date +%s)

# Start GPU monitoring
nvidia-smi dmon -s pucvmet -d 5 -f gpu_usage_conservative.log &
GPU_MONITOR_PID=$!

# Run with ULTRA-CONSERVATIVE GPU parameters
log_info "Starting GPU Conservative pipeline..."

ign-lidar-hd process \
    --config-file "$CONFIG" \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    cache_dir="$CACHE_DIR" \
    \
    processor.use_gpu=true \
    processor.architecture=direct \
    processor.processing_mode=enriched_only \
    processor.batch_size=128 \
    processor.generate_patches=false \
    processor.patch_size=null \
    processor.patch_overlap=null \
    processor.num_points=null \
    processor.direct_enrichment=true \
    processor.use_stitching=false \
    processor.buffer_size=0.0 \
    processor.apply_reclassification_inline=false \
    processor.reclassification.enabled=false \
    \
    processing.mode=enriched_only \
    processing.architecture=direct \
    processing.generate_patches=false \
    processing.use_gpu=true \
    processing.num_workers=1 \
    \
    features.use_gpu=true \
    features.gpu_batch_size=2000000 \
    features.vram_utilization_target=0.7 \
    features.num_cuda_streams=2 \
    features.adaptive_chunk_sizing=true \
    features.enable_async_processing=false \
    features.memory_pool_enabled=false \
    features.k_neighbors=10 \
    features.search_radius=0.8 \
    features.use_nir=false \
    features.use_infrared=false \
    features.compute_ndvi=false \
    features.compute_verticality=false \
    features.compute_horizontality=false \
    features.compute_sphericity=false \
    features.compute_curvature=false \
    features.compute_linearity=false \
    features.gpu_optimization.enable_mixed_precision=false \
    features.gpu_optimization.adaptive_memory_management=true \
    features.gpu_optimization.enable_memory_pooling=false \
    features.gpu_optimization.enable_tensor_cores=false \
    features.cpu_optimization.enabled=true \
    features.cpu_optimization.enable_numba_acceleration=true \
    features.cpu_optimization.enable_parallel_processing=true \
    \
    preprocess.enabled=false \
    stitching.enabled=false \
    \
    ground_truth.enabled=true \
    ground_truth.update_classification=true \
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    ground_truth.optimization.force_method=strtree \
    ground_truth.optimization.enable_monitoring=false \
    ground_truth.optimization.enable_auto_tuning=false \
    \
    data_sources.bd_topo_enabled=true \
    data_sources.bd_topo_buildings=true \
    data_sources.bd_topo_roads=true \
    data_sources.bd_topo_water=true \
    data_sources.bd_topo_vegetation=false \
    data_sources.cadastre_enabled=false \
    data_sources.bd_foret_enabled=false \
    data_sources.rpg_enabled=false \
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

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ GPU Conservative Pipeline completed successfully!"
    echo "======================================================================"
    log_info "Processing completed in ${DURATION_MIN}m ${DURATION_SEC}s"
    log_info "Output directory: $OUTPUT_DIR"
    echo ""
    
    # GPU Status après
    echo "GPU Status après traitement:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    echo ""
    
    if [ -d "$OUTPUT_DIR" ]; then
        ENRICHED_COUNT=$(find "$OUTPUT_DIR" -name "*enriched*.laz" | wc -l)
        PATCH_COUNT=$(find "$OUTPUT_DIR" -name "*patch*.laz" | wc -l 2>/dev/null || echo "0")
        
        echo "Results:"
        echo "✅ Enriched LAZ files: $ENRICHED_COUNT"
        if [ "$PATCH_COUNT" -gt 0 ]; then
            echo "⚠️  WARNING: Patches still generated: $PATCH_COUNT (should be 0!)"
        else
            echo "✅ Patches: 0 (perfect!)"
        fi
        
        echo ""
        echo "Sample output files:"
        find "$OUTPUT_DIR" -name "*.laz" | head -5
    fi
    
    echo ""
    echo "GPU Performance Analysis:"
    if [ -f "gpu_usage_conservative.log" ]; then
        echo "GPU utilization moyenne:"
        tail -n +3 gpu_usage_conservative.log | awk 'NF>=4 && $4 ~ /^[0-9]+$/ {sum+=$4; count++} END {
            if(count>0) printf "%.1f%%\n", sum/count; else print "N/A"
        }'
        echo "Température GPU max:"
        tail -n +3 gpu_usage_conservative.log | awk 'NF>=6 && $6 ~ /^[0-9]+$/ {if($6>max) max=$6} END {print max "°C"}'
        echo "Log complet: gpu_usage_conservative.log"
    fi
    
    echo ""
    echo "Conservative GPU Settings Used:"
    echo "✅ Batch size: 2M points (ultra-conservative)"
    echo "✅ VRAM target: 70% (ultra-conservative)"  
    echo "✅ Mixed precision: OFF (stability)"
    echo "✅ Tensor cores: OFF (stability)"
    echo "✅ Memory pooling: OFF (stability)"
    echo "✅ CPU fallback: ENABLED"
    
    echo ""
    echo "Performance:"
    if [ $DURATION -gt 0 ]; then
        echo "Processing rate: $(echo "scale=1; $FILE_COUNT * 60 / $DURATION" | bc 2>/dev/null || echo "N/A") files/hour"
        echo "Time per tile: $(echo "scale=1; $DURATION / $FILE_COUNT" | bc 2>/dev/null || echo "N/A") seconds/tile"
    fi
    
else
    echo ""
    echo "======================================================================"
    echo "❌ GPU Conservative Pipeline failed!"
    echo "======================================================================"
    log_error "Processing failed after ${DURATION_MIN}m ${DURATION_SEC}s"
    
    echo ""
    echo "GPU Status during failure:"
    if [ -f "gpu_usage_conservative.log" ]; then
        echo "Dernières métriques GPU:"
        tail -10 gpu_usage_conservative.log
    fi
    
    echo ""
    echo "Troubleshooting suggestions:"
    echo "1. Check GPU temperature: nvidia-smi"
    echo "2. Check available GPU memory"
    echo "3. Try CPU-only mode: features.use_gpu=false"
    echo "4. Check CUDA/CuPy installation"
    
    exit 1
fi