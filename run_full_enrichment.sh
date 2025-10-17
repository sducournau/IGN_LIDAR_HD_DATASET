#!/bin/bash
# Launch IGN LiDAR HD Pipeline with RTX 4080 Super Optimized Configuration
# Date: October 16, 2025
# Input: D:\ign\selected_tiles\asprs\tiles
# Output: D:\ign\preprocessed\asprs
# Hardware: NVIDIA RTX 4080 Super (16GB VRAM)

# Function to log with timestamp
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Function to check GPU status
check_gpu_status() {
    log_info "Checking GPU status..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
        echo ""
        # Check if GPU is available for CUDA
        if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
            python -c "import torch; print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null
        fi
        # Check CuPy availability for GPU optimizations
        if python -c "import cupy; print(f'CuPy version: {cupy.__version__}')" 2>/dev/null; then
            log_info "CuPy available - GPU optimizations will be used"
        else
            log_info "CuPy not available - consider installing for maximum performance"
        fi
    else
        log_error "nvidia-smi not found - GPU monitoring unavailable"
    fi
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

# Configuration - Updated to use RTX 4080 optimized config
CONFIG="configs/config_asprs_rtx4080.yaml"
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/mnt/d/ign/preprocessed/asprs/enriched_tiles"
CACHE_DIR="/mnt/d/ign/cache"

# Validate paths are accessible (convert Windows paths if needed)
if [ ! -d "$INPUT_DIR" ] && [ -d "/mnt/d/ign/selected_tiles/asprs/tiles" ]; then
    INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
elif [ ! -d "$INPUT_DIR" ] && [ -d "D:/ign/selected_tiles/asprs/tiles" ]; then
    INPUT_DIR="D:/ign/selected_tiles/asprs/tiles"
fi

if [ ! -d "$(dirname "$OUTPUT_DIR")" ]; then
    if [ -d "/mnt/d/ign/preprocessed" ]; then
        OUTPUT_DIR="/mnt/d/ign/preprocessed/asprs/enriched_tiles"
    elif [ -d "D:/ign/preprocessed" ]; then
        OUTPUT_DIR="D:/ign/preprocessed/asprs/enriched_tiles"
    fi
fi

# Create output and cache directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Check GPU status and optimization capabilities
check_gpu_status

# Apply optimized ground truth computation
log_info "Applying optimized ground truth computation..."
python3 -c "
from ign_lidar.optimization.auto_select import auto_optimize
auto_optimize()
print('✅ Ground truth optimization applied')
"

# Display configuration
echo "======================================================================"
echo "IGN LiDAR HD - RTX 4080 Super Optimized Enrichment Pipeline"
echo "======================================================================"
echo "Input:        $INPUT_DIR"
echo "Output:       $OUTPUT_DIR"
echo "Cache:        $CACHE_DIR"
echo "Config:       $CONFIG"
echo "GPU:          RTX 4080 Super (16GB VRAM)"
echo "Optimization: Optimized ground truth computation"
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

echo ""

# Start GPU monitoring in background if available
if command -v nvidia-smi &> /dev/null; then
    log_info "Starting GPU monitoring..."
    # Start monitoring with more frequent sampling (every 1 second)
    nvidia-smi dmon -s pucvmet -i 0 -d 1 -f nvidia_gpu_usage.log &
    GPU_MONITOR_PID=$!
    trap "kill $GPU_MONITOR_PID 2>/dev/null" EXIT
    # Give monitoring a moment to start
    sleep 2
fi

# Record start time for performance tracking
START_TIME=$(date +%s)

# Run the pipeline with RTX 4080 optimized parameters
# Note: num_workers=1 is required for GPU processing due to CUDA context limitations
log_info "Starting IGN LiDAR HD processing with RTX 4080 configuration..."
ign-lidar-hd process \
    --config-file "$CONFIG" \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    cache_dir="$CACHE_DIR" \
    processor.use_gpu=true \
    ground_truth.enabled=true \
    ground_truth.update_classification=true \
    ground_truth.use_ndvi=true \
    data_sources.bd_topo.enabled=true \
    data_sources.cadastre.enabled=true \
    log_level=INFO

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# Stop GPU monitoring
if [ ! -z "$GPU_MONITOR_PID" ]; then
    kill $GPU_MONITOR_PID 2>/dev/null
fi

# Check exit status and provide detailed feedback
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ RTX 4080 Super Pipeline completed successfully!"
    echo "======================================================================"
    log_info "Processing completed in ${DURATION_MIN}m ${DURATION_SEC}s"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Cache directory: $CACHE_DIR"
    echo ""
    echo "Output files:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -lh "$OUTPUT_DIR" | head -20
        TOTAL_FILES=$(find "$OUTPUT_DIR" -type f | wc -l)
        if [ $TOTAL_FILES -gt 20 ]; then
            echo "... and $((TOTAL_FILES - 20)) more files"
        fi
    fi
    echo ""
    echo "Performance summary:"
    if [ -f "nvidia_gpu_usage.log" ]; then
        echo "GPU usage log: nvidia_gpu_usage.log"
        echo "Average GPU utilization:"
        if command -v awk &> /dev/null; then
            # Better parsing: skip header lines and handle various formats
            GPU_UTIL=$(awk 'NR>2 && NF>=4 && $4 ~ /^[0-9]+$/ {sum+=$4; count++} END {
                if(count>0) 
                    printf "%.1f%%\n", sum/count
                else 
                    print "N/A (no valid data points)"
            }' nvidia_gpu_usage.log 2>/dev/null)
            echo "$GPU_UTIL"
            
            # Show data points count for debugging
            DATA_POINTS=$(awk 'NR>2 && NF>=4 && $4 ~ /^[0-9]+$/ {count++} END {print count+0}' nvidia_gpu_usage.log 2>/dev/null || echo "0")
            echo "Data points collected: $DATA_POINTS"
        fi
    else
        echo "GPU usage log: Not available"
    fi
    echo "Files processed: $FILE_COUNT"
    echo "Processing rate: $(echo "scale=2; $FILE_COUNT / ($DURATION / 60)" | bc 2>/dev/null || echo "N/A") files/minute"
else
    echo ""
    echo "======================================================================"
    echo "❌ Pipeline failed with errors"
    echo "======================================================================"
    log_error "Processing failed after ${DURATION_MIN}m ${DURATION_SEC}s"
    log_error "Check the logs above for error details"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check GPU memory usage: nvidia-smi"
    echo "2. Verify input files are valid LiDAR data"
    echo "3. Ensure adequate disk space in output directory"
    echo "4. Check CUDA/CuPy installation if GPU errors occur"
    exit 1
fi
