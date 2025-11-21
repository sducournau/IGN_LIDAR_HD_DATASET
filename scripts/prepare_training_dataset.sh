#!/bin/bash
# ============================================================================
# IGN LiDAR HD - Prepare Training Dataset
# ============================================================================
# Script to prepare training dataset for PointNet++/Transformer hybrid
#
# Usage:
#   ./scripts/prepare_training_dataset.sh
#
# Requirements:
#   - IGN LiDAR HD library installed (pip install ign-lidar-hd)
#   - CUDA-capable GPU (RTX 4080 or similar)
#   - Input tiles in /home/simon/ign_data/unified_dataset_rgb
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}IGN LiDAR HD - Training Dataset Preparation${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Paths
INPUT_DIR="/home/simon/ign_data/unified_dataset_rgb"
OUTPUT_DIR="/home/simon/ign_data/training_patches_50m_32k"
CONFIG_FILE="examples/config_pointnet_transformer_hybrid_training.yaml"
LOG_FILE="training_dataset_preparation.log"

# Optional: Set BD TOPO path (if available)
# BD_TOPO_PATH="/path/to/BDTOPO"

echo -e "Configuration:"
echo -e "  Input:  ${INPUT_DIR}"
echo -e "  Output: ${OUTPUT_DIR}"
echo -e "  Config: ${CONFIG_FILE}"
echo -e "  Log:    ${LOG_FILE}"
echo ""

# ============================================================================
# Pre-flight checks
# ============================================================================

echo -e "${YELLOW}Running pre-flight checks...${NC}"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}ERROR: Input directory not found: ${INPUT_DIR}${NC}"
    exit 1
fi

# Count LAZ files
NUM_TILES=$(find "$INPUT_DIR" -name "*.laz" -o -name "*.las" | wc -l)
if [ "$NUM_TILES" -eq 0 ]; then
    echo -e "${RED}ERROR: No LAZ/LAS files found in ${INPUT_DIR}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Found ${NUM_TILES} tiles${NC}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}ERROR: Config file not found: ${CONFIG_FILE}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Config file found${NC}"

# Check if ign-lidar-hd is installed
if ! command -v ign-lidar-hd &> /dev/null; then
    echo -e "${RED}ERROR: ign-lidar-hd not found. Please install:${NC}"
    echo -e "  pip install ign-lidar-hd"
    exit 1
fi
echo -e "${GREEN}✓ ign-lidar-hd installed${NC}"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ GPU detected: ${GPU_NAME} (${GPU_MEM})${NC}"
else
    echo -e "${YELLOW}⚠ No GPU detected. Processing will use CPU (slower).${NC}"
    echo -e "  Consider setting processor.use_gpu=false in config."
fi

echo ""

# ============================================================================
# Estimate processing time and disk space
# ============================================================================

echo -e "${YELLOW}Estimating processing requirements...${NC}"

# Average processing time per tile (minutes)
AVG_TIME_PER_TILE=7

# Estimated disk space per tile (MB)
AVG_DISK_PER_TILE=2500

# Calculate estimates
TOTAL_TIME_MIN=$((NUM_TILES * AVG_TIME_PER_TILE))
TOTAL_TIME_HOURS=$((TOTAL_TIME_MIN / 60))
TOTAL_DISK_MB=$((NUM_TILES * AVG_DISK_PER_TILE))
TOTAL_DISK_GB=$((TOTAL_DISK_MB / 1024))

echo -e "  Tiles to process: ${NUM_TILES}"
echo -e "  Estimated time:   ${TOTAL_TIME_HOURS}h ${TOTAL_TIME_MIN}m (${AVG_TIME_PER_TILE}min/tile)"
echo -e "  Estimated disk:   ${TOTAL_DISK_GB} GB (~${AVG_DISK_PER_TILE}MB/tile)"
echo ""

# Check available disk space
AVAILABLE_SPACE=$(df -BG "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
if [ -z "$AVAILABLE_SPACE" ]; then
    # If output dir doesn't exist, check parent
    AVAILABLE_SPACE=$(df -BG "$(dirname "$OUTPUT_DIR")" | awk 'NR==2 {print $4}' | sed 's/G//')
fi

if [ "$AVAILABLE_SPACE" -lt "$TOTAL_DISK_GB" ]; then
    echo -e "${RED}WARNING: Insufficient disk space!${NC}"
    echo -e "  Available: ${AVAILABLE_SPACE} GB"
    echo -e "  Required:  ${TOTAL_DISK_GB} GB"
    echo -e ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Sufficient disk space available: ${AVAILABLE_SPACE} GB${NC}"
fi

echo ""

# ============================================================================
# Confirm and start processing
# ============================================================================

echo -e "${YELLOW}Ready to start processing.${NC}"
echo -e "Press CTRL+C to cancel or wait 5 seconds to continue..."
sleep 5

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}Starting dataset preparation...${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start timestamp
START_TIME=$(date +%s)
START_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "Started at: $START_DATETIME" | tee "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# Run processing
# ============================================================================

# Run ign-lidar-hd with the config
ign-lidar-hd process \
    -c "$CONFIG_FILE" \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Post-processing summary
# ============================================================================

END_TIME=$(date +%s)
END_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MIN=$(((ELAPSED % 3600) / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo "" | tee -a "$LOG_FILE"
echo -e "${GREEN}============================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}Processing completed${NC}" | tee -a "$LOG_FILE"
echo -e "${GREEN}============================================================================${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Started:  $START_DATETIME" | tee -a "$LOG_FILE"
echo "Finished: $END_DATETIME" | tee -a "$LOG_FILE"
echo "Elapsed:  ${ELAPSED_HOURS}h ${ELAPSED_MIN}m ${ELAPSED_SEC}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Processing completed successfully!${NC}" | tee -a "$LOG_FILE"
    
    # Count output files
    if [ -d "$OUTPUT_DIR/patches" ]; then
        NUM_PATCHES=$(find "$OUTPUT_DIR/patches" -name "*.laz" | wc -l)
        echo -e "  Generated patches: ${NUM_PATCHES}" | tee -a "$LOG_FILE"
    fi
    
    # Disk usage
    DISK_USED=$(du -sh "$OUTPUT_DIR" | cut -f1)
    echo -e "  Disk usage: ${DISK_USED}" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo -e "Output location: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    echo -e "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    echo -e "${GREEN}Next steps:${NC}" | tee -a "$LOG_FILE"
    echo -e "  1. Verify dataset: python scripts/validate_training_dataset.py" | tee -a "$LOG_FILE"
    echo -e "  2. Train model: python train.py --data ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    
else
    echo -e "${RED}✗ Processing failed with exit code ${EXIT_CODE}${NC}" | tee -a "$LOG_FILE"
    echo -e "  Check log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

echo ""
echo -e "${GREEN}============================================================================${NC}"
