#!/bin/bash
# ============================================================================
# Multi-Scale Training Pipeline - Quick Start Script
# ============================================================================
# This script automates the multi-scale training data generation process
# for LOD3 architectural classification with hybrid models.
#
# Usage:
#   ./run_multiscale_training.sh [--parallel] [--skip-merge]
#
# Options:
#   --parallel     Generate all scales in parallel (requires ~64GB RAM)
#   --skip-merge   Skip dataset merging step
#   --dry-run      Show commands without executing
# ============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET"
PARALLEL=false
SKIP_MERGE=false
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel     Generate all scales in parallel (requires ~64GB RAM)"
            echo "  --skip-merge   Skip dataset merging step"
            echo "  --dry-run      Show commands without executing"
            echo "  --help, -h     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

# Helper function to run or show commands
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${BLUE}[DRY RUN]${NC} $@"
    else
        echo -e "${GREEN}[RUNNING]${NC} $@"
        eval "$@"
    fi
}

# Helper function to check if output directory exists
check_output() {
    local dir=$1
    local scale=$2
    
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        echo -e "${YELLOW}⚠️  Warning: $dir already contains files${NC}"
        read -p "Do you want to overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}ℹ️  Skipping ${scale} generation${NC}"
            return 1
        fi
        run_cmd "rm -rf $dir/*"
    fi
    return 0
}

# Print banner
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}   Multi-Scale Training Pipeline for LOD3 Hybrid Model${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "Project Root: ${GREEN}$PROJECT_ROOT${NC}"
echo -e "Parallel Mode: ${GREEN}$PARALLEL${NC}"
echo -e "Skip Merge: ${GREEN}$SKIP_MERGE${NC}"
echo -e "Dry Run: ${GREEN}$DRY_RUN${NC}"
echo ""

# Change to project directory
cd "$PROJECT_ROOT"

# Check if configs exist
for config in config_lod3_training_50m.yaml config_lod3_training_100m.yaml config_lod3_training_150m.yaml; do
    if [ ! -f "examples/$config" ]; then
        echo -e "${RED}❌ Error: Configuration file not found: examples/$config${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ All configuration files found${NC}"
echo ""

# ============================================================================
# Phase 1: Generate 50m patches (Fine Details)
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 1: Generating 50m patches (Fine architectural details)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Estimated time: ${YELLOW}4-6 hours${NC}"
echo -e "Expected output: ${YELLOW}~15,000 patches (~12GB)${NC}"
echo ""

if check_output "/mnt/c/Users/Simon/ign/patches_50m" "50m"; then
    if [ "$PARALLEL" = true ]; then
        run_cmd "ign-lidar-hd process --config-file examples/config_lod3_training_50m.yaml &"
        PID_50M=$!
        echo -e "${GREEN}✓ Started 50m generation in background (PID: $PID_50M)${NC}"
    else
        run_cmd "ign-lidar-hd process --config-file examples/config_lod3_training_50m.yaml"
        echo -e "${GREEN}✓ 50m patches generated successfully${NC}"
    fi
fi
echo ""

# ============================================================================
# Phase 2: Generate 100m patches (Balanced Context)
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 2: Generating 100m patches (Balanced detail/context)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Estimated time: ${YELLOW}3-4 hours${NC}"
echo -e "Expected output: ${YELLOW}~8,000 patches (~15GB)${NC}"
echo ""

if check_output "/mnt/c/Users/Simon/ign/patches_100m" "100m"; then
    if [ "$PARALLEL" = true ]; then
        run_cmd "ign-lidar-hd process --config-file examples/config_lod3_training_100m.yaml &"
        PID_100M=$!
        echo -e "${GREEN}✓ Started 100m generation in background (PID: $PID_100M)${NC}"
    else
        run_cmd "ign-lidar-hd process --config-file examples/config_lod3_training_100m.yaml"
        echo -e "${GREEN}✓ 100m patches generated successfully${NC}"
    fi
fi
echo ""

# ============================================================================
# Phase 3: Generate 150m patches (Full Context)
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 3: Generating 150m patches (Full building context)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "Estimated time: ${YELLOW}2-3 hours${NC}"
echo -e "Expected output: ${YELLOW}~5,000 patches (~18GB)${NC}"
echo ""

if check_output "/mnt/c/Users/Simon/ign/patches_150m" "150m"; then
    if [ "$PARALLEL" = true ]; then
        run_cmd "ign-lidar-hd process --config-file examples/config_lod3_training.yaml &"
        PID_150M=$!
        echo -e "${GREEN}✓ Started 150m generation in background (PID: $PID_150M)${NC}"
    else
        run_cmd "ign-lidar-hd process --config-file examples/config_lod3_training.yaml"
        echo -e "${GREEN}✓ 150m patches generated successfully${NC}"
    fi
fi
echo ""

# Wait for parallel jobs to complete
if [ "$PARALLEL" = true ] && [ "$DRY_RUN" = false ]; then
    echo -e "${YELLOW}⏳ Waiting for all parallel jobs to complete...${NC}"
    echo -e "   Monitor progress with: ${BLUE}tail -f /tmp/ign_lidar_*.log${NC}"
    echo ""
    
    if [ ! -z "$PID_50M" ]; then
        echo -e "   50m generation (PID: $PID_50M)..."
        wait $PID_50M && echo -e "   ${GREEN}✓ 50m completed${NC}" || echo -e "   ${RED}✗ 50m failed${NC}"
    fi
    
    if [ ! -z "$PID_100M" ]; then
        echo -e "   100m generation (PID: $PID_100M)..."
        wait $PID_100M && echo -e "   ${GREEN}✓ 100m completed${NC}" || echo -e "   ${RED}✗ 100m failed${NC}"
    fi
    
    if [ ! -z "$PID_150M" ]; then
        echo -e "   150m generation (PID: $PID_150M)..."
        wait $PID_150M && echo -e "   ${GREEN}✓ 150m completed${NC}" || echo -e "   ${RED}✗ 150m failed${NC}"
    fi
    
    echo -e "${GREEN}✓ All patch generation completed${NC}"
    echo ""
fi

# ============================================================================
# Phase 4: Merge Multi-Scale Datasets
# ============================================================================
if [ "$SKIP_MERGE" = false ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Phase 4: Merging Multi-Scale Datasets${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "This will combine all scales into train/val/test splits"
    echo ""
    
    run_cmd "python examples/merge_multiscale_dataset.py \
        --output patches_multiscale \
        --weights 0.4 0.3 0.3 \
        --train-split 0.7 \
        --val-split 0.15 \
        --test-split 0.15"
    
    echo -e "${GREEN}✓ Multi-scale dataset merged successfully${NC}"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ Multi-Scale Training Pipeline Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}📊 Generated Datasets:${NC}"

if [ -d "/mnt/c/Users/Simon/ign/patches_50m" ]; then
    COUNT_50M=$(find /mnt/c/Users/Simon/ign/patches_50m -name "*.npz" 2>/dev/null | wc -l)
    SIZE_50M=$(du -sh /mnt/c/Users/Simon/ign/patches_50m 2>/dev/null | cut -f1)
    echo -e "   ✓ 50m patches:  ${YELLOW}${COUNT_50M} files (${SIZE_50M})${NC}"
fi

if [ -d "/mnt/c/Users/Simon/ign/patches_100m" ]; then
    COUNT_100M=$(find /mnt/c/Users/Simon/ign/patches_100m -name "*.npz" 2>/dev/null | wc -l)
    SIZE_100M=$(du -sh /mnt/c/Users/Simon/ign/patches_100m 2>/dev/null | cut -f1)
    echo -e "   ✓ 100m patches: ${YELLOW}${COUNT_100M} files (${SIZE_100M})${NC}"
fi

if [ -d "/mnt/c/Users/Simon/ign/patches_150m" ]; then
    COUNT_150M=$(find /mnt/c/Users/Simon/ign/patches_150m -name "*.npz" 2>/dev/null | wc -l)
    SIZE_150M=$(du -sh /mnt/c/Users/Simon/ign/patches_150m 2>/dev/null | cut -f1)
    echo -e "   ✓ 150m patches: ${YELLOW}${COUNT_150M} files (${SIZE_150M})${NC}"
fi

if [ "$SKIP_MERGE" = false ] && [ -d "/mnt/c/Users/Simon/ign/patches_multiscale" ]; then
    echo ""
    echo -e "${GREEN}📦 Merged Multi-Scale Dataset:${NC}"
    COUNT_TRAIN=$(find /mnt/c/Users/Simon/ign/patches_multiscale/train -name "*.npz" 2>/dev/null | wc -l)
    COUNT_VAL=$(find /mnt/c/Users/Simon/ign/patches_multiscale/val -name "*.npz" 2>/dev/null | wc -l)
    COUNT_TEST=$(find /mnt/c/Users/Simon/ign/patches_multiscale/test -name "*.npz" 2>/dev/null | wc -l)
    
    echo -e "   ✓ Train: ${YELLOW}${COUNT_TRAIN} patches${NC}"
    echo -e "   ✓ Val:   ${YELLOW}${COUNT_VAL} patches${NC}"
    echo -e "   ✓ Test:  ${YELLOW}${COUNT_TEST} patches${NC}"
fi

echo ""
echo -e "${GREEN}🎯 Next Steps:${NC}"
echo -e "   1. Inspect dataset: ${BLUE}python examples/inspect_patches.py${NC}"
echo -e "   2. Start training: ${BLUE}python train_multiscale_hybrid.py${NC}"
echo -e "   3. Monitor metrics: ${BLUE}tensorboard --logdir logs/${NC}"
echo ""
echo -e "${YELLOW}📚 Documentation:${NC}"
echo -e "   See ${BLUE}examples/MULTI_SCALE_TRAINING_STRATEGY.md${NC} for training strategies"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
