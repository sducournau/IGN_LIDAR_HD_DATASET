#!/bin/bash
# ============================================================================
# Complete Multi-Scale Training Pipeline - Automated Execution
# ============================================================================
# This script orchestrates the entire multi-scale training pipeline:
# 1. Tile selection from unified dataset
# 2. Preprocessing with full features
# 3. Multi-scale patch generation (50m, 100m, 150m)
# 4. Dataset merging
# 5. Model training (ASPRS → LOD2 → LOD3)
#
# Usage:
#   ./run_complete_pipeline.sh [OPTIONS]
#
# Options:
#   --unified-dataset PATH    Path to unified_dataset (default: /mnt/c/Users/Simon/ign/unified_dataset)
#   --output-base PATH        Base output directory (default: /mnt/c/Users/Simon/ign)
#   --phases PHASES           Phases to run: 1,2,3,4,5 or 'all' (default: all)
#   --parallel-patches        Generate patches in parallel (requires more RAM)
#   --skip-existing           Skip already processed files
#   --gpu                     Enable GPU acceleration
#   --dry-run                 Show commands without executing
#   --help                    Show this help message
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
UNIFIED_DATASET="/mnt/c/Users/Simon/ign/unified_dataset"
OUTPUT_BASE="/mnt/c/Users/Simon/ign"
PHASES="all"
PARALLEL_PATCHES=false
SKIP_EXISTING=true
GPU=true
DRY_RUN=false
PROJECT_ROOT="/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unified-dataset)
            UNIFIED_DATASET="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --phases)
            PHASES="$2"
            shift 2
            ;;
        --parallel-patches)
            PARALLEL_PATCHES=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            shift
            ;;
        --gpu)
            GPU=true
            shift
            ;;
        --no-gpu)
            GPU=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -n 25 "$0" | tail -n 20
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
print_header() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║ $1${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

run_command() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $1"
    else
        print_info "Running: $1"
        eval "$1"
        if [ $? -eq 0 ]; then
            print_success "Command completed successfully"
        else
            print_error "Command failed with exit code $?"
            return 1
        fi
    fi
}

# Check if phase should run
should_run_phase() {
    local phase=$1
    if [ "$PHASES" = "all" ]; then
        return 0
    else
        if [[ ",$PHASES," == *",$phase,"* ]]; then
            return 0
        else
            return 1
        fi
    fi
}

# Create directory structure
create_directories() {
    print_header "Creating Directory Structure"
    
    local dirs=(
        "$OUTPUT_BASE/selected_tiles/asprs"
        "$OUTPUT_BASE/selected_tiles/lod2"
        "$OUTPUT_BASE/selected_tiles/lod3"
        "$OUTPUT_BASE/preprocessed/asprs"
        "$OUTPUT_BASE/preprocessed/lod2"
        "$OUTPUT_BASE/preprocessed/lod3"
        "$OUTPUT_BASE/patches/asprs/50m"
        "$OUTPUT_BASE/patches/asprs/100m"
        "$OUTPUT_BASE/patches/asprs/150m"
        "$OUTPUT_BASE/patches/lod2/50m"
        "$OUTPUT_BASE/patches/lod2/100m"
        "$OUTPUT_BASE/patches/lod2/150m"
        "$OUTPUT_BASE/patches/lod3/50m"
        "$OUTPUT_BASE/patches/lod3/100m"
        "$OUTPUT_BASE/patches/lod3/150m"
        "$OUTPUT_BASE/merged_datasets"
        "$OUTPUT_BASE/models/asprs"
        "$OUTPUT_BASE/models/lod2"
        "$OUTPUT_BASE/models/lod3"
        "$OUTPUT_BASE/logs"
        "$OUTPUT_BASE/cache/neighbors"
        "$OUTPUT_BASE/cache/ground_truth"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            run_command "mkdir -p \"$dir\""
        else
            print_info "Directory already exists: $dir"
        fi
    done
    
    print_success "Directory structure created"
}

# Phase 1: Tile Selection
phase1_tile_selection() {
    if ! should_run_phase 1; then
        print_warning "Skipping Phase 1: Tile Selection"
        return
    fi
    
    print_header "Phase 1: Tile Selection from Unified Dataset"
    
    # Analyze unified dataset
    print_info "Step 1.1: Analyzing unified dataset..."
    run_command "python \"$PROJECT_ROOT/scripts/analyze_unified_dataset.py\" \
        --input \"$UNIFIED_DATASET\" \
        --output \"$OUTPUT_BASE/analysis_report.json\""
    
    # Select optimal tiles
    print_info "Step 1.2: Selecting optimal tiles..."
    run_command "python \"$PROJECT_ROOT/scripts/select_optimal_tiles.py\" \
        --input \"$UNIFIED_DATASET\" \
        --analysis \"$OUTPUT_BASE/analysis_report.json\" \
        --output \"$OUTPUT_BASE/selected_tiles\" \
        --asprs-count 100 \
        --lod2-count 80 \
        --lod3-count 60"
    
    # Create symbolic links
    print_info "Step 1.3: Creating tile links..."
    run_command "python \"$PROJECT_ROOT/scripts/create_tile_links.py\" \
        --source \"$UNIFIED_DATASET\" \
        --target \"$OUTPUT_BASE/selected_tiles\" \
        --lists \"$OUTPUT_BASE/selected_tiles/*.txt\""
    
    print_success "Phase 1 completed: Tiles selected and linked"
}

# Phase 2: Preprocessing
phase2_preprocessing() {
    if ! should_run_phase 2; then
        print_warning "Skipping Phase 2: Preprocessing"
        return
    fi
    
    print_header "Phase 2: Preprocessing and Feature Enrichment"
    
    # ASPRS preprocessing
    print_info "Step 2.1: Preprocessing ASPRS tiles..."
    run_command "ign-lidar-hd process \
        --config-file \"$PROJECT_ROOT/configs/multiscale/config_unified_asprs_preprocessing.yaml\""
    
    # LOD2 preprocessing
    print_info "Step 2.2: Preprocessing LOD2 tiles..."
    run_command "ign-lidar-hd process \
        --config-file \"$PROJECT_ROOT/configs/multiscale/config_unified_lod2_preprocessing.yaml\""
    
    # LOD3 preprocessing
    print_info "Step 2.3: Preprocessing LOD3 tiles..."
    run_command "ign-lidar-hd process \
        --config-file \"$PROJECT_ROOT/configs/multiscale/config_unified_lod3_preprocessing.yaml\""
    
    print_success "Phase 2 completed: All tiles preprocessed and enriched"
}

# Phase 3: Multi-scale Patch Generation
phase3_patch_generation() {
    if ! should_run_phase 3; then
        print_warning "Skipping Phase 3: Patch Generation"
        return
    fi
    
    print_header "Phase 3: Multi-Scale Patch Generation"
    
    local scales=(50 100 150)
    local lods=(asprs lod2 lod3)
    
    if [ "$PARALLEL_PATCHES" = true ]; then
        print_info "Generating patches in parallel (requires more RAM)..."
        
        # Generate all patches in parallel
        for lod in "${lods[@]}"; do
            for scale in "${scales[@]}"; do
                (
                    print_info "Generating ${lod} patches at ${scale}m scale..."
                    ign-lidar-hd process \
                        --config-file "$PROJECT_ROOT/configs/multiscale/${lod}/config_${lod}_patches_${scale}m.yaml"
                ) &
            done
        done
        
        wait # Wait for all parallel jobs
        print_success "All patches generated in parallel"
        
    else
        # Sequential generation
        for lod in "${lods[@]}"; do
            print_info "Generating ${lod^^} patches..."
            for scale in "${scales[@]}"; do
                print_info "  → ${scale}m scale..."
                run_command "ign-lidar-hd process \
                    --config-file \"$PROJECT_ROOT/configs/multiscale/${lod}/config_${lod}_patches_${scale}m.yaml\""
            done
        done
        
        print_success "Phase 3 completed: All multi-scale patches generated"
    fi
}

# Phase 4: Dataset Merging
phase4_dataset_merging() {
    if ! should_run_phase 4; then
        print_warning "Skipping Phase 4: Dataset Merging"
        return
    fi
    
    print_header "Phase 4: Multi-Scale Dataset Merging"
    
    # Merge ASPRS datasets
    print_info "Step 4.1: Merging ASPRS multi-scale patches..."
    run_command "python \"$PROJECT_ROOT/examples/merge_multiscale_dataset.py\" \
        --input-dirs \
            \"$OUTPUT_BASE/patches/asprs/50m\" \
            \"$OUTPUT_BASE/patches/asprs/100m\" \
            \"$OUTPUT_BASE/patches/asprs/150m\" \
        --output \"$OUTPUT_BASE/merged_datasets/asprs_multiscale\" \
        --strategy balanced \
        --split 0.7 0.15 0.15 \
        --balance-classes"
    
    # Merge LOD2 datasets
    print_info "Step 4.2: Merging LOD2 multi-scale patches..."
    run_command "python \"$PROJECT_ROOT/examples/merge_multiscale_dataset.py\" \
        --input-dirs \
            \"$OUTPUT_BASE/patches/lod2/50m\" \
            \"$OUTPUT_BASE/patches/lod2/100m\" \
            \"$OUTPUT_BASE/patches/lod2/150m\" \
        --output \"$OUTPUT_BASE/merged_datasets/lod2_multiscale\" \
        --strategy weighted \
        --weights 0.3 0.4 0.3 \
        --split 0.7 0.15 0.15 \
        --balance-classes"
    
    # Merge LOD3 datasets
    print_info "Step 4.3: Merging LOD3 multi-scale patches..."
    run_command "python \"$PROJECT_ROOT/examples/merge_multiscale_dataset.py\" \
        --input-dirs \
            \"$OUTPUT_BASE/patches/lod3/50m\" \
            \"$OUTPUT_BASE/patches/lod3/100m\" \
            \"$OUTPUT_BASE/patches/lod3/150m\" \
        --output \"$OUTPUT_BASE/merged_datasets/lod3_multiscale\" \
        --strategy adaptive \
        --split 0.7 0.15 0.15 \
        --balance-classes \
        --oversample-rare"
    
    print_success "Phase 4 completed: All datasets merged"
}

# Phase 5: Model Training
phase5_training() {
    if ! should_run_phase 5; then
        print_warning "Skipping Phase 5: Model Training"
        return
    fi
    
    print_header "Phase 5: Model Training (ASPRS → LOD2 → LOD3)"
    
    # ASPRS Training
    print_info "Step 5.1: Training ASPRS models..."
    
    print_info "  → PointNet++ ASPRS..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/asprs/pointnet++_asprs.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/asprs_multiscale\" \
        --output \"$OUTPUT_BASE/models/asprs/pointnet++\" \
        --epochs 100 --batch-size 32 --lr 0.001 --patience 15"
    
    print_info "  → Point Transformer ASPRS..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/asprs/point_transformer_asprs.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/asprs_multiscale\" \
        --output \"$OUTPUT_BASE/models/asprs/point_transformer\" \
        --epochs 150 --batch-size 16 --lr 0.0005 --patience 20"
    
    print_info "  → Intelligent Index ASPRS..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/asprs/intelligent_index_asprs.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/asprs_multiscale\" \
        --output \"$OUTPUT_BASE/models/asprs/intelligent_index\" \
        --epochs 120 --batch-size 24 --lr 0.0008 --patience 18"
    
    # LOD2 Training (with ASPRS pretraining)
    print_info "Step 5.2: Training LOD2 models (with ASPRS pretraining)..."
    
    print_info "  → PointNet++ LOD2..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/lod2/pointnet++_lod2.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/lod2_multiscale\" \
        --output \"$OUTPUT_BASE/models/lod2/pointnet++\" \
        --pretrained \"$OUTPUT_BASE/models/asprs/pointnet++/best_model.pth\" \
        --epochs 150 --batch-size 24 --lr 0.0005 --patience 20 --freeze-backbone 10"
    
    print_info "  → Point Transformer LOD2..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/lod2/point_transformer_lod2.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/lod2_multiscale\" \
        --output \"$OUTPUT_BASE/models/lod2/point_transformer\" \
        --pretrained \"$OUTPUT_BASE/models/asprs/point_transformer/best_model.pth\" \
        --epochs 200 --batch-size 12 --lr 0.0003 --patience 25 --freeze-backbone 15"
    
    print_info "  → Intelligent Index LOD2..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/lod2/intelligent_index_lod2.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/lod2_multiscale\" \
        --output \"$OUTPUT_BASE/models/lod2/intelligent_index\" \
        --pretrained \"$OUTPUT_BASE/models/asprs/intelligent_index/best_model.pth\" \
        --epochs 180 --batch-size 16 --lr 0.0004 --patience 22 --freeze-backbone 12"
    
    # LOD3 Training (with LOD2 pretraining)
    print_info "Step 5.3: Training LOD3 models (with LOD2 pretraining)..."
    
    print_info "  → PointNet++ LOD3..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/lod3/pointnet++_lod3.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/lod3_multiscale\" \
        --output \"$OUTPUT_BASE/models/lod3/pointnet++\" \
        --pretrained \"$OUTPUT_BASE/models/lod2/pointnet++/best_model.pth\" \
        --epochs 200 --batch-size 16 --lr 0.0003 --patience 25 \
        --freeze-backbone 15 --class-weights auto"
    
    print_info "  → Point Transformer LOD3..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/lod3/point_transformer_lod3.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/lod3_multiscale\" \
        --output \"$OUTPUT_BASE/models/lod3/point_transformer\" \
        --pretrained \"$OUTPUT_BASE/models/lod2/point_transformer/best_model.pth\" \
        --epochs 250 --batch-size 8 --lr 0.0002 --patience 30 \
        --freeze-backbone 20 --class-weights auto --focal-loss"
    
    print_info "  → Intelligent Index LOD3..."
    run_command "python -m ign_lidar.core.train \
        --config-file \"$PROJECT_ROOT/configs/training/lod3/intelligent_index_lod3.yaml\" \
        --data \"$OUTPUT_BASE/merged_datasets/lod3_multiscale\" \
        --output \"$OUTPUT_BASE/models/lod3/intelligent_index\" \
        --pretrained \"$OUTPUT_BASE/models/lod2/intelligent_index/best_model.pth\" \
        --epochs 220 --batch-size 12 --lr 0.00025 --patience 28 \
        --freeze-backbone 18 --class-weights auto --focal-loss"
    
    print_success "Phase 5 completed: All models trained"
}

# Main execution
main() {
    print_header "Multi-Scale Training Pipeline - Starting"
    
    # Print configuration
    echo -e "${CYAN}Configuration:${NC}"
    echo -e "  Unified Dataset: ${YELLOW}$UNIFIED_DATASET${NC}"
    echo -e "  Output Base: ${YELLOW}$OUTPUT_BASE${NC}"
    echo -e "  Phases: ${YELLOW}$PHASES${NC}"
    echo -e "  Parallel Patches: ${YELLOW}$PARALLEL_PATCHES${NC}"
    echo -e "  Skip Existing: ${YELLOW}$SKIP_EXISTING${NC}"
    echo -e "  GPU: ${YELLOW}$GPU${NC}"
    echo -e "  Dry Run: ${YELLOW}$DRY_RUN${NC}"
    echo ""
    
    # Create directory structure
    create_directories
    
    # Execute phases
    phase1_tile_selection
    phase2_preprocessing
    phase3_patch_generation
    phase4_dataset_merging
    phase5_training
    
    print_header "Pipeline Completed Successfully!"
    
    # Print summary
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    PIPELINE SUMMARY                            ║${NC}"
    echo -e "${GREEN}╠════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║${NC} Preprocessed tiles: $OUTPUT_BASE/preprocessed"
    echo -e "${GREEN}║${NC} Training patches: $OUTPUT_BASE/patches"
    echo -e "${GREEN}║${NC} Merged datasets: $OUTPUT_BASE/merged_datasets"
    echo -e "${GREEN}║${NC} Trained models: $OUTPUT_BASE/models"
    echo -e "${GREEN}║${NC} Logs: $OUTPUT_BASE/logs"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    
    print_info "Next steps:"
    echo "  1. Evaluate models: python -m ign_lidar.core.evaluate ..."
    echo "  2. Classify LAZ files: python -m ign_lidar.core.classify ..."
    echo "  3. Check training logs: $OUTPUT_BASE/logs/"
}

# Run main function
main
