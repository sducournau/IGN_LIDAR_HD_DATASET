#!/bin/bash
# Script de validation de l'accélération GPU - IGN LiDAR HD
# Vérifie l'utilisation GPU et mesure les performances
# Date: October 17, 2025

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/configs/config.yaml"
TEST_INPUT="$PROJECT_DIR/data/test_integration"
TEST_OUTPUT="/tmp/ign_gpu_test_$(date +%s)"

# Fonction de logging
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Fonction de nettoyage
cleanup() {
    if [ -d "$TEST_OUTPUT" ]; then
        log_info "Cleaning up test output: $TEST_OUTPUT"
        rm -rf "$TEST_OUTPUT"
    fi
}

# Trap pour nettoyer en cas d'interruption
trap cleanup EXIT

echo "======================================================================"
echo "🚀 IGN LiDAR HD - GPU Acceleration Validation"
echo "======================================================================"
echo "Project Directory: $PROJECT_DIR"
echo "Config File: $CONFIG"
echo "Test Input: $TEST_INPUT"
echo "Test Output: $TEST_OUTPUT"
echo "======================================================================"

# 1. Vérifier la disponibilité GPU
echo ""
echo "📊 1. GPU Hardware Check"
echo "---------------------"

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. GPU drivers not installed?"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ $GPU_COUNT -eq 0 ]; then
    log_error "No NVIDIA GPUs detected"
    exit 1
fi

log_info "GPU(s) detected: $GPU_COUNT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 2. Vérifier l'environnement conda
echo ""
echo "🐍 2. Environment Check"
echo "----------------------"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

if [ $? -ne 0 ]; then
    log_error "Failed to activate ign_gpu environment"
    exit 1
fi

log_info "Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"

# Vérifier les packages GPU
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 3. Vérifier la configuration
echo ""
echo "⚙️  3. Configuration Validation"
echo "------------------------------"

if [ ! -f "$CONFIG" ]; then
    log_error "Configuration file not found: $CONFIG"
    exit 1
fi

log_info "Configuration file found: $CONFIG"

# Extraire les paramètres GPU clés
GPU_BATCH_SIZE=$(grep -E "gpu_batch_size:|batch_size:" "$CONFIG" | head -1 | awk '{print $2}' | tr -d '_,')
VRAM_TARGET=$(grep "vram_utilization_target:" "$CONFIG" | awk '{print $2}')
ACCELERATION_MODE=$(grep "acceleration_mode:" "$CONFIG" | awk '{print $2}')
FORCE_METHOD=$(grep "force_method:" "$CONFIG" | grep -v "#" | awk '{print $2}' | tr -d '"')

echo "GPU Batch Size: ${GPU_BATCH_SIZE:-"Not found"}"
echo "VRAM Utilization Target: ${VRAM_TARGET:-"Not found"}"
echo "Acceleration Mode: ${ACCELERATION_MODE:-"Not found"}"
echo "Ground Truth Force Method: ${FORCE_METHOD:-"Not found"}"

# Vérifier les paramètres optimaux
if [ "$ACCELERATION_MODE" = "cpu" ]; then
    log_warning "⚠️  Acceleration mode is CPU - GPU acceleration disabled!"
fi

if [ "$FORCE_METHOD" = "auto" ] || [ "$FORCE_METHOD" = "strtree" ]; then
    log_warning "⚠️  Ground truth force method may fallback to CPU"
fi

# 4. Rechercher un fichier de test
echo ""
echo "📁 4. Test Data Search"
echo "--------------------"

TEST_FILE=""
if [ -d "$TEST_INPUT" ]; then
    TEST_FILE=$(find "$TEST_INPUT" -name "*.laz" -o -name "*.las" | head -1)
fi

if [ -z "$TEST_FILE" ]; then
    # Chercher dans d'autres répertoires
    for dir in "$PROJECT_DIR/data" "/mnt/d/ign" "/tmp"; do
        if [ -d "$dir" ]; then
            TEST_FILE=$(find "$dir" -name "*.laz" -o -name "*.las" 2>/dev/null | head -1)
            if [ -n "$TEST_FILE" ]; then
                break
            fi
        fi
    done
fi

if [ -z "$TEST_FILE" ]; then
    log_warning "No LiDAR test files found. Skipping performance test."
    echo ""
    echo "✅ GPU validation completed (hardware check only)"
    echo "To run full performance test, place a .laz/.las file in:"
    echo "  - $TEST_INPUT"
    echo "  - $PROJECT_DIR/data/"
    exit 0
fi

log_info "Test file found: $TEST_FILE"
FILE_SIZE=$(ls -lh "$TEST_FILE" | awk '{print $5}')
log_info "File size: $FILE_SIZE"

# 5. Test de performance
echo ""
echo "🚀 5. Performance Test"
echo "--------------------"

mkdir -p "$TEST_OUTPUT"

# Démarrer le monitoring GPU en arrière-plan
GPU_LOG="$TEST_OUTPUT/gpu_usage.log"
nvidia-smi dmon -s u -d 1 -o DT > "$GPU_LOG" &
GPU_MONITOR_PID=$!

log_info "Starting GPU monitoring (PID: $GPU_MONITOR_PID)"
log_info "GPU log: $GPU_LOG"

# Créer un input temporaire avec un seul fichier
TEST_INPUT_SINGLE="$TEST_OUTPUT/input"
mkdir -p "$TEST_INPUT_SINGLE"
cp "$TEST_FILE" "$TEST_INPUT_SINGLE/"

# Lancer le processing avec monitoring
log_info "Starting processing with GPU acceleration..."
START_TIME=$(date +%s)

ign-lidar-hd process \
    --config-file "$CONFIG" \
    input_dir="$TEST_INPUT_SINGLE" \
    output_dir="$TEST_OUTPUT/output" \
    cache_dir="$TEST_OUTPUT/cache" \
    verbose=true

PROCESS_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Arrêter le monitoring
kill $GPU_MONITOR_PID 2>/dev/null
sleep 2

# 6. Analyser les résultats
echo ""
echo "📊 6. Performance Analysis"
echo "------------------------"

if [ $PROCESS_EXIT_CODE -eq 0 ]; then
    log_info "✅ Processing completed successfully"
    log_info "Duration: ${DURATION}s"
    
    # Analyser l'utilisation GPU
    if [ -f "$GPU_LOG" ] && [ -s "$GPU_LOG" ]; then
        echo ""
        echo "GPU Utilization Analysis:"
        echo "------------------------"
        
        # Exclure la première ligne (header) et calculer les stats
        tail -n +2 "$GPU_LOG" | awk '
        {
            if (NF >= 3 && $3 ~ /^[0-9]+$/) {
                util[NR] = $3
                count++
                sum += $3
                if ($3 > max) max = $3
                if (min == 0 || $3 < min) min = $3
            }
        }
        END {
            if (count > 0) {
                avg = sum / count
                printf "GPU Utilization: Min=%d%%, Max=%d%%, Avg=%.1f%%\n", min, max, avg
                if (avg > 80) {
                    print "✅ EXCELLENT: High GPU utilization (>80%)"
                } else if (avg > 50) {
                    print "⚠️  MODERATE: Medium GPU utilization (50-80%)"
                } else if (avg > 10) {
                    print "❌ POOR: Low GPU utilization (10-50%)"
                } else {
                    print "❌ CRITICAL: Very low GPU utilization (<10%)"
                }
            } else {
                print "❌ ERROR: No valid GPU utilization data found"
            }
        }'
        
        echo ""
        echo "Sample GPU usage (last 10 measurements):"
        tail -10 "$GPU_LOG"
    else
        log_warning "GPU monitoring log is empty or missing"
    fi
    
    # Vérifier les outputs
    OUTPUT_COUNT=$(find "$TEST_OUTPUT/output" -name "*.laz" 2>/dev/null | wc -l)
    log_info "Output files generated: $OUTPUT_COUNT"
    
    if [ $OUTPUT_COUNT -gt 0 ]; then
        echo ""
        echo "Output files:"
        find "$TEST_OUTPUT/output" -name "*.laz" -exec ls -lh {} \;
    fi
    
else
    log_error "❌ Processing failed (exit code: $PROCESS_EXIT_CODE)"
    echo ""
    echo "Common GPU issues to check:"
    echo "1. CUDA out of memory → Reduce batch sizes"
    echo "2. GPU drivers → Update NVIDIA drivers"
    echo "3. CUDA version compatibility → Check PyTorch CUDA version"
    echo "4. Configuration → Check acceleration_mode and force_method"
fi

echo ""
echo "======================================================================"
echo "🏁 GPU Validation Complete"
echo "======================================================================"

if [ $PROCESS_EXIT_CODE -eq 0 ]; then
    echo "✅ Status: SUCCESS"
    echo "📈 Performance: Check GPU utilization above"
    echo "📁 Test output: $TEST_OUTPUT"
    echo ""
    echo "Next steps:"
    echo "1. If GPU utilization <80%, review batch sizes"
    echo "2. Run full processing: ./run_ground_truth_reclassification.sh"
    echo "3. Monitor with: watch -n 1 nvidia-smi"
else
    echo "❌ Status: FAILED"
    echo "🔧 Action needed: Fix configuration issues above"
fi