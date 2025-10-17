#!/bin/bash
# Test GPU Stable - Paramètres conservatifs pour éviter fallback CPU
# Date: October 17, 2025

echo "======================================================================"
echo "TEST GPU STABLE - Paramètres conservatifs (éviter fallback CPU)"
echo "======================================================================"

# Configuration
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/tmp/ign_test_gpu_stable"
CACHE_DIR="/mnt/d/ign/cache"
CONFIG="configs/config_asprs_rtx4080.yaml"

# Trouver le premier fichier LAZ
FIRST_FILE=$(find "$INPUT_DIR" -name "*.laz" | head -1)

if [ -z "$FIRST_FILE" ]; then
    echo "❌ Aucun fichier LAZ trouvé dans $INPUT_DIR"
    exit 1
fi

echo "Fichier de test: $(basename "$FIRST_FILE")"
echo "Output: $OUTPUT_DIR"
echo "GPU Batch Size: 4M points (conservatif)"
echo "VRAM Target: 75% (conservatif)"
echo "Mixed Precision: DISABLED (stabilité)"
echo ""

# Créer répertoire de sortie temporaire
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*

# Activer environnement
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

# Vérifier GPU status avant
echo "Status GPU avant traitement:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Extraire coordonnées pour bbox du nom de fichier
COORDS=$(echo "$(basename "$FIRST_FILE")" | grep -o '[0-9]\{4\}_[0-9]\{4\}')
if [ -n "$COORDS" ]; then
    XMIN=$(echo "$COORDS" | cut -d_ -f1)000
    YMIN=$(echo "$COORDS" | cut -d_ -f2)000
    XMAX=$(($(echo "$COORDS" | cut -d_ -f1) + 1))000
    YMAX=$(($(echo "$COORDS" | cut -d_ -f2) + 1))000
    echo "BBox: ($XMIN, $YMIN, $XMAX, $YMAX)"
else
    XMIN=null
    YMIN=null
    XMAX=null
    YMAX=null
    echo "BBox: Auto-detect from file"
fi

echo ""
echo "Lancement du test avec GPU stable (4M batch, 75% VRAM)..."
START_TIME=$(date +%s)

# Monitor GPU en arrière-plan
nvidia-smi dmon -s um -d 2 -f gpu_monitor.log &
GPU_PID=$!

time ign-lidar-hd process \
    --config-file "$CONFIG" \
    input_dir="$(dirname "$FIRST_FILE")" \
    output_dir="$OUTPUT_DIR" \
    cache_dir="$CACHE_DIR" \
    \
    processor.architecture=direct \
    processor.processing_mode=enriched_only \
    processor.generate_patches=false \
    processor.use_gpu=true \
    processor.batch_size=256 \
    processor.apply_reclassification_inline=false \
    processor.reclassification.enabled=false \
    \
    processing.mode=enriched_only \
    processing.architecture=direct \
    \
    features.gpu_batch_size=4000000 \
    features.use_gpu=true \
    features.vram_utilization_target=0.75 \
    features.num_cuda_streams=4 \
    features.k_neighbors=12 \
    features.search_radius=0.8 \
    features.use_nir=false \
    features.compute_ndvi=false \
    features.gpu_optimization.enable_mixed_precision=false \
    features.adaptive_chunk_sizing=true \
    \
    ground_truth.enabled=true \
    ground_truth.update_classification=true \
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    \
    data_sources.bd_topo_enabled=true \
    data_sources.bd_topo_buildings=true \
    data_sources.bd_topo_roads=true \
    data_sources.bd_topo_water=true \
    data_sources.bd_topo_vegetation=false \
    data_sources.cadastre_enabled=false \
    data_sources.cadastre.enabled=false \
    \
    preprocess.enabled=false \
    stitching.enabled=false \
    \
    bbox.xmin="$XMIN" \
    bbox.ymin="$YMIN" \
    bbox.xmax="$XMAX" \
    bbox.ymax="$YMAX" \
    \
    log_level=INFO \
    verbose=true

TEST_RESULT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Arrêter monitoring GPU
kill $GPU_PID 2>/dev/null

echo ""
echo "======================================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ TEST GPU STABLE RÉUSSI!"
    echo ""
    echo "Temps d'exécution: ${DURATION} secondes"
    echo ""
    
    # Vérifier GPU status après
    echo "Status GPU après traitement:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
    
    echo "Fichiers générés:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    
    ENRICHED_COUNT=$(find "$OUTPUT_DIR" -name "*enriched*.laz" | wc -l)
    PATCH_COUNT=$(find "$OUTPUT_DIR" -name "*patch*.laz" | wc -l 2>/dev/null || echo "0")
    TOTAL_LAZ=$(find "$OUTPUT_DIR" -name "*.laz" | wc -l)
    
    echo "Résultats:"
    echo "✅ Total fichiers LAZ: $TOTAL_LAZ"
    echo "✅ Enriched LAZ: $ENRICHED_COUNT"
    if [ "$PATCH_COUNT" -gt 0 ]; then
        echo "⚠️  ATTENTION: Patches générés: $PATCH_COUNT (devrait être 0!)"
    else
        echo "✅ Patches: 0 (parfait!)"
    fi
    
    echo ""
    echo "Analyse GPU Performance:"
    if [ -f "gpu_monitor.log" ]; then
        echo "GPU utilization moyenne:"
        tail -n +3 gpu_monitor.log | awk '{sum+=$2; count++} END {if(count>0) printf "%.1f%%\n", sum/count; else print "N/A"}'
        echo "Mémoire GPU max utilisée:"
        tail -n +3 gpu_monitor.log | awk '{if($3>max) max=$3} END {print max " MB"}'
        echo ""
        echo "Log GPU complet dans: gpu_monitor.log"
    fi
    
    echo ""
    echo "🎯 GPU fonctionne correctement avec ces paramètres:"
    echo "   - Batch size: 4M points"  
    echo "   - VRAM target: 75%"
    echo "   - Mixed precision: OFF"
    echo "   - Streams: 4"
    echo ""
    echo "Prêt pour traitement complet avec paramètres stables!"
    
else
    echo "❌ TEST ÉCHOUÉ!"
    echo "Temps écoulé: ${DURATION} secondes"
    echo ""
    echo "Vérifier:"
    echo "1. GPU disponible et non utilisé par autre processus"
    echo "2. Mémoire GPU suffisante: nvidia-smi"
    echo "3. CUDA/CuPy installés correctement"
    echo "4. Logs pour identifier si fallback CPU activé"
    
    if [ -f "gpu_monitor.log" ]; then
        echo ""
        echo "Dernières métriques GPU:"
        tail -5 gpu_monitor.log
    fi
fi
echo "======================================================================"