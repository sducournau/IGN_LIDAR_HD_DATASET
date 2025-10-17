#!/bin/bash
# Force GPU Batch Size - Écrasement Complet de Tous les Paramètres
# Date: October 17, 2025

echo "======================================================================"
echo "FORCE GPU BATCH SIZE - Debug et correction complète"
echo "======================================================================"

# Configuration
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/tmp/ign_force_batch_size"
CACHE_DIR="/mnt/d/ign/cache"
CONFIG="configs/config_asprs_rtx4080.yaml"

# Trouver le premier fichier LAZ
FIRST_FILE=$(find "$INPUT_DIR" -name "*.laz" | head -1)

if [ -z "$FIRST_FILE" ]; then
    echo "❌ Aucun fichier LAZ trouvé dans $INPUT_DIR"
    exit 1
fi

echo "Fichier de test: $(basename "$FIRST_FILE")"
echo "STRATEGIE: Force TOUS les paramètres batch_size possibles"
echo "Target: batch_size=2,000,000 (2M) dans les logs"
echo ""

# Créer répertoire de sortie temporaire
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*

# Activer environnement
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

echo "Lancement avec FORCE TOUS les paramètres batch_size..."

# STRATEGY: Force EVERY possible batch_size parameter
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
    processor.batch_size=64 \
    processor.prefetch_factor=2 \
    \
    processing.mode=enriched_only \
    processing.architecture=direct \
    processing.batch_size=64 \
    \
    features.use_gpu=true \
    features.gpu_batch_size=2000000 \
    features.batch_size=2000000 \
    features.chunk_size=2000000 \
    features.max_batch_size=2000000 \
    features.gpu_chunk_size=2000000 \
    features.processing_batch_size=2000000 \
    features.feature_batch_size=2000000 \
    features.computation_batch_size=2000000 \
    features.vram_utilization_target=0.6 \
    features.num_cuda_streams=2 \
    features.k_neighbors=10 \
    features.search_radius=0.8 \
    features.use_nir=false \
    features.compute_ndvi=false \
    features.adaptive_chunk_sizing=false \
    features.enable_async_processing=false \
    features.memory_pool_enabled=false \
    features.gpu_optimization.enable_mixed_precision=false \
    features.gpu_optimization.adaptive_memory_management=false \
    features.gpu_optimization.enable_memory_pooling=false \
    features.gpu_optimization.enable_tensor_cores=false \
    features.gpu_optimization.enable_kernel_fusion=false \
    \
    ground_truth.enabled=true \
    ground_truth.update_classification=true \
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    ground_truth.batch_size=2000000 \
    ground_truth.chunk_size=2000000 \
    ground_truth.gpu_batch_size=2000000 \
    ground_truth.optimization.gpu_chunk_size=2000000 \
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
    log_level=DEBUG \
    verbose=true

TEST_RESULT=$?

echo ""
echo "======================================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ TEST TERMINÉ - Vérifier les logs ci-dessus"
    echo ""
    echo "À CHERCHER dans les logs:"
    echo "🎯 '🚀 GPU mode enabled (batch_size=2,000,000)' ✅"
    echo "❌ '🚀 GPU mode enabled (batch_size=8,000,000)' (si encore 8M)"
    echo ""
    echo "Et aussi:"
    echo "🎯 'radius=0.80m' (au lieu de 0.60m) ✅"
    echo "🎯 'k=10' (au lieu de 8) ✅"
    
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo "Fichiers générés:"
        ls -lah "$OUTPUT_DIR"
        
        TOTAL_FILES=$(find "$OUTPUT_DIR" -name "*.laz" | wc -l)
        PATCH_FILES=$(find "$OUTPUT_DIR" -name "*patch*.laz" | wc -l 2>/dev/null || echo "0")
        
        echo ""
        echo "✅ Fichiers LAZ: $TOTAL_FILES"
        if [ "$PATCH_FILES" -eq 0 ]; then
            echo "✅ Patches: 0 (enriched_only fonctionne)"
        else
            echo "⚠️  Patches: $PATCH_FILES (enriched_only ne fonctionne pas!)"
        fi
    fi
    
else
    echo "❌ ERREUR PENDANT LE TEST"
    echo ""
    echo "Vérifier les logs ci-dessus pour:"
    echo "1. Erreurs GPU/CUDA"
    echo "2. Paramètres qui ne sont pas pris en compte"
    echo "3. Messages d'erreur spécifiques"
fi

echo ""
echo "ANALYSE:"
echo "Si batch_size = 8M encore → Le paramètre est hard-codé quelque part"
echo "Si radius = 0.60m encore → search_radius pas pris en compte"
echo "Si patches générés → architecture pas vraiment 'direct'"
echo ""
echo "SOLUTION suivante:"
echo "1. Si batch_size toujours 8M → Modif code source ou config différente"
echo "2. Si radius toujours 0.60m → Paramètre dans autre section"
echo "3. Si patches générés → Force mode CPU pur"
echo "======================================================================"