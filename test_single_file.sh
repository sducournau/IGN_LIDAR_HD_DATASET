#!/bin/bash
# Test ULTRA-FAST sur un seul fichier pour validation
# Date: October 17, 2025

echo "======================================================================"
echo "TEST ULTRA-FAST - Un seul fichier pour validation"
echo "======================================================================"

# Configuration
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/tmp/ign_test_ultra_fast"
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
echo ""

# Créer répertoire de sortie temporaire
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*

# Activer environnement
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

# Tester avec UN seul fichier et parameters FORCES
echo "Lancement du test avec paramètres forcés..."
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
    processor.batch_size=512 \
    processor.reclassification.enabled=false \
    processor.apply_reclassification_inline=false \
    \
    processing.mode=enriched_only \
    processing.architecture=direct \
    \
    features.gpu_batch_size=16000000 \
    features.use_gpu=true \
    features.use_nir=false \
    features.compute_ndvi=false \
    features.k_neighbors=8 \
    features.search_radius=0.6 \
    \
    ground_truth.enabled=true \
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    \
    data_sources.cadastre_enabled=false \
    data_sources.cadastre.enabled=false \
    data_sources.bd_topo_vegetation=false \
    \
    preprocess.enabled=false \
    stitching.enabled=false \
    \
    bbox.xmin=$(echo "$FIRST_FILE" | grep -o '[0-9]\{4\}_[0-9]\{4\}' | cut -d_ -f1 | head -1)000 \
    bbox.ymin=$(echo "$FIRST_FILE" | grep -o '[0-9]\{4\}_[0-9]\{4\}' | cut -d_ -f2 | head -1)000 \
    bbox.xmax=$(($(echo "$FIRST_FILE" | grep -o '[0-9]\{4\}_[0-9]\{4\}' | cut -d_ -f1 | head -1) + 1))000 \
    bbox.ymax=$(($(echo "$FIRST_FILE" | grep -o '[0-9]\{4\}_[0-9]\{4\}' | cut -d_ -f2 | head -1) + 1))000 \
    \
    log_level=INFO \
    verbose=true

TEST_RESULT=$?

echo ""
echo "======================================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ TEST RÉUSSI!"
    echo ""
    echo "Fichiers générés:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    
    ENRICHED_COUNT=$(find "$OUTPUT_DIR" -name "*.laz" -not -name "*patch*" | wc -l)
    PATCH_COUNT=$(find "$OUTPUT_DIR" -name "*patch*.laz" | wc -l 2>/dev/null || echo "0")
    
    echo "✅ Fichiers LAZ enrichis: $ENRICHED_COUNT"
    if [ "$PATCH_COUNT" -gt 0 ]; then
        echo "⚠️  ATTENTION: Patches générés: $PATCH_COUNT (devrait être 0!)"
        find "$OUTPUT_DIR" -name "*patch*" | head -3
    else
        echo "✅ Patches: 0 (parfait!)"
    fi
    
    echo ""
    echo "🎯 Prêt pour le traitement complet avec run_forced_ultra_fast.sh"
else
    echo "❌ TEST ÉCHOUÉ!"
    echo "Vérifier les logs ci-dessus pour identifier le problème"
fi
echo "======================================================================"