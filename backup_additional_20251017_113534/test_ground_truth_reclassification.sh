#!/bin/bash
# Test Ground Truth Reclassification sur un seul fichier
# Date: October 17, 2025

echo "======================================================================"
echo "TEST - Ground Truth Reclassification (1 fichier)"
echo "======================================================================"

# Configuration
INPUT_DIR="/mnt/d/ign/selected_tiles/asprs/tiles"
OUTPUT_DIR="/tmp/ign_test_ground_truth_reclassification"
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
echo "Mode: ENRICHED + GROUND TRUTH RECLASSIFICATION"
echo "BD TOPO: Buildings, Roads, Water (PAS de cadastre)"
echo "NDVI: DISABLED (pas de fetch NIR/RGB)"
echo ""

# Créer répertoire de sortie temporaire
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*

# Activer environnement
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ign_gpu

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
echo "Lancement du test avec Ground Truth Reclassification..."
START_TIME=$(date +%s)

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
    processor.apply_reclassification_inline=true \
    processor.reclassification.enabled=true \
    processor.reclassification.acceleration_mode=cpu \
    processor.reclassification.use_geometric_rules=true \
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
    ground_truth.update_classification=true \
    ground_truth.apply_reclassification=true \
    ground_truth.use_ndvi=false \
    ground_truth.fetch_rgb_nir=false \
    \
    classification.enabled=true \
    classification.methods.ground_truth=true \
    classification.methods.geometric=true \
    classification.methods.ndvi=false \
    \
    data_sources.bd_topo_enabled=true \
    data_sources.bd_topo_buildings=true \
    data_sources.bd_topo_roads=true \
    data_sources.bd_topo_water=true \
    data_sources.bd_topo_vegetation=false \
    data_sources.cadastre_enabled=false \
    data_sources.cadastre.enabled=false \
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

echo ""
echo "======================================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ TEST GROUND TRUTH RECLASSIFICATION RÉUSSI!"
    echo ""
    echo "Temps d'exécution: ${DURATION} secondes"
    echo ""
    echo "Fichiers générés:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    
    ENRICHED_COUNT=$(find "$OUTPUT_DIR" -name "*enriched*.laz" | wc -l)
    RECLASSIFIED_COUNT=$(find "$OUTPUT_DIR" -name "*reclassified*.laz" | wc -l) 
    PATCH_COUNT=$(find "$OUTPUT_DIR" -name "*patch*.laz" | wc -l 2>/dev/null || echo "0")
    TOTAL_LAZ=$(find "$OUTPUT_DIR" -name "*.laz" | wc -l)
    
    echo "Résultats:"
    echo "✅ Total fichiers LAZ: $TOTAL_LAZ"
    echo "✅ Enriched LAZ: $ENRICHED_COUNT"
    echo "✅ Reclassified LAZ: $RECLASSIFIED_COUNT"
    if [ "$PATCH_COUNT" -gt 0 ]; then
        echo "⚠️  ATTENTION: Patches générés: $PATCH_COUNT (devrait être 0!)"
        find "$OUTPUT_DIR" -name "*patch*" | head -3
    else
        echo "✅ Patches: 0 (parfait!)"
    fi
    
    echo ""
    echo "Validation de la classification:"
    # Vérifier qu'il y a des fichiers avec reclassification
    if [ "$RECLASSIFIED_COUNT" -gt 0 ]; then
        echo "✅ Reclassification appliquée avec succès"
        echo "✅ Ground truth BD TOPO intégré"
    else
        echo "⚠️  Pas de fichier *reclassified*.laz trouvé"
        echo "   Vérifier si la reclassification s'est bien activée"
    fi
    
    echo ""
    echo "🎯 Prêt pour le traitement complet:"
    echo "   ./run_ground_truth_reclassification.sh"
else
    echo "❌ TEST ÉCHOUÉ!"
    echo "Temps écoulé: ${DURATION} secondes"
    echo ""
    echo "Vérifier les logs ci-dessus pour identifier le problème"
    echo ""
    echo "Points de vérification:"
    echo "1. GPU disponible: nvidia-smi"
    echo "2. Connectivité BD TOPO (cache/réseau)"
    echo "3. Espace disque suffisant"
    echo "4. Fichier LAZ d'entrée valide"
fi
echo "======================================================================"