#!/bin/bash
# Script pour enrichir les tuiles LAZ de manière sécurisée (évite les OOM)

set -e

# Configuration
INPUT_DIR="/mnt/c/Users/Simon/ign/raw_tiles"
OUTPUT_DIR="/mnt/c/Users/Simon/ign/enriched_tiles"
K_NEIGHBORS=20
NUM_WORKERS=2  # Réduit à 2 workers pour éviter les OOM
MODE="building"  # Mode: 'core' (basic) ou 'building' (full)

# Activer l'environnement virtuel
source /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_downloader/.venv/bin/activate

# Lancer l'enrichissement avec 2 workers
echo "==================================================================="
echo "Enrichissement des tuiles LAZ avec configuration sécurisée"
echo "==================================================================="
echo "Input:   ${INPUT_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "Workers: ${NUM_WORKERS} (réduit pour éviter les problèmes mémoire)"
echo "==================================================================="
echo ""

python -m ign_lidar.cli enrich \
  --input-dir "${INPUT_DIR}" \
  --output "${OUTPUT_DIR}" \
  --k-neighbors ${K_NEIGHBORS} \
  --num-workers ${NUM_WORKERS} \
  --mode ${MODE}

echo ""
echo "==================================================================="
echo "✓ Enrichissement terminé avec succès !"
echo "==================================================================="
