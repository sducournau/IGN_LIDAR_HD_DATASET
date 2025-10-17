#!/bin/bash
# Script de nettoyage du repository IGN LiDAR HD
# Supprime les fichiers temporaires, logs, et artefacts de développement
# 
# Usage: ./scripts/cleanup_repo.sh [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DRY_RUN=false

# Parse arguments
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "🔍 Mode dry-run activé - aucun fichier ne sera supprimé"
fi

# Fonction de suppression avec log
safe_remove() {
    local pattern="$1"
    local description="$2"
    
    echo "🧹 $description"
    
    if [ "$DRY_RUN" = true ]; then
        find "$PROJECT_DIR" -name "$pattern" -type f 2>/dev/null | while read -r file; do
            echo "  [DRY-RUN] Supprimerait: $(basename "$file")"
        done
    else
        local count=0
        find "$PROJECT_DIR" -name "$pattern" -type f 2>/dev/null | while read -r file; do
            rm -f "$file"
            echo "  ✅ Supprimé: $(basename "$file")"
            count=$((count + 1))
        done
        if [ $count -eq 0 ]; then
            echo "  ℹ️  Aucun fichier trouvé"
        fi
    fi
}

# Fonction de suppression de répertoires
safe_remove_dir() {
    local pattern="$1"
    local description="$2"
    
    echo "🧹 $description"
    
    if [ "$DRY_RUN" = true ]; then
        find "$PROJECT_DIR" -name "$pattern" -type d 2>/dev/null | while read -r dir; do
            echo "  [DRY-RUN] Supprimerait répertoire: $(basename "$dir")"
        done
    else
        find "$PROJECT_DIR" -name "$pattern" -type d 2>/dev/null | while read -r dir; do
            rm -rf "$dir"
            echo "  ✅ Répertoire supprimé: $(basename "$dir")"
        done
    fi
}

echo "🚀 Nettoyage du repository IGN LiDAR HD"
echo "======================================"
echo "Répertoire: $PROJECT_DIR"
echo ""

cd "$PROJECT_DIR"

# 1. Fichiers de log
safe_remove "*.log" "Fichiers de log"
safe_remove "*_log.txt" "Fichiers de log texte"
safe_remove "gpu_usage*.log" "Logs d'utilisation GPU"
safe_remove "performance_*.json" "Métriques de performance"

# 2. Fichiers temporaires
safe_remove "*.tmp" "Fichiers temporaires"
safe_remove "*.temp" "Fichiers temporaires"
safe_remove "*~" "Fichiers de backup éditeur"
safe_remove ".DS_Store" "Fichiers macOS"

# 3. Caches Python
safe_remove_dir "__pycache__" "Caches Python"
safe_remove "*.pyc" "Fichiers bytecode Python"
safe_remove "*.pyo" "Fichiers optimisés Python"

# 4. Artefacts de test
safe_remove_dir ".pytest_cache" "Cache pytest"
safe_remove ".coverage" "Fichiers de couverture"
safe_remove "htmlcov" "Rapports de couverture HTML"
safe_remove "coverage.xml" "Rapports XML de couverture"

# 5. Builds et distributions
safe_remove_dir "build" "Répertoires de build"
safe_remove_dir "dist" "Répertoires de distribution"
safe_remove_dir "*.egg-info" "Métadonnées d'installation"

# 6. Notebooks checkpoints
safe_remove_dir ".ipynb_checkpoints" "Checkpoints Jupyter"

# 7. Fichiers d'environnement
safe_remove ".env" "Fichiers d'environnement local"
safe_remove ".env.local" "Fichiers d'environnement local"

# 8. Fichiers de dump/debug spécifiques au projet
safe_remove "*_dump.json" "Fichiers de dump JSON"
safe_remove "*_debug.txt" "Fichiers de debug"
safe_remove "debug_*.py" "Scripts de debug temporaires"

# 9. Outputs de processing temporaires
safe_remove_dir "/tmp/ign_*" "Sorties temporaires IGN"

echo ""
echo "🎉 Nettoyage terminé!"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "💡 Pour effectuer le nettoyage réel, lancez:"
    echo "   ./scripts/cleanup_repo.sh"
fi

echo ""
echo "📋 Fichiers conservés:"
echo "   ✅ Configurations v4.0 (configs/)"
echo "   ✅ Scripts utilitaires (scripts/)"
echo "   ✅ Code source (ign_lidar/)"
echo "   ✅ Documentation (docs/)"
echo "   ✅ Tests (tests/)"
echo "   ✅ Archives legacy (*_legacy_*)"