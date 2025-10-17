#!/bin/bash
# IGN LiDAR HD - Script de Processing Unifié v4.0
# ==============================================
# 
# Script unifié remplaçant tous les scripts spécialisés :
# - run_ground_truth_reclassification.sh
# - run_gpu_conservative.sh  
# - run_forced_ultra_fast.sh
# - run_full_enrichment.sh
# - run_gpu_processing.sh
#
# Usage:
#   ./scripts/run_processing.sh --preset gpu_optimized --input /data/tiles --output /data/processed
#   ./scripts/run_processing.sh --config configs/presets/asprs_classification.yaml
#   ./scripts/run_processing.sh --help
#
# Date: 2025-10-17
# Version: 4.0.0

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Valeurs par défaut
DEFAULT_PRESET="gpu_optimized"
DEFAULT_CONFIG="$PROJECT_DIR/configs/config.yaml"
DEFAULT_INPUT_DIR=""
DEFAULT_OUTPUT_DIR=""
DEFAULT_CACHE_DIR="$HOME/.cache/ign_lidar"
DEFAULT_HARDWARE="auto"

# Variables de script
PRESET=""
CONFIG=""
INPUT_DIR=""
OUTPUT_DIR=""
CACHE_DIR=""
HARDWARE_PROFILE=""
CONDA_ENV="ign_gpu"
DRY_RUN=false
VERBOSE=true
MONITOR_GPU=false
VALIDATE_ONLY=false

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

# Logging avec timestamps
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  INFO: $1"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ SUCCESS: $1"
}

# Affichage d'aide
show_help() {
    cat << 'EOF'
IGN LiDAR HD - Script de Processing Unifié v4.0
==============================================

USAGE:
    ./scripts/run_processing.sh [OPTIONS]

OPTIONS PRINCIPALES:
    --preset PRESET           Utiliser un preset prédéfini
                              Valeurs: gpu_optimized, asprs_classification, 
                                      enrichment_only, minimal
    
    --config CONFIG           Chemin vers fichier de configuration custom
    
    --input INPUT_DIR         Répertoire d'entrée (fichiers LAS/LAZ)
    --output OUTPUT_DIR       Répertoire de sortie
    --cache CACHE_DIR         Répertoire de cache (optionnel)
    
    --hardware PROFILE        Profil hardware: auto, rtx4080, rtx3080, cpu_only

OPTIONS AVANCÉES:
    --conda-env ENV           Environnement conda (défaut: ign_gpu)
    --monitor-gpu             Activer monitoring GPU temps-réel
    --validate-only           Validation configuration uniquement
    --dry-run                 Affichage commande sans exécution
    --quiet                   Mode silencieux (moins de logs)
    
    --help                    Afficher cette aide

EXEMPLES:
    # Processing GPU optimisé (recommandé)
    ./scripts/run_processing.sh --preset gpu_optimized \\
        --input /data/tiles --output /data/processed
    
    # Classification ASPRS standard
    ./scripts/run_processing.sh --preset asprs_classification \\
        --input /data/tiles
    
    # Test rapide sur échantillon
    ./scripts/run_processing.sh --preset minimal \\
        --input /data/test --dry-run
    
    # Avec profil hardware spécifique
    ./scripts/run_processing.sh --preset gpu_optimized \\
        --hardware rtx4080 --monitor-gpu
    
    # Configuration custom
    ./scripts/run_processing.sh --config my_config.yaml \\
        --input /data/tiles

PRESETS DISPONIBLES:
    gpu_optimized           Performance maximale GPU (RTX 4080/3080)
    asprs_classification    Classification ASPRS standard, qualité prioritaire
    enrichment_only         LAZ enrichis uniquement, mode rapide
    minimal                 Configuration minimale pour tests

PROFILS HARDWARE:
    auto                    Détection automatique
    rtx4080                 Optimisé NVIDIA RTX 4080 (16GB VRAM)
    rtx3080                 Optimisé NVIDIA RTX 3080 (10GB VRAM)  
    cpu_only                CPU uniquement (fallback)

CODES DE RETOUR:
    0                       Succès
    1                       Erreur arguments/configuration
    2                       Erreur environnement/dépendances
    3                       Erreur processing
EOF
}

# Validation des arguments
validate_args() {
    # Validation preset OU config (mutuellement exclusifs)
    if [ -n "$PRESET" ] && [ -n "$CONFIG" ]; then
        log_error "Spécifiez soit --preset soit --config, pas les deux"
        return 1
    fi
    
    # Au moins un des deux requis
    if [ -z "$PRESET" ] && [ -z "$CONFIG" ]; then
        log_warning "Aucun preset ou config spécifié, utilisation du preset par défaut: $DEFAULT_PRESET"
        PRESET="$DEFAULT_PRESET"
    fi
    
    # Validation répertoire d'entrée
    if [ -z "$INPUT_DIR" ]; then
        log_error "Répertoire d'entrée requis (--input)"
        return 1
    fi
    
    if [ ! -d "$INPUT_DIR" ]; then
        log_error "Répertoire d'entrée inexistant: $INPUT_DIR"
        return 1
    fi
    
    # Validation fichiers LiDAR
    file_count=$(find "$INPUT_DIR" -name "*.laz" -o -name "*.las" | wc -l)
    if [ "$file_count" -eq 0 ]; then
        log_error "Aucun fichier LAS/LAZ trouvé dans: $INPUT_DIR"
        return 1
    fi
    
    log_info "Fichiers LiDAR trouvés: $file_count"
    
    # Répertoire de sortie par défaut
    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="${INPUT_DIR%/}_processed"
        log_info "Répertoire de sortie auto: $OUTPUT_DIR"
    fi
    
    # Cache par défaut
    if [ -z "$CACHE_DIR" ]; then
        CACHE_DIR="$DEFAULT_CACHE_DIR"
    fi
    
    return 0
}

# Détection du hardware automatique
detect_hardware() {
    if [ "$HARDWARE_PROFILE" = "auto" ]; then
        log_info "Détection automatique du hardware..."
        
        if command -v nvidia-smi &> /dev/null; then
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
            log_info "GPU détecté: $gpu_name"
            
            if echo "$gpu_name" | grep -qi "rtx.*40[789]0"; then
                HARDWARE_PROFILE="rtx4080"
                log_info "Profil hardware sélectionné: RTX 4080 série"
            elif echo "$gpu_name" | grep -qi "rtx.*30[789]0"; then
                HARDWARE_PROFILE="rtx3080" 
                log_info "Profil hardware sélectionné: RTX 3080 série"
            else
                HARDWARE_PROFILE="rtx3080"  # Fallback GPU générique
                log_warning "GPU non reconnu, utilisation profil RTX 3080 générique"
            fi
        else
            HARDWARE_PROFILE="cpu_only"
            log_warning "Aucun GPU NVIDIA détecté, mode CPU uniquement"
        fi
    fi
}

# Résolution du fichier de configuration
resolve_config() {
    if [ -n "$PRESET" ]; then
        CONFIG="$PROJECT_DIR/configs/presets/${PRESET}.yaml"
        if [ ! -f "$CONFIG" ]; then
            log_error "Preset inexistant: $PRESET"
            log_info "Presets disponibles:"
            ls -1 "$PROJECT_DIR/configs/presets/"*.yaml 2>/dev/null | xargs -I {} basename {} .yaml | sed 's/^/  - /' || echo "  (aucun preset trouvé)"
            return 1
        fi
        log_info "Configuration preset: $CONFIG"
    else
        if [ ! -f "$CONFIG" ]; then
            log_error "Fichier de configuration inexistant: $CONFIG"
            return 1
        fi
        log_info "Configuration custom: $CONFIG"
    fi
    
    return 0
}

# Setup de l'environnement conda
setup_conda_env() {
    log_info "Configuration environnement conda: $CONDA_ENV"
    
    # Vérifier conda
    if ! command -v conda &> /dev/null; then
        log_error "conda non trouvé. Installez Anaconda/Miniconda"
        return 2
    fi
    
    # Initialiser conda pour ce shell
    eval "$(conda shell.bash hook)" 2>/dev/null || {
        source "$(conda info --base)/etc/profile.d/conda.sh"
    }
    
    # Vérifier environnement
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        log_error "Environnement conda '$CONDA_ENV' non trouvé"
        log_info "Créez l'environnement avec: conda create -n $CONDA_ENV python=3.10"
        return 2
    fi
    
    # Activer environnement
    conda activate "$CONDA_ENV" || {
        log_error "Impossible d'activer l'environnement conda: $CONDA_ENV"
        return 2
    }
    
    log_success "Environnement conda activé: $CONDA_ENV"
    
    # Vérifier packages essentiels
    if ! python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        log_warning "PyTorch non disponible dans l'environnement"
    fi
    
    if ! python -c "import cupy; print(f'CuPy: {cupy.__version__}')" 2>/dev/null; then
        log_warning "CuPy non disponible - performances GPU limitées"
    fi
    
    return 0
}

# Vérification GPU
check_gpu_status() {
    if command -v nvidia-smi &> /dev/null; then
        log_info "Statut GPU:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=',' read -r name total used free util; do
            echo "  GPU: $(echo $name | xargs)"
            echo "  VRAM: ${used}/${total} MB ($(echo "$free" | xargs) MB libre)"
            echo "  Utilisation: $(echo "$util" | xargs)%"
        done
        
        # Vérifier CUDA dans Python
        if python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')" 2>/dev/null | grep -q "True"; then
            log_success "CUDA opérationnel"
        else
            log_warning "CUDA non disponible - fallback CPU"
        fi
    else
        log_warning "nvidia-smi non disponible - impossible de vérifier le GPU"
    fi
}

# Monitoring GPU en arrière-plan
start_gpu_monitoring() {
    if [ "$MONITOR_GPU" = true ] && command -v nvidia-smi &> /dev/null; then
        log_info "Démarrage monitoring GPU..."
        
        # Créer répertoire logs
        LOG_DIR="$OUTPUT_DIR/logs"
        mkdir -p "$LOG_DIR"
        
        # Script de monitoring en arrière-plan
        "$PROJECT_DIR/scripts/gpu_monitor.sh" 3600 > "$LOG_DIR/gpu_monitoring.log" 2>&1 &
        GPU_MONITOR_PID=$!
        
        log_info "Monitoring GPU démarré (PID: $GPU_MONITOR_PID)"
        echo $GPU_MONITOR_PID > "$LOG_DIR/gpu_monitor.pid"
    fi
}

# Arrêt monitoring GPU
stop_gpu_monitoring() {
    if [ -n "$GPU_MONITOR_PID" ]; then
        log_info "Arrêt monitoring GPU..."
        kill $GPU_MONITOR_PID 2>/dev/null || true
        wait $GPU_MONITOR_PID 2>/dev/null || true
    fi
    
    # Cleanup PID file
    if [ -f "$OUTPUT_DIR/logs/gpu_monitor.pid" ]; then
        rm -f "$OUTPUT_DIR/logs/gpu_monitor.pid"
    fi
}

# Construction de la commande ign-lidar-hd
build_command() {
    local cmd="ign-lidar-hd process"
    
    # Configuration principale
    cmd="$cmd --config-file \"$CONFIG\""
    
    # Paramètres I/O
    cmd="$cmd input_dir=\"$INPUT_DIR\""
    cmd="$cmd output_dir=\"$OUTPUT_DIR\""
    cmd="$cmd cache_dir=\"$CACHE_DIR\""
    
    # Profil hardware (si applicable)
    if [ "$HARDWARE_PROFILE" != "auto" ] && [ -f "$PROJECT_DIR/configs/hardware/${HARDWARE_PROFILE}.yaml" ]; then
        cmd="$cmd --config-file \"$PROJECT_DIR/configs/hardware/${HARDWARE_PROFILE}.yaml\""
    fi
    
    # Mode verbose
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd verbose=true"
        cmd="$cmd monitoring.log_level=INFO"
    else
        cmd="$cmd verbose=false"
        cmd="$cmd monitoring.log_level=WARNING"
    fi
    
    echo "$cmd"
}

# Validation de configuration seulement
validate_config_only() {
    log_info "Validation de la configuration..."
    
    # Construire commande de validation
    local validation_cmd
    validation_cmd=$(build_command)
    validation_cmd="$validation_cmd --validate-only"
    
    log_info "Commande de validation: $validation_cmd"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Mode dry-run - validation simulée"
        return 0
    fi
    
    # Exécuter validation
    eval "$validation_cmd"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_success "Configuration valide"
    else
        log_error "Configuration invalide"
    fi
    
    return $exit_code
}

# Exécution du processing principal
run_processing() {
    log_info "Démarrage du processing IGN LiDAR HD..."
    
    # Créer répertoires
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$CACHE_DIR"
    
    # Construire commande
    local processing_cmd
    processing_cmd=$(build_command)
    
    log_info "Commande de processing:"
    echo "  $processing_cmd"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Mode dry-run - processing simulé"
        return 0
    fi
    
    # Enregistrer timestamp de début
    local start_time
    start_time=$(date +%s)
    
    # Démarrer monitoring GPU si demandé
    start_gpu_monitoring
    
    # Exécuter processing
    log_info "Exécution en cours..."
    eval "$processing_cmd"
    local exit_code=$?
    
    # Arrêter monitoring
    stop_gpu_monitoring
    
    # Calculer durée
    local end_time duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Rapport final
    log_info "Processing terminé en ${duration}s"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Processing réussi!"
        
        # Statistiques de sortie
        if [ -d "$OUTPUT_DIR" ]; then
            local output_count
            output_count=$(find "$OUTPUT_DIR" -name "*.laz" | wc -l)
            log_info "Fichiers générés: $output_count"
            
            # Taille totale
            local total_size
            total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "N/A")
            log_info "Taille totale sortie: $total_size"
        fi
        
        # Monitoring GPU summary
        if [ -f "$OUTPUT_DIR/logs/gpu_monitoring.log" ]; then
            log_info "Log monitoring GPU: $OUTPUT_DIR/logs/gpu_monitoring.log"
        fi
    else
        log_error "Processing échoué (code: $exit_code)"
    fi
    
    return $exit_code
}

# Cleanup à la sortie
cleanup() {
    stop_gpu_monitoring
}

# ============================================================================
# PARSING DES ARGUMENTS
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cache)
            CACHE_DIR="$2"
            shift 2
            ;;
        --hardware)
            HARDWARE_PROFILE="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --monitor-gpu)
            MONITOR_GPU=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Option inconnue: $1"
            echo ""
            echo "Utilisez --help pour voir les options disponibles"
            exit 1
            ;;
    esac
done

# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

# Trap pour cleanup
trap cleanup EXIT

# Header
echo "======================================================================"
echo "🚀 IGN LiDAR HD - Script de Processing Unifié v4.0"
echo "======================================================================"

# Valeurs par défaut pour hardware
if [ -z "$HARDWARE_PROFILE" ]; then
    HARDWARE_PROFILE="$DEFAULT_HARDWARE"
fi

# Validation des arguments
if ! validate_args; then
    exit 1
fi

# Détection hardware
detect_hardware

# Résolution configuration
if ! resolve_config; then
    exit 1
fi

# Setup environnement
if ! setup_conda_env; then
    exit 2
fi

# Vérification GPU
check_gpu_status

# Affichage configuration finale
echo ""
echo "Configuration finale:"
echo "  Preset/Config: ${PRESET:-custom} (${CONFIG})"
echo "  Hardware: $HARDWARE_PROFILE"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Cache: $CACHE_DIR"
echo "  Conda env: $CONDA_ENV"
echo "  GPU monitoring: $MONITOR_GPU"
echo "  Dry run: $DRY_RUN"
echo ""

# Exécution selon le mode
if [ "$VALIDATE_ONLY" = true ]; then
    validate_config_only
    exit_code=$?
else
    run_processing
    exit_code=$?
fi

# Exit avec le code approprié
exit $exit_code