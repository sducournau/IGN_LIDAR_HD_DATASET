"""
Configuration centralisée pour le package ign_lidar.

Ce module contient toutes les configurations par défaut utilisées
dans les workflows de création de datasets IA.
"""

from pathlib import Path
from typing import Dict, Any

# ============================================================================
# CONFIGURATIONS PAR DÉFAUT
# ============================================================================

# Patch Configuration
DEFAULT_PATCH_SIZE = 150.0  # Taille des patches en mètres
DEFAULT_NUM_POINTS = 16384  # Points par patch (optimal GPU ≥12GB)
DEFAULT_OVERLAP = 0.5       # Chevauchement entre patches (50%)

# Feature Computation
# Optimal k for IGN LiDAR HD (~0.5m radius, good for building extraction)
DEFAULT_K_NEIGHBORS = 20

# Memory Management
# Chunk size for processing large point clouds to avoid OOM errors
# Adjusted for system: 31GB RAM, 20 cores, RTX 4080 SUPER (16GB VRAM)
# Smaller = less memory but slower, larger = faster but more memory
DEFAULT_CHUNK_SIZE_LARGE = 10_000_000   # For files >30M points (high-end system)
DEFAULT_CHUNK_SIZE_MEDIUM = 20_000_000  # For files 15-30M points (high-end system)
# Files <15M points: no chunking (process all at once)
# Note: With 31GB RAM, we can process larger chunks efficiently

# Dataset Split
DEFAULT_TRAIN_SPLIT = 0.8   # 80% train, 20% validation

# Paths Configuration
DEFAULT_OUTPUT_DIR = Path.home() / "ign_lidar_data"
DEFAULT_DOWNLOAD_DIR = Path.home() / "ign_lidar_data" / "downloads"
DEFAULT_FEATURES_DIR = Path.home() / "ign_lidar_data" / "features"
DEFAULT_PATCHES_DIR = Path.home() / "ign_lidar_data" / "patches"

# ============================================================================
# LOD CONFIGURATIONS
# ============================================================================

LOD2_NUM_CLASSES = 6   # Classes LOD2 (simplifié)
LOD3_NUM_CLASSES = 30  # Classes LOD3 (détaillé)

# ============================================================================
# DOWNLOAD CONFIGURATIONS
# ============================================================================

# IGN LIDAR HD WFS Service
IGN_WFS_URL = "https://data.geopf.fr/wfs"
IGN_WFS_TYPENAME = "LIDARHD_FXX"

# Download Settings
DEFAULT_NUM_TILES = 60       # Nombre de tuiles à télécharger par défaut
DOWNLOAD_TIMEOUT = 300       # Timeout en secondes
DOWNLOAD_RETRY_ATTEMPTS = 3  # Nombre de tentatives de téléchargement

# ============================================================================
# FEATURE DIMENSIONS
# ============================================================================

FEATURE_DIMENSIONS = {
    # === CORE FEATURES (always computed) ===
    'xyz': 3,                    # Coordonnées XYZ
    'intensity': 1,              # Intensité
    'return_number': 1,          # Numéro de retour
    'normals': 3,                # Normales (X, Y, Z) - orientation
    'curvature': 1,              # Courbure
    'height_above_ground': 1,    # Hauteur au-dessus du sol
    'density': 1,                # Densité locale
    'planarity': 1,              # Planéité (surfaces planes)
    'linearity': 1,              # Linéarité (arêtes, câbles)
    'sphericity': 1,             # Sphéricité (végétation, bruit)
    'anisotropy': 1,             # Anisotropie (structure directionnelle)
    'roughness': 1,              # Rugosité (lisse vs rugueux)
    
    # === EXTRA FEATURES (computed if include_extra=True) ===
    # CRITICAL for building extraction
    'z_absolute': 1,             # Hauteur absolue Z
    'z_normalized': 1,           # Hauteur normalisée [0, 1]
    'z_from_ground': 1,          # Hauteur depuis le sol (Z - Z_min)
    'z_from_median': 1,          # Hauteur relative à la médiane
    'distance_to_center': 1,     # Distance euclidienne au centre
    
    # POWERFUL but EXPENSIVE for large clouds
    'vertical_std': 1,           # Écart-type vertical voisinage
    'neighborhood_extent': 1,    # Étendue du voisinage (max dist)
    'height_extent_ratio': 1,    # Ratio hauteur/étendue
    'local_roughness': 1,        # Rugosité locale (std au plan)
    'verticality': 1,            # Verticalité (murs vs toits)
}

TOTAL_FEATURE_DIM = sum(FEATURE_DIMENSIONS.values())  # 28 features total

# Feature sets pour DataLoader
FEATURE_SETS = {
    # Fast - only core features (18 dims)
    'minimal': ['xyz', 'intensity', 'return_number', 'normals'],  # 8 dims
    
    # Standard - good balance (14 dims)
    'geometric': [
        'xyz', 'intensity', 'return_number', 'normals',
        'curvature', 'height_above_ground', 'planarity', 'linearity'
    ],
    
    # Core - all fast features (18 dims)
    'core': [
        'xyz', 'intensity', 'return_number', 'normals',
        'curvature', 'height_above_ground', 'density',
        'planarity', 'linearity', 'sphericity', 'anisotropy', 'roughness'
    ],
    
    # Building - optimized for building extraction (23 dims)
    'building': [
        'xyz', 'intensity', 'return_number', 'normals',
        'curvature', 'height_above_ground', 'density',
        'planarity', 'linearity', 'verticality',
        'z_absolute', 'z_normalized', 'z_from_ground',
        'vertical_std', 'height_extent_ratio'
    ],
    
    # Full - everything (28 dims, expensive)
    'full': list(FEATURE_DIMENSIONS.keys())
}

# ============================================================================
# LAZ EXTRA DIMENSIONS
# ============================================================================

# Dimensions supplémentaires pour LAZ enrichi
LAZ_EXTRA_DIMS = [
    # Normals
    'normal_x',
    'normal_y',
    'normal_z',
    # Core geometric features
    'curvature',
    'height_above_ground',
    'density',
    'planarity',
    'linearity',
    'sphericity',
    'anisotropy',
    'roughness',
    # Height features (critical for buildings)
    'z_normalized',
    'z_from_ground',
    'verticality',
    # Local statistics
    'vertical_std',
    'height_extent_ratio',
    # Labels
    'label_lod2',
    'label_lod3',
]

# ============================================================================
# OUTPUT FORMAT PREFERENCES
# ============================================================================

# Prefer augmented LAZ (LAZ 1.4, format 6+) over QGIS format
# When True: Save enriched LAZ in native format with all extra dimensions
# When False: Optionally convert to QGIS format (LAZ 1.2, format 3)
PREFER_AUGMENTED_LAZ = True

# Auto-convert to QGIS format after enrichment
# Only applies when PREFER_AUGMENTED_LAZ is False
AUTO_CONVERT_TO_QGIS = False


# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

VALIDATION_SAMPLE_SIZE = 100  # Patches à échantillonner pour validation
MIN_POINTS_PER_PATCH = 1000   # Minimum de points requis par patch
MAX_NAN_RATIO = 0.01          # Ratio maximum de NaN acceptable


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_default_config() -> Dict[str, Any]:
    """
    Retourne un dictionnaire avec toutes les configurations par défaut.
    
    Returns:
        Dict contenant toutes les valeurs de configuration
    """
    return {
        # Patch settings
        'patch_size': DEFAULT_PATCH_SIZE,
        'num_points': DEFAULT_NUM_POINTS,
        'overlap': DEFAULT_OVERLAP,
        
        # Feature settings
        'k_neighbors': DEFAULT_K_NEIGHBORS,
        
        # Dataset settings
        'train_split': DEFAULT_TRAIN_SPLIT,
        'num_tiles': DEFAULT_NUM_TILES,
        
        # Paths
        'output_dir': DEFAULT_OUTPUT_DIR,
        'download_dir': DEFAULT_DOWNLOAD_DIR,
        'features_dir': DEFAULT_FEATURES_DIR,
        'patches_dir': DEFAULT_PATCHES_DIR,
        
        # LOD
        'lod2_classes': LOD2_NUM_CLASSES,
        'lod3_classes': LOD3_NUM_CLASSES,
        
        # Features
        'feature_dimensions': FEATURE_DIMENSIONS,
        'total_feature_dim': TOTAL_FEATURE_DIM,
        'feature_sets': FEATURE_SETS,
        
        # Download
        'wfs_url': IGN_WFS_URL,
        'wfs_typename': IGN_WFS_TYPENAME,
        'download_timeout': DOWNLOAD_TIMEOUT,
        'download_retry_attempts': DOWNLOAD_RETRY_ATTEMPTS,
        
        # Validation
        'validation_sample_size': VALIDATION_SAMPLE_SIZE,
        'min_points_per_patch': MIN_POINTS_PER_PATCH,
        'max_nan_ratio': MAX_NAN_RATIO,
    }


def get_feature_set_dims(feature_set: str = 'full') -> int:
    """
    Retourne le nombre de dimensions pour un feature set donné.
    
    Args:
        feature_set: Nom du feature set ('minimal', 'geometric', 'full')
        
    Returns:
        Nombre total de dimensions
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Feature set inconnu: {feature_set}. "
            f"Choix: {list(FEATURE_SETS.keys())}"
        )
    
    total_dims = 0
    for feature_name in FEATURE_SETS[feature_set]:
        total_dims += FEATURE_DIMENSIONS[feature_name]
    
    return total_dims


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valide une configuration utilisateur.
    
    Args:
        config: Dictionnaire de configuration à valider
        
    Returns:
        True si la configuration est valide
        
    Raises:
        ValueError: Si la configuration est invalide
    """
    if 'patch_size' in config and config['patch_size'] <= 0:
        raise ValueError("patch_size doit être > 0")
    
    if 'num_points' in config and config['num_points'] < MIN_POINTS_PER_PATCH:
        raise ValueError(f"num_points doit être >= {MIN_POINTS_PER_PATCH}")
    
    if 'train_split' in config:
        if not 0 < config['train_split'] < 1:
            raise ValueError("train_split doit être entre 0 et 1")
    
    if 'overlap' in config:
        if not 0 <= config['overlap'] < 1:
            raise ValueError("overlap doit être entre 0 et 1")
    
    if 'k_neighbors' in config and config['k_neighbors'] < 3:
        raise ValueError("k_neighbors doit être >= 3")
    
    return True


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    'DEFAULT_PATCH_SIZE',
    'DEFAULT_NUM_POINTS',
    'DEFAULT_OVERLAP',
    'DEFAULT_K_NEIGHBORS',
    'DEFAULT_TRAIN_SPLIT',
    'DEFAULT_OUTPUT_DIR',
    'DEFAULT_DOWNLOAD_DIR',
    'DEFAULT_FEATURES_DIR',
    'DEFAULT_PATCHES_DIR',
    'DEFAULT_NUM_TILES',
    'LOD2_NUM_CLASSES',
    'LOD3_NUM_CLASSES',
    'IGN_WFS_URL',
    'IGN_WFS_TYPENAME',
    'FEATURE_DIMENSIONS',
    'TOTAL_FEATURE_DIM',
    'FEATURE_SETS',
    'LAZ_EXTRA_DIMS',
    'PREFER_AUGMENTED_LAZ',
    'AUTO_CONVERT_TO_QGIS',
    'VALIDATION_SAMPLE_SIZE',
    'MIN_POINTS_PER_PATCH',
    'MAX_NAN_RATIO',
    
    # Functions
    'get_default_config',
    'get_feature_set_dims',
    'validate_config',
]
