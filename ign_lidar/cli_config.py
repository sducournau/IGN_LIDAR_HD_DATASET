"""Central configuration constants for CLI operations."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CLIDefaults:
    """Default values for CLI parameters."""
    
    # Processing defaults
    DEFAULT_K_NEIGHBORS: int = 10
    DEFAULT_RADIUS: Optional[float] = None
    DEFAULT_PATCH_SIZE: float = 150.0
    DEFAULT_PATCH_OVERLAP: float = 0.0
    DEFAULT_NUM_POINTS: int = 4096
    DEFAULT_NUM_WORKERS: Optional[int] = None  # None = CPU count
    DEFAULT_LOD_LEVEL: int = 2
    
    # Feature verification
    MAX_SAMPLE_POINTS: int = 1000
    EXPECTED_GEOMETRIC_FEATURES: list[str] = field(default_factory=lambda: [
        'linearity', 'planarity', 'sphericity', 'anisotropy',
        'curvature', 'omnivariance', 'eigensum', 'roughness'
    ])
    RGB_FEATURES: list[str] = field(default_factory=lambda: [
        'red', 'green', 'blue'
    ])
    INFRARED_FEATURES: list[str] = field(default_factory=lambda: [
        'infrared', 'nir'
    ])
    
    # Memory management
    MIN_GB_PER_WORKER_CORE: float = 2.5
    MIN_GB_PER_WORKER_FULL: float = 5.0
    LARGE_FILE_THRESHOLD_MB: int = 500
    MEDIUM_FILE_THRESHOLD_MB: int = 300
    MAX_WORKERS_LARGE_FILES: int = 3
    MAX_WORKERS_MEDIUM_FILES: int = 4
    
    # File patterns
    LAZ_PATTERN: str = "*.laz"
    LAZ_RECURSIVE_PATTERN: str = "**/*.laz"
    JSON_PATTERN: str = "*.json"
    TXT_PATTERN: str = "*.txt"
    
    # Logging
    LOG_SEPARATOR: str = "=" * 70
    LOG_SUB_SEPARATOR: str = "-" * 70


@dataclass
class PreprocessingDefaults:
    """Default preprocessing configuration."""
    
    # Statistical Outlier Removal (SOR)
    SOR_K: int = 12
    SOR_STD_MULTIPLIER: float = 2.0
    SOR_ENABLED: bool = True
    
    # Radius Outlier Removal (ROR)
    ROR_RADIUS: float = 1.0
    ROR_MIN_NEIGHBORS: int = 4
    ROR_ENABLED: bool = True
    
    # Voxel downsampling
    VOXEL_SIZE: float = 0.5
    VOXEL_METHOD: str = 'centroid'  # or 'random'
    VOXEL_ENABLED: bool = False


@dataclass
class AugmentationDefaults:
    """Default augmentation configuration."""
    
    DEFAULT_NUM_AUGMENTATIONS: int = 0
    MAX_ROTATION_DEGREES: float = 45.0
    MAX_SCALE_FACTOR: float = 0.1
    MAX_TRANSLATION_METERS: float = 1.0
    ENABLE_JITTER: bool = True
    JITTER_SIGMA: float = 0.01


# Global configuration instances
CLI_DEFAULTS = CLIDefaults()
PREPROCESSING_DEFAULTS = PreprocessingDefaults()
AUGMENTATION_DEFAULTS = AugmentationDefaults()


def get_preprocessing_config(
    enable_sor: bool = True,
    enable_ror: bool = True,
    enable_voxel: bool = False,
    sor_k: Optional[int] = None,
    sor_std: Optional[float] = None,
    ror_radius: Optional[float] = None,
    ror_neighbors: Optional[int] = None,
    voxel_size: Optional[float] = None,
    voxel_method: Optional[str] = None
) -> dict:
    """Build preprocessing configuration dictionary.
    
    Args:
        enable_sor: Enable Statistical Outlier Removal
        enable_ror: Enable Radius Outlier Removal
        enable_voxel: Enable voxel downsampling
        sor_k: Number of neighbors for SOR
        sor_std: Standard deviation multiplier for SOR
        ror_radius: Radius for ROR
        ror_neighbors: Minimum neighbors for ROR
        voxel_size: Voxel size for downsampling
        voxel_method: Voxel method ('centroid' or 'random')
    
    Returns:
        Preprocessing configuration dictionary
    """
    return {
        'sor': {
            'enable': enable_sor,
            'k': sor_k or PREPROCESSING_DEFAULTS.SOR_K,
            'std_multiplier': sor_std or PREPROCESSING_DEFAULTS.SOR_STD_MULTIPLIER
        },
        'ror': {
            'enable': enable_ror,
            'radius': ror_radius or PREPROCESSING_DEFAULTS.ROR_RADIUS,
            'min_neighbors': ror_neighbors or PREPROCESSING_DEFAULTS.ROR_MIN_NEIGHBORS
        },
        'voxel': {
            'enable': enable_voxel,
            'voxel_size': voxel_size or PREPROCESSING_DEFAULTS.VOXEL_SIZE,
            'method': voxel_method or PREPROCESSING_DEFAULTS.VOXEL_METHOD
        }
    }
