"""
Feature extraction modules for IGN LiDAR HD.

This package contains geometric and radiometric feature calculation:
- features: CPU-based feature extraction
- features_gpu: GPU-accelerated feature extraction (CuPy)
- features_gpu_chunked: GPU feature extraction for large files
- features_boundary: Boundary-aware feature extraction
- factory: Factory for creating feature computers
- architectural_styles: Architectural style classification system
"""

from .features import (
    compute_normals,
    compute_curvature,
    extract_geometric_features,
    compute_all_features_optimized,
    compute_all_features_with_gpu,
)
from .factory import (
    FeatureComputerFactory,
    BaseFeatureComputer,
)
from .architectural_styles import (
    ARCHITECTURAL_STYLES,
    STYLE_NAME_TO_ID,
    CHARACTERISTIC_TO_STYLE,
)

__all__ = [
    'compute_normals',
    'compute_curvature',
    'extract_geometric_features',
    'compute_all_features_optimized',
    'compute_all_features_with_gpu',
    'FeatureComputerFactory',
    'BaseFeatureComputer',
    'ARCHITECTURAL_STYLES',
    'STYLE_NAME_TO_ID',
    'CHARACTERISTIC_TO_STYLE',
]
