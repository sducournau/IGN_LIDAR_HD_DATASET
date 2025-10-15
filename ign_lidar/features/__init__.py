"""
Feature extraction modules for IGN LiDAR HD.

This package contains geometric and radiometric feature calculation:
- features: CPU-based feature extraction (includes all geometric features)
- features_gpu: GPU-accelerated feature extraction (CuPy)
- features_gpu_chunked: GPU feature extraction for large files
- features_boundary: Boundary-aware feature extraction
- feature_modes: LOD2/LOD3 feature configuration
- factory: Factory for creating feature computers
- architectural_styles: Architectural style classification system
"""

from .features import (
    compute_normals,
    compute_curvature,
    extract_geometric_features,
    compute_all_features_optimized,
    compute_all_features_with_gpu,
    compute_features_by_mode,
    # Additional feature functions
    compute_eigenvalue_features,
    compute_architectural_features,
    compute_density_features,
    compute_verticality,
    compute_building_scores,
    # NEW: Enhanced geometric features for building classification
    compute_horizontality,
    compute_edge_strength,
    compute_facade_score,
    compute_roof_plane_score,
    compute_opening_likelihood,
    compute_structural_element_score,
)
from .factory import (
    FeatureComputerFactory,
    BaseFeatureComputer,
)
from .orchestrator import (
    FeatureOrchestrator,
)
from .architectural_styles import (
    ARCHITECTURAL_STYLES,
    STYLE_NAME_TO_ID,
    CHARACTERISTIC_TO_STYLE,
    get_tile_architectural_style,
    get_patch_architectural_style,
    compute_architectural_style_features,
    get_architectural_style_id,
    get_style_name,
    infer_multi_styles_from_characteristics,
)
from .feature_modes import (
    FeatureMode,
    FeatureSet,
    get_feature_config,
    LOD2_FEATURES,
    LOD3_FEATURES,
    FEATURE_DESCRIPTIONS,
)

__all__ = [
    # Core feature functions
    'compute_normals',
    'compute_curvature',
    'extract_geometric_features',
    'compute_all_features_optimized',
    'compute_all_features_with_gpu',
    'compute_features_by_mode',
    # Additional feature functions (merged from features_enhanced)
    'compute_eigenvalue_features',
    'compute_architectural_features',
    'compute_density_features',
    'compute_verticality',
    'compute_building_scores',
    # NEW: Enhanced geometric features for building classification
    'compute_horizontality',
    'compute_edge_strength',
    'compute_facade_score',
    'compute_roof_plane_score',
    'compute_opening_likelihood',
    'compute_structural_element_score',
    # Factory
    'FeatureComputerFactory',
    'BaseFeatureComputer',
    # Orchestrator (Phase 4)
    'FeatureOrchestrator',
    # Architectural styles
    'ARCHITECTURAL_STYLES',
    'STYLE_NAME_TO_ID',
    'CHARACTERISTIC_TO_STYLE',
    'get_tile_architectural_style',
    'get_patch_architectural_style',
    'compute_architectural_style_features',
    'get_architectural_style_id',
    'get_style_name',
    'infer_multi_styles_from_characteristics',
    # Feature modes
    'FeatureMode',
    'FeatureSet',
    'get_feature_config',
    'LOD2_FEATURES',
    'LOD3_FEATURES',
    'FEATURE_DESCRIPTIONS',
]
