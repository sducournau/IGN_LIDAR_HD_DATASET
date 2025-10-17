"""
Feature extraction modules for IGN LiDAR HD.

This package provides unified feature computation with mode-based selection:

Unified API (recommended):
    from ign_lidar.features import compute_verticality, extract_geometric_features
    
    # Mode-based selection
    verticality = compute_verticality(normals, mode='cpu')
    verticality = compute_verticality(normals, mode='gpu') 
    features = extract_geometric_features(points, normals, mode='auto')

Legacy modules (deprecated):
- features: CPU-based feature extraction
- features_gpu: GPU-accelerated feature extraction (CuPy)
- features_gpu_chunked: GPU feature extraction for large files  
- features_boundary: Boundary-aware feature extraction
"""

# Core implementations (V5 consolidated)
from .core import (
    compute_verticality,
    extract_geometric_features,
    compute_all_features,
    compute_normals,
    compute_curvature,
    compute_all_features as core_compute_all_features,
    ComputeMode,
)

# Legacy imports (deprecated - use unified API instead)
from .features import (
    compute_all_features_optimized,
    compute_all_features_with_gpu,
    compute_features_by_mode,
    # Additional feature functions
    compute_eigenvalue_features,
    compute_architectural_features,
    compute_density_features,
    compute_building_scores,
    # Enhanced geometric features for building classification
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
    # Unified API (recommended)
    'FeatureMode',
    'compute_verticality',
    'extract_geometric_features', 
    'compute_all_features',
    'compute_normals',
    'compute_curvature',
    
    # Core implementations
    'core_compute_all_features',
    'ComputeMode',
    
    # Legacy functions (deprecated)
    'compute_all_features_optimized',
    'compute_all_features_with_gpu',
    'compute_features_by_mode',
    'compute_eigenvalue_features',
    'compute_architectural_features',
    'compute_density_features',
    'compute_building_scores',
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
