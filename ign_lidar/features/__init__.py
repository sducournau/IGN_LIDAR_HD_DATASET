"""
Feature extraction modules for IGN LiDAR HD.

This package provides unified feature computation with Strategy pattern (Week 2 refactoring):

Strategy Pattern API (NEW - Week 2):
    from ign_lidar.features import BaseFeatureStrategy, CPUStrategy, GPUStrategy
    
    # Automatic strategy selection
    strategy = BaseFeatureStrategy.auto_select(n_points=1_000_000, mode='auto')
    features = strategy.compute(points, intensities, rgb, nir)
    
    # Manual strategy selection
    strategy = GPUChunkedStrategy(chunk_size=5_000_000, batch_size=250_000)
    features = strategy.compute(points)

Unified API (recommended):
    from ign_lidar.features import compute_verticality, extract_geometric_features
    
    # Mode-based selection
    verticality = compute_verticality(normals, mode='cpu')
    verticality = compute_verticality(normals, mode='gpu') 
    features = extract_geometric_features(points, normals, mode='auto')

Legacy modules (deprecated - will be removed):
- features: CPU-based feature extraction
- features_gpu: GPU-accelerated feature extraction (CuPy)
- features_gpu_chunked: GPU feature extraction for large files  
- features_boundary: Boundary-aware feature extraction
- factory: Factory pattern (replaced by Strategy pattern)
"""

# Strategy Pattern (NEW - Week 2 refactoring)
from .strategies import (
    BaseFeatureStrategy,
    FeatureComputeMode,
    estimate_optimal_batch_size,
)
from .strategy_cpu import CPUStrategy
try:
    from .strategy_gpu import GPUStrategy
    from .strategy_gpu_chunked import GPUChunkedStrategy
except ImportError:
    # GPU strategies not available without CuPy
    GPUStrategy = None
    GPUChunkedStrategy = None
from .strategy_boundary import BoundaryAwareStrategy

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

# Import optimized feature computation (preferred implementation)
try:
    from .core.features import compute_all_features as compute_all_features_optimized
except ImportError:
    # Fallback if Numba not available
    compute_all_features_optimized = None
    
# Additional core features
from .core.eigenvalues import compute_eigenvalue_features
from .core.architectural import (
    compute_architectural_features,
    compute_horizontality,
    compute_facade_score,
)
from .core.density import compute_density_features

# Note: The following functions were removed during Phase 2 cleanup:
# - compute_all_features_with_gpu -> Use GPUStrategy instead
# - compute_features_by_mode -> Use Strategy pattern (BaseFeatureStrategy.auto_select)
# - compute_building_scores -> Not found in core modules
# - compute_edge_strength -> Not found in core modules  
# - compute_roof_plane_score -> Defined but never used
# - compute_opening_likelihood -> Defined but never used
# - compute_structural_element_score -> Defined but never used

# Factory Pattern has been removed - use Strategy Pattern instead
# All functionality moved to Strategy pattern (strategies.py, strategy_*.py)

from .orchestrator import (
    FeatureOrchestrator,
)

# FeatureComputer (Phase 4 - automatic mode selection)
try:
    from .feature_computer import (
        FeatureComputer,
        get_feature_computer,
    )
    FEATURE_COMPUTER_AVAILABLE = True
except ImportError:
    FeatureComputer = None
    get_feature_computer = None
    FEATURE_COMPUTER_AVAILABLE = False

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
    # FeatureComputer (Phase 4 - automatic mode selection)
    'FeatureComputer',
    'get_feature_computer',
    
    # Strategy Pattern (NEW - Week 2)
    'BaseFeatureStrategy',
    'FeatureComputeMode',
    'CPUStrategy',
    'GPUStrategy',
    'GPUChunkedStrategy',
    'BoundaryAwareStrategy',
    'estimate_optimal_batch_size',
    
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
    
    # Additional core features (available)
    'compute_all_features_optimized',
    'compute_eigenvalue_features',
    'compute_architectural_features',
    'compute_density_features',
    'compute_horizontality',
    'compute_facade_score',
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
