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
# Note: Moved from .core to .compute in v3.1.0
from .compute import (
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
    from .compute.features import compute_all_features as compute_all_features_optimized
except ImportError:
    # Fallback if Numba not available
    compute_all_features_optimized = None
    
# Additional core features
from .compute.eigenvalues import compute_eigenvalue_features
from .compute.architectural import (
    compute_architectural_features,
    compute_horizontality,
    compute_facade_score,
)
from .compute.density import compute_density_features

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

# Backward compatibility aliases (v3.x only - will be removed in v4.0)
# These allow old code to continue working with deprecation warnings
try:
    from .gpu_processor import GPUProcessor
    # Aliases for deprecated classes
    GPUFeatureComputer = GPUProcessor  # Alias for features_gpu.GPUFeatureComputer
    GPUFeatureComputerChunked = GPUProcessor  # Alias for features_gpu_chunked.GPUChunkedFeatureComputer
    GPUChunkedFeatureComputer = GPUProcessor  # Alternative alias
except ImportError:
    GPUProcessor = None
    GPUFeatureComputer = None
    GPUFeatureComputerChunked = None
    GPUChunkedFeatureComputer = None

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
    
    # Backward compatibility aliases (v3.x - deprecated, remove in v4.0)
    'GPUProcessor',
    'GPUFeatureComputer',  # Deprecated alias for GPUProcessor
    'GPUFeatureComputerChunked',  # Deprecated alias for GPUProcessor
    'GPUChunkedFeatureComputer',  # Deprecated alias for GPUProcessor
    
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

# Backward compatibility: features.core moved to features.compute in v3.1.0
# This allows old imports to continue working with a deprecation warning
import sys
import warnings
from types import ModuleType


class _CoreCompatibilityModule(ModuleType):
    """
    Compatibility shim for features.core → features.compute rename.
    
    This allows code using the old path to continue working:
        from ign_lidar.features.core.eigenvalues import compute_eigenvalues
    
    While showing a deprecation warning guiding users to the new path:
        from ign_lidar.features.compute.eigenvalues import compute_eigenvalues
    """
    
    def __getattr__(self, name):
        # Handle special module attributes without deprecation warnings
        if name in ('__path__', '__file__', '__package__', '__spec__', '__loader__', '__cached__'):
            # Get the actual compute module to access its attributes
            import importlib
            compute_module = importlib.import_module('ign_lidar.features.compute')
            return getattr(compute_module, name, None)
        
        # For all other attributes, show deprecation warning
        warnings.warn(
            f"Importing from 'ign_lidar.features.core' is deprecated. "
            f"Use 'ign_lidar.features.compute' instead. "
            f"The 'features.core' path will be removed in v4.0.0.\n"
            f"  OLD: from ign_lidar.features.core.{name} import ...\n"
            f"  NEW: from ign_lidar.features.compute.{name} import ...",
            DeprecationWarning,
            stacklevel=2
        )
        # Import the actual module from new location
        import importlib
        module_path = f'ign_lidar.features.compute.{name}'
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' from 'ign_lidar.features.core' (now 'ign_lidar.features.compute'). "
                f"Original error: {e}"
            ) from e


# Register the compatibility module
sys.modules['ign_lidar.features.core'] = _CoreCompatibilityModule('ign_lidar.features.core')
