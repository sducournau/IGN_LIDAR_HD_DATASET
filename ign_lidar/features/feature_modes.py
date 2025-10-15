"""
Feature Computation Modes for LOD2/LOD3/ASPRS Training

This module defines feature sets and computation modes optimized for different
levels of detail and training objectives:

- MINIMAL: Ultra-fast minimal features for quick processing (~8 features)
- LOD2_SIMPLIFIED: Essential features for basic building detection (~17 features)
- LOD3_FULL: Complete feature set for detailed architectural modeling (~43 features)
- ASPRS_CLASSES: Optimized features for ASPRS classification (~30 features)
- FULL: All available features (~45 features)
- CUSTOM: User-defined feature selection

Feature Categories:
1. Core Geometric Features: normals, curvature, shape descriptors
2. Eigenvalue Features: eigenvalues, sum, entropy
3. Building-Specific Features: height, wall/roof scores, architectural features
4. Density & Neighborhood: local density, point counts, spatial extents
5. Spectral Features: RGB, NIR, NDVI

References:
- Weinmann et al. (2015) - Geometric feature formulas
- Demantké et al. (2011) - Eigenvalue-based descriptors
- ASPRS LAS 1.4 Specification - Classification codes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'FeatureMode',
    'FeatureSet',
    'get_feature_config',
    'FEATURE_DESCRIPTIONS',
    'LOD2_FEATURES',
    'LOD3_FEATURES',
    'ASPRS_FEATURES',
]


class FeatureMode(Enum):
    """Predefined feature computation modes."""
    MINIMAL = "minimal"              # Ultra-fast: ~8 features
    LOD2_SIMPLIFIED = "lod2"        # Essential: ~11 features for LOD2
    LOD3_FULL = "lod3"              # Complete: ~35 features for LOD3
    ASPRS_CLASSES = "asprs_classes" # ASPRS classification: ~25 features
    FULL = "full"                   # All available features
    CUSTOM = "custom"               # User-defined selection


# =============================================================================
# FEATURE DESCRIPTIONS
# =============================================================================

FEATURE_DESCRIPTIONS = {
    # Core Coordinates
    'xyz': 'Point coordinates (X, Y, Z)',
    
    # Surface Normals (3 features)
    'normal_x': 'X component of surface normal',
    'normal_y': 'Y component of surface normal', 
    'normal_z': 'Z component of surface normal (verticality indicator)',
    
    # Curvature Features (2 features)
    'curvature': 'Local surface curvature',
    'change_curvature': 'Rate of curvature change (using eigenvalue variance)',
    
    # Shape Descriptors (6 features) - Weinmann et al.
    'planarity': 'Planarity measure [0,1] - high for flat surfaces (roofs, walls)',
    'linearity': 'Linearity measure [0,1] - high for edges, cables',
    'sphericity': 'Sphericity measure [0,1] - high for vegetation, noise',
    'roughness': 'Surface roughness [0,1] - texture measure',
    'anisotropy': 'Anisotropy measure [0,1] - directional variation',
    'omnivariance': 'Omnivariance - 3D dispersion measure',
    
    # Eigenvalues (5 features)
    'eigenvalue_1': 'Largest eigenvalue (λ₀)',
    'eigenvalue_2': 'Medium eigenvalue (λ₁)',
    'eigenvalue_3': 'Smallest eigenvalue (λ₂)',
    'sum_eigenvalues': 'Sum of eigenvalues (Σλ)',
    'eigenentropy': 'Shannon entropy of eigenvalues',
    
    # Height Features (2 features)
    'height_above_ground': 'Height above ground level (meters)',
    'vertical_std': 'Standard deviation of Z in neighborhood',
    
    # Building-Specific Scores (legacy)
    'verticality': 'Verticality score [0,1] - 1 for vertical surfaces',
    'wall_score': 'Wall likelihood (planarity × verticality) - LEGACY',
    'roof_score': 'Roof likelihood (planarity × horizontality) - LEGACY',
    
    # Canonical Architectural Features (from architectural.py)
    'horizontality': 'Horizontality score [0,1] - 1 for horizontal surfaces',
    'wall_likelihood': 'Wall probability (canonical from architectural.py)',
    'roof_likelihood': 'Roof probability (canonical from architectural.py)',
    'facade_score': 'Facade characteristic score (verticality + height + planarity)',
    
    # Density & Neighborhood (4 features)
    'density': 'Local point density (points per unit volume)',
    'num_points_2m': 'Number of points within 2m radius',
    'neighborhood_extent': 'Maximum distance to k-th neighbor',
    'height_extent_ratio': 'Ratio of vertical std to spatial extent',
    
    # Advanced Architectural Features (4 features)
    'edge_strength': 'Edge detection strength (high eigenvalue variance)',
    'corner_likelihood': 'Corner probability (3D structure measure)',
    'overhang_indicator': 'Overhang/protrusion detection',
    'surface_roughness': 'Fine-scale surface texture',
    
    # NEW: Enhanced Building Classification Features (8 features)
    'horizontality': 'Horizontality score [0,1] - 1 for horizontal surfaces (roofs)',
    'facade_score': 'Facade likelihood (combined vertical + planar + elevated)',
    'flat_roof_score': 'Flat roof likelihood (horizontal + planar + elevated)',
    'sloped_roof_score': 'Sloped roof likelihood (15-45° tilt)',
    'steep_roof_score': 'Steep roof likelihood (45-70° tilt)',
    'opening_likelihood': 'Window/door likelihood (low planarity + high linearity)',
    'structural_element_score': 'Pillar/column likelihood (linear + vertical + organized)',
    'edge_strength_enhanced': 'Enhanced edge strength from eigenvalues (λ1-λ2)/λ1',
    
    # Spectral Features (5 features)
    'red': 'Red channel [0-255]',
    'green': 'Green channel [0-255]',
    'blue': 'Blue channel [0-255]',
    'nir': 'Near-infrared channel [0-255]',
    'ndvi': 'Normalized Difference Vegetation Index [-1,1]',
}


# =============================================================================
# FEATURE SETS BY MODE
# =============================================================================

# MINIMAL: Ultra-fast processing (~8 features)
MINIMAL_FEATURES = {
    'normal_z',           # Verticality from normal
    'planarity',          # Main shape descriptor
    'height_above_ground',# Essential for building detection
    'density',            # Local point density
}

# LOD2_SIMPLIFIED: Essential features for basic building classification (~17 features)
LOD2_FEATURES = {
    # Coordinates
    'xyz',                # 3 features (x, y, z)
    
    # Essential geometric
    'normal_z',           # Verticality indicator
    'planarity',          # Flat surface detection (from architectural.py)
    'linearity',          # Edge detection
    'anisotropy',         # Structure detection
    
    # Building-specific
    'height_above_ground',# Height feature
    'verticality',        # Wall detection (from architectural.py)
    'horizontality',      # Roof detection (from architectural.py)
    
    # Architectural features (from canonical architectural.py)
    'wall_likelihood',    # Wall probability (from architectural.py)
    'roof_likelihood',    # Roof probability (from architectural.py)
    'facade_score',       # Facade detection (from architectural.py)
    
    # Legacy building scores (for compatibility)
    'wall_score',         # Legacy wall likelihood
    'roof_score',         # Legacy roof likelihood
    
    # Spectral
    'red', 'green', 'blue',  # RGB colors (3 features)
    'ndvi',               # Vegetation index
}  # Total: 19 features (xyz=3, geometric=4, building=10, spectral=4)

# LOD3_FULL: Complete feature set for detailed modeling (~43 features)
LOD3_FEATURES = {
    # Coordinates
    'xyz',                # 3 features
    
    # Normals (3 features)
    'normal_x',
    'normal_y', 
    'normal_z',
    
    # Curvature (2 features)
    'curvature',
    'change_curvature',
    
    # Shape descriptors (6 features)
    'planarity',
    'linearity',
    'sphericity',
    'roughness',
    'anisotropy',
    'omnivariance',
    
    # Eigenvalues (5 features)
    'eigenvalue_1',
    'eigenvalue_2',
    'eigenvalue_3',
    'sum_eigenvalues',
    'eigenentropy',
    
    # Height features (2 features)
    'height_above_ground',
    'vertical_std',
    
    # Building scores (3 features - legacy)
    'verticality',
    'wall_score',
    'roof_score',
    
    # Canonical Architectural Features (from architectural.py)
    'horizontality',        # Horizontal surface detection (from architectural.py)
    'wall_likelihood',      # Wall probability (from architectural.py)
    'roof_likelihood',      # Roof probability (from architectural.py)
    'facade_score',         # Facade detection (from architectural.py)
    
    # Density (4 features)
    'density',
    'num_points_2m',
    'neighborhood_extent',
    'height_extent_ratio',
    
    # Legacy Architectural (4 features - for backward compatibility)
    'legacy_edge_strength',
    'legacy_corner_likelihood',
    'legacy_overhang_indicator',
    'legacy_surface_roughness',
    
    # Additional Building Classification Features (4 features)
    'flat_roof_score',           # Flat roof detection
    'sloped_roof_score',         # Sloped roof detection
    'steep_roof_score',          # Steep roof detection
    'opening_likelihood',        # Window/door detection
    
    # Spectral (5 features)
    'red', 'green', 'blue',
    'nir',
    'ndvi',
}  # Total: ~43 features (was 35, added 8 new building features)

# ASPRS_CLASSES: Features optimized for ASPRS classification (~30 features)
ASPRS_FEATURES = {
    # Coordinates
    'xyz',                # 3 features
    
    # Essential normals for surface orientation
    'normal_x',
    'normal_y',
    'normal_z',           # Critical for ground/vegetation/building separation
    
    # Core shape descriptors (essential for classification)
    'planarity',          # Flat surfaces (ground, roads, roofs)
    'linearity',          # Linear features (wires, edges)
    'sphericity',         # Vegetation detection
    'roughness',          # Surface texture
    'anisotropy',         # Structure detection
    
    # Eigenvalues for advanced classification
    'eigenvalue_1',
    'eigenvalue_2',
    'eigenvalue_3',
    'sum_eigenvalues',
    'eigenentropy',
    
    # Height features (critical for multi-class separation)
    'height_above_ground',# Essential for vegetation height classes
    'vertical_std',       # Local height variation
    
    # Building detection (canonical architectural features)
    'verticality',        # Walls vs ground (from architectural.py)
    'horizontality',      # Ground and flat roofs (from architectural.py)
    'wall_likelihood',    # Wall probability (from architectural.py)
    'roof_likelihood',    # Roof probability (from architectural.py)
    'facade_score',       # Facade detection (from architectural.py)
    
    # Legacy building scores (for compatibility)
    'wall_score',         # Legacy wall score
    'roof_score',         # Legacy roof score
    
    # Density features for classification
    'density',            # Point density varies by class
    'num_points_2m',
    
    # Spectral features (if available)
    'red', 'green', 'blue',  # RGB for better classification
    'nir',                   # NIR for vegetation
    'ndvi',                  # Vegetation index
}  # Total: ~30 features (optimized for ASPRS multi-class classification)


@dataclass
class FeatureSet:
    """
    Configuration for a specific feature computation mode.
    
    Attributes:
        mode: Feature mode identifier
        features: Set of feature names to compute
        requires_rgb: Whether RGB data is needed
        requires_nir: Whether NIR data is needed
        k_neighbors: Recommended number of neighbors
        use_radius: Use radius-based search instead of k-NN
        radius: Search radius in meters (if use_radius=True)
    """
    mode: FeatureMode
    features: Set[str]
    requires_rgb: bool = False
    requires_nir: bool = False
    k_neighbors: int = 20
    use_radius: bool = True
    radius: Optional[float] = None
    
    def __post_init__(self):
        """Validate and compute derived properties."""
        # Check if RGB/NIR are required
        rgb_features = {'red', 'green', 'blue'}
        if rgb_features & self.features:
            self.requires_rgb = True
        if 'nir' in self.features or 'ndvi' in self.features:
            self.requires_nir = True
    
    @property
    def num_features(self) -> int:
        """Total number of features (counting xyz as 3)."""
        count = len(self.features)
        if 'xyz' in self.features:
            count += 2  # xyz counts as 3 features
        return count
    
    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names."""
        # Standard ordering for consistent output
        order = [
            'xyz',
            'normal_x', 'normal_y', 'normal_z',
            'curvature', 'change_curvature',
            'planarity', 'linearity', 'sphericity',
            'roughness', 'anisotropy', 'omnivariance',
            'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
            'sum_eigenvalues', 'eigenentropy',
            'height_above_ground', 'vertical_std',
            'verticality', 'wall_score', 'roof_score',
            'density', 'num_points_2m', 'neighborhood_extent',
            'height_extent_ratio',
            'edge_strength', 'corner_likelihood',
            'overhang_indicator', 'surface_roughness',
            'red', 'green', 'blue', 'nir', 'ndvi',
        ]
        return [f for f in order if f in self.features]
    
    def get_description(self) -> str:
        """Get human-readable description of this feature set."""
        mode_descriptions = {
            FeatureMode.MINIMAL: "Minimal features for fast processing",
            FeatureMode.LOD2_SIMPLIFIED: "Essential features for LOD2 building classification",
            FeatureMode.LOD3_FULL: "Complete features for LOD3 architectural modeling",
            FeatureMode.ASPRS_CLASSES: "Optimized features for ASPRS classification",
            FeatureMode.FULL: "All available features",
            FeatureMode.CUSTOM: "Custom user-defined features",
        }
        desc = mode_descriptions.get(self.mode, "Custom feature set")
        return f"{desc} ({self.num_features} features)"


# =============================================================================
# FEATURE CONFIGURATION FACTORY
# =============================================================================

def get_feature_config(
    mode: str = "lod3",
    custom_features: Optional[Set[str]] = None,
    k_neighbors: int = 20,
    use_radius: bool = True,
    radius: Optional[float] = None,
    has_rgb: Optional[bool] = None,
    has_nir: Optional[bool] = None,
    log_config: bool = True,
) -> FeatureSet:
    """
    Get feature configuration for a specific mode.
    
    Args:
        mode: Feature mode ('minimal', 'lod2', 'lod3', 'full', 'custom')
        custom_features: Set of features for custom mode
        k_neighbors: Number of neighbors for feature computation
        use_radius: Use radius-based search (recommended)
        radius: Search radius in meters (auto-estimated if None)
        has_rgb: Whether RGB data is available (None = unknown, don't log)
        has_nir: Whether NIR data is available (None = unknown, don't log)
        log_config: Whether to log the configuration (default True)
    
    Returns:
        FeatureSet configuration
        
    Examples:
        >>> # LOD3 training with full features
        >>> config = get_feature_config("lod3", k_neighbors=30)
        >>> print(config.num_features)  # ~35
        
        >>> # LOD2 training with essential features
        >>> config = get_feature_config("lod2", k_neighbors=20)
        >>> print(config.num_features)  # ~11
        
        >>> # Custom features
        >>> features = {'xyz', 'normal_z', 'planarity', 'height_above_ground'}
        >>> config = get_feature_config("custom", custom_features=features)
    """
    # Convert string to enum
    try:
        mode_enum = FeatureMode(mode.lower())
    except ValueError:
        logger.warning(f"Unknown mode '{mode}', defaulting to LOD3_FULL")
        mode_enum = FeatureMode.LOD3_FULL
    
    # Select feature set
    if mode_enum == FeatureMode.MINIMAL:
        features = MINIMAL_FEATURES.copy()
    elif mode_enum == FeatureMode.LOD2_SIMPLIFIED:
        features = LOD2_FEATURES.copy()
    elif mode_enum == FeatureMode.LOD3_FULL:
        features = LOD3_FEATURES.copy()
    elif mode_enum == FeatureMode.ASPRS_CLASSES:
        features = ASPRS_FEATURES.copy()
    elif mode_enum == FeatureMode.FULL:
        features = set(FEATURE_DESCRIPTIONS.keys())
    elif mode_enum == FeatureMode.CUSTOM:
        if custom_features is None:
            raise ValueError("custom_features must be provided for CUSTOM mode")
        features = custom_features.copy()
    else:
        features = LOD3_FEATURES.copy()
    
    # Create feature set
    feature_set = FeatureSet(
        mode=mode_enum,
        features=features,
        k_neighbors=k_neighbors,
        use_radius=use_radius,
        radius=radius,
    )
    
    # Log configuration if requested
    if log_config:
        logger.info(f"📊 Feature Configuration: {feature_set.get_description()}")
        logger.info(f"   Features: {', '.join(feature_set.feature_names)}")
        
        # Log RGB/NIR requirements with data availability context
        if feature_set.requires_rgb:
            if has_rgb is None:
                # Unknown availability - just state the requirement
                logger.info("   ℹ️  Feature set includes RGB channels")
            elif has_rgb:
                # Data is available - confirmation
                logger.info("   ✓ RGB channels available")
            else:
                # Data is NOT available - warning
                logger.warning("   ⚠️  RGB channels required but not available in input data")
        
        if feature_set.requires_nir:
            if has_nir is None:
                # Unknown availability - just state the requirement
                logger.info("   ℹ️  Feature set includes NIR channel for NDVI")
            elif has_nir:
                # Data is available - confirmation
                logger.info("   ✓ NIR channel available for NDVI")
            else:
                # Data is NOT available - warning
                logger.warning("   ⚠️  NIR channel required for NDVI but not available in input data")
    
    return feature_set


# =============================================================================
# AUGMENTATION-SAFE FEATURES
# =============================================================================

# Features that should NOT be augmented (absolute geometric properties)
AUGMENTATION_INVARIANT_FEATURES = {
    'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
    'sum_eigenvalues', 'eigenentropy',
    'planarity', 'linearity', 'sphericity',
    'anisotropy', 'roughness', 'omnivariance',
    'curvature', 'density',
}

# Features that CAN be safely augmented (relative/invariant properties)
AUGMENTATION_SAFE_FEATURES = {
    'xyz',  # Transformed by augmentation
    'normal_x', 'normal_y', 'normal_z',  # Rotated with points
    'height_above_ground',  # Relative height preserved
    'vertical_std', 'neighborhood_extent',  # Local properties
    'wall_score', 'roof_score', 'verticality',  # Rotation-invariant
    'red', 'green', 'blue', 'nir', 'ndvi',  # Colors unchanged
}


def get_augmentation_strategy(feature_set: FeatureSet) -> Dict[str, bool]:
    """
    Get augmentation strategy for each feature.
    
    Returns:
        Dictionary mapping feature names to whether they should be augmented
        
    Example:
        >>> config = get_feature_config("lod3")
        >>> strategy = get_augmentation_strategy(config)
        >>> strategy['normal_x']  # True - rotate with points
        >>> strategy['eigenvalue_1']  # False - don't augment
    """
    strategy = {}
    for feature in feature_set.features:
        # Default: augment if in safe list, otherwise don't
        if feature in AUGMENTATION_SAFE_FEATURES:
            strategy[feature] = True
        elif feature in AUGMENTATION_INVARIANT_FEATURES:
            strategy[feature] = False
        else:
            # Unknown feature - be conservative
            strategy[feature] = False
            logger.warning(f"Unknown feature '{feature}' - not augmenting")
    
    return strategy
