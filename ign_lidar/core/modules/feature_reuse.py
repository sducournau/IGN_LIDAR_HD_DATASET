"""
Feature Reuse Module

This module provides functionality to detect and reuse existing computed features
from LAZ files to avoid redundant computation. Particularly useful when:
- Processing already-enriched LAZ files
- Re-processing with different parameters but same features
- Continuing interrupted workflows

Key Features:
- Detect existing features in LAZ extra dimensions
- Optionally override or preserve existing values
- Configurable reuse policies per feature type
- Performance optimization by skipping computation

Author: imagodata
Date: October 17, 2025
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import laspy
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    laspy = None


# ============================================================================
# Feature Categories
# ============================================================================

# Features that are always reusable (stable across processing)
STABLE_FEATURES = {
    'intensity', 'return_number', 'classification',
    'red', 'green', 'blue', 'nir'
}

# Geometric features that depend on k_neighbors parameter
NEIGHBOR_DEPENDENT_FEATURES = {
    'normal_x', 'normal_y', 'normal_z',
    'curvature', 'planarity', 'linearity', 'sphericity',
    'anisotropy', 'eigenentropy', 'omnivariance',
    'sum_eigenvalues', 'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
    'density', 'local_point_density', 'neighborhood_extent'
}

# Features that depend on ground classification (height-based)
GROUND_DEPENDENT_FEATURES = {
    'height', 'height_above_ground', 'height_extent_ratio'
}

# Radiometric features (depend on RGB/NIR availability)
RADIOMETRIC_FEATURES = {
    'red', 'green', 'blue', 'nir', 'ndvi'
}

# All reusable geometric features
GEOMETRIC_FEATURES = NEIGHBOR_DEPENDENT_FEATURES | GROUND_DEPENDENT_FEATURES


@dataclass
class FeatureReusePolicy:
    """Policy for reusing existing features from LAZ files.
    
    Attributes:
        reuse_rgb: Reuse existing RGB if available (default: True)
        reuse_nir: Reuse existing NIR if available (default: True)
        reuse_normals: Reuse existing normals if available (default: False)
        reuse_curvature: Reuse existing curvature if available (default: False)
        reuse_height: Reuse existing height if available (default: False)
        reuse_geometric: Reuse all geometric features if available (default: False)
        reuse_all: Override to reuse everything if available (default: False)
        override_rgb: Force recompute RGB even if present (default: False)
        override_nir: Force recompute NIR even if present (default: False)
        override_normals: Force recompute normals even if present (default: False)
        override_all: Force recompute all features (default: False)
        check_k_neighbors: Only reuse if k_neighbors matches stored value (default: True)
        k_neighbors_tolerance: Allow k_neighbors to differ by this amount (default: 0)
    """
    # Reuse policies (what to reuse if available)
    reuse_rgb: bool = True
    reuse_nir: bool = True
    reuse_normals: bool = False
    reuse_curvature: bool = False
    reuse_height: bool = False
    reuse_geometric: bool = False
    reuse_all: bool = False
    
    # Override policies (force recomputation)
    override_rgb: bool = False
    override_nir: bool = False
    override_normals: bool = False
    override_all: bool = False
    
    # Validation policies
    check_k_neighbors: bool = True
    k_neighbors_tolerance: int = 0
    min_points_threshold: int = 100  # Skip reuse if feature has too few valid points
    
    def __post_init__(self):
        """Apply override flags."""
        if self.reuse_all:
            self.reuse_rgb = True
            self.reuse_nir = True
            self.reuse_normals = True
            self.reuse_curvature = True
            self.reuse_height = True
            self.reuse_geometric = True
        
        if self.override_all:
            self.override_rgb = True
            self.override_nir = True
            self.override_normals = True


@dataclass
class FeatureInventory:
    """Inventory of available features in a LAZ file.
    
    Attributes:
        available_standard: Standard LAS/LAZ fields (RGB, NIR, classification, etc.)
        available_extra: Extra dimension feature names
        normals_available: Whether normals (normal_x/y/z) are present
        rgb_available: Whether RGB is present
        nir_available: Whether NIR is present
        geometric_available: Set of available geometric features
        metadata: Metadata about features (k_neighbors, etc.)
    """
    available_standard: Set[str] = field(default_factory=set)
    available_extra: Set[str] = field(default_factory=set)
    normals_available: bool = False
    rgb_available: bool = False
    nir_available: bool = False
    geometric_available: Set[str] = field(default_factory=set)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    @property
    def has_any_features(self) -> bool:
        """Check if any extra features are available."""
        return len(self.available_extra) > 0
    
    @property
    def has_normals(self) -> bool:
        """Check if normals are available."""
        return self.normals_available
    
    @property
    def has_rgb(self) -> bool:
        """Check if RGB is available."""
        return self.rgb_available
    
    @property
    def has_nir(self) -> bool:
        """Check if NIR is available."""
        return self.nir_available


# ============================================================================
# Feature Detection
# ============================================================================

def detect_available_features(laz_path: Path) -> Optional[FeatureInventory]:
    """
    Detect what features are available in a LAZ file.
    
    Args:
        laz_path: Path to LAZ file
        
    Returns:
        FeatureInventory object, or None if file cannot be read
    """
    if not LASPY_AVAILABLE:
        logger.warning("laspy not available - cannot detect features")
        return None
    
    try:
        with laspy.open(str(laz_path)) as f:
            las = f.read()
    except Exception as e:
        logger.debug(f"Could not read LAZ file {laz_path}: {e}")
        return None
    
    inventory = FeatureInventory()
    
    # Check standard fields
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        inventory.available_standard.update(['red', 'green', 'blue'])
        inventory.rgb_available = True
    
    if hasattr(las, 'nir'):
        inventory.available_standard.add('nir')
        inventory.nir_available = True
    
    if hasattr(las, 'classification'):
        inventory.available_standard.add('classification')
    
    if hasattr(las, 'intensity'):
        inventory.available_standard.add('intensity')
    
    # Check extra dimensions
    if hasattr(las.point_format, 'extra_dimension_names'):
        extra_dims = set(las.point_format.extra_dimension_names)
        inventory.available_extra = extra_dims
        
        # Check for normals
        if {'normal_x', 'normal_y', 'normal_z'}.issubset(extra_dims):
            inventory.normals_available = True
        
        # Identify geometric features
        inventory.geometric_available = extra_dims & GEOMETRIC_FEATURES
    
    # Try to extract metadata (k_neighbors, etc.) if stored
    # Look for metadata in VLR (Variable Length Records) or extra bytes descriptions
    try:
        if hasattr(las, 'vlrs'):
            for vlr in las.vlrs:
                if vlr.description and 'k_neighbors' in vlr.description:
                    # Try to parse k_neighbors from description
                    parts = vlr.description.split('k_neighbors=')
                    if len(parts) > 1:
                        try:
                            k_val = int(parts[1].split()[0])
                            inventory.metadata['k_neighbors'] = k_val
                        except:
                            pass
    except Exception as e:
        logger.debug(f"Could not extract metadata: {e}")
    
    return inventory


def load_existing_features(
    laz_path: Path,
    feature_names: Set[str]
) -> Dict[str, np.ndarray]:
    """
    Load existing features from a LAZ file.
    
    Args:
        laz_path: Path to LAZ file
        feature_names: Set of feature names to load
        
    Returns:
        Dictionary mapping feature names to numpy arrays
    """
    if not LASPY_AVAILABLE:
        logger.warning("laspy not available - cannot load features")
        return {}
    
    try:
        with laspy.open(str(laz_path)) as f:
            las = f.read()
    except Exception as e:
        logger.warning(f"Could not read LAZ file {laz_path}: {e}")
        return {}
    
    features = {}
    
    for feature_name in feature_names:
        try:
            if hasattr(las, feature_name):
                data = getattr(las, feature_name)
                
                # Normalize if needed
                if feature_name in {'red', 'green', 'blue', 'nir'}:
                    features[feature_name] = np.array(data, dtype=np.float32) / 65535.0
                else:
                    features[feature_name] = np.array(data, dtype=np.float32)
            elif feature_name in las.point_format.dimension_names:
                data = getattr(las, feature_name)
                features[feature_name] = np.array(data, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Could not load feature {feature_name}: {e}")
            continue
    
    # Special handling for normals - combine into single array
    if {'normal_x', 'normal_y', 'normal_z'}.issubset(feature_names):
        if all(f in features for f in ['normal_x', 'normal_y', 'normal_z']):
            normals = np.vstack([
                features['normal_x'],
                features['normal_y'],
                features['normal_z']
            ]).T
            features['normals'] = normals
            # Remove individual components
            del features['normal_x']
            del features['normal_y']
            del features['normal_z']
    
    return features


# ============================================================================
# Feature Reuse Decision Logic
# ============================================================================

def should_reuse_feature(
    feature_name: str,
    policy: FeatureReusePolicy,
    inventory: FeatureInventory,
    current_k_neighbors: Optional[int] = None
) -> bool:
    """
    Determine if a feature should be reused from existing LAZ file.
    
    Args:
        feature_name: Name of feature to check
        policy: FeatureReusePolicy with reuse settings
        inventory: FeatureInventory from LAZ file
        current_k_neighbors: Current k_neighbors setting
        
    Returns:
        True if feature should be reused, False if it should be recomputed
    """
    # Check if override is forced
    if policy.override_all:
        return False
    
    if feature_name in RADIOMETRIC_FEATURES:
        if feature_name in {'red', 'green', 'blue'}:
            if policy.override_rgb:
                return False
            return policy.reuse_rgb and inventory.rgb_available
        elif feature_name == 'nir':
            if policy.override_nir:
                return False
            return policy.reuse_nir and inventory.nir_available
        elif feature_name == 'ndvi':
            # NDVI should be recomputed if RGB or NIR are being recomputed
            return policy.reuse_nir and policy.reuse_rgb
    
    if feature_name in {'normal_x', 'normal_y', 'normal_z', 'normals'}:
        if policy.override_normals:
            return False
        if not (policy.reuse_normals or policy.reuse_all):
            return False
        if not inventory.normals_available:
            return False
        # Check k_neighbors compatibility
        if policy.check_k_neighbors and current_k_neighbors is not None:
            stored_k = inventory.metadata.get('k_neighbors')
            if stored_k is not None:
                diff = abs(stored_k - current_k_neighbors)
                if diff > policy.k_neighbors_tolerance:
                    logger.info(
                        f"  ℹ️  Skipping normals reuse: k_neighbors mismatch "
                        f"(stored={stored_k}, current={current_k_neighbors})"
                    )
                    return False
        return True
    
    if feature_name == 'curvature':
        return (policy.reuse_curvature or policy.reuse_all) and \
               feature_name in inventory.available_extra
    
    if feature_name in GROUND_DEPENDENT_FEATURES:
        return (policy.reuse_height or policy.reuse_all) and \
               feature_name in inventory.available_extra
    
    if feature_name in GEOMETRIC_FEATURES:
        return (policy.reuse_geometric or policy.reuse_all) and \
               feature_name in inventory.available_extra
    
    # Default: don't reuse unknown features
    return False


def create_reuse_plan(
    laz_path: Path,
    requested_features: Set[str],
    policy: FeatureReusePolicy,
    current_k_neighbors: Optional[int] = None
) -> Tuple[Set[str], Set[str], FeatureInventory]:
    """
    Create a plan for which features to reuse and which to compute.
    
    Args:
        laz_path: Path to LAZ file
        requested_features: Set of all features that are needed
        policy: FeatureReusePolicy with reuse settings
        current_k_neighbors: Current k_neighbors setting
        
    Returns:
        Tuple of (features_to_reuse, features_to_compute, inventory)
    """
    inventory = detect_available_features(laz_path)
    if inventory is None:
        # Cannot read file, compute everything
        return set(), requested_features, FeatureInventory()
    
    features_to_reuse = set()
    features_to_compute = set()
    
    for feature_name in requested_features:
        if should_reuse_feature(feature_name, policy, inventory, current_k_neighbors):
            features_to_reuse.add(feature_name)
        else:
            features_to_compute.add(feature_name)
    
    return features_to_reuse, features_to_compute, inventory


# ============================================================================
# Logging and Reporting
# ============================================================================

def log_reuse_plan(
    features_to_reuse: Set[str],
    features_to_compute: Set[str],
    logger_instance: Optional[logging.Logger] = None
):
    """
    Log the feature reuse plan.
    
    Args:
        features_to_reuse: Features that will be reused
        features_to_compute: Features that will be computed
        logger_instance: Optional logger instance
    """
    log = logger_instance or logger
    
    if features_to_reuse:
        log.info(f"  ♻️  Reusing {len(features_to_reuse)} existing features: "
                f"{', '.join(sorted(features_to_reuse))}")
    
    if features_to_compute:
        log.info(f"  🔧 Computing {len(features_to_compute)} features: "
                f"{', '.join(sorted(features_to_compute))}")
    
    if not features_to_compute:
        log.info("  ✨ All features available - no computation needed!")


# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Data classes
    'FeatureReusePolicy',
    'FeatureInventory',
    
    # Feature sets
    'STABLE_FEATURES',
    'NEIGHBOR_DEPENDENT_FEATURES',
    'GROUND_DEPENDENT_FEATURES',
    'RADIOMETRIC_FEATURES',
    'GEOMETRIC_FEATURES',
    
    # Functions
    'detect_available_features',
    'load_existing_features',
    'should_reuse_feature',
    'create_reuse_plan',
    'log_reuse_plan',
]
