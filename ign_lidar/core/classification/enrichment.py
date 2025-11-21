"""
Feature Enrichment Module

This module handles all feature enrichment operations for LiDAR point clouds:
- RGB color fetching from orthophotos
- Infrared/NIR fetching from orthophotos
- NDVI computation from RGB + NIR
- Geometric feature computation (normals, curvature, planarity, etc.)
- Boundary-aware processing with tile stitching
- GPU/CPU processing paths

Extracted from processor.py as part of Phase 4 refactoring.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ...utils.normalization import normalize_rgb, normalize_nir

# Optional dependency handling
try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    laspy = None

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class EnrichmentConfig:
    """Configuration for point cloud enrichment.
    
    Attributes:
        include_rgb: Whether to fetch RGB colors from orthophotos
        include_infrared: Whether to fetch NIR from infrared orthophotos
        compute_ndvi: Whether to compute NDVI from RGB + NIR
        include_extra_features: Whether to include extra geometric features
        k_neighbors: Number of neighbors for feature computation (None = auto)
        use_gpu: Whether to use GPU for feature computation
        use_gpu_chunked: Whether to use chunked GPU processing for large datasets
        gpu_batch_size: Batch size for GPU chunked processing
        use_stitching: Whether to use boundary-aware tile stitching
        reuse_existing_features: Whether to reuse features from LAZ if available
        override_rgb: Force recompute RGB even if present in LAZ
        override_nir: Force recompute NIR even if present in LAZ
        override_normals: Force recompute normals even if present in LAZ
        override_all_features: Force recompute all features (ignore existing)
    """
    include_rgb: bool = True
    include_infrared: bool = False
    compute_ndvi: bool = False
    include_extra_features: bool = True
    k_neighbors: Optional[int] = None
    use_gpu: bool = False
    use_gpu_chunked: bool = True
    gpu_batch_size: int = 100_000
    use_stitching: bool = False
    # Feature reuse options (NEW)
    reuse_existing_features: bool = True
    override_rgb: bool = False
    override_nir: bool = False
    override_normals: bool = False
    override_all_features: bool = False


@dataclass
class EnrichmentResult:
    """Result of enrichment operation.
    
    Attributes:
        rgb: RGB colors [N, 3] normalized to [0, 1], or None
        nir: Near-infrared values [N] normalized to [0, 1], or None
        ndvi: NDVI values [N] in range [-1, 1], or None
        normals: Surface normals [N, 3], or None
        curvature: Curvature values [N], or None
        height: Height above local minimum [N], or None
        geo_features: Dict of geometric features (planarity, linearity, etc.)
        num_boundary_points: Number of points affected by boundary processing
        processing_time: Total processing time in seconds
        used_boundary_aware: Whether boundary-aware processing was used
    """
    rgb: Optional[np.ndarray] = None
    nir: Optional[np.ndarray] = None
    ndvi: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    curvature: Optional[np.ndarray] = None
    height: Optional[np.ndarray] = None
    geo_features: Optional[Dict[str, np.ndarray]] = None
    num_boundary_points: int = 0
    processing_time: float = 0.0
    used_boundary_aware: bool = False


# ============================================================================
# RGB/NIR Fetching Functions
# ============================================================================

def fetch_rgb_colors(
    points: np.ndarray,
    rgb_fetcher: Any,
    logger_instance: Optional[logging.Logger] = None
) -> Optional[np.ndarray]:
    """Fetch RGB colors from orthophotos using the RGB fetcher.
    
    Args:
        points: Point cloud [N, 3] with XYZ coordinates
        rgb_fetcher: RGB fetcher instance (IGNRGBFetcher)
        logger_instance: Optional logger instance
        
    Returns:
        RGB colors [N, 3] normalized to [0, 1], or None if failed
    """
    log = logger_instance or logger
    
    if rgb_fetcher is None:
        log.warning("RGB fetcher not available")
        return None
    
    log.info("  🎨 Fetching RGB from IGN orthophotos...")
    try:
        # Calculate tile bounding box
        tile_bbox = (
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max()
        )
        
        # Fetch RGB using the fetcher
        rgb = rgb_fetcher.augment_points_with_rgb(points, bbox=tile_bbox)
        
        if rgb is not None:
            # Normalize to [0, 1] using utility
            rgb = normalize_rgb(rgb, use_gpu=False)
            log.info(f"  ✓ RGB fetched from orthophotos")
            return rgb
        else:
            log.warning(f"  ⚠️  RGB fetch returned None")
            return None
            
    except Exception as e:
        log.warning(f"  ⚠️  RGB fetch failed: {e}")
        return None


def fetch_infrared(
    points: np.ndarray,
    infrared_fetcher: Any,
    logger_instance: Optional[logging.Logger] = None
) -> Optional[np.ndarray]:
    """Fetch NIR (near-infrared) values from infrared orthophotos.
    
    Args:
        points: Point cloud [N, 3] with XYZ coordinates
        infrared_fetcher: Infrared fetcher instance
        logger_instance: Optional logger instance
        
    Returns:
        NIR values [N] normalized to [0, 1], or None if failed
    """
    log = logger_instance or logger
    
    if infrared_fetcher is None:
        log.warning("Infrared fetcher not available")
        return None
    
    log.info("  📡 Fetching NIR from infrared orthophotos...")
    try:
        # Calculate tile bounding box
        tile_bbox = (
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max()
        )
        
        # Fetch NIR using the fetcher
        nir = infrared_fetcher.augment_points_with_infrared(points, bbox=tile_bbox)
        
        if nir is not None:
            # Normalize to [0, 1] using utility
            nir = normalize_nir(nir, use_gpu=False)
            log.info(f"  ✓ NIR fetched from orthophotos")
            return nir
        else:
            log.warning(f"  ⚠️  NIR fetch returned None")
            return None
            
    except Exception as e:
        log.warning(f"  ⚠️  NIR fetch failed: {e}")
        return None


def compute_ndvi(
    rgb: Optional[np.ndarray],
    nir: Optional[np.ndarray],
    logger_instance: Optional[logging.Logger] = None
) -> Optional[np.ndarray]:
    """Compute NDVI (Normalized Difference Vegetation Index) from RGB and NIR.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        rgb: RGB colors [N, 3] normalized to [0, 1]
        nir: NIR values [N] normalized to [0, 1]
        logger_instance: Optional logger instance
        
    Returns:
        NDVI values [N] in range [-1, 1], or None if inputs unavailable
    """
    log = logger_instance or logger
    
    if nir is None:
        log.warning(f"  ⚠️  NDVI requested but NIR not available")
        return None
    
    if rgb is None:
        log.warning(f"  ⚠️  NDVI requested but RGB not available")
        return None
    
    # Extract red channel
    red = rgb[:, 0]
    
    # Compute NDVI with safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)
    
    log.info(f"  ✓ NDVI computed")
    return ndvi


# ============================================================================
# Geometric Feature Computation
# ============================================================================

def compute_geometric_features_standard(
    points: np.ndarray,
    classification: np.ndarray,
    feature_computer_factory: Any,
    k_neighbors: Optional[int] = None,
    use_gpu: bool = False,
    use_gpu_chunked: bool = True,
    gpu_batch_size: int = 100_000,
    include_extra: bool = True,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, np.ndarray]:
    """Compute geometric features using standard (non-boundary-aware) method.
    
    Args:
        points: Point cloud [N, 3] with XYZ coordinates
        classification: Point classifications [N]
        feature_computer_factory: Factory for creating feature computers
        k_neighbors: Number of neighbors (None = auto)
        use_gpu: Whether to use GPU processing
        use_gpu_chunked: Whether to use chunked GPU processing
        gpu_batch_size: Batch size for GPU chunked processing
        include_extra: Whether to include extra features
        logger_instance: Optional logger instance
        
    Returns:
        Dict with keys: normals, curvature, height, geo_features
    """
    log = logger_instance or logger
    
    num_points = len(points)
    k_value = k_neighbors if k_neighbors else 20
    
    # Create computer using factory (handles GPU availability, chunking, etc.)
    computer = feature_computer_factory.create(
        use_gpu=use_gpu,
        use_chunked=use_gpu_chunked and num_points > 500_000,
        gpu_batch_size=gpu_batch_size,
        k_neighbors=k_value
    )
    
    # Log processing method
    if use_gpu and use_gpu_chunked and num_points > 500_000:
        log.info(
            f"🚀 Using GPU chunked processing "
            f"({num_points:,} points, batch_size={gpu_batch_size:,})"
        )
    elif use_gpu:
        log.info(f"🚀 Using GPU processing ({num_points:,} points)")
    else:
        log.info(f"💻 Using CPU processing ({num_points:,} points)")
    
    # Compute features
    feature_dict = computer.compute_features(
        points=points,
        classification=classification,
        auto_k=(k_neighbors is None),
        include_extra=include_extra,
        patch_center=np.mean(points, axis=0) if include_extra else None
    )
    
    # Extract individual features
    normals = feature_dict.get('normals')
    curvature = feature_dict.get('curvature')
    height = feature_dict.get('height')
    geo_features = feature_dict.get('geo_features', {})
    
    # Ensure verticality is present
    if isinstance(geo_features, dict) and 'verticality' not in geo_features:
        if normals is not None:
            verticality = np.abs(normals[:, 2])
            geo_features['verticality'] = verticality
    
    return {
        'normals': normals,
        'curvature': curvature,
        'height': height,
        'geo_features': geo_features
    }


def compute_geometric_features_boundary_aware(
    laz_file: Path,
    stitcher: Any,
    k_neighbors: Optional[int] = None,
    logger_instance: Optional[logging.Logger] = None
) -> Tuple[Dict[str, np.ndarray], int]:
    """Compute geometric features using boundary-aware tile stitching.
    
    Args:
        laz_file: Path to the LAZ file being processed
        stitcher: Tile stitcher instance
        k_neighbors: Number of neighbors (None = auto, uses 20)
        logger_instance: Optional logger instance
        
    Returns:
        Tuple of (feature_dict, num_boundary_points)
        feature_dict contains: normals, curvature, height, geo_features
    """
    log = logger_instance or logger
    
    k_value = k_neighbors if k_neighbors else 20
    
    log.info("  🔗 Using tile stitching for boundary features...")
    
    # Load adjacent tiles and compute boundary-aware features
    features = stitcher.compute_boundary_aware_features(
        laz_file=laz_file,
        k=k_value
    )
    
    # Extract feature components
    normals = features['normals']
    curvature = features['curvature']
    
    # Extract geometric features
    geo_features = None
    if 'geometric_features' in features:
        geo_dict = features['geometric_features']
        
        # Convert to standardized dict format
        if isinstance(geo_dict, dict):
            geo_features = geo_dict
        else:
            # Assume it's an array [N, 4+] with columns: planarity, linearity, sphericity, verticality
            geo_features = {
                'planarity': geo_dict[:, 0],
                'linearity': geo_dict[:, 1],
                'sphericity': geo_dict[:, 2],
                'verticality': geo_dict[:, 3] if geo_dict.shape[1] > 3 else np.abs(normals[:, 2])
            }
    
    # Height feature (relative to local minimum)
    # Note: In boundary-aware mode, this might be computed differently
    # Extract from features dict if available
    if 'height' in features:
        height = features['height']
    else:
        # Fallback: compute from points if available
        if 'points' in features:
            points = features['points']
            height = points[:, 2] - points[:, 2].min()
        else:
            height = None
    
    num_boundary = features.get('num_boundary_points', 0)
    log.info(
        f"  ✓ Boundary-aware features computed "
        f"({num_boundary} boundary points affected)"
    )
    
    return {
        'normals': normals,
        'curvature': curvature,
        'height': height,
        'geo_features': geo_features
    }, num_boundary


# ============================================================================
# Main Enrichment Function
# ============================================================================

def enrich_point_cloud(
    points: np.ndarray,
    classification: np.ndarray,
    config: EnrichmentConfig,
    rgb_fetcher: Optional[Any] = None,
    infrared_fetcher: Optional[Any] = None,
    feature_computer_factory: Optional[Any] = None,
    stitcher: Optional[Any] = None,
    laz_file: Optional[Path] = None,
    rgb_from_laz: Optional[np.ndarray] = None,
    nir_from_laz: Optional[np.ndarray] = None,
    logger_instance: Optional[logging.Logger] = None
) -> EnrichmentResult:
    """Main enrichment function - orchestrates all enrichment operations.
    
    This function:
    1. Fetches RGB colors from orthophotos (if requested and not from LAZ)
    2. Fetches NIR from infrared orthophotos (if requested and not from LAZ)
    3. Computes NDVI from RGB + NIR (if requested)
    4. Computes geometric features (normals, curvature, height, etc.)
       - Uses boundary-aware processing if stitcher available and neighbors exist
       - Falls back to standard processing otherwise
    
    Args:
        points: Point cloud [N, 3] with XYZ coordinates
        classification: Point classifications [N]
        config: Enrichment configuration
        rgb_fetcher: Optional RGB fetcher for orthophoto colors
        infrared_fetcher: Optional infrared fetcher for NIR
        feature_computer_factory: Factory for creating feature computers
        stitcher: Optional tile stitcher for boundary-aware processing
        laz_file: Path to LAZ file (required for boundary-aware processing)
        rgb_from_laz: Optional RGB already loaded from LAZ file
        nir_from_laz: Optional NIR already loaded from LAZ file
        logger_instance: Optional logger instance
        
    Returns:
        EnrichmentResult with all computed features
    """
    log = logger_instance or logger
    start_time = time.time()
    
    result = EnrichmentResult()
    
    # ===== Feature Reuse Check (NEW) =====
    reused_features = {}
    if config.reuse_existing_features and laz_file is not None:
        try:
            from .feature_reuse import (
                FeatureReusePolicy,
                create_reuse_plan,
                load_existing_features,
                log_reuse_plan
            )
            
            # Create reuse policy from config
            policy = FeatureReusePolicy(
                reuse_rgb=not config.override_rgb,
                reuse_nir=not config.override_nir,
                reuse_normals=not config.override_normals,
                reuse_curvature=True,
                reuse_height=True,
                reuse_geometric=config.include_extra_features,
                override_all=config.override_all_features,
                check_k_neighbors=True
            )
            
            # Determine which features are requested
            requested_features = {'normals', 'curvature', 'height'}
            if config.include_rgb:
                requested_features.update({'red', 'green', 'blue'})
            if config.include_infrared:
                requested_features.add('nir')
            if config.include_extra_features:
                requested_features.update({
                    'planarity', 'linearity', 'sphericity', 'anisotropy',
                    'verticality', 'horizontality', 'density'
                })
            
            # Create reuse plan
            to_reuse, to_compute, inventory = create_reuse_plan(
                laz_file, requested_features, policy, config.k_neighbors
            )
            
            if to_reuse:
                log_reuse_plan(to_reuse, to_compute, log)
                reused_features = load_existing_features(laz_file, to_reuse)
                
                # Apply reused features to result
                if 'normals' in reused_features:
                    result.normals = reused_features['normals']
                if 'curvature' in reused_features:
                    result.curvature = reused_features['curvature']
                if 'height' in reused_features:
                    result.height = reused_features['height']
                
                # Store geometric features
                result.geo_features = {}
                for feat_name in ['planarity', 'linearity', 'sphericity', 'anisotropy',
                                  'verticality', 'horizontality', 'density']:
                    if feat_name in reused_features:
                        result.geo_features[feat_name] = reused_features[feat_name]
        
        except Exception as e:
            log.warning(f"  ⚠️  Feature reuse failed: {e}")
            reused_features = {}
    
    # ===== Step 1: RGB Colors =====
    if config.include_rgb:
        # Check if reused from file
        if 'red' in reused_features and 'green' in reused_features and 'blue' in reused_features:
            result.rgb = np.vstack([
                reused_features['red'],
                reused_features['green'],
                reused_features['blue']
            ]).T
            log.info("  ✓ Using RGB from existing LAZ features")
        elif rgb_from_laz is not None:
            log.info("  ✓ Using RGB from LAZ file")
            result.rgb = rgb_from_laz
        elif rgb_fetcher is not None:
            result.rgb = fetch_rgb_colors(points, rgb_fetcher, log)
        else:
            log.warning("  ⚠️  RGB requested but no RGB fetcher or LAZ RGB available")
    
    # ===== Step 2: Infrared/NIR =====
    if config.include_infrared:
        # Check if reused from file
        if 'nir' in reused_features:
            result.nir = reused_features['nir']
            log.info("  ✓ Using NIR from existing LAZ features")
        elif nir_from_laz is not None:
            log.info("  ✓ Using NIR from LAZ file")
            result.nir = nir_from_laz
        elif infrared_fetcher is not None:
            result.nir = fetch_infrared(points, infrared_fetcher, log)
        else:
            log.warning("  ⚠️  NIR requested but no infrared fetcher or LAZ NIR available")
    
    # ===== Step 3: NDVI =====
    if config.compute_ndvi:
        result.ndvi = compute_ndvi(result.rgb, result.nir, log)
    
    # ===== Step 4: Geometric Features =====
    # Only compute if not already reused
    skip_geometry = (result.normals is not None and result.curvature is not None and 
                     result.height is not None and result.geo_features is not None)
    
    if not skip_geometry:
        log.info("  🔧 Computing geometric features...")
        feature_start = time.time()
    
        # Determine if we should use boundary-aware stitching
        use_boundary_aware = False
        if config.use_stitching and stitcher is not None and laz_file is not None:
            # Check if neighbors exist for boundary-aware processing
            neighbors_exist = stitcher.check_neighbors_exist(laz_file)
            if neighbors_exist:
                use_boundary_aware = True
                
                try:
                    feature_dict, num_boundary = compute_geometric_features_boundary_aware(
                        laz_file=laz_file,
                        stitcher=stitcher,
                        k_neighbors=config.k_neighbors,
                        logger_instance=log
                    )
                    
                    result.normals = feature_dict['normals']
                    result.curvature = feature_dict['curvature']
                    result.height = feature_dict['height']
                    result.geo_features = feature_dict['geo_features']
                    result.num_boundary_points = num_boundary
                    result.used_boundary_aware = True
                    
                except Exception as e:
                    log.warning(
                        f"  ⚠️  Tile stitching failed, falling back to standard: {e}"
                    )
                    use_boundary_aware = False
        
        # Standard feature computation (no stitching or fallback)
        if not use_boundary_aware:
            if feature_computer_factory is None:
                log.error("  ❌ Feature computer factory required for standard processing")
                raise ValueError("feature_computer_factory required when not using boundary-aware processing")
            
            feature_dict = compute_geometric_features_standard(
                points=points,
                classification=classification,
                feature_computer_factory=feature_computer_factory,
                k_neighbors=config.k_neighbors,
                use_gpu=config.use_gpu,
                use_gpu_chunked=config.use_gpu_chunked,
                gpu_batch_size=config.gpu_batch_size,
                include_extra=config.include_extra_features,
                logger_instance=log
            )
            
            result.normals = feature_dict['normals']
            result.curvature = feature_dict['curvature']
            result.height = feature_dict['height']
            result.geo_features = feature_dict['geo_features']
            result.used_boundary_aware = False
        
        feature_time = time.time() - feature_start
        log.info(f"  ⏱️  Features computed: {feature_time:.1f}s")
    else:
        log.info("  ✨ All geometric features reused from LAZ file - computation skipped!")
    
    # Record total processing time
    result.processing_time = time.time() - start_time
    
    return result


# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Configuration
    'EnrichmentConfig',
    'EnrichmentResult',
    
    # RGB/NIR functions
    'fetch_rgb_colors',
    'fetch_infrared',
    'compute_ndvi',
    
    # Geometric feature functions
    'compute_geometric_features_standard',
    'compute_geometric_features_boundary_aware',
    
    # Main function
    'enrich_point_cloud',
]
