#!/usr/bin/env python3
"""
Test Parcel Classification on Versailles Tile

This script tests the new parcel-based classification system on a Versailles
tile with cadastre, BD Forêt, RPG, and BD TOPO ground truth.

Usage:
    python test_parcel_classification_versailles.py
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ign_lidar.io import LASReader
from ign_lidar.io.cadastre import CadastreFetcher
from ign_lidar.io.bd_foret import BDForetFetcher
from ign_lidar.io.rpg import RPGFetcher
from ign_lidar.io.bd_topo import BDTopoFetcher
from ign_lidar.core.classification import AdvancedClassifier
from ign_lidar.preprocessing.features import FeatureComputer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_parcel_classification():
    """Test parcel classification on Versailles tile."""
    
    # Configuration
    versailles_dir = Path("/mnt/c/Users/Simon/ign/versailles")
    cache_dir = Path("/mnt/c/Users/Simon/ign/cache")
    
    # Find first LAZ file
    laz_files = list(versailles_dir.glob("*.laz"))
    if not laz_files:
        laz_files = list(versailles_dir.glob("*.las"))
    
    if not laz_files:
        logger.error(f"No LAS/LAZ files found in {versailles_dir}")
        return False
    
    input_file = laz_files[0]
    logger.info(f"Testing with file: {input_file.name}")
    
    # ========================================================================
    # 1. Load Point Cloud
    # ========================================================================
    
    logger.info("=" * 80)
    logger.info("STEP 1: Loading point cloud")
    logger.info("=" * 80)
    
    reader = LASReader(str(input_file))
    points = reader.read_points()
    
    logger.info(f"  Loaded: {len(points):,} points")
    logger.info(f"  Bounds: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    logger.info(f"  Bounds: Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    logger.info(f"  Bounds: Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Create bounding box
    bbox = (
        points[:, 0].min(), points[:, 1].min(),
        points[:, 0].max(), points[:, 1].max()
    )
    
    # ========================================================================
    # 2. Fetch Ground Truth Data
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Fetching ground truth data")
    logger.info("=" * 80)
    
    ground_truth_features = {}
    
    # Cadastre (required for parcel classification)
    logger.info("  Fetching cadastre (BD Parcellaire)...")
    cadastre_fetcher = CadastreFetcher(cache_dir=cache_dir / "cadastre")
    cadastre_gdf = cadastre_fetcher.fetch_parcels(bbox=bbox)
    if cadastre_gdf is not None and len(cadastre_gdf) > 0:
        ground_truth_features['cadastre'] = cadastre_gdf
        logger.info(f"    ✓ Found {len(cadastre_gdf)} cadastral parcels")
    else:
        logger.warning("    ✗ No cadastre data found")
    
    # BD Forêt (optional - improves forest classification)
    logger.info("  Fetching BD Forêt...")
    bd_foret_fetcher = BDForetFetcher(cache_dir=cache_dir / "bd_foret")
    bd_foret_gdf = bd_foret_fetcher.fetch_forest(bbox=bbox)
    if bd_foret_gdf is not None and len(bd_foret_gdf) > 0:
        ground_truth_features['forest'] = bd_foret_gdf
        logger.info(f"    ✓ Found {len(bd_foret_gdf)} forest polygons")
    else:
        logger.info("    ○ No BD Forêt data found (optional)")
    
    # RPG (optional - improves agriculture classification)
    logger.info("  Fetching RPG...")
    try:
        rpg_fetcher = RPGFetcher(cache_dir=cache_dir / "rpg")
        rpg_gdf = rpg_fetcher.fetch_parcels(bbox=bbox)
        if rpg_gdf is not None and len(rpg_gdf) > 0:
            ground_truth_features['rpg'] = rpg_gdf
            logger.info(f"    ✓ Found {len(rpg_gdf)} agricultural parcels")
        else:
            logger.info("    ○ No RPG data found (optional)")
    except Exception as e:
        logger.info(f"    ○ RPG fetch failed (optional): {e}")
    
    # BD TOPO (standard ground truth)
    logger.info("  Fetching BD TOPO...")
    bd_topo_fetcher = BDTopoFetcher(cache_dir=cache_dir / "bd_topo")
    bd_topo_features = bd_topo_fetcher.fetch_all_features(
        bbox=bbox,
        include_buildings=True,
        include_roads=True,
        include_water=True
    )
    if bd_topo_features:
        ground_truth_features.update(bd_topo_features)
        logger.info(f"    ✓ BD TOPO features:")
        for key, gdf in bd_topo_features.items():
            if gdf is not None and len(gdf) > 0:
                logger.info(f"      - {key}: {len(gdf)}")
    
    # ========================================================================
    # 3. Compute Features (Simplified - just what we need)
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Computing features")
    logger.info("=" * 80)
    
    # For testing, create mock features
    # In production, use FeatureComputer
    logger.info("  Creating mock features for testing...")
    
    n_points = len(points)
    height = np.abs(points[:, 2] - points[:, 2].min())  # Height above min
    ndvi = np.random.uniform(0.2, 0.7, n_points)  # Mock NDVI
    planarity = np.random.uniform(0.3, 0.9, n_points)  # Mock planarity
    verticality = np.random.uniform(0.1, 0.8, n_points)  # Mock verticality
    curvature = np.random.uniform(0.0, 0.5, n_points)  # Mock curvature
    
    logger.info(f"    ✓ Height: min={height.min():.2f}, max={height.max():.2f}")
    logger.info(f"    ✓ NDVI: min={ndvi.min():.2f}, max={ndvi.max():.2f}")
    logger.info(f"    ✓ Planarity: min={planarity.min():.2f}, max={planarity.max():.2f}")
    
    # ========================================================================
    # 4. Run Classification (WITH and WITHOUT parcel classification)
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Running classification comparison")
    logger.info("=" * 80)
    
    # 4a. Traditional classification (baseline)
    logger.info("\n--- 4a. Traditional Classification (Baseline) ---")
    classifier_traditional = AdvancedClassifier(
        use_parcel_classification=False,
        use_ground_truth=True,
        use_ndvi=True,
        use_geometric=True
    )
    
    labels_traditional = classifier_traditional.classify_points(
        points=points,
        ground_truth_features=ground_truth_features,
        ndvi=ndvi,
        height=height,
        planarity=planarity,
        curvature=curvature,
        verticality=verticality
    )
    
    # Print distribution
    unique_traditional, counts_traditional = np.unique(labels_traditional, return_counts=True)
    logger.info("\n  Traditional classification distribution:")
    for label, count in zip(unique_traditional, counts_traditional):
        pct = 100 * count / n_points
        logger.info(f"    Class {label:2d}: {count:8,} points ({pct:5.2f}%)")
    
    # 4b. Parcel-based classification
    if 'cadastre' in ground_truth_features:
        logger.info("\n--- 4b. Parcel-Based Classification (NEW) ---")
        
        parcel_config = {
            'min_parcel_points': 20,
            'min_parcel_area': 10.0,
            'parcel_confidence_threshold': 0.6,
            'refine_points': True,
            'refinement_method': 'feature_based',
            'forest_ndvi_min': 0.5,
            'agriculture_ndvi_min': 0.2,
            'agriculture_ndvi_max': 0.6,
            'building_ndvi_max': 0.15
        }
        
        classifier_parcel = AdvancedClassifier(
            use_parcel_classification=True,
            parcel_classification_config=parcel_config,
            use_ground_truth=True,
            use_ndvi=True,
            use_geometric=True
        )
        
        labels_parcel = classifier_parcel.classify_points(
            points=points,
            ground_truth_features=ground_truth_features,
            ndvi=ndvi,
            height=height,
            planarity=planarity,
            curvature=curvature,
            verticality=verticality
        )
        
        # Print distribution
        unique_parcel, counts_parcel = np.unique(labels_parcel, return_counts=True)
        logger.info("\n  Parcel-based classification distribution:")
        for label, count in zip(unique_parcel, counts_parcel):
            pct = 100 * count / n_points
            logger.info(f"    Class {label:2d}: {count:8,} points ({pct:5.2f}%)")
        
        # Compare results
        logger.info("\n--- Classification Comparison ---")
        different = np.sum(labels_traditional != labels_parcel)
        pct_diff = 100 * different / n_points
        logger.info(f"  Points reclassified: {different:,} ({pct_diff:.2f}%)")
        
        # Agreement matrix (simplified)
        agreement = np.sum(labels_traditional == labels_parcel)
        pct_agree = 100 * agreement / n_points
        logger.info(f"  Point agreement: {agreement:,} ({pct_agree:.2f}%)")
    else:
        logger.warning("\n  ✗ Skipping parcel classification (no cadastre data)")
    
    # ========================================================================
    # 5. Summary
    # ========================================================================
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Total points: {n_points:,}")
    logger.info(f"  Ground truth sources: {len(ground_truth_features)}")
    logger.info(f"  Parcel classification: {'✓ TESTED' if 'cadastre' in ground_truth_features else '✗ SKIPPED'}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_parcel_classification()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
