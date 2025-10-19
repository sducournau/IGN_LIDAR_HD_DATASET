#!/usr/bin/env python3
"""
Test Ground Truth Refinement Module

Demonstrates the improvements from ground truth refinement:
1. Water & Roads: Validates flat, horizontal surfaces
2. Vegetation: Multi-feature segmentation
3. Buildings: Polygon expansion and validation

Usage:
    python scripts/test_ground_truth_refinement.py
"""

import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_water_refinement():
    """Test water classification refinement."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Water Refinement")
    logger.info("=" * 80)
    
    from ign_lidar.core.modules.ground_truth_refinement import (
        GroundTruthRefiner,
        GroundTruthRefinementConfig
    )
    
    # Create mock data
    n_points = 1000
    labels = np.ones(n_points, dtype=np.int32) * 9  # All water initially
    points = np.random.rand(n_points, 3) * 100
    
    # Create realistic features
    # 90% are actual water (near ground, flat)
    height = np.random.rand(n_points) * 0.2  # Most near ground
    height[900:1000] = np.random.rand(100) * 5.0 + 5.0  # 10% Bridge over water (elevated)
    
    planarity = np.random.rand(n_points) * 0.05 + 0.92  # Very flat
    planarity[900:1000] = np.random.rand(100) * 0.5  # Bridge is not flat
    
    curvature = np.random.rand(n_points) * 0.015  # Very smooth
    normals = np.zeros((n_points, 3))
    normals[:, 2] = 0.96 + np.random.rand(n_points) * 0.04  # Horizontal
    normals[900:1000, 2] = np.random.rand(100) * 0.5  # Bridge normals vary
    
    water_mask = labels == 9
    
    # Refine water classification
    refiner = GroundTruthRefiner()
    refined_labels, stats = refiner.refine_water_classification(
        labels=labels,
        points=points,
        water_mask=water_mask,
        height=height,
        planarity=planarity,
        curvature=curvature,
        normals=normals
    )
    
    # Validate results
    logger.info("\nResults:")
    logger.info(f"  Original water points: {np.sum(labels == 9)}")
    logger.info(f"  Validated water points: {stats['water_validated']}")
    logger.info(f"  Rejected water points: {stats['water_rejected']}")
    logger.info(f"  Expected rejections: ~100 (bridge over water)")
    
    assert stats['water_validated'] > 700, "Should validate most water points (>70%)"
    assert stats['water_rejected'] > 50, "Should reject bridge points"
    logger.info("  ✓ Test PASSED")


def test_road_refinement():
    """Test road classification refinement with tree canopy detection."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Road Refinement (with tree canopy detection)")
    logger.info("=" * 80)
    
    from ign_lidar.core.modules.ground_truth_refinement import (
        GroundTruthRefiner,
        GroundTruthRefinementConfig
    )
    
    # Create mock data
    n_points = 1000
    labels = np.ones(n_points, dtype=np.int32) * 11  # All road initially
    points = np.random.rand(n_points, 3) * 100
    
    # Create realistic features
    height = np.random.rand(n_points) * 0.5  # Road surface
    height[300:400] = np.random.rand(100) * 5.0 + 3.0  # Tree canopy over road
    
    planarity = np.random.rand(n_points) * 0.1 + 0.85  # Very flat
    curvature = np.random.rand(n_points) * 0.03  # Very smooth
    normals = np.zeros((n_points, 3))
    normals[:, 2] = 0.90 + np.random.rand(n_points) * 0.10  # Horizontal
    
    # NDVI: low for road, high for tree canopy
    ndvi = np.random.rand(n_points) * 0.1  # Low NDVI for road
    ndvi[300:400] = np.random.rand(100) * 0.5 + 0.4  # High NDVI for trees
    
    road_mask = labels == 11
    
    # Refine road classification
    refiner = GroundTruthRefiner()
    refined_labels, stats = refiner.refine_road_classification(
        labels=labels,
        points=points,
        road_mask=road_mask,
        height=height,
        planarity=planarity,
        curvature=curvature,
        normals=normals,
        ndvi=ndvi
    )
    
    # Validate results
    logger.info("\nResults:")
    logger.info(f"  Original road points: {np.sum(labels == 11)}")
    logger.info(f"  Validated road points: {stats['road_validated']}")
    logger.info(f"  Rejected road points: {stats['road_rejected']}")
    logger.info(f"  Tree canopy reclassified: {stats['road_vegetation_override']}")
    logger.info(f"  Expected tree canopy: ~100")
    
    assert stats['road_validated'] > 800, "Should validate most road points"
    assert stats['road_vegetation_override'] > 50, "Should detect tree canopy"
    logger.info("  ✓ Test PASSED")


def test_vegetation_refinement():
    """Test multi-feature vegetation classification."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Vegetation Refinement (multi-feature)")
    logger.info("=" * 80)
    
    from ign_lidar.core.modules.ground_truth_refinement import (
        GroundTruthRefiner,
        GroundTruthRefinementConfig
    )
    
    # Create mock data
    n_points = 1000
    labels = np.ones(n_points, dtype=np.int32)  # All unclassified
    
    # Create realistic features
    # Low vegetation (grass): low height, moderate NDVI
    ndvi = np.random.rand(n_points) * 0.2 + 0.3  # Start with good NDVI
    height = np.random.rand(n_points) * 0.5
    curvature = np.random.rand(n_points) * 0.05 + 0.03  # Higher curvature
    planarity = np.random.rand(n_points) * 0.4  # Lower planarity
    
    # Medium vegetation (shrubs): medium height, high NDVI
    ndvi[300:500] = np.random.rand(200) * 0.2 + 0.5  # Higher NDVI
    height[300:500] = np.random.rand(200) * 1.5 + 0.5
    curvature[300:500] = np.random.rand(200) * 0.06 + 0.04
    
    # High vegetation (trees): high height, very high NDVI
    ndvi[700:900] = np.random.rand(200) * 0.2 + 0.6  # Very high NDVI
    height[700:900] = np.random.rand(200) * 10.0 + 2.0
    curvature[700:900] = np.random.rand(200) * 0.08 + 0.05
    
    # Refine vegetation classification
    refiner = GroundTruthRefiner()
    refined_labels, stats = refiner.refine_vegetation_with_features(
        labels=labels,
        ndvi=ndvi,
        height=height,
        curvature=curvature,
        planarity=planarity
    )
    
    # Validate results
    logger.info("\nResults:")
    logger.info(f"  Total vegetation detected: {stats['vegetation_added']}")
    logger.info(f"  Low vegetation (0-0.5m): {stats['low_veg']}")
    logger.info(f"  Medium vegetation (0.5-2m): {stats['medium_veg']}")
    logger.info(f"  High vegetation (>2m): {stats['high_veg']}")
    
    assert stats['vegetation_added'] > 200, "Should detect most vegetation (>20%)"
    assert stats['medium_veg'] > 50, "Should detect medium vegetation"
    assert stats['high_veg'] > 50, "Should detect high vegetation"
    logger.info("  ✓ Test PASSED")


def test_building_refinement():
    """Test building polygon expansion and validation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Building Refinement (polygon expansion)")
    logger.info("=" * 80)
    
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        logger.warning("  ⚠ Skipping test - geopandas/shapely not available")
        return
    
    from ign_lidar.core.modules.ground_truth_refinement import (
        GroundTruthRefiner,
        GroundTruthRefinementConfig
    )
    
    # Create mock data
    n_points = 1000
    labels = np.ones(n_points, dtype=np.int32)  # All unclassified
    
    # Create points around a building
    # Building at (50, 50) with 10x10 size
    points = np.random.rand(n_points, 3) * 100
    points[200:400, 0] = np.random.rand(200) * 10 + 45  # Inside building
    points[200:400, 1] = np.random.rand(200) * 10 + 45
    points[400:500, 0] = np.random.rand(100) * 12 + 44  # Building edges (within 0.5m buffer)
    points[400:500, 1] = np.random.rand(100) * 12 + 44
    
    # Create realistic features
    height = np.random.rand(n_points) * 0.5  # Ground level
    height[200:500] = np.random.rand(300) * 5.0 + 2.0  # Building elevated
    
    planarity = np.random.rand(n_points) * 0.3
    planarity[200:500] = np.random.rand(300) * 0.2 + 0.7  # Building has high planarity
    
    verticality = np.random.rand(n_points) * 0.3
    verticality[200:500] = np.random.rand(300) * 0.3 + 0.5  # Building walls
    
    ndvi = np.random.rand(n_points) * 0.3
    ndvi[200:500] = np.random.rand(300) * 0.15  # Building has low NDVI
    
    # Create building polygon (exact building footprint)
    building_poly = Polygon([
        (45, 45),
        (55, 45),
        (55, 55),
        (45, 55),
        (45, 45)
    ])
    building_gdf = gpd.GeoDataFrame({'geometry': [building_poly]}, crs='EPSG:2154')
    
    # Refine building classification
    refiner = GroundTruthRefiner()
    refined_labels, stats = refiner.refine_building_with_expanded_polygons(
        labels=labels,
        points=points,
        building_polygons=building_gdf,
        height=height,
        planarity=planarity,
        verticality=verticality,
        ndvi=ndvi
    )
    
    # Validate results
    logger.info("\nResults:")
    logger.info(f"  Building points validated: {stats['building_validated']}")
    logger.info(f"  Building points expanded: {stats['building_expanded']}")
    logger.info(f"  Building candidates rejected: {stats['building_rejected']}")
    logger.info(f"  Expected expansion: ~100 (building edges)")
    
    assert stats['building_expanded'] > 50, "Should expand to capture building edges"
    logger.info("  ✓ Test PASSED")


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("Ground Truth Refinement Test Suite")
    logger.info("=" * 80)
    
    try:
        test_water_refinement()
        test_road_refinement()
        test_vegetation_refinement()
        test_building_refinement()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 80)
        
    except AssertionError as e:
        logger.error(f"\nTest FAILED: {e}")
        return 1
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
