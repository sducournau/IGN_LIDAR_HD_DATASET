"""
Demo: Adaptive Building Classification

This script demonstrates the new adaptive building classification approach where:
1. Ground truth polygons guide classification but don't strictly define boundaries
2. Point cloud features drive final classification decisions
3. Multi-feature confidence scoring combines evidence from multiple sources
4. Adaptive expansion captures building points outside polygons
5. Intelligent rejection filters non-building points inside polygons

Author: Building Classification Enhancement
Date: October 20, 2025
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_adaptive_building_classification():
    """
    Demonstrate adaptive building classification with synthetic data.
    
    This example shows how the adaptive classifier handles:
    1. Misaligned polygons
    2. Missing walls in ground truth
    3. Vegetation near buildings
    4. Small structures not in ground truth
    """
    
    logger.info("=" * 80)
    logger.info("ADAPTIVE BUILDING CLASSIFICATION DEMO")
    logger.info("=" * 80)
    
    try:
        from ign_lidar.core.classification.adaptive_building_classifier import (
            AdaptiveBuildingClassifier,
            BuildingFeatureSignature
        )
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError as e:
        logger.error(f"Required modules not available: {e}")
        logger.error("Please install: geopandas, shapely, scipy")
        return
    
    # =========================================================================
    # 1. Create synthetic building point cloud
    # =========================================================================
    logger.info("\n1. Creating synthetic building point cloud...")
    
    # Building 1: 10m x 10m, height 3-8m (ground at z=0)
    building1_points = []
    
    # Walls (vertical surfaces)
    for i in range(100):
        # Front wall (x=0)
        building1_points.append([0, np.random.uniform(0, 10), np.random.uniform(3, 8)])
        # Back wall (x=10)
        building1_points.append([10, np.random.uniform(0, 10), np.random.uniform(3, 8)])
        # Left wall (y=0)
        building1_points.append([np.random.uniform(0, 10), 0, np.random.uniform(3, 8)])
        # Right wall (y=10)
        building1_points.append([np.random.uniform(0, 10), 10, np.random.uniform(3, 8)])
    
    # Roof (horizontal surface at z=8m)
    for i in range(200):
        building1_points.append([
            np.random.uniform(0, 10),
            np.random.uniform(0, 10),
            8 + np.random.normal(0, 0.1)  # Flat roof with small noise
        ])
    
    building1_points = np.array(building1_points)
    
    # Add some vegetation points near building (should be rejected)
    vegetation_near = []
    for i in range(50):
        vegetation_near.append([
            np.random.uniform(2, 8),
            np.random.uniform(11, 13),  # Just outside building
            np.random.uniform(1, 6)
        ])
    vegetation_near = np.array(vegetation_near)
    
    # Add small extension not in ground truth polygon (should be expanded)
    extension_points = []
    for i in range(30):
        extension_points.append([
            np.random.uniform(10, 12),  # Beyond original polygon
            np.random.uniform(4, 6),
            np.random.uniform(3, 8)
        ])
    extension_points = np.array(extension_points)
    
    # Combine all points
    points = np.vstack([building1_points, vegetation_near, extension_points])
    n_points = len(points)
    
    logger.info(f"Created {n_points:,} points:")
    logger.info(f"  - Building structure: {len(building1_points)}")
    logger.info(f"  - Vegetation nearby: {len(vegetation_near)}")
    logger.info(f"  - Extension (not in GT): {len(extension_points)}")
    
    # =========================================================================
    # 2. Compute synthetic features
    # =========================================================================
    logger.info("\n2. Computing synthetic features...")
    
    # Height above ground (z coordinate in this case)
    height = points[:, 2].copy()
    
    # Planarity (roofs have high planarity, walls have low)
    planarity = np.zeros(n_points)
    planarity[:len(building1_points)] = 0.3  # Walls
    planarity[len(building1_points) - 200:len(building1_points)] = 0.95  # Roof
    planarity[len(building1_points):len(building1_points) + len(vegetation_near)] = 0.4  # Vegetation
    planarity[len(building1_points) + len(vegetation_near):] = 0.3  # Extension walls
    
    # Verticality (walls have high verticality)
    verticality = np.zeros(n_points)
    verticality[:400] = 0.85  # Building walls
    verticality[400:len(building1_points)] = 0.1  # Roof (horizontal)
    verticality[len(building1_points):len(building1_points) + len(vegetation_near)] = 0.3  # Vegetation
    verticality[len(building1_points) + len(vegetation_near):] = 0.80  # Extension walls
    
    # Curvature (buildings have low curvature)
    curvature = np.random.uniform(0.01, 0.05, n_points)
    curvature[len(building1_points):len(building1_points) + len(vegetation_near)] = np.random.uniform(0.1, 0.3, len(vegetation_near))  # Vegetation higher
    
    # NDVI (vegetation has high NDVI, buildings low)
    ndvi = np.zeros(n_points)
    ndvi[:len(building1_points)] = np.random.uniform(0.05, 0.15, len(building1_points))  # Building
    ndvi[len(building1_points):len(building1_points) + len(vegetation_near)] = np.random.uniform(0.4, 0.7, len(vegetation_near))  # Vegetation
    ndvi[len(building1_points) + len(vegetation_near):] = np.random.uniform(0.05, 0.15, len(extension_points))  # Extension
    
    # Normals (approximate for demo)
    normals = np.zeros((n_points, 3))
    normals[:, 2] = 0.1  # Default low Z
    normals[len(building1_points) - 200:len(building1_points), 2] = 0.98  # Roof normals point up
    
    logger.info("Features computed successfully")
    
    # =========================================================================
    # 3. Create ground truth polygon (intentionally imperfect)
    # =========================================================================
    logger.info("\n3. Creating ground truth polygon (intentionally imperfect)...")
    
    # Polygon is slightly misaligned and doesn't include extension
    polygon = box(-0.5, -0.5, 9.5, 10.5)  # Slightly shifted, doesn't cover extension at x=10-12
    
    building_gdf = gpd.GeoDataFrame(
        {'geometry': [polygon]},
        crs='EPSG:2154'
    )
    
    logger.info(f"Ground truth polygon: {polygon.bounds}")
    logger.info("  NOTE: Polygon is shifted 0.5m and doesn't include extension")
    
    # =========================================================================
    # 4. Initialize adaptive classifier
    # =========================================================================
    logger.info("\n4. Initializing adaptive classifier...")
    
    signature = BuildingFeatureSignature(
        min_height=1.5,
        typical_height_range=(2.5, 50.0),
        wall_verticality_min=0.65,
        roof_planarity_min=0.75,
        ndvi_max=0.25
    )
    
    classifier = AdaptiveBuildingClassifier(
        signature=signature,
        fuzzy_boundary_inner=0.0,
        fuzzy_boundary_outer=2.0,
        fuzzy_decay_function="gaussian",
        enable_adaptive_expansion=True,
        max_expansion_distance=3.0,
        expansion_confidence_threshold=0.7,
        enable_intelligent_rejection=True,
        rejection_confidence_threshold=0.4,
        enable_spatial_clustering=True,
        spatial_radius=2.0,
        min_classification_confidence=0.5
    )
    
    logger.info("Classifier initialized with:")
    logger.info("  - Fuzzy boundaries: 0-2m")
    logger.info("  - Adaptive expansion: up to 3m")
    logger.info("  - Intelligent rejection: confidence < 0.4")
    logger.info("  - Spatial clustering: 2m radius")
    
    # =========================================================================
    # 5. Run adaptive classification
    # =========================================================================
    logger.info("\n5. Running adaptive classification...")
    
    labels, confidences, stats = classifier.classify_buildings_adaptive(
        points=points,
        building_polygons=building_gdf,
        height=height,
        planarity=planarity,
        verticality=verticality,
        curvature=curvature,
        normals=normals,
        ndvi=ndvi
    )
    
    # =========================================================================
    # 6. Analyze results
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS ANALYSIS")
    logger.info("=" * 80)
    
    # Count classifications
    building_mask = labels == 6  # ASPRS Building
    n_classified = np.sum(building_mask)
    
    # Analyze by point group
    building_structure_classified = np.sum(building_mask[:len(building1_points)])
    vegetation_classified = np.sum(building_mask[len(building1_points):len(building1_points) + len(vegetation_near)])
    extension_classified = np.sum(building_mask[len(building1_points) + len(vegetation_near):])
    
    logger.info(f"\nClassification breakdown:")
    logger.info(f"  Building structure: {building_structure_classified}/{len(building1_points)} "
                f"({building_structure_classified/len(building1_points)*100:.1f}%)")
    logger.info(f"  Vegetation (should reject): {vegetation_classified}/{len(vegetation_near)} "
                f"({vegetation_classified/len(vegetation_near)*100:.1f}%)")
    logger.info(f"  Extension (should expand): {extension_classified}/{len(extension_points)} "
                f"({extension_classified/len(extension_points)*100:.1f}%)")
    
    # Confidence analysis
    if n_classified > 0:
        avg_confidence = np.mean(confidences[building_mask])
        min_confidence = np.min(confidences[building_mask])
        max_confidence = np.max(confidences[building_mask])
        
        logger.info(f"\nConfidence scores:")
        logger.info(f"  Average: {avg_confidence:.3f}")
        logger.info(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")
    
    # Key metrics
    logger.info(f"\nKey metrics:")
    logger.info(f"  ✓ Building points captured: {building_structure_classified} "
                f"({building_structure_classified/len(building1_points)*100:.1f}%)")
    logger.info(f"  ✓ Vegetation correctly rejected: {len(vegetation_near) - vegetation_classified} "
                f"({(len(vegetation_near) - vegetation_classified)/len(vegetation_near)*100:.1f}%)")
    logger.info(f"  ✓ Extension points expanded: {extension_classified} "
                f"({extension_classified/len(extension_points)*100:.1f}%)")
    
    # Display statistics from classifier
    logger.info(f"\nClassifier statistics:")
    logger.info(f"  Walls detected: {stats.get('walls_detected', 0)}")
    logger.info(f"  Roofs detected: {stats.get('roofs_detected', 0)}")
    logger.info(f"  Adaptive expansion: {stats.get('expanded', 0)} points")
    logger.info(f"  Intelligent rejection: {stats.get('rejected', 0)} points")
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo completed successfully!")
    logger.info("=" * 80)
    
    return labels, confidences, stats


if __name__ == "__main__":
    demonstrate_adaptive_building_classification()
