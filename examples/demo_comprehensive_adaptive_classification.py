#!/usr/bin/env python3
"""
Comprehensive Adaptive Classification Demo

Demonstrates adaptive classification for ALL feature types:
- Buildings: Fuzzy boundaries, geometry-driven
- Vegetation: Multi-feature confidence voting
- Roads: Tree canopy detection, bridge identification
- Water: Extreme flatness validation

Key principle: Ground truth is GUIDANCE, not absolute truth.
Point cloud features are the PRIMARY classification signal.

Usage:
    python examples/demo_comprehensive_adaptive_classification.py
"""

import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_adaptive_water_classification():
    """Demo adaptive water classification."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 1: Adaptive Water Classification")
    logger.info("="*80)
    
    from ign_lidar.core.classification.adaptive_classifier import (
        ComprehensiveAdaptiveClassifier,
        AdaptiveReclassificationConfig
    )
    
    # Create synthetic data
    n_points = 2000
    labels = np.ones(n_points, dtype=np.int32) * 9  # All water initially
    points = np.random.rand(n_points, 3) * 100
    
    # Create realistic water features
    height = np.random.randn(n_points) * 0.1  # Very low, near zero
    planarity = np.random.rand(n_points) * 0.05 + 0.95  # Very flat
    curvature = np.random.rand(n_points) * 0.01  # Very smooth
    
    # Add some non-water points (elevated, rough)
    height[500:700] = np.random.rand(200) * 3.0 + 2.0  # Elevated (buildings/trees)
    planarity[500:700] = np.random.rand(200) * 0.5  # Rough
    
    # Normals - horizontal for water
    normals = np.zeros((n_points, 3))
    normals[:, 2] = 0.95 + np.random.rand(n_points) * 0.05  # Very horizontal
    normals[500:700, 2] = 0.3 + np.random.rand(200) * 0.4  # Non-horizontal
    
    ndvi = np.random.rand(n_points) * 0.1  # Low NDVI for water
    roughness = np.random.rand(n_points) * 0.01  # Very smooth
    
    # Run adaptive classification
    classifier = ComprehensiveAdaptiveClassifier()
    refined, stats = classifier.refine_water_adaptive(
        points=points,
        labels=labels,
        gt_water=None,  # No ground truth - purely feature-based
        height=height,
        planarity=planarity,
        curvature=curvature,
        normals=normals,
        ndvi=ndvi,
        roughness=roughness
    )
    
    logger.info(f"\nResults:")
    logger.info(f"  Initial water points: {np.sum(labels == 9):,}")
    logger.info(f"  Validated water: {stats['water_validated']:,}")
    logger.info(f"  Rejected (non-water): {stats['water_rejected']:,}")
    logger.info(f"  → Rejection rate: {stats['water_rejected']/np.sum(labels==9)*100:.1f}%")
    
    return refined, stats


def demo_adaptive_road_classification():
    """Demo adaptive road classification with tree canopy detection."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 2: Adaptive Road Classification + Tree Canopy Detection")
    logger.info("="*80)
    
    from ign_lidar.core.classification.adaptive_classifier import (
        ComprehensiveAdaptiveClassifier
    )
    
    # Create synthetic data
    n_points = 2000
    labels = np.ones(n_points, dtype=np.int32) * 11  # All road initially
    points = np.random.rand(n_points, 3) * 100
    
    # Road surface (0-1000)
    height = np.random.rand(1000) * 0.3  # Low, flat
    planarity = np.random.rand(1000) * 0.1 + 0.85  # Very flat
    curvature = np.random.rand(1000) * 0.03  # Smooth
    ndvi = np.random.rand(1000) * 0.15  # Low NDVI
    roughness = np.random.rand(1000) * 0.03  # Smooth
    verticality = np.random.rand(1000) * 0.15  # Not vertical
    
    # Tree canopy over road (1000-1400)
    height = np.concatenate([height, np.random.rand(400) * 5.0 + 3.0])  # Elevated
    planarity = np.concatenate([planarity, np.random.rand(400) * 0.3])  # Irregular
    curvature = np.concatenate([curvature, np.random.rand(400) * 0.08 + 0.02])  # Complex
    ndvi = np.concatenate([ndvi, np.random.rand(400) * 0.5 + 0.4])  # High NDVI
    roughness = np.concatenate([roughness, np.random.rand(400) * 0.1 + 0.05])  # Rough
    verticality = np.concatenate([verticality, np.random.rand(400) * 0.3])
    
    # Bridge (1400-1600)
    height = np.concatenate([height, np.random.rand(200) * 5.0 + 8.0])  # Very elevated
    planarity = np.concatenate([planarity, np.random.rand(200) * 0.1 + 0.85])  # Flat
    curvature = np.concatenate([curvature, np.random.rand(200) * 0.03])  # Smooth
    ndvi = np.concatenate([ndvi, np.random.rand(200) * 0.15])  # Low NDVI
    roughness = np.concatenate([roughness, np.random.rand(200) * 0.03])  # Smooth
    verticality = np.concatenate([verticality, np.random.rand(200) * 0.15])
    
    # Non-road (buildings) (1600-2000)
    height = np.concatenate([height, np.random.rand(400) * 15.0 + 3.0])  # Buildings
    planarity = np.concatenate([planarity, np.random.rand(400) * 0.3 + 0.6])
    curvature = np.concatenate([curvature, np.random.rand(400) * 0.05])
    ndvi = np.concatenate([ndvi, np.random.rand(400) * 0.2])  # Low NDVI
    roughness = np.concatenate([roughness, np.random.rand(400) * 0.05])
    verticality = np.concatenate([verticality, np.random.rand(400) * 0.5 + 0.5])  # Vertical
    
    # Normals
    normals = np.zeros((n_points, 3))
    normals[:, 2] = 0.92 + np.random.rand(n_points) * 0.08  # Mostly horizontal
    normals[1600:2000, 2] = 0.2 + np.random.rand(400) * 0.3  # Vertical structures
    
    # Run adaptive classification
    classifier = ComprehensiveAdaptiveClassifier()
    refined, stats = classifier.refine_roads_adaptive(
        points=points,
        labels=labels,
        gt_roads=None,  # No ground truth
        height=height,
        planarity=planarity,
        curvature=curvature,
        normals=normals,
        ndvi=ndvi,
        roughness=roughness,
        verticality=verticality
    )
    
    logger.info(f"\nResults:")
    logger.info(f"  Initial road points: {np.sum(labels == 11):,}")
    logger.info(f"  Validated roads: {stats['road_validated']:,}")
    logger.info(f"  Tree canopy detected: {stats['tree_canopy']:,}")
    logger.info(f"  Bridges detected: {stats['bridge']:,}")
    logger.info(f"  Rejected (non-road): {stats['road_rejected']:,}")
    
    return refined, stats


def demo_adaptive_vegetation_classification():
    """Demo adaptive vegetation classification."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 3: Adaptive Vegetation Classification")
    logger.info("="*80)
    
    from ign_lidar.core.classification.adaptive_classifier import (
        ComprehensiveAdaptiveClassifier
    )
    
    # Create synthetic data
    n_points = 2000
    labels = np.ones(n_points, dtype=np.int32) * 1  # Unclassified
    points = np.random.rand(n_points, 3) * 100
    
    # Vegetation points (0-1200)
    ndvi = np.random.rand(1200) * 0.5 + 0.3  # High NDVI
    height = np.random.rand(1200) * 8.0  # Various heights
    curvature = np.random.rand(1200) * 0.08 + 0.02  # Complex surfaces
    planarity = np.random.rand(1200) * 0.4  # Low planarity
    roughness = np.random.rand(1200) * 0.1 + 0.03  # Rough
    sphericity = np.random.rand(1200) * 0.5 + 0.3  # Organic shapes
    
    # Non-vegetation (1200-2000)
    ndvi = np.concatenate([ndvi, np.random.rand(800) * 0.2])  # Low NDVI
    height = np.concatenate([height, np.random.rand(800) * 5.0])
    curvature = np.concatenate([curvature, np.random.rand(800) * 0.02])  # Smooth
    planarity = np.concatenate([planarity, np.random.rand(800) * 0.3 + 0.7])  # High planarity
    roughness = np.concatenate([roughness, np.random.rand(800) * 0.03])  # Smooth
    sphericity = np.concatenate([sphericity, np.random.rand(800) * 0.3])  # Regular shapes
    
    # Run adaptive classification
    classifier = ComprehensiveAdaptiveClassifier()
    refined, stats = classifier.refine_vegetation_adaptive(
        points=points,
        labels=labels,
        gt_vegetation=None,  # No ground truth
        ndvi=ndvi,
        height=height,
        curvature=curvature,
        planarity=planarity,
        roughness=roughness,
        sphericity=sphericity
    )
    
    logger.info(f"\nResults:")
    logger.info(f"  Total vegetation: {stats['veg_total']:,}")
    logger.info(f"    - Low (0-0.5m): {stats['low_veg']:,}")
    logger.info(f"    - Medium (0.5-2m): {stats['medium_veg']:,}")
    logger.info(f"    - High (>2m): {stats['high_veg']:,}")
    
    return refined, stats


def demo_complete_adaptive_pipeline():
    """Demo complete adaptive classification pipeline."""
    logger.info("\n" + "="*80)
    logger.info("DEMO 4: Complete Adaptive Classification Pipeline")
    logger.info("="*80)
    
    from ign_lidar.core.classification.adaptive_classifier import (
        ComprehensiveAdaptiveClassifier
    )
    
    # Create realistic mixed scene
    n_points = 5000
    labels = np.ones(n_points, dtype=np.int32) * 1  # All unclassified
    points = np.random.rand(n_points, 3) * 100
    
    # Generate features for mixed scene
    features = {
        'height': np.random.rand(n_points) * 10.0,
        'planarity': np.random.rand(n_points),
        'curvature': np.random.rand(n_points) * 0.1,
        'ndvi': np.random.rand(n_points) * 0.8 - 0.1,
        'roughness': np.random.rand(n_points) * 0.1,
        'verticality': np.random.rand(n_points),
        'sphericity': np.random.rand(n_points),
        'normals': np.random.rand(n_points, 3)
    }
    features['normals'][:, 2] = np.abs(features['normals'][:, 2])  # Ensure valid normals
    
    # Run complete adaptive classification
    classifier = ComprehensiveAdaptiveClassifier()
    refined, all_stats = classifier.classify_all_adaptive(
        points=points,
        labels=labels,
        ground_truth_data=None,  # No ground truth - purely feature-based
        features=features
    )
    
    logger.info(f"\nFinal Classification Summary:")
    logger.info(f"  Water: {np.sum(refined == 9):,} points")
    logger.info(f"  Ground: {np.sum(refined == 2):,} points")
    logger.info(f"  Roads: {np.sum(refined == 11):,} points")
    logger.info(f"  Bridges: {np.sum(refined == 17):,} points")
    logger.info(f"  Low vegetation: {np.sum(refined == 3):,} points")
    logger.info(f"  Medium vegetation: {np.sum(refined == 4):,} points")
    logger.info(f"  High vegetation: {np.sum(refined == 5):,} points")
    logger.info(f"  Unclassified: {np.sum(refined == 1):,} points")
    
    return refined, all_stats


def main():
    """Run all demos."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ADAPTIVE CLASSIFICATION DEMOS")
    logger.info("Ground Truth as Guidance - Point Cloud as Primary Signal")
    logger.info("=" * 80)
    
    try:
        # Demo 1: Water
        demo_adaptive_water_classification()
        
        # Demo 2: Roads + tree canopy
        demo_adaptive_road_classification()
        
        # Demo 3: Vegetation
        demo_adaptive_vegetation_classification()
        
        # Demo 4: Complete pipeline
        demo_complete_adaptive_pipeline()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
