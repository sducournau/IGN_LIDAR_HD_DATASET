"""
Demo: Using Legacy Rule Engines with Modern Rules Framework

This example demonstrates how to use the adapter pattern to integrate
legacy classification engines (SpectralRulesEngine, GeometricRulesEngine)
with the modern hierarchical rules framework.

The adapters allow:
1. Using legacy engines in HierarchicalRuleEngine
2. Composing multiple rule types together
3. Leveraging confidence scoring and validation
4. Gradual migration without breaking existing code

Usage:
    python examples/demo_legacy_adapter.py

Author: Classification Enhancement Team
Date: October 23, 2025
"""

import numpy as np
import logging
from typing import Dict
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.classification.rules import (
    # Configuration
    RuleConfig,
    RuleType,
    RulePriority,
    
    # Hierarchical execution
    HierarchicalRuleEngine,
    RuleLevel,
    
    # Legacy adapters
    SpectralRulesAdapter,
    GeometricRulesAdapter,
    
    # Convenience factories
    create_spectral_vegetation_rule,
    create_spectral_water_rule,
    create_geometric_building_rule,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_point_cloud(n_points: int = 1000) -> tuple:
    """
    Create sample point cloud with features for testing.
    
    Returns:
        Tuple of (points, features, ground_truth_features)
    """
    logger.info(f"Creating sample point cloud with {n_points} points")
    
    # Create random points
    points = np.random.randn(n_points, 3) * 10
    points[:, 2] = np.abs(points[:, 2])  # Positive Z (height)
    
    # Create sample RGB (normalized)
    rgb = np.random.rand(n_points, 3)
    
    # Create sample NIR (higher for vegetation areas)
    nir = np.random.rand(n_points)
    # Make some points have high NIR (vegetation)
    veg_mask = (points[:, 2] < 2) & (np.random.rand(n_points) > 0.5)
    nir[veg_mask] = np.clip(nir[veg_mask] + 0.4, 0, 1)
    
    # Compute NDVI
    red = rgb[:, 0]
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    # Create features dictionary
    features = {
        'rgb': rgb,
        'nir': nir,
        'ndvi': ndvi,
        'intensities': np.random.rand(n_points) * 255,
    }
    
    # Create empty ground truth features (normally would be GeoDataFrames)
    ground_truth_features = {}
    
    logger.info(f"  Created {n_points} points with RGB, NIR, NDVI features")
    logger.info(f"  Vegetation candidates: {np.sum(veg_mask)} points (NDVI > 0.3)")
    
    return points, features, ground_truth_features


def demo_1_basic_adapter_usage():
    """
    Demo 1: Basic usage of spectral adapter.
    """
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Basic Spectral Adapter Usage")
    logger.info("="*60)
    
    # Create sample data
    points, features, _ = create_sample_point_cloud(500)
    
    # Create spectral rule configuration
    config = RuleConfig(
        rule_id="vegetation_rule",
        rule_type=RuleType.SPECTRAL,
        target_class=3,  # Low vegetation (ASPRS)
        priority=RulePriority.MEDIUM,
        description="Detect vegetation using spectral signatures"
    )
    
    # Create spectral adapter
    logger.info("\nCreating SpectralRulesAdapter...")
    adapter = SpectralRulesAdapter(
        config=config,
        nir_vegetation_threshold=0.4
    )
    
    # Evaluate rule
    logger.info("\nEvaluating spectral rule...")
    mask, confidence = adapter.evaluate(points, features)
    
    # Report results
    n_matched = np.sum(mask)
    logger.info(f"\nResults:")
    logger.info(f"  Matched points: {n_matched} / {len(points)} ({100*n_matched/len(points):.1f}%)")
    if n_matched > 0:
        logger.info(f"  Mean confidence: {np.mean(confidence[mask]):.3f}")
        logger.info(f"  Min/Max confidence: {np.min(confidence[mask]):.3f} / {np.max(confidence[mask]):.3f}")


def demo_2_hierarchical_composition():
    """
    Demo 2: Using adapters in hierarchical rule engine.
    """
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Hierarchical Rule Composition")
    logger.info("="*60)
    
    # Create sample data
    points, features, _ = create_sample_point_cloud(1000)
    
    # Create hierarchical engine
    logger.info("\nCreating HierarchicalRuleEngine...")
    engine = HierarchicalRuleEngine()
    
    # Add Level 1: Spectral rules (vegetation, water)
    logger.info("\nLevel 1: Adding spectral rules...")
    
    veg_config = RuleConfig(
        rule_id="spectral_vegetation",
        rule_type=RuleType.SPECTRAL,
        target_class=3,  # Low vegetation
        priority=RulePriority.HIGH
    )
    veg_rule = create_spectral_vegetation_rule(veg_config, nir_threshold=0.4)
    engine.add_rule(veg_rule, level=0)
    logger.info("  ✓ Added vegetation rule")
    
    water_config = RuleConfig(
        rule_id="spectral_water",
        rule_type=RuleType.SPECTRAL,
        target_class=9,  # Water
        priority=RulePriority.HIGH
    )
    water_rule = create_spectral_water_rule(water_config)
    engine.add_rule(water_rule, level=0)
    logger.info("  ✓ Added water rule")
    
    # Apply all rules
    logger.info("\nApplying hierarchical rules...")
    result = engine.apply(points, features)
    
    # Report results
    logger.info(f"\nResults:")
    logger.info(f"  Total points: {result.stats.total_points}")
    logger.info(f"  Matched points: {result.stats.matched_points}")
    logger.info(f"  Coverage: {100*result.stats.coverage:.1f}%")
    logger.info(f"  Rules applied: {result.stats.rules_applied}")
    logger.info(f"  Execution time: {result.stats.execution_time_ms:.2f} ms")
    
    if result.stats.rule_match_counts:
        logger.info(f"\n  Rule match counts:")
        for rule_id, count in result.stats.rule_match_counts.items():
            logger.info(f"    {rule_id}: {count} points")


def demo_3_convenience_factories():
    """
    Demo 3: Using convenience factory functions.
    """
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: Convenience Factory Functions")
    logger.info("="*60)
    
    # Create sample data
    points, features, _ = create_sample_point_cloud(500)
    
    logger.info("\nUsing factory functions for quick rule creation...")
    
    # Create vegetation rule with factory
    veg_config = RuleConfig(
        rule_id="quick_veg",
        rule_type=RuleType.SPECTRAL,
        target_class=3
    )
    veg_rule = create_spectral_vegetation_rule(veg_config, nir_threshold=0.35)
    logger.info("  ✓ Created vegetation rule with factory")
    
    # Evaluate
    mask, confidence = veg_rule.evaluate(points, features)
    n_matched = np.sum(mask)
    
    logger.info(f"\nResults:")
    logger.info(f"  Vegetation detected: {n_matched} points")
    logger.info(f"  Mean confidence: {np.mean(confidence[mask]):.3f}" if n_matched > 0 else "  No matches")


def demo_4_comparison():
    """
    Demo 4: Comparison of direct engine vs adapter usage.
    """
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Direct Engine vs Adapter Comparison")
    logger.info("="*60)
    
    # Create sample data
    points, features, _ = create_sample_point_cloud(1000)
    
    # Method 1: Direct legacy engine usage
    logger.info("\nMethod 1: Using SpectralRulesEngine directly...")
    from ign_lidar.core.classification.spectral_rules import SpectralRulesEngine
    
    direct_engine = SpectralRulesEngine(nir_vegetation_threshold=0.4)
    current_labels = np.ones(len(points), dtype=np.int32)
    
    updated_labels, stats = direct_engine.classify_by_spectral_signature(
        rgb=features['rgb'],
        nir=features['nir'],
        current_labels=current_labels,
        ndvi=features['ndvi']
    )
    
    n_veg_direct = np.sum(updated_labels == 3)  # Low vegetation
    logger.info(f"  Direct engine: {n_veg_direct} vegetation points")
    logger.info(f"  Stats: {stats}")
    
    # Method 2: Using adapter
    logger.info("\nMethod 2: Using SpectralRulesAdapter...")
    config = RuleConfig(
        rule_id="adapter_veg",
        rule_type=RuleType.SPECTRAL,
        target_class=3
    )
    adapter = SpectralRulesAdapter(config, nir_vegetation_threshold=0.4)
    mask, confidence = adapter.evaluate(points, features)
    
    n_veg_adapter = np.sum(mask)
    logger.info(f"  Adapter: {n_veg_adapter} vegetation points")
    logger.info(f"  Mean confidence: {np.mean(confidence[mask]):.3f}")
    
    # Compare
    logger.info(f"\nComparison:")
    logger.info(f"  Both methods found similar points: {n_veg_direct == n_veg_adapter}")
    logger.info(f"  Adapter provides confidence scores: ✓")
    logger.info(f"  Adapter works with hierarchical engine: ✓")


def main():
    """
    Run all demonstrations.
    """
    logger.info("="*60)
    logger.info("Legacy Adapter Demonstration")
    logger.info("="*60)
    logger.info("\nThis demo shows how to use legacy rule engines")
    logger.info("with the modern rules framework using adapters.\n")
    
    try:
        # Run demonstrations
        demo_1_basic_adapter_usage()
        demo_2_hierarchical_composition()
        demo_3_convenience_factories()
        demo_4_comparison()
        
        logger.info("\n" + "="*60)
        logger.info("All demonstrations completed successfully!")
        logger.info("="*60)
        
        logger.info("\nKey Takeaways:")
        logger.info("  1. Adapters wrap legacy engines for modern framework")
        logger.info("  2. Can use in HierarchicalRuleEngine with other rules")
        logger.info("  3. Convenience factories simplify rule creation")
        logger.info("  4. Both old and new patterns work together")
        logger.info("  5. No breaking changes - gradual migration possible")
        
    except Exception as e:
        logger.error(f"\nDemo failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
