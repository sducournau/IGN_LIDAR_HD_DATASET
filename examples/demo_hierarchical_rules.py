#!/usr/bin/env python3
"""
Demo: Hierarchical Rule Engine

This example demonstrates how to use the HierarchicalRuleEngine for
multi-level classification with complex rule dependencies.

Features demonstrated:
- Organizing rules into hierarchical levels
- Different execution strategies (first_match, priority, weighted)
- Conflict resolution between rules
- Performance tracking per level
"""

import numpy as np
from typing import Dict, Any, Optional, Set

from ign_lidar.core.classification.rules import (
    BaseRule,
    RuleResult,
    RuleType,
    RulePriority,
    RuleConfig,
    HierarchicalRuleEngine,
    RuleLevel,
    ExecutionStrategy,
    ConflictResolution,
    calculate_confidence_threshold,
    validate_required_features,
)


# Define simple example rules

class GroundRule(BaseRule):
    """Level 1: Identify ground points."""
    
    def __init__(self):
        config = RuleConfig(
            name="ground_detection",
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.CRITICAL,
            min_confidence=0.7,
        )
        super().__init__(config)
    
    def get_required_features(self) -> Set[str]:
        return {"height_above_ground", "planarity"}
    
    def evaluate(self, points: np.ndarray, features: Dict[str, np.ndarray],
                 labels: Optional[np.ndarray] = None, **kwargs) -> RuleResult:
        n_points = len(points)
        height = features["height_above_ground"]
        planarity = features["planarity"]
        
        # Ground: very low height + high planarity
        ground_mask = (height < 0.2) & (planarity > 0.8)
        confidence_scores = calculate_confidence_threshold(
            height[ground_mask], threshold=0.2, reverse=True
        )
        
        classifications = np.zeros(n_points, dtype=np.int32)
        confidence = np.zeros(n_points, dtype=np.float32)
        classifications[ground_mask] = 2  # Ground class
        confidence[ground_mask] = confidence_scores
        
        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=ground_mask,
            metadata={"n_ground": np.sum(ground_mask)},
        )


class BuildingRule(BaseRule):
    """Level 2: Identify building points."""
    
    def __init__(self):
        config = RuleConfig(
            name="building_detection",
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.HIGH,
            min_confidence=0.6,
        )
        super().__init__(config)
    
    def get_required_features(self) -> Set[str]:
        return {"height_above_ground", "planarity", "ndvi"}
    
    def evaluate(self, points: np.ndarray, features: Dict[str, np.ndarray],
                 labels: Optional[np.ndarray] = None, **kwargs) -> RuleResult:
        n_points = len(points)
        height = features["height_above_ground"]
        planarity = features["planarity"]
        ndvi = features["ndvi"]
        
        # Buildings: elevated + planar + low NDVI
        building_mask = (
            (height > 2.0) & 
            (height < 50.0) &
            (planarity > 0.7) &
            (ndvi < 0.2)
        )
        
        # Skip already classified points if labels provided
        if labels is not None:
            building_mask &= (labels == 0)
        
        confidence_scores = np.ones(np.sum(building_mask), dtype=np.float32) * 0.8
        
        classifications = np.zeros(n_points, dtype=np.int32)
        confidence = np.zeros(n_points, dtype=np.float32)
        classifications[building_mask] = 6  # Building class
        confidence[building_mask] = confidence_scores
        
        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=building_mask,
            metadata={"n_buildings": np.sum(building_mask)},
        )


class VegetationRule(BaseRule):
    """Level 2: Identify vegetation points."""
    
    def __init__(self):
        config = RuleConfig(
            name="vegetation_detection",
            rule_type=RuleType.SPECTRAL,
            priority=RulePriority.HIGH,
            min_confidence=0.6,
        )
        super().__init__(config)
    
    def get_required_features(self) -> Set[str]:
        return {"ndvi", "height_above_ground"}
    
    def evaluate(self, points: np.ndarray, features: Dict[str, np.ndarray],
                 labels: Optional[np.ndarray] = None, **kwargs) -> RuleResult:
        n_points = len(points)
        ndvi = features["ndvi"]
        height = features["height_above_ground"]
        
        # Vegetation: high NDVI
        veg_mask = ndvi > 0.4
        
        # Skip already classified
        if labels is not None:
            veg_mask &= (labels == 0)
        
        # Classify by height
        veg_height = height[veg_mask]
        classifications = np.zeros(n_points, dtype=np.int32)
        confidence = np.zeros(n_points, dtype=np.float32)
        
        veg_indices = np.where(veg_mask)[0]
        low_veg = veg_height <= 2.0
        high_veg = veg_height > 2.0
        
        classifications[veg_indices[low_veg]] = 3  # Low vegetation
        classifications[veg_indices[high_veg]] = 5  # High vegetation
        confidence[veg_mask] = 0.7
        
        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=veg_mask,
            metadata={
                "n_low_veg": np.sum(low_veg),
                "n_high_veg": np.sum(high_veg),
            },
        )


class RefinementRule(BaseRule):
    """Level 3: Refine classifications based on context."""
    
    def __init__(self):
        config = RuleConfig(
            name="context_refinement",
            rule_type=RuleType.CONTEXTUAL,
            priority=RulePriority.NORMAL,
            min_confidence=0.5,
        )
        super().__init__(config)
    
    def get_required_features(self) -> Set[str]:
        return {"height_above_ground"}
    
    def evaluate(self, points: np.ndarray, features: Dict[str, np.ndarray],
                 labels: Optional[np.ndarray] = None, **kwargs) -> RuleResult:
        n_points = len(points)
        
        # Simple refinement: reclassify very low points as ground
        # (This would be more sophisticated in practice)
        classifications = np.zeros(n_points, dtype=np.int32)
        confidence = np.zeros(n_points, dtype=np.float32)
        mask = np.zeros(n_points, dtype=bool)
        
        if labels is not None:
            height = features["height_above_ground"]
            # Points classified as vegetation but very low -> ground
            refinement_mask = (labels == 3) & (height < 0.1)
            
            if np.any(refinement_mask):
                classifications[refinement_mask] = 2  # Reclassify as ground
                confidence[refinement_mask] = 0.6
                mask = refinement_mask
        
        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=mask,
            metadata={"n_refined": np.sum(mask)},
        )


def demo_hierarchical_first_match():
    """Demonstrate hierarchical execution with first_match strategy."""
    print("=" * 70)
    print("Demo 1: Hierarchical Execution - First Match Strategy")
    print("=" * 70)
    
    # Create synthetic data
    n_points = 5000
    points = np.random.rand(n_points, 3) * 100
    
    features = {
        "height_above_ground": np.random.rand(n_points) * 30,
        "planarity": np.random.rand(n_points),
        "ndvi": np.random.rand(n_points),
    }
    
    # Define hierarchical levels
    levels = [
        RuleLevel(
            name="ground_detection",
            priority=1,
            rules=[GroundRule()],
            strategy=ExecutionStrategy.FIRST_MATCH,
            description="Identify ground points first",
        ),
        RuleLevel(
            name="primary_classification",
            priority=2,
            rules=[BuildingRule(), VegetationRule()],
            strategy=ExecutionStrategy.PRIORITY,
            description="Classify buildings and vegetation",
        ),
        RuleLevel(
            name="refinement",
            priority=3,
            rules=[RefinementRule()],
            strategy=ExecutionStrategy.FIRST_MATCH,
            description="Refine classifications",
        ),
    ]
    
    # Create and configure engine
    engine = HierarchicalRuleEngine(
        levels=levels,
        conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE,
        enable_stats=True,
    )
    
    print(f"\nEngine configuration:")
    print(f"  Levels: {len(engine.levels)}")
    print(f"  Total rules: {sum(len(level.rules) for level in engine.levels)}")
    print(f"  Conflict resolution: {engine.conflict_resolution.value}")
    
    # Apply rules
    print(f"\nApplying rules to {n_points} points...\n")
    result = engine.apply_rules(points, features)
    
    # Print results
    print("Results:")
    print(f"  Points classified: {result.n_classified} ({result.n_classified/n_points*100:.1f}%)")
    print(f"  Mean confidence: {result.mean_confidence:.3f}")
    
    # Classification breakdown
    unique_labels, counts = np.unique(
        result.classifications[result.mask], return_counts=True
    )
    print(f"\nClassification breakdown:")
    for label, count in zip(unique_labels, counts):
        pct = count / result.n_classified * 100
        print(f"  Class {label}: {count:4d} points ({pct:5.1f}%)")
    
    # Per-level statistics
    if result.metadata and "level_stats" in result.metadata:
        print(f"\nPer-level performance:")
        for level_name, stats in result.metadata["level_stats"].items():
            print(f"  {level_name}:")
            print(f"    Classified: {stats['n_classified']} points")
            print(f"    Rules executed: {stats['n_rules']}")
            print(f"    Execution time: {stats['execution_time']:.3f}s")


def demo_hierarchical_weighted():
    """Demonstrate hierarchical execution with weighted strategy."""
    print("\n" + "=" * 70)
    print("Demo 2: Hierarchical Execution - Weighted Strategy")
    print("=" * 70)
    
    n_points = 3000
    points = np.random.rand(n_points, 3) * 100
    
    features = {
        "height_above_ground": np.random.rand(n_points) * 30,
        "planarity": np.random.rand(n_points),
        "ndvi": np.random.rand(n_points),
    }
    
    # Single level with weighted combination
    levels = [
        RuleLevel(
            name="weighted_classification",
            priority=1,
            rules=[BuildingRule(), VegetationRule()],
            strategy=ExecutionStrategy.WEIGHTED_COMBINATION,
            description="Combine multiple rules with weighting",
        ),
    ]
    
    engine = HierarchicalRuleEngine(
        levels=levels,
        conflict_resolution=ConflictResolution.WEIGHTED_AVERAGE,
    )
    
    print(f"\nUsing weighted combination strategy...")
    print(f"  This allows multiple rules to contribute to final classification")
    print(f"  based on their confidence scores.\n")
    
    result = engine.apply_rules(points, features)
    
    print("Results:")
    print(f"  Points classified: {result.n_classified}")
    print(f"  Mean confidence: {result.mean_confidence:.3f}")
    
    unique_labels, counts = np.unique(
        result.classifications[result.mask], return_counts=True
    )
    print(f"\nClassification breakdown:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} points")


def demo_level_strategies():
    """Demonstrate different level execution strategies."""
    print("\n" + "=" * 70)
    print("Demo 3: Comparing Level Execution Strategies")
    print("=" * 70)
    
    n_points = 2000
    points = np.random.rand(n_points, 3) * 100
    features = {
        "height_above_ground": np.random.rand(n_points) * 30,
        "planarity": np.random.rand(n_points),
        "ndvi": np.random.rand(n_points),
    }
    
    strategies = [
        ExecutionStrategy.FIRST_MATCH,
        ExecutionStrategy.ALL_MATCHES,
        ExecutionStrategy.PRIORITY,
    ]
    
    print("\nTesting different strategies on the same data:\n")
    
    for strategy in strategies:
        levels = [
            RuleLevel(
                name="classification",
                priority=1,
                rules=[BuildingRule(), VegetationRule()],
                strategy=strategy,
            ),
        ]
        
        engine = HierarchicalRuleEngine(levels=levels)
        result = engine.apply_rules(points, features)
        
        print(f"Strategy: {strategy.value}")
        print(f"  Classified: {result.n_classified} points")
        print(f"  Mean confidence: {result.mean_confidence:.3f}")
        print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Hierarchical Rule Engine - Usage Examples")
    print("=" * 70)
    print("\nThis demo shows how to use the HierarchicalRuleEngine for")
    print("multi-level classification with complex rule dependencies.\n")
    
    demo_hierarchical_first_match()
    demo_hierarchical_weighted()
    demo_level_strategies()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Organize rules into hierarchical levels (coarse → fine)")
    print("2. Use different execution strategies per level")
    print("3. First match stops after first rule, priority uses rule priority")
    print("4. Weighted combination blends multiple rule outputs")
    print("5. Engine tracks performance per level and rule")
    print("6. Conflict resolution handles overlapping classifications")
    print("\nFor more details, see the rules module documentation.")


if __name__ == "__main__":
    main()
