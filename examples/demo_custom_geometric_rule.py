#!/usr/bin/env python3
"""
Demo: Creating Custom Geometric Rules using the Rules Framework

This example demonstrates how to create custom geometric rules for point cloud
classification using the new rules infrastructure from Phase 4B.

Features demonstrated:
- Creating a custom rule class
- Using confidence scoring methods
- Feature validation
- Rule execution and result handling
"""

import numpy as np
from typing import Dict, Any, Optional, Set

# Import the rules framework
from ign_lidar.core.classification.rules import (
    BaseRule,
    RuleResult,
    RuleType,
    RulePriority,
    RuleConfig,
    calculate_confidence_linear,
    calculate_confidence_threshold,
    validate_required_features,
    validate_point_cloud_shape,
)


class FlatSurfaceRule(BaseRule):
    """
    Custom rule to identify flat surfaces based on planarity and roughness.
    
    This rule classifies points as flat surfaces (e.g., roads, parking lots)
    when they have high planarity and low roughness values.
    """
    
    def __init__(
        self,
        planarity_threshold: float = 0.7,
        roughness_threshold: float = 0.05,
        min_confidence: float = 0.6,
        priority: RulePriority = RulePriority.NORMAL,
    ):
        """
        Initialize the flat surface detection rule.
        
        Args:
            planarity_threshold: Minimum planarity value (0-1)
            roughness_threshold: Maximum roughness value
            min_confidence: Minimum confidence to accept classification
            priority: Rule execution priority
        """
        config = RuleConfig(
            name="flat_surface_detection",
            rule_type=RuleType.GEOMETRIC,
            priority=priority,
            min_confidence=min_confidence,
            enabled=True,
        )
        super().__init__(config)
        
        self.planarity_threshold = planarity_threshold
        self.roughness_threshold = roughness_threshold
    
    def get_required_features(self) -> Set[str]:
        """Features required by this rule."""
        return {"planarity", "roughness"}
    
    def get_optional_features(self) -> Set[str]:
        """Optional features that enhance the rule."""
        return {"height_above_ground", "intensity"}
    
    def evaluate(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        **kwargs
    ) -> RuleResult:
        """
        Evaluate the rule on point cloud data.
        
        Args:
            points: Point cloud (N, 3) array
            features: Dictionary of feature arrays
            labels: Existing classification labels (optional)
            **kwargs: Additional parameters
            
        Returns:
            RuleResult with classifications and confidence scores
        """
        # Validate inputs
        validate_point_cloud_shape(points)
        validate_required_features(features, self.get_required_features())
        
        n_points = len(points)
        
        # Extract features
        planarity = features["planarity"]
        roughness = features["roughness"]
        
        # Initialize result arrays
        classifications = np.zeros(n_points, dtype=np.int32)
        confidence = np.zeros(n_points, dtype=np.float32)
        mask = np.zeros(n_points, dtype=bool)
        
        # Apply geometric criteria
        flat_mask = (
            (planarity >= self.planarity_threshold) &
            (roughness <= self.roughness_threshold)
        )
        
        if not np.any(flat_mask):
            return self._create_empty_result(n_points)
        
        # Calculate confidence based on how well points meet criteria
        # Use linear confidence for planarity (higher is better)
        planarity_conf = calculate_confidence_linear(
            planarity[flat_mask],
            min_value=self.planarity_threshold,
            max_value=1.0,
        )
        
        # Use threshold confidence for roughness (lower is better)
        roughness_conf = calculate_confidence_threshold(
            roughness[flat_mask],
            threshold=self.roughness_threshold,
            reverse=True,  # Lower values are better
        )
        
        # Combine confidences (weighted average)
        combined_confidence = 0.6 * planarity_conf + 0.4 * roughness_conf
        
        # Optional: Adjust confidence based on height
        if "height_above_ground" in features:
            height = features["height_above_ground"][flat_mask]
            # Flat surfaces are typically near ground
            height_factor = np.clip(1.0 - height / 5.0, 0.3, 1.0)
            combined_confidence *= height_factor
        
        # Apply minimum confidence threshold
        confident_mask = combined_confidence >= self.config.min_confidence
        
        if not np.any(confident_mask):
            return self._create_empty_result(n_points)
        
        # Set classifications (assuming label 2 = ground/road)
        flat_indices = np.where(flat_mask)[0]
        confident_indices = flat_indices[confident_mask]
        
        classifications[confident_indices] = 2  # Ground class
        confidence[confident_indices] = combined_confidence[confident_mask]
        mask[confident_indices] = True
        
        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=mask,
            metadata={
                "n_classified": np.sum(mask),
                "mean_confidence": np.mean(confidence[mask]) if np.any(mask) else 0.0,
                "mean_planarity": np.mean(planarity[confident_indices]),
                "mean_roughness": np.mean(roughness[confident_indices]),
                "planarity_threshold": self.planarity_threshold,
                "roughness_threshold": self.roughness_threshold,
            },
        )


class VegetationHeightRule(BaseRule):
    """
    Custom rule to classify vegetation based on height and NDVI.
    
    Distinguishes between low vegetation (grass), medium vegetation (shrubs),
    and high vegetation (trees) based on height above ground.
    """
    
    def __init__(
        self,
        low_veg_max_height: float = 0.5,
        medium_veg_max_height: float = 3.0,
        ndvi_threshold: float = 0.3,
        priority: RulePriority = RulePriority.HIGH,
    ):
        """
        Initialize the vegetation classification rule.
        
        Args:
            low_veg_max_height: Maximum height for low vegetation (meters)
            medium_veg_max_height: Maximum height for medium vegetation (meters)
            ndvi_threshold: Minimum NDVI value for vegetation
            priority: Rule execution priority
        """
        config = RuleConfig(
            name="vegetation_height_classification",
            rule_type=RuleType.GEOMETRIC,
            priority=priority,
            min_confidence=0.5,
            enabled=True,
        )
        super().__init__(config)
        
        self.low_veg_max = low_veg_max_height
        self.medium_veg_max = medium_veg_max_height
        self.ndvi_threshold = ndvi_threshold
    
    def get_required_features(self) -> Set[str]:
        """Features required by this rule."""
        return {"height_above_ground", "ndvi"}
    
    def evaluate(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None,
        **kwargs
    ) -> RuleResult:
        """Evaluate vegetation classification based on height and NDVI."""
        validate_point_cloud_shape(points)
        validate_required_features(features, self.get_required_features())
        
        n_points = len(points)
        height = features["height_above_ground"]
        ndvi = features["ndvi"]
        
        # Initialize results
        classifications = np.zeros(n_points, dtype=np.int32)
        confidence = np.zeros(n_points, dtype=np.float32)
        mask = np.zeros(n_points, dtype=bool)
        
        # Identify vegetation by NDVI
        veg_mask = ndvi >= self.ndvi_threshold
        
        if not np.any(veg_mask):
            return self._create_empty_result(n_points)
        
        # Classify by height
        veg_height = height[veg_mask]
        veg_ndvi = ndvi[veg_mask]
        
        # Low vegetation (class 3)
        low_veg = veg_height <= self.low_veg_max
        
        # Medium vegetation (class 4)
        medium_veg = (veg_height > self.low_veg_max) & (veg_height <= self.medium_veg_max)
        
        # High vegetation (class 5)
        high_veg = veg_height > self.medium_veg_max
        
        # Calculate confidence based on NDVI strength
        veg_confidence = calculate_confidence_linear(
            veg_ndvi,
            min_value=self.ndvi_threshold,
            max_value=0.8,
        )
        
        # Apply classifications
        veg_indices = np.where(veg_mask)[0]
        
        classifications[veg_indices[low_veg]] = 3
        classifications[veg_indices[medium_veg]] = 4
        classifications[veg_indices[high_veg]] = 5
        
        confidence[veg_indices] = veg_confidence
        mask[veg_indices] = True
        
        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=mask,
            metadata={
                "n_low_veg": np.sum(low_veg),
                "n_medium_veg": np.sum(medium_veg),
                "n_high_veg": np.sum(high_veg),
                "mean_ndvi": np.mean(veg_ndvi),
                "mean_height": np.mean(veg_height),
            },
        )


def demo_single_rule():
    """Demonstrate using a single custom rule."""
    print("=" * 70)
    print("Demo 1: Single Rule Evaluation")
    print("=" * 70)
    
    # Create synthetic point cloud data
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100
    
    # Create synthetic features
    features = {
        "planarity": np.random.rand(n_points),
        "roughness": np.random.rand(n_points) * 0.1,
        "height_above_ground": np.random.rand(n_points) * 10,
    }
    
    # Create and configure rule
    rule = FlatSurfaceRule(
        planarity_threshold=0.7,
        roughness_threshold=0.05,
        min_confidence=0.6,
    )
    
    print(f"\nRule: {rule.config.name}")
    print(f"Type: {rule.config.rule_type.value}")
    print(f"Priority: {rule.config.priority.value}")
    print(f"Required features: {rule.get_required_features()}")
    print(f"Optional features: {rule.get_optional_features()}")
    
    # Evaluate rule
    result = rule.evaluate(points, features)
    
    print(f"\nResults:")
    print(f"  Points classified: {result.n_classified}")
    print(f"  Mean confidence: {result.mean_confidence:.3f}")
    print(f"  Classification rate: {result.n_classified / n_points * 100:.1f}%")
    
    if result.metadata:
        print(f"\nMetadata:")
        for key, value in result.metadata.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")


def demo_multiple_rules():
    """Demonstrate using multiple rules together."""
    print("\n" + "=" * 70)
    print("Demo 2: Multiple Rule Evaluation")
    print("=" * 70)
    
    # Create synthetic point cloud with mixed features
    n_points = 2000
    points = np.random.rand(n_points, 3) * 100
    
    # Create features with distinct regions
    features = {
        "planarity": np.random.rand(n_points),
        "roughness": np.random.rand(n_points) * 0.15,
        "height_above_ground": np.random.rand(n_points) * 20,
        "ndvi": np.random.rand(n_points),
    }
    
    # Create multiple rules
    rules = [
        FlatSurfaceRule(min_confidence=0.6),
        VegetationHeightRule(ndvi_threshold=0.3),
    ]
    
    print(f"\nEvaluating {len(rules)} rules on {n_points} points...\n")
    
    # Evaluate each rule
    results = []
    for rule in rules:
        result = rule.evaluate(points, features)
        results.append(result)
        
        print(f"Rule: {rule.config.name}")
        print(f"  Classified: {result.n_classified} points ({result.n_classified/n_points*100:.1f}%)")
        print(f"  Mean confidence: {result.mean_confidence:.3f}")
        print()
    
    # Combine results (simple approach: first rule wins)
    final_labels = np.zeros(n_points, dtype=np.int32)
    final_confidence = np.zeros(n_points, dtype=np.float32)
    
    for result in results:
        mask = result.mask
        final_labels[mask] = result.classifications[mask]
        final_confidence[mask] = result.confidence[mask]
    
    n_classified = np.sum(final_labels > 0)
    print(f"Combined Results:")
    print(f"  Total classified: {n_classified} points ({n_classified/n_points*100:.1f}%)")
    print(f"  Mean confidence: {np.mean(final_confidence[final_labels > 0]):.3f}")
    
    # Count classifications
    unique_labels, counts = np.unique(final_labels[final_labels > 0], return_counts=True)
    print(f"\nClassification breakdown:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} points ({count/n_classified*100:.1f}%)")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("Custom Geometric Rules - Usage Examples")
    print("=" * 70)
    print("\nThis demo shows how to create and use custom rules with the")
    print("rules framework introduced in Phase 4B.\n")
    
    # Run demonstrations
    demo_single_rule()
    demo_multiple_rules()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Inherit from BaseRule and implement required methods")
    print("2. Use confidence calculation utilities for scoring")
    print("3. Validate inputs using provided validation functions")
    print("4. Return RuleResult with classifications and metadata")
    print("5. Combine multiple rules for comprehensive classification")
    print("\nSee docs/ for migration guides and more examples.")


if __name__ == "__main__":
    main()
