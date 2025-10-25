---
sidebar_position: 5
title: Rules Framework
---

# Rules-Based Classification Framework

**Version:** 3.2.0+  
**Status:** Production Ready

The Rules Framework provides an extensible, plugin-based system for implementing custom classification rules with confidence scoring, hierarchical execution, and conflict resolution.

---

## 🎯 Overview

The Rules Framework enables you to:

- **Create custom rules** without modifying the framework
- **Score confidence** using 7 different methods
- **Combine rules hierarchically** with multiple strategies
- **Resolve conflicts** between competing classifications
- **Track performance** per rule and execution level
- **Validate features** with comprehensive quality checks

### Key Concepts

```python
from ign_lidar.core.classification.rules import (
    BaseRule,           # Abstract base for custom rules
    RuleEngine,         # Execute rules with conflict resolution
    HierarchicalRuleEngine,  # Multi-level classification
    RuleResult,         # Type-safe results
    ConfidenceMethod,   # 7 scoring methods
)
```

---

## 🚀 Quick Start

### Creating Your First Rule

```python
from ign_lidar.core.classification.rules import BaseRule, RuleResult
import numpy as np

class BuildingHeightRule(BaseRule):
    """Classify points as buildings based on height."""

    def __init__(self, min_height: float = 3.0):
        super().__init__(
            name="building_height",
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.HIGH,
        )
        self.min_height = min_height

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Classify points above minimum height as buildings."""
        # Get height feature
        height = context.additional_features.get('height')

        # Apply rule
        mask = height > self.min_height
        point_indices = np.where(mask)[0]

        # Assign building class (6 in ASPRS)
        classifications = np.full(len(point_indices), 6, dtype=np.int32)

        # Calculate confidence (linear scaling)
        confidence = np.clip((height[mask] - self.min_height) / 10.0, 0, 1)

        return RuleResult(
            point_indices=point_indices,
            classifications=classifications,
            confidence_scores=confidence,
        )
```

### Using the Rule

```python
from ign_lidar.core.classification.rules import RuleEngine

# Create engine and add rule
engine = RuleEngine()
engine.add_rule(BuildingHeightRule(min_height=3.0))

# Execute on point cloud
points = np.array([[x, y, z], ...])
labels = np.zeros(len(points), dtype=np.int32)
features = {'height': height_array}

result = engine.execute(
    points=points,
    labels=labels,
    additional_features=features,
)

print(f"Classified {len(result.point_indices)} points")
print(f"Average confidence: {result.confidence_scores.mean():.2f}")
```

---

## 📊 Confidence Scoring Methods

The framework provides 7 confidence calculation methods:

### 1. Binary Confidence

Simple 0 or 1 based on threshold:

```python
from ign_lidar.core.classification.rules import calculate_binary_confidence

confidence = calculate_binary_confidence(
    values=height_array,
    threshold=3.0,
    above=True  # True for values > threshold
)
```

### 2. Linear Confidence

Linear scaling between min and max:

```python
confidence = calculate_linear_confidence(
    values=height_array,
    min_value=0.0,
    max_value=20.0,
)
```

### 3. Sigmoid Confidence

Smooth S-curve transition:

```python
confidence = calculate_sigmoid_confidence(
    values=height_array,
    midpoint=5.0,
    steepness=1.0,
)
```

### 4. Gaussian Confidence

Bell curve around target value:

```python
confidence = calculate_gaussian_confidence(
    values=planarity_array,
    target=1.0,
    sigma=0.2,
)
```

### 5. Threshold-Based

Stepped confidence levels:

```python
confidence = calculate_threshold_confidence(
    values=ndvi_array,
    thresholds=[0.2, 0.4, 0.6],
    confidences=[0.25, 0.5, 0.75, 1.0],
)
```

### 6. Exponential Confidence

Exponential growth/decay:

```python
confidence = calculate_exponential_confidence(
    values=curvature_array,
    rate=2.0,
    increasing=False,
)
```

### 7. Composite Confidence

Combine multiple features:

```python
confidence = calculate_composite_confidence(
    feature_dict={
        'height': height_array,
        'planarity': planarity_array,
        'verticality': verticality_array,
    },
    weights={
        'height': 0.4,
        'planarity': 0.3,
        'verticality': 0.3,
    },
    methods={
        'height': ConfidenceMethod.LINEAR,
        'planarity': ConfidenceMethod.GAUSSIAN,
        'verticality': ConfidenceMethod.LINEAR,
    },
)
```

---

## 🏗️ Hierarchical Classification

Process classifications in multiple levels with different strategies:

```python
from ign_lidar.core.classification.rules import (
    HierarchicalRuleEngine,
    RuleLevel,
    ExecutionStrategy,
)

# Create hierarchical engine
engine = HierarchicalRuleEngine()

# Level 1: Coarse classification
level1 = RuleLevel(
    name="coarse",
    rules=[
        GroundRule(),
        VegetationRule(),
        BuildingRule(),
    ],
    strategy=ExecutionStrategy.PRIORITY,
)
engine.add_level(level1)

# Level 2: Fine refinement
level2 = RuleLevel(
    name="fine",
    rules=[
        LowVegetationRule(),
        HighVegetationRule(),
        WallRule(),
        RoofRule(),
    ],
    strategy=ExecutionStrategy.WEIGHTED,
)
engine.add_level(level2)

# Execute hierarchically
result = engine.execute(points, labels, features)
```

### Execution Strategies

- **FIRST_MATCH**: Stop at first rule that matches (fast)
- **ALL_MATCHES**: Execute all rules (comprehensive)
- **PRIORITY**: Execute by priority, stop on match
- **WEIGHTED**: Combine results by confidence weights

---

## 🔧 Feature Validation

Validate features before classification:

```python
from ign_lidar.core.classification.rules import (
    validate_features,
    FeatureRequirements,
)

# Define requirements
requirements = FeatureRequirements(
    required=['height', 'planarity', 'normals'],
    optional=['ndvi', 'intensity'],
    shapes={
        'height': (n_points,),
        'planarity': (n_points,),
        'normals': (n_points, 3),
    },
)

# Validate
is_valid, missing, invalid_shapes = validate_features(
    features=feature_dict,
    requirements=requirements,
)

if not is_valid:
    raise ValueError(f"Missing features: {missing}")
```

---

## 📈 Performance Tracking

Track rule execution performance:

```python
result = engine.execute(points, labels, features)

# Per-rule statistics
for rule_name, stats in result.rule_stats.items():
    print(f"{rule_name}:")
    print(f"  Execution time: {stats.execution_time:.3f}s")
    print(f"  Points classified: {stats.points_classified}")
    print(f"  Average confidence: {stats.confidence_mean:.2f}")
```

---

## 🎓 Complete Examples

### Multi-Feature Building Detection

```python
class BuildingDetectionRule(BaseRule):
    """Detect buildings using multiple features."""

    def evaluate(self, context: RuleContext) -> RuleResult:
        # Get features
        height = context.additional_features['height']
        planarity = context.additional_features['planarity']
        verticality = context.additional_features['verticality']
        ndvi = context.additional_features.get('ndvi', None)

        # Building criteria
        height_mask = height > 2.0
        planar_mask = planarity > 0.7
        vertical_mask = verticality > 0.5

        # Vegetation filter
        if ndvi is not None:
            veg_mask = ndvi < 0.3
            mask = height_mask & planar_mask & vertical_mask & veg_mask
        else:
            mask = height_mask & planar_mask & vertical_mask

        # Calculate composite confidence
        confidence = calculate_composite_confidence(
            feature_dict={
                'height': height[mask],
                'planarity': planarity[mask],
                'verticality': verticality[mask],
            },
            weights={'height': 0.3, 'planarity': 0.4, 'verticality': 0.3},
            methods={
                'height': ConfidenceMethod.LINEAR,
                'planarity': ConfidenceMethod.GAUSSIAN,
                'verticality': ConfidenceMethod.LINEAR,
            },
        )

        return RuleResult(
            point_indices=np.where(mask)[0],
            classifications=np.full(mask.sum(), 6, dtype=np.int32),
            confidence_scores=confidence,
        )
```

### Context-Aware Classification

```python
class ContextAwareGroundRule(BaseRule):
    """Classify ground considering neighborhood context."""

    def evaluate(self, context: RuleContext) -> RuleResult:
        # Get features
        height = context.additional_features['height']
        planarity = context.additional_features['planarity']

        # Identify low, flat points
        mask = (height < 0.5) & (planarity > 0.9)

        # Check neighborhood consistency
        from scipy.spatial import cKDTree
        tree = cKDTree(context.points)

        candidates = np.where(mask)[0]
        confirmed = []
        confidences = []

        for idx in candidates:
            # Find neighbors
            neighbors = tree.query_ball_point(context.points[idx], r=2.0)
            neighbor_heights = height[neighbors]

            # Check if neighbors are also low
            if np.mean(neighbor_heights < 1.0) > 0.7:
                confirmed.append(idx)
                # Higher confidence with more consistent neighbors
                conf = np.mean(neighbor_heights < 1.0)
                confidences.append(conf)

        return RuleResult(
            point_indices=np.array(confirmed, dtype=np.int32),
            classifications=np.full(len(confirmed), 2, dtype=np.int32),  # Ground
            confidence_scores=np.array(confidences, dtype=np.float32),
        )
```

---

## 📚 Additional Resources

- **Quick Reference**: [`RULES_FRAMEWORK_QUICK_REFERENCE.md`](../../RULES_FRAMEWORK_QUICK_REFERENCE.md)
- **Developer Guide**: [`RULES_FRAMEWORK_DEVELOPER_GUIDE.md`](../../RULES_FRAMEWORK_DEVELOPER_GUIDE.md)
- **Architecture**: [`RULES_FRAMEWORK_ARCHITECTURE.md`](../../RULES_FRAMEWORK_ARCHITECTURE.md)
- **Examples**: [`examples/`](../../examples/) - demo_custom_geometric_rule.py, demo_hierarchical_rules.py, demo_confidence_scoring.py

---

## 🔗 API Reference

See [Rules API Reference](../api/rules.md) for complete API documentation.

---

## 💡 Best Practices

1. **Start Simple**: Begin with single-feature rules, then combine
2. **Validate Features**: Always check feature availability and quality
3. **Use Appropriate Confidence**: Choose method matching your data distribution
4. **Test Incrementally**: Test each rule independently before combining
5. **Monitor Performance**: Track execution time and classified points
6. **Document Rules**: Clear docstrings explaining criteria and thresholds
7. **Handle Edge Cases**: Check for empty results, missing features, invalid values

---

## 🐛 Troubleshooting

### No Points Classified

```python
# Check feature values
print(f"Height range: {height.min():.2f} - {height.max():.2f}")
print(f"Points above threshold: {(height > 3.0).sum()}")

# Adjust thresholds
rule = BuildingHeightRule(min_height=2.0)  # Lower threshold
```

### Low Confidence Scores

```python
# Check confidence method
confidence = calculate_linear_confidence(
    values=height,
    min_value=height.min(),  # Use actual data range
    max_value=height.max(),
)
```

### Performance Issues

```python
# Use efficient strategies
engine = HierarchicalRuleEngine()
level = RuleLevel(
    rules=[...],
    strategy=ExecutionStrategy.FIRST_MATCH,  # Stop on first match
)
```

---

**Framework Version:** 3.2.0+  
**Last Updated:** October 25, 2025
