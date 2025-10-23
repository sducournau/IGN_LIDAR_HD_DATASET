# Rules Framework Developer Guide

**Version:** 3.2.0  
**Phase:** 4B - Rules Infrastructure  
**Status:** Production Ready  
**Last Updated:** October 23, 2025

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Creating Custom Rules](#creating-custom-rules)
5. [Confidence Scoring](#confidence-scoring)
6. [Hierarchical Classification](#hierarchical-classification)
7. [Feature Management](#feature-management)
8. [Best Practices](#best-practices)
9. [Performance Optimization](#performance-optimization)
10. [Testing and Validation](#testing-and-validation)
11. [Integration Patterns](#integration-patterns)
12. [Troubleshooting](#troubleshooting)
13. [API Reference](#api-reference)

---

## Introduction

### What is the Rules Framework?

The rules framework provides a modern, extensible infrastructure for implementing rule-based point cloud classification in the IGN LIDAR HD Dataset project. It was introduced in **Phase 4B** of the classification module consolidation.

### Key Features

- ✅ **Type-safe**: Dataclasses and enums throughout
- ✅ **Extensible**: Plugin architecture via abstract base classes
- ✅ **Flexible**: Multiple execution strategies and confidence methods
- ✅ **Performant**: Vectorized operations with NumPy
- ✅ **Well-tested**: Comprehensive validation and error handling
- ✅ **Production-ready**: Used in active classification pipelines

### When to Use This Framework

**Use the rules framework when**:

- Implementing new classification rules
- Combining multiple criteria for classification decisions
- Need hierarchical or sequential classification
- Require confidence scoring for classifications
- Want type-safe, maintainable rule implementations

**Consider alternatives when**:

- Simple threshold-based classification (use thresholds module)
- Deep learning classification (use separate ML pipeline)
- Performance-critical inner loops (optimize after profiling)

---

## Quick Start

### Installation

The rules framework is included in the IGN LIDAR HD Dataset package:

```bash
pip install ign-lidar-hd
# Or for development
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
cd IGN_LIDAR_HD_DATASET
pip install -e .
```

### Minimal Example

```python
import numpy as np
from ign_lidar.core.classification.rules import (
    BaseRule, RuleResult, RuleConfig, RuleType, RulePriority
)

class SimpleGroundRule(BaseRule):
    """Classify points as ground if height < threshold."""

    def __init__(self, height_threshold: float = 0.2):
        config = RuleConfig(
            name="simple_ground",
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.HIGH,
        )
        super().__init__(config)
        self.height_threshold = height_threshold

    def get_required_features(self):
        return {"height_above_ground"}

    def evaluate(self, points, features, labels=None, **kwargs):
        height = features["height_above_ground"]
        ground_mask = height < self.height_threshold

        classifications = np.zeros(len(points), dtype=np.int32)
        classifications[ground_mask] = 2  # Ground class

        confidence = np.ones(len(points), dtype=np.float32)
        confidence[ground_mask] = 0.9

        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=ground_mask,
        )

# Use the rule
rule = SimpleGroundRule(height_threshold=0.2)
points = np.random.rand(1000, 3) * 100
features = {"height_above_ground": np.random.rand(1000) * 10}
result = rule.evaluate(points, features)

print(f"Classified {result.n_classified} points as ground")
print(f"Mean confidence: {result.mean_confidence:.3f}")
```

### Running Examples

The package includes three comprehensive examples:

```bash
cd examples

# 1. Custom rule creation
python demo_custom_geometric_rule.py

# 2. Hierarchical classification
python demo_hierarchical_rules.py

# 3. Confidence scoring methods
python demo_confidence_scoring.py
```

See `examples/README_RULES_EXAMPLES.md` for detailed documentation of all examples.

---

## Core Concepts

### Architecture Overview

```
BaseRule (Abstract)
├─ evaluate()           # Core classification logic
├─ validate_features()  # Feature validation
└─ get_required/optional_features()  # Feature dependencies

RuleEngine (Abstract)
├─ apply_rules()        # Execute multiple rules
├─ add_rule()           # Add rules dynamically
└─ get_rules()          # Query registered rules

HierarchicalRuleEngine (Concrete)
├─ Sequential levels    # Process levels in order
├─ Multiple strategies  # Different execution per level
└─ Conflict resolution  # Handle overlapping classifications
```

### Key Classes

**`BaseRule`** - Abstract base class for all rules

- Defines interface for rule implementation
- Handles configuration and metadata
- Provides validation utilities

**`RuleEngine`** - Abstract engine for rule execution

- Manages collection of rules
- Orchestrates rule application
- Provides querying capabilities

**`HierarchicalRuleEngine`** - Multi-level rule execution

- Organizes rules into hierarchical levels
- Supports different execution strategies per level
- Handles conflicts between rules

**`RuleResult`** - Container for classification results

- Classifications array (int32)
- Confidence scores array (float32)
- Boolean mask of classified points
- Metadata dictionary for debugging

### Key Enums

**`RuleType`** - Type of rule logic:

- `GEOMETRIC` - Based on geometric features
- `SPECTRAL` - Based on spectral/color features
- `CONTEXTUAL` - Based on spatial context
- `HYBRID` - Combination of multiple types
- `LEARNING_BASED` - ML-based rules
- `CUSTOM` - User-defined types

**`RulePriority`** - Execution priority:

- `CRITICAL` (value: 100) - Must execute first
- `HIGH` (value: 75) - Important rules
- `NORMAL` (value: 50) - Standard priority
- `LOW` (value: 25) - Optional refinements

**`ExecutionStrategy`** - How to execute multiple rules:

- `FIRST_MATCH` - Stop after first successful rule
- `ALL_MATCHES` - Execute all rules
- `PRIORITY` - Execute by priority, stop at first match
- `WEIGHTED_COMBINATION` - Combine results with weights

**`ConflictResolution`** - How to handle overlapping classifications:

- `HIGHEST_CONFIDENCE` - Use classification with highest confidence
- `HIGHEST_PRIORITY` - Use classification from highest priority rule
- `WEIGHTED_AVERAGE` - Average confidences weighted by priority
- `FIRST_WINS` - Keep first classification
- `LAST_WINS` - Keep last classification

---

## Creating Custom Rules

### Step-by-Step Guide

#### Step 1: Define Your Rule Class

```python
from typing import Dict, Set, Optional
import numpy as np
from ign_lidar.core.classification.rules import (
    BaseRule, RuleResult, RuleConfig, RuleType, RulePriority
)

class MyCustomRule(BaseRule):
    """Brief description of what this rule does."""

    def __init__(self, param1: float, param2: str = "default"):
        # Create configuration
        config = RuleConfig(
            name="my_custom_rule",
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.NORMAL,
            min_confidence=0.6,
            enabled=True,
        )
        super().__init__(config)

        # Store parameters
        self.param1 = param1
        self.param2 = param2
```

#### Step 2: Declare Feature Requirements

```python
    def get_required_features(self) -> Set[str]:
        """Features that MUST be present."""
        return {"height_above_ground", "planarity"}

    def get_optional_features(self) -> Set[str]:
        """Features that enhance the rule if available."""
        return {"intensity", "return_number"}
```

#### Step 3: Implement Evaluation Logic

```python
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
        # 1. Validate inputs
        from ign_lidar.core.classification.rules import (
            validate_point_cloud_shape,
            validate_required_features,
        )
        validate_point_cloud_shape(points)
        validate_required_features(features, self.get_required_features())

        n_points = len(points)

        # 2. Extract features
        height = features["height_above_ground"]
        planarity = features["planarity"]

        # 3. Apply classification logic
        my_mask = (height > self.param1) & (planarity > 0.7)

        # Skip already classified if labels provided
        if labels is not None:
            my_mask &= (labels == 0)  # Only unclassified

        # 4. Calculate confidence
        from ign_lidar.core.classification.rules import (
            calculate_confidence_linear,
        )
        confidence_scores = calculate_confidence_linear(
            planarity[my_mask],
            min_value=0.7,
            max_value=1.0,
        )

        # 5. Build result arrays
        classifications = np.zeros(n_points, dtype=np.int32)
        confidence = np.zeros(n_points, dtype=np.float32)

        classifications[my_mask] = 6  # Your class code
        confidence[my_mask] = confidence_scores

        # 6. Return result
        return RuleResult(
            rule_name=self.config.name,
            rule_type=self.config.rule_type,
            classifications=classifications,
            confidence=confidence,
            mask=my_mask,
            metadata={
                "n_classified": np.sum(my_mask),
                "mean_confidence": np.mean(confidence_scores),
                "param1_used": self.param1,
            },
        )
```

### Rule Implementation Checklist

- [ ] Inherit from `BaseRule`
- [ ] Create `RuleConfig` in `__init__`
- [ ] Implement `get_required_features()`
- [ ] Implement `get_optional_features()` (if needed)
- [ ] Implement `evaluate()` method
- [ ] Validate inputs (point cloud shape, required features)
- [ ] Handle optional `labels` parameter (skip classified points)
- [ ] Calculate confidence scores appropriately
- [ ] Return `RuleResult` with all fields populated
- [ ] Add metadata for debugging/logging
- [ ] Write docstrings for class and methods

---

## Confidence Scoring

### Available Methods

The framework provides **7 confidence calculation methods**:

#### 1. Binary Confidence

**Use case**: Simple threshold-based decisions

```python
from ign_lidar.core.classification.rules import calculate_confidence_binary

# Returns 1.0 if value >= threshold, else 0.0
confidence = calculate_confidence_binary(
    values=ndvi,
    threshold=0.3,
    reverse=False,  # Set True to invert logic
)
```

**Example**: NDVI > 0.3 → vegetation (confidence 1.0), else not vegetation (confidence 0.0)

#### 2. Linear Confidence

**Use case**: Gradual increase over a range

```python
from ign_lidar.core.classification.rules import calculate_confidence_linear

# Linear mapping from [min_value, max_value] to [0, 1]
confidence = calculate_confidence_linear(
    values=planarity,
    min_value=0.7,
    max_value=1.0,
)
```

**Example**: Planarity 0.7 → 0.0 confidence, 0.85 → 0.5 confidence, 1.0 → 1.0 confidence

#### 3. Sigmoid Confidence

**Use case**: Smooth S-curve transitions

```python
from ign_lidar.core.classification.rules import calculate_confidence_sigmoid

# S-curve centered at midpoint
confidence = calculate_confidence_sigmoid(
    values=height,
    midpoint=5.0,
    steepness=2.0,  # Higher = sharper transition
)
```

**Example**: Smooth transition around 5m height, sharp near midpoint

#### 4. Gaussian Confidence

**Use case**: Target value with tolerance

```python
from ign_lidar.core.classification.rules import calculate_confidence_gaussian

# Bell curve around target
confidence = calculate_confidence_gaussian(
    values=height,
    mean=3.0,      # Target height
    std=1.0,       # Tolerance
)
```

**Example**: Looking for objects ~3m tall, confidence decreases with distance from 3m

#### 5. Threshold Confidence

**Use case**: Distance from threshold

```python
from ign_lidar.core.classification.rules import calculate_confidence_threshold

# Confidence based on distance from threshold
confidence = calculate_confidence_threshold(
    values=roughness,
    threshold=0.05,
    reverse=True,  # Lower values = higher confidence
)
```

**Example**: Smooth surfaces (low roughness) get high confidence

#### 6. Exponential Confidence

**Use case**: Rapid growth for strong signals

```python
from ign_lidar.core.classification.rules import calculate_confidence_exponential

# Exponential growth
confidence = calculate_confidence_exponential(
    values=intensity,
    rate=3.0,  # Growth rate
)
```

**Example**: Rapidly increase confidence for high-intensity returns

#### 7. Composite Confidence

**Use case**: Combine multiple methods

```python
# Combine different confidence methods
conf1 = calculate_confidence_linear(planarity, 0.7, 1.0)
conf2 = calculate_confidence_threshold(roughness, 0.05, reverse=True)
conf3 = calculate_confidence_gaussian(height, mean=5.0, std=2.0)

# Weighted combination
from ign_lidar.core.classification.rules import combine_confidence_weighted
final_conf = combine_confidence_weighted(
    [conf1, conf2, conf3],
    weights=[0.5, 0.3, 0.2]
)
```

### Combination Strategies

The framework provides **6 confidence combination strategies**:

```python
from ign_lidar.core.classification.rules import (
    combine_confidence_weighted,
    combine_confidence_max,
    combine_confidence_min,
    combine_confidence_product,
    combine_confidence_geometric_mean,
    combine_confidence_harmonic_mean,
)

# 1. Weighted Average (most common)
conf = combine_confidence_weighted([conf1, conf2], weights=[0.7, 0.3])

# 2. Maximum (optimistic - accept if ANY confident)
conf = combine_confidence_max([conf1, conf2, conf3])

# 3. Minimum (conservative - ALL must be confident)
conf = combine_confidence_min([conf1, conf2, conf3])

# 4. Product (independent evidence)
conf = combine_confidence_product([conf1, conf2])

# 5. Geometric Mean (balanced)
conf = combine_confidence_geometric_mean([conf1, conf2])

# 6. Harmonic Mean (emphasizes lower values)
conf = combine_confidence_harmonic_mean([conf1, conf2])
```

### When to Use Each Method

| Method      | Use Case                   | Example                |
| ----------- | -------------------------- | ---------------------- |
| Binary      | Simple yes/no              | NDVI > threshold       |
| Linear      | Gradual transition         | Planarity 0-1          |
| Sigmoid     | Soft threshold             | Height around target   |
| Gaussian    | Specific value ± tolerance | Window height ~1.5m    |
| Threshold   | Distance-based             | Roughness < limit      |
| Exponential | Strong signal emphasis     | High intensity returns |

| Combination  | Use Case              | Example                 |
| ------------ | --------------------- | ----------------------- |
| Weighted Avg | Different importance  | Planarity 70%, NDVI 30% |
| Max          | Any criterion passes  | Vegetation OR water     |
| Min          | All criteria required | Building AND elevated   |
| Product      | Independent evidence  | Multiple weak signals   |
| Geometric    | Balanced combination  | Equal-weighted criteria |
| Harmonic     | Conservative blend    | Emphasize weakest       |

---

## Hierarchical Classification

### Concept

Hierarchical classification organizes rules into levels that execute sequentially:

```
Level 1 (Ground Detection) → CRITICAL priority
  └─ Identifies ground points first

Level 2 (Primary Objects) → HIGH priority
  ├─ Buildings
  ├─ Vegetation
  └─ Water

Level 3 (Refinement) → NORMAL priority
  └─ Context-based adjustments
```

### Creating a Hierarchical Engine

```python
from ign_lidar.core.classification.rules import (
    HierarchicalRuleEngine,
    RuleLevel,
    ExecutionStrategy,
    ConflictResolution,
)

# Define levels
levels = [
    RuleLevel(
        name="ground_detection",
        priority=1,
        rules=[GroundRule()],
        strategy=ExecutionStrategy.FIRST_MATCH,
        description="Identify ground points",
    ),
    RuleLevel(
        name="primary_classification",
        priority=2,
        rules=[BuildingRule(), VegetationRule(), WaterRule()],
        strategy=ExecutionStrategy.PRIORITY,
        description="Classify main object types",
    ),
    RuleLevel(
        name="refinement",
        priority=3,
        rules=[RefinementRule()],
        strategy=ExecutionStrategy.ALL_MATCHES,
        description="Refine classifications",
    ),
]

# Create engine
engine = HierarchicalRuleEngine(
    levels=levels,
    conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE,
    enable_stats=True,
)

# Apply rules
result = engine.apply_rules(points, features)
```

### Execution Strategies

#### FIRST_MATCH

Stop after first rule succeeds:

```python
RuleLevel(
    name="ground_detection",
    rules=[GroundRule()],
    strategy=ExecutionStrategy.FIRST_MATCH,  # Stop after ground found
)
```

**Use when**: One rule should handle all cases in level

#### ALL_MATCHES

Execute all rules in level:

```python
RuleLevel(
    name="refinement",
    rules=[Refine1(), Refine2(), Refine3()],
    strategy=ExecutionStrategy.ALL_MATCHES,  # Run all refinements
)
```

**Use when**: Multiple complementary rules that don't overlap

#### PRIORITY

Execute by priority, stop at first match:

```python
RuleLevel(
    name="classification",
    rules=[
        BuildingRule(priority=RulePriority.HIGH),
        VegetationRule(priority=RulePriority.NORMAL),
        DefaultRule(priority=RulePriority.LOW),
    ],
    strategy=ExecutionStrategy.PRIORITY,  # Try high priority first
)
```

**Use when**: Rules have fallback hierarchy

#### WEIGHTED_COMBINATION

Combine all rule outputs with weights:

```python
RuleLevel(
    name="fusion",
    rules=[GeometricRule(), SpectralRule(), ContextRule()],
    strategy=ExecutionStrategy.WEIGHTED_COMBINATION,  # Blend all
)
```

**Use when**: Multiple rules provide complementary evidence

### Conflict Resolution

When multiple rules classify the same point:

```python
engine = HierarchicalRuleEngine(
    levels=levels,
    conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE,
)
```

**Options**:

- `HIGHEST_CONFIDENCE` - Keep classification with highest confidence
- `HIGHEST_PRIORITY` - Keep classification from highest priority rule
- `WEIGHTED_AVERAGE` - Blend confidences weighted by rule priority
- `FIRST_WINS` - Keep first classification
- `LAST_WINS` - Keep last classification (overwrite)

---

## Feature Management

### Required vs Optional Features

**Required features** must be present:

```python
def get_required_features(self) -> Set[str]:
    return {"height_above_ground", "planarity"}
```

If missing, `validate_required_features()` raises an error.

**Optional features** enhance the rule:

```python
def get_optional_features(self) -> Set[str]:
    return {"intensity", "return_number", "ndvi"}

def evaluate(self, points, features, **kwargs):
    # Check if optional feature available
    if "intensity" in features:
        # Use intensity to refine classification
        intensity_factor = ...
    else:
        # Proceed without intensity
        intensity_factor = 1.0
```

### Feature Validation

The framework provides validation utilities:

```python
from ign_lidar.core.classification.rules import (
    validate_point_cloud_shape,
    validate_required_features,
    validate_feature_shape,
    validate_feature_range,
    validate_feature_quality,
)

def evaluate(self, points, features, **kwargs):
    # Validate point cloud
    validate_point_cloud_shape(points)  # Checks (N, 3) shape

    # Validate required features present
    validate_required_features(features, {"height", "planarity"})

    # Validate feature shape
    validate_feature_shape(features["height"], len(points))

    # Validate value range
    validate_feature_range(
        features["planarity"],
        min_value=0.0,
        max_value=1.0,
        feature_name="planarity"
    )

    # Validate quality (no NaN/inf)
    validate_feature_quality(
        features["height"],
        allow_nan=False,
        allow_inf=False,
    )
```

### Feature Requirements Dataclass

For complex requirements:

```python
from ign_lidar.core.classification.rules import FeatureRequirements

requirements = FeatureRequirements(
    required={"height_above_ground", "planarity"},
    optional={"intensity", "ndvi"},
    shape_requirements={"height_above_ground": (None,)},  # 1D array
    dtype_requirements={"planarity": np.float32},
    value_ranges={"planarity": (0.0, 1.0)},
)

# Validate all at once
requirements.validate(features, n_points=len(points))
```

---

## Best Practices

### 1. Rule Design

**✅ DO**:

- Keep rules focused on single classification task
- Use descriptive names (`BuildingDetectionRule` not `Rule1`)
- Document required features and thresholds
- Include metadata for debugging
- Handle edge cases gracefully

**❌ DON'T**:

- Create monolithic rules doing multiple things
- Hard-code magic numbers (use class parameters)
- Ignore optional labels parameter
- Skip input validation
- Forget to handle empty results

### 2. Confidence Scoring

**✅ DO**:

- Choose confidence method matching your criterion
- Combine multiple criteria with appropriate strategy
- Calibrate confidence to match actual accuracy
- Use meaningful confidence ranges (0.0 to 1.0)

**❌ DON'T**:

- Always use binary confidence (too restrictive)
- Combine confidences without considering method
- Return constant confidence for all points
- Use confidence > 1.0 or < 0.0

### 3. Feature Usage

**✅ DO**:

- Validate required features are present
- Check optional features before using
- Handle missing/invalid values gracefully
- Document feature requirements clearly

**❌ DON'T**:

- Assume features exist without checking
- Fail silently on missing optional features
- Use features without understanding their meaning
- Mix incompatible feature types

### 4. Performance

**✅ DO**:

- Use vectorized NumPy operations
- Pre-compute expensive features once
- Use boolean indexing for masking
- Profile before optimizing

**❌ DON'T**:

- Loop over individual points
- Recompute same features multiple times
- Create unnecessary copies of large arrays
- Optimize prematurely

### 5. Testing

**✅ DO**:

- Test with real point cloud data
- Test edge cases (empty, single point, all classified)
- Verify confidence ranges
- Check feature validation works

**❌ DON'T**:

- Test only with synthetic data
- Skip edge case testing
- Assume validation catches everything
- Ignore performance in tests

---

## Performance Optimization

### Memory Efficiency

**Problem**: Large point clouds consume memory

**Solution**: Use views and masking

```python
# ✅ GOOD: Use boolean mask
mask = height < 0.2
ground_points = points[mask]  # Creates view, not copy

# ❌ BAD: Create unnecessary copies
ground_points = points.copy()
for i in range(len(points)):
    if height[i] < 0.2:
        ground_points.append(points[i])
```

### Computational Efficiency

**Problem**: Slow rule execution

**Solution**: Vectorize operations

```python
# ✅ GOOD: Vectorized
mask = (height < 5) & (planarity > 0.8) & (ndvi < 0.2)
classifications[mask] = 6

# ❌ BAD: Loops
for i in range(len(points)):
    if height[i] < 5 and planarity[i] > 0.8 and ndvi[i] < 0.2:
        classifications[i] = 6
```

### Pre-computation

**Problem**: Recomputing same features

**Solution**: Compute once, reuse

```python
# ✅ GOOD: Compute features once
features = {
    "height": compute_height(points),
    "planarity": compute_planarity(points),
    "ndvi": compute_ndvi(points, colors),
}

# Apply multiple rules
for rule in rules:
    result = rule.evaluate(points, features)  # Reuses features

# ❌ BAD: Recompute per rule
for rule in rules:
    height = compute_height(points)  # Slow!
    result = rule.evaluate(points, {"height": height})
```

### Profiling

Use profiling to find bottlenecks:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code
result = engine.apply_rules(points, features)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 slowest functions
```

---

## Testing and Validation

### Unit Testing Rules

```python
import unittest
import numpy as np
from my_rules import MyCustomRule

class TestMyCustomRule(unittest.TestCase):
    def setUp(self):
        self.rule = MyCustomRule(param1=5.0)
        self.points = np.random.rand(1000, 3) * 100
        self.features = {
            "height_above_ground": np.random.rand(1000) * 20,
            "planarity": np.random.rand(1000),
        }

    def test_required_features(self):
        """Test required features are declared."""
        required = self.rule.get_required_features()
        self.assertIn("height_above_ground", required)
        self.assertIn("planarity", required)

    def test_evaluate_returns_result(self):
        """Test evaluate returns RuleResult."""
        result = self.rule.evaluate(self.points, self.features)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.classifications), len(self.points))
        self.assertEqual(len(result.confidence), len(self.points))

    def test_confidence_range(self):
        """Test confidence scores in [0, 1]."""
        result = self.rule.evaluate(self.points, self.features)
        self.assertTrue(np.all(result.confidence >= 0.0))
        self.assertTrue(np.all(result.confidence <= 1.0))

    def test_empty_points(self):
        """Test with empty point cloud."""
        empty_points = np.empty((0, 3))
        empty_features = {
            "height_above_ground": np.empty(0),
            "planarity": np.empty(0),
        }
        result = self.rule.evaluate(empty_points, empty_features)
        self.assertEqual(result.n_classified, 0)

    def test_missing_feature_raises(self):
        """Test missing required feature raises error."""
        bad_features = {"planarity": self.features["planarity"]}
        with self.assertRaises(ValueError):
            self.rule.evaluate(self.points, bad_features)
```

### Integration Testing

```python
def test_hierarchical_pipeline():
    """Test complete hierarchical classification."""
    # Setup
    points = load_test_point_cloud()
    features = compute_features(points)

    # Create pipeline
    levels = [
        RuleLevel("ground", rules=[GroundRule()], ...),
        RuleLevel("objects", rules=[BuildingRule(), TreeRule()], ...),
    ]
    engine = HierarchicalRuleEngine(levels=levels)

    # Execute
    result = engine.apply_rules(points, features)

    # Validate
    assert result.n_classified > 0
    assert result.mean_confidence > 0.5
    assert np.all(result.classifications >= 0)

    # Check coverage
    coverage = result.n_classified / len(points)
    assert coverage > 0.8  # At least 80% classified
```

### Validation Utilities

```python
from ign_lidar.core.classification.rules import validate_rule_result

def validate_rule_output(rule, points, features):
    """Validate rule output is correct."""
    result = rule.evaluate(points, features)

    # Use built-in validation
    validate_rule_result(result, n_points=len(points))

    # Additional checks
    assert result.rule_name == rule.config.name
    assert result.rule_type == rule.config.rule_type
    assert result.n_classified <= len(points)

    return result
```

---

## Integration Patterns

### Pattern 1: Single Rule Application

```python
# Simple single-rule classification
rule = BuildingDetectionRule(min_height=3.0, min_planarity=0.8)
result = rule.evaluate(points, features)
classifications = result.classifications
```

### Pattern 2: Sequential Rules

```python
# Apply rules in sequence, accumulating results
labels = np.zeros(len(points), dtype=np.int32)

for rule in [GroundRule(), BuildingRule(), VegetationRule()]:
    result = rule.evaluate(points, features, labels=labels)
    # Update labels where rule classified
    labels[result.mask] = result.classifications[result.mask]
```

### Pattern 3: Hierarchical Engine

```python
# Use hierarchical engine for complex pipelines
engine = HierarchicalRuleEngine(levels=levels)
result = engine.apply_rules(points, features)
labels = result.classifications
confidences = result.confidence
```

### Pattern 4: Confidence-Based Filtering

```python
# Apply rule and filter by confidence
result = rule.evaluate(points, features)

# Only accept high-confidence classifications
high_conf_mask = result.confidence > 0.8
labels = np.zeros(len(points), dtype=np.int32)
labels[high_conf_mask] = result.classifications[high_conf_mask]
```

### Pattern 5: Combining with Existing Classifiers

```python
# Integrate rules with existing classification pipeline

# 1. Run existing classifier
initial_labels = existing_classifier.predict(points)

# 2. Apply rules for refinement
rule_result = refinement_rule.evaluate(
    points, features, labels=initial_labels
)

# 3. Merge results (rules override where confident)
final_labels = initial_labels.copy()
override_mask = rule_result.confidence > 0.9
final_labels[override_mask] = rule_result.classifications[override_mask]
```

---

## Troubleshooting

### Common Issues

#### Issue: "Feature not found" error

**Problem**: Rule requires feature that isn't in features dict

**Solution**:

```python
# Check what features are required
print(rule.get_required_features())

# Ensure all required features are computed
required = rule.get_required_features()
for feat in required:
    if feat not in features:
        features[feat] = compute_feature(points, feat)
```

#### Issue: Low classification coverage

**Problem**: Very few points classified

**Solution**:

```python
# 1. Check confidence threshold
rule.config.min_confidence = 0.5  # Lower threshold

# 2. Debug which criteria fail
result = rule.evaluate(points, features)
print(f"Classified: {result.n_classified} / {len(points)}")
print(f"Metadata: {result.metadata}")

# 3. Check feature values
print(f"Height range: {features['height'].min():.2f} - {features['height'].max():.2f}")
print(f"Planarity range: {features['planarity'].min():.2f} - {features['planarity'].max():.2f}")
```

#### Issue: Confidence scores always 0 or 1

**Problem**: Using binary confidence when should use gradual

**Solution**:

```python
# ❌ BAD: Binary confidence
conf = calculate_confidence_binary(planarity, threshold=0.8)

# ✅ GOOD: Linear confidence
conf = calculate_confidence_linear(planarity, min_value=0.7, max_value=1.0)
```

#### Issue: Memory error with large point clouds

**Problem**: Loading entire cloud into memory

**Solution**:

```python
# Process in chunks
chunk_size = 10_000_000  # 10M points

for i in range(0, len(points), chunk_size):
    chunk_points = points[i:i+chunk_size]
    chunk_features = {
        k: v[i:i+chunk_size] for k, v in features.items()
    }
    chunk_result = rule.evaluate(chunk_points, chunk_features)
    # Process chunk_result...
```

#### Issue: Slow rule execution

**Problem**: Not using vectorized operations

**Solution**:

```python
# Profile to find bottleneck
import cProfile
cProfile.run('rule.evaluate(points, features)')

# Use vectorized operations (see Performance section)
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ign_lidar.core.classification.rules')

# Now rules will log detailed info
result = rule.evaluate(points, features)
```

### Validation Checklist

When a rule doesn't work as expected:

- [ ] Check required features are present
- [ ] Verify feature value ranges are sensible
- [ ] Confirm confidence calculation is appropriate
- [ ] Test with small synthetic dataset
- [ ] Print intermediate results/masks
- [ ] Check metadata for debugging info
- [ ] Verify point cloud shape is (N, 3)
- [ ] Ensure no NaN/inf values in features

---

## API Reference

### Quick Reference

**Import paths**:

```python
from ign_lidar.core.classification.rules import (
    # Base classes
    BaseRule, RuleEngine, HierarchicalRuleEngine,

    # Data containers
    RuleResult, RuleConfig, RuleLevel, FeatureRequirements,

    # Enums
    RuleType, RulePriority, ExecutionStrategy, ConflictResolution,

    # Confidence calculation
    calculate_confidence_binary,
    calculate_confidence_linear,
    calculate_confidence_sigmoid,
    calculate_confidence_gaussian,
    calculate_confidence_threshold,
    calculate_confidence_exponential,

    # Confidence combination
    combine_confidence_weighted,
    combine_confidence_max,
    combine_confidence_min,
    combine_confidence_product,
    combine_confidence_geometric_mean,
    combine_confidence_harmonic_mean,

    # Utilities
    normalize_confidence,
    calibrate_confidence,

    # Validation
    validate_required_features,
    validate_point_cloud_shape,
    validate_feature_shape,
    validate_feature_range,
    validate_feature_quality,
)
```

### Core Classes

See `ign_lidar/core/classification/rules/__init__.py` for complete API.

**BaseRule** - Abstract base class for rules

- `evaluate(points, features, labels, **kwargs) -> RuleResult`
- `get_required_features() -> Set[str]`
- `get_optional_features() -> Set[str]`
- `validate_features(features) -> bool`

**HierarchicalRuleEngine** - Multi-level rule execution

- `apply_rules(points, features, labels) -> RuleResult`
- `add_level(level: RuleLevel)`
- `get_level(name: str) -> RuleLevel`
- `get_stats() -> Dict`

**RuleResult** - Classification result container

- `classifications: np.ndarray` - Class labels (int32)
- `confidence: np.ndarray` - Confidence scores (float32)
- `mask: np.ndarray` - Classified points (bool)
- `metadata: Dict` - Additional information
- `n_classified: int` - Property: number of classified points
- `mean_confidence: float` - Property: average confidence

### Full Documentation

For complete API documentation, see:

- `ign_lidar/core/classification/rules/base.py` - Base classes
- `ign_lidar/core/classification/rules/confidence.py` - Confidence utilities
- `ign_lidar/core/classification/rules/validation.py` - Validation utilities
- `ign_lidar/core/classification/rules/hierarchy.py` - Hierarchical engine

---

## Additional Resources

### Documentation

- **Examples**: `examples/README_RULES_EXAMPLES.md`
- **Phase 4B Summary**: `docs/PHASE_4B_INFRASTRUCTURE_COMPLETE.md`
- **Project Overview**: `docs/PROJECT_CONSOLIDATION_SUMMARY.md`

### Example Code

- `examples/demo_custom_geometric_rule.py` - Creating custom rules
- `examples/demo_hierarchical_rules.py` - Hierarchical classification
- `examples/demo_confidence_scoring.py` - Confidence methods

### Related Modules

- **Thresholds**: `ign_lidar.core.classification.thresholds`
- **Building**: `ign_lidar.core.classification.building`
- **Transport**: `ign_lidar.core.classification.transport`

---

## Version History

- **v3.2.0** (2025-10-23) - Initial release
  - Complete rules infrastructure (Phase 4B)
  - 40+ public API exports
  - 7 confidence methods, 6 combination strategies
  - Hierarchical execution engine
  - Comprehensive validation utilities

---

## Support

For questions, issues, or contributions:

- **GitHub Issues**: [IGN_LIDAR_HD_DATASET/issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- **Documentation**: [Project Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- **Examples**: `examples/` directory

---

## License

MIT License - See [LICENSE](../LICENSE) file for details.
