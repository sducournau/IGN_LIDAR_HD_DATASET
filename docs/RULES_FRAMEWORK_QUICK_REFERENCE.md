# Rules Framework - Quick Reference Card

> **One-page reference for the IGN LiDAR HD Rules Framework (v3.2.0)**  
> For detailed documentation, see: `RULES_FRAMEWORK_DEVELOPER_GUIDE.md`

---

## 📦 Quick Start

```python
from ign_lidar.core.classification.rules import (
    BaseRule, RuleEngine, HierarchicalRuleEngine,
    RuleContext, RuleResult, RulePriority
)
import numpy as np

# 1. Define a custom rule
class MyRule(BaseRule):
    def evaluate(self, context: RuleContext) -> RuleResult:
        # Your classification logic
        mask = context.points[:, 2] > 5.0  # Height > 5m
        confidence = np.ones(mask.sum()) * 0.9
        return RuleResult(
            point_indices=np.where(mask)[0],
            classifications=np.full(mask.sum(), 6),  # Building
            confidence_scores=confidence
        )

# 2. Create engine and run
engine = RuleEngine()
engine.add_rule(MyRule(name="height_filter", priority=RulePriority.HIGH))
result = engine.execute(points=points, labels=labels)
```

---

## 🎯 Core Classes

### **BaseRule** - Abstract base for all rules

```python
class MyRule(BaseRule):
    def validate_features(self, context: RuleContext) -> bool:
        return 'height' in context.additional_features

    def evaluate(self, context: RuleContext) -> RuleResult:
        # Return RuleResult with indices, classes, confidences
        pass

    def get_required_features(self) -> List[str]:
        return ['height', 'intensity']
```

### **RuleEngine** - Execute rules sequentially

```python
engine = RuleEngine(
    execution_strategy=ExecutionStrategy.SEQUENTIAL,
    conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE
)
engine.add_rule(rule1, priority=RulePriority.HIGH)
engine.add_rule(rule2, priority=RulePriority.MEDIUM)
result = engine.execute(points, labels)
```

### **HierarchicalRuleEngine** - Multi-level classification

```python
from ign_lidar.core.classification.rules import RuleLevel

engine = HierarchicalRuleEngine()
engine.add_level(RuleLevel(
    name="coarse",
    rules=[ground_rule, non_ground_rule],
    priority=1
))
engine.add_level(RuleLevel(
    name="fine",
    rules=[building_rule, vegetation_rule],
    priority=2
))
result = engine.execute_hierarchical(points, labels)
```

---

## 📊 Data Structures

### **RuleContext** - Input to rule evaluation

```python
@dataclass
class RuleContext:
    points: np.ndarray              # (N, 3+) point coordinates
    labels: np.ndarray              # (N,) current classifications
    additional_features: Dict       # Custom features
    metadata: Dict                  # Processing metadata
```

### **RuleResult** - Output from rule evaluation

```python
@dataclass
class RuleResult:
    point_indices: np.ndarray       # Which points to reclassify
    classifications: np.ndarray     # New class labels
    confidence_scores: np.ndarray   # Confidence [0-1]
    metadata: Dict                  # Optional statistics
```

### **ExecutionResult** - Final engine output

```python
result = engine.execute(points, labels)
print(f"Changed: {result.points_modified}")
print(f"Rules applied: {result.rules_applied}")
print(f"Time: {result.execution_time:.2f}s")
labels = result.updated_labels  # Updated classifications
```

---

## 🔢 Confidence Methods

```python
from ign_lidar.core.classification.rules.confidence import (
    calculate_confidence, ConfidenceMethod
)

# 1. Binary (0.0 or 1.0)
confidence = calculate_confidence(
    feature_values=height,
    method=ConfidenceMethod.BINARY,
    threshold=5.0
)

# 2. Linear ramp
confidence = calculate_confidence(
    feature_values=height,
    method=ConfidenceMethod.LINEAR,
    threshold=5.0,
    range_width=2.0  # 5.0 to 7.0
)

# 3. Sigmoid (smooth)
confidence = calculate_confidence(
    feature_values=height,
    method=ConfidenceMethod.SIGMOID,
    threshold=5.0,
    steepness=2.0
)

# 4. Gaussian (peak-based)
confidence = calculate_confidence(
    feature_values=height,
    method=ConfidenceMethod.GAUSSIAN,
    threshold=5.0,
    sigma=1.0
)

# 5. Threshold range
confidence = calculate_confidence(
    feature_values=height,
    method=ConfidenceMethod.THRESHOLD_RANGE,
    threshold_min=3.0,
    threshold_max=10.0
)

# 6. Exponential decay
confidence = calculate_confidence(
    feature_values=distance,
    method=ConfidenceMethod.EXPONENTIAL,
    decay_rate=0.5
)

# 7. Composite (multiple features)
confidence = calculate_confidence(
    feature_values=[height, intensity, ndvi],
    method=ConfidenceMethod.COMPOSITE,
    weights=[0.5, 0.3, 0.2],
    sub_methods=[ConfidenceMethod.LINEAR,
                 ConfidenceMethod.SIGMOID,
                 ConfidenceMethod.THRESHOLD_RANGE]
)
```

---

## 🔄 Confidence Combination

```python
from ign_lidar.core.classification.rules.confidence import combine_confidences

# Multiple rules classify the same point differently
confidences = [0.8, 0.9, 0.7]

# Strategy 1: Maximum (default)
final = combine_confidences(confidences, strategy='max')

# Strategy 2: Average
final = combine_confidences(confidences, strategy='average')

# Strategy 3: Weighted average
final = combine_confidences(
    confidences,
    strategy='weighted_average',
    weights=[0.5, 0.3, 0.2]
)

# Strategy 4: Minimum (conservative)
final = combine_confidences(confidences, strategy='min')

# Strategy 5: Product (all must agree)
final = combine_confidences(confidences, strategy='product')

# Strategy 6: Weighted product
final = combine_confidences(
    confidences,
    strategy='weighted_product',
    weights=[0.5, 0.3, 0.2]
)
```

---

## ✅ Feature Validation

```python
from ign_lidar.core.classification.rules.validation import (
    validate_rule_context,
    validate_rule_result,
    validate_feature_array,
    FeatureRequirements
)

# Validate context before rule execution
is_valid, errors = validate_rule_context(
    context,
    required_features=['height', 'intensity']
)
if not is_valid:
    print(f"Errors: {errors}")

# Validate result after rule execution
is_valid, errors = validate_rule_result(
    result,
    num_points=len(points)
)

# Validate individual feature
is_valid, error = validate_feature_array(
    feature_array=height,
    feature_name='height',
    expected_shape=(len(points),),
    allow_nan=False
)

# Define feature requirements
requirements = FeatureRequirements(
    required=['height', 'intensity'],
    optional=['rgb', 'ndvi'],
    min_points=100,
    allow_empty_labels=False
)
```

---

## 🎭 Enums & Constants

### **RuleType**

```python
RuleType.GEOMETRIC      # Height, planarity, etc.
RuleType.SPECTRAL       # RGB, intensity, NDVI
RuleType.CONTEXTUAL     # Neighborhood analysis
RuleType.HYBRID         # Multiple types
RuleType.CUSTOM         # User-defined
```

### **RulePriority**

```python
RulePriority.CRITICAL   # 100 - Execute first
RulePriority.HIGH       # 75
RulePriority.MEDIUM     # 50  (default)
RulePriority.LOW        # 25
RulePriority.OPTIONAL   # 10  - Execute last
```

### **ExecutionStrategy**

```python
ExecutionStrategy.SEQUENTIAL           # One rule at a time
ExecutionStrategy.PARALLEL             # Multiple rules
ExecutionStrategy.ADAPTIVE             # Auto-optimize
ExecutionStrategy.CONDITIONAL          # Rule dependencies
```

### **ConflictResolution**

```python
ConflictResolution.HIGHEST_CONFIDENCE  # Max confidence wins
ConflictResolution.FIRST_WINS          # Priority order
ConflictResolution.LAST_WINS           # Last rule applied
ConflictResolution.MAJORITY_VOTE       # Most common class
ConflictResolution.WEIGHTED_VOTE       # Weighted by confidence
```

---

## 📋 Common Patterns

### **Pattern 1: Height-based rule**

```python
class HeightRule(BaseRule):
    def __init__(self, min_height: float, class_label: int):
        super().__init__(name=f"height_{class_label}",
                        rule_type=RuleType.GEOMETRIC)
        self.min_height = min_height
        self.class_label = class_label

    def evaluate(self, context: RuleContext) -> RuleResult:
        height = context.additional_features['height']
        mask = height > self.min_height
        confidence = calculate_confidence(
            height[mask],
            method=ConfidenceMethod.LINEAR,
            threshold=self.min_height,
            range_width=2.0
        )
        return RuleResult(
            point_indices=np.where(mask)[0],
            classifications=np.full(mask.sum(), self.class_label),
            confidence_scores=confidence
        )
```

### **Pattern 2: Multi-feature rule**

```python
class VegetationRule(BaseRule):
    def evaluate(self, context: RuleContext) -> RuleResult:
        height = context.additional_features['height']
        ndvi = context.additional_features['ndvi']
        intensity = context.additional_features['intensity']

        # Combined criteria
        veg_mask = (height > 0.5) & (height < 30) & \
                   (ndvi > 0.3) & (intensity < 200)

        # Multi-feature confidence
        conf_height = calculate_confidence(
            height[veg_mask], ConfidenceMethod.GAUSSIAN,
            threshold=5.0, sigma=3.0
        )
        conf_ndvi = calculate_confidence(
            ndvi[veg_mask], ConfidenceMethod.LINEAR,
            threshold=0.3, range_width=0.3
        )

        # Combine confidences
        confidence = combine_confidences(
            [conf_height, conf_ndvi],
            strategy='average'
        )

        return RuleResult(
            point_indices=np.where(veg_mask)[0],
            classifications=np.full(veg_mask.sum(), 5),
            confidence_scores=confidence
        )
```

### **Pattern 3: Contextual rule with ground truth**

```python
class GroundTruthRule(BaseRule):
    def __init__(self, ground_truth_gdf, buffer_distance: float = 0.1):
        super().__init__(name="ground_truth",
                        rule_type=RuleType.CONTEXTUAL)
        self.gdf = ground_truth_gdf
        self.buffer = buffer_distance

    def evaluate(self, context: RuleContext) -> RuleResult:
        from shapely.geometry import Point
        from shapely.strtree import STRtree

        # Build spatial index
        tree = STRtree(self.gdf.geometry)

        indices = []
        classes = []
        confidences = []

        for i, point in enumerate(context.points):
            pt = Point(point[0], point[1])
            nearby = tree.query(pt.buffer(self.buffer))

            if len(nearby) > 0:
                # Use nearest ground truth feature
                nearest = nearby[0]
                indices.append(i)
                classes.append(self.gdf.iloc[nearest]['class'])
                confidences.append(0.95)  # High confidence

        return RuleResult(
            point_indices=np.array(indices),
            classifications=np.array(classes),
            confidence_scores=np.array(confidences)
        )
```

---

## 🚀 Performance Tips

1. **Use NumPy vectorization** - Avoid Python loops
2. **Validate once** - Check features before engine execution
3. **Set appropriate priorities** - Higher priority = earlier execution
4. **Use hierarchical engines** - Break complex tasks into levels
5. **Cache expensive computations** - Store in context.metadata
6. **Profile with statistics** - Use result.execution_time, result.rules_applied
7. **Batch similar operations** - Group geometric/spectral rules
8. **Use spatial indices** - STRtree for ground truth queries

---

## 📚 Examples

See `examples/` directory:

- `demo_custom_geometric_rule.py` - Basic height/planarity rules
- `demo_hierarchical_rules.py` - Multi-level classification
- `demo_confidence_scoring.py` - All confidence methods
- `README_RULES_EXAMPLES.md` - Comprehensive examples guide

---

## 🔗 API Reference

**Full documentation**: `docs/RULES_FRAMEWORK_DEVELOPER_GUIDE.md`

### Key modules:

- `rules.base` - BaseRule, RuleEngine, HierarchicalRuleEngine
- `rules.confidence` - Confidence calculation and combination
- `rules.validation` - Feature and result validation
- `rules.hierarchy` - Multi-level rule execution

### Import everything:

```python
from ign_lidar.core.classification.rules import *
```

---

## 🎓 Learning Path

1. **Start here**: Read Quick Start section (top of this page)
2. **Run examples**: `python examples/demo_custom_geometric_rule.py`
3. **Read guide**: `docs/RULES_FRAMEWORK_DEVELOPER_GUIDE.md#quick-start`
4. **Create simple rule**: Height or intensity based
5. **Add confidence**: Use `calculate_confidence()`
6. **Combine rules**: Use `RuleEngine`
7. **Go hierarchical**: Use `HierarchicalRuleEngine`
8. **Optimize**: Profile and improve performance

---

## 📞 Support

- **Documentation**: [sducournau.github.io/IGN_LIDAR_HD_DATASET](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- **Issues**: [github.com/sducournau/IGN_LIDAR_HD_DATASET/issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- **Examples**: `examples/README_RULES_EXAMPLES.md`
- **Developer Guide**: `docs/RULES_FRAMEWORK_DEVELOPER_GUIDE.md`

---

**Version**: 3.2.0 | **Date**: October 23, 2025 | **License**: MIT
