---
sidebar_position: 6
title: Rules API Reference
---

# Rules Framework API Reference

Complete API documentation for the rule-based classification framework introduced in v3.2.0.

---

## Core Classes

### BaseRule

Abstract base class for all classification rules.

```python
from ign_lidar.core.classification.rules import BaseRule

class CustomRule(BaseRule):
    def __init__(self, name: str = "custom"):
        super().__init__(
            name=name,
            rule_type=RuleType.GEOMETRIC,
            priority=RulePriority.MEDIUM,
        )

    def evaluate(self, context: RuleContext) -> RuleResult:
        """Implement classification logic."""
        pass
```

**Methods:**

- `evaluate(context: RuleContext) -> RuleResult`: **Abstract** - Implement your classification logic
- `validate_context(context: RuleContext) -> bool`: Validate context has required features
- `get_required_features() -> List[str]`: Return list of required feature names

**Properties:**

- `name: str`: Unique rule identifier
- `rule_type: RuleType`: Type of rule (GEOMETRIC, SPECTRAL, CONTEXTUAL, etc.)
- `priority: RulePriority`: Execution priority (LOW, MEDIUM, HIGH, CRITICAL)

---

### RuleEngine

Execute rules with conflict resolution.

```python
from ign_lidar.core.classification.rules import RuleEngine

engine = RuleEngine(
    conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE,
    min_confidence=0.5,
)

engine.add_rule(rule1)
engine.add_rule(rule2)

result = engine.execute(points, labels, features)
```

**Constructor Parameters:**

- `conflict_resolution: ConflictResolution = HIGHEST_CONFIDENCE`: How to resolve conflicts
  - `FIRST`: Use first classification
  - `LAST`: Use last classification
  - `HIGHEST_CONFIDENCE`: Use classification with highest confidence
  - `WEIGHTED_AVERAGE`: Weight by confidence (for numeric classes)
  - `MAJORITY_VOTE`: Most common classification
- `min_confidence: float = 0.0`: Minimum confidence threshold (0-1)

**Methods:**

- `add_rule(rule: BaseRule)`: Add rule to engine
- `remove_rule(name: str)`: Remove rule by name
- `clear_rules()`: Remove all rules
- `execute(points, labels, additional_features) -> RuleResult`: Execute all rules

---

### HierarchicalRuleEngine

Multi-level rule execution with strategies.

```python
from ign_lidar.core.classification.rules import (
    HierarchicalRuleEngine,
    RuleLevel,
    ExecutionStrategy,
)

engine = HierarchicalRuleEngine()

# Add levels
level1 = RuleLevel(
    name="coarse",
    rules=[...],
    strategy=ExecutionStrategy.PRIORITY,
)
engine.add_level(level1)

result = engine.execute(points, labels, features)
```

**Methods:**

- `add_level(level: RuleLevel)`: Add classification level
- `remove_level(name: str)`: Remove level by name
- `clear_levels()`: Remove all levels
- `execute(points, labels, additional_features) -> RuleResult`: Execute hierarchically

---

### RuleLevel

Container for rules at one hierarchical level.

```python
level = RuleLevel(
    name="building_detection",
    rules=[BuildingRule(), RoofRule()],
    strategy=ExecutionStrategy.WEIGHTED,
    description="Detect building structures",
)
```

**Parameters:**

- `name: str`: Level identifier
- `rules: List[BaseRule]`: Rules to execute
- `strategy: ExecutionStrategy`: How to combine results
- `description: str = ""`: Optional description

**Execution Strategies:**

- `FIRST_MATCH`: Stop at first rule that classifies each point
- `ALL_MATCHES`: Execute all rules
- `PRIORITY`: Execute in priority order, stop on classification
- `WEIGHTED`: Combine results weighted by confidence

---

## Data Classes

### RuleContext

Input context for rule evaluation.

```python
@dataclass
class RuleContext:
    points: np.ndarray              # [N, 3] XYZ coordinates
    labels: np.ndarray              # [N] current classifications
    additional_features: Dict[str, np.ndarray]  # Feature arrays
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

### RuleResult

Result from rule execution.

```python
@dataclass
class RuleResult:
    point_indices: np.ndarray       # [M] indices of classified points
    classifications: np.ndarray     # [M] class assignments
    confidence_scores: np.ndarray   # [M] confidence values (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    rule_stats: Dict[str, RuleStats] = field(default_factory=dict)
```

---

### RuleStats

Performance statistics for a rule.

```python
@dataclass
class RuleStats:
    execution_time: float           # Seconds
    points_classified: int          # Number of points
    confidence_mean: float          # Average confidence
    confidence_std: float           # Confidence std dev
    confidence_min: float           # Minimum confidence
    confidence_max: float           # Maximum confidence
```

---

## Enums

### RuleType

```python
class RuleType(Enum):
    GEOMETRIC = "geometric"         # Geometry-based (height, planarity, etc.)
    SPECTRAL = "spectral"          # Color/spectral (RGB, NIR, NDVI)
    CONTEXTUAL = "contextual"      # Neighborhood context
    GROUND_TRUTH = "ground_truth"  # External data (BD TOPO, etc.)
    HYBRID = "hybrid"              # Multiple types
```

---

### RulePriority

```python
class RulePriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
```

---

### ConflictResolution

```python
class ConflictResolution(Enum):
    FIRST = "first"
    LAST = "last"
    HIGHEST_CONFIDENCE = "highest_confidence"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
```

---

### ExecutionStrategy

```python
class ExecutionStrategy(Enum):
    FIRST_MATCH = "first_match"
    ALL_MATCHES = "all_matches"
    PRIORITY = "priority"
    WEIGHTED = "weighted"
```

---

## Confidence Calculation Functions

### calculate_binary_confidence

```python
def calculate_binary_confidence(
    values: np.ndarray,
    threshold: float,
    above: bool = True,
) -> np.ndarray:
    """
    Binary confidence (0 or 1) based on threshold.

    Args:
        values: Feature values
        threshold: Threshold value
        above: True for values > threshold, False for values < threshold

    Returns:
        Confidence array (0 or 1)
    """
```

---

### calculate_linear_confidence

```python
def calculate_linear_confidence(
    values: np.ndarray,
    min_value: float,
    max_value: float,
    invert: bool = False,
) -> np.ndarray:
    """
    Linear scaling between min and max.

    Args:
        values: Feature values
        min_value: Minimum value (maps to 0)
        max_value: Maximum value (maps to 1)
        invert: If True, invert the scaling

    Returns:
        Confidence array [0, 1]
    """
```

---

### calculate_sigmoid_confidence

```python
def calculate_sigmoid_confidence(
    values: np.ndarray,
    midpoint: float,
    steepness: float = 1.0,
) -> np.ndarray:
    """
    Sigmoid (S-curve) confidence.

    Args:
        values: Feature values
        midpoint: Center of sigmoid (0.5 confidence)
        steepness: Curve steepness (higher = sharper transition)

    Returns:
        Confidence array [0, 1]
    """
```

---

### calculate_gaussian_confidence

```python
def calculate_gaussian_confidence(
    values: np.ndarray,
    target: float,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian (bell curve) confidence around target.

    Args:
        values: Feature values
        target: Target value (peak confidence)
        sigma: Standard deviation (spread)

    Returns:
        Confidence array [0, 1]
    """
```

---

### calculate_threshold_confidence

```python
def calculate_threshold_confidence(
    values: np.ndarray,
    thresholds: List[float],
    confidences: List[float],
) -> np.ndarray:
    """
    Stepped confidence based on thresholds.

    Args:
        values: Feature values
        thresholds: Threshold values (sorted)
        confidences: Confidence for each range (len = len(thresholds) + 1)

    Returns:
        Confidence array

    Example:
        thresholds = [2.0, 5.0, 10.0]
        confidences = [0.25, 0.5, 0.75, 1.0]
        # values < 2.0: 0.25
        # 2.0 <= values < 5.0: 0.5
        # 5.0 <= values < 10.0: 0.75
        # values >= 10.0: 1.0
    """
```

---

### calculate_exponential_confidence

```python
def calculate_exponential_confidence(
    values: np.ndarray,
    rate: float = 1.0,
    increasing: bool = True,
) -> np.ndarray:
    """
    Exponential growth or decay confidence.

    Args:
        values: Feature values
        rate: Exponential rate
        increasing: True for growth, False for decay

    Returns:
        Confidence array [0, 1]
    """
```

---

### calculate_composite_confidence

```python
def calculate_composite_confidence(
    feature_dict: Dict[str, np.ndarray],
    weights: Dict[str, float],
    methods: Dict[str, ConfidenceMethod],
    method_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Combine multiple features with different methods.

    Args:
        feature_dict: Feature name -> array mapping
        weights: Feature name -> weight mapping (should sum to 1)
        methods: Feature name -> ConfidenceMethod mapping
        method_params: Feature name -> method parameters

    Returns:
        Combined confidence array [0, 1]

    Example:
        confidence = calculate_composite_confidence(
            feature_dict={
                'height': height_array,
                'planarity': planarity_array,
            },
            weights={
                'height': 0.6,
                'planarity': 0.4,
            },
            methods={
                'height': ConfidenceMethod.LINEAR,
                'planarity': ConfidenceMethod.GAUSSIAN,
            },
            method_params={
                'height': {'min_value': 0, 'max_value': 20},
                'planarity': {'target': 1.0, 'sigma': 0.2},
            },
        )
    """
```

---

## Validation Functions

### validate_features

```python
def validate_features(
    features: Dict[str, np.ndarray],
    requirements: FeatureRequirements,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate feature dictionary against requirements.

    Args:
        features: Feature name -> array mapping
        requirements: Feature requirements

    Returns:
        (is_valid, missing_features, invalid_shapes)
    """
```

---

### check_feature_quality

```python
def check_feature_quality(
    feature_array: np.ndarray,
    feature_name: str,
    expected_range: Optional[Tuple[float, float]] = None,
    max_nan_ratio: float = 0.1,
) -> Tuple[bool, str]:
    """
    Check feature array quality.

    Args:
        feature_array: Feature values
        feature_name: Feature name (for error messages)
        expected_range: (min, max) expected values
        max_nan_ratio: Maximum allowed NaN ratio

    Returns:
        (is_valid, error_message)
    """
```

---

## Configuration Classes

### FeatureRequirements

```python
@dataclass
class FeatureRequirements:
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
```

---

## Complete Usage Example

```python
from ign_lidar.core.classification.rules import *

# Define custom rule
class VegetationRule(BaseRule):
    def __init__(self):
        super().__init__(
            name="vegetation_ndvi",
            rule_type=RuleType.SPECTRAL,
            priority=RulePriority.HIGH,
        )

    def evaluate(self, context: RuleContext) -> RuleResult:
        ndvi = context.additional_features['ndvi']
        height = context.additional_features['height']

        # Low vegetation: NDVI > 0.3, height < 2m
        mask_low = (ndvi > 0.3) & (height < 2.0)

        # High vegetation: NDVI > 0.3, height >= 2m
        mask_high = (ndvi > 0.3) & (height >= 2.0)

        # Combine
        indices = np.concatenate([
            np.where(mask_low)[0],
            np.where(mask_high)[0],
        ])

        classes = np.concatenate([
            np.full(mask_low.sum(), 3),   # Low veg
            np.full(mask_high.sum(), 4),  # Med veg
        ])

        # Confidence based on NDVI strength
        confidence = calculate_linear_confidence(
            ndvi[indices],
            min_value=0.3,
            max_value=0.8,
        )

        return RuleResult(
            point_indices=indices,
            classifications=classes,
            confidence_scores=confidence,
        )

# Create engine
engine = RuleEngine(
    conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE,
    min_confidence=0.5,
)

# Add rules
engine.add_rule(VegetationRule())
engine.add_rule(BuildingRule())
engine.add_rule(GroundRule())

# Execute
result = engine.execute(
    points=points,
    labels=np.zeros(len(points)),
    additional_features={'ndvi': ndvi, 'height': height},
)

# Apply results
final_labels = labels.copy()
final_labels[result.point_indices] = result.classifications

print(f"Classified {len(result.point_indices)} points")
print(f"Mean confidence: {result.confidence_scores.mean():.2f}")
```

---

## See Also

- [Rules Framework Guide](../features/rules-framework.md)
- [Quick Reference](../../RULES_FRAMEWORK_QUICK_REFERENCE.md)
- [Developer Guide](../../RULES_FRAMEWORK_DEVELOPER_GUIDE.md)
- [Examples](../../examples/)

---

**API Version:** 3.2.0+  
**Last Updated:** October 25, 2025
