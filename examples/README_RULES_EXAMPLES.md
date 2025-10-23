# Rules Framework Examples

**Phase 4B Infrastructure** - Custom rule-based classification examples

This directory contains practical examples demonstrating how to use the rules framework introduced in **Phase 4B** of the classification module consolidation.

---

## 📚 Available Examples

### 1. **Custom Geometric Rules** (`demo_custom_geometric_rule.py`)

**Purpose**: Learn how to create custom rule classes for point cloud classification

**Topics covered**:

- Creating custom rule classes by inheriting from `BaseRule`
- Implementing required methods (`evaluate`, `get_required_features`, etc.)
- Using confidence scoring utilities (`calculate_confidence_linear`, `calculate_confidence_threshold`)
- Feature validation (`validate_required_features`, `validate_point_cloud_shape`)
- Combining multiple rules for comprehensive classification

**Example rules**:

- `FlatSurfaceRule`: Detects flat surfaces (roads, parking lots) using planarity and roughness
- `VegetationHeightRule`: Classifies vegetation by height (low/medium/high) using NDVI

**Usage**:

```bash
cd examples
python demo_custom_geometric_rule.py
```

**Output**:

```
Demo 1: Single Rule Evaluation
  Points classified: 234 (23.4%)
  Mean confidence: 0.785

Demo 2: Multiple Rule Evaluation
  Combined Results: 867 points (43.4%)
  Classification breakdown by class
```

---

### 2. **Hierarchical Rule Engine** (`demo_hierarchical_rules.py`)

**Purpose**: Learn how to organize rules into multi-level hierarchies for complex classification

**Topics covered**:

- Creating hierarchical levels with `RuleLevel`
- Different execution strategies (`first_match`, `priority`, `weighted`)
- Conflict resolution between overlapping rules
- Performance tracking per level
- Sequential classification (coarse → fine)

**Example architecture**:

```
Level 1 (Ground Detection)
  └─ GroundRule (priority: CRITICAL)

Level 2 (Primary Classification)
  ├─ BuildingRule (priority: HIGH)
  └─ VegetationRule (priority: HIGH)

Level 3 (Refinement)
  └─ RefinementRule (priority: NORMAL)
```

**Usage**:

```bash
cd examples
python demo_hierarchical_rules.py
```

**Output**:

```
Demo 1: First Match Strategy
  Level 1 (ground_detection): 1,234 points
  Level 2 (primary): 2,456 points
  Level 3 (refinement): 123 points

Demo 2: Weighted Strategy
  Confidence weighted combination across rules

Demo 3: Strategy Comparison
  Comparing first_match vs priority vs all_matches
```

---

### 3. **Confidence Scoring** (`demo_confidence_scoring.py`)

**Purpose**: Master confidence calculation and combination strategies

**Topics covered**:

- **7 confidence calculation methods**:
  - Binary: Simple threshold-based (pass/fail)
  - Linear: Gradual increase over range
  - Sigmoid: Smooth S-curve transitions
  - Gaussian: Bell curve around target value
  - Threshold: Distance from threshold
  - Exponential: Rapid growth for strong signals
  - Composite: Combine multiple methods
- **6 confidence combination strategies**:
  - Weighted average: Different importance per rule
  - Maximum: Optimistic (accept if any confident)
  - Minimum: Conservative (require all confident)
  - Product: Independent evidence multiplication
  - Geometric mean: Balanced combination
  - Harmonic mean: Emphasizes lower values
- **Calibration and normalization**:
  - Normalize scores to [0, 1] range
  - Calibrate to target distribution

**Usage**:

```bash
cd examples
python demo_confidence_scoring.py
```

**Output**:

```
Demo 1: Confidence Calculation Methods
  Binary: Below 0.5: 0.00, Above 0.5: 1.00
  Linear: At 0.2: 0.00, At 0.5: 0.50, At 0.8: 1.00
  Sigmoid: At 0.3: 0.02, At 0.5: 0.50, At 0.7: 0.98
  ... (all 7 methods with examples)

Demo 2: Confidence Combination
  Weighted Average: [0.78, 0.67, 0.83, ...]
  Maximum: [0.90, 0.80, 0.90, ...]
  Minimum: [0.70, 0.50, 0.60, ...]
  ... (all 6 strategies)

Demo 3: Calibration
  Before: mean=0.712, std=0.142
  After:  mean=0.700, std=0.150 (target matched)

Demo 4: Practical Example - Building Detection
  10 points evaluated with 4 criteria
  7/10 classified as buildings with confidence >0.6
```

---

## 🚀 Quick Start

### Prerequisites

The examples require the rules infrastructure from Phase 4B:

```python
from ign_lidar.core.classification.rules import (
    BaseRule,              # Abstract base class for rules
    RuleEngine,            # Basic rule engine
    HierarchicalRuleEngine,  # Multi-level engine
    RuleResult,            # Result container
    RuleConfig,            # Rule configuration
    RuleLevel,             # Level definition
    # Enums
    RuleType,              # GEOMETRIC, SPECTRAL, CONTEXTUAL, etc.
    RulePriority,          # CRITICAL, HIGH, NORMAL, LOW
    ExecutionStrategy,     # FIRST_MATCH, PRIORITY, WEIGHTED, etc.
    ConflictResolution,    # HIGHEST_CONFIDENCE, WEIGHTED_AVERAGE, etc.
    # Confidence utilities
    calculate_confidence_linear,
    calculate_confidence_sigmoid,
    combine_confidence_weighted,
    # Validation utilities
    validate_required_features,
    validate_point_cloud_shape,
)
```

### Installation

```bash
# Install package in development mode
cd IGN_LIDAR_HD_DATASET
pip install -e .

# Or install from repository
pip install git+https://github.com/sducournau/IGN_LIDAR_HD_DATASET.git
```

### Run All Examples

```bash
# Navigate to examples directory
cd examples

# Run each demo
python demo_custom_geometric_rule.py
python demo_hierarchical_rules.py
python demo_confidence_scoring.py
```

---

## 📖 Learning Path

**Recommended order for learning the framework**:

1. **Start with confidence scoring** (`demo_confidence_scoring.py`)

   - Understand how confidence is calculated and combined
   - Learn when to use each method
   - See practical examples

2. **Create custom rules** (`demo_custom_geometric_rule.py`)

   - Learn the `BaseRule` interface
   - Implement simple rules
   - Combine multiple rules

3. **Build hierarchical systems** (`demo_hierarchical_rules.py`)
   - Organize rules into levels
   - Use different execution strategies
   - Handle conflicts between rules

---

## 🎯 Use Cases

### Use Case 1: Simple Threshold-Based Classification

**Goal**: Classify points as ground if height < 0.2m

```python
class SimpleGroundRule(BaseRule):
    def evaluate(self, points, features, **kwargs):
        height = features["height_above_ground"]
        ground_mask = height < 0.2

        confidence = calculate_confidence_binary(height, threshold=0.2, reverse=True)

        classifications = np.zeros(len(points), dtype=np.int32)
        classifications[ground_mask] = 2  # Ground class

        return RuleResult(
            rule_name="simple_ground",
            classifications=classifications,
            confidence=confidence,
            mask=ground_mask,
        )
```

### Use Case 2: Multi-Criteria Classification

**Goal**: Classify buildings using height, planarity, NDVI, and roughness

```python
class BuildingRule(BaseRule):
    def evaluate(self, points, features, **kwargs):
        # Calculate confidence for each criterion
        conf_height = calculate_confidence_gaussian(
            features["height"], mean=15.0, std=10.0
        )
        conf_planarity = calculate_confidence_linear(
            features["planarity"], min_value=0.7, max_value=1.0
        )
        conf_ndvi = calculate_confidence_threshold(
            features["ndvi"], threshold=0.3, reverse=True
        )
        conf_roughness = calculate_confidence_threshold(
            features["roughness"], threshold=0.1, reverse=True
        )

        # Combine with weights
        final_confidence = combine_confidence_weighted(
            [conf_height, conf_planarity, conf_ndvi, conf_roughness],
            weights=[0.3, 0.35, 0.2, 0.15]
        )

        # Apply threshold
        building_mask = final_confidence > 0.6

        classifications = np.zeros(len(points), dtype=np.int32)
        classifications[building_mask] = 6  # Building class

        return RuleResult(
            rule_name="building_detection",
            classifications=classifications,
            confidence=final_confidence,
            mask=building_mask,
        )
```

### Use Case 3: Hierarchical Classification Pipeline

**Goal**: Ground → Buildings/Vegetation → Refinement

```python
# Define levels
levels = [
    RuleLevel(
        name="ground_detection",
        priority=1,
        rules=[GroundRule()],
        strategy=ExecutionStrategy.FIRST_MATCH,
    ),
    RuleLevel(
        name="primary_objects",
        priority=2,
        rules=[BuildingRule(), VegetationRule()],
        strategy=ExecutionStrategy.PRIORITY,
    ),
    RuleLevel(
        name="refinement",
        priority=3,
        rules=[RefinementRule()],
        strategy=ExecutionStrategy.FIRST_MATCH,
    ),
]

# Create engine
engine = HierarchicalRuleEngine(
    levels=levels,
    conflict_resolution=ConflictResolution.HIGHEST_CONFIDENCE,
)

# Apply rules
result = engine.apply_rules(points, features)
```

---

## 🔧 Customization Tips

### Tip 1: Adjust Confidence Thresholds

```python
config = RuleConfig(
    name="my_rule",
    min_confidence=0.6,  # Lower = more permissive
    # min_confidence=0.8  # Higher = more conservative
)
```

### Tip 2: Use Optional Features

```python
def get_optional_features(self) -> Set[str]:
    return {"intensity", "return_number"}

def evaluate(self, points, features, **kwargs):
    # Check if optional feature available
    if "intensity" in features:
        # Use intensity to refine confidence
        intensity_factor = ...
        confidence *= intensity_factor
```

### Tip 3: Combine Multiple Confidence Methods

```python
# Use different methods for different aspects
conf_primary = calculate_confidence_sigmoid(
    value, midpoint=0.5, steepness=10
)
conf_secondary = calculate_confidence_linear(
    value, min_value=0.0, max_value=1.0
)

# Combine with weights
final_conf = combine_confidence_weighted(
    [conf_primary, conf_secondary],
    weights=[0.7, 0.3]
)
```

### Tip 4: Add Metadata for Debugging

```python
return RuleResult(
    rule_name=self.config.name,
    classifications=classifications,
    confidence=confidence,
    mask=mask,
    metadata={
        "n_classified": np.sum(mask),
        "mean_confidence": np.mean(confidence[mask]),
        "mean_planarity": np.mean(features["planarity"][mask]),
        "threshold_used": self.planarity_threshold,
        "processing_time": elapsed_time,
    },
)
```

---

## 📊 Performance Considerations

### Memory Efficiency

- Rules process full point clouds (N points)
- Confidence arrays are float32 (4 bytes per point)
- For 1M points: ~4MB per confidence array
- Use masking to reduce memory: `confidence[mask]` instead of full array

### Computational Efficiency

- Vectorized operations (NumPy) are fast
- Avoid loops over points
- Use boolean indexing: `points[mask]` instead of loops
- Pre-compute features once, reuse across rules

### Example: Efficient Rule Implementation

```python
# ✅ GOOD: Vectorized operations
ground_mask = (height < 0.2) & (planarity > 0.8)
confidence[ground_mask] = calculate_confidence(...)

# ❌ BAD: Loops over points
for i in range(len(points)):
    if height[i] < 0.2 and planarity[i] > 0.8:
        confidence[i] = ...  # Slow!
```

---

## 🐛 Troubleshooting

### Issue: ImportError for rules module

**Problem**: Cannot import from `ign_lidar.core.classification.rules`

**Solution**:

```bash
# Reinstall package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/IGN_LIDAR_HD_DATASET:$PYTHONPATH
```

### Issue: "Feature not found" error

**Problem**: Rule requires feature that isn't computed

**Solution**:

```python
# Check required features
print(rule.get_required_features())

# Ensure features are computed
features = {
    "height_above_ground": compute_height(points),
    "planarity": compute_planarity(points),
    "ndvi": compute_ndvi(points, colors),
}
```

### Issue: Low confidence scores

**Problem**: All confidence scores are near 0

**Solution**:

```python
# Check value ranges
print(f"Feature range: {features['planarity'].min():.3f} - {features['planarity'].max():.3f}")

# Adjust confidence calculation parameters
conf = calculate_confidence_linear(
    values,
    min_value=0.0,  # Adjust to actual data range
    max_value=1.0,
)
```

---

## 📚 Additional Resources

### Documentation

- **Rules Module API**: `ign_lidar/core/classification/rules/__init__.py`
- **Phase 4B Summary**: `docs/PHASE_4B_INFRASTRUCTURE_COMPLETE.md`
- **Phase 4A Analysis**: `docs/PHASE_4A_RULES_GRAMMAR_ANALYSIS.md`
- **Project Summary**: `docs/PROJECT_CONSOLIDATION_SUMMARY.md`

### Migration Guides

If migrating from older code:

- **Thresholds**: `docs/THRESHOLD_MIGRATION_GUIDE.md`
- **Building Module**: `docs/BUILDING_MODULE_MIGRATION_GUIDE.md`
- **Transport Module**: `docs/TRANSPORT_MODULE_MIGRATION_GUIDE.md`

### Other Examples

- `demo_adaptive_building_classification.py` - Adaptive building classification
- `demo_parcel_classification.py` - Parcel-based classification
- `demo_wall_detection.py` - Wall detection using geometric features
- `demo_variable_object_filtering.py` - Object filtering examples

---

## 🤝 Contributing

### Adding New Examples

1. Create a new demo file: `demo_your_feature.py`
2. Follow the existing structure:
   - Header docstring explaining purpose
   - Import statements
   - Example classes/functions
   - Multiple demo functions
   - Main function with clear output
3. Add entry to this README
4. Test with: `python demo_your_feature.py`

### Example Template

```python
#!/usr/bin/env python3
"""
Demo: Your Feature Name

Brief description of what this demo shows.

Features demonstrated:
- Feature 1
- Feature 2
- Feature 3
"""

import numpy as np
from ign_lidar.core.classification.rules import ...

# Your classes and functions here

def demo_feature_1():
    """Demonstrate feature 1."""
    print("Demo 1: Feature 1")
    # Implementation

def demo_feature_2():
    """Demonstrate feature 2."""
    print("Demo 2: Feature 2")
    # Implementation

def main():
    """Run all demonstrations."""
    print("Your Feature - Usage Examples")
    demo_feature_1()
    demo_feature_2()
    print("Demo Complete!")

if __name__ == "__main__":
    main()
```

---

## 📝 Version History

- **v3.2.0** (2025-10-23) - Initial release
  - Added `demo_custom_geometric_rule.py`
  - Added `demo_hierarchical_rules.py`
  - Added `demo_confidence_scoring.py`
  - Created README for rules examples

---

## 📧 Support

For questions, issues, or contributions related to the rules framework:

- **GitHub Issues**: [IGN_LIDAR_HD_DATASET/issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- **Documentation**: [Project Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- **Phase 4B Summary**: See `docs/PHASE_4B_INFRASTRUCTURE_COMPLETE.md`

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) file for details.
