# Implementation Roadmap: Feature-Based Classification

**Date:** October 19, 2025  
**Status:** Ready for Implementation  
**Priority:** HIGH

---

## Quick Reference

**Objective:** Upgrade classification to use geometric + radiometric features with ground truth validation, removing BD Topo vegetation dependency.

**Timeline:** 3 weeks  
**Expected Impact:** +14% overall accuracy, +18% vegetation accuracy  
**Risk:** Low (builds on existing infrastructure)

---

## Week 1: Core Feature Validation (Days 1-5)

### Day 1-2: Create Feature Validator Module

**File:** `ign_lidar/core/modules/feature_validator.py` (NEW)

```python
"""
Feature-based ground truth validation module.

Validates ground truth classifications using geometric and radiometric features.
"""

class FeatureValidator:
    """Validate classifications using multi-feature signatures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with feature signature thresholds."""
        self.vegetation_signature = config['vegetation_signature']
        self.building_signature = config['building_signature']
        self.road_signature = config['road_signature']

    def validate_ground_truth(
        self,
        labels: np.ndarray,
        ground_truth_type: str,
        features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate ground truth using feature signatures.

        Returns: (validated_labels, confidence_scores)
        """
        pass  # Implement as per CLASSIFICATION_VEGETATION_AUDIT_2025.md

    def check_vegetation_signature(self, features: Dict) -> Tuple[bool, float]:
        """Check if features match vegetation signature."""
        pass

    def check_building_signature(self, features: Dict) -> Tuple[bool, float]:
        """Check if features match building signature."""
        pass

    def check_road_signature(self, features: Dict) -> Tuple[bool, float]:
        """Check if features match road signature."""
        pass
```

**Testing:**

- Unit test each signature checker
- Test with synthetic feature data
- Validate confidence scoring

### Day 3-4: Update Advanced Classification Module

**File:** `ign_lidar/core/modules/advanced_classification.py`

**Changes Required:**

1. **Remove BD Topo vegetation from priority order** (line ~438):

```python
# OLD (line 438):
priority_order = [
    ('vegetation', self.ASPRS_MEDIUM_VEGETATION),  # ← REMOVE
    ('water', self.ASPRS_WATER),
    ...
]

# NEW:
priority_order = [
    # ('vegetation', self.ASPRS_MEDIUM_VEGETATION),  # ← REMOVED
    ('water', self.ASPRS_WATER),
    ('cemeteries', self.ASPRS_CEMETERY),
    ...
]
```

2. **Add feature-based vegetation classification** (new method):

```python
def _classify_vegetation_feature_aware(
    self,
    labels: np.ndarray,
    confidence: np.ndarray,
    ndvi: np.ndarray,
    height: np.ndarray,
    features: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify vegetation using NDVI + geometric features.

    Validates NDVI-based classification with curvature and planarity.
    """
    # Extract features
    curvature = features.get('curvature')
    planarity = features.get('planarity')
    nir = features.get('nir')

    # Multi-level NDVI with feature validation
    for i in range(len(labels)):
        if ndvi[i] >= 0.6:
            # Dense forest - validate with features
            if curvature[i] > 0.3 and planarity[i] < 0.5:
                labels[i] = self.ASPRS_HIGH_VEGETATION
                confidence[i] = 0.95
        elif ndvi[i] >= 0.5:
            # Healthy trees
            if curvature[i] > 0.25 and planarity[i] < 0.6:
                labels[i] = self.ASPRS_HIGH_VEGETATION if height[i] > 2.0 else self.ASPRS_MEDIUM_VEGETATION
                confidence[i] = 0.85
        # ... continue for other NDVI levels

    return labels, confidence
```

3. **Integrate feature validator in ground truth classification**:

```python
def _classify_by_ground_truth(self, ...):
    # ... existing code ...

    # NEW: Validate with features
    if self.use_feature_validation:
        labels, confidence = self.feature_validator.validate_ground_truth(
            labels=labels,
            ground_truth_types=ground_truth_types,
            features=all_features
        )

    return labels
```

**Testing:**

- Test on Versailles sample tile
- Compare accuracy before/after
- Verify vegetation classification without BD Topo

### Day 5: Update Geometric Rules Module

**File:** `ign_lidar/core/modules/geometric_rules.py`

**Changes Required:**

1. **Update NDVI thresholds** (add multi-level):

```python
# Add class constants
NDVI_DENSE_FOREST = 0.60
NDVI_HEALTHY_TREES = 0.50
NDVI_MODERATE_VEG = 0.40
NDVI_GRASS = 0.30
NDVI_SPARSE_VEG = 0.20
NDVI_NON_VEG = 0.15
```

2. **Enhance apply_ndvi_refinement()** to use features:

```python
def apply_ndvi_refinement(
    self,
    points: np.ndarray,
    labels: np.ndarray,
    ndvi: np.ndarray,
    curvature: Optional[np.ndarray] = None,
    planarity: Optional[np.ndarray] = None
) -> int:
    """Apply NDVI refinement WITH feature validation."""
    # ... existing code ...

    # NEW: Validate high NDVI with geometric features
    if curvature is not None and planarity is not None:
        high_ndvi_mask = (ndvi >= self.ndvi_vegetation_threshold)
        # Only classify as vegetation if features confirm
        is_vegetation = high_ndvi_mask & (curvature > 0.15) & (planarity < 0.7)
        # ... apply classification
```

**Testing:**

- Test NDVI refinement with/without features
- Validate false positive reduction

---

## Week 2: Feature Integration & Testing (Days 6-10)

### Day 6-7: Multi-Feature Decision Fusion

**File:** `ign_lidar/core/modules/multi_feature_classifier.py` (NEW)

```python
"""
Multi-feature decision fusion for classification.
"""

class MultiFeatureClassifier:
    """Combine evidence from multiple feature sources."""

    def classify_point_fusion(
        self,
        geometric_features: Dict[str, float],
        radiometric_features: Dict[str, float],
        ground_truth_context: Dict[str, bool]
    ) -> Tuple[int, float]:
        """
        Classify a single point using feature fusion.

        Returns: (class_id, confidence)
        """
        # Implement decision tree from CLASSIFICATION_VEGETATION_AUDIT_2025.md
        pass

    def classify_batch_fusion(
        self,
        geometric_features: Dict[str, np.ndarray],
        radiometric_features: Dict[str, np.ndarray],
        ground_truth_context: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized classification for efficiency.

        Returns: (labels, confidence_scores)
        """
        pass
```

**Implementation:**

- Copy decision tree logic from audit document
- Vectorize for performance
- Add feature weighting

### Day 8-9: Integration Testing

**Test Suites:**

1. **Unit Tests** (`tests/test_feature_validation.py`):

```python
def test_vegetation_signature_validation():
    """Test vegetation feature signature detection."""
    features = {
        'curvature': np.array([0.35, 0.05]),
        'planarity': np.array([0.45, 0.90]),
        'ndvi': np.array([0.55, 0.10])
    }
    validator = FeatureValidator(config)
    is_veg, conf = validator.check_vegetation_signature(features)
    assert is_veg[0] == True   # First point is vegetation
    assert is_veg[1] == False  # Second point is not vegetation

def test_building_false_positive_filtering():
    """Test roof vegetation detection."""
    # Point on building with high NDVI (roof vegetation)
    features = {
        'curvature': 0.30,
        'planarity': 0.50,
        'ndvi': 0.65,
        'height': 5.0
    }
    label, conf = validate_ground_truth(
        ground_truth_label=ASPRS_BUILDING,
        ground_truth_type='building',
        features=features
    )
    assert label == ASPRS_MEDIUM_VEGETATION  # Overridden to vegetation
    assert conf > 0.80
```

2. **Integration Tests** (Versailles tiles):

```bash
# Run classification with new feature-first approach
python -m ign_lidar.cli.process \
    --config examples/config_versailles_asprs_v5.0_feature_first.yaml \
    --tile 0646_6862 \
    --output test_output/

# Compare with old approach
python scripts/compare_classifications.py \
    --old test_output/old/0646_6862.laz \
    --new test_output/new/0646_6862.laz \
    --metrics confusion_matrix,accuracy,f1_score
```

### Day 10: Performance Benchmarking

**Benchmark Script:** `scripts/benchmark_feature_classification.py`

```python
import time
from ign_lidar.core.modules import AdvancedClassifier

# Load test tile
points, features = load_test_tile("0646_6862")

# Benchmark old approach
start = time.time()
labels_old = classify_old_approach(points, features)
time_old = time.time() - start

# Benchmark new approach
start = time.time()
labels_new = classify_new_approach(points, features)
time_new = time.time() - start

# Compare results
print(f"Old approach: {time_old:.2f}s")
print(f"New approach: {time_new:.2f}s")
print(f"Speedup: {time_old/time_new:.2f}x")
print(f"Accuracy improvement: {compute_accuracy_improvement(labels_old, labels_new):.1f}%")
```

---

## Week 3: Optimization & Deployment (Days 11-15)

### Day 11-12: Configuration Updates

**File:** `examples/config_versailles_asprs_v5.0_feature_first.yaml` (NEW)

```yaml
# Feature-first classification configuration
version: "5.0"
strategy: "asprs"

# Advanced classification with feature validation
advanced_classification:
  enabled: true
  strategy: "feature_first"

  # Feature-based classification
  use_multi_feature_fusion: true
  use_feature_validation: true

  # Feature weights
  feature_weights:
    geometric: 0.5
    radiometric: 0.3
    ground_truth: 0.2

  # Ground truth configuration
  use_bd_topo_vegetation: false # ← DISABLED
  validate_ground_truth: true
  ground_truth_confidence_threshold: 0.6

  # Feature signatures for validation
  vegetation_signature:
    curvature_min: 0.15
    planarity_max: 0.70
    ndvi_min: 0.20
    nir_min: 0.25

  building_signature:
    curvature_max: 0.10
    planarity_min: 0.70
    ndvi_max: 0.15
    verticality_min: 0.60

  road_signature:
    curvature_max: 0.05
    planarity_min: 0.85
    ndvi_max: 0.15
    normal_z_min: 0.90
    height_max: 2.0

  # Multi-level NDVI thresholds
  ndvi_thresholds:
    dense_forest: 0.60
    healthy_trees: 0.50
    moderate_veg: 0.40
    grass: 0.30
    sparse_veg: 0.20
    non_veg: 0.15

# Data sources
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
      vegetation: false # ← DISABLED
      railways: true
      bridges: true
      parking: true
      sports: true
```

### Day 13: Documentation

**Update Documentation:**

1. **User Guide** (`docs/docs/guides/feature-classification.md`):

   - Explain feature-first approach
   - Configuration examples
   - Best practices

2. **API Reference** (`docs/docs/reference/feature-validation-api.md`):

   - FeatureValidator class
   - MultiFeatureClassifier class
   - Configuration parameters

3. **Migration Guide** (`docs/docs/guides/migration-v4-to-v5.md`):
   - How to upgrade configs
   - Breaking changes
   - Backward compatibility

### Day 14: Full Pipeline Testing

**Test Plan:**

1. **Accuracy Validation:**

   - Process 10 Versailles tiles
   - Compare with manual labels
   - Measure accuracy improvements

2. **Performance Validation:**

   - Measure processing time
   - Check memory usage
   - Verify GPU compatibility

3. **Edge Cases:**
   - Roof vegetation detection
   - Tree canopies over roads
   - Mixed vegetation types
   - Building shadows

### Day 15: Deployment Preparation

**Checklist:**

- [ ] All unit tests passing
- [ ] Integration tests validated
- [ ] Performance benchmarks documented
- [ ] Documentation complete
- [ ] Configuration examples tested
- [ ] Migration guide reviewed
- [ ] Backward compatibility verified
- [ ] GPU support tested
- [ ] Code review completed
- [ ] Release notes prepared

---

## Success Metrics

### Quantitative Metrics

| Metric              | Target     | Measurement Method                |
| ------------------- | ---------- | --------------------------------- |
| Overall Accuracy    | +10-15%    | Confusion matrix vs manual labels |
| Vegetation Accuracy | +15-20%    | F1-score for vegetation classes   |
| Building Accuracy   | +5-10%     | Precision/recall for buildings    |
| False Positive Rate | -50%       | Compare with ground truth         |
| Processing Speed    | +10-20%    | Time per tile                     |
| Memory Usage        | ≤ Baseline | Peak memory measurement           |

### Qualitative Metrics

- [ ] Roof vegetation correctly detected
- [ ] Tree canopies over roads correctly classified
- [ ] No regression in building/road detection
- [ ] Feature validation prevents obvious errors
- [ ] Confidence scores correlate with accuracy
- [ ] Configuration is intuitive

---

## Rollback Plan

If issues arise:

1. **Immediate rollback:**

   ```bash
   git revert <commit_hash>
   ```

2. **Configuration fallback:**

   ```yaml
   advanced_classification:
     strategy: "ground_truth_first" # Revert to old behavior
     use_bd_topo_vegetation: true
   ```

3. **Feature flag:**
   ```python
   if config.get('use_feature_validation', False):
       # New approach
   else:
       # Old approach (fallback)
   ```

---

## Contact & Support

**Implementation Team:**

- Lead Developer: [Assign]
- Feature Engineering: [Assign]
- Testing: [Assign]
- Documentation: [Assign]

**Questions?**

- See: `CLASSIFICATION_VEGETATION_AUDIT_2025.md`
- See: `CLASSIFICATION_OPTIMIZATION_SUMMARY.md`
- GitHub Issues: Tag with `feature-classification`

---

**Document Status:** Implementation Guide  
**Last Updated:** October 19, 2025  
**Next Review:** After Week 1 completion
