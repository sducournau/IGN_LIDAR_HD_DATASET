# Comprehensive Adaptive Classification Guide

## Overview

This guide describes the **comprehensive adaptive classification system** that treats ground truth as **guidance rather than absolute truth** for ALL feature types:

- **Buildings**: Fuzzy boundaries, geometry-driven classification
- **Vegetation**: Multi-feature confidence voting
- **Roads**: Tree canopy detection, bridge identification
- **Water**: Extreme flatness validation with geometric features

## Core Philosophy

### Traditional Approach (Problem)

```
Ground Truth → Absolute Boundaries → Classification
```

**Issues:**

- Building polygons don't match point cloud (misaligned, wrong dimensions)
- Road polygons include tree canopy
- Water polygons may include elevated structures
- Vegetation polygons miss complex organic shapes

### Adaptive Approach (Solution)

```
Ground Truth → Spatial Guidance ↘
                                  → Confidence Vote → Classification
Point Cloud Features → Primary Signal ↗
```

**Advantages:**

- Point cloud features are the **primary classification signal**
- Ground truth provides **spatial context** (where to look)
- **Fuzzy boundaries** allow smooth transitions
- **Confidence-based** voting handles uncertainty
- **Adaptive** to polygon misalignment

---

## Architecture

### 1. Comprehensive Adaptive Classifier

```python
from ign_lidar.core.classification.adaptive_classifier import (
    ComprehensiveAdaptiveClassifier,
    AdaptiveReclassificationConfig
)

# Initialize
config = AdaptiveReclassificationConfig()
classifier = ComprehensiveAdaptiveClassifier(config)

# Classify all features adaptively
refined_labels, stats = classifier.classify_all_adaptive(
    points=points,
    labels=initial_labels,
    ground_truth_data=gt_data,  # Optional guidance
    features=computed_features
)
```

### 2. Configuration

```python
@dataclass
class AdaptiveReclassificationConfig:
    # Confidence thresholds
    MIN_CONFIDENCE = 0.5          # Minimum to classify
    HIGH_CONFIDENCE = 0.7         # High confidence
    GT_WEIGHT = 0.20              # Ground truth (guidance only)
    GEOMETRY_WEIGHT = 0.80        # Geometric features (primary)

    # Fuzzy boundaries
    BUFFER_DISTANCE = 3.0         # Fuzzy boundary distance (meters)
    DECAY_RATE = 0.5              # Confidence decay outside polygon

    # Feature-specific thresholds
    WATER_PLANARITY_MIN = 0.90    # Water is extremely flat
    ROAD_NDVI_MAX = 0.20          # Roads have low NDVI
    VEG_CURVATURE_MIN = 0.02      # Vegetation has complex surfaces
    ...
```

---

## Feature-Specific Classification

### Water Classification

**Principle:** Water has the most distinctive geometric signature (flat, horizontal, smooth).

**Features used (by weight):**

1. **Planarity (30%)**: Water is extremely flat (>0.90)
2. **Height (25%)**: Near ground level (<0.5m)
3. **Normals (20%)**: Horizontal surface (Z component >0.92)
4. **Curvature (10%)**: Very low curvature (<0.02)
5. **NDVI (5%)**: Low NDVI (<0.15)
6. **Roughness (5%)**: Very smooth (<0.02)
7. **Ground truth (5%)**: Spatial guidance only

**Example:**

```python
refined, stats = classifier.refine_water_adaptive(
    points=points,
    labels=labels,
    gt_water=water_polygons,  # Optional
    height=height,
    planarity=planarity,
    curvature=curvature,
    normals=normals,
    ndvi=ndvi,
    roughness=roughness
)

# Output:
# Water: ✓12,450 +320 ✗180
#   ✓ = validated
#   + = added
#   ✗ = rejected (elevated structures, rough surfaces)
```

---

### Road Classification

**Principle:** Roads are flat, horizontal, near ground, but less distinctive than water. Special handling for tree canopy and bridges.

**Features used (by weight):**

1. **Planarity (25%)**: Very flat (>0.75)
2. **Height (20%)**: Near ground (-0.5m to 1.5m)
3. **Normals (15%)**: Horizontal (Z >0.85)
4. **NDVI (15%)**: Low NDVI (<0.20) - excludes vegetation
5. **Verticality (10%)**: Not vertical (<0.30)
6. **Curvature (5%)**: Smooth (<0.05)
7. **Roughness (5%)**: Smooth (<0.05)
8. **Ground truth (5%)**: Spatial guidance

**Special cases:**

- **Tree canopy**: High NDVI + elevated → reclassify as vegetation
- **Bridges**: Elevated (>3m) but road-like → classify as bridge

**Example:**

```python
refined, stats = classifier.refine_roads_adaptive(
    points=points,
    labels=labels,
    gt_roads=road_polygons,
    height=height,
    planarity=planarity,
    curvature=curvature,
    normals=normals,
    ndvi=ndvi,
    roughness=roughness,
    verticality=verticality
)

# Output:
# Roads: ✓45,680 +1,240 ✗3,580
#   🌳 Tree canopy: 2,850
#   🌉 Bridges: 420
```

---

### Vegetation Classification

**Principle:** Vegetation has complex organic surfaces with high NDVI. Multi-feature confidence captures diverse vegetation types.

**Features used (by weight):**

1. **NDVI (40%)**: Primary indicator (>0.25)
2. **Curvature (20%)**: Complex surfaces (>0.02)
3. **Planarity inverse (15%)**: Low planarity (<0.50)
4. **Roughness (10%)**: Irregular surfaces (>0.03)
5. **Sphericity (10%)**: Organic shapes
6. **Ground truth (5%)**: Minimal spatial guidance

**Height-based classification:**

- **Low vegetation**: 0-0.5m (grass, shrubs)
- **Medium vegetation**: 0.5-2m (bushes)
- **High vegetation**: >2m (trees)

**Example:**

```python
refined, stats = classifier.refine_vegetation_adaptive(
    points=points,
    labels=labels,
    gt_vegetation=veg_polygons,  # Optional
    ndvi=ndvi,
    height=height,
    curvature=curvature,
    planarity=planarity,
    roughness=roughness,
    sphericity=sphericity
)

# Output:
# Vegetation: 89,450 (L:12,340 M:38,560 H:38,550)
#   ✗ Rejected: 1,280
```

---

### Building Classification

**Principle:** Buildings have structured geometry (flat roofs, vertical walls). Fuzzy boundaries handle polygon misalignment.

**Features used (by weight):**

1. **Ground truth (25%)**: Spatial guidance (higher weight for buildings)
2. **Height (20%)**: Elevated structures (>2.5m)
3. **Planarity (15%)**: Flat surfaces (>0.60)
4. **Verticality (15%)**: Vertical walls (>0.60)
5. **NDVI inverse (10%)**: Not vegetation (<0.25)
6. **Wall/roof scores (10%)**: Architectural features
7. **Normals (5%)**: Horizontal roofs or vertical walls

**Fuzzy boundaries:**

- Inside polygon: confidence = 1.0
- Outside polygon: confidence = exp(-distance / decay_rate)
- Buffer distance: 3.0m by default

**Example:**

```python
refined, stats = classifier.classify_buildings_adaptive(
    points=points,
    labels=labels,
    gt_buildings=building_polygons,
    height=height,
    planarity=planarity,
    verticality=verticality,
    ndvi=ndvi,
    normals=normals,
    wall_score=wall_score,
    roof_score=roof_score
)

# Output:
# Building: ✓32,580 +4,230 ✗890
#   🏢 Walls: 18,450, Roofs: 14,130
```

---

## Complete Pipeline

### Step 1: Compute Features

```python
from ign_lidar.features import compute_geometric_features

features = compute_geometric_features(
    points=points,
    normals=normals,
    k=20
)
# Returns: height, planarity, curvature, roughness, verticality, etc.
```

### Step 2: Load Ground Truth (Optional)

```python
import geopandas as gpd

ground_truth_data = {
    'buildings': gpd.read_file('buildings.geojson'),
    'roads': gpd.read_file('roads.geojson'),
    'water': gpd.read_file('water.geojson'),
    'vegetation': gpd.read_file('vegetation.geojson')
}
```

### Step 3: Run Adaptive Classification

```python
from ign_lidar.core.classification.adaptive_classifier import (
    refine_all_classifications_adaptive
)

refined_labels, stats = refine_all_classifications_adaptive(
    points=points,
    labels=initial_labels,
    ground_truth_data=ground_truth_data,  # Optional
    features=features
)
```

### Step 4: Analyze Results

```python
print("Classification Results:")
print(f"  Water: {np.sum(refined_labels == 9):,}")
print(f"  Roads: {np.sum(refined_labels == 11):,}")
print(f"  Bridges: {np.sum(refined_labels == 17):,}")
print(f"  Low vegetation: {np.sum(refined_labels == 3):,}")
print(f"  Medium vegetation: {np.sum(refined_labels == 4):,}")
print(f"  High vegetation: {np.sum(refined_labels == 5):,}")
print(f"  Buildings: {np.sum(refined_labels == 6):,}")
```

---

## Validation & Quality Control

### 1. Confidence Scores

Each classification has a confidence score (0-1):

- **>0.7**: High confidence (reliable)
- **0.5-0.7**: Medium confidence (check manually)
- **<0.5**: Low confidence (likely misclassified)

### 2. Feature Importance

Different features contribute differently:

- **Water**: Planarity (30%) + Height (25%)
- **Roads**: Planarity (25%) + Height (20%) + NDVI (15%)
- **Vegetation**: NDVI (40%) + Curvature (20%)
- **Buildings**: GT (25%) + Height (20%)

### 3. Quality Metrics

```python
# Compute classification quality
from sklearn.metrics import classification_report

print(classification_report(
    y_true=ground_truth_labels,
    y_pred=refined_labels,
    target_names=['Ground', 'Water', 'Road', 'Vegetation', 'Building']
))
```

---

## Best Practices

### 1. Feature Computation

- **Use appropriate neighborhood size**: k=20 for geometric features
- **Normalize features**: Scale to [0, 1] range
- **Handle missing features**: System adapts automatically

### 2. Ground Truth Usage

- **Optional**: System works without ground truth
- **Guidance only**: Don't trust polygons absolutely
- **Fuzzy boundaries**: Allow 3-5m buffer for misalignment
- **Quality check**: Validate GT before using

### 3. Parameter Tuning

Adjust thresholds for your data:

```python
config = AdaptiveReclassificationConfig()
config.WATER_PLANARITY_MIN = 0.92  # Stricter for very flat water
config.VEG_NDVI_MIN = 0.30         # Higher for healthy vegetation
config.BUFFER_DISTANCE = 5.0       # Larger buffer for poor GT
```

### 4. Iterative Refinement

```python
# First pass: Water (most distinctive)
labels, _ = classifier.refine_water_adaptive(...)

# Second pass: Roads (after water removed)
labels, _ = classifier.refine_roads_adaptive(...)

# Third pass: Vegetation (after roads removed)
labels, _ = classifier.refine_vegetation_adaptive(...)

# Final pass: Buildings (after all others)
labels, _ = classifier.classify_buildings_adaptive(...)
```

---

## Troubleshooting

### Problem: Too many water points rejected

**Cause**: Water features not distinctive enough

**Solution**: Lower planarity threshold

```python
config.WATER_PLANARITY_MIN = 0.85  # Default: 0.90
```

### Problem: Tree canopy not detected

**Cause**: NDVI threshold too high

**Solution**: Lower NDVI threshold for vegetation

```python
config.VEG_NDVI_MIN = 0.20  # Default: 0.25
```

### Problem: Buildings not detected outside polygons

**Cause**: Buffer distance too small

**Solution**: Increase buffer distance

```python
config.BUFFER_DISTANCE = 5.0  # Default: 3.0
```

### Problem: Roads include vegetation

**Cause**: NDVI not used or threshold too high

**Solution**: Ensure NDVI is provided and lower threshold

```python
config.ROAD_NDVI_MAX = 0.15  # Default: 0.20
```

---

## Performance

### Computational Complexity

- **Water**: O(n) - simple geometric checks
- **Roads**: O(n) - linear feature evaluation
- **Vegetation**: O(n) - multi-feature voting
- **Buildings**: O(n log m) - spatial indexing for GT proximity

### Memory Usage

- **Per point**: ~100 bytes (all features)
- **1M points**: ~100 MB
- **10M points**: ~1 GB

### Optimization Tips

1. **Chunk processing**: Process 1M points at a time
2. **Feature caching**: Compute features once, reuse
3. **Spatial indexing**: Use STRtree for GT polygons
4. **GPU acceleration**: Use CuPy for large datasets

---

## References

1. **Ground Truth Refinement**: `docs/guides/ground-truth-refinement.md`
2. **Building Fusion**: `docs/BUILDING_FUSION_GUIDE.md`
3. **Classification Thresholds**: `ign_lidar/core/classification/classification_thresholds.py`
4. **Feature Computation**: `ign_lidar/features/`

---

## Examples

See `examples/demo_comprehensive_adaptive_classification.py` for complete working examples.

Run demos:

```bash
python examples/demo_comprehensive_adaptive_classification.py
```

---

## API Reference

### ComprehensiveAdaptiveClassifier

```python
class ComprehensiveAdaptiveClassifier:
    def __init__(self, config: AdaptiveReclassificationConfig = None)

    def refine_water_adaptive(...)  # Water classification
    def refine_roads_adaptive(...)  # Roads + canopy + bridges
    def refine_vegetation_adaptive(...)  # Vegetation (low/med/high)
    def classify_buildings_adaptive(...)  # Buildings (from ground_truth_refinement.py)
    def classify_all_adaptive(...)  # Complete pipeline
```

### Convenience Function

```python
def refine_all_classifications_adaptive(
    points: np.ndarray,
    labels: np.ndarray,
    ground_truth_data: Optional[Dict] = None,
    features: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[AdaptiveReclassificationConfig] = None
) -> Tuple[np.ndarray, Dict[str, Any]]
```

---

## Summary

The comprehensive adaptive classification system provides:

✅ **Point cloud-driven** classification (features are primary)  
✅ **Ground truth as guidance** (not absolute truth)  
✅ **Fuzzy boundaries** (handles polygon misalignment)  
✅ **Confidence-based voting** (multi-feature validation)  
✅ **All feature types** (water, roads, vegetation, buildings)  
✅ **Special handling** (tree canopy, bridges, height-based vegetation)  
✅ **Robust & adaptive** (works with or without ground truth)

This approach significantly improves classification accuracy for real-world LiDAR data where ground truth polygons are imperfect.
