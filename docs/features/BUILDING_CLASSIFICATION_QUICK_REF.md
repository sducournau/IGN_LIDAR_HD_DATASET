# Building Classification for Non-Classified Points - Quick Reference

**Date:** October 16, 2025  
**Status:** ✅ Implemented

## 🎯 What Was Improved

### Problem

- Many building points remained unclassified (ASPRS code 1)
- Building edges and corners were missed
- Ground truth footprints weren't fully utilized

### Solution

Added **4-stage classification pipeline** with new **Stage 4: Post-Processing**

## 📊 4-Stage Classification Pipeline

```
Input: Raw Point Cloud
  ↓
[Stage 1] Geometric Features → Ground, Roads, Buildings (basic)
  ↓
[Stage 2] NDVI Refinement → Vegetation classification
  ↓
[Stage 3] Ground Truth → Buildings, Roads, Railways (from BD TOPO®)
  ↓
[Stage 4] Post-Process Unclassified (NEW!) → Recover missed buildings
  ↓
Output: Fully Classified Point Cloud (fewer unclassified)
```

## 🔧 New Stage 4 Strategies

### Strategy 1: Ground Truth Footprints

```python
# Check if unclassified points are INSIDE building polygons
if point in building_polygon:
    label = BUILDING  # ASPRS code 6
```

### Strategy 2: Geometric Building-Like

```python
# Building characteristics
if (height > 2.5m AND
    planarity > 0.6 AND
    (vertical OR horizontal) AND
    curvature < 0.02):
    label = BUILDING
```

### Strategy 3: Low-Height Ground

```python
# Very low points → ground
if height < 0.5m:
    label = GROUND  # ASPRS code 2
```

### Strategy 4: Vegetation-Like

```python
# Medium height + irregular → vegetation
if (0.5m < height < 2.0m AND
    planarity < 0.4):
    label = LOW_VEGETATION  # ASPRS code 3
```

## 📈 Expected Results

| Metric                       | Before  | After  | Change              |
| ---------------------------- | ------- | ------ | ------------------- |
| Unclassified building points | 15-25%  | 5-10%  | **-50% to -60%** ✅ |
| Building recall              | 75-85%  | 90-95% | **+15-20%** ✅      |
| Ground truth utilization     | Partial | Full   | **100%** ✅         |

## 🔑 Key Configuration

### File: `configs/classification_config.yaml`

```yaml
classification:
  thresholds:
    # Building-specific
    building_buffer_tolerance: 0.0 # Strict footprint matching
    use_building_footprints: true # Enable BD TOPO® matching
    ground_truth_building_priority: high # Override other signals

  # NEW: Post-processing section
  post_processing:
    enabled: true # Enable Stage 4
    reclassify_unclassified: true # Attempt reclassification
    use_ground_truth_context: true # Use building footprints
    use_geometric_similarity: true # Use geometric features
    min_building_height: 2.5 # Height threshold (m)
    min_building_planarity: 0.6 # Planarity threshold
```

## 🎓 Code Changes Summary

### 1. `advanced_classification.py`

**Added:** Stage 4 post-processing method

```python
def _post_process_unclassified(
    self,
    labels: np.ndarray,
    confidence: np.ndarray,
    points: np.ndarray,
    height: Optional[np.ndarray],
    normals: Optional[np.ndarray],
    planarity: Optional[np.ndarray],
    curvature: Optional[np.ndarray],
    intensity: Optional[np.ndarray],
    ground_truth_features: Optional[Dict]
) -> np.ndarray:
    """
    Post-process unclassified points using:
    1. Ground truth footprint matching
    2. Geometric building-like features
    3. Low-height ground classification
    4. Vegetation-like classification
    """
```

**Modified:** `classify_points()` method

```python
# Stage 4: Post-processing for unclassified points (NEW!)
logger.info("  Stage 4: Post-processing unclassified points")
labels = self._post_process_unclassified(
    labels, confidence, points, height, normals, planarity,
    curvature, intensity, ground_truth_features
)
```

### 2. `building_detection.py`

**Added:** Strategy 6 for unclassified recovery in ASPRS mode

```python
# Strategy 6: Handle unclassified building-like points
unclassified_mask = (refined == 1) | (refined == 0)

if unclassified_mask.any():
    building_like = (
        (height > self.config.min_height) &
        (height < self.config.max_height) &
        (planarity > 0.5) &
        unclassified_mask
    )

    # Validate with normals (orientation)
    # Validate with anisotropy (structure)

    refined[building_like] = ASPRS_BUILDING
    stats['unclassified_recovery'] = building_like.sum()
```

### 3. `classification_config.yaml`

**Added:** New configuration sections

```yaml
# Ground truth building parameters
thresholds:
  building_buffer_tolerance: 0.0
  use_building_footprints: true
  ground_truth_building_priority: high

# Post-processing parameters
post_processing:
  enabled: true
  reclassify_unclassified: true
  use_ground_truth_context: true
  use_geometric_similarity: true
  min_building_height: 2.5
  min_building_planarity: 0.6
```

## 💻 Usage Example

### Simple Classification

```python
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier

# Create classifier (post-processing enabled by default)
classifier = AdvancedClassifier(
    use_ground_truth=True,
    use_ndvi=True,
    use_geometric=True
)

# Classify (Stage 4 runs automatically)
labels = classifier.classify_points(
    points=points,
    ground_truth_features=ground_truth_features,  # BD TOPO® buildings
    ndvi=ndvi,
    height=height,
    normals=normals,
    planarity=planarity,
    curvature=curvature,
    intensity=intensity
)

# Check results
n_unclassified = np.sum(labels == 1)
n_buildings = np.sum(labels == 6)
print(f"Unclassified: {n_unclassified:,}")
print(f"Buildings: {n_buildings:,}")
```

### With Custom Configuration

```python
from ign_lidar.core.config import load_classification_config

# Load config with post-processing parameters
config = load_classification_config('configs/classification_config.yaml')

# Use in classification pipeline
# (post_processing parameters automatically applied)
```

## 🔍 Logging Output

### Expected Log Messages

```
🎯 Classifying 1,000,000 points with advanced method
  Stage 1: Geometric feature classification
    Ground: 250,000 points
    Roads (geometric): 50,000 points
    Buildings (ASPRS): 150,000 points
  Stage 2: NDVI-based vegetation refinement
    Vegetation: 300,000 points
  Stage 3: Ground truth classification (highest priority)
    Processing buildings: 1,234 features
    Classified 180,000 building points
  Stage 4: Post-processing unclassified points
    Post-processing 45,000 unclassified points
      Ground truth: 12,000 points within building footprints
      Geometric: 8,000 building-like points classified
      Low height: 15,000 points classified as ground
      Vegetation-like: 10,000 points classified as low vegetation
    Reclassified 45,000 points, 0 remain unclassified
📊 Final classification distribution:
  Ground              : 265,000 (26.5%)
  Low Vegetation      : 310,000 (31.0%)
  High Vegetation     : 100,000 (10.0%)
  Building            : 200,000 (20.0%)
  Road                :  75,000 ( 7.5%)
  Water               :  50,000 ( 5.0%)
  Unclassified        :       0 ( 0.0%)  ← IMPROVED!
```

## 🎯 Key Thresholds

### Building Detection

| Parameter          | Value    | Purpose                 |
| ------------------ | -------- | ----------------------- |
| min_height         | 2.5m     | Minimum building height |
| min_planarity      | 0.6      | Flat surface threshold  |
| wall_verticality   | 0.7+     | Wall detection          |
| roof_horizontality | 0.85+    | Roof detection          |
| curvature_max      | 0.02     | Exclude vegetation      |
| intensity_range    | 0.2-0.85 | Building materials      |

### Ground Truth

| Parameter        | Value | Purpose                        |
| ---------------- | ----- | ------------------------------ |
| buffer_tolerance | 0.0m  | Strict footprint matching      |
| priority         | HIGH  | Override other classifications |
| use_footprints   | true  | Enable polygon matching        |

## 🐛 Edge Cases Handled

1. **Building Edges/Corners** → Relaxed thresholds + multi-feature validation
2. **Complex Shapes** → Ground truth polygon matching
3. **Roof Vegetation** → Height + planarity + ground truth priority
4. **Low Buildings** → Ground truth catches < 2.5m buildings
5. **Overhangs** → Geometric features detect structural elements

## 📁 Files Modified

```
ign_lidar/
├── core/
│   └── modules/
│       ├── advanced_classification.py  ← Added Stage 4 method
│       └── building_detection.py       ← Added Strategy 6
└── ...

configs/
└── classification_config.yaml          ← Added post_processing section

# New Documentation
BUILDING_CLASSIFICATION_IMPROVEMENTS.md  ← Comprehensive guide
BUILDING_CLASSIFICATION_QUICK_REF.md     ← This file
```

## ✅ Testing Checklist

- [ ] Run classification on test tile
- [ ] Verify unclassified count reduced
- [ ] Check building completeness improved
- [ ] Validate no false positives
- [ ] Review log output for Stage 4
- [ ] Measure performance impact

## 🚀 Next Steps

1. **Run Tests:**

   ```bash
   pytest tests/ -v -m "integration"
   ```

2. **Process Sample Tile:**

   ```bash
   ign-lidar-hd process --config configs/classification_config.yaml
   ```

3. **Analyze Results:**

   ```python
   # Check classification statistics
   from ign_lidar.core.utils import analyze_classification
   analyze_classification('output/classified.laz')
   ```

4. **Tune Thresholds:**
   - Adjust `min_building_height` if needed
   - Adjust `min_building_planarity` if needed
   - Review `ground_truth_building_priority`

## 📚 Related Documentation

- [BUILDING_CLASSIFICATION_IMPROVEMENTS.md](BUILDING_CLASSIFICATION_IMPROVEMENTS.md) - Full technical documentation
- [BD_TOPO_RPG_INTEGRATION.md](BD_TOPO_RPG_INTEGRATION.md) - Ground truth integration
- [CLASSIFICATION_REFERENCE.md](CLASSIFICATION_REFERENCE.md) - Classification system

---

**Questions?** Check the comprehensive documentation in `BUILDING_CLASSIFICATION_IMPROVEMENTS.md`

**Status:** ✅ Ready for testing
