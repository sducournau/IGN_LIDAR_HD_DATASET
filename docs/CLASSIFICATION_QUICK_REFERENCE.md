# Classification System Quick Reference

## 🔧 Quick Fixes Needed

### Critical (Fix This Week)

```bash
# Issue #8: Unify Thresholds
# Create: ign_lidar/core/modules/classification_thresholds.py
# Update: transport_detection.py, classification_refinement.py, advanced_classification.py

# Issue #14: Add Spatial Indexing
# Update: advanced_classification.py::_classify_roads_with_buffer()
# Update: advanced_classification.py::_classify_railways_with_buffer()
```

---

## 📊 Current Threshold Values

### Buildings

```python
HEIGHT_MIN = 2.5m          # ✅ Consistent
HEIGHT_MAX = 200.0m        # ✅ Consistent
WALL_VERTICALITY = 0.65-0.75  # ✅ Mode-dependent
ROOF_HORIZONTALITY = 0.80-0.85 # ✅ Mode-dependent
```

### Roads

```python
HEIGHT_MAX = 0.5m or 1.5m  # ❌ INCONSISTENT - FIX
HEIGHT_MIN = -0.3m         # ✅ Reasonable
PLANARITY_MIN = 0.6 or 0.8 # ⚠️ Review
INTENSITY = 0.15-0.7       # ⚠️ Material-specific
```

### Railways

```python
HEIGHT_MAX = 0.8m or 1.2m  # ❌ INCONSISTENT - FIX
HEIGHT_MIN = -0.2m         # ✅ Reasonable
PLANARITY_MIN = 0.5 or 0.75 # ⚠️ Review
BUFFER_MULTIPLIER = 1.2    # ✅ Good
```

---

## 🎯 Classification Priority Order

```
1. Ground Truth (IGN BD TOPO®)    [HIGHEST]
   ├─ Buildings
   ├─ Roads (with buffering)
   ├─ Railways (with buffering)
   ├─ Bridges
   ├─ Water
   └─ Others

2. NDVI Vegetation
   ├─ High vegetation (>2m)
   ├─ Medium vegetation (0.5-2m)
   └─ Low vegetation (<0.5m)

3. Geometric Features
   ├─ Buildings (multi-mode)
   ├─ Transport (multi-mode)
   └─ Ground

4. Height-based
5. Default/Fallback
```

---

## 🏗️ Building Detection Modes

### ASPRS Mode (General)

```python
mode = 'asprs'
output = ASPRS Class 6 (Building)
use_case = "General classification"
```

### LOD2 Mode (Elements)

```python
mode = 'lod2'
output = {0: Wall, 1: Roof_flat, 2: Roof_gable, 3: Roof_hip}
use_case = "Building reconstruction training"
```

### LOD3 Mode (Details)

```python
mode = 'lod3'
output = LOD2 + {13: Door, 14: Window, 15: Balcony, ...}
use_case = "Architectural modeling"
```

---

## 🚗 Transport Detection Modes

### ASPRS Standard

```python
mode = 'asprs_standard'
output = {10: Rail, 11: Road}
use_case = "Basic classification"
```

### ASPRS Extended

```python
mode = 'asprs_extended'
output = {32: Motorway, 33: Primary, ...}
use_case = "Detailed road types"
```

### LOD2

```python
mode = 'lod2'
output = Class 9 (Ground)
use_case = "Training data"
```

---

## 🔍 Key Functions

### Advanced Classification

```python
from ign_lidar.core.modules.advanced_classification import (
    AdvancedClassifier,
    classify_with_all_methods
)

# Main classifier
classifier = AdvancedClassifier(
    use_ground_truth=True,
    use_ndvi=True,
    use_geometric=True,
    building_detection_mode='lod2',
    transport_detection_mode='asprs_standard'
)

labels = classifier.classify_points(
    points=xyz,
    ground_truth_features=gt_data,
    ndvi=ndvi_values,
    height=height_values,
    normals=normals,
    planarity=planarity
)
```

### Building Detection

```python
from ign_lidar.core.modules.building_detection import (
    BuildingDetectionMode,
    detect_buildings_multi_mode
)

labels, stats = detect_buildings_multi_mode(
    labels=initial_labels,
    features={'height': h, 'planarity': p, ...},
    mode='lod2',
    ground_truth_mask=building_gt,
    config=None  # Use defaults
)
```

### Transport Detection

```python
from ign_lidar.core.modules.transport_detection import (
    TransportDetectionMode,
    detect_transport_multi_mode
)

labels, stats = detect_transport_multi_mode(
    labels=initial_labels,
    features={'height': h, 'planarity': p, ...},
    mode='asprs_standard',
    road_ground_truth_mask=road_gt,
    rail_ground_truth_mask=rail_gt
)
```

### Classification Refinement

```python
from ign_lidar.core.modules.classification_refinement import (
    refine_classification,
    RefinementConfig
)

config = RefinementConfig()
refined, stats = refine_classification(
    labels=initial_labels,
    features={'ndvi': ndvi, 'height': h, ...},
    ground_truth_data={'building_mask': gt},
    config=config,
    lod_level='LOD2'
)
```

---

## ⚙️ Configuration

### YAML Config

```yaml
# configs/classification_config.yaml

classification:
  methods:
    geometric: true
    ndvi: true
    ground_truth: true

  thresholds:
    ndvi_vegetation: 0.35
    height_low_veg: 0.5
    height_medium_veg: 2.0
    planarity_road: 0.8
    planarity_building: 0.7
```

### Python Config

```python
from ign_lidar.core.modules.classification_refinement import RefinementConfig

config = RefinementConfig()
config.ROAD_HEIGHT_MAX = 1.5
config.BUILDING_HEIGHT_MIN = 2.5
config.USE_GROUND_TRUTH = True
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Tests

```bash
# Building detection
pytest tests/test_building_detection_modes.py -v

# Classification refinement
pytest tests/test_classification_refinement.py -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=ign_lidar --cov-report=html
```

---

## 📝 Common Issues & Solutions

### Issue: Roads Not Detected

```python
# Check height threshold
if max_height > 1.5:
    # Points filtered out - adjust threshold
    config.ROAD_HEIGHT_MAX = 2.0

# Check planarity
if road_planarity < 0.6:
    # Surface not flat enough - lower threshold
    config.ROAD_PLANARITY_MIN = 0.5
```

### Issue: Buildings Misclassified as Vegetation

```python
# Use NDVI refinement
classifier = AdvancedClassifier(
    use_ndvi=True,
    ndvi_building_threshold=0.15  # Lower = stricter
)

# Or use ground truth
ground_truth_features = fetcher.fetch_buildings(bbox)
```

### Issue: Performance Slow

```python
# Current: O(n*m) without indexing
# Solution: Add spatial indexing
from shapely.strtree import STRtree

tree = STRtree(point_geoms)
candidates = tree.query(polygon)
```

### Issue: Threshold Inconsistencies

```python
# Problem: Different values in different modules
# Solution: Use unified thresholds
from ign_lidar.core.modules.classification_thresholds import UnifiedThresholds

HEIGHT_MAX = UnifiedThresholds.ROAD_HEIGHT_MAX
```

---

## 🚀 Quick Start Examples

### Example 1: Basic Classification

```python
from ign_lidar.core.modules.advanced_classification import classify_with_all_methods

labels = classify_with_all_methods(
    points=xyz,
    labels=initial_labels,
    height=height_values,
    normals=normals,
    planarity=planarity,
    building_detection_mode='asprs',
    transport_detection_mode='asprs_standard'
)
```

### Example 2: With Ground Truth

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

fetcher = IGNGroundTruthFetcher()
gt_features = fetcher.fetch_all_features(
    bbox=(xmin, ymin, xmax, ymax),
    include_buildings=True,
    include_roads=True
)

classifier = AdvancedClassifier(use_ground_truth=True)
labels = classifier.classify_points(
    points=xyz,
    ground_truth_features=gt_features,
    height=height
)
```

### Example 3: LOD2 Training Data

```python
# Full pipeline for LOD2 training
labels, stats = detect_buildings_multi_mode(
    labels=initial,
    features=geometric_features,
    mode='lod2',
    ground_truth_mask=building_gt
)

refined, ref_stats = refine_classification(
    labels=labels,
    features=all_features,
    ground_truth_data=gt_data,
    lod_level='LOD2'
)
```

---

## 📚 Key Files

### Core Modules

- `ign_lidar/core/modules/advanced_classification.py` - Main classifier
- `ign_lidar/core/modules/building_detection.py` - Building detection
- `ign_lidar/core/modules/transport_detection.py` - Transport detection
- `ign_lidar/core/modules/classification_refinement.py` - Refinement

### Configuration

- `configs/classification_config.yaml` - Main config
- `ign_lidar/core/modules/classification_refinement.py::RefinementConfig`

### Tests

- `tests/test_building_detection_modes.py`
- `tests/test_classification_refinement.py`

### Documentation

- `docs/ADVANCED_CLASSIFICATION_GUIDE.md` - User guide
- `docs/CLASSIFICATION_AUDIT_REPORT.md` - Full audit
- `docs/AUDIT_ACTION_PLAN.md` - Implementation plan

---

## 🐛 Debug Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ign_lidar')
logger.setLevel(logging.DEBUG)
```

### Check Classification Stats

```python
labels, stats = detect_buildings_multi_mode(...)
print(f"Buildings detected: {stats['total_building']}")
print(f"From ground truth: {stats.get('ground_truth_building', 0)}")
print(f"From geometry: {stats.get('geometric_building', 0)}")
```

### Validate Thresholds

```python
# Check current configuration
config = RefinementConfig()
print(f"Road height max: {config.ROAD_HEIGHT_MAX}")
print(f"Building height min: {config.BUILDING_HEIGHT_MIN}")
print(f"Road planarity min: {config.ROAD_PLANARITY_MIN}")
```

---

## 🔗 Related Resources

- Main README: `/README.md`
- Configuration Guide: `docs/CONFIG_QUICK_REFERENCE.md`
- Road Segmentation: `docs/ROAD_SEGMENTATION_IMPROVEMENTS.md`
- Testing Guide: `docs/TESTING.md`

---

**Last Updated:** October 16, 2025  
**Maintained By:** Development Team
