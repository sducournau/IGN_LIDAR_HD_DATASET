# Classification Module - Quick Reference Guide

**Date:** October 23, 2025  
**Module:** `ign_lidar/core/classification`  
**Status:** Production Ready (v3.1+)

---

## 🚀 Quick Start

### Basic Classification

```python
from ign_lidar.core.classification import UnifiedClassifier

# Create classifier with default settings
classifier = UnifiedClassifier(
    strategy='comprehensive',  # 'basic', 'adaptive', or 'comprehensive'
    mode='asprs_extended'      # 'asprs_standard', 'asprs_extended', 'lod2', 'lod3'
)

# Classify points
labels = classifier.classify_points(
    points=points,      # [N, 3] array (x, y, z)
    features=features   # Dict with 'height', 'planarity', 'ndvi', etc.
)
```

### Hierarchical Classification (ASPRS → LOD2 → LOD3)

```python
from ign_lidar.core.classification import HierarchicalClassifier
from ign_lidar.core.classification.hierarchical_classifier import ClassificationLevel

# Create hierarchical classifier
classifier = HierarchicalClassifier(
    target_level=ClassificationLevel.LOD2,
    use_confidence_scores=True
)

# Classify with hierarchy
result = classifier.classify(
    points=points,
    features=features,
    initial_labels=asprs_labels  # Starting from ASPRS
)

# Access results
labels = result.labels
confidence = result.confidence_scores
stats = result.get_statistics()
```

---

## 📚 Common Imports

### Essential Imports

```python
# Classification schema (ASPRS, LOD2, LOD3 classes)
from ign_lidar.classification_schema import (
    ASPRSClass, LOD2Class, LOD3Class,
    ClassificationMode,
    get_class_name, get_class_color
)

# Thresholds
from ign_lidar.core.classification.thresholds import (
    get_thresholds, ThresholdConfig
)

# Main classifiers
from ign_lidar.core.classification import (
    UnifiedClassifier,
    HierarchicalClassifier
)
```

### Building Module

```python
from ign_lidar.core.classification.building import (
    AdaptiveBuildingClassifier,
    BuildingDetector,
    BuildingClusterer,
    BuildingFusion,
    BuildingMode
)
```

### Transport Module

```python
from ign_lidar.core.classification.transport import (
    TransportDetector,
    enhance_classification
)
```

### Rules Framework (New in v3.2)

```python
from ign_lidar.core.classification.rules import (
    BaseRule,
    RuleEngine,
    validate_features,
    compute_confidence_linear
)
```

---

## 🎨 Classification Modes

### Mode Comparison

| Mode | Classes | Use Case | ASPRS Codes |
|------|---------|----------|-------------|
| `asprs_standard` | 22 | Standard LAS output | 0-31 |
| `asprs_extended` | 100+ | BD TOPO integration | 0-255 |
| `lod2` | 15 | Building-focused training | Custom |
| `lod3` | 30 | Detailed architecture | Custom |

### Example: Mode Selection

```python
from ign_lidar.classification_schema import ClassificationMode

# For standard LAS export
mode = ClassificationMode.ASPRS_STANDARD

# For BD TOPO/cadastre integration
mode = ClassificationMode.ASPRS_EXTENDED

# For building ML training
mode = ClassificationMode.LOD2

# For detailed 3D modeling
mode = ClassificationMode.LOD3
```

---

## 🏗️ Common Patterns

### Pattern 1: Get Thresholds for Context

```python
from ign_lidar.core.classification.thresholds import get_thresholds

# Standard thresholds
thresholds = get_thresholds(mode='lod2', strict=False)

# Strict thresholds for urban areas
thresholds_urban = get_thresholds(mode='lod2', strict=True)

# Access specific thresholds
road_height = thresholds.height.road_height_max
building_planarity = thresholds.building.min_planarity_lod2
ndvi_vegetation = thresholds.ndvi.vegetation_min
```

### Pattern 2: Building Detection

```python
from ign_lidar.core.classification.building import BuildingDetector

detector = BuildingDetector(mode='lod2')

# Detect buildings
result = detector.detect(
    points=points,
    features=features,
    ground_truth_polygons=bdtopo_buildings  # Optional
)

# Access results
building_mask = result.mask
building_confidence = result.confidence
```

### Pattern 3: Feature Validation

```python
from ign_lidar.core.classification.rules.validation import validate_features

# Check required features are present
required_features = {'height', 'planarity', 'ndvi'}
is_valid = validate_features(features, required_features)

if not is_valid:
    missing = required_features - set(features.keys())
    print(f"Missing features: {missing}")
```

### Pattern 4: Ground Truth Refinement

```python
from ign_lidar.core.classification import GroundTruthRefiner

refiner = GroundTruthRefiner(
    enable_buildings=True,
    enable_roads=True,
    enable_vegetation=True
)

# Refine classification with ground truth
refined_labels = refiner.refine(
    points=points,
    initial_labels=labels,
    bdtopo_buildings=buildings_gdf,
    bdtopo_roads=roads_gdf,
    oso_vegetation=vegetation_gdf
)
```

### Pattern 5: Parcel-Based Classification

```python
from ign_lidar.core.classification import ParcelClassifier

classifier = ParcelClassifier(
    min_building_area=10.0,
    buffer_distance=0.5
)

# Classify using cadastral parcels
result = classifier.classify_by_parcels(
    points=points,
    features=features,
    parcels=cadastre_gdf,
    initial_labels=labels
)
```

---

## 🔧 Configuration Examples

### Basic Configuration

```python
from ign_lidar.core.classification import UnifiedClassifierConfig

config = UnifiedClassifierConfig(
    strategy='comprehensive',
    mode='asprs_extended',
    use_ground_truth=True,
    use_parcel_classification=False,
    strict_mode=False
)

classifier = UnifiedClassifier(config=config)
```

### Advanced Configuration with All Options

```python
config = UnifiedClassifierConfig(
    # Strategy
    strategy='comprehensive',
    mode='lod2',
    
    # Ground truth
    use_ground_truth=True,
    ground_truth_priority=0.9,
    
    # Features
    required_features={'height', 'planarity'},
    optional_features={'ndvi', 'intensity'},
    
    # Thresholds
    strict_mode=True,
    custom_thresholds=None,
    
    # Building detection
    building_detection_mode='multi_source',
    min_building_area=10.0,
    
    # Transport detection
    transport_detection_mode='geometry',
    road_buffer_distance=2.0,
    
    # Parcel classification
    use_parcel_classification=True,
    parcel_classification_config={
        'min_building_area': 15.0,
        'buffer_distance': 1.0
    },
    
    # Performance
    use_gpu=True,
    batch_size=10000,
    n_jobs=-1
)
```

---

## 📊 Feature Requirements

### Required Features by Classification Type

```python
# Water classification
water_features = ['height', 'planarity', 'curvature', 'normals']

# Road classification
road_features = ['height', 'planarity', 'curvature', 'normals', 'ndvi']

# Vegetation classification
vegetation_features = ['ndvi', 'height', 'curvature', 'planarity', 
                       'sphericity', 'roughness']

# Building classification
building_features = ['height', 'planarity', 'verticality', 'ndvi']
```

### Feature Dictionary Structure

```python
features = {
    'height': np.array([...]),        # Height above ground (meters)
    'planarity': np.array([...]),     # Flatness [0-1]
    'verticality': np.array([...]),   # Wall-like measure [0-1]
    'curvature': np.array([...]),     # Surface curvature [0-∞]
    'roughness': np.array([...]),     # Surface roughness [0-∞]
    'sphericity': np.array([...]),    # Shape sphericity [0-1]
    'ndvi': np.array([...]),          # Normalized Difference Vegetation Index [-1, 1]
    'intensity': np.array([...]),     # LiDAR intensity [0-1]
    'normals': np.array([..., 3])     # Surface normals [N, 3]
}
```

---

## 🎯 Class Codes Reference

### ASPRS Standard (0-31)

```python
from ign_lidar.classification_schema import ASPRSClass

ASPRSClass.GROUND            # 2
ASPRSClass.LOW_VEGETATION    # 3
ASPRSClass.MEDIUM_VEGETATION # 4
ASPRSClass.HIGH_VEGETATION   # 5
ASPRSClass.BUILDING          # 6
ASPRSClass.WATER             # 9
ASPRSClass.RAIL              # 10
ASPRSClass.ROAD_SURFACE      # 11
ASPRSClass.BRIDGE_DECK       # 17
```

### ASPRS Extended - Buildings (50-69)

```python
ASPRSClass.BUILDING_RESIDENTIAL  # 50
ASPRSClass.BUILDING_COMMERCIAL   # 51
ASPRSClass.BUILDING_INDUSTRIAL   # 52
ASPRSClass.BUILDING_RELIGIOUS    # 53
ASPRSClass.BUILDING_ROOF         # 58
ASPRSClass.BUILDING_WALL         # 59
ASPRSClass.BUILDING_FACADE       # 60
```

### LOD2 Building Classes (0-14)

```python
from ign_lidar.classification_schema import LOD2Class

LOD2Class.WALL               # 0
LOD2Class.ROOF_FLAT          # 1
LOD2Class.ROOF_GABLE         # 2
LOD2Class.ROOF_HIP           # 3
LOD2Class.CHIMNEY            # 4
LOD2Class.BALCONY            # 6
LOD2Class.GROUND             # 9
LOD2Class.VEGETATION_LOW     # 10
LOD2Class.VEGETATION_HIGH    # 11
```

---

## 🚨 Common Issues and Solutions

### Issue 1: Missing Features

**Problem:**
```python
# Error: Required feature 'planarity' not found
labels = classifier.classify_points(points, features={'height': heights})
```

**Solution:**
```python
from ign_lidar.core.classification.enrichment import compute_geometric_features_standard

# Compute missing features
features = compute_geometric_features_standard(
    points=points,
    k_neighbors=20,
    search_radius=1.0
)

# Now classify
labels = classifier.classify_points(points, features)
```

### Issue 2: Import Errors (Deprecated Modules)

**Problem:**
```python
# DeprecationWarning: classification_thresholds.py is deprecated
from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds
```

**Solution:**
```python
# ✅ Use new module
from ign_lidar.core.classification.thresholds import get_thresholds

thresholds = get_thresholds(mode='lod2')
```

### Issue 3: Incorrect Feature Shape

**Problem:**
```python
# ValueError: Expected normals shape (N, 3), got (N,)
```

**Solution:**
```python
# Ensure normals are [N, 3]
if normals.ndim == 1:
    normals = normals.reshape(-1, 3)

# Validate before classification
from ign_lidar.core.classification.rules.validation import validate_feature_shape
validate_feature_shape(normals, expected_shape=(len(points), 3))
```

### Issue 4: Low Classification Confidence

**Problem:**
```python
# Many points with low confidence scores
```

**Solution:**
```python
# 1. Check feature quality
from ign_lidar.core.classification.rules.validation import check_feature_quality
quality_report = check_feature_quality(features)

# 2. Use strict thresholds
thresholds = get_thresholds(mode='lod2', strict=True)

# 3. Enable ground truth refinement
classifier = UnifiedClassifier(
    use_ground_truth=True,
    ground_truth_priority=0.9
)
```

---

## 🔍 Debugging and Validation

### Validate Input Data

```python
from ign_lidar.core.classification.loader import validate_lidar_data

# Check data quality
is_valid, errors = validate_lidar_data(
    points=points,
    intensity=intensity,
    rgb=rgb,
    nir=nir
)

if not is_valid:
    print(f"Validation errors: {errors}")
```

### Check Classification Statistics

```python
# Get class distribution
unique, counts = np.unique(labels, return_counts=True)
distribution = dict(zip(unique, counts))

print("Classification distribution:")
for class_code, count in distribution.items():
    class_name = get_class_name(class_code)
    percentage = count / len(labels) * 100
    print(f"  {class_name} ({class_code}): {count:,} ({percentage:.1f}%)")
```

### Visualize Confidence Scores

```python
import matplotlib.pyplot as plt

# Plot confidence histogram
if hasattr(result, 'confidence_scores') and result.confidence_scores is not None:
    plt.figure(figsize=(10, 6))
    plt.hist(result.confidence_scores, bins=50, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Points')
    plt.title('Classification Confidence Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Statistics
    print(f"Mean confidence: {np.mean(result.confidence_scores):.3f}")
    print(f"Median confidence: {np.median(result.confidence_scores):.3f}")
    print(f"Low confidence (<0.5): {np.sum(result.confidence_scores < 0.5):,} points")
```

---

## 📖 Migration Guide (Quick Version)

### From Old Thresholds

```python
# ❌ Old (deprecated)
from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds
height = ClassificationThresholds.ROAD_HEIGHT_MAX

# ✅ New
from ign_lidar.core.classification.thresholds import get_thresholds
thresholds = get_thresholds()
height = thresholds.height.road_height_max
```

### From Old Building Classifier

```python
# ❌ Old (deprecated)
from ign_lidar.core.classification.adaptive_building_classifier import AdaptiveBuildingClassifier

# ✅ New
from ign_lidar.core.classification.building import AdaptiveBuildingClassifier
```

### From Old Transport Detection

```python
# ❌ Old (deprecated)
from ign_lidar.core.classification.transport_detection import TransportDetector

# ✅ New
from ign_lidar.core.classification.transport import TransportDetector
```

---

## 🎓 Best Practices

### 1. Always Validate Features First

```python
from ign_lidar.core.classification.rules.validation import validate_features

required = {'height', 'planarity', 'ndvi'}
if not validate_features(features, required):
    # Compute missing features or raise error
    pass
```

### 2. Use Appropriate Mode for Task

```python
# For standard LAS export → asprs_standard
# For BD TOPO integration → asprs_extended
# For ML training → lod2 or lod3
# For 3D modeling → lod3
```

### 3. Enable Ground Truth When Available

```python
classifier = UnifiedClassifier(
    use_ground_truth=True,
    ground_truth_priority=0.9  # High confidence in ground truth
)
```

### 4. Check Confidence Scores

```python
if result.confidence_scores is not None:
    low_conf_mask = result.confidence_scores < 0.5
    if low_conf_mask.any():
        # Flag for manual review or refinement
        print(f"Warning: {low_conf_mask.sum()} points with low confidence")
```

### 5. Use Hierarchical Classification for Progressive Refinement

```python
# Start with ASPRS, refine to LOD2, then LOD3
asprs_result = asprs_classifier.classify(points, features)
lod2_result = lod2_classifier.classify(points, features, asprs_result.labels)
lod3_result = lod3_classifier.classify(points, features, lod2_result.labels)
```

---

## 📚 Additional Resources

### Documentation
- [Full Analysis Report](./CLASSIFICATION_ANALYSIS_REPORT_2025.md)
- [Action Plan](./CLASSIFICATION_ACTION_PLAN.md)
- [Executive Summary](./CLASSIFICATION_EXECUTIVE_SUMMARY.md)
- [Project Summary](./PROJECT_CONSOLIDATION_SUMMARY.md)

### Migration Guides
- [Threshold Migration Guide](./THRESHOLD_MIGRATION_GUIDE.md)
- [Building Module Migration Guide](./BUILDING_MODULE_MIGRATION_GUIDE.md)
- [Transport Module Migration Guide](./TRANSPORT_MODULE_MIGRATION_GUIDE.md)

### Examples
- See `examples/` directory for complete working examples
- `demo_adaptive_building_classification.py`
- `demo_hierarchical_rules.py`
- `demo_parcel_classification.py`

---

## 🆘 Getting Help

1. **Check documentation** in `docs/` directory
2. **Review examples** in `examples/` directory
3. **Check docstrings** in source code
4. **File GitHub issue** with minimal reproduction

---

**Last Updated:** October 23, 2025  
**Module Version:** v3.2.0  
**Status:** Production Ready
