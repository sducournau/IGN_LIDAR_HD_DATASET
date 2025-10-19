# Quick Reference: Classification Enhancements V2.0

## 🎯 What Was Implemented

### 1. **Building Detection & Clustering** ✅

- **File**: `ign_lidar/core/classification/building_clustering.py`
- **New class**: `BuildingClusterer` - clusters points by building footprint
- **Features**: Centroid attraction, polygon adjustment, multi-source fusion

### 2. **Enhanced Vegetation (NDVI)** ✅

- **File**: `ign_lidar/core/classification/advanced_classification.py` (modified)
- **Increased thresholds**: All NDVI thresholds raised by +0.05
- **Original preservation**: New feature to preserve vegetation where NDVI confirms

### 3. **RGE ALTI® DTM Integration** ✅

- **File**: `ign_lidar/io/rge_alti_fetcher.py`
- **New class**: `RGEALTIFetcher` - fetches and processes French national DTM
- **Features**: WCS download, local files, caching, ground point generation

### 4. **Ground-Referenced Roads/Water** ✅

- **Status**: Already implemented in `advanced_classification.py`
- **Uses**: Height above ground for filtering elevated points (bridges)

### 5. **MNT Height Separation** ✅

- **Status**: Already implemented via `classification_thresholds.py`
- **Enhanced**: Can now use RGE ALTI for true ground reference

---

## 🚀 Quick Start

### Example 1: Use Building Clustering

```python
from ign_lidar.core.classification.building_clustering import cluster_buildings_multi_source

building_ids, clusters = cluster_buildings_multi_source(
    points=points,
    ground_truth_features={'buildings': buildings_gdf},
    labels=labels,
    building_classes=[6]
)

print(f"Found {len(clusters)} buildings")
for cluster in clusters[:5]:
    print(f"  Building {cluster.building_id}: {cluster.n_points} pts, {cluster.volume:.0f}m³")
```

### Example 2: Use RGE ALTI for Ground Height

```python
from ign_lidar.io.rge_alti_fetcher import RGEALTIFetcher

fetcher = RGEALTIFetcher(cache_dir="/cache", use_wcs=True)
height = fetcher.compute_height_above_ground(points, bbox)
```

### Example 3: Augment Ground with DTM

```python
from ign_lidar.io.rge_alti_fetcher import augment_ground_with_rge_alti

points_aug, labels_aug = augment_ground_with_rge_alti(
    points, labels, bbox, spacing=2.0
)
print(f"Added {len(points_aug) - len(points)} synthetic ground points")
```

### Example 4: Enhanced Vegetation with Preservation

```python
from ign_lidar.core.classification.advanced_classification import AdvancedClassifier

classifier = AdvancedClassifier(
    ndvi_veg_threshold=0.35,  # Increased sensitivity
    use_ndvi=True
)

labels = classifier.classify_points(
    points=points,
    ndvi=ndvi,
    height=height  # Use RGE ALTI-derived height!
)
```

---

## 📊 Key Changes Summary

| Component                 | Before  | After              | Impact                  |
| ------------------------- | ------- | ------------------ | ----------------------- |
| **NDVI Dense Forest**     | 0.60    | **0.65**           | More selective          |
| **NDVI Healthy Trees**    | 0.50    | **0.55**           | Higher quality          |
| **NDVI Moderate Veg**     | 0.40    | **0.45**           | Better separation       |
| **NDVI Grass**            | 0.30    | **0.35**           | Improved detection      |
| **NDVI Sparse Veg**       | 0.20    | **0.25**           | More sensitive          |
| **Building Clustering**   | ❌ None | ✅ **New module**  | Spatial coherence       |
| **RGE ALTI Integration**  | ❌ None | ✅ **New module**  | True ground reference   |
| **Original Preservation** | ❌ None | ✅ **New feature** | Protects quality labels |

---

## 📁 New Files Created

1. **`ign_lidar/core/classification/building_clustering.py`**

   - `BuildingCluster` dataclass
   - `BuildingClusterer` class
   - `cluster_buildings_multi_source()` function

2. **`ign_lidar/io/rge_alti_fetcher.py`**

   - `RGEALTIFetcher` class
   - `augment_ground_with_rge_alti()` function

3. **`docs/CLASSIFICATION_ENHANCEMENTS_V2.md`**
   - Complete documentation
   - Integration guide
   - Usage examples

---

## 🔧 Dependencies

### Required

```bash
pip install numpy shapely geopandas laspy
```

### Optional (for RGE ALTI)

```bash
pip install rasterio affine requests
```

---

## ⚙️ Configuration Updates

Add to your YAML config:

```yaml
# NEW: RGE ALTI configuration
rge_alti:
  enabled: true
  cache_dir: /data/rge_alti_cache
  resolution: 1.0
  use_wcs: true
  augment_ground: true
  ground_spacing: 2.0

# UPDATED: NDVI thresholds (increased sensitivity)
classification:
  ndvi_dense_forest: 0.65 # was 0.60
  ndvi_healthy_trees: 0.55 # was 0.50
  ndvi_moderate_veg: 0.45 # was 0.40
  ndvi_grass: 0.35 # was 0.30
  ndvi_sparse_veg: 0.25 # was 0.20

# NEW: Building clustering
building_clustering:
  enabled: true
  use_centroid_attraction: true
  attraction_radius: 5.0
  min_points_per_building: 10
```

---

## 📖 Full Documentation

See **`docs/CLASSIFICATION_ENHANCEMENTS_V2.md`** for:

- Detailed implementation guide
- Complete code examples
- Integration pipeline
- Testing strategies
- Performance notes

---

**Version**: 2.0  
**Date**: October 19, 2025  
**Status**: ✅ All Enhancements Completed
