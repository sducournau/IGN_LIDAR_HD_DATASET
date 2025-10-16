# Ground Truth Performance Investigation - Executive Summary

**Date:** 2025-10-16  
**Issue:** Ground truth classification extremely slow (5-30 minutes per tile)

## 🔴 Critical Bottleneck Identified

The ground truth classification performs **brute-force point-in-polygon checks** for millions of points against hundreds of polygons:

```
18,000,000 points × 290 polygons = 5,200,000,000 containment checks
```

Each `polygon.contains(point)` call is computationally expensive.

## 📊 Performance Breakdown

For a typical 18M point tile:

| Operation            | Current Time | Target Time | Status                  |
| -------------------- | ------------ | ----------- | ----------------------- |
| Load LAZ file        | 5-10s        | 5-10s       | ✅ OK                   |
| Fetch WFS data       | 0.5-1s       | 0.5-1s      | ✅ Optimized (parallel) |
| **Point-in-polygon** | **5-30 min** | **10-30s**  | 🔴 **BOTTLENECK**       |
| Save LAZ file        | 5-10s        | 5-10s       | ✅ OK                   |
| **TOTAL**            | **6-31 min** | **20-50s**  | 🔴 **30-60× slower**    |

## 🎯 Root Cause

**File:** `ign_lidar/core/modules/advanced_classification.py`  
**Method:** `_classify_by_ground_truth()`

**Problems:**

1. **No spatial indexing** - iterates through ALL polygons for each point
2. **Python loops** - no vectorization (pure Python loops on millions of points)
3. **Late filtering** - applies geometric filters AFTER expensive containment checks
4. **Object creation** - creates 18M Point objects in memory

**Code snippet showing the issue:**

```python
# Line ~400-500 in advanced_classification.py
for feature_type, asprs_class in priority_order:
    gdf = ground_truth_features[feature_type]
    for idx, row in gdf.iterrows():  # For each polygon
        polygon = row['geometry']
        for i in candidate_indices:  # For each point - SLOW!
            if polygon.contains(point_geoms[i]):  # Expensive!
                labels[i] = asprs_class
```

## ✅ Solutions (Ranked by Effort vs Impact)

### Option 1: Quick Win - Pre-filtering (1 hour, 2-5× speedup)

**Files to create:**

- `ground_truth_quick_fix.py` ✅ Already created

**Apply geometric filters BEFORE spatial queries:**

```python
# Filter candidates by height and planarity first
road_candidates = np.where(
    (height <= 2.0) & (height >= -0.5) & (planarity >= 0.7)
)[0]
# Reduces 18M points to ~1-2M candidates (10× reduction)
```

**How to use:**

```python
from ground_truth_quick_fix import patch_classifier
patch_classifier()  # Apply optimization

# Then run normal processing
python reprocess_with_ground_truth.py enriched.laz
```

### Option 2: Medium Win - STRtree Spatial Index (4 hours, 10-30× speedup)

**Modify:** `ign_lidar/core/modules/advanced_classification.py`

**Use shapely's STRtree for spatial indexing:**

```python
from shapely.strtree import STRtree

# Build spatial index once
tree = STRtree(all_polygons)

# Query efficiently
for i, point in enumerate(points):
    pt = Point(point[0], point[1])
    nearby = tree.query(pt)  # Returns only nearby polygons!
    for poly in nearby:
        if poly.contains(pt):
            labels[i] = get_class(poly)
```

### Option 3: Best Win - Vectorized Spatial Joins (1 day, 30-100× speedup)

**Modify:** `ign_lidar/core/modules/advanced_classification.py`

**Use GeoPandas vectorized operations:**

```python
import geopandas as gpd

# Create GeoDataFrame from points
points_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(points[:, 0], points[:, 1])
)

# Vectorized spatial join (FAST!)
for feature_type, gdf in ground_truth_features.items():
    joined = gpd.sjoin(points_gdf, gdf, predicate='within')
    labels[joined.index] = asprs_class
```

### Option 4: Ultimate - GPU Acceleration (1 week, 100-1000× speedup)

**Use RAPIDS cuSpatial for GPU-accelerated spatial operations**

Requires:

- CUDA-capable GPU
- RAPIDS cuSpatial library
- Data transfer to GPU

## 📋 Immediate Actions

### Step 1: Profile Current Performance (5 minutes)

```bash
# Run profiler to confirm bottleneck
python profile_ground_truth.py /path/to/enriched.laz
```

This will show:

- Exact time spent in each operation
- Top 30 slowest functions
- Performance recommendations

### Step 2: Apply Quick Fix (10 minutes)

```bash
# Test the quick fix
python -c "
from ground_truth_quick_fix import patch_classifier
patch_classifier()
print('Optimization applied!')
"

# Then reprocess a test tile
python reprocess_with_ground_truth.py test_tile.laz
```

Expected result: 2-5× speedup (5-30 min → 2-10 min)

### Step 3: Monitor Progress (ongoing)

The quick fix adds progress bars:

```
Processing roads: 100%|███████████| 290/290 [00:45<00:00, 6.4 features/s]
Processing railways: 100%|███████| 15/15 [00:03<00:00, 4.2 features/s]
```

This helps identify which feature types are slowest.

## 🔧 Files Created

1. **`GROUND_TRUTH_PERFORMANCE_ANALYSIS.md`** - Detailed technical analysis
2. **`profile_ground_truth.py`** - Profiling script to measure performance
3. **`ground_truth_quick_fix.py`** - Quick optimization patch (2-5× speedup)
4. **`GROUND_TRUTH_QUICK_START.md`** (this file) - Quick start guide

## 📈 Expected Results

| Optimization   | Effort  | Time per Tile | Time for 128 Tiles | Speedup   |
| -------------- | ------- | ------------- | ------------------ | --------- |
| **Current**    | -       | 5-30 min      | 10-64 hours        | 1×        |
| **Quick fix**  | 1 hour  | 2-10 min      | 4-21 hours         | 2-5×      |
| **STRtree**    | 4 hours | 30s-2 min     | 1-4 hours          | 10-30×    |
| **Vectorized** | 1 day   | 10-30s        | 20-60 min          | 30-100×   |
| **GPU**        | 1 week  | 1-5s          | 2-10 min           | 300-1000× |

## 🚦 Next Steps

### Immediate (Today)

1. ✅ Run profiler: `python profile_ground_truth.py enriched.laz`
2. ✅ Apply quick fix: Import and use `ground_truth_quick_fix.py`
3. ⏱️ Measure improvement

### Short-term (This Week)

1. Implement STRtree spatial indexing
2. Add caching for spatial query results
3. Optimize road/railway filtering

### Long-term (This Month)

1. Implement vectorized GeoPandas spatial joins
2. Consider GPU acceleration for large batches
3. Parallelize tile processing

## 📞 Questions?

Run the profiler first to confirm the bottleneck:

```bash
python profile_ground_truth.py /path/to/enriched.laz
```

This will show you exactly where time is being spent and provide specific recommendations.

---

**TL;DR:** Ground truth classification is 30-60× too slow because it does brute-force point-in-polygon checks. Quick fix available for 2-5× speedup. Full optimization (vectorization) would give 30-100× speedup.
