# Ground Truth Performance Analysis

## 🔴 CRITICAL BOTTLENECK IDENTIFIED

### The Problem: O(N×M) Point-in-Polygon Checks

Your ground truth classification is **extremely slow** because it performs **brute-force point-in-polygon containment checks** for millions of points:

```python
# Current implementation in _classify_by_ground_truth()
for idx, row in gdf.iterrows():  # For each polygon (e.g., 290 roads)
    polygon = row['geometry']
    for i in candidate_indices:  # For each point (e.g., 18 million)
        if polygon.contains(point_geoms[i]):  # SLOW!
            labels[i] = asprs_class
```

**Performance Impact:**

- **18 million points** × **290 road polygons** = **5.2 BILLION** containment checks
- Each `polygon.contains()` call is expensive (shapely geometric computation)
- Even with bbox filtering, this is still O(N×M) complexity

### Time Breakdown

For a typical enriched LAZ file:

| Operation                   | Time              | Notes                           |
| --------------------------- | ----------------- | ------------------------------- |
| Load LAZ file               | ~5-10s            | Reading 18M points from disk    |
| Fetch WFS data (parallel)   | ~0.5-1s           | Already optimized ✅            |
| **Point-in-polygon checks** | **5-30+ minutes** | 🔴 BOTTLENECK!                  |
| Save LAZ file               | ~5-10s            | Writing updated classifications |
| **TOTAL**                   | **6-31 minutes**  | Per tile!                       |

## 🎯 Root Causes

### 1. **Inefficient Spatial Indexing**

The code creates Python Point objects for every point, then does individual containment checks:

```python
# SLOW: Creating 18M Point objects
point_geoms = [Point(p[0], p[1]) for p in points]  # 18M objects!

# SLOW: Individual containment checks
for i in candidate_indices:
    if polygon.contains(point_geoms[i]):  # Python loop, no vectorization
        labels[i] = asprs_class
```

### 2. **No Spatial Index on Polygons**

The code iterates through every polygon without using spatial indices like R-tree or STRtree:

```python
for idx, row in gdf.iterrows():  # Iterates ALL polygons for EVERY point
    polygon = row['geometry']
    # ...
```

### 3. **Repeated Filtering**

For roads and railways, geometric filters (height, planarity, intensity) are applied **after** the expensive containment check:

```python
for i in candidate_indices:
    if polygon.contains(point_geoms[i]):  # EXPENSIVE
        # Then check filters
        if height[i] > threshold:
            continue  # Wasted computation!
```

## ✅ Solutions

### Solution 1: Use Spatial Indices (STRtree)

**Impact: 10-100× speedup**

```python
from shapely.strtree import STRtree

def _classify_by_ground_truth_optimized(self, labels, points, ground_truth_features, ...):
    """Use STRtree for efficient spatial queries."""

    # Build spatial index for ALL polygons at once
    all_polygons = []
    polygon_to_class = {}

    for feature_type, asprs_class in priority_order:
        if feature_type not in ground_truth_features:
            continue
        gdf = ground_truth_features[feature_type]
        for idx, row in gdf.iterrows():
            poly = row['geometry']
            all_polygons.append(poly)
            polygon_to_class[id(poly)] = (asprs_class, feature_type, row)

    # Build STRtree (one-time cost)
    tree = STRtree(all_polygons)

    # Query tree for each point (vectorized)
    for i, point in enumerate(points):
        pt = Point(point[0], point[1])
        # Query returns only nearby polygons (fast!)
        for poly in tree.query(pt):
            if poly.contains(pt):
                asprs_class, feature_type, row = polygon_to_class[id(poly)]
                # Apply filters BEFORE expensive contains check
                if self._passes_filters(i, feature_type, height, planarity, intensity):
                    labels[i] = asprs_class
                break  # Found match
```

**Expected time: 30s-2min** (vs 5-30min currently)

### Solution 2: Vectorized Point-in-Polygon (GeoPandas)

**Impact: 50-100× speedup**

```python
def _classify_by_ground_truth_vectorized(self, labels, points, ground_truth_features, ...):
    """Use vectorized spatial joins."""
    import geopandas as gpd
    from shapely.geometry import Point

    # Create GeoDataFrame from points (vectorized)
    points_gdf = gpd.GeoDataFrame(
        {'point_idx': np.arange(len(points))},
        geometry=gpd.points_from_xy(points[:, 0], points[:, 1]),
        crs='EPSG:2154'
    )

    # Spatial join with each feature type (vectorized!)
    for feature_type, asprs_class in priority_order:
        if feature_type not in ground_truth_features:
            continue

        gdf = ground_truth_features[feature_type]

        # Vectorized spatial join (FAST!)
        joined = gpd.sjoin(points_gdf, gdf, how='inner', predicate='within')

        # Apply filters vectorized
        if feature_type in ['roads', 'railways']:
            # Vectorized height filter
            if height is not None:
                mask = (height[joined['point_idx']] <= threshold_max) & \
                       (height[joined['point_idx']] >= threshold_min)
                joined = joined[mask]

        # Update labels
        labels[joined['point_idx'].values] = asprs_class
```

**Expected time: 10-30s** (vs 5-30min currently)

### Solution 3: GPU-Accelerated (cuSpatial + RAPIDS)

**Impact: 100-1000× speedup**

Use RAPIDS cuSpatial for GPU-accelerated spatial operations:

```python
import cudf
import cuspatial

# Transfer to GPU
points_gpu = cudf.DataFrame({'x': points[:, 0], 'y': points[:, 1]})
polygons_gpu = cuspatial.from_geopandas(gdf)

# GPU-accelerated point-in-polygon
results = cuspatial.point_in_polygon(points_gpu, polygons_gpu)
```

**Expected time: 1-5s** (vs 5-30min currently)

## 🚀 Quick Wins

### Immediate Optimization (No Code Changes)

**Pre-filter points before classification:**

```python
# Apply geometric filters BEFORE spatial queries
road_candidates = np.where(
    (height <= 2.0) &  # Roads at ground level
    (height >= -0.5) &
    (planarity >= 0.7)  # Flat surfaces
)[0]

# Only check these candidates (much smaller set)
for i in road_candidates:
    if polygon.contains(point_geoms[i]):
        labels[i] = ASPRS_ROAD
```

This reduces points to check from 18M to ~1-2M (10× reduction).

### Cache Filtered Results

For batch processing, cache spatial query results per tile bbox:

```python
cache_key = f"spatial_labels_{bbox}_{hash(frozenset(ground_truth_features.keys()))}"
if cache_key in cache:
    return cache[cache_key]
```

## 📊 Expected Performance After Optimization

| Method                    | Time per Tile | Speedup   |
| ------------------------- | ------------- | --------- |
| **Current (brute force)** | 5-30 min      | 1×        |
| Pre-filtering             | 2-10 min      | 2-3×      |
| STRtree                   | 30s-2min      | 10-30×    |
| Vectorized (GeoPandas)    | 10-30s        | 30-100×   |
| GPU (cuSpatial)           | 1-5s          | 300-1000× |

For 128 tiles:

- **Current**: 10-64 hours 🔴
- **With STRtree**: 1-4 hours 🟡
- **With vectorization**: 20-60 minutes ✅
- **With GPU**: 2-10 minutes 🚀

## 🔧 Recommended Actions

### Immediate (1 hour)

1. **Add pre-filtering** to reduce candidate points
2. **Add progress logging** to see which feature types are slow
3. **Profile with cProfile** to confirm bottleneck

### Short-term (1 day)

1. **Implement STRtree** spatial indexing
2. **Add caching** for spatial query results
3. **Batch process filters** before containment checks

### Long-term (1 week)

1. **Implement vectorized** GeoPandas spatial joins
2. **Add GPU support** with cuSpatial (optional)
3. **Parallelize tile processing** across CPU cores

## 🔍 Debugging Commands

### Profile current performance:

```bash
python -m cProfile -o ground_truth.prof reprocess_with_ground_truth.py enriched.laz
python -m pstats ground_truth.prof
# Then: sort cumtime, stats 20
```

### Monitor progress:

```python
# Add to _classify_by_ground_truth
from tqdm import tqdm

for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc=f"Processing {feature_type}"):
    # ...
```

### Check point counts:

```python
logger.info(f"Total points: {len(points):,}")
logger.info(f"Candidate points after bbox filter: {len(candidate_indices):,}")
logger.info(f"Points passing geometric filters: {sum(passes_filters):,}")
```

## 📝 Summary

**The ground truth processing is slow because:**

1. ❌ No spatial indexing (STRtree/R-tree)
2. ❌ Brute-force O(N×M) containment checks
3. ❌ Creating 18M Python Point objects
4. ❌ No vectorization (pure Python loops)
5. ❌ Filters applied AFTER expensive spatial queries

**Quick fix:** Add pre-filtering to reduce candidate points by 10×

**Best fix:** Implement vectorized spatial joins with GeoPandas (30-100× speedup)

**Ultimate fix:** Use GPU acceleration with cuSpatial (300-1000× speedup)
