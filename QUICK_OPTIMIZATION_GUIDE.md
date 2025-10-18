# 🚀 Quick Optimization Guide - Action Items

**Date**: October 18, 2025  
**Priority**: Immediate Performance Improvements  
**Expected Gain**: +20-50% throughput in 4-8 hours of work

---

## 🎯 Top 3 Bottlenecks to Fix NOW

### 1. 🔴 DataFrame `.iterrows()` Anti-Pattern (CRITICAL)

**Impact**: 10-100× slower than vectorized operations  
**Effort**: 2-4 hours  
**Risk**: Low  
**Expected Gain**: +20-40% for ground truth processing

#### Files to Fix (in priority order):

1. **`ign_lidar/optimization/strtree.py:199`** (HIGHEST IMPACT)

   - Used in spatial indexing for ground truth
   - Processes every polygon in tile bounding box
   - Fix = 10-50× speedup

2. **`ign_lidar/core/modules/transport_enhancement.py`**

   - Lines: 327, 387, 478, 510
   - Road and railway buffering operations
   - Fix = 5-20× speedup

3. **`ign_lidar/io/wfs_ground_truth.py`**

   - Lines: 209, 307, 611, 946, 1039
   - Ground truth feature processing
   - Fix = 10-30× speedup

4. **`ign_lidar/optimization/*.py`** (Multiple files)
   - `prefilter.py:126`
   - `ground_truth.py:346`
   - `optimizer.py:479, 682`
   - `gpu_optimized.py:225`
   - `cpu_optimized.py:341`

#### Quick Fix Template:

```python
# ❌ SLOW (current code)
for idx, row in gdf.iterrows():
    geom = row['geometry']
    result = process(geom, row['attribute'])

# ✅ FAST (vectorized)
# Option 1: Use .apply() for complex operations
gdf['result'] = gdf.apply(lambda row: process(row.geometry, row['attribute']), axis=1)

# Option 2: Use itertuples() if you must iterate (5× faster than iterrows)
for row in gdf.itertuples():
    geom = row.geometry
    result = process(geom, row.attribute)

# Option 3: Vectorize completely (best)
geometries = gdf.geometry.values
attributes = gdf['attribute'].values
results = vectorized_process(geometries, attributes)
```

---

### 2. 🟡 Hierarchical Classifier Loop (MEDIUM)

**Impact**: 50-100× slower for point classification  
**Effort**: 1-2 hours  
**Risk**: Low  
**Expected Gain**: +5-10% overall

#### File to Fix:

**`ign_lidar/core/modules/hierarchical_classifier.py:326`**

```python
# ❌ SLOW (current)
for i in range(n_points):
    if condition[i]:
        labels[i] = new_value

# ✅ FAST (vectorized)
mask = condition  # Boolean array
labels[mask] = new_value

# Or for complex logic:
labels = np.where(condition, new_value, labels)
```

---

### 3. 🟢 Memory Cleanup Between Operations (LOW-MEDIUM)

**Impact**: Reduces OOM errors, smoother processing  
**Effort**: 1 hour  
**Risk**: None  
**Expected Gain**: +5-10% stability

#### File to Enhance:

**`ign_lidar/features/features_gpu.py`**

```python
# Add proactive cleanup between major operations
def compute_features_with_cleanup(self, points, k=20):
    # Compute normals
    normals = self.compute_normals(points, k)

    # Cleanup between operations
    if self.use_gpu and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()

    # Compute next feature
    curvature = self.compute_curvature(points, normals, k)

    return normals, curvature
```

---

## 📋 Step-by-Step Implementation Plan

### Step 1: Setup & Baseline (15 minutes)

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

# Run current benchmark
python scripts/benchmark_bottleneck_fixes.py > baseline_before.txt

# Or run on actual data
time ign-lidar-hd process \
  -c ign_lidar/configs/presets/asprs_rtx4080.yaml \
  input_dir="/path/to/test/tiles" \
  output_dir="/path/to/output" \
  2>&1 | tee baseline_before.log
```

### Step 2: Fix `strtree.py` (30-60 minutes)

**File**: `ign_lidar/optimization/strtree.py`

1. **Backup original**:

   ```bash
   cp ign_lidar/optimization/strtree.py ign_lidar/optimization/strtree.py.backup
   ```

2. **Find the loop** (around line 199):

   ```python
   for idx, row in gdf.iterrows():
       polygon = row['geometry']
       # ... processing ...
   ```

3. **Replace with vectorized version**:

   ```python
   # Filter valid geometries (vectorized)
   valid_mask = gdf.geometry.apply(lambda g: isinstance(g, (Polygon, MultiPolygon)))
   valid_gdf = gdf[valid_mask].copy()

   # Apply buffer if needed (vectorized)
   if feature_type == 'roads' and self.road_buffer_tolerance > 0:
       valid_gdf['geometry'] = valid_gdf.geometry.buffer(self.road_buffer_tolerance)

   # Prepare geometries
   geometries = valid_gdf.geometry.values
   prepared_geoms = [prep(g) if self.use_prepared_geometries else None
                     for g in geometries]

   # Build metadata (minimal iteration)
   for (idx, row), prepared_geom in zip(valid_gdf.iterrows(), prepared_geoms):
       polygon = row['geometry']
       metadata = PolygonMetadata(
           feature_type=feature_type,
           asprs_class=asprs_class,
           properties=dict(row),
           prepared_geom=prepared_geom
       )
       all_polygons.append(polygon)
       metadata_map[id(polygon)] = metadata
   ```

4. **Test**:

   ```bash
   # Run unit tests
   pytest tests/test_ground_truth_optimizer.py -v

   # Run validation
   python scripts/validate_bottleneck_fixes.py
   ```

### Step 3: Fix `transport_enhancement.py` (30-45 minutes)

**File**: `ign_lidar/core/modules/transport_enhancement.py`

1. **Find road buffering loop** (line ~327):

   ```python
   for idx, row in roads_gdf.iterrows():
       geometry = row['geometry']
       # ... buffering ...
   ```

2. **Replace with**:

   ```python
   # Filter LineStrings (vectorized)
   line_mask = roads_gdf.geometry.apply(lambda g: isinstance(g, LineString))
   roads_lines = roads_gdf[line_mask].copy()

   # Vectorized buffer
   def buffer_road(row):
       base_width = row.get('width_m', 4.0)
       return adaptive_buffer(row['geometry'], base_width, self.config)

   enhanced_geometries = roads_lines.apply(buffer_road, axis=1)
   enhanced_roads = enhanced_geometries.tolist()
   ```

3. **Repeat for railway buffering** (line ~387, similar pattern)

4. **Test**:
   ```bash
   pytest tests/test_enriched_save.py -v
   ```

### Step 4: Fix `hierarchical_classifier.py` (20-30 minutes)

**File**: `ign_lidar/core/modules/hierarchical_classifier.py`

1. **Find point-by-point loop** (line ~326)

2. **Replace with vectorized operations** using NumPy boolean indexing

3. **Test**:
   ```bash
   pytest tests/ -k hierarchical -v
   ```

### Step 5: Benchmark & Compare (15 minutes)

```bash
# Run benchmark again
python scripts/benchmark_bottleneck_fixes.py > baseline_after.txt

# Compare
diff baseline_before.txt baseline_after.txt

# Or run on actual data
time ign-lidar-hd process \
  -c ign_lidar/configs/presets/asprs_rtx4080.yaml \
  input_dir="/path/to/test/tiles" \
  output_dir="/path/to/output" \
  2>&1 | tee baseline_after.log
```

### Step 6: Commit Changes (10 minutes)

```bash
# Run full test suite
pytest tests/ -v

# If all pass, commit
git add ign_lidar/optimization/strtree.py
git add ign_lidar/core/modules/transport_enhancement.py
git add ign_lidar/core/modules/hierarchical_classifier.py
git commit -m "perf: vectorize DataFrame operations for 20-50% speedup

- Replace .iterrows() with vectorized operations in strtree.py
- Vectorize road/railway buffering in transport_enhancement.py
- Vectorize point classification in hierarchical_classifier.py

Expected performance improvement: +20-50% for ground truth processing
Closes #<issue_number>"
```

---

## 🧪 Testing Checklist

- [ ] Unit tests pass: `pytest tests/ -v`
- [ ] Integration tests pass: `pytest tests/test_integration_e2e.py -v`
- [ ] Benchmark shows improvement: `python scripts/benchmark_bottleneck_fixes.py`
- [ ] Manual test on sample tile completes successfully
- [ ] Memory usage remains stable (check logs)
- [ ] No regression in output quality (spot check LAZ files)

---

## 📊 Expected Results

### Before Optimization

```
Processing 18.6M point tile:
- Ground truth fetch: 8-12s
- Spatial indexing: 5-8s
- Feature computation: 12-14s
Total: ~25-34s per tile
```

### After Optimization

```
Processing 18.6M point tile:
- Ground truth fetch: 2-4s (3-4× faster ✅)
- Spatial indexing: 1-2s (4-5× faster ✅)
- Feature computation: 12-14s (unchanged)
Total: ~15-20s per tile (40-50% improvement! 🎉)
```

---

## 🚨 Troubleshooting

### Issue: Tests fail after changes

**Solution**:

```bash
# Revert specific file
git checkout ign_lidar/optimization/strtree.py

# Debug with minimal example
python -c "
import geopandas as gpd
from shapely.geometry import Point
gdf = gpd.GeoDataFrame({'geometry': [Point(0, 0)]})
print(gdf.apply(lambda row: row.geometry.buffer(1), axis=1))
"
```

### Issue: Performance not improved

**Solution**:

```bash
# Profile specific function
python -m cProfile -s cumulative -o profile.stats scripts/validate_bottleneck_fixes.py
python -c "import pstats; pstats.Stats('profile.stats').print_stats(30)"

# Check if vectorization is actually being used
python -c "
import numpy as np
import pandas as pd
# Your test case here
"
```

### Issue: Memory usage increased

**Solution**:

- Check if you're creating unnecessary copies with `.copy()`
- Use `.loc[]` instead of boolean indexing when possible
- Add manual cleanup: `del temp_variable; gc.collect()`

---

## 📚 Resources

### Pandas Performance

- [Pandas Performance Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [GeoPandas Performance](https://geopandas.org/en/stable/docs/user_guide/io.html#performance)

### Profiling Tools

```bash
# Install profiling tools
pip install line_profiler memory_profiler

# Profile line-by-line
kernprof -l -v script.py

# Profile memory
mprof run script.py
mprof plot
```

### Example: Vectorized vs Loop

```python
import numpy as np
import time

# Setup
data = np.random.rand(1_000_000)

# ❌ Loop (slow)
start = time.time()
result = []
for x in data:
    result.append(x * 2 + 1)
loop_time = time.time() - start

# ✅ Vectorized (fast)
start = time.time()
result = data * 2 + 1
vectorized_time = time.time() - start

print(f"Loop: {loop_time:.3f}s")
print(f"Vectorized: {vectorized_time:.3f}s")
print(f"Speedup: {loop_time/vectorized_time:.1f}×")
# Typical output: Speedup: 100-1000×
```

---

## 🎯 Next Steps After Quick Wins

Once you've completed these optimizations and verified the improvements:

1. **Document** the performance gains in `OPTIMIZATION_IMPROVEMENTS.md`
2. **Update** benchmark baselines
3. **Consider** Phase 2 optimizations (see CODEBASE_AUDIT_REPORT.md)
4. **Share** results with team/community

---

## ✅ Success Criteria

You'll know you're successful when:

- ✅ Processing time reduced by 20-50%
- ✅ All tests pass
- ✅ No increase in memory usage
- ✅ Code is cleaner and more maintainable
- ✅ Benchmark results documented

**Good luck!** 🚀

---

**Questions or issues?**

- Check full audit report: `CODEBASE_AUDIT_REPORT.md`
- Review optimization docs: `OPTIMIZATION_IMPROVEMENTS.md`
- Run validation: `python scripts/validate_bottleneck_fixes.py`
