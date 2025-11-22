# Ground Truth v2.0 Migration Guide

**GroundTruthHub - Unified API for Ground Truth Operations**

This guide helps you migrate from the v1.x ground truth API (4 separate classes) to the new v2.0 unified hub API.

**Version**: 2.0.0  
**Date**: November 22, 2025  
**Breaking Changes**: None - Full backward compatibility maintained

---

## Table of Contents

1. [Overview](#overview)
2. [What's New](#whats-new)
3. [Migration Paths](#migration-paths)
4. [API Comparison](#api-comparison)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Overview

### Why v2.0?

**Problem in v1.x**: 4 separate classes with unclear responsibilities and 3 different caching implementations.

```python
# v1.x - Confusing, scattered API
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
from ign_lidar.core.ground_truth_manager import GroundTruthManager
from ign_lidar.core.classification.ground_truth_refinement import GroundTruthRefiner

fetcher = IGNGroundTruthFetcher(config)
optimizer = GroundTruthOptimizer()
manager = GroundTruthManager()
refiner = GroundTruthRefiner(config)
```

**Solution in v2.0**: Single unified hub with composition pattern (same as GPU Manager v3.1).

```python
# v2.0 - Unified, clear API
from ign_lidar.core import ground_truth

# High-level convenience methods
labels = ground_truth.fetch_and_label(tile_path, points)

# Low-level component access (when needed)
buildings = ground_truth.fetcher.fetch_buildings(bbox)
labels = ground_truth.optimizer.label_points(points, buildings)
```

### Key Benefits

✅ **Single Entry Point** - Import once: `from ign_lidar.core import ground_truth`  
✅ **Lazy Loading** - Components only created when accessed (performance)  
✅ **Unified Caching** - One cache implementation (not 3)  
✅ **Backward Compatible** - All old code continues to work  
✅ **Clear Hierarchy** - HIGH-LEVEL hub → LOW-LEVEL components  
✅ **Easier Testing** - Mock entire hub or individual components

---

## What's New

### New GroundTruthHub Class

```python
from ign_lidar.core import GroundTruthHub, ground_truth

# Singleton instance (one per process)
hub = GroundTruthHub()  # Same as 'ground_truth'

# Lazy-loaded properties
hub.fetcher    # IGNGroundTruthFetcher
hub.optimizer  # GroundTruthOptimizer
hub.manager    # GroundTruthManager
hub.refiner    # GroundTruthRefiner

# Convenience methods (NEW)
hub.fetch_and_label(tile_path, points)
hub.prefetch_batch(tile_paths)
hub.process_tile_complete(tile_path, points, features)
hub.clear_all_caches()
hub.get_statistics()
```

### Architecture

```
GroundTruthHub (HIGH-LEVEL)
├── .fetcher  → IGNGroundTruthFetcher (lazy-loaded)
├── .optimizer → GroundTruthOptimizer (lazy-loaded)
├── .manager  → GroundTruthManager (lazy-loaded)
└── .refiner  → GroundTruthRefiner (lazy-loaded)
```

---

## Migration Paths

### Path 1: Quick Migration (Recommended)

**Use new convenience methods** - Easiest migration path.

```python
# OLD v1.x
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

fetcher = IGNGroundTruthFetcher()
data = fetcher.fetch_all_features(bbox)
optimizer = GroundTruthOptimizer()
labels = optimizer.label_points(points, data)

# NEW v2.0 (single call)
from ign_lidar.core import ground_truth

labels, metadata = ground_truth.fetch_and_label(tile_path, points)
```

### Path 2: Component Access

**Use hub properties** - When you need fine-grained control.

```python
# OLD v1.x
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

fetcher = IGNGroundTruthFetcher()
optimizer = GroundTruthOptimizer()

# NEW v2.0 (via hub)
from ign_lidar.core import ground_truth

# Access components through hub properties
fetcher = ground_truth.fetcher
optimizer = ground_truth.optimizer
```

### Path 3: No Changes (Backward Compatible)

**Keep existing code** - Works with deprecation warnings.

```python
# OLD v1.x (still works in v2.0)
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

fetcher = IGNGroundTruthFetcher()  # ⚠️ Deprecation warning
optimizer = GroundTruthOptimizer()  # ✅ No warning (no change)

# Works exactly as before
```

**Note**: Deprecation warnings guide you to new API. Old imports will be removed in v4.0.0 (12+ months).

---

## API Comparison

### Fetching Ground Truth

#### v1.x - Direct Class Usage

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

fetcher = IGNGroundTruthFetcher(config)
buildings = fetcher.fetch_buildings(bbox)
roads = fetcher.fetch_roads_with_polygons(bbox)
all_features = fetcher.fetch_all_features(bbox)
```

#### v2.0 - Via Hub Property

```python
from ign_lidar.core import ground_truth

# Same methods, accessed through hub
buildings = ground_truth.fetcher.fetch_buildings(bbox)
roads = ground_truth.fetcher.fetch_roads_with_polygons(bbox)
all_features = ground_truth.fetcher.fetch_all_features(bbox)
```

### Labeling Points

#### v1.x - Manual Coordination

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

# Step 1: Fetch data
fetcher = IGNGroundTruthFetcher()
data = fetcher.fetch_all_features(bbox)

# Step 2: Label points
optimizer = GroundTruthOptimizer(force_method="strtree")
labels = optimizer.label_points(points, data)
```

#### v2.0 - Unified Convenience Method

```python
from ign_lidar.core import ground_truth

# Single call handles fetch + label
labels, metadata = ground_truth.fetch_and_label(
    tile_path="tile.laz",
    points=points
)

print(f"Labeled {metadata['n_labeled']} points")
```

### Prefetching for Multiple Tiles

#### v1.x - Manual Loop

```python
from ign_lidar.core.ground_truth_manager import GroundTruthManager

manager = GroundTruthManager(data_sources_config)

for tile_path in tile_paths:
    manager.prefetch_ground_truth_for_tile(tile_path)
```

#### v2.0 - Batch Method

```python
from ign_lidar.core import ground_truth

# Parallel prefetch
stats = ground_truth.prefetch_batch(
    tile_paths=tile_paths,
    num_workers=8
)

print(f"Prefetched {stats['n_success']}/{stats['n_tiles']} tiles")
```

### Complete Pipeline

#### v1.x - Manual Steps

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
from ign_lidar.core.classification.ground_truth_refinement import GroundTruthRefiner

# Step 1: Fetch
fetcher = IGNGroundTruthFetcher()
data = fetcher.fetch_all_features(bbox)

# Step 2: Label
optimizer = GroundTruthOptimizer()
labels = optimizer.label_points(points, data)

# Step 3: Refine
refiner = GroundTruthRefiner(config)
labels_refined, meta = refiner.refine_all(points, labels, features)
```

#### v2.0 - Single Method

```python
from ign_lidar.core import ground_truth

# Complete pipeline: fetch → label → refine
labels, stats = ground_truth.process_tile_complete(
    tile_path="tile.laz",
    points=points,
    features=features,
    refine=True
)

print(f"Processing time: {stats['duration']:.2f}s")
```

### Cache Management

#### v1.x - Scattered Cache Clearing

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
from ign_lidar.core.ground_truth_manager import GroundTruthManager

# Clear each cache separately
fetcher._cache.clear()
optimizer.clear_cache()
manager.clear_cache()
```

#### v2.0 - Unified Cache Clearing

```python
from ign_lidar.core import ground_truth

# Clear all caches at once
cleared = ground_truth.clear_all_caches()

print(f"Cleared {sum(cleared.values())} cache entries")
```

---

## Examples

### Example 1: Simple Tile Processing

```python
from ign_lidar.core import ground_truth
import laspy

# Load tile
las = laspy.read("tile_0500_6250.laz")
points = np.vstack([las.x, las.y, las.z]).T

# Fetch and label (single call)
labels, metadata = ground_truth.fetch_and_label(
    tile_path="tile_0500_6250.laz",
    points=points,
    feature_types=['buildings', 'roads']
)

print(f"Labeled {metadata['label_rate']:.1%} of points")
print(f"Classes: {metadata['unique_labels']}")
```

### Example 2: Batch Processing

```python
from ign_lidar.core import ground_truth
from pathlib import Path

# Get all tiles
tile_dir = Path("data/tiles")
tile_paths = list(tile_dir.glob("*.laz"))

# Prefetch ground truth for all tiles (parallel)
stats = ground_truth.prefetch_batch(
    tile_paths=[str(p) for p in tile_paths],
    feature_types=['buildings', 'roads', 'vegetation'],
    num_workers=8
)

print(f"Prefetched: {stats['n_success']}/{stats['n_tiles']} tiles")
print(f"Cache hits: {stats['cache_hits']}")

# Now process each tile (will use cached data)
for tile_path in tile_paths:
    las = laspy.read(tile_path)
    points = np.vstack([las.x, las.y, las.z]).T

    labels, meta = ground_truth.fetch_and_label(tile_path, points)
    print(f"{tile_path.name}: {meta['n_labeled']} points labeled")
```

### Example 3: Custom Component Configuration

```python
from ign_lidar.core import ground_truth

# Access and configure individual components
ground_truth.optimizer.force_method = "gpu"  # Force GPU mode
ground_truth.optimizer.gpu_chunk_size = 5_000_000  # 5M points per chunk

ground_truth.fetcher.verbose = True  # Enable verbose logging

# Use configured components
labels = ground_truth.optimizer.label_points(points, polygons)
```

### Example 4: Complete Pipeline with Refinement

```python
from ign_lidar.core import ground_truth
from ign_lidar.features import compute_all_features_optimized
import laspy

# Load tile
las = laspy.read("tile.laz")
points = np.vstack([las.x, las.y, las.z]).T

# Compute features for refinement
features = compute_all_features_optimized(points, k_neighbors=30)

# Complete pipeline: fetch → label → refine
labels, stats = ground_truth.process_tile_complete(
    tile_path="tile.laz",
    points=points,
    features=features,
    refine=True
)

print(f"Labeled {stats['n_labeled']} points in {stats['duration']:.2f}s")
print(f"Refinement applied: {stats['refined']}")
if stats['refined']:
    print(f"Refined {stats['refine_metadata']['n_refined']} points")
```

### Example 5: Low-Level Component Access

```python
from ign_lidar.core import ground_truth

# When you need fine-grained control
bbox = (500000, 6250000, 501000, 6251000)

# Step 1: Fetch specific features
buildings = ground_truth.fetcher.fetch_buildings(bbox)
roads = ground_truth.fetcher.fetch_roads_with_polygons(bbox)

# Step 2: Label with specific method
labels_buildings = ground_truth.optimizer.label_points(
    points=points,
    geometries={'buildings': buildings},
    use_cache=True
)

labels_roads = ground_truth.optimizer.label_points(
    points=points,
    geometries={'roads': roads},
    use_cache=True
)

# Combine labels (custom logic)
combined_labels = np.maximum(labels_buildings, labels_roads)

# Step 3: Refine specific classes
refined_labels, meta = ground_truth.refiner.refine_building_with_expanded_polygons(
    points=points,
    labels=combined_labels,
    features=features
)
```

---

## Troubleshooting

### Issue 1: ImportError after upgrade

**Problem**:

```python
ImportError: cannot import name 'ground_truth' from 'ign_lidar.core'
```

**Solution**: Upgrade to latest version and ensure proper import:

```bash
pip install --upgrade ign-lidar-hd
```

```python
# Correct import
from ign_lidar.core import ground_truth  # v2.0+

# Old import (still works with warning)
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher  # v1.x
```

### Issue 2: Components not lazy loading

**Problem**: All components instantiated even if not used.

**Solution**: Check if you're accessing properties unnecessarily:

```python
# ❌ Bad - instantiates all components
hub = GroundTruthHub()
print(hub.fetcher, hub.optimizer, hub.manager, hub.refiner)

# ✅ Good - only instantiates what you use
hub = GroundTruthHub()
labels = hub.optimizer.label_points(points, polygons)  # Only creates optimizer
```

### Issue 3: Cache not cleared

**Problem**: `clear_all_caches()` doesn't clear my component's cache.

**Solution**: Component must be instantiated first:

```python
# ❌ Bad - optimizer not instantiated yet
ground_truth.clear_all_caches()  # Won't clear optimizer cache

# ✅ Good - use optimizer first
_ = ground_truth.optimizer.label_points(points, polygons)
ground_truth.clear_all_caches()  # Now clears optimizer cache
```

### Issue 4: Deprecation warnings

**Problem**: Getting deprecation warnings for old imports.

**Solution**: Migrate to new API (warnings won't break code):

```python
# ⚠️ Old import (works but deprecated)
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
fetcher = IGNGroundTruthFetcher()

# ✅ New import (no warnings)
from ign_lidar.core import ground_truth
fetcher = ground_truth.fetcher
```

### Issue 5: Performance regression

**Problem**: Code seems slower after migrating to v2.0.

**Solution**: Ensure you're not re-instantiating hub:

```python
# ❌ Bad - creates new hub each iteration
for tile in tiles:
    hub = GroundTruthHub()  # Don't do this!
    labels = hub.fetch_and_label(tile, points)

# ✅ Good - reuse singleton
from ign_lidar.core import ground_truth

for tile in tiles:
    labels, _ = ground_truth.fetch_and_label(tile, points)
```

---

## FAQ

### Q1: Do I need to change my code?

**A**: No, but it's recommended. All old imports continue to work with deprecation warnings.

### Q2: When will old imports be removed?

**A**: Old imports will be removed in v4.0.0 (estimated 12+ months from now). You have plenty of time to migrate.

### Q3: Is there a performance difference?

**A**: No performance difference. Lazy loading may actually improve performance for partial usage.

### Q4: Can I mix old and new APIs?

**A**: Yes, but not recommended. Pick one approach for consistency:

```python
# ✅ Consistent (new API)
from ign_lidar.core import ground_truth
labels = ground_truth.fetch_and_label(tile, points)

# ⚠️ Inconsistent (mixing old and new)
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
from ign_lidar.core import ground_truth
optimizer = GroundTruthOptimizer()  # Why not use ground_truth.optimizer?
```

### Q5: How do I test with the new API?

**A**: Mock the hub or individual components:

```python
from unittest.mock import Mock, patch

# Option 1: Mock entire hub
with patch('ign_lidar.core.ground_truth') as mock_hub:
    mock_hub.fetch_and_label.return_value = (labels, metadata)
    # Your test code

# Option 2: Mock individual component
with patch('ign_lidar.core.ground_truth.optimizer') as mock_optimizer:
    mock_optimizer.label_points.return_value = labels
    # Your test code
```

### Q6: Does this affect GPU usage?

**A**: No, GPU functionality unchanged. GroundTruthOptimizer still has GPU support:

```python
from ign_lidar.core import ground_truth

# GPU still works as before
ground_truth.optimizer.force_method = "gpu"
labels = ground_truth.optimizer.label_points(points, polygons)
```

### Q7: What about configuration?

**A**: Component configuration unchanged, just accessed through hub:

```python
# v1.x
optimizer = GroundTruthOptimizer(
    force_method="strtree",
    enable_cache=True,
    max_cache_size_mb=1000
)

# v2.0 - same configuration
ground_truth.optimizer.force_method = "strtree"
ground_truth.optimizer.enable_cache = True
ground_truth.optimizer.max_cache_size_mb = 1000
```

### Q8: Can I still instantiate components directly?

**A**: Yes, but using the hub is recommended:

```python
# ✅ Recommended (via hub)
from ign_lidar.core import ground_truth
optimizer = ground_truth.optimizer

# ✅ Still works (direct instantiation)
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
optimizer = GroundTruthOptimizer()
```

---

## Migration Checklist

Use this checklist to track your migration progress:

### Phase 1: Assessment

- [ ] Identify all ground truth imports in codebase
- [ ] List all usages of IGNGroundTruthFetcher
- [ ] List all usages of GroundTruthOptimizer
- [ ] List all usages of GroundTruthManager
- [ ] List all usages of GroundTruthRefiner

### Phase 2: Update Imports

- [ ] Replace `from ign_lidar.io.wfs_ground_truth import` with `from ign_lidar.core import ground_truth`
- [ ] Update instantiation to use hub properties
- [ ] Run tests to verify functionality

### Phase 3: Use Convenience Methods

- [ ] Identify fetch + label patterns → replace with `fetch_and_label()`
- [ ] Identify batch prefetch patterns → replace with `prefetch_batch()`
- [ ] Identify full pipeline patterns → replace with `process_tile_complete()`

### Phase 4: Cleanup

- [ ] Remove deprecated imports
- [ ] Update documentation
- [ ] Update tests
- [ ] Verify no deprecation warnings

---

## Additional Resources

- **Design Document**: `docs/GROUND_TRUTH_CONSOLIDATION_DESIGN.md`
- **Test Examples**: `tests/test_ground_truth_hub.py`
- **API Reference**: See docstrings in `ign_lidar/core/ground_truth_hub.py`
- **GPU Manager Migration** (similar pattern): `docs/GPU_MANAGER_V3.1_MIGRATION.md`

---

## Support

If you encounter issues or have questions:

1. **Check this guide** - Most common issues are covered above
2. **Check examples** - See `tests/test_ground_truth_hub.py` for working code
3. **GitHub Issues** - Report bugs at https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues

---

**Version**: 2.0.0  
**Last Updated**: November 22, 2025  
**Status**: ✅ Production Ready
