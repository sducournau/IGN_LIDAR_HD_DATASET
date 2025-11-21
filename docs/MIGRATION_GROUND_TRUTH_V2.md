# Migration Guide: GroundTruthOptimizer V2

## 📢 Important Changes (November 2025)

The `GroundTruthOptimizer` has been **consolidated and enhanced** with V2 features:

- **Old location**: `ign_lidar.io.ground_truth_optimizer` (deprecated)
- **New location**: `ign_lidar.optimization.ground_truth` (recommended)

All V2 cache features are now integrated into the main module.

---

## 🚀 What's New in V2

### 1. Intelligent Caching System

- **Spatial hashing**: Cache keys based on tile bounds and features
- **LRU eviction**: Automatic cache management when limits reached
- **30-50% speedup**: For repeated tiles in production pipelines
- **Disk persistence**: Optional disk cache for multi-session use

### 2. Batch Processing

- **Optimized for multiple tiles**: Process entire directories efficiently
- **Cache reuse**: Automatic detection of previously processed tiles
- **Reduced I/O overhead**: Shared spatial indexes across tiles

### 3. Enhanced API

- **Backward compatible**: All existing code continues to work
- **New parameters**: `enable_cache`, `cache_dir`, `max_cache_size_mb`
- **Cache statistics**: Monitor hit ratios and performance

---

## 📝 Migration Steps

### Step 1: Update Imports (Required by v4.0)

**Before (deprecated)**:

```python
from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer
```

**After (recommended)**:

```python
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
```

### Step 2: Enable Caching (Optional)

**Basic usage (cache enabled by default)**:

```python
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

# Memory-only cache (default)
optimizer = GroundTruthOptimizer(
    enable_cache=True,  # Default: True
    max_cache_size_mb=500,  # Default: 500MB
    max_cache_entries=100,  # Default: 100
    verbose=True
)

labels = optimizer.label_points(points, ground_truth_features)

# Check cache performance
stats = optimizer.get_cache_stats()
print(f"Cache hit ratio: {stats['hit_ratio']:.1%}")
```

**With disk cache (persistent across sessions)**:

```python
from pathlib import Path

optimizer = GroundTruthOptimizer(
    enable_cache=True,
    cache_dir=Path("cache/ground_truth"),  # Persistent cache
    max_cache_size_mb=1000,  # 1GB cache
    verbose=True
)

labels = optimizer.label_points(points, ground_truth_features)
```

**Disable cache (same as v1 behavior)**:

```python
optimizer = GroundTruthOptimizer(
    enable_cache=False,  # No caching
    verbose=True
)
```

### Step 3: Use Batch Processing (Optional)

**Process multiple tiles efficiently**:

```python
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer

optimizer = GroundTruthOptimizer(
    enable_cache=True,
    cache_dir=Path("cache/ground_truth")
)

# Prepare tiles (list of dicts)
tiles = [
    {'points': points1, 'ndvi': ndvi1},
    {'points': points2},  # NDVI optional
    {'points': points3, 'ndvi': ndvi3},
]

# Batch process (automatic cache reuse)
labels_list = optimizer.label_points_batch(
    tile_data_list=tiles,
    ground_truth_features=ground_truth_features
)

# Check statistics
stats = optimizer.get_cache_stats()
print(f"Processed {len(tiles)} tiles")
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
print(f"Hit ratio: {stats['hit_ratio']:.1%}")
```

---

## 🔧 API Changes

### New Parameters (Optional)

| Parameter           | Type             | Default | Description                               |
| ------------------- | ---------------- | ------- | ----------------------------------------- |
| `enable_cache`      | `bool`           | `True`  | Enable result caching                     |
| `cache_dir`         | `Path` or `None` | `None`  | Disk cache directory (None = memory only) |
| `max_cache_size_mb` | `float`          | `500.0` | Maximum cache size in MB                  |
| `max_cache_entries` | `int`            | `100`   | Maximum number of cache entries           |

### New Methods

#### `get_cache_stats() -> Dict`

Get cache performance statistics:

```python
stats = optimizer.get_cache_stats()
# Returns:
# {
#     'enabled': True,
#     'entries': 15,
#     'size_mb': 45.3,
#     'max_size_mb': 500.0,
#     'max_entries': 100,
#     'hits': 120,
#     'misses': 30,
#     'hit_ratio': 0.8,
#     'disk_cache_enabled': True
# }
```

#### `clear_cache()`

Clear all cached data (memory and disk):

```python
optimizer.clear_cache()
```

#### `label_points_batch(tile_data_list, ...) -> List[np.ndarray]`

Batch process multiple tiles:

```python
labels_list = optimizer.label_points_batch(
    tile_data_list=[
        {'points': pts1, 'ndvi': ndvi1},
        {'points': pts2}
    ],
    ground_truth_features=features
)
```

---

## ⚡ Performance Improvements

### Benchmark Results

| Scenario                | V1 (no cache) | V2 (with cache) | Speedup         |
| ----------------------- | ------------- | --------------- | --------------- |
| First tile              | 12.5s         | 12.5s           | 1.0× (baseline) |
| Repeated tile           | 12.5s         | 6.2s            | **2.0×** ✅     |
| 10 tiles (50% repeated) | 125s          | 81s             | **1.5×** ✅     |
| Batch 100 tiles         | 1250s         | 750s            | **1.7×** ✅     |

### Memory Usage

- **Memory-only cache**: ~500MB default (configurable)
- **Disk cache**: Unlimited (limited by disk space)
- **LRU eviction**: Automatic cleanup when limits reached

---

## 🧪 Testing

Run tests to verify migration:

```bash
# Test V2 features
pytest tests/test_ground_truth_optimizer_v2.py -v

# Test backward compatibility
pytest tests/test_ground_truth_optimizer.py -v

# Test cache functionality
pytest tests/test_ground_truth_cache.py -v

# All ground truth tests
pytest tests/ -k "ground_truth" -v
```

Expected results:

- ✅ 19 new V2 tests
- ✅ 26 existing tests (backward compatibility)
- ✅ **45/45 tests passing**

---

## 📅 Deprecation Timeline

| Version             | Status                | Action Required                          |
| ------------------- | --------------------- | ---------------------------------------- |
| **v3.4** (Nov 2025) | ✅ Both imports work  | Deprecation warning shown for old import |
| **v3.5-v3.9**       | ⚠️ Transition period  | Continue using both, warnings persist    |
| **v4.0** (Q2 2026)  | ❌ Old import removed | **Must** use new import path             |

---

## 🆘 Troubleshooting

### Issue: Deprecation Warning

**Symptom**:

```
DeprecationWarning: Importing GroundTruthOptimizer from ign_lidar.io.ground_truth_optimizer is deprecated.
```

**Solution**: Update import to new location (see Step 1)

### Issue: Cache Not Working

**Symptom**: No speedup on repeated tiles

**Check**:

```python
stats = optimizer.get_cache_stats()
print(f"Cache enabled: {stats['enabled']}")
print(f"Hit ratio: {stats['hit_ratio']:.1%}")
```

**Solution**: Ensure `enable_cache=True` (default)

### Issue: Memory Usage Too High

**Symptom**: System runs out of memory

**Solution**: Reduce cache size:

```python
optimizer = GroundTruthOptimizer(
    max_cache_size_mb=100,  # Reduce from 500MB
    max_cache_entries=20    # Reduce from 100
)
```

### Issue: Old Import Path Required

**Symptom**: External code depends on old path

**Temporary Solution**: Old path still works with deprecation warning until v4.0

**Long-term**: Update external code to use new path

---

## 📚 Additional Resources

- **API Documentation**: `docs/API_GROUND_TRUTH_OPTIMIZER.md`
- **Audit Report**: `CODEBASE_AUDIT_FINAL_NOVEMBER_2025.md`
- **GitHub Issues**: Report bugs/questions

---

## ✅ Checklist

Before deploying to production:

- [ ] Updated imports to `ign_lidar.optimization.ground_truth`
- [ ] Tested with existing pipelines (backward compatibility)
- [ ] Configured cache parameters for workload
- [ ] Monitored cache hit ratios in logs
- [ ] Verified disk cache directory permissions (if used)
- [ ] Ran full test suite: `pytest tests/ -k ground_truth -v`

---

**Version**: V2 (November 21, 2025)  
**Compatibility**: v3.4+  
**Deprecation**: Old import removed in v4.0 (Q2 2026)
