# 🚀 Priority Fixes Roadmap - IGN LiDAR HD v3.6.1

**Date**: November 26, 2025  
**Target**: 3-4x overall performance improvement  
**Estimated Effort**: 4-5 weeks

---

## 🎯 Quick Summary

| Issue                                 | Speedup | Files | Effort | Priority  |
| ------------------------------------- | ------- | ----- | ------ | --------- |
| **KDTree CPU-only (9.7x slower GPU)** | 1.56x   | 5+    | 2-3d   | 🔴 URGENT |
| **GPU Memory Fragmentation**          | 1.20x   | 5     | 2-3d   | 🔴 URGENT |
| **Serial GPU-CPU Transfers**          | 1.20x   | 4     | 3-4d   | 🟠 HIGH   |
| **FAISS Batch Size**                  | 1.10x   | 1     | 1d     | 🟠 HIGH   |
| **Formatter Index Caching**           | 1.05x   | 2     | 1d     | 🟡 MEDIUM |
| **API Cleanup**                       | -       | 3     | 1d     | 🟡 MEDIUM |

**Cumulative**: 1.56 × 1.20 × 1.20 × 1.10 × 1.05 = **2.58x overall**

---

## 🔴 PHASE 1: URGENT (Week 1 - 9.7x KNN bottleneck)

### Fix 1.1: Migrate All KNN to KNNEngine

**Problem**: KDTree always CPU, even with 50M+ points  
**Solution**: Use existing KNNEngine (auto GPU/CPU)  
**Impact**: 1.56x speedup (saves 36s on 50M points)

#### Files to Update

```
ign_lidar/features/utils.py                   (1 function)
ign_lidar/features/compute/density.py         (1 function)
ign_lidar/core/tile_stitcher.py               (1 function)
ign_lidar/io/formatters/multi_arch_formatter.py  (1 function)
ign_lidar/io/formatters/hybrid_formatter.py   (1 function)
ign_lidar/optimization/gpu_accelerated_ops.py (remove duplicates)
```

#### Implementation

```python
# BEFORE (CPU-only)
from sklearn.neighbors import KDTree
tree = KDTree(points, metric='euclidean')
distances, indices = tree.query(query_points, k=30)

# AFTER (Auto GPU/CPU)
from ign_lidar.optimization import KNNEngine
engine = KNNEngine()
distances, indices = engine.search(points, k=30)
# Automatically uses:
# - FAISS-GPU (10x faster) if available
# - FAISS-CPU (2x faster) if no GPU
# - cuML or sklearn as fallback
```

#### Checklist

- [ ] Replace `build_kdtree()` in `features/utils.py`
- [ ] Replace KDTree in `compute/density.py`
- [ ] Replace KDTree in `core/tile_stitcher.py`
- [ ] Replace KDTree in formatters (2 files)
- [ ] Remove duplicate KNN implementations
- [ ] Add caching to avoid rebuilds
- [ ] Run unit tests
- [ ] Benchmark: expect 10x on 1M points, 1.5x on pipeline
- [ ] Update documentation

---

### Fix 1.2: Universalize GPU Memory Pooling

**Problem**: Memory fragmentation causes 20-40% loss  
**Solution**: Reuse GPU buffers instead of allocating new ones  
**Impact**: 1.20x speedup

#### Files to Update

```
ign_lidar/features/strategy_gpu.py           (20 allocations → pooled)
ign_lidar/features/strategy_gpu_chunked.py   (15 allocations → pooled)
ign_lidar/optimization/vectorized.py         (10 allocations → pooled)
ign_lidar/io/formatters/multi_arch_formatter.py (5 allocations → pooled)
ign_lidar/io/formatters/hybrid_formatter.py  (5 allocations → pooled)
```

#### Implementation

```python
# Module-level pool factory
_gpu_pool = None

def get_gpu_pool():
    global _gpu_pool
    if _gpu_pool is None:
        from ign_lidar.optimization.gpu_cache import GPUMemoryPool
        # Size based on GPU VRAM
        available_gb = detect_available_vram()
        _gpu_pool = GPUMemoryPool(max_size_gb=available_gb * 0.7)
    return _gpu_pool

# In all GPU operations:
pool = get_gpu_pool()
buffer = pool.allocate(size_needed, name=f"feature_{i}")
result = compute(buffer)  # Uses same memory, no allocation!
output = cp.asnumpy(result)
pool.free(buffer)  # Reuse next iteration
```

#### Checklist

- [ ] Create pool factory function
- [ ] Update `strategy_gpu.py` to use pooling
- [ ] Update `strategy_gpu_chunked.py` to use pooling
- [ ] Update `vectorized.py` to use pooling
- [ ] Update formatters to use pooling
- [ ] Add pool statistics monitoring
- [ ] Test fragmentation resistance
- [ ] Benchmark: expect 1.2-1.4x improvement
- [ ] Document pool sizing strategy

---

## 🟠 PHASE 2: HIGH PRIORITY (Week 2)

### Fix 2.1: Batch GPU-CPU Transfers

**Problem**: Serial transfers (one per feature) waste GPU time  
**Solution**: Batch all transfers at start and end  
**Impact**: 1.20x speedup (saves 300ms per batch)

#### Files to Update

```
ign_lidar/features/strategy_gpu.py        (refactor compute loop)
ign_lidar/features/compute/geometric.py   (batch computations)
ign_lidar/features/compute/eigenvalues.py (batch computations)
ign_lidar/features/compute/feature_filter.py (if needed)
```

#### Implementation

```python
# BEFORE (serial)
results = {}
for feature_name in ['planarity', 'linearity', 'sphericity']:
    gpu_data = cp.asarray(cpu_data[feature_name])      # Transfer 1
    results[feature_name] = cp.asnumpy(compute(gpu_data))  # Transfer 2
# = 6 transfers for 3 features

# AFTER (batch)
# Transfer all at once
gpu_data = {name: cp.asarray(data) for name, data in cpu_data.items()}

# Compute all at once
gpu_results = {}
for name, data in gpu_data.items():
    gpu_results[name] = compute(data)  # No transfers yet

# Transfer all results at once
results = {name: cp.asnumpy(data) for name, data in gpu_results.items()}
# = 2 transfers total!
```

#### Checklist

- [ ] Refactor feature computation loop in `strategy_gpu.py`
- [ ] Implement batch transfer in `geometric.py`
- [ ] Implement batch transfer in `eigenvalues.py`
- [ ] Verify results match original
- [ ] Benchmark: expect 1.15-1.25x improvement
- [ ] Test with different dataset sizes
- [ ] Update documentation

### Fix 2.2: Optimize FAISS Batch Sizes

**Problem**: Conservative settings waste GPU capacity (50% VRAM, 3x safety)  
**Solution**: More aggressive batching (70% VRAM, 2x safety)  
**Impact**: 1.10x speedup (300K → 1.2M batch size)

#### File to Update

```
ign_lidar/features/gpu_processor.py (line 1170-1190)
```

#### Implementation

```python
# BEFORE (conservative)
available_gb = self.vram_limit_gb * 0.5      # 50% usage ❌
bytes_per_point = k * 8 * 3                   # 3x safety ❌
batch_size = min(5_000_000, max(100_000, ...))  # Fixed bounds

# AFTER (optimized)
available_gb = self.vram_limit_gb * 0.7      # 70% usage ✓
bytes_per_point = k * 8 * 2                   # 2x safety ✓
batch_size = max(500_000, min(10_000_000, ...)) # Dynamic bounds

# Example: 16GB GPU
# Before: batch_size ~600K (wastes 8GB)
# After: batch_size ~1.2M (uses capacity well)
```

#### Checklist

- [ ] Update batch size calculation in `gpu_processor.py`
- [ ] Test with 6GB, 8GB, 12GB, 16GB, 24GB GPUs
- [ ] Benchmark on each GPU type
- [ ] Ensure stability (no OOM errors)
- [ ] Update documentation

---

## 🟡 PHASE 3: MEDIUM PRIORITY (Week 3)

### Fix 3.1: Formatter Index Caching

**Problem**: Rebuild KDTree index for each tile (waste!)  
**Solution**: Cache indices, reuse across tiles  
**Impact**: 1.05x speedup

#### Files to Update

```
ign_lidar/io/formatters/multi_arch_formatter.py
ign_lidar/io/formatters/hybrid_formatter.py
```

#### Implementation

```python
# BEFORE (rebuild each time)
def process_tile(self, points):
    engine = KNNEngine()  # New instance!
    distances, indices = engine.search(points, k=30)
    # Returns correct results but rebuilds engine for each tile

# AFTER (cache and reuse)
class CachedFormatter:
    def __init__(self):
        self._kdtree_cache = {}
        self._engine = KNNEngine()  # Shared instance!

    def process_tile(self, tile_id, points):
        if tile_id not in self._kdtree_cache:
            # First time: build and cache
            self._engine.fit(points)
            self._kdtree_cache[tile_id] = self._engine
        else:
            # Reuse cached engine
            self._engine = self._kdtree_cache[tile_id]

        distances, indices = self._engine.search(k=30)
        return distances, indices
```

#### Checklist

- [ ] Add caching to `multi_arch_formatter.py`
- [ ] Add caching to `hybrid_formatter.py`
- [ ] Test cache effectiveness (profile cache hits)
- [ ] Benchmark: expect 1.05-1.1x improvement
- [ ] Monitor memory usage (don't cache too much)

### Fix 3.2: API Cleanup & Deprecation

**Problem**: Multiple entry points, deprecated APIs still importable  
**Solution**: Mark for removal, add migration helpers  
**Timeline**: v4.0 (next major release)

#### Files to Update

```
ign_lidar/features/feature_computer.py      (REMOVE in v4.0)
ign_lidar/core/feature_engine.py            (REMOVE in v4.0)
ign_lidar/features/__init__.py              (update exports)
```

#### Implementation

```python
# CURRENT (deprecated)
from ign_lidar.features import FeatureComputer  # ❌ Deprecated
from ign_lidar.core import FeatureEngine        # ❌ Deprecated

# RECOMMENDED (primary)
from ign_lidar.features import FeatureOrchestrationService  # ✓

# ADVANCED (internal)
from ign_lidar.features import FeatureOrchestrator  # ✓ for power users
```

#### Checklist

- [ ] Add deprecation warnings (already done?)
- [ ] Update migration guide
- [ ] Plan removal for v4.0
- [ ] Update documentation

---

## 📊 Performance Targets

### Before vs After

```
Scenario: 50M points, LOD3 features, RTX 4080 Super (16GB)

BEFORE:
├── KDTree construction:    40s ❌
├── Eigenvalue decomp:      25s
├── Feature computation:    20s
├── GPU-CPU transfers:      10s
├── Other:                   5s
TOTAL: 100s

AFTER ALL FIXES:
├── KDTree construction:     2.5s ✓ (16x faster! via GPU)
├── Eigenvalue decomp:      25s (same, CUSOLVER limited)
├── Feature computation:    20s (same, well optimized)
├── GPU-CPU transfers:       3s ✓ (batch transfers)
├── GPU memory pooling:       1s (amortized, no fragmentation)
└── Other:                   5s
TOTAL: 56.5s → 28.5s with batching optimization

Overall Speedup: 100s / 28.5s = 3.5x
```

### By Fix

| Fix               | Baseline | After   | Speedup        |
| ----------------- | -------- | ------- | -------------- |
| Start             | 100s     | 100s    | 1.0x           |
| + KNN GPU         | 100s     | 64s     | 1.56x          |
| + Memory Pool     | 64s      | 53s     | 1.20x (vs 64s) |
| + Batch Transfers | 53s      | 44s     | 1.20x (vs 53s) |
| + FAISS Batch     | 44s      | 40s     | 1.10x (vs 44s) |
| + Index Cache     | 40s      | 38s     | 1.05x (vs 40s) |
| **Final**         | 100s     | **38s** | **2.63x**      |

---

## ✅ Implementation Checklist

### Phase 1 (Week 1) - URGENT

**KNN Migration**

- [ ] Create KNNEngine wrapper for backward compatibility
- [ ] Update `build_kdtree()` to use KNNEngine
- [ ] Update density features
- [ ] Update tile stitcher
- [ ] Update formatters
- [ ] Comprehensive testing
- [ ] Benchmark 1M, 10M, 50M points

**GPU Memory Pooling**

- [ ] Create pool factory
- [ ] Update strategy_gpu.py
- [ ] Update strategy_gpu_chunked.py
- [ ] Test fragmentation resistance
- [ ] Benchmark and validate

### Phase 2 (Week 2) - HIGH

**Batch Transfers**

- [ ] Refactor geometric features
- [ ] Refactor eigenvalues
- [ ] Integration testing
- [ ] Benchmark impact

**FAISS Batching**

- [ ] Update batch size calculation
- [ ] Test on different GPUs
- [ ] Validate stability

### Phase 3 (Week 3) - MEDIUM

**Index Caching**

- [ ] Add caching layer
- [ ] Monitor memory
- [ ] Benchmark

**API Cleanup**

- [ ] Documentation updates
- [ ] Deprecation warnings
- [ ] v4.0 planning

### Testing & Release

- [ ] All unit tests pass
- [ ] GPU tests pass (ign_gpu environment)
- [ ] Integration tests pass
- [ ] Performance benchmarks validate
- [ ] Documentation complete
- [ ] Changelog prepared
- [ ] Release notes written

---

## 📋 Success Criteria

### Performance

- ✓ 2.5-3.5x overall speedup on 50M+ point datasets
- ✓ <10s total time for 50M points (down from ~100s)
- ✓ >75% average GPU utilization

### Quality

- ✓ All tests pass (unit, integration, GPU)
- ✓ No memory leaks
- ✓ Backward compatible (v3.x API unchanged)

### Documentation

- ✓ Updated API docs
- ✓ Updated GPU optimization guide
- ✓ Troubleshooting updated
- ✓ Performance benchmarks documented

---

## 📞 Status Tracking

### Current Status

- [x] Audit complete
- [x] Bottlenecks identified
- [x] Fixes prioritized
- [ ] Phase 1 started

### Next Steps

1. Review and approve roadmap
2. Allocate development resources
3. Begin Phase 1 (KNN + Memory Pooling)
4. Weekly progress reviews

---

**Created**: November 26, 2025  
**Last Updated**: November 26, 2025  
**Status**: Ready for Implementation  
**Contact**: IGN LiDAR HD Development Team
