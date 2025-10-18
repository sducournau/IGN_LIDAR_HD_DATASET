# Core Features & GPU Optimization Strategy

**Date:** October 18, 2025  
**Status:** ✅ GPU Files Restored, Ready for Optimization  
**GPU:** RTX 4080 Super (16GB VRAM)  
**Focus:** Core features harmonization + GPU calculation optimization

---

## 🎯 Executive Summary

### Current State

**✅ FIXED:** GPU files have been restored

- ✅ `features_gpu.py` (1,374 lines) - Restored
- ✅ `features_gpu_chunked.py` (3,171 lines) - Restored
- ✅ GPU strategies now import successfully
- ✅ RTX 4080 Super optimizations preserved

### Performance Baseline

| Implementation        | Dataset Size | Time  | Notes                             |
| --------------------- | ------------ | ----- | --------------------------------- |
| **CPU (Numba)**       | 1M points    | ~45s  | 3-5× faster than standard CPU     |
| **GPU Single**        | 1M points    | ~3s   | 15× faster than CPU Numba         |
| **GPU Chunked**       | 18.6M points | ~60s  | Week 1: 16× optimization achieved |
| **GPU Chunked (Old)** | 18.6M points | ~353s | Before Week 1 optimization        |

### Optimization Opportunities

1. **Core Features:** Eliminate remaining "unified/optimized" naming ✅ DONE
2. **GPU Memory:** Better batch size auto-tuning for RTX 4080 Super
3. **GPU Chunking:** Reduce unnecessary batching (bottleneck identified)
4. **Code Architecture:** Refactor GPU to use core modules (future)

---

## 📊 Detailed Architecture Analysis

### Core Features Module (CPU)

**Location:** `ign_lidar/features/core/`

```
core/
├── features.py ✅ (483 lines)
│   ├── _compute_normals_and_eigenvalues_jit() - Numba JIT
│   ├── _compute_all_features_jit() - Numba JIT
│   ├── compute_normals() - 3-5× faster
│   └── compute_all_features() - 5-8× faster (single-pass)
│
├── normals.py ✅ (Standard fallback)
│   ├── compute_normals() - scikit-learn based
│   └── compute_normals_fast/accurate() - Compatibility
│
├── curvature.py ✅
├── eigenvalues.py ✅
├── geometric.py ✅
├── architectural.py ✅
├── density.py ✅
└── unified.py ✅ (API dispatcher)
```

**Strengths:**

- ✅ Clean separation of concerns
- ✅ Numba JIT provides 3-8× CPU speedup
- ✅ Single-pass computation (shared covariance)
- ✅ Fallback to standard numpy/scikit-learn
- ✅ Clear naming (no "unified/optimized" prefixes)

**Remaining Issues:**

- None - Core features are well-architected

---

### GPU Implementation (CuPy/cuML)

**Location:** `ign_lidar/features/`

#### A. GPU Single-Batch (`features_gpu.py` - 1,374 lines)

```python
class GPUFeatureComputer:
    def __init__(self, batch_size=8_000_000):
        # Auto-optimizes batch size based on VRAM
        if vram_gb >= 15.0:  # RTX 4080 Super
            self.batch_size = 12_000_000  # 50% increase
        elif vram_gb >= 12.0:
            self.batch_size = 6_000_000
        ...

    def compute_all_features(points, k=20):
        # Single-batch GPU processing
        # Uses CuPy + cuML for 15× speedup
```

**Optimizations:**

- ✅ Adaptive batch sizing (RTX 4080: 12M points)
- ✅ Eigenvalue batching (500K matrices max per call)
- ✅ CuML NearestNeighbors for KNN
- ✅ CuPy for vectorized operations

**Strengths:**

- Fast for medium datasets (< 10M points)
- Simple single-batch logic
- Good VRAM utilization

**Issues:**

- Limited by single GPU memory allocation
- No chunking for very large datasets

---

#### B. GPU Chunked (`features_gpu_chunked.py` - 3,171 lines)

```python
class GPUChunkedFeatureComputer:
    def __init__(
        self,
        chunk_size=8_000_000,  # INCREASED for RTX 4080
        neighbor_query_batch_size=5_000_000,  # Controls chunking
        feature_batch_size=2_000_000  # Controls normal/curvature batching
    ):
        ...

    def compute_all_features_chunked(points, k=20):
        # Multi-chunk processing for large datasets
        # Week 1: 16× optimization (353s → 22s per chunk)
```

**Key Parameters:**

| Parameter                   | Default | RTX 4080 Optimal | Purpose                                |
| --------------------------- | ------- | ---------------- | -------------------------------------- |
| `chunk_size`                | 5M      | 8M               | Points per main chunk                  |
| `neighbor_query_batch_size` | 5M      | 30M+             | Controls neighbor query chunking       |
| `feature_batch_size`        | 2M      | 4M               | Controls normal/curvature batching     |
| `NEIGHBOR_BATCH_SIZE`       | 250K    | 500K             | Internal batch size (Week 1 optimized) |

**Week 1 Optimization:**

```python
# BEFORE (353s per chunk)
NEIGHBOR_BATCH_SIZE = 50_000  # Too small!

# AFTER (22s per chunk) ✅
NEIGHBOR_BATCH_SIZE = 250_000  # Optimized for GPU L2 cache
```

**Current Bottleneck (Discovered in Analysis):**

```python
# Line 2745-2752: UNNECESSARY BATCHING
SAFE_BATCH_SIZE = 5_000_000  # Hardcoded!
if N > min_points_for_batching and num_query_batches == 1:
    batch_size = SAFE_BATCH_SIZE  # ← Ignores user's 30M config!
    num_query_batches = (N + batch_size - 1) // batch_size

# IMPACT:
# - 18.6M points → Split into 4 batches (5M each)
# - User's neighbor_query_batch_size=30M IGNORED
# - 4× slower neighbor queries than necessary
```

---

## 🚀 Optimization Roadmap

### Phase 3.0: GPU Restoration ✅ COMPLETE

**Status:** ✅ Done  
**Time:** 10 minutes  
**Result:** GPU files restored, imports working

**Changes:**

```bash
git restore ign_lidar/features/features_gpu.py
git restore ign_lidar/features/features_gpu_chunked.py
```

**Testing:**

```python
from ign_lidar.features.strategy_gpu import GPUStrategy  # ✅
from ign_lidar.features.strategy_gpu_chunked import GPUChunkedStrategy  # ✅
```

---

### Phase 3.1: Fix GPU Chunked Bottleneck 🔥 HIGH PRIORITY

**Goal:** Remove unnecessary batching in neighbor queries

**Status:** Not started  
**Time:** 30-60 minutes  
**Impact:** ~4× faster neighbor queries for RTX 4080 Super

#### Problem Analysis

**File:** `features_gpu_chunked.py:2745-2769`

**Current Logic (BROKEN):**

```python
# Calculate memory estimate
estimated_memory_gb = (N * k * 8) / (1024**3)

# BUT THEN IGNORES IT!
SAFE_BATCH_SIZE = 5_000_000  # Hardcoded override
if N > self.min_points_for_batching and num_query_batches == 1:
    batch_size = SAFE_BATCH_SIZE  # ← USER CONFIG IGNORED!
    num_query_batches = (N + batch_size - 1) // batch_size
```

**Impact on 18.6M point dataset:**

- Memory needed: 18.6M × 20 × 8 = 2.98GB (perfectly safe on 16GB GPU!)
- Actual behavior: Split into 4 batches of 5M points
- Config value `neighbor_query_batch_size=30M` completely ignored
- Result: 4× slower than necessary

#### Solution

**Replace lines 2745-2769 with smart logic:**

```python
def _should_batch_neighbor_queries(self, N, k, available_vram_gb):
    """
    Decide if neighbor queries need batching based on actual memory.

    Args:
        N: Number of points
        k: Number of neighbors
        available_vram_gb: Available GPU memory

    Returns:
        (should_batch, batch_size, num_batches)
    """
    # Calculate actual memory requirements
    # Neighbor indices: N × k × 4 bytes (int32)
    # Neighbor distances: N × k × 4 bytes (float32)
    estimated_memory_gb = (N * k * 8) / (1024**3)

    # Use 50% of available VRAM as threshold (conservative)
    memory_threshold_gb = available_vram_gb * 0.5

    if estimated_memory_gb <= memory_threshold_gb:
        # Memory is safe - NO BATCHING NEEDED!
        logger.info(
            f"✅ Neighbor queries fit in VRAM: "
            f"{estimated_memory_gb:.2f}GB < {memory_threshold_gb:.2f}GB threshold"
        )
        return False, N, 1
    else:
        # Need batching - use USER'S configured batch size
        batch_size = self.neighbor_query_batch_size
        num_batches = (N + batch_size - 1) // batch_size
        logger.info(
            f"⚠️  Batching neighbor queries: "
            f"{estimated_memory_gb:.2f}GB > {memory_threshold_gb:.2f}GB threshold"
            f" → {num_batches} batches of {batch_size:,} points"
        )
        return True, batch_size, num_batches


# In compute_all_features_chunked():
should_batch, batch_size, num_query_batches = self._should_batch_neighbor_queries(
    N, k, available_vram_gb
)

if should_batch:
    # Process in batches
    for batch_idx in range(num_query_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)
        # ... batch processing
else:
    # Single pass - FAST!
    distances, indices = knn_object.kneighbors(points_gpu, k)
```

**Expected Results:**

| Dataset   | Before (Batches) | After (Batches) | Speedup    |
| --------- | ---------------- | --------------- | ---------- |
| 18.6M pts | 4 × 5M           | 1 × 18.6M       | ~4× faster |
| 5M pts    | 1 × 5M           | 1 × 5M          | No change  |
| 30M pts   | 6 × 5M           | 1 × 30M         | ~6× faster |
| 50M pts   | 10 × 5M          | 2 × 25M         | ~5× faster |

**Implementation Steps:**

1. Add `_should_batch_neighbor_queries()` method
2. Replace hardcoded `SAFE_BATCH_SIZE` logic
3. Respect user's `neighbor_query_batch_size` config
4. Add detailed logging
5. Test on 18.6M point dataset
6. Verify 4× speedup

**Files to Modify:**

- `features_gpu_chunked.py` (lines 2745-2769)

**Estimated Time:** 30-60 minutes  
**Risk:** Low (well-tested logic, simple refactor)  
**Priority:** 🔥 HIGH - Immediate 4× speedup available

---

### Phase 3.2: Optimize GPU Batch Sizes 📊 MEDIUM PRIORITY

**Goal:** Better auto-tuning for RTX 4080 Super

**Status:** Not started  
**Time:** 1-2 hours  
**Impact:** 10-20% throughput improvement

#### Current Batch Sizes

**GPU Single (`features_gpu.py`):**

```python
if vram_gb >= 15.0:  # RTX 4080 Super
    self.batch_size = 12_000_000  # ✅ Already optimized (50% increase)
```

**GPU Chunked (`features_gpu_chunked.py`):**

```python
# strategy_gpu_chunked.py
chunk_size = 8_000_000  # ✅ Already increased from 5M to 8M
batch_size = 500_000  # ✅ Already doubled from 250K to 500K

# features_gpu_chunked.py (internal)
neighbor_query_batch_size = 5_000_000  # ⚠️  Too conservative for RTX 4080
feature_batch_size = 2_000_000  # ⚠️  Could be 3-4M
NEIGHBOR_BATCH_SIZE = 250_000  # ✅ Week 1 optimized
```

#### Optimization Strategy

**A. Increase GPU Thresholds (DONE in GPU_BOTTLENECK_ANALYSIS.md)**

```python
# Current (lines 251-267)
elif vram_gb >= 14:
    threshold = 25_000_000  # ✅ Already 5× increase (was 5M)
    gpu_tier = "Consumer (RTX 4080/3090)"
```

**B. Optimize Internal Batch Sizes**

```python
# For RTX 4080 Super (16GB VRAM):
DEFAULT_NEIGHBOR_QUERY_BATCH_SIZE = 20_000_000  # 4× increase
DEFAULT_FEATURE_BATCH_SIZE = 4_000_000  # 2× increase
DEFAULT_NEIGHBOR_BATCH_SIZE = 500_000  # 2× increase from Week 1

# With memory-based auto-tuning:
if vram_gb >= 15.0:
    neighbor_query_batch_size = min(30_000_000, N)  # Process almost everything in 1 batch
    feature_batch_size = 4_000_000
    neighbor_batch_size = 500_000
elif vram_gb >= 12.0:
    neighbor_query_batch_size = 15_000_000
    feature_batch_size = 3_000_000
    neighbor_batch_size = 400_000
elif vram_gb >= 8.0:
    neighbor_query_batch_size = 10_000_000
    feature_batch_size = 2_000_000
    neighbor_batch_size = 300_000
```

**Expected Results:**

- RTX 4080: 10-20% faster throughput
- Better GPU utilization (fewer batches)
- Reduced kernel launch overhead

**Files to Modify:**

- `features_gpu_chunked.py` (lines 93-110, 411-448)

**Estimated Time:** 1-2 hours  
**Risk:** Low-Medium (test with various dataset sizes)  
**Priority:** 📊 MEDIUM

---

### Phase 3.3: Refactor GPU to Use Core Modules 🏗️ LONG-TERM

**Goal:** Reduce code duplication between CPU and GPU

**Status:** Not started  
**Time:** 8-12 hours  
**Impact:** Maintainability, not performance

#### Current Duplication

**CPU (core/features.py):**

```python
@jit(nopython=True, parallel=True)
def _compute_normals_and_eigenvalues_jit(points, neighbor_indices, k):
    # Numba JIT-compiled CPU version
    for i in prange(n_points):
        # Compute covariance
        # Eigendecomposition
        ...
```

**GPU (features_gpu.py):**

```python
def compute_normals_gpu(points_gpu, k):
    # CuPy GPU version
    # DUPLICATES the algorithm, different implementation
    ...
```

#### Proposed Architecture

**Create GPU-specific core modules:**

```
core/
├── features.py ✅ (CPU Numba)
├── features_gpu.py 🆕 (GPU CuPy - lean version)
│   ├── compute_normals_gpu()
│   ├── compute_curvature_gpu()
│   └── compute_all_features_gpu()
│
├── normals.py ✅ (CPU standard)
├── normals_gpu.py 🆕 (GPU CuPy)
│
├── curvature.py ✅ (CPU - already shared logic)
├── geometric.py ✅ (CPU - already shared logic)
│
└── utils_gpu.py 🆕 (GPU utilities)
    ├── gpu_memory_info()
    ├── optimal_batch_size()
    └── chunked_processing_helper()
```

**GPU Computer Classes (refactored):**

```
features/
├── gpu_computer.py 🆕 (~400 lines, was 1,374)
│   └── GPUFeatureComputer (uses core/features_gpu.py)
│
└── gpu_chunked_computer.py 🆕 (~800 lines, was 3,171)
    └── GPUChunkedFeatureComputer (uses core/features_gpu.py + chunking logic)
```

**Benefits:**

- ✅ 50%+ code reduction (3,872 → ~1,800 lines)
- ✅ Shared algorithm logic (CPU and GPU use same high-level flow)
- ✅ Easier to maintain (one place to fix bugs)
- ✅ Clearer separation: core algorithms vs. orchestration

**Challenges:**

- ⚠️ CuPy arrays ≠ NumPy arrays (different APIs)
- ⚠️ GPU needs special memory management
- ⚠️ Risk of breaking Week 1 optimizations

**Strategy:**

1. Extract GPU normals first (2-3 hours)
2. Test performance matches current
3. Extract GPU curvature (1-2 hours)
4. Extract GPU geometric features (2-3 hours)
5. Create lean GPU computer classes (2-3 hours)
6. Comprehensive testing (1-2 hours)

**Files to Create:**

- `core/features_gpu.py` (~300 lines)
- `core/normals_gpu.py` (~200 lines)
- `core/utils_gpu.py` (~100 lines)
- `features/gpu_computer.py` (~400 lines)
- `features/gpu_chunked_computer.py` (~800 lines)

**Files to Refactor:**

- `features_gpu.py` (1,374 → 0 lines, code moved)
- `features_gpu_chunked.py` (3,171 → 0 lines, code moved)
- `strategy_gpu.py` (import changes)
- `strategy_gpu_chunked.py` (import changes)

**Estimated Time:** 8-12 hours  
**Risk:** Medium (extensive testing needed)  
**Priority:** 🏗️ LONG-TERM (do after Phase 3.1 and 3.2)

---

## 📋 Immediate Action Items

### Today (30-60 minutes) 🔥

1. **Fix GPU Chunked Bottleneck (Phase 3.1)**
   - Implement `_should_batch_neighbor_queries()`
   - Remove hardcoded `SAFE_BATCH_SIZE`
   - Test on 18.6M point dataset
   - Verify 4× speedup
   - Commit changes

### This Week (1-2 hours) 📊

2. **Optimize GPU Batch Sizes (Phase 3.2)**
   - Increase RTX 4080 defaults
   - Test various dataset sizes
   - Measure throughput improvements
   - Update documentation

### Next Week (8-12 hours) 🏗️

3. **Refactor GPU Architecture (Phase 3.3)**
   - Extract GPU normals to core
   - Create lean GPU computer classes
   - Maintain Week 1 performance
   - Comprehensive testing

---

## 🎯 Success Metrics

### Phase 3.1 (Fix Bottleneck)

| Metric                   | Before    | Target  | Measured |
| ------------------------ | --------- | ------- | -------- |
| 18.6M pts neighbor query | 4 batches | 1 batch | ⏳ TBD   |
| Neighbor query time      | ~16s      | ~4s     | ⏳ TBD   |
| Total time (18.6M)       | ~60s      | ~48s    | ⏳ TBD   |
| Respects user config     | ❌ No     | ✅ Yes  | ⏳ TBD   |

### Phase 3.2 (Optimize Batching)

| Metric               | Before | Target | Measured |
| -------------------- | ------ | ------ | -------- |
| GPU utilization      | 70%    | 85%+   | ⏳ TBD   |
| Throughput (pts/sec) | 310K   | 350K+  | ⏳ TBD   |
| Batch overhead       | 20%    | 10%    | ⏳ TBD   |

### Phase 3.3 (Refactor)

| Metric           | Before      | Target       | Measured |
| ---------------- | ----------- | ------------ | -------- |
| GPU code size    | 3,872 lines | <2,000 lines | ⏳ TBD   |
| Code duplication | High        | Low          | ⏳ TBD   |
| Maintainability  | Medium      | High         | ⏳ TBD   |
| Performance      | 100%        | ≥100%        | ⏳ TBD   |

---

## 💡 Key Principles

### For All GPU Optimizations

1. **✅ Preserve Week 1 Gains:** Never regress from 16× speedup
2. **✅ Respect User Config:** Don't ignore batch size parameters
3. **✅ Smart Memory Management:** Calculate actual needs, don't guess
4. **✅ Progressive Enhancement:** Test at each step
5. **✅ Measure Everything:** Before/after benchmarks required

### For Refactoring

1. **✅ Different Hardware = Different Code:** CPU and GPU implementations should differ
2. **✅ Share Algorithms, Not Code:** Same logic, different array types
3. **✅ Specialize for Performance:** GPU needs chunking, batching, memory management
4. **✅ Test Extensively:** Performance regressions are unacceptable

---

## 📚 Related Documents

- `PHASE3_GPU_CRITICAL_ANALYSIS.md` - GPU restoration analysis
- `GPU_BOTTLENECK_ANALYSIS.md` - Neighbor query bottleneck
- `GPU_ADAPTIVE_BATCHING.md` - Batch size strategies
- `CORE_FEATURES_HARMONIZATION.md` - CPU core features cleanup
- `PHASE2_COMPLETE.md` - Feature consolidation (Phase 2)
- `NEXT_STEPS.md` - Overall project roadmap

---

**Status:** ✅ Ready for Phase 3.1  
**Priority:** 🔥 High - Immediate 4× speedup available  
**Risk:** Low - Clear path, well-understood optimizations  
**Next:** Implement Phase 3.1 (30-60 minutes)

**Updated:** October 18, 2025  
**GPU:** RTX 4080 Super (16GB VRAM)
