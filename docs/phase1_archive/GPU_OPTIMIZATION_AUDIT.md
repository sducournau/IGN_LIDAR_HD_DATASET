# GPU Implementation Audit & Optimization Plan

**Date:** October 18, 2025  
**Target:** RTX 4080 Super (16GB VRAM)  
**Issue:** Processing taking too long on 18.6M point tiles

## 🔍 BOTTLENECK ANALYSIS

### Current Performance Issue

From logs:

```
2025-10-18 21:32:21 - Batching FAISS queries: 10 batches of 2,000,000 points
```

**Root Cause:** FAISS batching is TOO CONSERVATIVE

- Batch size: 2M points
- Number of batches: 10 for 18.6M points
- Memory usage per batch: ~305MB (2M × 10 neighbors × 4 bytes × 2 + overhead)
- Available VRAM: 14.7GB (91.8% free!)

### Critical Issues Identified

#### 1. **FAISS Query Batching is Inefficient**

Location: `features_gpu_chunked.py:2826`

```python
batch_size = 2_000_000  # TOO SMALL!
num_batches = 10  # TOO MANY!
```

**Problem:** Hardcoded 2M batch size wastes GPU capacity

- Current: 10 sequential batches (no parallelism)
- GPU underutilized: Only using ~300MB of 14.7GB available
- Loop overhead: 10× function call overhead
- Memory fragmentation: 10× allocation/deallocation cycles

**Solution:** Increase to 10-15M points per batch

- Expected batches: 2 instead of 10 (5× reduction)
- Better VRAM utilization: ~1.5GB per batch (still safe)
- Reduce loop overhead by 80%

#### 2. **Duplicate Code Between GPU Strategies**

- `strategy_gpu.py` - Single batch GPU (1-10M points)
- `strategy_gpu_chunked.py` - Chunked GPU (>10M points)
- `features_gpu_chunked.py` - Core implementation

**Problems:**

- Code duplication leads to maintenance burden
- Both strategies call `features_gpu_chunked.py` differently
- Inconsistent optimizations between paths
- FAISS path only in chunked version

#### 3. **Feature Computation Pipeline Inefficiency**

From logs - THREE separate phases:

```
Phase 1: Build KDTree/FAISS index
Phase 2: Query neighbors (BOTTLENECK HERE)
Phase 3: Compute features from neighbors
```

**Problem:** Sequential processing, no pipelining

- Build index → wait
- Query ALL neighbors → wait (SLOW!)
- Compute ALL features → wait

**Better approach:** Stream processing

- Query batch 1 → start computing features
- While computing batch 1, query batch 2
- Overlap neighbor queries with feature computation

#### 4. **Memory Transfer Overhead**

Location: `features_gpu_chunked.py:2813-2815`

```python
points_np = cp.asnumpy(points_gpu).astype(np.float32)  # GPU → CPU
faiss_index = self._build_faiss_index(points_np, k)     # CPU → GPU
global_indices_all_gpu = cp.asarray(indices_all)        # CPU → GPU
```

**Problem:** Unnecessary round-trip transfers

1. Points start on GPU (CuPy array)
2. Transfer to CPU for FAISS (230MB transfer @ 15GB/s = 15ms)
3. FAISS builds GPU index (transfers back to GPU)
4. Results transfer back to CPU
5. Transfer results to GPU for feature computation

**Solution:** Keep data on GPU throughout

- Use FAISS GPU resources directly
- Avoid CPU intermediate steps

#### 5. **Configuration Complexity**

Multiple overlapping parameters:

```yaml
processor:
  gpu_batch_size: 30000000 # Used where?
  gpu_memory_target: 0.9 # Redundant with vram_limit_gb?
  chunk_size: null # Overridden by features?
  vram_limit_gb: 14
  ground_truth_chunk_size: 20000000 # Another chunk size!

features:
  gpu_batch_size: 30000000 # Duplicate!
  neighbor_query_batch_size: 50000000 # Different value!
  feature_batch_size: 30000000 # Yet another!
```

**Problem:** Confusing, inconsistent, redundant

- 5 different size parameters
- Unclear which takes precedence
- No unified memory management

## 📋 REFACTORING PLAN

### Phase 1: Immediate Performance Fix (30 mins)

**Priority:** Critical  
**Impact:** 5-10× speedup on neighbor queries

#### 1.1 Increase FAISS Batch Size

File: `ign_lidar/features/features_gpu_chunked.py:2820`

```python
# BEFORE
batch_size = 2_000_000  # Conservative

# AFTER - Dynamic based on available VRAM
def _calculate_optimal_faiss_batch_size(self, N: int, k: int, available_vram_gb: float) -> int:
    """
    Calculate optimal FAISS query batch size based on available VRAM.

    Memory requirements:
    - Index storage: Already built (accounted for)
    - Query points: N × 3 × 4 bytes
    - Result indices: N × k × 4 bytes
    - Result distances: N × k × 4 bytes
    - Working memory: 2× (for safety)

    Args:
        N: Total points
        k: Neighbors per point
        available_vram_gb: Available VRAM in GB

    Returns:
        Optimal batch size
    """
    # Memory per point (in bytes)
    memory_per_point = (
        3 * 4 +      # Query points (xyz, float32)
        k * 4 * 2 +  # Results: indices + distances (int32 + float32)
        32           # Overhead
    )

    # Use 60% of available VRAM for batch (leave room for index + overhead)
    usable_vram = available_vram_gb * 0.6 * (1024**3)  # Convert to bytes

    # Calculate max batch size
    max_batch = int(usable_vram / memory_per_point)

    # Clamp to reasonable range
    batch_size = min(
        max_batch,
        N,           # Don't exceed total points
        20_000_000   # Hard cap at 20M for safety
    )
    batch_size = max(batch_size, 1_000_000)  # Minimum 1M

    return batch_size
```

#### 1.2 Add Progress Logging

```python
# Add time tracking per batch
for batch_idx in range(num_batches):
    batch_start_time = time.time()
    # ... query ...
    batch_time = time.time() - batch_start_time
    logger.info(
        f"        ✓ Batch {batch_idx + 1}/{num_batches}: {batch_time:.1f}s "
        f"({batch_n_points/batch_time/1e6:.2f}M pts/s)"
    )
```

**Expected Results:**

- Batch size: 2M → 12-15M points
- Batches: 10 → 2 batches
- Query time: ~30s → ~6-10s (3-5× faster)

### Phase 2: Unify GPU Strategies (2 hours)

**Priority:** High  
**Impact:** Reduce code duplication by 70%, easier maintenance

#### 2.1 Create Unified GPU Strategy

New file: `ign_lidar/features/strategy_gpu_unified.py`

```python
"""
Unified GPU Strategy - Handles both single-batch and chunked processing.

Auto-selects optimal processing mode based on:
- Dataset size
- Available VRAM
- User configuration

This replaces both strategy_gpu.py and strategy_gpu_chunked.py.
"""

class GPUUnifiedStrategy(BaseFeatureStrategy):
    """
    Unified GPU strategy with automatic mode selection.

    Automatically chooses between:
    1. Single-batch mode (< 10M points, fits in VRAM)
    2. Chunked mode (> 10M points, needs batching)
    3. FAISS fast path (when available)

    Key features:
    - Dynamic batch sizing based on available VRAM
    - FAISS integration for 50-100× speedup
    - Transparent fallback to CPU if GPU fails
    - Progress tracking and performance metrics
    """

    def __init__(self, **kwargs):
        """Auto-configure based on system capabilities."""
        super().__init__(**kwargs)

        # Detect capabilities
        self.gpu_available = GPU_AVAILABLE
        self.cuml_available = CUML_AVAILABLE
        self.faiss_available = FAISS_AVAILABLE

        # Auto-detect VRAM
        if self.gpu_available:
            free_vram, total_vram = cp.cuda.runtime.memGetInfo()
            self.total_vram_gb = total_vram / (1024**3)
            self.available_vram_gb = free_vram / (1024**3)

        # Initialize unified computer
        self.computer = GPUChunkedFeatureComputer(**kwargs)

    def compute(self, points, **kwargs):
        """
        Compute features with automatic optimization.

        Decision tree:
        1. Try FAISS path (fastest)
        2. Fall back to cuML chunked
        3. Fall back to CPU if GPU fails
        """
        N = len(points)

        # Auto-select processing mode
        if N < 10_000_000 and self._fits_in_vram(N):
            mode = "single_batch"
        else:
            mode = "chunked"

        logger.info(f"  🚀 Auto-selected mode: {mode} ({N:,} points)")

        # Use FAISS if available (fastest)
        if self.faiss_available:
            return self._compute_with_faiss(points, mode, **kwargs)
        elif self.cuml_available:
            return self._compute_with_cuml(points, mode, **kwargs)
        else:
            return self._compute_with_cpu(points, **kwargs)
```

#### 2.2 Deprecate Old Strategies

- Keep `strategy_gpu.py` and `strategy_gpu_chunked.py` as thin wrappers
- Forward all calls to unified implementation
- Add deprecation warnings

### Phase 3: Stream Processing Pipeline (3 hours)

**Priority:** Medium  
**Impact:** 20-30% speedup through overlap

#### 3.1 Implement Pipelined Feature Computation

```python
def compute_features_pipelined(self, points, k, features_to_compute):
    """
    Pipeline neighbor queries with feature computation.

    Pipeline stages (overlap in time):
    ┌─────────────────────────────────────────┐
    │ Batch 1: Query neighbors               │
    └──────────┬──────────────────────────────┘
               │ Batch 1: Compute features    │
               └──────────┬───────────────────┘
                          │ Batch 2: Query    │
                          └──────────┬────────┘
                                     │ Batch 2: Compute │
                                     └─────────┘

    Using CUDA streams for overlap.
    """
    # Implementation using CUDA streams
    # Similar to existing _compute_normals_with_streams
    # But for full feature pipeline
```

### Phase 4: Simplify Configuration (1 hour)

**Priority:** High  
**Impact:** Better UX, fewer bugs

#### 4.1 Unified Memory Configuration

```yaml
# NEW - Single unified config section
gpu:
  memory:
    vram_limit_gb: 14 # Total VRAM cap
    vram_safety_margin: 0.1 # 10% safety margin
    batch_size_auto: true # Auto-calculate batch sizes
    batch_size_override: null # Manual override if needed

  optimization:
    use_faiss: true # Enable FAISS (auto-detect)
    use_cuda_streams: true # Overlap processing
    enable_memory_pooling: true # Reduce alloc overhead

  fallback:
    auto_fallback_cpu: true # Fall back to CPU on OOM
    retry_with_smaller_batch: true # Retry with 50% batch size
```

#### 4.2 Remove Redundant Parameters

Mark as deprecated in code:

- `processor.gpu_batch_size` → use `gpu.memory.batch_size_override`
- `processor.chunk_size` → auto-calculated
- `features.gpu_batch_size` → use unified `gpu.memory.*`
- `features.neighbor_query_batch_size` → auto-calculated
- `features.feature_batch_size` → auto-calculated

### Phase 5: Testing & Validation (1 hour)

#### 5.1 Performance Benchmarks

Create: `scripts/benchmark_gpu_unified.py`

```python
"""
Benchmark unified GPU implementation.

Tests:
1. Small dataset (1M points)
2. Medium dataset (5M points)
3. Large dataset (20M points)
4. FAISS vs cuML comparison
5. Memory usage tracking
"""
```

#### 5.2 Regression Tests

```python
def test_gpu_unified_matches_legacy():
    """Ensure unified implementation produces identical results."""
    # Generate test data
    points = np.random.randn(100000, 3)

    # Legacy path
    from ign_lidar.features.strategy_gpu_chunked import GPUChunkedStrategy
    legacy = GPUChunkedStrategy()
    result_legacy = legacy.compute(points)

    # Unified path
    from ign_lidar.features.strategy_gpu_unified import GPUUnifiedStrategy
    unified = GPUUnifiedStrategy()
    result_unified = unified.compute(points)

    # Compare
    np.testing.assert_allclose(
        result_legacy['normals'],
        result_unified['normals'],
        rtol=1e-5
    )
```

## 📊 EXPECTED IMPROVEMENTS

### Current Performance (18.6M point tile)

- FAISS index build: ~5s
- FAISS queries: ~30-60s (10 batches) ⚠️ BOTTLENECK
- Feature computation: ~15s
- **Total: ~50-80s per tile**

### After Phase 1 (Immediate Fix)

- FAISS index build: ~5s
- FAISS queries: ~6-10s (2 batches) ✅ 5× FASTER
- Feature computation: ~15s
- **Total: ~26-30s per tile** (2-3× speedup)

### After Phase 3 (Pipelined)

- Index build + pipelined queries + features: ~20-25s
- **Total: ~20-25s per tile** (3-4× speedup)

### After All Phases

- **Total: ~15-20s per tile** (4-5× speedup)
- 128 tiles: ~32-43 minutes (down from 2-3 hours)

## 🎯 IMPLEMENTATION ORDER

### Week 1: Critical Path

1. ✅ **Day 1** - Phase 1.1: Dynamic FAISS batch sizing
2. ✅ **Day 1** - Phase 1.2: Progress logging
3. **Day 2** - Phase 4: Simplify configuration
4. **Day 3** - Test on real data, validate speedup

### Week 2: Refactoring

5. **Day 4-5** - Phase 2: Unified GPU strategy
6. **Day 6-7** - Phase 5: Testing & validation

### Week 3: Advanced Optimizations

7. **Day 8-10** - Phase 3: Stream processing pipeline
8. **Day 11** - Performance benchmarking
9. **Day 12** - Documentation update

## 🚀 QUICK START: Apply Phase 1 Now

Run these commands to apply immediate fix:

```bash
# 1. Create backup
cp ign_lidar/features/features_gpu_chunked.py \
   ign_lidar/features/features_gpu_chunked.py.backup

# 2. Apply Phase 1 patch (will be created next)
# See GPU_PHASE1_PATCH.md

# 3. Test on single tile
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/asprs_rtx4080_fast.yaml" \
  input_dir="/mnt/d/ign/selected_tiles/asprs/tiles" \
  output_dir="/mnt/d/ign/test_output" \
  --max-tiles=1

# 4. Benchmark improvement
python scripts/benchmark_gpu_unified.py
```

## 📝 NOTES

### Why FAISS Batching is the Bottleneck

From the logs, we see:

```
21:32:21 - Building FAISS index... (fast)
21:32:24 - Batching FAISS queries: 10 batches
[... long pause here ...]
```

The index build is fast (~3s), but then we batch queries into 10 pieces:

- Each batch: function overhead + GPU kernel launch + memory allocation
- 10 batches = 10× overhead
- No parallelism between batches (sequential loop)

**Solution:** Fewer, larger batches = less overhead, better GPU utilization

### Memory Calculation Details

For 18.6M points × 10 neighbors with current 2M batch size:

```
Per batch:
- Query points: 2M × 3 × 4 bytes = 24MB
- Indices result: 2M × 10 × 4 bytes = 80MB
- Distances result: 2M × 10 × 4 bytes = 80MB
- Overhead: ~50MB
Total: ~234MB per batch

With 14.7GB available, we could easily do:
- 12M point batches = ~1.4GB per batch
- Only 2 batches needed (vs 10 currently)
```

### Why Not One Big Batch?

Could do all 18.6M in one query, BUT:

- FAISS IVF index already in VRAM (~2GB)
- Result arrays: 18.6M × 10 × 8 bytes = 1.5GB
- Working memory for FAISS: ~1-2GB
- Total: ~5GB needed

This fits! But we batch for:

1. Safety margin (avoid OOM crashes)
2. Progress reporting (user feedback)
3. Memory defragmentation (cleanup between batches)

Optimal: 2-3 batches balances safety and performance.
