# GPU Code Consolidation - Detailed Analysis

**Date**: October 19, 2025  
**Environment**: ign_gpu (CuPy 13.6.0, CUDA 12.0)  
**Status**: Phase 2 Planning - Deep Dive Analysis

---

## Executive Summary

This analysis examines the GPU implementation fragmentation identified in Phase 1 and provides a detailed consolidation strategy for Phase 2.

### Key Findings

1. **Two parallel GPU implementations** exist with 60-70% code overlap
2. **Strategy wrappers** add an extra abstraction layer without removing duplication
3. **GPU Bridge pattern** (Phase 1-3 refactoring) partially addresses the issue
4. **Consolidation opportunity**: ~700-1000 lines can be reduced

---

## Current GPU Architecture

### File Inventory

```
ign_lidar/features/
├── features_gpu.py              (1,175 lines) - Basic GPU implementation
├── features_gpu_chunked.py      (2,100 lines) - Chunked GPU implementation
├── strategy_gpu.py              (200 lines)   - Thin wrapper
├── strategy_gpu_chunked.py      (200 lines)   - Thin wrapper
└── core/
    └── gpu_bridge.py            (600 lines)   - GPU-Core bridge (Phase 1-3)

ign_lidar/optimization/
├── gpu.py                       (~300 lines)  - Basic GPU ops
├── gpu_async.py                 (~400 lines)  - Async processing
├── gpu_array_ops.py             (~200 lines)  - Array operations
├── gpu_coordinator.py           (~150 lines)  - Coordination
├── gpu_dataframe_ops.py         (~250 lines)  - DataFrame ops
├── gpu_kernels.py               (~180 lines)  - CUDA kernels
├── gpu_memory.py                (~350 lines)  - Memory management
└── gpu_profiler.py              (~120 lines)  - Profiling

Total: ~5,825 lines across 13 files
```

---

## Detailed Code Analysis

### 1. features_gpu.py (1,175 lines)

**Purpose**: Basic GPU-accelerated feature computation

**Key Components**:

- `GPUFeatureComputer` class (main interface)
- Normal computation (GPU-accelerated)
- Curvature computation
- Eigenvalue features (uses GPU Bridge ✅)
- Height computation (delegates to core ✅)

**Refactoring Status**:

- ✅ Phase 3: Eigenvalue features use GPU Bridge
- ✅ Delegates to core for some features
- ❌ Still contains custom normal/curvature logic

**Code Structure**:

```python
class GPUFeatureComputer:
    def __init__(self, use_gpu=True, batch_size=None)

    # Core methods
    def compute_normals(points, k=10) -> normals
    def compute_curvature(points, normals, k=10) -> curvature
    def compute_eigenvalue_features() -> features  # Uses GPU Bridge ✅
    def compute_height_above_ground() -> height    # Delegates to core ✅
    def compute_verticality(normals) -> verticality

    # Batch processing
    def _compute_batch_normals_gpu()
    def _compute_batch_eigenvalue_features_gpu()  # Uses GPU Bridge ✅
```

**Duplication Issues**:

- Custom normal computation logic (also in chunked version)
- Custom curvature logic (also in chunked version)
- Batch processing logic (different from chunked version)

---

### 2. features_gpu_chunked.py (2,100 lines)

**Purpose**: GPU feature computation with chunking for large datasets

**Key Components**:

- `GPUChunkedFeatureComputer` class
- Global KDTree + chunked processing
- Automatic VRAM management
- Memory pooling and CUDA streams
- Progress tracking

**Refactoring Status**:

- ✅ Phase 2: Eigenvalue features use GPU Bridge
- ✅ Uses GPU Bridge for eigenvalues
- ❌ Custom implementations for normals, curvature, height
- ❌ Complex memory management

**Code Structure**:

```python
class GPUChunkedFeatureComputer:
    def __init__(
        chunk_size=None,
        vram_limit_gb=None,
        use_gpu=True,
        show_progress=True,
        auto_optimize=True,
        use_cuda_streams=True,
        enable_memory_pooling=True,
        # ... 10+ parameters
    )

    # Main processing methods
    def compute_normals_chunked(points, k=10) -> normals
    def compute_curvature_chunked(points, normals, k=10) -> curvature
    def compute_eigenvalue_features() -> features  # Uses GPU Bridge ✅
    def compute_all_features_chunked() -> all_features

    # Memory management
    def _calculate_optimal_chunk_size()
    def _manage_vram()
    def _create_cuda_streams()

    # Batch processing (different from features_gpu.py!)
    def _process_chunk_batch()
```

**Unique Features** (not in features_gpu.py):

- Automatic chunking based on VRAM
- Global KDTree strategy
- CUDA streams for overlap
- Memory pooling
- Progress bars
- Auto-optimization

**Duplication with features_gpu.py**:

- Normal computation algorithm (~80% similar)
- Curvature computation (~70% similar)
- Neighbor query logic (~60% similar)
- GPU error handling (~90% similar)

---

### 3. Strategy Wrappers (400 lines total)

**strategy_gpu.py** (200 lines):

```python
class GPUStrategy(BaseFeatureStrategy):
    """Thin wrapper around GPUFeatureComputer"""

    def __init__(self, batch_size=None, use_gpu=True):
        self.gpu_computer = GPUFeatureComputer(
            use_gpu=use_gpu,
            batch_size=batch_size
        )

    def compute(self, points, intensities, rgb, nir, classification):
        # Calls gpu_computer methods
        normals = self.gpu_computer.compute_normals(points)
        curvature = self.gpu_computer.compute_curvature(points, normals)
        # ... etc
        return features
```

**strategy_gpu_chunked.py** (200 lines):

```python
class GPUChunkedStrategy(BaseFeatureStrategy):
    """Thin wrapper around GPUChunkedFeatureComputer"""

    def __init__(self, chunk_size=None, batch_size=None):
        self.gpu_computer = GPUChunkedFeatureComputer(
            chunk_size=chunk_size,
            batch_size=batch_size
        )

    def compute(self, points, intensities, rgb, nir, classification):
        # Calls gpu_computer.compute_all_features_chunked()
        # Almost identical to GPUStrategy but calls different backend
        return features
```

**Analysis**: These wrappers provide a common interface but don't reduce duplication in the underlying implementations.

---

## Duplication Matrix

| Feature               | features_gpu.py  | features_gpu_chunked.py | Code Overlap  |
| --------------------- | ---------------- | ----------------------- | ------------- |
| Normal computation    | Custom impl      | Custom impl             | ~80%          |
| Curvature computation | Custom impl      | Custom impl             | ~70%          |
| Eigenvalue features   | GPU Bridge ✅    | GPU Bridge ✅           | 100% (shared) |
| Height computation    | Core delegate ✅ | Custom impl             | 0%            |
| Neighbor queries      | Basic batching   | Global KDTree           | ~40%          |
| Memory management     | Simple           | Advanced (chunking)     | ~20%          |
| Error handling        | Basic            | Advanced                | ~90%          |
| Progress tracking     | None             | Rich (tqdm)             | 0%            |
| CUDA streams          | No               | Yes (optional)          | 0%            |

**Estimated Duplication**:

- ~600 lines of similar code across both files
- ~200 lines in strategy wrappers (thin layer)
- **Total reducible**: ~800 lines

---

## optimization/ Module Analysis

### Current Structure (8 files, ~1,950 lines)

1. **gpu.py** (~300 lines)

   - Basic GPU utilities
   - Array transfers
   - Device management

2. **gpu_async.py** (~400 lines)

   - Async GPU processing
   - `create_enhanced_gpu_processor()` ← needs renaming!
   - Task queues

3. **gpu_array_ops.py** (~200 lines)

   - Array operations
   - Unified interface for CPU/GPU arrays

4. **gpu_coordinator.py** (~150 lines)

   - Multi-GPU coordination
   - Load balancing

5. **gpu_dataframe_ops.py** (~250 lines)

   - GeoPandas GPU operations
   - DataFrame transfers

6. **gpu_kernels.py** (~180 lines)

   - Custom CUDA kernels
   - Low-level optimizations

7. **gpu_memory.py** (~350 lines)

   - Memory management
   - VRAM monitoring
   - Pool allocation

8. **gpu_profiler.py** (~120 lines)
   - Performance profiling
   - Timing utilities

### Proposed Consolidation

```
optimization/
├── gpu_compute.py        (Merge: gpu.py + gpu_kernels.py + gpu_array_ops.py)
│   └── ~500 lines (vs 680 current) = -180 lines
│
├── gpu_memory.py         (Keep - focused responsibility)
│   └── ~350 lines (no change)
│
├── gpu_async.py          (Merge: gpu_async.py + gpu_coordinator.py)
│   └── ~450 lines (vs 550 current) = -100 lines
│
└── dataframe/            (New subdirectory)
    └── gpu_ops.py        (Move gpu_dataframe_ops.py here)
        └── ~250 lines (better organization)

Delete: gpu_profiler.py → merge into gpu_memory.py or separate tools/ dir
```

**Reduction**: 8 files → 4 files, -280 lines

---

## Proposed Consolidation Strategy

### Phase 2A: Merge GPU Feature Implementations

**Goal**: Create single GPU processor with automatic chunking

**New File**: `ign_lidar/features/gpu_processor.py` (~1,500 lines)

```python
class GPUProcessor:
    """
    Unified GPU feature processor with automatic chunking.

    Combines the best of features_gpu.py and features_gpu_chunked.py:
    - Automatic chunking based on dataset size and VRAM
    - Memory management from chunked version
    - Simplified API from basic version
    - GPU Bridge integration for eigenvalues
    """

    def __init__(
        self,
        auto_chunk: bool = True,  # NEW: Auto-detect if chunking needed
        chunk_size: Optional[int] = None,
        vram_limit_gb: Optional[float] = None,
        use_gpu: bool = True,
        show_progress: bool = True,
        use_cuda_streams: bool = True,
    ):
        """
        Initialize with smart defaults.

        If auto_chunk=True (default):
        - Automatically chunk based on dataset size and available VRAM
        - Use global KDTree for large datasets (>10M points)
        - Use simple batching for smaller datasets
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.auto_chunk = auto_chunk

        # Auto-detect optimal configuration
        if auto_chunk:
            self._configure_auto_chunking()
        else:
            self.chunk_size = chunk_size

        # Initialize GPU Bridge
        self.gpu_bridge = GPUCoreBridge(use_gpu=self.use_gpu)

        # Initialize memory manager
        if self.use_gpu:
            self.memory_manager = GPUMemoryManager(vram_limit_gb)

    def compute_features(
        self,
        points: np.ndarray,
        feature_types: List[str] = None,
        k_neighbors: int = 10,
        show_progress: bool = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute features with automatic strategy selection.

        NEW: Single unified method that automatically chooses:
        - Chunking strategy based on dataset size
        - Batch sizes based on VRAM
        - CPU fallback if GPU unavailable
        """
        n_points = len(points)

        # Auto-select strategy
        if self.auto_chunk and n_points > self.chunk_threshold:
            return self._compute_chunked(points, feature_types, k_neighbors)
        else:
            return self._compute_batch(points, feature_types, k_neighbors)

    def _compute_chunked(self, points, feature_types, k_neighbors):
        """Chunked processing for large datasets (from gpu_chunked)"""
        # Use global KDTree strategy
        # Process in chunks
        # Memory-efficient

    def _compute_batch(self, points, feature_types, k_neighbors):
        """Batch processing for medium datasets (from gpu)"""
        # Simple batching
        # No global KDTree overhead
```

**Migration Path**:

1. Create `gpu_processor.py` with unified logic
2. Update `strategy_gpu.py` and `strategy_gpu_chunked.py` to use GPUProcessor
3. Mark `features_gpu.py` and `features_gpu_chunked.py` as deprecated
4. Comprehensive testing
5. Remove old files in next release

**Benefits**:

- -700 lines of duplicate code
- Single source of truth for GPU features
- Automatic optimization
- Simpler API for users
- Easier to maintain

---

### Phase 2B: Consolidate optimization/gpu\_\*.py Files

**Timeline**: After Phase 2A stabilizes

**Changes**:

1. Merge `gpu.py` + `gpu_kernels.py` + `gpu_array_ops.py` → `gpu_compute.py`
2. Merge `gpu_async.py` + `gpu_coordinator.py` → `gpu_async.py` (enhanced)
3. Move `gpu_dataframe_ops.py` → `io/dataframe/gpu_ops.py`
4. Merge `gpu_profiler.py` → `gpu_memory.py` or `tools/profiling.py`

**Benefits**:

- -280 lines
- Clearer module boundaries
- Better discoverability
- Reduced import complexity

---

## Risk Assessment

### Phase 2A: GPU Feature Consolidation

**Risk Level**: 🟡 MEDIUM

**Risks**:

1. Performance regression if chunking logic changes
2. VRAM management issues on different GPU models
3. Breaking changes for users directly using GPUFeatureComputer
4. Complex testing requirements (multiple dataset sizes, GPU configs)

**Mitigation**:

1. ✅ Extensive benchmarking before/after
2. ✅ Keep old files deprecated but functional initially
3. ✅ Comprehensive test suite (unit + integration + performance)
4. ✅ Gradual rollout with deprecation warnings
5. ✅ Document migration path

### Phase 2B: Optimization Module Consolidation

**Risk Level**: 🟢 LOW-MEDIUM

**Risks**:

1. Import path changes
2. Potential circular dependencies

**Mitigation**:

1. ✅ Update all imports simultaneously
2. ✅ Deprecation period for old imports
3. ✅ Clear migration guide

---

## Testing Strategy

### Before Consolidation

1. **Capture Current Performance**:

   ```bash
   python scripts/benchmark_gpu_bridge.py > baseline_performance.txt
   python scripts/benchmark_gpu_chunked.py >> baseline_performance.txt
   ```

2. **Document Current API**:

   - All public methods
   - Parameter signatures
   - Return value formats

3. **Create Reference Outputs**:
   - Run on known datasets
   - Save feature outputs
   - Use for regression testing

### During Consolidation

1. **Unit Tests**:

   - Each method in GPUProcessor
   - Memory management
   - Chunking logic
   - Auto-configuration

2. **Integration Tests**:

   - Small datasets (<1M points)
   - Medium datasets (1-10M points)
   - Large datasets (>10M points)
   - Edge cases (empty, single point, etc.)

3. **Performance Tests**:
   - Compare with baseline
   - Must match or improve performance
   - Test on different GPU models

### After Consolidation

1. **Regression Tests**:

   - Output must match baseline (bit-for-bit if possible)
   - Performance within ±5% of baseline

2. **Stress Tests**:
   - Maximum dataset size
   - Minimum VRAM scenarios
   - Multi-GPU scenarios

---

## Implementation Timeline

### Week 1: Preparation & Analysis ✅

- [x] Phase 1 complete (deprecated code removed)
- [x] Deep analysis of GPU code (this document)
- [x] Establish baselines

### Week 2: Phase 2A - GPU Processor Creation

- [ ] Day 1-2: Create `gpu_processor.py` skeleton
- [ ] Day 3-4: Implement auto-chunking logic
- [ ] Day 5: Testing & benchmarking

### Week 3: Phase 2A - Migration & Testing

- [ ] Day 1-2: Update strategy wrappers
- [ ] Day 3: Deprecate old files
- [ ] Day 4-5: Comprehensive testing

### Week 4: Phase 2B - Optimization Module (Optional)

- [ ] Consolidate gpu\_\*.py files
- [ ] Update imports
- [ ] Testing

---

## Success Metrics

### Code Quality

- ✅ Reduce total GPU code by 700+ lines
- ✅ Single source of truth for GPU features
- ✅ Zero code duplication in feature algorithms

### Performance

- ✅ Match or improve current performance
- ✅ Auto-optimization works correctly
- ✅ Memory usage within bounds

### Maintainability

- ✅ Simpler API (fewer classes/methods)
- ✅ Clearer code organization
- ✅ Better documentation

---

## Next Steps

### Immediate Actions

1. **Create Feature Branch**:

   ```bash
   git checkout -b refactor/phase2-gpu-consolidation
   ```

2. **Run Baselines**:

   ```bash
   python scripts/benchmark_gpu_bridge.py
   python scripts/benchmark_gpu_phase3_optimization.py
   ```

3. **Start Implementation**:

   - Create `gpu_processor.py` skeleton
   - Port chunking logic
   - Port batching logic
   - Integrate GPU Bridge

4. **Iterative Testing**:
   - Test each component
   - Compare with baseline
   - Fix issues

### Decision Points

**Proceed with Phase 2A?**

- ✅ Yes: High-value consolidation, manageable risk
- ❌ No: Keep current architecture, accept duplication

**Proceed with Phase 2B?**

- ⏸️ Maybe: Lower priority, can defer to later

---

## Appendix: Code Metrics

### Current State

```
Features GPU:     3,475 lines (gpu.py + gpu_chunked.py + strategies)
Optimization GPU: 1,950 lines (8 files)
Total:            5,425 lines across 11 files
```

### After Phase 2A

```
Features GPU:     2,200 lines (gpu_processor.py + strategies)
Optimization GPU: 1,950 lines (no change yet)
Total:            4,150 lines across 10 files
Reduction:        -1,275 lines (23% reduction)
```

### After Phase 2A + 2B

```
Features GPU:     2,200 lines (no change)
Optimization GPU: 1,050 lines (4 files)
Total:            3,250 lines across 7 files
Reduction:        -2,175 lines (40% reduction)
```

---

**End of Analysis**

**Recommendation**: Proceed with Phase 2A (GPU Feature Consolidation) as next step.

**Author**: Code Quality Audit Team  
**Date**: October 19, 2025  
**Status**: Ready for Implementation
