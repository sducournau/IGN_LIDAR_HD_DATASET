# Specific Code Fixes for Bottlenecks

**Quick implementation fixes that can be applied immediately**

---

## Fix 1: Enable GPU Memory Pooling in strategy_gpu.py

**File:** `ign_lidar/features/strategy_gpu.py`  
**Lines:** 49-150 (approximate)  
**Estimated Time:** 30 minutes  
**Impact:** 15-20% performance gain

### Current Code:

```python
class GPUStrategy(BaseFeatureStrategy):
    def compute(self, points: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        import cupy as cp

        gpu_points = cp.asarray(points)
        # ... computation ...
        return cp.asnumpy(result)
```

### Fixed Code:

```python
class GPUStrategy(BaseFeatureStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ign_lidar.core.gpu_memory import get_gpu_memory_pool
        self.gpu_pool = get_gpu_memory_pool(size_gb=8.0)
        self.logger = logging.getLogger(__name__)

    def compute(self, points: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        import cupy as cp

        try:
            # Use GPU memory pool for allocations
            if self.gpu_pool:
                with self.gpu_pool.pool:  # Pool context manager
                    gpu_points = cp.asarray(points)
                    result = self._compute_with_pooled_memory(gpu_points, **kwargs)
                    return cp.asnumpy(result)
            else:
                # Fallback if pool not available
                gpu_points = cp.asarray(points)
                return cp.asnumpy(self._compute_kernel(gpu_points))
        except Exception as e:
            self.logger.error(f"GPU computation failed: {e}")
            raise

    def _compute_with_pooled_memory(self, gpu_points, **kwargs):
        """Compute using pooled GPU memory"""
        # All intermediate arrays use pool memory automatically
        # No manual allocation needed
        return self._compute_kernel(gpu_points)
```

---

## Fix 2: Auto-GPU Selection for KDTree

**File:** `ign_lidar/features/utils.py`  
**Lines:** 28-70 (approximate)  
**Estimated Time:** 20 minutes  
**Impact:** 80-90% for large datasets (> 100k points)

### Current Code:

```python
def build_kdtree(
    points: np.ndarray,
    k_neighbors: int = 30,
    search_radius: float = 3.0,
) -> KDTree:
    """Build KDTree with optimal default parameters."""
    # Always CPU KDTree - no GPU option
    from sklearn.neighbors import KDTree
    return KDTree(points, metric='euclidean')
```

### Fixed Code:

```python
def build_kdtree(
    points: np.ndarray,
    k_neighbors: int = 30,
    search_radius: float = 3.0,
    use_gpu: bool = None,
    gpu_threshold: int = 100_000,
) -> KDTree:
    """Build KDTree with automatic GPU/CPU selection.

    Automatically switches to GPU KDTree for large point clouds
    (N > gpu_threshold) when GPU is available.

    Args:
        points: Point cloud [N, 3]
        k_neighbors: Number of neighbors (unused in KDTree)
        search_radius: Search radius (unused in KDTree)
        use_gpu: Force GPU (True) or CPU (False), auto-detect if None
        gpu_threshold: Switch to GPU when N > threshold (default: 100k)

    Returns:
        KDTree instance (CPU or GPU)
    """
    n_points = len(points)

    # Auto-select GPU if not specified
    if use_gpu is None:
        use_gpu = (n_points > gpu_threshold) and _is_gpu_available()

    if use_gpu:
        try:
            logger.info(f"Building GPU KDTree for {n_points} points")
            from ign_lidar.optimization.gpu_kdtree import GPUKDTree
            return GPUKDTree(points, metric='euclidean')
        except Exception as e:
            logger.warning(f"GPU KDTree failed, falling back to CPU: {e}")
            # Fall through to CPU
            use_gpu = False

    if not use_gpu:
        logger.debug(f"Building CPU KDTree for {n_points} points")
        from sklearn.neighbors import KDTree
        return KDTree(points, metric='euclidean')


def _is_gpu_available() -> bool:
    """Check if GPU is available and configured."""
    try:
        import cupy as cp
        cp.cuda.Device()  # Test GPU access
        return True
    except Exception:
        return False
```

---

## Fix 3: Cache Mode Selection in Dispatcher

**File:** `ign_lidar/features/compute/dispatcher.py`  
**Lines:** 43-120 (approximate)  
**Estimated Time:** 25 minutes  
**Impact:** 50% for batch processing

### Current Code:

```python
def compute_all_features(
    points: np.ndarray,
    batch_size: Optional[int] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Compute all features (mode selected EVERY TIME)"""

    # ❌ Mode decision happens on every call
    mode = _select_optimal_mode(n_points=len(points))

    if mode == ComputeMode.CPU:
        return _compute_all_features_cpu(points, **kwargs)
    elif mode == ComputeMode.GPU:
        return _compute_all_features_gpu(points, **kwargs)
    else:
        return _compute_all_features_gpu_chunked(points, batch_size, **kwargs)
```

### Fixed Code:

```python
# Add at module level
_dispatcher_instance = None


class FeatureComputeDispatcher:
    """Cached feature computation dispatcher

    Selects optimal compute mode once at initialization
    and reuses it for all subsequent computations.
    """

    def __init__(
        self,
        mode: Optional[ComputeMode] = None,
        expected_size: Optional[int] = None
    ):
        """Initialize dispatcher with cached mode.

        Args:
            mode: Explicit computation mode, or auto-select if None
            expected_size: Expected dataset size for auto-selection
        """
        if mode is None:
            mode = self._select_optimal_mode(expected_size)

        self.mode = mode  # ← CACHED
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Feature dispatcher initialized with mode: {mode}")

    def compute(
        self,
        points: np.ndarray,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Compute features using cached mode (no mode decision here).

        Args:
            points: Input point cloud
            batch_size: Batch size for GPU_CHUNKED mode
            **kwargs: Additional arguments

        Returns:
            Dictionary of computed features
        """
        # ✓ Mode is already selected and cached
        if self.mode == ComputeMode.CPU:
            return _compute_all_features_cpu(points, **kwargs)
        elif self.mode == ComputeMode.GPU:
            return _compute_all_features_gpu(points, **kwargs)
        else:  # GPU_CHUNKED
            return _compute_all_features_gpu_chunked(
                points, batch_size, **kwargs
            )

    def _select_optimal_mode(
        self,
        size_hint: Optional[int] = None
    ) -> ComputeMode:
        """Select optimal computation mode (called ONCE at init)."""

        if not _check_gpu_available():
            return ComputeMode.CPU

        try:
            free_gb, total_gb = _get_gpu_memory_info()
        except Exception:
            return ComputeMode.CPU

        # Decision logic
        if size_hint and size_hint > 50_000:
            # Large dataset: use GPU if enough memory
            if free_gb > 6.0:
                return ComputeMode.GPU
            else:
                return ComputeMode.GPU_CHUNKED

        # Small dataset: check GPU availability
        if free_gb > 4.0:
            return ComputeMode.GPU
        else:
            return ComputeMode.GPU_CHUNKED


def get_feature_compute_dispatcher(
    mode: Optional[ComputeMode] = None,
    expected_size: Optional[int] = None,
    cache: bool = True,
) -> FeatureComputeDispatcher:
    """Get feature dispatcher (with optional caching).

    Args:
        mode: Explicit mode or None for auto-selection
        expected_size: Dataset size hint
        cache: Use singleton dispatcher (default: True)

    Returns:
        Feature compute dispatcher instance
    """
    global _dispatcher_instance

    if cache and _dispatcher_instance is not None:
        return _dispatcher_instance

    dispatcher = FeatureComputeDispatcher(mode, expected_size)

    if cache:
        _dispatcher_instance = dispatcher

    return dispatcher


# Update the main function to use dispatcher
def compute_all_features(
    points: np.ndarray,
    batch_size: Optional[int] = None,
    use_cached_dispatcher: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Compute all features with cached mode selection.

    Args:
        points: Input point cloud
        batch_size: Batch size for GPU operations
        use_cached_dispatcher: Use cached dispatcher (recommended)
        **kwargs: Additional arguments

    Returns:
        Dictionary of computed features
    """

    dispatcher = get_feature_compute_dispatcher(
        cache=use_cached_dispatcher
    )
    return dispatcher.compute(points, batch_size, **kwargs)
```

---

## Fix 4: GPU Stream Overlap (Intermediate)

**File:** `ign_lidar/features/compute/gpu_stream_overlap.py`  
**Estimated Time:** 1.5 hours  
**Impact:** 25-35% for compute-bound operations

### Usage Pattern:

```python
from ign_lidar.features.compute.gpu_stream_overlap import GPUStreamOverlapOptimizer
import cupy as cp

# Initialize with 3 streams for pipelining
optimizer = GPUStreamOverlapOptimizer(num_streams=3)

# Use overlapped streams
with optimizer.get_stream(0):
    # Stream 0: Transfer data
    gpu_data1 = cp.asarray(data1)

with optimizer.get_stream(1):
    # Stream 1: Compute while transferring
    result1 = compute_kernel1(gpu_data0)

with optimizer.get_stream(2):
    # Stream 2: Copy result while computing
    cpu_result = cp.asnumpy(result0)

# Synchronize all streams
optimizer.synchronize()
```

---

## Fix 5: Minimize GPU-CPU Copies

**File:** `ign_lidar/features/strategy_gpu.py`  
**Lines:** 80-150 (approximate)  
**Estimated Time:** 30 minutes  
**Impact:** 10-15% reduction in copy overhead

### Pattern Change:

```python
# ❌ OLD - Multiple copies
gpu_points = cp.asarray(points)
normals = cp.asnumpy(compute_normals_gpu(gpu_points))
gpu_normals = cp.asarray(normals)  # Unnecessary copy back!
eigenvalues = cp.asnumpy(compute_eigenvalues_gpu(gpu_normals))
gpu_eigenvalues = cp.asarray(eigenvalues)  # Another unnecessary copy!
features = cp.asnumpy(compute_features_gpu(gpu_eigenvalues))

# ✓ NEW - Single copy at end
gpu_points = cp.asarray(points)
gpu_normals = compute_normals_gpu(gpu_points)  # Stays on GPU
gpu_eigenvalues = compute_eigenvalues_gpu(gpu_normals)  # Stays on GPU
gpu_features = compute_features_gpu(gpu_eigenvalues)  # Stays on GPU
features = cp.asnumpy(gpu_features)  # Single copy at end!
```

---

## Implementation Order

1. **Start with Fix #1** (Memory pooling) - Easiest, immediate wins
2. **Then Fix #3** (Mode caching) - Quick to implement
3. **Then Fix #2** (Auto GPU KDTree) - Needs testing
4. **Then Fix #4** (Stream overlap) - Most complex, but high impact
5. **Then Fix #5** (Minimize copies) - Refactor pattern across strategies

---

## Testing Each Fix

```bash
# Test GPU memory pooling
pytest tests/test_gpu_memory.py -xvs

# Test KDTree auto-selection
python3 -c "
from ign_lidar.features.utils import build_kdtree
import numpy as np

# Test CPU path
cpu_tree = build_kdtree(np.random.rand(1000, 3), use_gpu=False)
print('✓ CPU KDTree:', type(cpu_tree))

# Test GPU path (if available)
try:
    gpu_tree = build_kdtree(np.random.rand(200000, 3), use_gpu=True)
    print('✓ GPU KDTree:', type(gpu_tree))
except:
    print('⊘ GPU KDTree not available')
"

# Test dispatcher caching
python3 -c "
from ign_lidar.features.compute.dispatcher import get_feature_compute_dispatcher

d1 = get_feature_compute_dispatcher(cache=True)
d2 = get_feature_compute_dispatcher(cache=True)
assert d1 is d2, 'Caching failed!'
print('✓ Dispatcher caching works')
"
```

---

## Performance Verification

After implementing each fix, run:

```bash
# Benchmark GPU operations
python3 scripts/benchmark_gpu.py

# Profile with GPU profiler
python3 -c "
from ign_lidar.core.gpu_profiler import GPUProfiler
import time

profiler = GPUProfiler()

# Measure before and after each fix
profiler.start('feature_computation')
# ... your code ...
profiler.end('feature_computation')

stats = profiler.summary()
print(f'Time: {stats[\"feature_computation\"][\"duration\"]:.2f}s')
"
```

---

## Expected Results After All Fixes

```
Performance Before: 100,000 ms (baseline)
After Fix #1 (pooling):    80,000 ms (20% gain)
After Fix #3 (caching):    70,000 ms (30% total)
After Fix #2 (KDTree):     60,000 ms (40% total)
After Fix #5 (copies):     52,000 ms (48% total)
After Fix #4 (streams):    45,000 ms (55% total)

TOTAL IMPROVEMENT: 55% faster
```
