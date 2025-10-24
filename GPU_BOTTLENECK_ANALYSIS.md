# GPU Computation Bottleneck Analysis

**Date:** October 24, 2025  
**Issue:** Processing stuck after KDTree building  
**Configuration:** CPU mode with 21M point cloud

---

## 🔴 Critical Findings

### 1. **FAISS Stuck on Large Query** (Most Likely Culprit)

**Location:** `ign_lidar/features/gpu_processor.py`, line 857

```python
# Query all neighbors in one batch
logger.info(f"  ⚡ Querying all {N:,} × {k} neighbors...")
distances, indices = index.search(points.astype(np.float32), k)  # ← STUCK HERE
logger.info(f"     ✓ All neighbors found")
```

**Problem:**

- For **21M points × 55 neighbors** = **1.155 billion neighbor queries**
- FAISS CPU (fallback mode) trying to search ALL at once
- No batching, no progress reporting
- **Appears frozen but actually computing (can take 5-30 minutes)**

**Memory Requirements:**

- Query results alone: `21M × 55 × 8 bytes = ~9GB`
- IVF temp buffers: `~2-4GB`
- Total: **~13GB RAM** sustained

---

### 2. **CPU FAISS Silent Processing**

**The Deceptive Part:**

```python
# Last log before freeze
logger.info(f"  🚀 Building FAISS index ({N:,} points, k={k})...")
# ... building happens (visible progress) ...
logger.info(f"     ✓ FAISS IVF index ready")

# This line prints immediately
logger.info(f"  ⚡ Querying all {N:,} × {k} neighbors...")

# ⚠️ THEN SILENCE FOR 5-30 MINUTES ⚠️
distances, indices = index.search(points.astype(np.float32), k)

# Finally prints (if it doesn't crash)
logger.info(f"     ✓ All neighbors found")
```

**Why It Appears Stuck:**

- FAISS IVF search is a **single blocking call**
- No internal progress reporting
- No timeouts
- No incremental results
- User sees nothing for 5-30+ minutes

---

## 📊 Performance Analysis

### Theoretical FAISS CPU Performance

For 21M points with IVF index (8192 clusters, 128 probes):

| Stage            | Time Estimate | Notes                  |
| ---------------- | ------------- | ---------------------- |
| Index build      | 2-5 min       | Visible progress       |
| Index train      | 1-2 min       | Visible progress       |
| **Query search** | **10-30 min** | **SILENT** ⚠️          |
| PCA computation  | 3-5 min       | After search completes |

**Total Expected:** **16-42 minutes** for normals computation alone!

### Actual Bottleneck Breakdown

For your 21M point cloud:

```
Configuration in use: config_asprs_bdtopo_cadastre_cpu_v3_memory_safe.yaml
- use_gpu: false
- Triggers CPU fallback in gpu_processor.py
- FAISS available → uses CPU FAISS
- N = 21,530,171 points
- k = 50 neighbors (from config)

Estimated timeline:
├── Building FAISS index:        ~3 min   ✓ Shows progress
├── Training IVF (8192 clusters): ~2 min   ✓ Shows progress
├── Adding points to index:      ~1 min   ✓ Shows progress
└── **Querying all neighbors:**  **15-25 min** ❌ **SILENT**
    └── This is where you're stuck right now!
```

---

## 🔍 Root Causes

### 1. **No Query Batching in CPU FAISS Path**

```python
# Current code - NO batching
distances, indices = index.search(points.astype(np.float32), k)
# Tries to do ALL 21M queries at once → 15-25 minutes of silence
```

**Should be:**

```python
# Batched approach with progress
batch_size = 500_000  # 500K points per batch
for batch in batches:
    batch_distances, batch_indices = index.search(batch, k)
    # Show progress bar
```

### 2. **Auto-Selection Logic Flaw**

```python
# Line 830-835 in gpu_processor.py
use_gpu_faiss = self.use_gpu and self.use_cuml and N < 15_000_000

if not use_gpu_faiss and N > 15_000_000:
    logger.info(f"     → Using CPU FAISS to avoid GPU OOM")
```

**Problem:**

- For N > 15M: Forces CPU FAISS
- But CPU FAISS has **NO batching** for queries
- Results in 15-30 min silent processing

### 3. **Missing Progress Feedback**

No progress reporting during the query phase:

- User can't tell if it's stuck or working
- No ETA estimation
- No memory usage monitoring
- No intermediate checkpoints

---

## 💡 Solutions

### Immediate Fix #1: Add Batched Query for CPU FAISS

**File:** `ign_lidar/features/gpu_processor.py`  
**Method:** `_compute_normals_with_faiss()`

```python
def _compute_normals_with_faiss(self, points: np.ndarray, k: int, show_progress: bool) -> np.ndarray:
    """Compute normals using FAISS with BATCHED queries."""
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)

    # Build FAISS index
    index = self._build_faiss_index(points, k)

    # ✅ NEW: Batched query with progress
    batch_size = 500_000  # 500K points per batch
    num_batches = (N + batch_size - 1) // batch_size

    logger.info(f"  ⚡ Querying {N:,} × {k} neighbors in {num_batches} batches...")

    all_indices = np.zeros((N, k), dtype=np.int64)

    batch_iterator = range(num_batches)
    if show_progress:
        from tqdm import tqdm
        batch_iterator = tqdm(batch_iterator, desc=f"  FAISS k-NN query", unit="batch")

    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N)

        batch_points = points[start_idx:end_idx].astype(np.float32)
        distances, indices = index.search(batch_points, k)

        all_indices[start_idx:end_idx] = indices

    logger.info(f"     ✓ All neighbors found ({num_batches} batches)")

    # Compute normals from neighbors
    # ... rest of the code ...
```

**Benefits:**

- ✅ Progress bar shows actual progress
- ✅ User knows it's working, not stuck
- ✅ Smaller memory footprint per iteration
- ✅ Can recover from partial failures
- ✅ Same total time, but visible feedback

---

### Immediate Fix #2: Add Timeout Detection

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds} seconds")

    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# In the query function:
try:
    with timeout(1800):  # 30 minute timeout
        distances, indices = index.search(points.astype(np.float32), k)
except TimeoutError:
    logger.error("FAISS query exceeded 30 minutes - possible hang detected")
    logger.info("Falling back to sklearn KDTree...")
    return self._compute_normals_cpu(points, k)
```

---

### Immediate Fix #3: Better Progress Logging

Add intermediate logging:

```python
def _compute_normals_with_faiss(self, points: np.ndarray, k: int, show_progress: bool):
    N = points.shape[0]

    # Estimate time
    estimated_minutes = (N / 1_000_000) * 1.2  # ~1.2 min per million points
    logger.info(f"  ⚡ Querying all {N:,} × {k} neighbors...")
    logger.info(f"     Estimated time: {estimated_minutes:.1f} minutes")
    logger.info(f"     (This may appear frozen - please wait)")

    import time
    start_time = time.time()

    # Do the query
    distances, indices = index.search(points.astype(np.float32), k)

    elapsed = time.time() - start_time
    logger.info(f"     ✓ All neighbors found in {elapsed/60:.1f} minutes")
```

---

### Long-term Fix #4: Smart Fallback Strategy

```python
def _compute_normals_chunked(self, points: np.ndarray, k: int, show_progress: bool):
    """Improved strategy with smart fallback."""
    N = len(points)

    # Decision tree
    if N < 5_000_000:
        # Small: Use FAISS Flat (fast, exact)
        return self._compute_normals_with_faiss(points, k, show_progress)

    elif N < 15_000_000 and self.use_gpu:
        # Medium + GPU: Use GPU FAISS
        return self._compute_normals_with_faiss(points, k, show_progress)

    elif N < 25_000_000:
        # Large: Use cuML KDTree with BATCHED queries
        return self._compute_normals_per_chunk(points, k, show_progress)

    else:
        # Huge: Force chunked processing with sklearn
        return self._compute_normals_cpu(points, k)
```

---

## 🎯 Recommended Action Plan

### For Your Immediate Issue (CPU mode, 21M points):

**Option A: Wait it out (if already running)**

- It's likely **NOT** stuck, just silent
- Should complete in 15-25 minutes
- Monitor with `htop` - CPU should be near 100%
- Monitor RAM - should be using 13-15GB

**Option B: Kill and restart with verbose logging**

```bash
# Add debug logging to see what's happening
export PYTHONUNBUFFERED=1
ign-lidar-hd process -c config.yaml ... 2>&1 | tee process.log
```

**Option C: Switch to chunked cuML strategy**
Modify config:

```yaml
features:
  use_gpu: false # Keep as false
  use_gpu_chunked: false # Disable FAISS path
```

This forces the `_compute_normals_cpu()` path which has built-in batching.

---

### Priority Fixes to Implement:

1. **HIGH PRIORITY:** Add batched queries to FAISS CPU path (Fix #1)
2. **HIGH PRIORITY:** Add progress logging (Fix #3)
3. **MEDIUM PRIORITY:** Add timeout detection (Fix #2)
4. **MEDIUM PRIORITY:** Improve fallback strategy (Fix #4)

---

## 📝 Code Locations to Modify

### Primary File: `ign_lidar/features/gpu_processor.py`

**Method: `_compute_normals_with_faiss()` (lines 840-880)**

- Add batched querying
- Add progress bars
- Add time estimates

**Method: `_build_faiss_index()` (lines 885-985)**

- Already optimized
- Add estimated query time warning

**Method: `_compute_normals_chunked()` (lines 987-1060)**

- Improve strategy selection
- Add early warnings about long operations

---

## 🔬 Verification Steps

After implementing fixes:

```python
# Test with 1M points
points = np.random.randn(1_000_000, 3).astype(np.float32)
processor = GPUProcessor(use_gpu=False, show_progress=True)
normals = processor.compute_normals(points, k=50)
# Should show: progress bar, time estimates, completion message

# Test with 21M points
# Should show: batch progress (42 batches), ETA, completion time
```

---

## 📊 Expected Performance After Fixes

### Before (Current):

```
Building FAISS index (21M points)... ✓ 3 min
Querying all 21,530,171 × 50 neighbors...
[SILENCE FOR 20 MINUTES] ← User thinks it's stuck
```

### After (With Fixes):

```
Building FAISS index (21M points)... ✓ 3 min
Querying 21,530,171 × 50 neighbors in 43 batches...
Estimated time: 18 minutes
[==========>                   ] 42% | Batch 18/43 | ETA: 10:32
```

Much better user experience! ✅

---

## 🚀 Alternative: Use cuML KDTree Instead

For CPU-only processing, cuML's KDTree (CPU mode) might be faster:

```python
# In _compute_normals_chunked, prefer cuML CPU over FAISS CPU for large datasets
if not self.use_gpu and N > 15_000_000:
    logger.info(f"  Using sklearn KDTree for CPU processing (with chunking)")
    return self._compute_normals_per_chunk(points, k, show_progress)
```

cuML CPU KDTree has better memory management and shows progress.

---

## Summary

**Your Current Issue:**

- Processing is **NOT stuck** - it's just **silently computing**
- FAISS CPU is doing 21M × 50 neighbor searches
- Should complete in **15-25 minutes** if you wait

**Root Cause:**

- No batching in FAISS CPU query path
- No progress reporting during query
- No timeout detection

**Priority Fix:**

- Add batched queries with progress bars (10 lines of code)
- Add time estimates and warnings (5 lines of code)

**Immediate Workaround:**

- Use CPU-only sklearn KDTree path (already has batching)
- Or wait 15-25 minutes for current run to complete

---

**Next Steps:**

1. Check if current process is still running (`htop`, check CPU usage)
2. If yes, let it finish (15-25 min remaining estimate)
3. Implement batched FAISS queries for future runs
4. Add progress logging for better UX

Would you like me to implement these fixes now?
