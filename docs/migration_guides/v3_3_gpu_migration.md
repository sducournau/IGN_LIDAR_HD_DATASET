# Migration Guide: v3.2 → v3.3 GPU Modules

**Version:** 3.3.0  
**Date:** November 23, 2025  
**Breaking Changes in:** v4.0 (planned Q1 2026)

---

## Overview

Version 3.3 reorganizes GPU modules for better clarity and maintainability:

- **GPU Profiler** consolidated into `core/`
- **Ground Truth Classifier** renamed for clarity
- **GPU Memory** split into modular `gpu_cache/` package

All changes are **backward compatible in v3.3** with deprecation warnings. Breaking changes will occur in **v4.0**.

---

## What Changed?

### 1. GPU Profiler: `optimization/` → `core/`

**Rationale:** Profiler is a core system utility, not an optimization component.

| Old (v3.0-v3.2)             | New (v3.3+)         | Status     |
| --------------------------- | ------------------- | ---------- |
| `optimization.gpu_profiler` | `core.gpu_profiler` | ✅ Moved   |
| `GPUOperationMetrics`       | `ProfileEntry`      | ✅ Renamed |
| `ProfilerSession`           | `ProfilingStats`    | ✅ Renamed |

**Migration:**

```python
# ❌ Old (v3.0-v3.2) - Deprecated, works with warning
from ign_lidar.optimization.gpu_profiler import GPUProfiler, GPUOperationMetrics

profiler = GPUProfiler(enable=True, session_name="eval")

# ✅ New (v3.3+) - Recommended
from ign_lidar.core.gpu_profiler import GPUProfiler, ProfileEntry

profiler = GPUProfiler(enabled=True, use_cuda_events=True)
```

**New Features in v3.3:**

- `get_bottleneck_analysis()`: Transfer vs compute analysis
- CUDA events support for precise timing (<1ms overhead)
- Improved performance reporting

---

### 2. Ground Truth Classifier: `gpu.py` → `ground_truth_classifier.py`

**Rationale:** Avoid confusion with `core/gpu.py` (GPUManager).

| Old (v3.0-v3.2)                             | New (v3.3+)                                                     | Status     |
| ------------------------------------------- | --------------------------------------------------------------- | ---------- |
| `optimization.gpu.GPUGroundTruthClassifier` | `optimization.ground_truth_classifier.GPUGroundTruthClassifier` | ✅ Renamed |

**Migration:**

```python
# ❌ Old (v3.0-v3.2) - Deprecated, works with warning
from ign_lidar.optimization.gpu import GPUGroundTruthClassifier

classifier = GPUGroundTruthClassifier()

# ✅ New (v3.3+) - Recommended
from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier

classifier = GPUGroundTruthClassifier()
```

**Type Hint Fix:**

- CuPy type hints now use string literals for optional dependency compatibility
- No functional changes, just better compatibility when CuPy is not installed

---

### 3. GPU Memory: Monolithic → Modular `gpu_cache/`

**Rationale:**

- Eliminate confusion with `core/gpu_memory.py` (system-level manager)
- Better organization: caching vs transfer optimization

| Old (v3.0-v3.2)                             | New (v3.3+)                                | Status   |
| ------------------------------------------- | ------------------------------------------ | -------- |
| `optimization.gpu_memory.GPUArrayCache`     | `optimization.gpu_cache.GPUArrayCache`     | ✅ Moved |
| `optimization.gpu_memory.GPUMemoryPool`     | `optimization.gpu_cache.GPUMemoryPool`     | ✅ Moved |
| `optimization.gpu_memory.TransferOptimizer` | `optimization.gpu_cache.TransferOptimizer` | ✅ Moved |

**New Structure:**

```
optimization/
├── gpu_cache/              # NEW in v3.3
│   ├── __init__.py        # Package exports
│   ├── arrays.py          # GPUArrayCache (smart caching)
│   └── transfer.py        # TransferOptimizer, GPUMemoryPool, utilities
└── gpu_memory.py          # Deprecation stub (backward compat)
```

**Migration:**

```python
# ❌ Old (v3.0-v3.2) - Deprecated, works with warning
from ign_lidar.optimization.gpu_memory import (
    GPUArrayCache,
    GPUMemoryPool,
    TransferOptimizer
)

# ✅ New (v3.3+) - Recommended
from ign_lidar.optimization.gpu_cache import (
    GPUArrayCache,
    GPUMemoryPool,
    TransferOptimizer
)
```

**Usage remains identical:**

```python
# No code changes needed beyond imports!
cache = GPUArrayCache(max_size_gb=8.0)
pool = GPUMemoryPool(max_arrays=20)
optimizer = TransferOptimizer()
```

---

## Complete Migration Example

### Before (v3.2)

```python
from ign_lidar.optimization.gpu_profiler import GPUProfiler
from ign_lidar.optimization.gpu import GPUGroundTruthClassifier
from ign_lidar.optimization.gpu_memory import GPUArrayCache, GPUMemoryPool

# Profiler
profiler = GPUProfiler(enable=True, session_name="processing")

# Caching
cache = GPUArrayCache(max_size_gb=8.0)
pool = GPUMemoryPool(max_arrays=20)

# Ground truth
classifier = GPUGroundTruthClassifier(chunk_size=5_000_000)

# Processing
with profiler.start_operation('labeling'):
    points_gpu = cache.get_or_upload('points', points_cpu)
    labels = classifier.label_points(points_gpu, buildings_gdf)
```

### After (v3.3)

```python
from ign_lidar.core.gpu_profiler import GPUProfiler
from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier
from ign_lidar.optimization.gpu_cache import GPUArrayCache, GPUMemoryPool

# Profiler (with new features!)
profiler = GPUProfiler(enabled=True, use_cuda_events=True)

# Caching (unchanged)
cache = GPUArrayCache(max_size_gb=8.0)
pool = GPUMemoryPool(max_arrays=20)

# Ground truth (unchanged)
classifier = GPUGroundTruthClassifier(chunk_size=5_000_000)

# Processing (new profiling API)
with profiler.profile('labeling', transfer='upload', size_mb=points_cpu.nbytes/1024**2):
    points_gpu = cache.get_or_upload('points', points_cpu)
    labels = classifier.label_points(points_gpu, buildings_gdf)

# NEW: Bottleneck analysis
analysis = profiler.get_bottleneck_analysis()
if analysis['bottleneck'] == 'transfer':
    print(f"⚠️ Transfer bottleneck: {analysis['transfer_pct']:.1f}%")
```

---

## Deprecation Timeline

| Version                    | Status                  | Description                                |
| -------------------------- | ----------------------- | ------------------------------------------ |
| **v3.3** (Nov 2025)        | ⚠️ Deprecation Warnings | Old imports work with `DeprecationWarning` |
| **v3.4-v3.9** (Q1-Q3 2026) | ⚠️ Continued Support    | Grace period for migration                 |
| **v4.0** (Q4 2026)         | 🚫 Breaking Changes     | Old imports removed, must use new paths    |

---

## Automated Migration Script

We provide a script to automatically update your codebase:

```bash
# Download migration script
wget https://raw.githubusercontent.com/sducournau/IGN_LIDAR_HD_DATASET/main/scripts/migrate_v3_3_gpu.py

# Run on your codebase
python migrate_v3_3_gpu.py /path/to/your/code

# Review changes
git diff

# Test thoroughly before committing!
pytest tests/
```

**Script performs:**

- ✅ Updates all import statements
- ✅ Renames `enable` → `enabled` parameter
- ✅ Updates `GPUOperationMetrics` → `ProfileEntry`
- ✅ Adds migration comments to your code

---

## Breaking Changes in v4.0

The following will **stop working** in v4.0:

### Removed Files

```python
# 🚫 These files will be DELETED in v4.0
optimization/gpu_profiler.py       # Use core.gpu_profiler
optimization/gpu.py                # Use ground_truth_classifier
optimization/gpu_memory.py         # Use gpu_cache
```

### Removed Aliases

```python
# 🚫 These aliases will be REMOVED in v4.0
GPUOperationMetrics  # Use ProfileEntry
ProfilerSession      # Use ProfilingStats
```

### Parameter Renames

```python
# 🚫 Old parameter names (v4.0)
GPUProfiler(enable=True)           # Use enabled=True

# ✅ New parameter names (v3.3+)
GPUProfiler(enabled=True)
```

### API Changes

```python
# 🚫 Session-based API (removed in v4.0)
profiler.start_operation('name')
profiler.end_operation('name')

# ✅ Context manager API (v3.3+)
with profiler.profile('name'):
    # ... code ...
```

---

## Testing Your Migration

### 1. Run Deprecation Checker

```bash
# Enable all deprecation warnings
python -W all your_script.py 2>&1 | grep DeprecationWarning

# Should output warnings for old imports
```

### 2. Update Tests

```python
import pytest
import warnings

def test_new_imports():
    """Verify new imports work."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Treat warnings as errors

        # Should NOT raise warnings
        from ign_lidar.core.gpu_profiler import GPUProfiler
        from ign_lidar.optimization.ground_truth_classifier import GPUGroundTruthClassifier
        from ign_lidar.optimization.gpu_cache import GPUArrayCache

def test_old_imports_deprecated():
    """Verify old imports trigger warnings."""
    with pytest.warns(DeprecationWarning):
        from ign_lidar.optimization.gpu_profiler import GPUProfiler
```

### 3. Run Full Test Suite

```bash
# All tests should pass
pytest tests/ -v

# Check for warnings
pytest tests/ -v -W all
```

---

## FAQ

### Q: Do I need to migrate immediately?

**A:** No. v3.3 is fully backward compatible. However, you should migrate before v4.0 (planned Q4 2026).

### Q: Will my code break if I upgrade to v3.3?

**A:** No. All old imports still work in v3.3, you'll just see deprecation warnings.

### Q: Can I use both old and new imports in the same codebase?

**A:** Yes, but not recommended. Migrate completely to avoid confusion.

### Q: What if I'm using a third-party library that hasn't migrated?

**A:** You can suppress deprecation warnings temporarily:

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='third_party_lib')
```

### Q: How do I find all deprecated imports in my codebase?

**A:** Use grep:

```bash
# Find old GPU profiler imports
grep -rn "from.*optimization.gpu_profiler import" .

# Find old GPU memory imports
grep -rn "from.*optimization.gpu_memory import" .

# Find old GPU imports
grep -rn "from.*optimization.gpu import" .
```

---

## Need Help?

- 📖 [GPU Stack Architecture](../architecture/gpu_stack.md)
- 💬 [GitHub Discussions](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions)
- 🐛 [Report Issues](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues)
- 📧 Contact: simon.ducournau@ign.fr

---

**Last Updated:** November 23, 2025  
**Authors:** IGN LiDAR HD Development Team
