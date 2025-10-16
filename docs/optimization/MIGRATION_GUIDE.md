# Ground Truth Optimization Migration Guide

## Overview

Ground truth optimization has been integrated into the main `ign-lidar-hd` package under the `ign_lidar.optimization` module. This guide explains how to update your code.

## What Changed

### Package Structure

**Before:**

```
/
├── optimize_ground_truth.py
├── optimize_ground_truth_strtree.py
├── optimize_ground_truth_vectorized.py
├── optimize_ground_truth_gpu.py
├── ground_truth_quick_fix.py
├── profile_ground_truth.py
└── benchmark_ground_truth.py
```

**After:**

```
ign_lidar/
└── optimization/
    ├── __init__.py         # Public API
    ├── auto_select.py      # Automatic optimization selection
    ├── strtree.py          # STRtree spatial indexing (10-30× speedup)
    ├── vectorized.py       # GeoPandas vectorized (30-100× speedup)
    ├── gpu.py              # GPU acceleration (100-1000× speedup)
    └── prefilter.py        # Pre-filtering (2-5× speedup)

scripts/
├── profile_ground_truth.py
└── benchmark_ground_truth.py

docs/optimization/
├── GROUND_TRUTH_PERFORMANCE_ANALYSIS.md
├── GROUND_TRUTH_QUICK_START.md
├── OPTIMIZATION_README.md
└── MIGRATION_GUIDE.md
```

## Migration Steps

### 1. Automatic Optimization (Recommended)

**Before:**

```python
from optimize_ground_truth import auto_optimize

# Apply best available optimization
auto_optimize()

# Or force specific level
auto_optimize(force_level="gpu")
```

**After:**

```python
from ign_lidar.optimization import auto_optimize

# Apply best available optimization
auto_optimize()

# Or force specific level
auto_optimize(force_level="gpu")
```

### 2. Specific Optimization

**Before:**

```python
# STRtree
from optimize_ground_truth_strtree import patch_advanced_classifier
patch_advanced_classifier()

# Vectorized
from optimize_ground_truth_vectorized import patch_advanced_classifier
patch_advanced_classifier()

# GPU
from optimize_ground_truth_gpu import patch_advanced_classifier
patch_advanced_classifier()

# Pre-filter
from ground_truth_quick_fix import patch_classifier
patch_classifier()
```

**After:**

```python
# STRtree
from ign_lidar.optimization import apply_strtree_optimization
apply_strtree_optimization()

# Vectorized
from ign_lidar.optimization import apply_vectorized_optimization
apply_vectorized_optimization()

# GPU
from ign_lidar.optimization import apply_gpu_optimization
apply_gpu_optimization()

# Pre-filter
from ign_lidar.optimization import apply_prefilter_optimization
apply_prefilter_optimization()
```

Or use the unified API:

```python
from ign_lidar.optimization.strtree import patch_advanced_classifier
patch_advanced_classifier()

from ign_lidar.optimization.vectorized import patch_advanced_classifier
patch_advanced_classifier()

from ign_lidar.optimization.gpu import patch_advanced_classifier
patch_advanced_classifier()

from ign_lidar.optimization.prefilter import patch_classifier
patch_classifier()
```

### 3. Configuration Files

If you have configuration files that reference optimization scripts, update them:

**Before:**

```yaml
# config.yaml
optimization:
  script: optimize_ground_truth.py
  level: auto
```

**After:**

```yaml
# config.yaml
optimization:
  enabled: true
  level: auto # or: gpu, vectorized, strtree, prefilter
```

And in your code:

```python
from ign_lidar.optimization import auto_optimize

config = load_config("config.yaml")
if config.get("optimization", {}).get("enabled", False):
    level = config["optimization"].get("level", "auto")
    if level == "auto":
        auto_optimize()
    else:
        auto_optimize(force_level=level)
```

### 4. Script Usage

**Before:**

```bash
python profile_ground_truth.py tile.laz --ground-truth ground_truth.geojson
python benchmark_ground_truth.py tile.laz ground_truth.geojson
```

**After:**

```bash
python scripts/profile_ground_truth.py tile.laz --ground-truth ground_truth.geojson
python scripts/benchmark_ground_truth.py tile.laz ground_truth.geojson
```

## Optional Dependencies

### GPU Acceleration

For GPU-accelerated optimization, install CuPy based on your CUDA version:

```bash
# CUDA 11.x
pip install ign-lidar-hd[gpu] cupy-cuda11x

# CUDA 12.x
pip install ign-lidar-hd[gpu] cupy-cuda12x
```

For full GPU spatial optimization with cuSpatial (best performance):

```bash
# Using conda (recommended)
conda install -c rapidsai -c conda-forge -c nvidia cuspatial cupy

# Or with pip extras
pip install ign-lidar-hd[gpu-spatial]
# Then manually: conda install -c rapidsai cuspatial
```

### GeoPandas Vectorized

GeoPandas is already included in base dependencies, no additional installation needed.

### STRtree

Shapely 2.0+ (already included) provides STRtree, no additional installation needed.

## Performance Expectations

| Optimization | Expected Speedup | Dependencies               | Best For                      |
| ------------ | ---------------- | -------------------------- | ----------------------------- |
| GPU          | 100-1000×        | cupy, cuspatial (optional) | Large tiles, GPU available    |
| Vectorized   | 30-100×          | geopandas (included)       | Medium-large tiles            |
| STRtree      | 10-30×           | shapely 2.0+ (included)    | Any size, no extra deps       |
| Pre-filter   | 2-5×             | numpy (included)           | Small tiles, minimal overhead |

## Backward Compatibility

The optimization module uses runtime patching and does not modify the original `AdvancedClassifier` source code. All existing code continues to work without changes.

However, to get the performance benefits, you must explicitly apply an optimization:

```python
from ign_lidar.optimization import auto_optimize

# Apply once at the start of your script
auto_optimize()

# Then use AdvancedClassifier as normal
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier
classifier = AdvancedClassifier()
# ... use classifier normally ...
```

## Example: Full Migration

**Before:**

```python
# old_script.py
from optimize_ground_truth import auto_optimize
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier

# Apply optimization
auto_optimize()

# Process tiles
classifier = AdvancedClassifier()
for tile_path in tiles:
    result = classifier.process(tile_path)
```

**After:**

```python
# new_script.py
from ign_lidar.optimization import auto_optimize
from ign_lidar.core.modules.advanced_classification import AdvancedClassifier

# Apply optimization (same API!)
auto_optimize()

# Process tiles (no changes needed!)
classifier = AdvancedClassifier()
for tile_path in tiles:
    result = classifier.process(tile_path)
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'ign_lidar.optimization'`:

1. Ensure you have the latest version: `pip install --upgrade ign-lidar-hd`
2. Check installation: `python -c "import ign_lidar.optimization; print('OK')"`
3. Reinstall if needed: `pip uninstall ign-lidar-hd && pip install ign-lidar-hd`

### GPU Optimization Fails

If GPU optimization fails to apply:

1. Check CuPy installation: `python -c "import cupy; print(cupy.__version__)"`
2. Verify CUDA version matches CuPy: `nvcc --version`
3. Fall back to vectorized: `auto_optimize(force_level="vectorized")`

### Performance Not Improving

If you don't see expected speedup:

1. Ensure optimization is applied **before** creating classifier
2. Check which optimization was selected: `auto_optimize(verbose=True)`
3. Profile your code: `python scripts/profile_ground_truth.py ...`
4. Verify ground truth data quality (complex polygons can reduce speedup)

## Getting Help

- **Documentation**: See `docs/optimization/OPTIMIZATION_README.md`
- **Quick Start**: See `docs/optimization/GROUND_TRUTH_QUICK_START.md`
- **Performance Analysis**: See `docs/optimization/GROUND_TRUTH_PERFORMANCE_ANALYSIS.md`
- **Issues**: Open an issue on GitHub with profiling results

## Summary

The optimization integration provides:

✅ **Professional package structure** - Clean PyPI distribution  
✅ **Backward compatible** - Existing code works unchanged  
✅ **Easy migration** - Mostly just import path changes  
✅ **Better documentation** - Centralized in docs/optimization/  
✅ **Optional GPU deps** - Install only what you need  
✅ **Automatic selection** - Best optimization chosen at runtime

**Migration checklist:**

- [ ] Update imports from `optimize_ground_truth` to `ign_lidar.optimization`
- [ ] Update script paths from `.` to `scripts/`
- [ ] Update documentation references to `docs/optimization/`
- [ ] Install GPU dependencies if needed
- [ ] Test with `auto_optimize(verbose=True)` to verify
- [ ] Profile to confirm performance improvements
