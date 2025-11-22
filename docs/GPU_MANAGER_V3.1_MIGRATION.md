# GPU Manager v3.1 Migration Guide

**Version**: 3.1.0  
**Date**: November 22, 2025  
**Breaking Changes**: None (fully backward compatible)

## Overview

GPUManager v3.1 introduces a **composition pattern** for unified GPU management. The new API provides cleaner access to GPU memory management and array caching while maintaining full backward compatibility.

## What's New

### Composition API (Recommended)

```python
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()

# v3.1+ Unified access to sub-components
gpu.memory.allocate(2.5)              # Memory management
gpu.cache.get_or_upload('key', array)  # Array caching
gpu.get_memory_info()                  # Convenience method
gpu.cleanup()                          # Full cleanup
```

### Before (Still Works!)

```python
from ign_lidar.core.gpu import GPUManager
from ign_lidar.core.gpu_memory import GPUMemoryManager
from ign_lidar.optimization.gpu_memory import GPUArrayCache

gpu = GPUManager()
gpu_mem = GPUMemoryManager()
cache = GPUArrayCache()
```

## Migration Paths

### Path 1: New Projects (Recommended)

Use the composition API from the start:

```python
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()

# Check GPU availability
if gpu.gpu_available:
    # Memory management
    if gpu.memory.allocate(required_gb=2.5):
        # Process with GPU
        process_on_gpu(data)

    # Array caching
    gpu_array = gpu.cache.get_or_upload('normals', normals_cpu)

    # Memory info
    info = gpu.get_memory_info()
    print(f"Free GPU memory: {info['free_gb']:.2f} GB")

    # Cleanup when done
    gpu.cleanup()
```

### Path 2: Existing Projects (Gradual Migration)

Your existing code will continue to work without changes:

```python
# OLD CODE - Still works in v3.1
from ign_lidar.core.gpu import GPU_AVAILABLE
from ign_lidar.core.gpu_memory import GPUMemoryManager

if GPU_AVAILABLE:
    gpu_mem = GPUMemoryManager()
    gpu_mem.allocate(2.5)

# NEW CODE - Use composition API
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()
if gpu.gpu_available:
    gpu.memory.allocate(2.5)
```

### Path 3: Mixed Approach (Transition Period)

You can mix old and new APIs during transition:

```python
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()

# Old style (still works)
from ign_lidar.core.gpu import GPU_AVAILABLE
if GPU_AVAILABLE:
    # Old code here
    pass

# New style (recommended)
if gpu.gpu_available:
    gpu.memory.allocate(2.5)
```

## API Comparison

| Old API                                  | New API (v3.1+)         | Notes                |
| ---------------------------------------- | ----------------------- | -------------------- |
| `GPUMemoryManager()`                     | `gpu.memory`            | Lazy-loaded property |
| `GPUArrayCache()`                        | `gpu.cache`             | Lazy-loaded property |
| `gpu_mem.get_memory_info()`              | `gpu.get_memory_info()` | Convenience method   |
| `cache.clear()` + `gpu_mem.free_cache()` | `gpu.cleanup()`         | Unified cleanup      |
| `GPU_AVAILABLE`                          | `gpu.gpu_available`     | Both still work      |

## Benefits of v3.1 Composition API

### 1. Single Import

```python
# Before: 3 imports
from ign_lidar.core.gpu import GPUManager
from ign_lidar.core.gpu_memory import GPUMemoryManager
from ign_lidar.optimization.gpu_memory import GPUArrayCache

# After: 1 import
from ign_lidar.core.gpu import GPUManager
```

### 2. Lazy Loading

Components are only loaded when needed:

```python
gpu = GPUManager()  # Lightweight

# These load on first access
mem = gpu.memory    # Loads GPUMemoryManager
cache = gpu.cache   # Loads GPUArrayCache
```

### 3. Unified Cleanup

```python
# Before: multiple cleanup calls
cache.clear()
gpu_mem.free_cache()
gc.collect()

# After: single call
gpu.cleanup()
```

### 4. Better Organization

```python
# Clear hierarchy
GPUManager
├── .memory     → Memory management
├── .cache      → Array caching
└── methods     → Convenience functions
```

## Common Usage Patterns

### Pattern 1: Feature Computation

```python
from ign_lidar.core.gpu import GPUManager
import numpy as np

def compute_features(points: np.ndarray, use_gpu: bool = True):
    gpu = GPUManager()

    if use_gpu and gpu.gpu_available:
        # Check memory
        required_gb = points.nbytes / (1024**3)
        if not gpu.memory.allocate(required_gb):
            print("Insufficient GPU memory, falling back to CPU")
            return compute_features_cpu(points)

        # Cache points
        points_gpu = gpu.cache.get_or_upload('points', points)

        # Process on GPU
        result = process_gpu(points_gpu)

        # Cleanup
        gpu.cleanup()
        return result
    else:
        return compute_features_cpu(points)
```

### Pattern 2: Batch Processing

```python
from ign_lidar.core.gpu import GPUManager

def process_tiles(tile_paths, use_gpu=True):
    gpu = GPUManager()

    for i, tile_path in enumerate(tile_paths):
        # Load tile
        points = load_tile(tile_path)

        if use_gpu and gpu.gpu_available:
            # Process with GPU
            result = process_gpu(gpu, points)
        else:
            result = process_cpu(points)

        save_result(result)

        # Cleanup after each batch
        if i % 10 == 0:
            gpu.cleanup()

    # Final cleanup
    gpu.cleanup()
```

### Pattern 3: Memory Monitoring

```python
from ign_lidar.core.gpu import GPUManager

def adaptive_processing(data, target_memory_usage=0.8):
    gpu = GPUManager()

    if not gpu.gpu_available:
        return process_cpu(data)

    # Check available memory
    info = gpu.get_memory_info()
    free_ratio = info['free_gb'] / info['total_gb']

    if free_ratio < target_memory_usage:
        # Use chunked processing
        return process_chunked_gpu(gpu, data)
    else:
        # Process all at once
        return process_full_gpu(gpu, data)
```

## Deprecation Timeline

### v3.1 (Current)

- ✅ New composition API added
- ✅ Full backward compatibility maintained
- ✅ No deprecation warnings

### v3.5 (Planned)

- ⚠️ Deprecation warnings for direct imports:
  - `from ign_lidar.core.gpu_memory import GPUMemoryManager`
  - `from ign_lidar.optimization.gpu_memory import GPUArrayCache`

### v4.0 (Future)

- ❌ Direct imports removed
- ✅ Only composition API available
- Migration guide will be provided

## Testing Your Migration

Run these tests to verify your code works with v3.1:

```python
from ign_lidar.core.gpu import GPUManager

# Test 1: Basic instantiation
gpu = GPUManager()
assert gpu is not None

# Test 2: Composition API
if gpu.gpu_available:
    assert hasattr(gpu, 'memory')
    assert hasattr(gpu, 'cache')

# Test 3: Convenience methods
info = gpu.get_memory_info()
assert isinstance(info, dict)

# Test 4: Backward compatibility
from ign_lidar.core.gpu import GPU_AVAILABLE
assert isinstance(GPU_AVAILABLE, bool)
```

## Troubleshooting

### Issue: "AttributeError: 'NoneType' object has no attribute 'allocate'"

**Cause**: Accessing `gpu.memory` without checking `gpu_available`

**Solution**:

```python
# Wrong
gpu = GPUManager()
gpu.memory.allocate(2.5)  # Fails if no GPU

# Right
gpu = GPUManager()
if gpu.gpu_available:
    gpu.memory.allocate(2.5)
```

### Issue: "Multiple GPUManager instances created"

**Cause**: Not using singleton pattern correctly

**Solution**:

```python
# All of these return the same instance
gpu1 = GPUManager()
gpu2 = GPUManager()
gpu3 = get_gpu_manager()

assert gpu1 is gpu2 is gpu3  # True
```

### Issue: "Lazy loading not working"

**Cause**: Accessing properties before GPU is available

**Solution**:

```python
# Check availability first
gpu = GPUManager()
if gpu.gpu_available:
    mem = gpu.memory  # Only loads if GPU available
```

## Getting Help

- **Documentation**: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Examples**: `examples/` directory

## Summary

✅ **v3.1 is fully backward compatible**  
✅ **New composition API is optional but recommended**  
✅ **No code changes required for existing projects**  
✅ **Gradual migration path available**

Start using `gpu.memory` and `gpu.cache` in new code, migrate existing code gradually.
