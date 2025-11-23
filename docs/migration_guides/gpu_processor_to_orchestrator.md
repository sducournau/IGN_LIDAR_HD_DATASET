# Migration Guide: GPUProcessor to FeatureOrchestrator

## Overview

`GPUProcessor` has been deprecated in favor of `FeatureOrchestrator`, which provides
a unified API for CPU/GPU feature computation with better performance and cleaner code.

## Before (v3.5.x - Deprecated)

```python
from ign_lidar.features.gpu_processor import GPUProcessor

processor = GPUProcessor(use_gpu=True, k_neighbors=30)
features = processor.compute_features(points)
normals = processor.compute_normals(points)
```

## After (v3.6.0+)

```python
from ign_lidar.features import FeatureOrchestrator

config = {
    'features': {
        'k_neighbors': 30,
        'use_gpu': True
    }
}

orchestrator = FeatureOrchestrator(config)
features = orchestrator.compute_features(points, mode='lod2')

# For normals specifically
from ign_lidar.features.compute import compute_normals
normals, eigenvalues = compute_normals(points, k_neighbors=30, use_gpu=True)
```

## Benefits

1. **Unified API**: Single interface for all feature modes (LOD2, LOD3, ASPRS, etc.)
2. **Better Configuration**: Hydra-based config system with validation
3. **Strategy Pattern**: Automatic CPU/GPU/GPU_CHUNKED selection
4. **Performance**: +20-30% throughput with optimized GPU transfers
5. **Memory Management**: Adaptive memory handling for large datasets

## Migration Checklist

- [ ] Replace `GPUProcessor` imports with `FeatureOrchestrator`
- [ ] Convert initialization to config-based
- [ ] Update `compute_features()` calls to include `mode` parameter
- [ ] Replace direct `compute_normals()` with canonical implementation
- [ ] Test with existing data to ensure compatibility
- [ ] Update configuration files (YAML)

## Need Help?

- See examples in `examples/`
- Check documentation: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- Open issue: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues

## Timeline

- **v3.6.0**: Deprecation warnings added
- **v3.7.0-3.9.0**: Continued support with warnings
- **v4.0.0**: `GPUProcessor` removed
