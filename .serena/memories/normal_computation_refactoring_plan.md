# Normal Computation Refactoring Plan

**Date:** October 26, 2025  
**Task:** Consolidate 9 compute_normals implementations  
**Status:** Phase 1 - Analysis Complete, Implementation Starting

## Current Architecture (GOOD!)

The codebase already has a well-designed architecture:

```
User Code
    ↓
FeatureComputer.compute_normals() [dispatcher]
    ↓
CPUStrategy.compute() or GPUStrategy.compute() [strategy pattern]
    ↓
compute_all_features_optimized() [optimized JIT] OR compute_normals() [fallback]
```

### Canonical Implementations ✅

1. **CPU Fallback (no JIT)**: `ign_lidar/features/compute/normals.py::compute_normals()`
   - Pure NumPy + scikit-learn
   - Used when Numba not available
   - Line 18-142

2. **CPU Optimized (with JIT)**: `ign_lidar/features/compute/features.py::compute_all_features_optimized()`
   - Numba JIT compiled
   - Computes all features in one pass
   - 5-8x faster than individual functions
   - Line 312+

3. **GPU**: `ign_lidar/optimization/gpu_kernels.py::compute_normals_and_eigenvalues()`
   - CuPy/RAPIDS accelerated
   - Line 439+

### Wrapper Functions ✅ (Keep - they add value)

4. **Fast wrapper**: `ign_lidar/features/compute/normals.py::compute_normals_fast()`
   - Convenience function (k=10, returns only normals)
   - Line 141-151

5. **Accurate wrapper**: `ign_lidar/features/compute/normals.py::compute_normals_accurate()`
   - Convenience function (k=50, returns normals + eigenvalues)
   - Line 159-178

### Dispatcher Layer ✅ (Keep - this is the public API)

6. **FeatureComputer.compute_normals()**: `ign_lidar/features/feature_computer.py::compute_normals()`
   - Public API method
   - Routes to appropriate strategy (CPU/GPU/GPU_CHUNKED)
   - Line 160-215

## Problematic Implementations (TO DEPRECATE)

### 7. ❌ `ign_lidar/features/feature_computer.py::compute_normals_with_boundary()`
- **Location**: Line 370+
- **Issue**: Duplicate logic, should call canonical version
- **Action**: Add deprecation warning, refactor to call normals.py

### 8. ❌ `ign_lidar/features/gpu_processor.py::compute_normals()` (method)
- **Location**: Line 358
- **Issue**: This file duplicates FeatureComputer functionality
- **Action**: Verify if gpu_processor.py is still needed, if yes add deprecation warning

### 9. ❌ `ign_lidar/features/gpu_processor.py::compute_normals()` (standalone)
- **Location**: Line 1502
- **Issue**: Standalone function duplicating method above
- **Action**: Remove or deprecate

### 10. ❌ `ign_lidar/features/compute/features.py::compute_normals()`
- **Location**: Line 237
- **Issue**: Duplicate implementation, people should use unified API
- **Action**: Add deprecation warning pointing to normals.py or unified API

## Refactoring Strategy

### Phase 1: Add Deprecation Warnings ✅ (Current Phase)

For each problematic implementation:
1. Add deprecation warning at function start
2. Point users to canonical implementation
3. Keep function working (backward compatibility)

Example:
```python
def compute_normals_OLD_LOCATION(...):
    warnings.warn(
        "This function is deprecated and will be removed in v4.0. "
        "Use ign_lidar.features.compute.normals.compute_normals() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Keep existing implementation for now
    ...
```

### Phase 2: Update Internal Callers (v3.x)

1. Find all internal calls to deprecated functions
2. Update them to use canonical implementations
3. Run full test suite to verify

### Phase 3: Remove Deprecated Functions (v4.0)

1. Remove deprecated implementations
2. Update documentation
3. Add migration guide

## Implementation Order

### Step 1: Add deprecation to `compute_normals_with_boundary`
- File: `ign_lidar/features/feature_computer.py`
- Line: 370
- Refactor to call `ign_lidar.features.compute.normals.compute_normals()` with boundary handling

### Step 2: Analyze and deprecate `gpu_processor.py`
- Verify if this file is still used
- Check imports across codebase
- If redundant, add deprecation warnings to all methods
- If needed, document its purpose clearly

### Step 3: Remove standalone `compute_normals()` in gpu_processor.py
- Line 1502
- Either remove or add deprecation

### Step 4: Deprecate `features.py::compute_normals()`
- Line 237
- Add warning pointing to unified API or normals.py

## Success Criteria

1. ✅ All duplicate implementations have deprecation warnings
2. ✅ No test failures
3. ✅ Documentation updated
4. ✅ Memory updated with progress
5. ✅ Clear migration path for users

## Timeline

- **Phase 1 (Today)**: Add deprecation warnings - 2 hours
- **Phase 2 (v3.2)**: Update internal callers - 4 hours
- **Phase 3 (v4.0)**: Remove deprecated functions - 2 hours

## Notes

- The architecture is actually GOOD - we have proper separation
- The issue is duplicate implementations that bypass the proper layers
- We're not fixing broken architecture, we're cleaning up duplicate paths
- Keep the strategy pattern and dispatcher - they add value
