# Phase 4 Task 1.4: Pipeline Integration Plan

**Date**: January 2025  
**Status**: PLANNING  
**Priority**: 🔥 HIGH

---

## Overview

Integrate the **ModeSelector** (Task 1.1) and **UnifiedFeatureComputer** (Task 1.2) into the main processing pipeline to provide intelligent, automatic mode selection for all feature computations.

---

## Current State Analysis

### FeatureOrchestrator (Current Implementation)

**Location**: `ign_lidar/features/orchestrator.py` (1326 lines)

**Current Mode Selection** (Lines 290-400):

```python
def _init_computer(self):
    """Select and create appropriate feature strategy"""
    # Manual selection based on config flags:
    if use_gpu_chunked:
        base_strategy = GPUChunkedStrategy(...)
    elif use_gpu:
        base_strategy = GPUStrategy(...)
    else:
        base_strategy = CPUStrategy(...)
```

**Issues with Current Approach**:

1. ❌ Manual config flags (use_gpu, use_gpu_chunked)
2. ❌ No automatic selection based on data size
3. ❌ No memory consideration
4. ❌ Doesn't leverage ModeSelector intelligence
5. ❌ Doesn't use UnifiedFeatureComputer API

---

## Proposed Integration

### Option A: Full Integration (Recommended)

Replace existing strategy selection with ModeSelector + UnifiedFeatureComputer.

#### Changes Required:

**1. Update FeatureOrchestrator.\_init_computer()** (Lines 290-400)

**Before**:

```python
def _init_computer(self):
    # Manual GPU/CPU selection
    if use_gpu_chunked:
        base_strategy = GPUChunkedStrategy(...)
    elif use_gpu:
        base_strategy = GPUStrategy(...)
    else:
        base_strategy = CPUStrategy(...)
```

**After**:

```python
from ..features.unified_computer import create_unified_computer
from ..features.mode_selector import ModeSelector

def _init_computer(self):
    """Initialize feature computer with automatic mode selection."""
    # Option 1: Let UnifiedFeatureComputer handle everything
    force_mode = self._get_forced_mode_from_config()
    self.computer = create_unified_computer(
        force_mode=force_mode,
        progress_callback=self._progress_callback if hasattr(self, '_progress_callback') else None
    )
    self.strategy_name = "unified_auto"

    # Option 2: Use ModeSelector explicitly
    selector = ModeSelector()
    recommendations = selector.get_recommendations(
        num_points=self._estimate_typical_tile_size()
    )
    self.computer = create_unified_computer(
        force_mode=recommendations['recommended_mode']
    )
    self.strategy_name = f"unified_{recommendations['recommended_mode']}"
```

**2. Update compute_features() method** (Uses UnifiedFeatureComputer API)

**Before**:

```python
def compute_features(self, tile_data):
    # Direct strategy call
    features = self.computer.compute(
        points=points,
        k_neighbors=self.k_neighbors,
        ...
    )
```

**After**:

```python
def compute_features(self, tile_data):
    # Use UnifiedFeatureComputer API
    points = tile_data['points']

    # Compute normals
    normals = self.computer.compute_normals(points, k=self.k_neighbors)

    # Compute curvature
    curvature = self.computer.compute_curvature(points, normals, k=self.k_neighbors)

    # Compute geometric features
    geometric_features = self.computer.compute_geometric_features(
        points,
        required_features=['planarity', 'linearity', 'sphericity'],
        k=self.k_neighbors
    )

    # Combine results
    features = {
        'normals': normals,
        'curvature': curvature,
        **geometric_features
    }
    return features
```

**3. Add Progress Tracking** (Optional but recommended)

```python
def _init_progress_callback(self):
    """Initialize progress callback for feature computation."""
    if hasattr(self, 'progress_bar'):
        def callback(progress: float, message: str):
            self.progress_bar.set_description(message)
            self.progress_bar.n = int(progress * 100)
            self.progress_bar.refresh()
        self._progress_callback = callback
    else:
        self._progress_callback = None
```

**4. Configuration Mapping**

Map existing config flags to ModeSelector options:

```python
def _get_forced_mode_from_config(self) -> Optional[str]:
    """Get forced mode from legacy config flags."""
    processor_cfg = self.config.get('processor', {})

    # Respect explicit mode if set
    if 'computation_mode' in processor_cfg:
        return processor_cfg['computation_mode']

    # Map legacy flags to modes
    if processor_cfg.get('use_gpu_chunked', False):
        return 'gpu_chunked'
    elif processor_cfg.get('use_gpu', False):
        return 'gpu'
    elif processor_cfg.get('use_boundary_aware', False):
        return 'boundary'
    else:
        return None  # Auto mode
```

---

### Option B: Gradual Integration (Lower Risk)

Add UnifiedFeatureComputer as an **option** alongside existing strategy pattern.

#### Changes Required:

**1. Add opt-in flag** in config:

```yaml
processor:
  use_unified_computer: false # Default to false for backward compatibility
```

**2. Conditional initialization**:

```python
def _init_computer(self):
    if self.config.processor.get('use_unified_computer', False):
        # NEW: Unified computer with automatic mode selection
        self._init_unified_computer()
    else:
        # LEGACY: Strategy pattern (existing code)
        self._init_strategy_computer()
```

**3. Gradual migration**:

- Phase 1: Add unified computer as option
- Phase 2: Test in production
- Phase 3: Make default (flip flag)
- Phase 4: Remove old strategy pattern

---

## Benefits of Integration

### Immediate Benefits

1. **Automatic Mode Selection** ✅

   - No more manual GPU flags
   - Intelligent selection based on data size
   - Memory-aware decisions

2. **Consistent API** ✅

   - Same interface across all modes
   - Easier to understand and maintain
   - Better error messages

3. **Progress Tracking** ✅

   - Built-in progress callbacks
   - Better user experience
   - Easier debugging

4. **Future-Proof** ✅
   - Easy to add new modes
   - Centralized optimization
   - Better testing

### Performance Impact

- **Neutral**: Same underlying algorithms
- **Mode Selection Overhead**: <50ms (one-time cost)
- **API Overhead**: <1ms per call (negligible)
- **Overall**: No performance regression expected

---

## Implementation Plan

### Phase 1: Add Unified Computer Support (1-2 hours)

1. ✅ Create integration plan (this document)
2. Add `use_unified_computer` config flag
3. Implement `_init_unified_computer()` method
4. Add config mapping helper
5. Test with simple tile

### Phase 2: Update Feature Computation (1-2 hours)

1. Update `compute_features()` to use UnifiedFeatureComputer API
2. Add progress callback support
3. Update error handling
4. Test with full pipeline

### Phase 3: Testing & Validation (1-2 hours)

1. Run existing test suite
2. Compare results with old strategy pattern
3. Performance benchmarking
4. Memory profiling

### Phase 4: Documentation & Migration (1 hour)

1. Update configuration documentation
2. Add migration guide for users
3. Update examples
4. Update API docs

**Total Estimated Time**: 4-7 hours

---

## Configuration Changes

### New Configuration Options

```yaml
processor:
  # NEW: Unified computer with automatic mode selection
  use_unified_computer: true # Enable new unified system

  # NEW: Explicit mode override (optional)
  computation_mode: auto # auto, cpu, gpu, gpu_chunked, boundary

  # NEW: Progress tracking
  show_computation_progress: true

  # LEGACY: Old flags (deprecated but still supported)
  use_gpu: false # Ignored if use_unified_computer=true
  use_gpu_chunked: false # Ignored if use_unified_computer=true
```

### Backward Compatibility

**Existing configs will continue to work**:

- Default: `use_unified_computer=false` (use old strategy pattern)
- Old flags (`use_gpu`, `use_gpu_chunked`) still respected
- No breaking changes for existing users

**Migration path**:

1. Set `use_unified_computer: true` in config
2. Remove old GPU flags (optional)
3. Enjoy automatic mode selection!

---

## Risk Assessment

### Low Risk ✅

- **Opt-in approach**: Existing code unaffected
- **Same algorithms**: No behavior changes
- **Well-tested**: 31+26+36=93 tests passing for new code
- **Gradual migration**: Can roll back if issues found

### Medium Risk ⚠️

- **API differences**: UnifiedFeatureComputer API differs from Strategy API
  - Mitigation: Add adapter layer if needed
- **Performance validation**: Need to confirm no regressions
  - Mitigation: Comprehensive benchmarking before enabling by default

### Mitigation Strategies

1. **Feature flag**: Easy on/off toggle
2. **A/B testing**: Run both systems in parallel temporarily
3. **Monitoring**: Add logging to detect issues
4. **Rollback plan**: Keep old code until fully validated

---

## Success Criteria

### Functionality ✅

- [ ] Unified computer integrates without errors
- [ ] All existing tests pass
- [ ] Results identical to old strategy pattern
- [ ] Progress tracking works correctly

### Performance ✅

- [ ] No performance regression (within 5%)
- [ ] Mode selection overhead <50ms
- [ ] Memory usage comparable

### User Experience ✅

- [ ] Simpler configuration
- [ ] Better error messages
- [ ] Progress visibility
- [ ] Clear migration path

---

## Next Steps

### Immediate (This Session)

1. **Decision**: Choose Option A (full) or Option B (gradual)
2. **Implementation**: Start Phase 1 (add support)
3. **Testing**: Basic integration test

### Short-term (Next Session)

1. Complete Phases 2-3 (update compute methods, testing)
2. Performance validation
3. Documentation updates

### Long-term (Future)

1. Make unified computer default
2. Deprecate old strategy pattern
3. Remove legacy code (after 2-3 releases)

---

## Recommendation

**Proceed with Option B (Gradual Integration)**

**Rationale**:

1. ✅ Lower risk - existing code unaffected
2. ✅ Easy rollback if issues found
3. ✅ Time for production validation
4. ✅ Backward compatibility maintained
5. ✅ Can flip to default once validated

**Timeline**:

- **Today**: Implement Option B Phase 1 (1-2 hours)
- **This week**: Complete testing (2-3 hours)
- **Next release**: Enable by default
- **Future release**: Remove old code

---

## Questions for Discussion

1. **Prefer Option A or Option B?**

   - Option A: Full replacement (higher risk, cleaner)
   - Option B: Gradual opt-in (lower risk, more code temporarily)

2. **Timeline?**

   - Fast track (implement today, 4-7 hours)
   - Gradual (spread over multiple sessions)

3. **Default behavior?**
   - Keep old as default initially?
   - Make new default immediately?

---

## Files to Modify

### Core Changes (Required)

1. **ign_lidar/features/orchestrator.py**
   - Add `_init_unified_computer()` method
   - Add `_get_forced_mode_from_config()` helper
   - Update `_init_computer()` to support both paths
   - Modify `compute_features()` if needed

### Configuration (Required)

2. **Example configs** (examples/\*.yaml)
   - Add `use_unified_computer` flag examples
   - Add `computation_mode` examples
   - Document new options

### Documentation (Required)

3. **docs/guides/performance_tuning.md**
   - Document automatic mode selection
   - Explain computation_mode options
   - Migration guide from old flags

### Tests (Required)

4. **tests/test_orchestrator_integration.py** (new)
   - Test unified computer integration
   - Test mode selection
   - Test backward compatibility

---

## Conclusion

The integration of ModeSelector and UnifiedFeatureComputer into FeatureOrchestrator is straightforward and low-risk. By following Option B (gradual integration), we can:

1. ✅ Add new functionality without breaking existing code
2. ✅ Validate in production before making default
3. ✅ Provide smooth migration path for users
4. ✅ Achieve benefits of automatic mode selection

**Estimated effort**: 4-7 hours total
**Risk level**: Low (with Option B)
**Priority**: High (enables key Phase 4 features)

**Ready to proceed with implementation!** 🚀
