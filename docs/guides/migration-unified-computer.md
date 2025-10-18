# Migration Guide: Strategy Pattern → UnifiedFeatureComputer

**Date**: October 18, 2025  
**Version**: Phase 4 Task 1.4  
**Status**: Production Ready

## Overview

This guide helps you migrate from the legacy Strategy Pattern to the new UnifiedFeatureComputer, which provides automatic computation mode selection and a simplified API.

## TL;DR - Quick Migration

### Before (Legacy Strategy Pattern)

```yaml
processor:
  use_gpu: true
  use_gpu_chunked: true
  use_strategy_pattern: true
  gpu_batch_size: 5000000
```

### After (UnifiedFeatureComputer - Automatic)

```yaml
processor:
  use_unified_computer: true
```

**That's it!** The system will automatically select the best computation mode.

---

## Why Migrate?

### Benefits of UnifiedFeatureComputer

✅ **Automatic Mode Selection** - No manual GPU/CPU configuration  
✅ **Simplified Configuration** - One flag instead of multiple  
✅ **Expert Recommendations** - System logs optimal settings  
✅ **Consistent API** - Same interface across all modes  
✅ **Future-Proof** - New modes added automatically

### When NOT to Migrate (Yet)

⚠️ **Custom Strategy Implementations** - If you have custom strategies  
⚠️ **Very Stable Pipeline** - If "if it ain't broke, don't fix it" applies  
⚠️ **Need Time to Test** - Migration is opt-in, no rush

---

## Migration Paths

### Path 1: Automatic Mode Selection (Recommended)

**Best for**: Most users who want simplicity and automatic optimization

**Before**:

```yaml
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000

features:
  k_neighbors: 20
```

**After**:

```yaml
processor:
  use_unified_computer: true

features:
  k_neighbors: 20
```

**What Changed**:

- Removed `use_gpu`, `use_gpu_chunked`, `gpu_batch_size`
- Added single `use_unified_computer: true` flag
- System automatically selects CPU/GPU/GPU_CHUNKED based on workload

**Configuration File**: `examples/config_unified_auto.yaml`

---

### Path 2: Forced GPU Chunked Mode

**Best for**: Large workloads where you know GPU_CHUNKED is optimal

**Before**:

```yaml
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000
```

**After**:

```yaml
processor:
  use_unified_computer: true
  computation_mode: "gpu_chunked"
```

**What Changed**:

- Explicit control via `computation_mode`
- No need for separate flags
- System respects your choice but may log recommendations

**Configuration File**: `examples/config_unified_gpu_chunked.yaml`

---

### Path 3: Forced CPU Mode

**Best for**: Small workloads, debugging, or systems without GPU

**Before**:

```yaml
processor:
  use_gpu: false
  use_strategy_pattern: true
```

**After**:

```yaml
processor:
  use_unified_computer: true
  computation_mode: "cpu"
```

**What Changed**:

- Explicit CPU mode
- More clear than `use_gpu: false`

**Configuration File**: `examples/config_unified_cpu.yaml`

---

### Path 4: No Migration (Keep Legacy)

**Best for**: Stable pipelines that don't need changes

**Config** (No Changes Required):

```yaml
processor:
  use_gpu: true
  use_gpu_chunked: true
  use_strategy_pattern: true
```

**Or Explicitly**:

```yaml
processor:
  use_unified_computer: false # Explicit
  use_gpu: true
  use_gpu_chunked: true
```

**Configuration File**: `examples/config_legacy_strategy.yaml`

---

## Configuration Reference

### UnifiedFeatureComputer Options

```yaml
processor:
  # Enable UnifiedFeatureComputer
  use_unified_computer: true # Default: false

  # Optional: Force specific mode (overrides automatic selection)
  computation_mode: "gpu_chunked" # Options: cpu, gpu, gpu_chunked, boundary

  # Optional: Help mode selector estimate workload
  typical_points_per_tile: 2000000 # Default: estimated from tile_size

  # Optional: Tile size hint (used for estimation if typical_points_per_tile not set)
  tile_size: 1000 # meters
```

### Mode Selection Logic

When `computation_mode` is not specified, the system automatically selects:

1. **Check GPU Availability**

   - No GPU → CPU mode
   - GPU available → Continue

2. **Estimate Workload Size**

   - Use `typical_points_per_tile` if provided
   - Else estimate from `tile_size` (tile_size² × 25 points/m²)
   - Default: 2M points

3. **Select Mode Based on Size**

   - < 500K points → GPU mode (full tile on GPU)
   - ≥ 500K points → GPU_CHUNKED mode (process in chunks)

4. **Log Decision**
   ```
   ℹ️  Automatic mode selection: GPU_CHUNKED
       Reason: Large workload (2.5M points), GPU available
       Recommendation: Consider GPU_CHUNKED for optimal performance
   ```

### Legacy Strategy Pattern Options

```yaml
processor:
  # Disable UnifiedFeatureComputer (use legacy)
  use_unified_computer: false # Default

  # Legacy flags
  use_gpu: true # Enable GPU
  use_gpu_chunked: true # Use chunked strategy (recommended)
  use_strategy_pattern: true # Use strategy pattern (vs factory)
  use_boundary_aware: false # Wrap with boundary-aware strategy

  # GPU settings
  gpu_batch_size: 5000000 # Points per batch

features:
  # Legacy GPU settings (alternative location)
  use_gpu_chunked: null # Inherits from processor
  gpu_batch_size: null # Inherits from processor
```

---

## Migration Checklist

### Pre-Migration

- [ ] **Backup Current Config** - Save working configuration
- [ ] **Identify Workload Type** - Small/Medium/Large point clouds?
- [ ] **Check GPU Availability** - Do you have GPU? CUDA installed?
- [ ] **Review Current Performance** - Baseline for comparison
- [ ] **Read Migration Guide** - This document

### Migration

- [ ] **Choose Migration Path** - Automatic/Forced/No Migration
- [ ] **Update Configuration** - Modify YAML files
- [ ] **Test on Small Dataset** - Single tile first
- [ ] **Compare Results** - Verify feature similarity
- [ ] **Check Performance** - Compare processing time
- [ ] **Review Logs** - Check mode selection decisions

### Post-Migration

- [ ] **Monitor Production** - Watch for issues
- [ ] **Collect Feedback** - Note any problems
- [ ] **Optimize if Needed** - Adjust `computation_mode` if needed
- [ ] **Update Documentation** - Record decisions
- [ ] **Remove Old Flags** - Clean up config (optional)

---

## Testing Your Migration

### Step 1: Test on Single Tile

```bash
# Process one tile with new config
python -m ign_lidar.cli.process \
    --config examples/config_unified_auto.yaml \
    --tiles 0830_6283 \
    --output ./test_output
```

### Step 2: Compare with Legacy

```bash
# Process same tile with legacy config
python -m ign_lidar.cli.process \
    --config examples/config_legacy_strategy.yaml \
    --tiles 0830_6283 \
    --output ./test_output_legacy
```

### Step 3: Verify Results

```python
import numpy as np
from ign_lidar.io.las_io import read_las

# Load both outputs
unified = read_las("test_output/0830_6283_processed.laz")
legacy = read_las("test_output_legacy/0830_6283_processed.laz")

# Compare normals (should be similar, not exact)
normals_unified = unified['normals']
normals_legacy = legacy['normals']

# Check they're both unit vectors
assert np.allclose(np.linalg.norm(normals_unified, axis=1), 1.0, atol=0.1)
assert np.allclose(np.linalg.norm(normals_legacy, axis=1), 1.0, atol=0.1)

print("✅ Results are numerically valid")
```

### Step 4: Performance Comparison

```python
import time

def benchmark(config_file, num_runs=3):
    times = []
    for _ in range(num_runs):
        start = time.time()
        # Run processing
        elapsed = time.time() - start
        times.append(elapsed)
    return np.mean(times), np.std(times)

# Benchmark both configs
unified_mean, unified_std = benchmark("config_unified_auto.yaml")
legacy_mean, legacy_std = benchmark("config_legacy_strategy.yaml")

print(f"Unified: {unified_mean:.2f}s ± {unified_std:.2f}s")
print(f"Legacy:  {legacy_mean:.2f}s ± {legacy_std:.2f}s")
print(f"Speedup: {legacy_mean/unified_mean:.2f}x")
```

---

## Troubleshooting

### Issue: "UnifiedFeatureComputer not available"

**Error Message**:

```
ERROR: UnifiedFeatureComputer not available: cannot import...
       Falling back to strategy pattern.
```

**Solution**:

```bash
# Reinstall package
pip install -e .

# Verify import works
python -c "from ign_lidar.features.unified_computer import UnifiedFeatureComputer"
```

---

### Issue: Mode Selection Not as Expected

**Symptom**: System selects CPU when you expect GPU

**Check**:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Solution**: Force mode explicitly

```yaml
processor:
  use_unified_computer: true
  computation_mode: "gpu" # Force GPU
```

---

### Issue: Performance Regression

**Symptom**: Unified computer slower than legacy

**Diagnosis**:

1. Check mode selection logs
2. Compare actual modes used
3. Verify workload size estimation

**Solution**: Provide tile size hint

```yaml
processor:
  use_unified_computer: true
  typical_points_per_tile: 2500000 # Your actual size
```

---

### Issue: Numerical Differences

**Symptom**: Results differ slightly from legacy

**Expected**: Minor differences due to implementation details

**Verify Valid**:

```python
# Check normals are unit vectors
norms = np.linalg.norm(features['normals'], axis=1)
assert np.allclose(norms, 1.0, atol=0.1), "Normals should be unit vectors"

# Check curvature is non-negative
assert np.all(features['curvature'] >= 0), "Curvature should be ≥0"
```

**Not a Problem If**:

- Normals are unit vectors
- Curvature values reasonable
- Visual quality unchanged

---

## FAQ

### Q: Do I have to migrate?

**A**: No! Legacy Strategy Pattern continues to work. Migration is opt-in.

### Q: Can I switch back if there are issues?

**A**: Yes! Just set `use_unified_computer: false` or remove the flag.

### Q: Will automatic mode always be optimal?

**A**: Usually yes, but you can override with `computation_mode` if needed.

### Q: What if I have custom strategies?

**A**: Keep using legacy Strategy Pattern until custom strategies are ported.

### Q: Does this affect output format?

**A**: No, output LAZ files are identical in structure.

### Q: Can I use this in production?

**A**: Yes! Tested and backward compatible. Start with non-critical pipelines.

### Q: What about boundary-aware processing?

**A**: Currently only available via legacy Strategy Pattern. UnifiedFeatureComputer support planned for future release.

### Q: How do I debug mode selection?

**A**: Enable logging:

```yaml
logging:
  level: "INFO"
  log_mode_selection: true
  log_recommendations: true
```

---

## Getting Help

### Check Logs

Mode selection decisions are logged:

```
ℹ️  Automatic mode selection: GPU_CHUNKED
    Reason: Large workload (2.5M points), GPU available
    Recommendation: Consider GPU_CHUNKED for optimal performance
```

### Run Tests

```bash
# Test unified computer integration
pytest tests/test_orchestrator_unified_integration.py -v

# Test mode selector
pytest tests/test_mode_selector.py -v

# Test unified computer
pytest tests/test_unified_computer.py -v
```

### Report Issues

If you encounter problems:

1. Include your config file
2. Include error logs
3. Specify workload size (points per tile)
4. Specify hardware (GPU model, CUDA version)

---

## Summary

**Recommended Migration Path**:

1. Start with `use_unified_computer: true` (automatic mode)
2. Test on small dataset
3. Compare performance and results
4. Deploy to production gradually
5. Remove legacy flags after validation

**Key Takeaway**: Migration is **safe**, **optional**, and **reversible**. Start with automatic mode and override only if needed.

---

**Last Updated**: October 18, 2025  
**Version**: Phase 4 Task 1.4  
**Feedback**: Welcome! Report issues or suggestions.
