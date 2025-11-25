# Phase 5 Advanced Use Cases & Patterns

**Version:** 3.6.0+  
**Date:** November 2025  
**Status:** Production Documentation

---

## Table of Contents

1. [Advanced GPU Stream Management](#advanced-gpu-stream-management)
2. [Performance Monitoring at Scale](#performance-monitoring-at-scale)
3. [Configuration Validation Patterns](#configuration-validation-patterns)
4. [Real-World Integration Scenarios](#real-world-integration-scenarios)
5. [Troubleshooting & Optimization](#troubleshooting--optimization)
6. [Migration Strategies](#migration-strategies)

---

## Advanced GPU Stream Management

### Pattern 1: Custom Stream Pools

```python
from ign_lidar.core import get_stream_manager

# Create custom-sized pool
manager = get_stream_manager(pool_size=8)

# Configure stream behavior
manager.configure(
    pool_size=8,
    default_priority=0,
    enable_profiling=True,
    auto_sync=True
)

# Monitor stream utilization
stats = manager.get_performance_stats()
print(f"Streams in use: {stats['streams_in_use']}")
print(f"Total transferred: {stats['bytes_transferred']} bytes")
```

### Pattern 2: Direct Stream Control

```python
from ign_lidar.core import get_stream_manager
import numpy as np

manager = get_stream_manager()

# Get specific stream for explicit control
stream0 = manager.get_stream(0)
stream1 = manager.get_stream(1)

# Transfer on different streams concurrently
src1, dst1 = np.random.rand(1000, 3), np.zeros((1000, 3))
src2, dst2 = np.random.rand(1000, 3), np.zeros((1000, 3))

stream0.transfer_async(src1, dst1)
stream1.transfer_async(src2, dst2)

# Sync each stream
stream0.synchronize()
stream1.synchronize()

# Get individual statistics
stats0 = stream0.get_stats()
stats1 = stream1.get_stats()

print(f"Stream 0: {stats0['transfer_count']} transfers")
print(f"Stream 1: {stats1['transfer_count']} transfers")
```

### Pattern 3: Batching with Memory Awareness

```python
from ign_lidar.core import get_stream_manager
import numpy as np

manager = get_stream_manager()

# Large number of transfers
large_batch = [
    (np.random.rand(500, 3), np.zeros((500, 3)))
    for _ in range(1000)
]

# Batch with automatic memory management
manager.batch_transfers(large_batch)

# This automatically:
# - Groups transfers by memory size
# - Distributes across stream pool
# - Handles GPU memory constraints
# - Retries on failure

manager.wait_all()
```

### Pattern 4: Stream Profiling

```python
from ign_lidar.core import get_stream_manager
import numpy as np

manager = get_stream_manager()

# Perform operations
for i in range(100):
    src = np.random.rand(500, 3)
    dst = np.zeros((500, 3))
    manager.async_transfer(src, dst)

manager.wait_all()

# Get comprehensive stats
stats = manager.get_performance_stats()

print("Performance Analysis:")
print(f"  Total transfers: {stats['transfer_stats']['count']}")
print(f"  Avg latency: {stats['transfer_stats']['avg_latency_us']:.2f} µs")
print(f"  Max latency: {stats['transfer_stats']['max_latency_us']:.2f} µs")
print(f"  Throughput: {stats['transfer_stats']['throughput_gb_s']:.2f} GB/s")
```

---

## Performance Monitoring at Scale

### Pattern 1: Hierarchical Phase Tracking

```python
from ign_lidar.core import get_performance_manager

manager = get_performance_manager()
manager.reset()

# Top-level phase
manager.start_phase("pipeline")

# Nested phases
manager.start_phase("loading")
# ... load data ...
manager.end_phase()

manager.start_phase("processing")
for batch_id in range(10):
    manager.start_phase(f"batch_{batch_id}")
    # ... process batch ...
    manager.record_metric(f"batch_{batch_id}_accuracy", 0.95)
    manager.end_phase()
manager.end_phase()

manager.end_phase()  # End pipeline

# Get full hierarchy
summary = manager.get_summary()
for phase_name, metrics in summary["phases"].items():
    print(f"{phase_name}: {metrics['duration']:.2f}s")
```

### Pattern 2: Custom Metric Collection

```python
from ign_lidar.core import get_performance_manager

manager = get_performance_manager()
manager.reset()

manager.start_phase("model_training")

# Collect metrics throughout phase
for epoch in range(100):
    manager.record_metric("epoch", epoch)
    manager.record_metric("loss", compute_loss())
    manager.record_metric("accuracy", compute_accuracy())
    manager.record_metric("learning_rate", lr_scheduler.get_lr())

manager.end_phase()

# Analyze metrics
stats = manager.get_phase_stats("model_training")
custom_metrics = stats["custom_metrics"]

print(f"Final loss: {custom_metrics['loss']}")
print(f"Final accuracy: {custom_metrics['accuracy']}")
```

### Pattern 3: Memory Profiling

```python
from ign_lidar.core import get_performance_manager

manager = get_performance_manager()
manager.configure(track_memory=True, track_gpu=True)

manager.reset()

manager.start_phase("memory_intensive_operation")

# Perform operations - memory is automatically tracked
# ...

manager.end_phase()

# Analyze memory usage
stats = manager.get_phase_stats("memory_intensive_operation")

print(f"Peak CPU memory: {stats['memory_mb']:.2f} MB")
if 'gpu_memory_mb' in stats:
    print(f"Peak GPU memory: {stats['gpu_memory_mb']:.2f} MB")
```

### Pattern 4: Multi-Phase Comparison

```python
from ign_lidar.core import get_performance_manager

manager = get_performance_manager()

# Run different implementations and compare
implementations = ["cpu", "gpu_single", "gpu_multi"]

results = {}

for impl in implementations:
    manager.reset()
    manager.start_phase(impl)
    
    # Run implementation
    run_implementation(impl)
    
    manager.end_phase()
    
    stats = manager.get_phase_stats(impl)
    results[impl] = stats["duration"]

# Compare results
print("Performance Comparison:")
for impl, duration in results.items():
    speedup = results["cpu"] / duration
    print(f"  {impl}: {duration:.3f}s ({speedup:.1f}x speedup)")
```

---

## Configuration Validation Patterns

### Pattern 1: Extensible Validation Rules

```python
from ign_lidar.core import get_config_validator

validator = get_config_validator()
validator.clear_rules()

# Add predefined validators
validator.add_lod_validator()
validator.add_gpu_validator()

# Add custom validators
def validate_model_path(config):
    """Custom rule: model path must exist."""
    from pathlib import Path
    if "model_path" in config:
        if not Path(config["model_path"]).exists():
            return False, f"Model not found: {config['model_path']}"
    return True, None

def validate_batch_divisible(config):
    """Custom rule: batch size divisible by 32."""
    if "batch_size" in config:
        if config["batch_size"] % 32 != 0:
            return False, "Batch size must be divisible by 32"
    return True, None

validator.add_rule("model_path_exists", validate_model_path)
validator.add_rule("batch_divisible", validate_batch_divisible)

# Validate with all rules
config = {
    "lod_level": "LOD2",
    "model_path": "/path/to/model.pth",
    "batch_size": 256,
}

is_valid, errors = validator.validate(config)
print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
if errors:
    for error in errors:
        print(f"  Error: {error}")
```

### Pattern 2: Conditional Validation

```python
from ign_lidar.core import get_config_validator

def build_validator_for_mode(mode):
    """Build validator based on processing mode."""
    validator = get_config_validator()
    validator.clear_rules()
    
    if mode == "training":
        validator.add_lod_validator()
        validator.add_numeric_range_validator("learning_rate", 0.0001, 0.1)
        validator.add_numeric_range_validator("batch_size", 16, 512)
    
    elif mode == "inference":
        validator.add_lod_validator()
        validator.add_gpu_validator()
        validator.add_numeric_range_validator("batch_size", 1, 10000)
    
    elif mode == "evaluation":
        validator.add_lod_validator()
        validator.add_required_field("test_dataset")
    
    return validator

# Use conditional validator
train_validator = build_validator_for_mode("training")
infer_validator = build_validator_for_mode("inference")

train_config = {"lod_level": "LOD2", "learning_rate": 0.001, "batch_size": 128}
infer_config = {"lod_level": "LOD3", "batch_size": 1024}

train_valid, _ = train_validator.validate(train_config)
infer_valid, _ = infer_validator.validate(infer_config)

print(f"Training config: {'VALID' if train_valid else 'INVALID'}")
print(f"Inference config: {'VALID' if infer_valid else 'INVALID'}")
```

### Pattern 3: Detailed Validation Reports

```python
from ign_lidar.core import get_config_validator

validator = get_config_validator()
validator.clear_rules()

validator.add_lod_validator()
validator.add_gpu_validator()
validator.add_numeric_range_validator("batch_size", 1, 10000)

config = {
    "lod_level": "LOD2",
    "gpu_memory_fraction": 0.8,
    "batch_size": 256,
    "extra_field": "ignored",  # Not validated
}

# Get detailed report
is_valid, report = validator.validate_detailed(config)

print("Validation Report:")
print(f"  Status: {'PASS' if is_valid else 'FAIL'}")
print(f"  Errors: {len(report.errors)}")

if report.errors:
    print("\n  Error Details:")
    for error in report.errors:
        print(f"    [{error.level}] {error.message}")

print(f"\n  Summary: {report.summary()}")
```

---

## Real-World Integration Scenarios

### Scenario 1: LiDAR Processing Pipeline

```python
from ign_lidar.core import (
    get_config_validator,
    get_performance_manager,
    get_stream_manager
)
import numpy as np

def process_lidar_pipeline(tile_path, config):
    """
    Complete LiDAR processing pipeline using Phase 5 managers.
    """
    
    # 1. VALIDATE CONFIGURATION
    validator = get_config_validator()
    validator.clear_rules()
    validator.add_lod_validator()
    validator.add_gpu_validator()
    validator.add_numeric_range_validator("num_workers", 1, 16)
    
    perf = get_performance_manager()
    perf.reset()
    
    perf.start_phase("validation")
    is_valid, errors = validator.validate(config)
    perf.end_phase()
    
    if not is_valid:
        raise ValueError(f"Invalid config: {errors}")
    
    # 2. LOAD DATA
    perf.start_phase("data_loading")
    point_cloud = load_laz_file(tile_path)
    metadata = extract_metadata(point_cloud)
    perf.record_metric("point_count", len(point_cloud))
    perf.end_phase()
    
    # 3. GPU TRANSFERS
    streams = get_stream_manager()
    
    perf.start_phase("gpu_transfer")
    
    # Split into batches
    batch_size = config.get("batch_size", 10000)
    batches = [
        (point_cloud[i:i+batch_size], np.zeros_like(point_cloud[i:i+batch_size]))
        for i in range(0, len(point_cloud), batch_size)
    ]
    
    # Transfer all batches
    streams.batch_transfers(batches)
    streams.wait_all()
    
    perf.record_metric("batch_count", len(batches))
    perf.end_phase()
    
    # 4. FEATURE COMPUTATION
    perf.start_phase("feature_computation")
    
    lod_level = config.get("lod_level", "LOD2")
    
    for i, batch in enumerate(batches):
        perf.start_phase(f"batch_{i}")
        features = compute_features(batch, lod_level)
        perf.record_metric(f"batch_{i}_features", len(features))
        perf.end_phase()
    
    perf.end_phase()
    
    # 5. CLASSIFICATION
    perf.start_phase("classification")
    
    classifications = classify_features(features, lod_level)
    perf.record_metric("classified_count", len(classifications))
    
    perf.end_phase()
    
    # 6. RESULTS
    perf.start_phase("output")
    
    save_results(classifications, f"{tile_path}.results")
    perf.record_metric("saved_classifications", len(classifications))
    
    perf.end_phase()
    
    # FINAL REPORT
    summary = perf.get_summary()
    print("\nProcessing Summary:")
    print(f"  Total time: {summary['total_time']:.2f}s")
    print(f"  Phases: {summary['num_phases']}")
    
    for phase_name, metrics in summary["phases"].items():
        print(f"  - {phase_name}: {metrics['duration']:.2f}s")
    
    return classifications
```

### Scenario 2: Batch Processing with Error Recovery

```python
from ign_lidar.core import (
    get_config_validator,
    get_performance_manager,
    get_stream_manager
)

def batch_process_tiles(tile_paths, config):
    """
    Process multiple tiles with error recovery and monitoring.
    """
    
    validator = get_config_validator()
    validator.clear_rules()
    validator.add_lod_validator()
    
    perf = get_performance_manager()
    perf.reset()
    
    streams = get_stream_manager()
    
    results = {}
    errors = {}
    
    perf.start_phase("batch_processing")
    
    for tile_path in tile_paths:
        perf.start_phase(f"tile_{os.path.basename(tile_path)}")
        
        try:
            # Validate config
            perf.start_phase("validate")
            is_valid, _ = validator.validate(config)
            perf.end_phase()
            
            if not is_valid:
                raise ValueError("Invalid configuration")
            
            # Process tile
            perf.start_phase("process")
            result = process_single_tile(tile_path, config, streams)
            perf.record_metric("success", 1)
            perf.end_phase()
            
            results[tile_path] = result
            
        except Exception as e:
            # Track error
            perf.record_metric("error", 1)
            perf.end_phase()  # End phase on error
            errors[tile_path] = str(e)
            print(f"Error processing {tile_path}: {e}")
    
    perf.end_phase()  # End batch processing
    
    # REPORT
    summary = perf.get_summary()
    
    print("\nBatch Processing Complete:")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(errors)}")
    print(f"  Total time: {summary['total_time']:.2f}s")
    
    return results, errors
```

---

## Troubleshooting & Optimization

### Issue 1: GPU Memory Exhaustion

**Problem:** GPU runs out of memory during transfers

**Solution:**

```python
from ign_lidar.core import get_stream_manager

manager = get_stream_manager()

# Reduce batch size
manager.batch_transfers(smaller_batch)  # Process fewer transfers

# Or use lower-level API for more control
stream = manager.get_stream()
for src, dst in data_in_chunks:
    stream.transfer_async(src, dst)
    stream.synchronize()  # Sync immediately to free memory
```

### Issue 2: Performance Overhead

**Problem:** Phase tracking causes noticeable slowdown

**Solution:**

```python
from ign_lidar.core import get_performance_manager

manager = get_performance_manager()

# Disable optional tracking if not needed
manager.configure(track_memory=False, track_gpu=False)

# Use batch metrics instead of per-item
for item in large_list:
    # DON'T DO THIS:
    # manager.record_metric(f"item_{i}", value)  # Creates many metrics
    
    # DO THIS INSTEAD:
    perf.record_metric("batch_values", value)  # Single metric
```

### Issue 3: Validation Bottleneck

**Problem:** Config validation is too strict or slow

**Solution:**

```python
from ign_lidar.core import get_config_validator

validator = get_config_validator()

# Create minimal validator for hot path
validator.clear_rules()
validator.add_lod_validator()  # Only essential rules

# Pre-validate config once, cache result
is_valid, _ = validator.validate(config)
if not is_valid:
    raise ValueError("Invalid config")

# Reuse config without re-validating
for i in range(1000):
    process_with_config(config)  # No validation needed
```

---

## Migration Strategies

### Strategy 1: Gradual Import Migration

```python
# OLD CODE
from ign_lidar.optimization.cuda_streams import get_stream
from ign_lidar.core.performance import ProcessorPerformanceMonitor

# NEW CODE - Phase 1: Import both
from ign_lidar.optimization.cuda_streams import get_stream  # Old
from ign_lidar.core import get_stream_manager  # New
from ign_lidar.core.performance import ProcessorPerformanceMonitor  # Old
from ign_lidar.core import get_performance_manager  # New

# Phase 2: Start using new imports
# stream = get_stream()
stream = get_stream_manager().get_stream()

# Phase 3: Remove old imports
# from ign_lidar.optimization.cuda_streams import get_stream
from ign_lidar.core import get_stream_manager
```

### Strategy 2: Wrapper Functions

```python
# Create compatibility wrapper
def legacy_get_stream(*args, **kwargs):
    """Deprecated: use get_stream_manager().get_stream() instead."""
    import warnings
    warnings.warn(
        "legacy_get_stream is deprecated, use get_stream_manager().get_stream()",
        DeprecationWarning,
        stacklevel=2
    )
    from ign_lidar.core import get_stream_manager
    return get_stream_manager().get_stream(*args, **kwargs)

# Can be imported by old code
# from ign_lidar.optimization.cuda_streams import get_stream as legacy_get_stream
```

### Strategy 3: Mixed Environment

```python
# Can have both old and new code running together
from ign_lidar.core import get_stream_manager, get_performance_manager

manager = get_stream_manager()
perf = get_performance_manager()

# New code
perf.start_phase("new_processing")
manager.async_transfer(src, dst)
perf.end_phase()

# Old code can coexist
from ign_lidar.optimization import cuda_streams  # Old code still works
stream = cuda_streams.get_stream()

# Both track to same singleton
summary = perf.get_summary()
```

---

## Best Practices

### ✅ DO:

1. Use HIGH-LEVEL API for 80% of code
2. Reset managers at start of workflow
3. Track phases for complete pipelines
4. Record meaningful metrics
5. Validate config early
6. Handle errors gracefully
7. Monitor performance regularly

### ❌ DON'T:

1. Create multiple manager instances (they're singletons)
2. Forget to call `wait_all()` for GPU transfers
3. Create thousands of unique metrics (group them)
4. Skip validation in production
5. Ignore deprecation warnings
6. Mix old and new APIs without planning
7. Assume validation rules never fail

---

## Summary

Phase 5 managers provide powerful, flexible APIs for GPU, performance, and configuration management. By following these patterns and best practices, you can build efficient, maintainable systems that scale from development to production.

For questions or issues, refer to:
- `REFACTORING_PHASE_5_SUMMARY.md` - Architecture overview
- `examples/phase5_managers_example.py` - Working examples
- Manager docstrings - Complete API reference
