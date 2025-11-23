# Verbose Mode and GPU Profiling Guide

**Version:** 3.8.0+  
**Date:** November 23, 2025  
**Purpose:** Complete guide to using verbose mode and GPU profiling for debugging and optimization

---

## Overview

The IGN LiDAR HD library provides comprehensive profiling and debugging capabilities through **verbose mode** and the **GPU profiler**. These tools help you:

- 🐛 Debug processing issues
- ⚡ Identify performance bottlenecks
- 💾 Monitor memory usage (CPU and GPU)
- 📊 Analyze GPU utilization
- 🔍 Track data transfer overhead
- 🎯 Optimize processing pipelines

---

## Quick Start

### Enable Verbose Mode

**Command Line:**

```bash
# Enable verbose logging
ign-lidar-process --verbose --use-gpu tiles/ output/

# Short form
ign-lidar-process -v --use-gpu tiles/ output/
```

**Python API:**

```python
import logging
from ign_lidar import LiDARProcessor

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Process with detailed output
processor = LiDARProcessor(config_path="config.yaml")
processor.process_directory("tiles/", "output/")
```

**Configuration File:**

```yaml
# config.yaml
processor:
  verbose: true
  use_gpu: true

logging:
  level: DEBUG
  show_timings: true
  show_memory: true
```

---

## What Verbose Mode Shows

### 1. Processing Progress

**Without verbose:**

```
Processing tiles: 100%|██████████| 10/10 [02:15<00:00, 13.5s/tile]
```

**With verbose:**

```
[DEBUG] Starting tile: LHD_FXX_0123_4567.laz
[DEBUG]   - Points: 2,345,678
[DEBUG]   - Size: 45.6 MB
[DEBUG]   - Chunks: 3 (800K points each)
[DEBUG]
[DEBUG] Phase 1: Loading and preprocessing
[DEBUG]   ✓ Load LAZ file: 1.23s
[DEBUG]   ✓ Outlier removal: 0.45s (removed 234 points, 0.01%)
[DEBUG]   ✓ RGB augmentation: 2.11s
[DEBUG]
[DEBUG] Phase 2: Feature computation
[DEBUG]   ✓ Geometric features (GPU): 3.45s
[DEBUG]     - Transfer to GPU: 0.23s
[DEBUG]     - Compute: 2.98s
[DEBUG]     - Transfer from GPU: 0.24s
[DEBUG]     - GPU utilization: 87%
[DEBUG]   ✓ Statistical features: 1.23s
[DEBUG]
[DEBUG] Phase 3: Classification
[DEBUG]   ✓ Ground truth fetch: 0.87s
[DEBUG]   ✓ Classification mapping: 0.12s
[DEBUG]
[DEBUG] Phase 4: Output
[DEBUG]   ✓ Tile stitching: 0.45s
[DEBUG]   ✓ Write enriched LAZ: 1.89s
[DEBUG]   ✓ Write patches: 2.34s
[DEBUG]
[DEBUG] Completed tile in 13.5s (173K points/sec)
[DEBUG] GPU memory: 2.3GB used / 8.0GB total (29%)
```

### 2. Memory Monitoring

**CPU Memory:**

```
[DEBUG] Memory status:
[DEBUG]   - Process RSS: 2.4GB
[DEBUG]   - Available: 12.3GB
[DEBUG]   - Swap used: 0%
[DEBUG]   - Memory pressure: 🟢 LOW
```

**GPU Memory:**

```
[DEBUG] GPU memory status:
[DEBUG]   - VRAM used: 3.2GB / 8.0GB (40%)
[DEBUG]   - VRAM free: 4.8GB
[DEBUG]   - Memory pools: 3.1GB allocated
[DEBUG]   - Fragmentation: 5% (good)
[DEBUG]   - Peak usage: 3.8GB
```

### 3. Performance Metrics

**Per-Operation Timings:**

```
[DEBUG] Performance breakdown:
[DEBUG]   Operation              Time     %Total  Points/sec
[DEBUG]   ───────────────────────────────────────────────────
[DEBUG]   Load LAZ              1.23s      9%     1.9M
[DEBUG]   Outlier removal       0.45s      3%     5.2M
[DEBUG]   RGB augmentation      2.11s     16%     1.1M
[DEBUG]   Feature computation   3.45s     26%     680K
[DEBUG]   Ground truth          0.87s      6%     2.7M
[DEBUG]   Classification        0.12s      1%     19.5M
[DEBUG]   Tile stitching        0.45s      3%     5.2M
[DEBUG]   Write outputs         4.23s     31%     554K
[DEBUG]   ───────────────────────────────────────────────────
[DEBUG]   TOTAL                13.91s    100%     168K
```

**GPU Bottleneck Analysis:**

```
[DEBUG] GPU performance analysis:
[DEBUG]   - Total GPU time: 3.45s
[DEBUG]   - Transfer time: 0.47s (14% overhead)
[DEBUG]   - Compute time: 2.98s (86% efficiency)
[DEBUG]   - GPU utilization: 87% (excellent)
[DEBUG]
[DEBUG]   Bottleneck: BALANCED ✅
[DEBUG]
[DEBUG]   Recommendations:
[DEBUG]     ✓ GPU utilization is high
[DEBUG]     ✓ Transfer overhead is low
[DEBUG]     ✓ No optimization needed
```

### 4. Error Diagnostics

**Enhanced Error Messages (verbose mode):**

```
[ERROR] GPU out of memory
[DEBUG]
[DEBUG] Diagnostic Information:
[DEBUG]   - VRAM usage: 7.8GB / 8.0GB (98% full)
[DEBUG]   - Peak memory: 7.9GB
[DEBUG]   - Chunk size: 1,000,000 points
[DEBUG]   - Total points: 5,234,567
[DEBUG]   - Number of chunks: 6
[DEBUG]
[DEBUG] Memory Timeline:
[DEBUG]   Chunk 1: 2.3GB ✓
[DEBUG]   Chunk 2: 4.1GB ✓
[DEBUG]   Chunk 3: 5.9GB ✓
[DEBUG]   Chunk 4: 7.8GB ✓
[DEBUG]   Chunk 5: FAILED (OOM)
[DEBUG]
[DEBUG] Suggested Actions:
[DEBUG]   1. Reduce chunk size to 500,000 points
[DEBUG]   2. Enable GPU chunked mode
[DEBUG]   3. Or switch to CPU mode
```

---

## GPU Profiler API

### Basic Usage

```python
from ign_lidar.optimization import GPUProfiler

# Initialize profiler
profiler = GPUProfiler(enable=True, session_name="feature_computation")

# Profile individual operations
with profiler.profile_operation('normals', data_size_mb=100):
    normals = compute_normals_gpu(points_gpu)

# Get summary
profiler.print_summary()
```

**Output:**

```
GPU Profiling Session: feature_computation
Duration: 5.23s

Operations:
  Operation        Count   Total Time   Avg Time   Data Size   Success
  ─────────────────────────────────────────────────────────────────────
  normals              3      2.45s     0.82s      300.0 MB      100%
  curvature            3      1.12s     0.37s      300.0 MB      100%
  eigenvalues          3      1.66s     0.55s      600.0 MB      100%
  ─────────────────────────────────────────────────────────────────────
  TOTAL               9      5.23s     0.58s      1200.0 MB     100%

Performance Metrics:
  - Total transfer time: 0.89s (17% overhead)
  - Total compute time: 4.34s (83% efficiency)
  - Average VRAM usage: 3.2GB
  - Peak VRAM usage: 4.1GB

Bottleneck Analysis:
  🎯 Compute is the bottleneck (83% of time)

  Recommendations:
    • Computation is dominating - this is expected
    • Transfer overhead is acceptable (<20%)
    • Consider larger batch sizes to amortize transfers
    • GPU is being used efficiently
```

### Advanced Profiling

```python
from ign_lidar.optimization import GPUProfiler

profiler = GPUProfiler(enable=True)

# Start profiling session
profiler.start_session("full_pipeline")

# Profile multiple stages
with profiler.profile_operation('load_data', data_size_mb=50):
    points = load_points(file_path)

with profiler.profile_operation('transfer_to_gpu', data_size_mb=50):
    points_gpu = cp.asarray(points)

with profiler.profile_operation('compute_features', data_size_mb=200):
    features_gpu = compute_all_features_gpu(points_gpu)

with profiler.profile_operation('transfer_from_gpu', data_size_mb=200):
    features = cp.asnumpy(features_gpu)

# End session and analyze
profiler.end_session()
profiler.print_summary()
profiler.save_report('profile_report.json')

# Get bottleneck analysis
analysis = profiler.get_bottleneck_analysis()
print(f"Bottleneck: {analysis['bottleneck']}")
print(f"Transfer overhead: {analysis['transfer_pct']:.1f}%")
print(f"Recommendations: {analysis['recommendation']}")
```

### Integration with FeatureOrchestrator

```python
from ign_lidar.features import FeatureOrchestrator
from ign_lidar.optimization import GPUProfiler

# Enable profiling
profiler = GPUProfiler(enable=True)

# Configure orchestrator with profiling
orchestrator = FeatureOrchestrator(
    mode='lod2',
    use_gpu=True,
    verbose=True,  # Enables automatic profiling output
    profiler=profiler  # Attach profiler
)

# Process with profiling
features = orchestrator.compute_features(points)

# View results
profiler.print_summary()
```

---

## Profiling Scenarios

### Scenario 1: Debugging Slow Performance

**Problem:** Processing is slower than expected

**Solution:**

```bash
# Run with verbose profiling
ign-lidar-process -v --use-gpu tiles/ output/ 2>&1 | tee profile.log

# Analyze output
grep "Performance breakdown" profile.log
grep "Bottleneck" profile.log
grep "points/sec" profile.log
```

**Look for:**

- Operations taking disproportionate time
- Low GPU utilization (<50%)
- High transfer overhead (>30%)
- Low points/sec throughput

### Scenario 2: GPU Memory Issues

**Problem:** Frequent GPU OOM errors

**Solution:**

```python
from ign_lidar.optimization import GPUProfiler
import logging

logging.basicConfig(level=logging.DEBUG)

profiler = GPUProfiler(enable=True)

# Monitor memory throughout processing
for chunk in chunks:
    with profiler.profile_operation(f'chunk_{i}', data_size_mb=chunk_size_mb):
        # Check memory before processing
        mem_info = profiler.get_memory_info()
        logger.debug(f"Pre-processing VRAM: {mem_info['used_gb']:.1f}GB")

        # Process
        features = compute_features_gpu(chunk)

        # Check memory after
        mem_info = profiler.get_memory_info()
        logger.debug(f"Post-processing VRAM: {mem_info['used_gb']:.1f}GB")

# Analyze memory usage patterns
profiler.print_summary()
```

**Look for:**

- Memory growth over chunks (indicates leak)
- Peak memory vs average (indicates spikes)
- Memory not freed after operations

### Scenario 3: Transfer vs Compute Optimization

**Problem:** Want to optimize GPU pipeline

**Solution:**

```python
profiler = GPUProfiler(enable=True)

# Profile full pipeline
with profiler.profile_operation('full_pipeline'):
    # CPU → GPU transfer
    with profiler.profile_operation('h2d_transfer', data_size_mb=100):
        points_gpu = cp.asarray(points)

    # GPU compute
    with profiler.profile_operation('gpu_compute', data_size_mb=100):
        features_gpu = compute_features_gpu(points_gpu)

    # GPU → CPU transfer
    with profiler.profile_operation('d2h_transfer', data_size_mb=200):
        features = cp.asnumpy(features_gpu)

# Analyze bottleneck
analysis = profiler.get_bottleneck_analysis()

if analysis['bottleneck'] == 'memory_transfer':
    print("🔴 Memory transfer is the bottleneck")
    print("   → Use larger chunk sizes")
    print("   → Enable CUDA streams")
    print("   → Keep data on GPU longer")
elif analysis['bottleneck'] == 'compute':
    print("🟢 Compute is the bottleneck (expected)")
    print("   → GPU is being used efficiently")
```

---

## Configuration Options

### Logging Configuration

```yaml
# config.yaml
logging:
  # Logging level
  level: DEBUG # DEBUG, INFO, WARNING, ERROR, CRITICAL

  # Show timing information
  show_timings: true

  # Show memory usage
  show_memory: true

  # Show GPU metrics
  show_gpu_metrics: true

  # Log file (optional)
  file: "processing.log"

  # Format
  format: "%(asctime)s [%(levelname)s] %(message)s"

  # Date format
  datefmt: "%Y-%m-%d %H:%M:%S"
```

### Profiler Configuration

```python
from ign_lidar.optimization import GPUProfiler

profiler = GPUProfiler(
    enable=True,                    # Enable profiling
    session_name="my_session",      # Session identifier
    track_memory=True,               # Track VRAM usage
    track_transfers=True,            # Track data transfers
    track_utilization=True,          # Track GPU utilization
    auto_print_summary=True,         # Print summary on session end
    save_report=True,                # Save JSON report
    report_path="profile_report.json"
)
```

---

## Performance Benchmarking

### Automated Benchmarking

```bash
# Run comprehensive benchmark with profiling
python scripts/benchmark_phase3.py \
    --quick \
    --verbose \
    --save-results benchmark_results.json

# View results
cat benchmark_results.json | jq '.performance_metrics'
```

### Custom Benchmarking

```python
from ign_lidar.optimization import GPUProfiler
import time

profiler = GPUProfiler(enable=True)

# Benchmark configuration
configs = [
    {'chunk_size': 500_000, 'use_gpu': True},
    {'chunk_size': 1_000_000, 'use_gpu': True},
    {'chunk_size': 2_000_000, 'use_gpu': True},
]

results = []

for config in configs:
    profiler.start_session(f"config_{config['chunk_size']}")

    # Run processing
    start = time.time()
    process_with_config(config)
    duration = time.time() - start

    profiler.end_session()

    # Collect metrics
    metrics = profiler.get_summary()
    results.append({
        'config': config,
        'duration': duration,
        'gpu_time': metrics['total_compute_time'],
        'transfer_time': metrics['total_transfer_time'],
        'throughput': num_points / duration
    })

# Find optimal configuration
best = max(results, key=lambda x: x['throughput'])
print(f"Optimal config: {best['config']}")
print(f"Throughput: {best['throughput']:.0f} points/sec")
```

---

## Troubleshooting

### No Profiling Output

**Problem:** Verbose mode enabled but no detailed output

**Possible causes:**

1. Logging level not set to DEBUG
2. Profiler not enabled
3. Output being redirected

**Solutions:**

```python
import logging

# Ensure DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Explicitly enable profiler
from ign_lidar.optimization import GPUProfiler
profiler = GPUProfiler(enable=True)
```

### Profiler Overhead

**Problem:** Profiling slows down processing

**Cause:** Profiler adds ~1-5% overhead

**Solutions:**

```python
# Disable profiling for production
profiler = GPUProfiler(enable=False)

# Or use selective profiling
profiler = GPUProfiler(
    enable=True,
    track_memory=False,         # Disable expensive memory tracking
    track_utilization=False,    # Disable utilization monitoring
)
```

### Memory Tracking Inaccurate

**Problem:** Memory numbers don't match `nvidia-smi`

**Cause:** CuPy memory pool caching

**Explanation:**

```python
# CuPy memory pool holds freed memory for reuse
# This is normal and improves performance

# To see actual allocated memory:
import cupy as cp
mempool = cp.get_default_memory_pool()

print(f"Used by arrays: {mempool.used_bytes() / 1e9:.1f}GB")
print(f"Held in pool: {mempool.total_bytes() / 1e9:.1f}GB")

# To free memory back to system:
mempool.free_all_blocks()
```

---

## Best Practices

### ✅ DO

1. **Use verbose mode for development and debugging**

   ```bash
   ign-lidar-process -v --use-gpu tiles/ output/
   ```

2. **Profile new features during development**

   ```python
   with profiler.profile_operation('my_new_feature'):
       result = my_new_feature(data)
   ```

3. **Monitor memory in production for large datasets**

   ```python
   if verbose:
       profiler = GPUProfiler(enable=True, track_memory=True)
   ```

4. **Save profiling reports for analysis**

   ```python
   profiler.save_report('profile.json')
   ```

5. **Use bottleneck analysis to guide optimization**
   ```python
   analysis = profiler.get_bottleneck_analysis()
   print(analysis['recommendation'])
   ```

### ❌ DON'T

1. **Don't leave verbose mode on in production scripts**
   - Adds overhead
   - Produces excessive logs
2. **Don't ignore profiler warnings**
   - High transfer overhead → optimize data movement
   - Low GPU utilization → increase batch sizes
3. **Don't profile in tight loops**

   ```python
   # BAD: Profiling every point
   for point in points:
       with profiler.profile_operation('process_point'):
           process_point(point)

   # GOOD: Profile batch
   with profiler.profile_operation('process_points'):
       for point in points:
           process_point(point)
   ```

4. **Don't use DEBUG logging for large production runs**
   - Use INFO or WARNING level
   - Selectively enable DEBUG for specific modules

---

## Related Documentation

- **GPU Acceleration Guide:** `docs/guides/gpu-acceleration.md`
- **Performance Optimization:** `docs/guides/performance.md`
- **Error Handling:** `docs/guides/error-handling.md`
- **Phase 3 Features:** `PHASE3_COMPLETE_SUMMARY.md`

---

## Example: Complete Profiling Workflow

```python
import logging
from ign_lidar import LiDARProcessor
from ign_lidar.optimization import GPUProfiler

# 1. Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

# 2. Initialize profiler
profiler = GPUProfiler(
    enable=True,
    session_name="production_run",
    track_memory=True,
    save_report=True,
    report_path='profile_report.json'
)

# 3. Configure processor with profiling
processor = LiDARProcessor(
    config_path='config.yaml',
    profiler=profiler,
    verbose=True
)

# 4. Process with profiling
try:
    processor.process_directory('tiles/', 'output/')
except Exception as e:
    logging.error(f"Processing failed: {e}")
    profiler.print_summary()  # Print profile even on failure
    raise

# 5. Analyze results
profiler.print_summary()
analysis = profiler.get_bottleneck_analysis()

logging.info("=" * 70)
logging.info("PROFILING SUMMARY")
logging.info("=" * 70)
logging.info(f"Bottleneck: {analysis['bottleneck']}")
logging.info(f"Transfer overhead: {analysis['transfer_pct']:.1f}%")
logging.info(f"Compute efficiency: {analysis['compute_pct']:.1f}%")
logging.info("\nRecommendations:")
logging.info(analysis['recommendation'])
logging.info("=" * 70)

# 6. Save detailed report
profiler.save_report('final_profile.json')
logging.info("Detailed profile saved to: final_profile.json")
```

---

**Version History:**

| Version | Date       | Changes                                  |
| ------- | ---------- | ---------------------------------------- |
| 1.0     | 2025-11-23 | Initial documentation                    |
| 1.1     | 2025-11-23 | Added GPU profiler integration           |
| 1.2     | 2025-11-23 | Added troubleshooting and best practices |

**Author:** IGN LiDAR HD Development Team  
**Last Updated:** November 23, 2025
