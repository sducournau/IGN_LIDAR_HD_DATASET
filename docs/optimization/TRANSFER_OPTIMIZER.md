# GPU Transfer Optimizer - Implementation Notes

**Date:** November 23, 2025  
**Version:** v3.3.0  
**Author:** @sducournau

## Overview

The `TransferOptimizer` module provides comprehensive tracking and optimization recommendations for CPU↔GPU data transfers in the IGN LiDAR HD processing pipeline.

## Motivation

GPU acceleration in LiDAR processing involves significant data transfers between CPU and GPU memory. Inefficient transfer patterns can become a major bottleneck, especially:

- Redundant uploads of the same data
- Unnecessary downloads that could stay on GPU
- Poor transfer batching
- Lack of visibility into transfer costs

The audit revealed several areas where transfers could be optimized, such as:

- `gpu_processor.py` mentions "4-6 transfers → 2 transfers per tile" optimization goal
- No systematic way to track transfer patterns across pipeline
- Cache hit rates available but not integrated with transfer analysis

## Solution: TransferOptimizer

### Key Features

1. **Transfer Tracking**

   - Upload events (CPU→GPU)
   - Download events (GPU→CPU)
   - Size, duration, bandwidth metrics
   - Data key identification

2. **Redundancy Detection**

   - Identifies repeated uploads of same data
   - Calculates potential cache savings
   - Tracks redundancy percentage

3. **Bandwidth Analysis**

   - Upload/download bandwidth in GB/s
   - Identifies transfer bottlenecks
   - Compares against hardware limits

4. **Hot Data Identification**

   - Tracks access frequency per data key
   - Recommends caching for frequently accessed data
   - Configurable thresholds for "hot" classification

5. **Actionable Recommendations**
   - Specific caching suggestions
   - Transfer pattern improvements
   - Batching opportunities

## Architecture

```
TransferOptimizer
├── TransferProfile (dataclass)
│   ├── Transfer metrics (counts, sizes, times)
│   ├── Events history
│   └── Redundancy tracking
│
├── TransferEvent (dataclass)
│   ├── Timestamp
│   ├── Size, duration
│   └── Data key
│
└── Methods
    ├── track_upload()
    ├── track_download()
    ├── get_report()
    ├── print_report()
    └── print_recommendations()
```

## Usage Examples

### Basic Tracking

```python
from ign_lidar.optimization import TransferOptimizer

# Initialize
optimizer = TransferOptimizer(enable_profiling=True)

# Track transfers
optimizer.track_upload(size_mb=100.0, duration_ms=50.0, data_key='points')
optimizer.track_download(size_mb=50.0, duration_ms=25.0, data_key='results')

# Get report
report = optimizer.get_report()
print(f"Total transfers: {report['total_transfer_mb']:.1f} MB")
print(f"Redundant uploads: {report['redundant_uploads']} ({report['potential_savings_pct']:.1f}% savings potential)")
```

### Integration with Processing Pipeline

```python
# In feature computation
def compute_features_gpu(points: np.ndarray, optimizer: TransferOptimizer):
    start = time.time()

    # Track upload
    size_mb = points.nbytes / (1024**2)
    points_gpu = cp.asarray(points)
    upload_time = (time.time() - start) * 1000
    optimizer.track_upload(size_mb, upload_time, data_key='points')

    # Compute
    features_gpu = _compute_on_gpu(points_gpu)

    # Track download
    start = time.time()
    features = cp.asnumpy(features_gpu)
    download_time = (time.time() - start) * 1000
    optimizer.track_download(features.nbytes / (1024**2), download_time, data_key='features')

    return features
```

### Getting Recommendations

```python
# After processing
optimizer.print_report()
optimizer.print_recommendations()

# Example output:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   GPU Transfer Optimization Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Uploads:    50 transfers, 5000.0 MB, 2500.0 ms
# Downloads:  30 transfers, 3000.0 MB, 1500.0 ms
# Total:      8000.0 MB in 4000.0 ms
#
# Upload BW:   2.00 GB/s
# Download BW: 2.00 GB/s
#
# Redundancy:  10 redundant uploads (20.0%)
#              1000.0 MB wasted (20.0% potential savings)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   Optimization Recommendations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 🔥 Hot Data (frequently accessed):
#    • 'points' - 15 accesses, 100.0 MB each → CACHE
#    • 'normals' - 8 accesses, 50.0 MB each → CACHE
#
# 💡 Suggestions:
#    • Enable caching for hot data keys
#    • Consider batching small transfers
#    • Monitor bandwidth vs hardware limit
```

## Testing

Comprehensive test suite in `tests/test_transfer_optimizer.py`:

- ✅ Creation and initialization
- ✅ Upload/download tracking
- ✅ Redundancy detection
- ✅ Bandwidth calculations
- ✅ Hot data identification
- ✅ Report generation
- ✅ Reset functionality
- ✅ Disabled profiling mode

All 12 tests passing.

## Integration Points

### 1. With GPUArrayCache

```python
from ign_lidar.optimization import GPUArrayCache, TransferOptimizer

cache = GPUArrayCache()
optimizer = TransferOptimizer()

# Cache miss → upload
if key not in cache:
    optimizer.track_upload(size_mb, duration_ms, data_key=key, cached=False)
    cache.put(key, data_gpu)
else:
    # Cache hit → no upload needed
    optimizer.track_upload(0, 0, data_key=key, cached=True)
```

### 2. With GPUStrategy

```python
class GPUStrategy:
    def __init__(self, transfer_optimizer: Optional[TransferOptimizer] = None):
        self.transfer_optimizer = transfer_optimizer or TransferOptimizer()

    def compute(self, data: np.ndarray):
        # Track transfers automatically
        if self.transfer_optimizer.enabled:
            # ... tracking code ...
        return results
```

### 3. With GroundTruthPerformanceMonitor

```python
class GroundTruthPerformanceMonitor:
    def __init__(self, transfer_optimizer: Optional[TransferOptimizer] = None):
        self.transfer_optimizer = transfer_optimizer

    def get_summary(self):
        summary = {...}
        if self.transfer_optimizer:
            summary['transfer_metrics'] = self.transfer_optimizer.get_report()
        return summary
```

## Performance Impact

- **Overhead:** Minimal (<1% when enabled)
- **Memory:** ~100 bytes per tracked event
- **Works without GPU:** Can track mock transfers for testing/profiling

## Configuration

```yaml
# In config.yaml
optimization:
  transfer_profiling:
    enabled: true
    hot_data_threshold: 3 # Accesses before considered "hot"
    report_interval: 100 # Report every N tiles
```

## Metrics Provided

| Metric                        | Description                           |
| ----------------------------- | ------------------------------------- |
| `total_uploads`               | Number of CPU→GPU uploads             |
| `total_downloads`             | Number of GPU→CPU downloads           |
| `total_upload_mb`             | Total data uploaded (MB)              |
| `total_download_mb`           | Total data downloaded (MB)            |
| `total_transfer_mb`           | Combined transfer size                |
| `avg_upload_bandwidth_gbps`   | Average upload speed (GB/s)           |
| `avg_download_bandwidth_gbps` | Average download speed (GB/s)         |
| `redundant_uploads`           | Count of redundant uploads            |
| `redundant_uploads_mb`        | Size of redundant data                |
| `potential_savings_pct`       | % savings from eliminating redundancy |
| `cache_efficiency`            | % of uploads that should be cached    |
| `hot_data`                    | List of frequently accessed data keys |

## Next Steps

1. **Integration with existing code**

   - Add to `GPUStrategy` and `GPUChunkedStrategy`
   - Hook into `gpu_processor.py`
   - Connect with `GPUArrayCache`

2. **Real-world validation**

   - Run on production datasets
   - Measure actual savings
   - Tune thresholds

3. **Advanced features**

   - Transfer batching detection
   - Pinned memory recommendations
   - CUDA stream optimization hints

4. **Monitoring dashboard**
   - Real-time transfer visualization
   - Historical trends
   - Comparative analysis across runs

## References

- Related to audit finding: "GPU transfer patterns could be optimized"
- Complements: `GPUArrayCache` metrics (cache_hits, cache_misses)
- Uses: `GroundTruthPerformanceMonitor` for benchmarking framework
- Target: Reduce 4-6 transfers → 2 transfers per tile (gpu_processor.py goal)

## Version History

- **v1.0** (2025-11-23): Initial implementation
  - Basic transfer tracking
  - Redundancy detection
  - Hot data identification
  - Bandwidth analysis
  - 12/12 tests passing
