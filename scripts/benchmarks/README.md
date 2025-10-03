# Benchmark Scripts

Performance benchmarking and optimization analysis.

## Scripts

### benchmark_gpu.py

Comprehensive CPU vs GPU performance comparison for feature computation.

**Features:**

- Compare CPU and GPU feature computation performance
- Test with real LAZ files or synthetic data
- Multi-size benchmarking (1K to 5M points)
- Automatic GPU detection and CPU fallback
- Detailed performance metrics and speedup calculations

**Usage:**

```bash
# Test with real LAZ file
python scripts/benchmarks/benchmark_gpu.py path/to/file.laz

# Quick test with synthetic data
python scripts/benchmarks/benchmark_gpu.py --synthetic

# Multi-size benchmark (comprehensive)
python scripts/benchmarks/benchmark_gpu.py --multi-size

# Custom parameters
python scripts/benchmarks/benchmark_gpu.py --synthetic --k 20 --runs 5
```

**Example Output:**

```
================================================================================
📊 BENCHMARK RESULTS
================================================================================
Dataset:        100,000 points
K-neighbors:    10

🖥️  CPU Performance:
  Best time:    0.42s
  Average:      0.50s ± 0.08s
  Throughput:   237,616 points/s

⚡ GPU Performance:
  Best time:    0.08s
  Average:      0.09s ± 0.01s
  Throughput:   1,250,000 points/s

🚀 Speedup:       5.25x faster on GPU
   Time saved:   0.34s (81.0%)
================================================================================
```

### benchmark_optimization.py

Benchmarks feature computation performance (moved from tests/).

**Usage:**

```bash
python scripts/benchmarks/benchmark_optimization.py <laz_file>
```

## Purpose

These scripts measure performance of various components to identify
bottlenecks and validate optimizations.
