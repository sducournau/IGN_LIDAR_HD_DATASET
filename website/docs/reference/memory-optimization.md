---
sidebar_position: 1
title: Memory Optimization
description: Guide to managing memory usage during LiDAR processing
keywords: [memory, optimization, performance, troubleshooting]
---

# Memory Optimization Guide

Learn how to optimize memory usage and avoid out-of-memory errors when processing large LiDAR datasets.

## Understanding Memory Requirements

LiDAR processing is memory-intensive, especially for building component analysis. Here's what you need to know about memory usage patterns.

### Memory Usage by Processing Mode

#### Core Mode Features

- **Base features**: ~40 bytes per point (normals, curvature, etc.)
- **KDTree**: ~24 bytes per point
- **Total**: ~70 bytes per point

#### Building Mode Features

- **Base features**: ~40 bytes per point
- **Building KDTree**: ~50 bytes per point
- **Additional features**: ~60 bytes per point
- **Total**: ~150 bytes per point

### File Size vs Memory Requirements

| File Size | Points      | Core Mode RAM | Building Mode RAM |
| --------- | ----------- | ------------- | ----------------- |
| 100MB     | ~2M points  | ~140MB        | ~300MB            |
| 200MB     | ~4M points  | ~280MB        | ~600MB            |
| 300MB     | ~6M points  | ~420MB        | ~900MB            |
| 500MB     | ~10M points | ~700MB        | ~1.5GB            |
| 1GB       | ~20M points | ~1.4GB        | ~3GB              |

## Automatic Memory Management

The library includes built-in memory management features:

### 1. Pre-flight Memory Check

Before processing starts, the system:

- Checks available RAM
- Detects swap usage
- Estimates memory needs
- Adjusts worker count automatically

```bash
# System automatically adjusts based on available memory
ign-lidar-hd enrich \
  --input-dir /path/to/tiles/ \
  --output /path/to/enriched/ \
  --mode full \
  --num-workers 4  # May be reduced automatically
```

**Console output example:**

```
Available RAM: 16.2 GB
High swap usage detected (65%) - reducing workers from 4 to 1
Processing with 1 worker for safety
```

### 2. Sequential Batching

For large files, the system automatically switches to sequential processing:

- **Small files** (under 200MB): Process multiple files in parallel
- **Medium files** (200-300MB): Reduce batch size
- **Large files** (over 300MB): Process one file at a time

### 3. Aggressive Chunking

For very large point clouds, memory-intensive operations are chunked:

```python
# Automatic chunking based on file size
if n_points > 5_000_000:
    chunk_size = 500_000      # Very aggressive
elif n_points > 3_000_000:
    chunk_size = 750_000      # Moderate
else:
    chunk_size = 1_000_000    # Standard
```

## Manual Memory Configuration

### Choosing Worker Count

Base your worker count on available RAM:

```bash
# For 8GB RAM systems
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode full \
  --num-workers 2

# For 16GB RAM systems
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode full \
  --num-workers 4

# For 32GB+ RAM systems
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode full \
  --num-workers 6
```

### Safe Worker Guidelines

| System RAM | Building Mode | Core Mode    |
| ---------- | ------------- | ------------ |
| 8GB        | 1-2 workers   | 2-3 workers  |
| 16GB       | 2-4 workers   | 4-6 workers  |
| 32GB       | 4-8 workers   | 8-12 workers |
| 64GB+      | 8-16 workers  | 16+ workers  |

### Force Conservative Processing

For maximum safety on constrained systems:

```bash
# Guaranteed to work (slowest but safest)
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode full \
  --num-workers 1
```

## Monitoring Memory Usage

### System Monitoring

Monitor your system during processing:

```bash
# Monitor memory and swap
htop

# Watch memory usage
watch -n 1 'free -h'

# Check swap usage
cat /proc/swaps
```

### Warning Signs

Watch for these indicators of memory pressure:

- **High swap usage** (over 50%)
- **System becoming unresponsive**
- **Process killed messages** in logs
- **Decreasing available memory** over time

### Log Messages

The system provides helpful log messages:

```
⚠️  High swap usage detected (75%)
⚠️  Reducing workers from 4 to 1 for safety
⚠️  Large file detected (523MB) - using sequential processing
✅ Memory check passed: 12.3GB available, 1.2GB needed
```

## Troubleshooting Memory Issues

### Out of Memory Crashes

**Symptoms:**

- Process terminated abruptly
- "Process pool was terminated" errors
- System freezes or becomes unresponsive

**Solutions:**

1. **Reduce worker count:**

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode full \
  --num-workers 1
```

2. **Process files individually:**

```bash
# Process each file separately
for file in tiles/*.laz; do
    ign-lidar-hd enrich \
      --input-dir "$file" \
      --output enriched/ \
      --mode full \
      --num-workers 1
done
```

3. **Use core mode instead of building mode:**

```bash
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode core \
  --num-workers 2
```

### High Memory Usage

**Symptoms:**

- Processing very slow
- High swap usage
- System warnings about low memory

**Solutions:**

1. **Close unnecessary applications**
2. **Increase system swap space:**

```bash
# Add temporary swap file (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

3. **Process smaller batches:**

```bash
# Split large datasets into smaller chunks
find tiles/ -name "*.laz" | head -10 | xargs -I {} ign-lidar-hd enrich --input {}
```

### Memory Leaks

**Symptoms:**

- Memory usage keeps increasing
- Eventually runs out of memory
- Processing gets slower over time

**Solutions:**

1. **Restart processing periodically:**

```bash
# Process in smaller batches
ign-lidar-hd enrich --input-dir batch1/ --output enriched/
ign-lidar-hd enrich --input-dir batch2/ --output enriched/
```

2. **Use smart skip to resume:**

```bash
# Safe to re-run - skips completed files
ign-lidar-hd enrich \
  --input-dir tiles/ \
  --output enriched/ \
  --mode full
```

## Performance Optimization

### SSD vs HDD

Using SSD storage significantly improves performance:

- **SSD**: Faster file I/O reduces memory pressure
- **HDD**: Slower I/O can cause memory buildup

### Memory Type

- **DDR4/DDR5**: Faster RAM improves processing speed
- **Sufficient capacity**: More important than speed for large datasets

### System Configuration

```bash
# Optimize system for large dataset processing (Linux)
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.overcommit_memory=1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Best Practices

### 1. Start Small

Test with a few files first:

```bash
# Test with 2-3 files
ign-lidar-hd enrich \
  --input-dir sample_tiles/ \
  --output test_enriched/ \
  --mode full \
  --num-workers 2
```

### 2. Monitor First Run

Watch system resources during your first processing run:

- Open `htop` or Task Manager
- Monitor memory and swap usage
- Note peak memory consumption
- Adjust workers accordingly for subsequent runs

### 3. Use Smart Skip

Let smart skip handle interrupted processing:

```bash
# Safe to interrupt and restart
ign-lidar-hd enrich \
  --input-dir large_dataset/ \
  --output enriched/ \
  --mode full
# Press Ctrl+C if needed, then re-run same command
```

### 4. Plan Disk Space

Enriched files are similar in size to input files:

- **Input**: 300MB LAZ file
- **Output**: ~310MB enriched LAZ file
- **Temporary**: Up to 2x during processing

Ensure sufficient disk space: `input_size × 3` for safety.

### 5. Batch Processing Strategy

For very large datasets:

```bash
# Process by geographic region
ign-lidar-hd enrich --input-dir paris_tiles/ --output enriched/
ign-lidar-hd enrich --input-dir lyon_tiles/ --output enriched/

# Or process by file size
ign-lidar-hd enrich --input-dir small_tiles/ --output enriched/
ign-lidar-hd enrich --input-dir large_tiles/ --output enriched/
```

## Advanced Configuration

### Environment Variables

```bash
# Set memory limits (if needed)
export MEMORY_LIMIT_GB=8
export MAX_WORKERS=2

# Then run processing
ign-lidar-hd enrich --input-dir tiles/ --output enriched/
```

### Python Memory Settings

```python
# For programmatic usage
from ign_lidar import LiDARProcessor

processor = LiDARProcessor(
    max_memory_gb=8,        # Limit memory usage
    chunk_size=500_000,     # Smaller chunks
    n_jobs=2                # Fewer workers
)
```

## See Also

- [Basic Usage Guide](../guides/basic-usage.md) - Essential processing workflows
- [CLI Commands](../guides/cli-commands.md) - Command-line options
- [Smart Skip Features](../features/smart-skip.md) - Resuming interrupted work
