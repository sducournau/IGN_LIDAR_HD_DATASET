# Memory Optimization V2 - Out of Memory Fixes

## Problem

When enriching LAZ files in building mode with `--num-workers 2`, the process crashes with:

```
A process in the process pool was terminated abruptly while the future was running or pending.
```

This is caused by Out-of-Memory (OOM) conditions when processing large point clouds (5-7M points) with building-specific features.

## Root Causes

1. **Building mode is extremely memory-intensive**:

   - Main features: ~40 bytes/point (normals, curvature, etc.)
   - KDTree for main features: ~24 bytes/point
   - Building features need ANOTHER KDTree: ~50 bytes/point
   - **Total: ~120-150 bytes per point**

2. **Multiple workers competing for memory**:

   - Each 5M point file needs ~750MB-1GB RAM
   - With 2 workers: ~1.5-2GB concurrent usage
   - System had 6.9GB swap in use (memory pressure)

3. **Inefficient batch processing**:
   - Large files processed concurrently
   - No early memory checks
   - Insufficient garbage collection

## Solutions Implemented

### 1. Improved Memory Estimation (cli.py:154-190)

```python
# More accurate memory estimation based on mode
if mode == 'building':
    bytes_per_point = 150  # Conservative for building mode
else:
    bytes_per_point = 70   # Conservative for core mode

# Require 60% safety margin (was 80%)
safety_factor = 0.6
if estimated_needed_mb > available_mb * safety_factor:
    # Abort processing this file
    return False
```

### 2. Pre-flight System Memory Check (cli.py:406-437)

Checks system memory BEFORE starting any processing:

```python
# Check swap usage
if swap_percent > 50:
    logger.warning("High swap usage detected")
    args.num_workers = 1  # Force single worker

# Calculate safe worker count
min_gb_per_worker = 5.0 if mode == 'building' else 2.5
max_safe_workers = int(available_gb / min_gb_per_worker)
```

### 3. Sequential Batching for Large Files (cli.py:531-547)

```python
if mode == 'building':
    if max_file_size > 300_000_000:
        batch_size = 1  # Process ONE file at a time
    elif max_file_size > 200_000_000:
        batch_size = max(1, args.num_workers // 2)
    else:
        batch_size = args.num_workers
```

### 4. Aggressive Chunking for Radius Queries (cli.py:337-348)

The `compute_num_points_in_radius` function is particularly memory-hungry:

```python
if n_points > 5_000_000:
    radius_chunk_size = 500_000   # Very aggressive
elif n_points > 3_000_000:
    radius_chunk_size = 750_000
else:
    radius_chunk_size = 1_000_000
```

### 5. Explicit Memory Cleanup (features.py:394-449)

Added explicit cleanup in `compute_num_points_in_radius`:

```python
# Clear temporary data after each chunk
del indices, chunk_counts, chunk

# Explicit cleanup of KDTree (can be large)
del nbrs
gc.collect()
```

### 6. Directory Structure Preservation (cli.py:443-481)

**NEW FEATURE**: Preserves input directory structure and copies metadata:

```python
# Copies:
# - Root-level JSON and TXT files (stats.json, etc.)
# - Per-file JSON metadata (LHD_*.json)
# - Maintains subdirectories (coastal_urban/, heritage_palace/, etc.)

# Calculate relative path to preserve directory structure
rel_path = laz.relative_to(input_path)
output_path = output_dir / rel_path
```

## Results

### Memory Usage Improvement

- **Before**: 2 workers × 5M points = ~1.5-2GB concurrent (OOM crash)
- **After**: Sequential batching + better estimation = stable processing

### Safety Features

1. ✅ Pre-flight memory check
2. ✅ Swap usage detection
3. ✅ Automatic worker reduction
4. ✅ Conservative memory estimation
5. ✅ Aggressive chunking for large files
6. ✅ Explicit garbage collection
7. ✅ Directory structure preservation
8. ✅ Metadata file copying

## Usage Recommendations

### For Large Files (>300MB / >5M points)

```bash
# Let the system auto-adjust workers
python -m ign_lidar.cli enrich \
  --input /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --k-neighbors 20 \
  --mode building \
  --num-workers 2
```

System will automatically:

- Detect high swap usage → reduce to 1 worker
- Detect large files → use batch_size=1
- Check available RAM → adjust worker count
- Preserve directory structure
- Copy all metadata files

### For Many Small Files (<200MB)

```bash
# Can use more workers efficiently
python -m ign_lidar.cli enrich \
  --input /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --k-neighbors 20 \
  --mode building \
  --num-workers 4
```

### Emergency: Force Single Worker

```bash
# Guaranteed to work (slowest but safest)
python -m ign_lidar.cli enrich \
  --input /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --k-neighbors 20 \
  --mode building \
  --num-workers 1
```

## Memory Requirements (Building Mode)

| File Size | Points      | RAM Needed | Safe Workers (32GB RAM) |
| --------- | ----------- | ---------- | ----------------------- |
| 100MB     | ~2M points  | ~300MB     | 8                       |
| 200MB     | ~4M points  | ~600MB     | 4                       |
| 300MB     | ~6M points  | ~900MB     | 2                       |
| 500MB     | ~10M points | ~1.5GB     | 1-2                     |
| 1GB       | ~20M points | ~3GB       | 1                       |

## Output Structure

The output directory now **exactly mirrors** the input structure:

```
input/
  ├── stats.json
  ├── coastal_residential/
  │   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.laz
  │   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.json
  │   └── ...
  ├── urban_dense/
  │   ├── LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.laz
  │   ├── LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.json
  │   └── ...
  └── ...

output/
  ├── stats.json                          ← COPIED
  ├── coastal_residential/                ← PRESERVED
  │   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.laz  (enriched)
  │   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.json ← COPIED
  │   └── ...
  ├── urban_dense/                        ← PRESERVED
  │   ├── LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.laz  (enriched)
  │   ├── LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.json ← COPIED
  │   └── ...
  └── ...
```

## Technical Details

### Files Modified

1. `ign_lidar/cli.py`:
   - Enhanced memory estimation
   - Pre-flight system checks
   - Sequential batching logic
   - Directory structure preservation
   - Metadata file copying
2. `ign_lidar/features.py`:
   - Improved `compute_num_points_in_radius` with aggressive chunking
   - Explicit KDTree cleanup
   - Better garbage collection

### Key Parameters

- `safety_factor = 0.6`: Require 40% free memory buffer
- `batch_size = 1`: For files >300MB in building mode
- `radius_chunk_size = 500k`: For files >5M points
- `bytes_per_point = 150`: Conservative estimate for building mode

## Monitoring

The system now logs:

- Available memory at start
- Swap usage warnings
- Automatic worker adjustments
- File size detection
- Batch processing status
- Metadata copying progress

Watch for warnings like:

- `⚠️ High swap usage detected (XX%)`
- `⚠️ Reducing workers from X to Y`
- `⚠️ Using sequential batching for large files`

These indicate the system is protecting itself from OOM.
