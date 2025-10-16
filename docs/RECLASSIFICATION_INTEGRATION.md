# Reclassification Integration Guide

## Overview

Reclassification is now available as an **optional feature** in the main processing pipeline. This allows you to apply optimized ground truth classification using spatial indexing for fast, accurate ASPRS code assignment.

## When to Use Reclassification

### Use Reclassification When:

- ✅ You want the most accurate ASPRS classification codes from BD TOPO®
- ✅ You need to update existing classifications without reprocessing everything
- ✅ You want GPU-accelerated classification updates
- ✅ You have ground truth data (BD TOPO®) available

### Don't Use Reclassification When:

- ❌ You only need basic geometric classification
- ❌ You don't have BD TOPO® data access
- ❌ You're doing initial exploratory processing

## Configuration

### Method 1: Enable in Main Pipeline

Add the `reclassification` section to your processor config:

```yaml
processor:
  lod_level: "LOD2"
  processing_mode: "both"
  num_workers: 4

  # Enable reclassification
  reclassification:
    enabled: true
    acceleration_mode: "auto" # cpu, gpu, gpu+cuml, or auto
    chunk_size: 100000
    show_progress: true
    use_geometric_rules: true
```

### Method 2: Use Standalone Reclassification Mode

For reclassification-only processing, use the dedicated config:

```bash
ign-lidar-hd process --config-file configs/reclassification_config.yaml
```

## Configuration Options

### Core Settings

| Parameter           | Type | Default | Description                         |
| ------------------- | ---- | ------- | ----------------------------------- |
| `enabled`           | bool | false   | Enable reclassification in pipeline |
| `acceleration_mode` | str  | "auto"  | Backend: cpu, gpu, gpu+cuml, auto   |
| `chunk_size`        | int  | 100000  | Points per processing chunk         |
| `show_progress`     | bool | true    | Show progress bars                  |
| `skip_existing`     | bool | true    | Skip already reclassified files     |

### GPU Performance Settings

| Parameter        | Type | Default | Description                   |
| ---------------- | ---- | ------- | ----------------------------- |
| `gpu_chunk_size` | int  | 500000  | Chunk size for GPU processing |
| `cpu_chunk_size` | int  | 100000  | Chunk size for CPU processing |

### Geometric Rules (Advanced)

| Parameter                          | Type  | Default | Description                           |
| ---------------------------------- | ----- | ------- | ------------------------------------- |
| `use_geometric_rules`              | bool  | true    | Apply geometric refinement rules      |
| `ndvi_vegetation_threshold`        | float | 0.3     | NDVI threshold for vegetation         |
| `ndvi_road_threshold`              | float | 0.15    | NDVI threshold for roads              |
| `road_vegetation_height_threshold` | float | 2.0     | Height (m) above road for vegetation  |
| `building_buffer_distance`         | float | 2.0     | Buffer (m) around buildings           |
| `max_building_height_difference`   | float | 3.0     | Max height (m) for building points    |
| `verticality_threshold`            | float | 0.7     | Verticality score (0-1) for buildings |
| `verticality_search_radius`        | float | 1.0     | Search radius (m) for verticality     |
| `min_vertical_neighbors`           | int   | 5       | Min neighbors for verticality check   |

## Usage Examples

### Example 1: Standard Pipeline with Reclassification

```yaml
# config.yaml
processor:
  lod_level: "LOD2"
  processing_mode: "both"

  reclassification:
    enabled: true
    acceleration_mode: "auto"
    use_geometric_rules: true

data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      railways: true
      water: true
```

Run:

```bash
ign-lidar-hd process --config-file config.yaml
```

### Example 2: GPU-Accelerated Reclassification

```yaml
processor:
  use_gpu: true

  reclassification:
    enabled: true
    acceleration_mode: "gpu+cuml" # Use full RAPIDS stack
    gpu_chunk_size: 500000
    show_progress: true
```

### Example 3: CPU-Only with Geometric Rules

```yaml
processor:
  reclassification:
    enabled: true
    acceleration_mode: "cpu"
    cpu_chunk_size: 100000
    use_geometric_rules: true
    ndvi_vegetation_threshold: 0.35 # More strict
```

### Example 4: Disable Reclassification (Default)

```yaml
processor:
  # Reclassification not configured = disabled (default behavior)
  lod_level: "LOD2"
  processing_mode: "both"
```

Or explicitly:

```yaml
processor:
  reclassification:
    enabled: false
```

## Processing Flow

### With Reclassification Enabled

```
1. Load tile
2. Compute geometric features
3. Apply initial classification
4. Fetch ground truth from BD TOPO®
5. Apply ground truth classification
6. ✨ APPLY RECLASSIFICATION (optimized)
7. Refine classification (NDVI, geometric rules)
8. Save enriched LAZ
9. Extract patches (if enabled)
```

### Without Reclassification (Default)

```
1. Load tile
2. Compute geometric features
3. Apply initial classification
4. Fetch ground truth from BD TOPO®
5. Apply ground truth classification
6. Refine classification (NDVI, geometric rules)
7. Save enriched LAZ
8. Extract patches (if enabled)
```

## Performance Comparison

### CPU Processing (18M points)

- Without reclassification: ~2-3 min
- With reclassification (CPU): ~7-13 min
- With reclassification (GPU): ~3-4 min

### GPU Processing (18M points)

- Without reclassification: ~1-2 min
- With reclassification (GPU): ~2-3 min
- With reclassification (GPU+cuML): ~1.5-2.5 min

## Backend Selection

### Auto Mode (Recommended)

```yaml
acceleration_mode: "auto"
```

Automatically selects the best available backend:

1. Try GPU+cuML (RAPIDS)
2. Fall back to GPU (CuPy)
3. Fall back to CPU (Shapely STRtree)

### Manual Selection

**CPU Mode:**

```yaml
acceleration_mode: "cpu"
```

- Uses Shapely STRtree for spatial indexing
- Works on any system
- Slower but reliable

**GPU Mode:**

```yaml
acceleration_mode: "gpu"
```

- Uses CuPy for GPU acceleration
- Requires: CUDA, CuPy
- 3-5x faster than CPU

**GPU+cuML Mode:**

```yaml
acceleration_mode: "gpu+cuml"
```

- Uses full RAPIDS stack
- Requires: CUDA, CuPy, cuML
- 5-10x faster than CPU

## Data Requirements

Reclassification requires:

- ✅ Ground truth data from BD TOPO®
- ✅ Point cloud with XYZ coordinates
- ✅ Valid tile bounding box

Optional (improves accuracy):

- NDVI values (for vegetation refinement)
- Height above ground (for building classification)
- Planarity/verticality (for geometric rules)

## Troubleshooting

### Issue: Reclassification Not Running

**Check:**

1. Is `reclassification.enabled: true`?
2. Is `data_sources.bd_topo.enabled: true`?
3. Are there any ground truth features?

### Issue: Slow Performance

**Solutions:**

1. Try GPU acceleration: `acceleration_mode: "gpu"`
2. Increase chunk size: `gpu_chunk_size: 500000`
3. Disable progress bars: `show_progress: false`

### Issue: GPU Out of Memory

**Solutions:**

1. Reduce chunk size: `gpu_chunk_size: 250000`
2. Fall back to CPU: `acceleration_mode: "cpu"`

### Issue: Inaccurate Results

**Solutions:**

1. Enable geometric rules: `use_geometric_rules: true`
2. Adjust NDVI thresholds
3. Check ground truth data quality

## Complete Example Configuration

See: [`configs/processing_with_reclassification.yaml`](processing_with_reclassification.yaml)

```bash
# Run with reclassification enabled
ign-lidar-hd process --config-file configs/processing_with_reclassification.yaml

# Override acceleration mode
ign-lidar-hd process \
  --config-file configs/processing_with_reclassification.yaml \
  processor.reclassification.acceleration_mode=gpu

# Disable reclassification at runtime
ign-lidar-hd process \
  --config-file configs/processing_with_reclassification.yaml \
  processor.reclassification.enabled=false
```

## Best Practices

1. **Start with auto mode**: Let the system choose the best backend
2. **Enable geometric rules**: Improves accuracy significantly
3. **Use caching**: Enable BD TOPO® caching for faster reruns
4. **Monitor performance**: Use `show_progress: true` initially
5. **Test on subset**: Use `max_tiles: 10` for testing

## Related Documentation

- [Reclassification Configuration Reference](reclassification_config.yaml)
- [GPU Reclassification Guide](../docs/GPU_RECLASSIFICATION_GUIDE.md)
- [Classification Audit Report](../docs/CLASSIFICATION_AUDIT_REPORT.md)

## Version History

- **v2.5.4** (2025-10-16): Added reclassification as optional pipeline feature
- **v2.5.3** (2025-10-16): Standalone reclassification mode
- **v2.5.0** (2025-10-15): Ground truth classification integration
