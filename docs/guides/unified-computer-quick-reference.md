# UnifiedFeatureComputer Quick Reference

**Quick lookup for UnifiedFeatureComputer configuration options**

## Basic Usage

### Automatic Mode Selection (Recommended)

```yaml
processor:
  use_unified_computer: true
```

### Force Specific Mode

```yaml
processor:
  use_unified_computer: true
  computation_mode: "gpu_chunked" # cpu | gpu | gpu_chunked | boundary
```

## Configuration Options

| Option                    | Type   | Default   | Description                                 |
| ------------------------- | ------ | --------- | ------------------------------------------- |
| `use_unified_computer`    | bool   | `false`   | Enable UnifiedFeatureComputer               |
| `computation_mode`        | string | `null`    | Force mode: cpu, gpu, gpu_chunked, boundary |
| `typical_points_per_tile` | int    | `2000000` | Hint for mode selection                     |
| `tile_size`               | float  | `1000`    | Tile size in meters (for estimation)        |

## Mode Selection Logic

### Automatic Selection

- **No GPU** → CPU
- **GPU + <500K points** → GPU
- **GPU + ≥500K points** → GPU_CHUNKED

### Manual Override

Set `computation_mode` to force specific mode:

- `"cpu"` - CPU computation
- `"gpu"` - Full GPU (small/medium workloads)
- `"gpu_chunked"` - Chunked GPU (large workloads)
- `"boundary"` - Boundary-aware processing

## Comparison: Legacy vs Unified

| Feature             | Legacy Strategy              | UnifiedFeatureComputer |
| ------------------- | ---------------------------- | ---------------------- |
| **Configuration**   | Multiple flags               | Single flag            |
| **Mode Selection**  | Manual                       | Automatic              |
| **API**             | Varies by strategy           | Consistent             |
| **GPU Control**     | `use_gpu`, `use_gpu_chunked` | `computation_mode`     |
| **Batch Size**      | Manual setting               | Automatic              |
| **Recommendations** | None                         | Logged                 |

## Example Configs

### Small Workload

```yaml
processor:
  use_unified_computer: true
  computation_mode: "cpu" # or let it auto-select
features:
  k_neighbors: 20
```

### Large Workload

```yaml
processor:
  use_unified_computer: true
  computation_mode: "gpu_chunked"
features:
  k_neighbors: 20
```

### Mixed Workload (Automatic)

```yaml
processor:
  use_unified_computer: true
  typical_points_per_tile: 1500000
features:
  k_neighbors: 20
```

## Log Messages

### Mode Selection

```
ℹ️  Automatic mode selection: GPU_CHUNKED
    Reason: Large workload (2.5M points), GPU available
```

### Recommendations

```
💡 Expert Recommendation:
    - Use GPU_CHUNKED for workloads >500K points
    - Consider increasing k_neighbors for better quality
```

## Troubleshooting

| Issue               | Solution                                        |
| ------------------- | ----------------------------------------------- |
| Not using GPU       | Set `computation_mode: "gpu"` explicitly        |
| Too slow            | Verify GPU available, check mode selection logs |
| Import error        | Run `pip install -e .` to reinstall             |
| Wrong mode selected | Provide `typical_points_per_tile` hint          |

## Feature Comparison

### Supported Features (Both Paths)

✅ Normals  
✅ Curvature  
✅ Geometric features (planarity, linearity, etc.)  
✅ Height (z-normalized)  
✅ Distance to center

### Strategy Pattern Only

⚠️ Boundary-aware wrapping  
⚠️ Custom strategies  
⚠️ Legacy factory pattern

## Performance Guidelines

| Points/Tile | Recommended Mode | Config                            |
| ----------- | ---------------- | --------------------------------- |
| <100K       | CPU              | `computation_mode: "cpu"`         |
| 100K-500K   | GPU              | `computation_mode: "gpu"`         |
| >500K       | GPU_CHUNKED      | `computation_mode: "gpu_chunked"` |
| Any (auto)  | Automatic        | `use_unified_computer: true`      |

## Migration Summary

### Old Config

```yaml
processor:
  use_gpu: true
  use_gpu_chunked: true
  gpu_batch_size: 5000000
```

### New Config

```yaml
processor:
  use_unified_computer: true
```

### Keep Legacy

```yaml
processor:
  use_unified_computer: false # or omit
  use_gpu: true
  use_gpu_chunked: true
```

## Links

- **Migration Guide**: `docs/guides/migration-unified-computer.md`
- **Example Configs**: `examples/config_unified_*.yaml`
- **Tests**: `tests/test_orchestrator_unified_integration.py`

---

**Last Updated**: October 18, 2025  
**Version**: Phase 4 Task 1.4
