# Quick Start: Reclassification in Main Pipeline

## TL;DR

Reclassification is now **optional** in the main pipeline. Add this to enable:

```yaml
processor:
  reclassification:
    enabled: true
```

## Three Ways to Use Reclassification

### 1. Main Pipeline with Reclassification (NEW ✨)

```bash
# Use the example config
ign-lidar-hd process --config-file configs/processing_with_reclassification.yaml
```

**When to use:** You want both enriched LAZ files and patches with accurate classification

### 2. Reclassification Only (Existing)

```bash
# Fast updates to existing files
ign-lidar-hd process --config-file configs/reclassification_config.yaml
```

**When to use:** You already have enriched files and just need to update classification

### 3. Standard Processing (Default)

```bash
# Regular processing without reclassification
ign-lidar-hd process --config-file configs/processing_config.yaml
```

**When to use:** Quick processing, testing, or when you don't need BD TOPO® accuracy

## Minimal Config Example

```yaml
# config_with_reclassification.yaml
processor:
  lod_level: "LOD2"
  processing_mode: "both"

  # Add this section to enable
  reclassification:
    enabled: true
    acceleration_mode: "auto"

data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      railways: true
```

Run:

```bash
ign-lidar-hd process --config-file config_with_reclassification.yaml
```

## Toggle at Runtime

```bash
# Enable reclassification
ign-lidar-hd process \
  --config-file configs/processing_config.yaml \
  processor.reclassification.enabled=true

# Disable reclassification
ign-lidar-hd process \
  --config-file configs/processing_with_reclassification.yaml \
  processor.reclassification.enabled=false

# Change acceleration mode
ign-lidar-hd process \
  --config-file configs/processing_with_reclassification.yaml \
  processor.reclassification.acceleration_mode=gpu
```

## Common Configurations

### Fast Processing (No Reclassification)

```yaml
processor:
  processing_mode: "patches_only"
  # No reclassification section
```

### Accurate Processing (With Reclassification)

```yaml
processor:
  processing_mode: "both"
  reclassification:
    enabled: true
    use_geometric_rules: true
```

### GPU-Accelerated Reclassification

```yaml
processor:
  use_gpu: true
  reclassification:
    enabled: true
    acceleration_mode: "gpu+cuml"
    gpu_chunk_size: 500000
```

## Performance Guide

| Mode                 | Time per Tile | Accuracy  | Use Case                 |
| -------------------- | ------------- | --------- | ------------------------ |
| No reclassification  | 2-3 min       | Good      | Fast processing, testing |
| CPU reclassification | 7-13 min      | Excellent | Standard production      |
| GPU reclassification | 3-4 min       | Excellent | Fast + accurate          |

## See Also

- **Full Guide:** [docs/RECLASSIFICATION_INTEGRATION.md](RECLASSIFICATION_INTEGRATION.md)
- **Example Config:** [configs/processing_with_reclassification.yaml](../configs/processing_with_reclassification.yaml)
- **Summary:** [docs/RECLASSIFICATION_INTEGRATION_SUMMARY.md](RECLASSIFICATION_INTEGRATION_SUMMARY.md)
