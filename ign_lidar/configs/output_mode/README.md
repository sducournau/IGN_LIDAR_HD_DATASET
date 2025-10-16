# Output Mode Configuration

This directory contains modular configuration files for different processing output modes.

## Available Output Modes

### 1. Enriched Only (`enriched_only.yaml`)

**What it does:** Creates only enriched LAZ files with added features

**Use cases:**

- Preprocessing for later patch extraction
- GIS analysis without ML training
- Quick feature enrichment
- Fastest option

**Output:**

- ✅ Enriched LAZ files
- ❌ No patches

### 2. Patches Only (`patches_only.yaml`)

**What it does:** Creates only ML training patches (NPZ/HDF5)

**Use cases:**

- ML model training
- Standard dataset creation
- When enriched LAZ not needed

**Output:**

- ❌ No enriched LAZ
- ✅ Training patches

### 3. Both (`both.yaml`)

**What it does:** Creates both enriched LAZ AND ML patches

**Use cases:**

- Complete dataset creation
- When you need both GIS and ML outputs
- Archival processing

**Output:**

- ✅ Enriched LAZ files
- ✅ Training patches

## Usage

### In Experiment Configs

```yaml
defaults:
  - /output_mode: enriched_only # or patches_only, both
  - _self_
```

### Overriding Settings

```yaml
defaults:
  - /output_mode: patches_only
  - _self_

# Override patch settings
patch_size: 50.0 # Smaller patches
num_points: 32768 # More points per patch
```

## Comparison

| Mode            | Enriched LAZ | Patches | Speed  | Use Case          |
| --------------- | ------------ | ------- | ------ | ----------------- |
| `enriched_only` | ✅           | ❌      | Fast   | GIS preprocessing |
| `patches_only`  | ❌           | ✅      | Medium | ML training       |
| `both`          | ✅           | ✅      | Slow   | Complete workflow |

## Migration from Old Configs

### Old Way

```yaml
processor:
  processing_mode: enriched_only
  output_format: laz
  patch_size: null
  # ... lots of settings
```

### New Way

```yaml
defaults:
  - /output_mode: enriched_only # That's it!
  - _self_
```

## See Also

- [Data Sources](../data_sources/README.md)
- [Processor Configs](../processor/README.md)
- [Experiment Configs](../experiment/README.md)
