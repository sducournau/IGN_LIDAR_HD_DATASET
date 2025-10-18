# Hydra Configuration System - Update Summary

## Problem

The example configuration files (`examples/config_versailles_*.yaml`) were using `preset: lod3` syntax, which is specific to `PresetConfigLoader` (Python API), not compatible with Hydra's `--config-file` loading system.

**Error encountered:**

```
Error: Missing key processor
    full_key: processor
    object_type=dict
```

## Root Cause

When loading configs via `--config-file`, the `HydraRunner` was:

1. Loading the YAML file with `OmegaConf.load()` (doesn't process `defaults`)
2. Not following Hydra's `defaults` inheritance chain
3. Result: preset configurations were never loaded, causing missing sections

## Solution

### 1. Updated Example Configs (Hydra Syntax)

All example configs now use proper Hydra `defaults` system:

```yaml
# BEFORE (PresetConfigLoader syntax - doesn't work with --config-file)
preset: lod3

# AFTER (Hydra defaults system - works with --config-file)
defaults:
  - presets/lod3
  - _self_
```

**Files updated:**

- ✅ `examples/config_versailles_lod3_v5.1.yaml`
- ✅ `examples/config_versailles_lod2_v5.1.yaml`
- ✅ `examples/config_versailles_asprs_v5.1.yaml`
- ✅ `examples/config_architectural_training_v5.1.yaml`
- ✅ `examples/config_architectural_analysis_v5.1.yaml`

### 2. Enhanced HydraRunner (Defaults Processing)

Updated `ign_lidar/cli/hydra_runner.py` to properly handle `defaults` section when using `--config-file`:

**New features:**

- ✅ Processes `defaults` list in custom config files
- ✅ Recursively loads referenced presets (e.g., `presets/lod3`)
- ✅ Handles relative paths (e.g., `../base`)
- ✅ Merges configs in correct order: base → defaults → user → overrides
- ✅ Supports nested defaults (presets that include other presets)

**Code changes:**

```python
# Handle Hydra 'defaults' section if present
if 'defaults' in user_cfg:
    defaults = user_cfg.get('defaults', [])
    # Load each default config and merge
    for default in defaults:
        if default == '_self_':
            continue
        # Load preset from config directory
        default_path = self.config_dir / f"{default}.yaml"
        if default_path.exists():
            default_cfg = OmegaConf.load(default_path)
            base_cfg = OmegaConf.merge(base_cfg, default_cfg)
```

## Usage

### Option 1: Using --config-file (Now Fixed!)

```bash
# Works correctly now - loads LOD3 preset via defaults
python -m ign_lidar.cli.main process \
    --config-file examples/config_versailles_lod3_v5.1.yaml

# Preview configuration
python -m ign_lidar.cli.main process \
    --config-file examples/config_versailles_lod3_v5.1.yaml \
    --show-config
```

### Option 2: Using Overrides (Alternative)

```bash
# Manually specify settings via overrides
python -m ign_lidar.cli.main process \
    input_dir=/mnt/c/Users/Simon/ign/versailles \
    output_dir=/mnt/c/Users/Simon/ign/versailles_output \
    processor.lod_level=LOD3 \
    processor.use_gpu=true \
    features.mode=lod3
```

### Option 3: Using PresetConfigLoader (Python API)

```python
from ign_lidar.config import load_config_with_preset

# This still works with preset: lod3 syntax
config = load_config_with_preset(
    preset="lod3",
    config_file="examples/config_versailles_lod3_v5.1.yaml"
)
```

## Verification

Test that configs load correctly:

```bash
# Check LOD3 config
python -m ign_lidar.cli.main process \
    --config-file examples/config_versailles_lod3_v5.1.yaml \
    --show-config | grep "processor:"

# Expected output:
# processor:
#   lod_level: LOD3
#   processing_mode: enriched_only
#   use_gpu: true
#   gpu_batch_size: 8000000
#   ...

# Check features section
python -m ign_lidar.cli.main process \
    --config-file examples/config_versailles_lod3_v5.1.yaml \
    --show-config | grep "features:"

# Expected output:
# features:
#   mode: lod3
#   k_neighbors: 30
#   use_gpu_chunked: true
#   ...
```

## Configuration Hierarchy

The system now properly follows this inheritance chain:

```
base.yaml
└── presets/lod3.yaml
    └── examples/config_versailles_lod3_v5.1.yaml
        └── CLI overrides (highest priority)
```

Each level overrides the previous:

1. **base.yaml**: Default settings for everything
2. **presets/lod3.yaml**: LOD3-specific overrides (GPU settings, features, etc.)
3. **config_versailles_lod3_v5.1.yaml**: User-specific paths and settings
4. **CLI overrides**: Runtime adjustments

## Migration Guide

If you have custom configs using old syntax:

```yaml
# OLD (doesn't work with --config-file)
preset: lod3
input_dir: /path/to/data
output_dir: /path/to/output

# NEW (works with --config-file)
defaults:
  - presets/lod3  # or presets/lod2, presets/asprs, etc.
  - _self_

input_dir: /path/to/data
output_dir: /path/to/output
```

**Note:** The `_self_` marker ensures user settings override preset settings.

## Testing

All example configs verified working:

```bash
# Test each config
for config in examples/config_*.yaml; do
    echo "Testing: $config"
    python -m ign_lidar.cli.main process \
        --config-file "$config" \
        --show-config | head -5
done
```

## Benefits

1. ✅ **Consistent syntax**: All configs use Hydra defaults system
2. ✅ **Preset inheritance**: Properly loads preset configurations
3. ✅ **Error prevention**: No more "Missing key processor" errors
4. ✅ **Flexibility**: Can override any preset setting
5. ✅ **Backward compatibility**: PresetConfigLoader still works for Python API

## Related Issues Fixed

- ❌ **BEFORE**: `--config-file` didn't load presets → missing processor section
- ✅ **AFTER**: `--config-file` fully supports Hydra defaults system

---

**Status**: ✅ All Hydra configs updated and verified working
**Package**: Reinstalled with `pip install -e .`
**Ready**: To run full pipeline with GPU synchronization fixes!
