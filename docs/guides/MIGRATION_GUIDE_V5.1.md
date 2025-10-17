# Migration Guide: V5.0 → V5.1

**From**: Verbose configuration files with 80%+ duplication  
**To**: Preset-based configuration with inheritance  
**Benefit**: 71% fewer lines, easier maintenance, clearer configs

---

## Overview

V5.1 introduces a preset-based configuration system that dramatically simplifies configuration management. This guide will help you migrate your existing V5.0 configs to the new system.

---

## Quick Migration

### 1. Identify Your Use Case

Choose the preset that matches your workflow:

| Your V5.0 Config                 | V5.1 Preset | Reason                          |
| -------------------------------- | ----------- | ------------------------------- |
| Building classification, facades | `lod2`      | LOD2 classes, building features |
| Detailed architecture, heritage  | `lod3`      | LOD3 classes, style detection   |
| Standard ASPRS classification    | `asprs`     | ASPRS LAS 1.4 codes             |
| Quick preview, testing           | `minimal`   | Fast, basic features only       |
| Training data, research          | `full`      | All features, both outputs      |

### 2. Create New Config

**Template**:

```yaml
# my_config_v5.1.yaml
preset: [chosen_preset]

input_dir: [your_input_path]
output_dir: [your_output_path]
# Add only custom overrides below
```

### 3. Add Custom Overrides

Compare your old config with the preset and add **only** what's different:

```bash
# View preset defaults
ign-lidar-hd presets --preset lod2
```

---

## Migration Examples

### Example 1: Versailles LOD2

**Before (V5.0)** - 188 lines:

```yaml
# config_versailles_lod2.yaml
defaults:
  - ../ign_lidar/configs/config
  - _self_

input_dir: /mnt/c/Users/Simon/ign/versailles
output_dir: /mnt/c/Users/Simon/ign/versailles_processed_lod2

processor:
  lod_level: LOD2
  architecture: hybrid
  use_gpu: true
  num_workers: 4
  patch_size: 50
  patch_overlap: 0.1
  num_points: 32768
  augment: false
  num_augmentations: 0
  pin_memory: true
  gpu_batch_size: 1_000_000
  use_optimized_ground_truth: true
  reclassification:
    enabled: true
    acceleration_mode: "auto"
    chunk_size: 5_000_000
    show_progress: true
    use_geometric_rules: true

features:
  mode: full
  k_neighbors: 30
  search_radius: 1.5
  include_extra: true
  use_rgb: true
  use_nir: true
  compute_ndvi: true
  include_architectural_style: false
  sampling_method: fps
  normalize_xyz: true
  normalize_features: true
  gpu_batch_size: 1000000
  use_gpu_chunked: true

preprocess:
  enabled: true
  sor_k: 12
  sor_std: 2.0
  ror_radius: 1.0
  ror_neighbors: 4
  voxel_enabled: false
  voxel_size: 0.1

stitching:
  enabled: true
  buffer_size: 15.0
  auto_detect_neighbors: true
  auto_download_neighbors: true
  cache_enabled: true

ground_truth:
  enabled: true
  update_classification: false
  include_buildings: true
  include_roads: true
  include_water: true
  include_vegetation: true
  use_ndvi: true
  fetch_rgb_nir: false
  ndvi_vegetation_threshold: 0.3
  ndvi_building_threshold: 0.15
  cache_dir: /mnt/c/Users/Simon/ign/cache/ground_truth
  save_ground_truth_vectors: true

output:
  format: laz,npz
  processing_mode: patches_only
  save_stats: true
  save_metadata: true
  compression: null

num_workers: 4
verbose: true
log_level: INFO

bbox:
  xmin: null
  ymin: null
  xmax: null
  ymax: null

hydra:
  run:
    dir: outputs/versailles_lod2/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    name: versailles_lod2
    chdir: false
```

**After (V5.1)** - 43 lines:

```yaml
# config_versailles_lod2_v5.1.yaml
preset: lod2

input_dir: /mnt/c/Users/Simon/ign/versailles
output_dir: /mnt/c/Users/Simon/ign/versailles_processed_lod2

ground_truth:
  cache_dir: /mnt/c/Users/Simon/ign/cache/ground_truth

hydra:
  run:
    dir: outputs/versailles_lod2/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    name: versailles_lod2
```

**Reduction**: 188 → 43 lines (**-77%**)

---

### Example 2: Architectural Analysis

**Before (V5.0)** - 174 lines:

```yaml
# config_architectural_analysis.yaml
processor:
  lod_level: LOD3
  architecture: hybrid
  use_gpu: true
  num_workers: 4
  patch_size: null
  patch_overlap: 0.0
  num_points: null
  augment: false
  num_augmentations: 0
  include_architectural_style: true
  style_encoding: constant
  # ... 150+ more lines
```

**After (V5.1)** - 63 lines:

```yaml
# config_architectural_analysis_v5.1.yaml
preset: full

processor:
  patch_size: null # No patching - enrich full tiles
  num_workers: 4

features:
  include_architectural_style: true
  style_encoding: constant

output:
  processing_mode: tiles_only
  format: laz,ply,las
```

**Reduction**: 174 → 63 lines (**-64%**)

---

### Example 3: Training Data Generation

**Before (V5.0)** - 180 lines with extensive augmentation config

**After (V5.1)** - 61 lines:

```yaml
preset: full

input_dir: /data/diverse_tiles
output_dir: /data/training

processor:
  patch_size: 100.0
  augment: true
  num_augmentations: 3
  num_workers: 2

features:
  include_architectural_style: true
  style_encoding: onehot

output:
  processing_mode: patches_only
  format: npz,laz
  save_stats: true
```

**Reduction**: 180 → 61 lines (**-66%**)

---

## Step-by-Step Migration

### Step 1: Understand Your Current Config

1. **Identify the LOD level**

   ```yaml
   processor:
     lod_level: LOD2 # → Use lod2 preset
   ```

2. **Check the processing mode**

   ```yaml
   output:
     processing_mode: enriched_only  # Matches most presets
     processing_mode: patches_only   # Training-focused
     processing_mode: both           # Use full preset
   ```

3. **Note custom paths**
   ```yaml
   input_dir: /your/path # Always custom
   output_dir: /your/path # Always custom
   ```

### Step 2: Choose the Right Preset

| If your config has...                  | Use preset |
| -------------------------------------- | ---------- |
| `lod_level: LOD2`                      | `lod2`     |
| `lod_level: LOD3`                      | `lod3`     |
| `lod_level: ASPRS`                     | `asprs`    |
| Minimal features, fast processing      | `minimal`  |
| `processing_mode: both` + all features | `full`     |

### Step 3: Extract Custom Overrides

**Method**: Compare your config with the preset

```bash
# View preset defaults
ign-lidar-hd presets --preset lod2

# Compare with your config
diff <(ign-lidar-hd presets --preset lod2) your_old_config.yaml
```

**Keep only**:

- Custom paths (`input_dir`, `output_dir`, cache directories)
- Custom Hydra settings (job names, output directories)
- Settings that differ from the preset defaults
- Project-specific overrides (num_workers, patch sizes, etc.)

### Step 4: Create New Config

```yaml
# my_project_v5.1.yaml
preset: [chosen_preset]

# Paths (always custom)
input_dir: [your_input]
output_dir: [your_output]

# Custom overrides (only what's different)
processor: [only_custom_settings]

features: [only_custom_settings]

ground_truth:
  cache_dir: [your_cache_dir] # If custom

hydra:
  run:
    dir: [your_output_dir]
  job:
    name: [your_job_name]
```

### Step 5: Test the New Config

```bash
# Dry run to validate
ign-lidar-hd process --config my_project_v5.1.yaml --dry-run

# Test with PresetConfigLoader
python -c "
from ign_lidar.config.preset_loader import PresetConfigLoader
from pathlib import Path

loader = PresetConfigLoader()
config = loader.load(config_file=Path('my_project_v5.1.yaml'))

# Verify key settings
print(f'LOD Level: {config[\"processor\"][\"lod_level\"]}')
print(f'Input Dir: {config[\"input_dir\"]}')
print(f'Features Mode: {config[\"features\"][\"mode\"]}')
"
```

### Step 6: Run and Compare

```bash
# Run with new config
ign-lidar-hd process --config my_project_v5.1.yaml

# Compare output with old config results
# Should be identical except for any intentional changes
```

---

## Common Migration Patterns

### Pattern 1: Simple Path Override

**Most common case** - just change paths

```yaml
preset: lod2

input_dir: /data/my_project/tiles
output_dir: /data/my_project/processed

ground_truth:
  cache_dir: /data/my_project/cache
```

**Lines**: ~15 (vs ~180 before)

### Pattern 2: Custom Worker Count

```yaml
preset: lod2

input_dir: /data/tiles
output_dir: /data/output

processor:
  num_workers: 8 # Override for more parallelism
```

### Pattern 3: Analysis Mode (No Patches)

```yaml
preset: lod3

input_dir: /data/heritage_tiles
output_dir: /data/analyzed

processor:
  patch_size: null # Disable patching

output:
  processing_mode: tiles_only # Only enriched tiles
  format: laz,ply,las # Multiple formats
```

### Pattern 4: Training with Augmentation

```yaml
preset: full

input_dir: /data/training_tiles
output_dir: /data/patches

processor:
  patch_size: 100.0
  augment: true
  num_augmentations: 3

output:
  processing_mode: patches_only
  format: npz,laz
```

### Pattern 5: Custom Feature Settings

```yaml
preset: lod3

input_dir: /data/tiles
output_dir: /data/output

features:
  k_neighbors: 50 # More neighbors for smoother features
  include_architectural_style: true
  style_encoding: onehot # One-hot for ML
```

---

## What Gets Inherited?

When you use a preset, you inherit:

### From base.yaml (All Presets)

- GPU acceleration settings
- Logging configuration
- Optimized batch sizes
- Ground truth BD TOPO integration
- Preprocessing settings (outlier removal)
- Stitching configuration

### From lod2.yaml

- LOD2 classification (15 classes)
- Full geometric + spectral features
- Enriched LAZ output only
- Building-focused feature set

### From lod3.yaml

- LOD3 classification (30 classes)
- Architectural style detection (13 styles)
- Full geometric + spectral features
- Enriched LAZ output only

### From asprs.yaml

- ASPRS LAS 1.4 classification
- Full geometric + spectral features
- Enriched LAZ output only

### From minimal.yaml

- ASPRS classification
- Basic geometric features only
- No RGB/NIR/NDVI
- No preprocessing
- No ground truth

### From full.yaml

- LOD3 classification (30 classes)
- Architectural style detection
- Full feature suite
- Both enriched LAZ + training patches
- All optimizations enabled

---

## Validation Checklist

After migration, verify:

- [ ] Config loads without errors
- [ ] LOD level matches your intent
- [ ] Input/output paths are correct
- [ ] Custom cache directories are set
- [ ] Worker count is appropriate
- [ ] Features match your requirements
- [ ] Output format and mode are correct
- [ ] Hydra job name and directories are set
- [ ] Test run produces expected output
- [ ] Output quality matches old config

---

## Troubleshooting

### Config Won't Load

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('my_config.yaml'))"

# Test with PresetConfigLoader
python -c "
from ign_lidar.config.preset_loader import PresetConfigLoader
from pathlib import Path
loader = PresetConfigLoader()
config = loader.load(config_file=Path('my_config.yaml'))
print('✅ Config loaded!')
"
```

### Preset Not Found

```bash
# List available presets
ign-lidar-hd presets

# Valid names: minimal, lod2, lod3, asprs, full
# (case-sensitive)
```

### Settings Not Applied

```bash
# View effective config
ign-lidar-hd presets --preset [your_preset] > effective_config.yaml

# Check if your override is there
grep "your_setting" effective_config.yaml
```

### Output Different from V5.0

1. **Check LOD level** - Preset might use different level
2. **Check feature mode** - Verify `features.mode` matches
3. **Check processing mode** - Verify `output.processing_mode` matches
4. **Check CLI overrides** - Old scripts might have had CLI args

---

## Benefits of Migration

### Before (V5.0)

❌ 914 lines across 5 example configs  
❌ 80%+ duplication  
❌ Hard to see what's custom vs default  
❌ Easy to miss updates to defaults  
❌ Difficult to switch between modes

### After (V5.1)

✅ 261 lines across 5 example configs (-71%)  
✅ Zero duplication (DRY principle)  
✅ Only custom overrides visible  
✅ Automatic updates from preset changes  
✅ Easy preset switching (`preset: lod2` → `preset: lod3`)

---

## Need Help?

- **Configuration Guide**: [docs/guides/CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **Preset Details**: `ign-lidar-hd presets --preset [name]`
- **Examples**: `examples/config_*_v5.1.yaml`
- **Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues

---

**Last Updated**: October 17, 2025  
**Version**: 5.1.0  
**Migration Tool**: Available in next release (V5.2.0)
