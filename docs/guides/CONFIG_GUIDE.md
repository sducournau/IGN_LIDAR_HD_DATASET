# Configuration Guide - IGN LiDAR HD Dataset V5.1

**Version**: 5.1.0  
**Date**: October 17, 2025  
**Week 3**: Configuration System Refactoring

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Preset System](#preset-system)
4. [Configuration Architecture](#configuration-architecture)
5. [Available Presets](#available-presets)
6. [Creating Custom Configs](#creating-custom-configs)
7. [CLI Usage](#cli-usage)
8. [Configuration Reference](#configuration-reference)
9. [Migration Guide](#migration-guide)
10. [Troubleshooting](#troubleshooting)

---

## Overview

IGN LiDAR HD V5.1 introduces a **preset-based configuration system** that dramatically simplifies configuration management:

- **71% fewer lines** in example configs
- **Single source of truth** (`base.yaml`)
- **5 clear presets** for common use cases
- **Smart inheritance** chain: base → preset → custom → CLI
- **No more duplication** across config files

### Philosophy

> **"Only specify what's different"**

Instead of copying hundreds of lines of configuration, you inherit a preset and override only what's unique to your use case.

---

## Quick Start

### 1. List Available Presets

```bash
ign-lidar-hd presets
```

Output:

```
Available presets (5):
  🚀 minimal  - Quick preview and testing
  🏃 lod2     - Building modeling and facade detection
  🏗️ lod3     - Detailed architectural classification
  📐 asprs    - ASPRS LAS 1.4 standard classification
  🔬 full     - Maximum detail with all features
```

### 2. View Preset Details

```bash
ign-lidar-hd presets --preset lod2
```

### 3. Use a Preset

**Option A: Direct preset usage**

```bash
ign-lidar-hd process \
  --preset lod2 \
  --input-dir /path/to/tiles \
  --output-dir /path/to/output
```

**Option B: Custom config file**

```yaml
# my_config.yaml
preset: lod2

input_dir: /path/to/tiles
output_dir: /path/to/output
```

```bash
ign-lidar-hd process --config my_config.yaml
```

**Option C: Preset + CLI overrides**

```bash
ign-lidar-hd process \
  --preset lod2 \
  --input-dir /path/to/tiles \
  --output-dir /path/to/output \
  --override processor.num_workers=8 \
  --override features.k_neighbors=50
```

---

## Preset System

### What is a Preset?

A **preset** is a pre-configured set of settings optimized for a specific use case. Each preset inherits from `base.yaml` and overrides settings to achieve a particular goal.

### Inheritance Chain

```
base.yaml (defaults for everything)
    ↓
[preset].yaml (optimized for use case)
    ↓
my_config.yaml (your custom overrides)
    ↓
CLI arguments (runtime overrides)
```

**Example**: Using the `lod2` preset

1. Start with `base.yaml` (all defaults)
2. Apply `lod2.yaml` overrides (LOD2 classification, building features)
3. Apply your config file overrides (custom paths, settings)
4. Apply CLI overrides (runtime adjustments)

### Benefits

✅ **DRY Principle**: Don't Repeat Yourself - no config duplication  
✅ **Maintainability**: Update presets once, all configs benefit  
✅ **Clarity**: Only see what's different from the preset  
✅ **Flexibility**: Easy to experiment with different presets  
✅ **Documentation**: Presets serve as templates and examples

---

## Configuration Architecture

### File Structure

```
ign_lidar/configs/
├── base.yaml              # Single source of truth (445 lines)
└── presets/
    ├── minimal.yaml       # Quick preview (107 lines)
    ├── lod2.yaml          # Building modeling (118 lines)
    ├── lod3.yaml          # Detailed architecture (128 lines)
    ├── asprs.yaml         # ASPRS classification (140 lines)
    └── full.yaml          # Maximum detail (173 lines)

examples/
├── config_versailles_lod2_v5.1.yaml       # 43 lines (was 188)
├── config_versailles_lod3_v5.1.yaml       # 45 lines (was 189)
├── config_versailles_asprs_v5.1.yaml      # 49 lines (was 183)
├── config_architectural_analysis_v5.1.yaml # 63 lines (was 174)
└── config_architectural_training_v5.1.yaml # 61 lines (was 180)
```

### base.yaml - The Foundation

`base.yaml` contains **all default settings** with:

- Smart defaults that work for 80% of use cases
- Auto-detection (GPU, workers, memory)
- Comprehensive inline documentation
- Validation-ready structure

**Key Sections**:

- `processor`: Processing pipeline and hardware settings
- `features`: Feature computation configuration
- `preprocessing`: Outlier removal and cleaning
- `ground_truth`: BD TOPO classification integration
- `output`: Output formats and modes
- `logging`: Logging and verbosity

---

## Available Presets

### 1. 🚀 Minimal - Quick Preview

**File**: `ign_lidar/configs/presets/minimal.yaml`

**Use Case**: Fast preview, testing, development

**Key Settings**:

- ASPRS classification (standard classes)
- Basic geometric features only (no RGB/NIR/NDVI)
- No preprocessing (raw data)
- No ground truth classification
- Enriched LAZ output only

**Speed**: ⚡ **FASTEST** (baseline)

**When to Use**:

- Quick data inspection
- Testing workflow before full processing
- Development and debugging
- Limited compute resources

**Example**:

```bash
ign-lidar-hd process --preset minimal input/ output/
```

---

### 2. 🏃 LOD2 - Building Modeling

**File**: `ign_lidar/configs/presets/lod2.yaml`

**Use Case**: Building facade detection, roof planes, urban modeling

**Key Settings**:

- LOD2 classification (15 building-focused classes)
- Full geometric features (curvature, planarity, height)
- RGB + NIR + NDVI spectral features
- Ground truth from BD TOPO
- Geometric reclassification refinement
- Statistical + radius outlier removal
- Enriched LAZ output

**Speed**: 🚀 **FAST** (1.5× slower than minimal)

**Classes**:

- **Structural**: wall
- **Roofs**: flat, gable, hip
- **Details**: chimney, dormer, balcony, overhang, foundation
- **Context**: ground, vegetation (low/high), water, vehicle, other

**When to Use**:

- Building facade detection and modeling
- Roof plane extraction
- Urban modeling and city digital twins
- Building footprint refinement
- Architectural analysis (coarse level)

**Example**:

```yaml
# my_lod2_config.yaml
preset: lod2

input_dir: /data/city_tiles
output_dir: /data/processed
```

---

### 3. 🏗️ LOD3 - Detailed Architecture

**File**: `ign_lidar/configs/presets/lod3.yaml`

**Use Case**: Detailed architectural classification, heritage sites

**Key Settings**:

- LOD3 classification (30 detailed classes)
- Full geometric + spectral features
- Architectural style detection (13 styles)
- Ground truth from BD TOPO
- Geometric reclassification refinement
- Statistical + radius outlier removal
- Enriched LAZ output

**Speed**: 🏃 **MODERATE** (2.5× slower than minimal)

**Classes** (30 total):

- **Walls**: plain, with windows, with door
- **Roofs**: flat, gable, hip, mansard, shed, gambrel, butterfly, dome, other
- **Openings**: window, door, balcony door
- **Details**: chimney, dormer, balcony, overhang, gutter, parapet
- **Context**: foundation, ground, vegetation, water, vehicle, other

**Architectural Styles** (13):

- unknown, classical, gothic, renaissance, baroque, haussmann
- modern, industrial, vernacular, art_deco, brutalist, glass_steel, fortress

**When to Use**:

- Heritage building documentation
- Detailed architectural analysis
- Building Information Modeling (BIM)
- Urban planning with detailed features
- Research requiring fine-grained classification

**Example**:

```yaml
# heritage_site_config.yaml
preset: lod3

input_dir: /data/heritage_tiles
output_dir: /data/detailed_output

features:
  include_architectural_style: true # Enable style detection
```

---

### 4. 📐 ASPRS - Standard Classification

**File**: `ign_lidar/configs/presets/asprs.yaml`

**Use Case**: ASPRS LAS 1.4 compliant classification

**Key Settings**:

- ASPRS classification scheme
- Full geometric + spectral features
- Ground truth from BD TOPO
- Geometric reclassification refinement
- Statistical + radius outlier removal
- Enriched LAZ output

**Speed**: 🚀 **FAST** (1.5× slower than minimal)

**Classes**: Standard ASPRS LAS 1.4

- Ground
- Low/Medium/High Vegetation
- Building
- Water
- Road Surface
- Bridge Deck
- etc.

**When to Use**:

- Standard LiDAR classification workflows
- GIS integration requiring ASPRS codes
- Compatibility with existing tools/pipelines
- Simple land cover classification

**Example**:

```bash
ign-lidar-hd process --preset asprs input/ output/
```

---

### 5. 🔬 Full - Maximum Detail

**File**: `ign_lidar/configs/presets/full.yaml`

**Use Case**: Research, training data generation, maximum quality

**Key Settings**:

- LOD3 classification (30 classes)
- Full geometric + spectral features
- Architectural style detection (13 styles)
- Ground truth from BD TOPO
- Geometric reclassification refinement
- Statistical + radius outlier removal
- Both enriched LAZ + training patches (NPZ)

**Speed**: 🐢 **SLOW** (4× slower than minimal)

**Output Modes**:

- Enriched LAZ tiles (full resolution)
- Training patches (NPZ format)
- Statistics and metadata
- Multi-format export (LAZ, NPZ, PLY, LAS)

**When to Use**:

- Machine learning training data generation
- Research requiring all features
- Maximum quality output needed
- Heritage site comprehensive documentation

**Example**:

```yaml
# ml_training_config.yaml
preset: full

input_dir: /data/diverse_tiles
output_dir: /data/training_patches

processor:
  patch_size: 100.0 # 100m patches
  augment: true # Enable augmentation
  num_augmentations: 3 # 3× data augmentation

output:
  processing_mode: patches_only # Only patches, no tiles
  format: npz,laz # Both formats
```

---

## Creating Custom Configs

### Minimal Override Pattern

**Best Practice**: Inherit a preset and override **only** what's different

```yaml
# my_custom_config.yaml

# Choose the preset that's closest to your use case
preset: lod2

# Override paths (always required)
input_dir: /path/to/your/tiles
output_dir: /path/to/your/output

# Override specific settings (optional)
processor:
  num_workers: 8 # More workers

features:
  k_neighbors: 50 # More neighbors for smoother features

ground_truth:
  cache_dir: /path/to/cache # Custom cache location

# All other settings inherited from lod2 preset!
```

### Multi-Level Overrides

Overrides are applied in order:

1. base.yaml
2. preset (e.g., lod2.yaml)
3. your config file
4. CLI arguments

**Example**:

```yaml
# my_config.yaml
preset: lod2
processor:
  num_workers: 4
```

```bash
# CLI override takes precedence
ign-lidar-hd process \
  --config my_config.yaml \
  --override processor.num_workers=8
```

Final value: `num_workers = 8` (from CLI)

### Deep Merge Behavior

Nested dictionaries are **deep merged**, not replaced:

```yaml
# lod2.yaml has:
processor:
  lod_level: "LOD2"
  use_gpu: true
  num_workers: 4

# Your config:
preset: lod2
processor:
  num_workers: 8

# Result (deep merged):
processor:
  lod_level: "LOD2"      # from lod2.yaml
  use_gpu: true          # from lod2.yaml
  num_workers: 8         # from your config
```

---

## CLI Usage

### Process Command with Presets

```bash
# Use preset directly
ign-lidar-hd process --preset lod2 input/ output/

# Use preset with overrides
ign-lidar-hd process \
  --preset lod2 \
  --input-dir input/ \
  --output-dir output/ \
  --override processor.num_workers=8 \
  --override features.k_neighbors=50

# Use custom config (which uses a preset)
ign-lidar-hd process --config my_config.yaml

# Use custom config with CLI overrides
ign-lidar-hd process \
  --config my_config.yaml \
  --override processor.use_gpu=false
```

### Presets Command

```bash
# List all presets
ign-lidar-hd presets

# Show preset details
ign-lidar-hd presets --preset lod2

# Show preset details with full config
ign-lidar-hd presets --preset lod2 --verbose
```

### Override Syntax

Use dot notation for nested keys:

```bash
--override processor.num_workers=8
--override features.k_neighbors=50
--override ground_truth.enabled=false
--override output.format=laz,npz
```

Multiple overrides:

```bash
ign-lidar-hd process \
  --preset lod3 \
  --input-dir input/ \
  --output-dir output/ \
  --override processor.num_workers=8 \
  --override processor.patch_size=100.0 \
  --override features.include_architectural_style=true \
  --override output.processing_mode=patches_only
```

---

## Configuration Reference

### Key Configuration Sections

#### 1. Processor

Controls the main processing pipeline:

```yaml
processor:
  lod_level: "ASPRS" # ASPRS, LOD2, or LOD3
  processing_mode: "enriched_only" # enriched_only, patches_only, or both
  use_gpu: true # Enable GPU acceleration
  num_workers: 4 # Parallel workers

  # Patch extraction (for training)
  patch_size: 150.0 # Patch size in meters (null = no patching)
  patch_overlap: 0.1 # Overlap fraction (0.0-0.5)
  num_points: 32768 # Points per patch

  # Data augmentation
  augment: false # Enable augmentation
  num_augmentations: 0 # Number of augmented versions

  # GPU settings
  gpu_batch_size: 1000000 # Points per GPU batch

  # Ground truth optimization
  use_optimized_ground_truth: true # Use fast GroundTruthOptimizer

  # Reclassification refinement
  reclassification:
    enabled: true # Geometric refinement pass
    acceleration_mode: "auto" # auto, gpu, cpu
    use_geometric_rules: true # Use planarity, height for refinement
```

#### 2. Features

Controls feature computation:

```yaml
features:
  mode: "lod2" # minimal, lod2, lod3, asprs, full

  # Geometric features
  k_neighbors: 30 # Neighbors for geometric features
  search_radius: 1.5 # Search radius in meters

  # Spectral features
  use_rgb: true # RGB colors
  use_nir: true # Near-infrared
  compute_ndvi: true # Vegetation index

  # Architectural features
  include_architectural_style: false # Style detection (13 styles)
  style_encoding: "constant" # constant or onehot

  # GPU acceleration
  gpu_batch_size: 1000000 # Points per GPU batch
  use_gpu_chunked: true # Chunked GPU processing
```

#### 3. Preprocessing

Outlier removal and cleaning:

```yaml
preprocess:
  enabled: true

  # Statistical Outlier Removal
  sor_k: 12 # Neighbors
  sor_std: 2.0 # Std dev threshold

  # Radius Outlier Removal
  ror_radius: 1.0 # Search radius (m)
  ror_neighbors: 4 # Min neighbors

  # Voxel downsampling
  voxel_enabled: false # Enable downsampling
  voxel_size: 0.1 # Voxel size (m)
```

#### 4. Ground Truth

BD TOPO classification integration:

```yaml
ground_truth:
  enabled: true
  update_classification: false # Modify input files

  # Feature selection
  include_buildings: true
  include_roads: true
  include_water: true
  include_vegetation: true

  # NDVI refinement
  use_ndvi: true
  ndvi_vegetation_threshold: 0.3
  ndvi_building_threshold: 0.15

  # Caching
  cache_dir: ".cache/ground_truth"
  cache_enabled: true
```

#### 5. Output

Output configuration:

```yaml
output:
  format: "laz" # laz, npz, ply, las (comma-separated)
  processing_mode: "enriched_only" # enriched_only, patches_only, both

  # Metadata
  save_stats: true # Save statistics
  save_metadata: true # Save metadata
  compression: null # Compression level
```

---

## Migration Guide

### From V5.0 to V5.1

**Before (V5.0)**: Verbose 188-line config

```yaml
# config_old.yaml (188 lines)
defaults:
  - ../ign_lidar/configs/config
  - _self_

input_dir: /data/tiles
output_dir: /data/output

processor:
  lod_level: LOD2
  architecture: hybrid
  use_gpu: true
  num_workers: 4
  patch_size: 150.0
  patch_overlap: 0.1
  # ... 160 more lines
```

**After (V5.1)**: Preset-based 43-line config

```yaml
# config_new.yaml (43 lines)
preset: lod2

input_dir: /data/tiles
output_dir: /data/output
# Everything else inherited from lod2 preset!
```

**Reduction**: 188 → 43 lines (**-77%**)

### Migration Steps

1. **Identify your use case**

   - Building modeling → `lod2`
   - Detailed architecture → `lod3`
   - Standard classification → `asprs`
   - Quick preview → `minimal`
   - Maximum detail → `full`

2. **Create new config with preset**

   ```yaml
   preset: [chosen_preset]

   input_dir: [your_path]
   output_dir: [your_path]
   ```

3. **Add only custom overrides**

   - Compare your old config with the preset
   - Add only settings that differ
   - Remove everything that matches the preset

4. **Test the new config**

   ```bash
   ign-lidar-hd process --config config_new.yaml --dry-run
   ```

5. **Validate output**
   - Check that settings match your expectations
   - Use `ign-lidar-hd presets --preset [name]` to see defaults

### Common Migration Patterns

**Pattern 1: Simple path override**

```yaml
# Old: 188 lines
# New: Just specify preset + paths
preset: lod2
input_dir: /data/tiles
output_dir: /data/output
```

**Pattern 2: Custom worker count**

```yaml
preset: lod2
input_dir: /data/tiles
output_dir: /data/output

processor:
  num_workers: 8 # Override default
```

**Pattern 3: Analysis mode (no patches)**

```yaml
preset: lod3
input_dir: /data/heritage
output_dir: /data/analyzed

processor:
  patch_size: null # Disable patching

output:
  processing_mode: tiles_only # Only enriched tiles
```

**Pattern 4: Training data generation**

```yaml
preset: full
input_dir: /data/diverse_tiles
output_dir: /data/training

processor:
  patch_size: 100.0
  augment: true
  num_augmentations: 3

output:
  processing_mode: patches_only
  format: npz,laz
```

---

## Troubleshooting

### Issue: Config not loading

**Symptom**: `FileNotFoundError` or `KeyError`

**Solution**:

```bash
# Check config file exists
ls -l my_config.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('my_config.yaml'))"

# Test with PresetConfigLoader
python -c "
from ign_lidar.config.preset_loader import PresetConfigLoader
from pathlib import Path
loader = PresetConfigLoader()
config = loader.load(config_file=Path('my_config.yaml'))
print('✅ Config loaded successfully!')
"
```

### Issue: Preset not found

**Symptom**: `Preset 'xyz' not found`

**Solution**:

```bash
# List available presets
ign-lidar-hd presets

# Check preset name spelling (case-sensitive)
# Valid: minimal, lod2, lod3, asprs, full
```

### Issue: Override not working

**Symptom**: CLI override not applied

**Solution**:

```bash
# Check dot notation syntax
--override processor.num_workers=8  # ✅ Correct
--override processor num_workers 8  # ❌ Wrong

# Check key path matches config structure
# Use ign-lidar-hd presets --preset [name] to see structure
```

### Issue: GPU not detected

**Symptom**: `use_gpu: true` but CPU is used

**Solution**:

```bash
# Check GPU libraries
python -c "import cupy; print('✅ CuPy available')"
python -c "from cuml import DBSCAN; print('✅ RAPIDS cuML available')"

# Force CPU if needed
ign-lidar-hd process --preset lod2 input/ output/ \
  --override processor.use_gpu=false
```

### Issue: Out of memory

**Symptom**: `OutOfMemoryError` or crashes

**Solution**:

```yaml
# Reduce batch sizes
preset: lod2

processor:
  gpu_batch_size: 500000  # Reduce from 1M

features:
  gpu_batch_size: 500000  # Reduce from 1M

# Or reduce workers
processor:
  num_workers: 2  # Reduce from 4
```

### Issue: Slow processing

**Symptom**: Processing takes too long

**Solution**:

```bash
# Use faster preset
ign-lidar-hd process --preset minimal input/ output/  # Fastest

# Or disable expensive features
--override features.compute_ndvi=false
--override ground_truth.enabled=false
--override preprocess.enabled=false
```

### Issue: Unknown classification scheme

**Symptom**: `Unknown LOD level: XYZ`

**Solution**:

```yaml
# Valid values: ASPRS, LOD2, LOD3
processor:
  lod_level: "LOD2" # Use quotes, case-sensitive
```

### Debug Mode

Enable verbose logging:

```bash
ign-lidar-hd process --config my_config.yaml --verbose

# Or in config:
log_level: DEBUG
verbose: true
```

---

## Best Practices

### 1. Start with Presets

Don't create configs from scratch. Choose the preset closest to your use case:

```yaml
# ✅ Good
preset: lod2
input_dir: /data
output_dir: /output

# ❌ Bad (reinventing the wheel)
processor:
  lod_level: LOD2
  use_gpu: true
  # ... 100+ lines of settings
```

### 2. Override Minimally

Only specify what's different:

```yaml
# ✅ Good (only 3 overrides)
preset: lod2
input_dir: /data
output_dir: /output
processor:
  num_workers: 8

# ❌ Bad (unnecessary overrides)
preset: lod2
input_dir: /data
output_dir: /output
processor:
  lod_level: LOD2      # Already in lod2 preset
  use_gpu: true        # Already in lod2 preset
  num_workers: 8
  patch_size: 150.0    # Already in lod2 preset
```

### 3. Document Your Overrides

Add comments explaining why you override:

```yaml
preset: lod3

input_dir: /data/heritage_site
output_dir: /data/processed

processor:
  num_workers: 8 # More workers for faster processing
  patch_size: null # No patching - keep full tiles for analysis

features:
  k_neighbors: 50 # More neighbors for smoother features on detailed architecture
```

### 4. Use Presets Command

Explore presets before choosing:

```bash
# List all presets
ign-lidar-hd presets

# View preset details
ign-lidar-hd presets --preset lod2

# Compare presets
ign-lidar-hd presets --preset lod2 > lod2.txt
ign-lidar-hd presets --preset lod3 > lod3.txt
diff lod2.txt lod3.txt
```

### 5. Version Your Configs

Use descriptive filenames:

```
config_paris_lod2_v5.1.yaml         # ✅ Good
config_versailles_heritage_lod3_v5.1.yaml  # ✅ Good
config.yaml                          # ❌ Bad (not descriptive)
```

---

## Additional Resources

- **Preset Files**: `ign_lidar/configs/presets/`
- **Base Config**: `ign_lidar/configs/base.yaml`
- **Examples**: `examples/config_*_v5.1.yaml`
- **Source Code**: `ign_lidar/config/preset_loader.py`
- **Tests**: `tests/test_preset_config_loader.py`
- **CLI Commands**: `ign_lidar/cli/commands/presets.py`

---

## Feedback & Contributions

Found an issue? Have a suggestion?

- **Issues**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Discussions**: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions

Want to add a new preset?

1. Create preset file in `ign_lidar/configs/presets/`
2. Add documentation header with use case, speed, classes
3. Test with `PresetConfigLoader`
4. Submit pull request

---

**Last Updated**: October 17, 2025  
**Version**: 5.1.0  
**Maintainer**: Simon Ducournau
