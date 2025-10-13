# Migration Guide: v2.4 → v2.5

**Target Audience:** Users upgrading from v2.4.x to v2.5.0+  
**Date:** October 13, 2025  
**Breaking Changes:** Yes (configuration system)

---

## 📋 Overview

Version 2.5.0 introduces significant improvements to the configuration system and internal architecture. This guide helps you migrate your existing code and configurations.

### What's Changing?

1. **Configuration System**: `PipelineConfig` → Hydra/OmegaConf
2. **CLI Interface**: Unified Hydra-based CLI
3. **Feature API**: New factory pattern with unified interface
4. **Internal Architecture**: Modular processor design

---

## 🚨 Breaking Changes

### 1. PipelineConfig Deprecated

**Status**: ⚠️ Deprecated in v2.4.4, removed in v2.5.0

#### Old Way (v2.4.x)

```python
from ign_lidar.core.pipeline_config import PipelineConfig

config = PipelineConfig("config.yaml")
processor = LiDARProcessor(
    input_dir=config.get('input_dir'),
    output_dir=config.get('output_dir'),
    # ... many parameters ...
)
```

#### New Way (v2.5.0+)

```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ign_lidar.core.processor import LiDARProcessor

# Option 1: Load from config directory
with initialize_config_dir(config_dir="../configs"):
    cfg = compose(config_name="config")

# Option 2: Load from file
cfg = OmegaConf.load("config.yaml")

# Create processor with config object
processor = LiDARProcessor(config=cfg)
```

### 2. CLI Interface

**Status**: ⚠️ Unified in v2.4.4+ (hydra_main deprecated in v2.5.0)

#### Old Way (v2.4.x - Multiple Entry Points)

```bash
# Entry Point 1: Click CLI with individual options
ign-lidar-hd process --config-file config.yaml --input data/ --output output/

# Entry Point 2: Pure Hydra (DEPRECATED)
python -m ign_lidar.cli.hydra_main input_dir=data/ output_dir=output/

# Entry Point 3: Hybrid (confusing!)
ign-lidar-hd process --config-path configs --config-name config \
  input_dir=data/ output_dir=output/
```

#### New Way (v2.4.4+)

```bash
# Single unified interface with Hydra overrides
ign-lidar-hd process input_dir=data/ output_dir=output/

# Use experiment presets
ign-lidar-hd process experiment=buildings_lod2 input_dir=data/ output_dir=output/

# With custom config file
ign-lidar-hd process --config-file config.yaml

# Config file + overrides (overrides have highest priority)
ign-lidar-hd process --config-file config.yaml \
  processor.use_gpu=true \
  features.k_neighbors=30

# Preview config without processing
ign-lidar-hd process --config-file config.yaml --show-config
```

**Key Changes:**

- ✅ Single command: `ign-lidar-hd process`
- ✅ All Hydra features available (overrides, experiments, composition)
- ✅ Config files supported with `--config-file` or `-c`
- ✅ Overrides always have highest priority
- ❌ No more `python -m ign_lidar.cli.hydra_main` (use `ign-lidar-hd process` instead)

**Migration:**

```bash
# Simply replace the module call with the main command
# OLD: python -m ign_lidar.cli.hydra_main input_dir=data/ output_dir=output/
# NEW: ign-lidar-hd process input_dir=data/ output_dir=output/
```

### 3. Feature Computation API

**Status**: ⚠️ New interface in v2.5.0

#### Old Way (v2.4.x)

```python
from ign_lidar.features import compute_all_features_optimized
from ign_lidar.features_gpu import compute_all_features_with_gpu

# Direct function calls
if use_gpu:
    features = compute_all_features_with_gpu(xyz, k=20)
else:
    features = compute_all_features_optimized(xyz, k=20)
```

#### New Way (v2.5.0+)

```python
from ign_lidar.features import FeatureComputerFactory

# Use factory pattern
computer = FeatureComputerFactory.create(
    backend='auto',  # or 'cpu', 'gpu', 'gpu_chunked'
    use_gpu=True
)
features = computer.compute_features(xyz, k_neighbors=20)
```

---

## 📝 Step-by-Step Migration

### Step 1: Update Configuration Files

Convert your YAML configs to Hydra format.

#### Example: Basic Config

**Old Format** (`config_old.yaml`):

```yaml
input_dir: "data/tiles/"
output_dir: "output/patches/"
use_gpu: true
k_neighbors: 20
patch_size: 100.0
num_points: 32768
processing_mode: "patches_only"
```

**New Format** (`config_new.yaml`):

```yaml
# Hydra format with structured configs
defaults:
  - _self_
  - processor: default
  - features: full
  - output: patches

# Paths
input_dir: "data/tiles/"
output_dir: "output/patches/"

# Processing settings
processor:
  use_gpu: true
  patch_size: 100.0
  num_points: 32768
  processing_mode: "patches_only"

# Feature settings
features:
  k_neighbors: 20
  mode: "full"
```

**Tool to Convert:**

```bash
# Use provided conversion script
python scripts/convert_config_to_hydra.py config_old.yaml config_new.yaml
```

---

### Step 2: Update Python Code

#### Scenario A: Using PipelineConfig

**Before:**

```python
from ign_lidar.core.pipeline_config import PipelineConfig
from ign_lidar.core.processor import LiDARProcessor

config = PipelineConfig("config.yaml")
processor = LiDARProcessor(
    input_dir=config.config['input_dir'],
    output_dir=config.config['output_dir'],
    use_gpu=config.config.get('use_gpu', False),
    # ... 40+ parameters ...
)
processor.process_directory()
```

**After:**

```python
from omegaconf import OmegaConf
from ign_lidar.core.processor import LiDARProcessor

# Load config
cfg = OmegaConf.load("config.yaml")

# Create processor with single config object
processor = LiDARProcessor(config=cfg)
processor.process_directory(cfg.input_dir, cfg.output_dir)
```

#### Scenario B: Direct Processor Instantiation

**Before:**

```python
processor = LiDARProcessor(
    input_dir="data/",
    output_dir="output/",
    patch_size=100.0,
    num_points=32768,
    use_gpu=True,
    k_neighbors=20,
    # ... 35 more parameters ...
)
```

**After:**

```python
from omegaconf import OmegaConf

# Create config programmatically
cfg = OmegaConf.create({
    'input_dir': 'data/',
    'output_dir': 'output/',
    'processor': {
        'patch_size': 100.0,
        'num_points': 32768,
        'use_gpu': True,
    },
    'features': {
        'k_neighbors': 20,
        'mode': 'full'
    }
})

processor = LiDARProcessor(config=cfg)
processor.process_directory(cfg.input_dir, cfg.output_dir)
```

---

### Step 3: Update CLI Commands

#### Download Command

**Before:**

```bash
ign-lidar-hd download --bbox 2.3,48.8,2.4,48.9 --output data/
```

**After:**

```bash
# Same syntax (backwards compatible)
ign-lidar-hd download bbox=[2.3,48.8,2.4,48.9] output_dir=data/
```

#### Process Command

**Before:**

```bash
ign-lidar-hd process \
  --config-file config.yaml \
  --input-dir data/ \
  --output-dir output/ \
  --use-gpu
```

**After:**

```bash
ign-lidar-hd process \
  --config-file config.yaml \
  input_dir=data/ \
  output_dir=output/ \
  processor.use_gpu=true
```

#### Verify Command

**Before:**

```bash
ign-lidar-hd verify --output-dir output/
```

**After:**

```bash
ign-lidar-hd verify output_dir=output/
```

---

### Step 4: Update Feature Computation

#### Direct Feature Usage

**Before:**

```python
from ign_lidar.features import compute_all_features_optimized

xyz = np.random.rand(10000, 3)
features = compute_all_features_optimized(xyz, k=20)
```

**After:**

```python
from ign_lidar.features import FeatureComputerFactory

xyz = np.random.rand(10000, 3)
computer = FeatureComputerFactory.create(backend='cpu')
features = computer.compute_features(xyz, k_neighbors=20)
```

#### GPU Features

**Before:**

```python
from ign_lidar.features_gpu import compute_all_features_with_gpu

features = compute_all_features_with_gpu(xyz, k=20)
```

**After:**

```python
from ign_lidar.features import FeatureComputerFactory

computer = FeatureComputerFactory.create(backend='gpu', use_gpu=True)
features = computer.compute_features(xyz, k_neighbors=20)
```

---

## 🔄 Automated Migration Tools

### Config Converter Script

```bash
# Convert old config to new format
python scripts/convert_config_to_hydra.py \
  examples/config_old.yaml \
  examples/config_new.yaml
```

### Batch Conversion

```bash
# Convert all configs in a directory
for config in configs/*.yaml; do
  python scripts/convert_config_to_hydra.py "$config" "configs_new/$(basename $config)"
done
```

---

## 🧪 Testing Your Migration

### 1. Verify Config Loading

```python
from omegaconf import OmegaConf

# Load your migrated config
cfg = OmegaConf.load("config_new.yaml")

# Check structure
print(OmegaConf.to_yaml(cfg))

# Access values
print(f"Input dir: {cfg.input_dir}")
print(f"GPU enabled: {cfg.processor.use_gpu}")
```

### 2. Test Processing

```bash
# Run with single tile for quick test
ign-lidar-hd process \
  --config-file config_new.yaml \
  processor.max_tiles=1
```

### 3. Compare Outputs

```bash
# Process with old version
ign-lidar-hd-old process --config-file config_old.yaml

# Process with new version
ign-lidar-hd process --config-file config_new.yaml

# Compare results
python scripts/compare_outputs.py output_old/ output_new/
```

---

## 📚 Configuration Examples

### Example 1: Minimal Config

```yaml
# config_minimal.yaml
input_dir: "data/"
output_dir: "output/"

processor:
  patch_size: 50.0
  num_points: 16384
```

### Example 2: GPU Config

```yaml
# config_gpu.yaml
defaults:
  - processor: gpu
  - features: full

input_dir: "data/"
output_dir: "output/"

processor:
  use_gpu: true
  use_chunked_processing: true
  chunk_size: 100000
```

### Example 3: Memory Optimized

```yaml
# config_memory.yaml
defaults:
  - processor: sequential
  - features: simplified

input_dir: "data/"
output_dir: "output/"

processor:
  max_concurrent_tiles: 1
  patch_size: 50.0
  num_points: 16384

features:
  k_neighbors: 15
  mode: "simplified"
```

---

## 🐛 Troubleshooting

### Issue 1: ImportError for PipelineConfig

**Error:**

```python
ImportError: cannot import name 'PipelineConfig' from 'ign_lidar.core.pipeline_config'
```

**Solution:**
PipelineConfig was removed in v2.5.0. Use Hydra configuration:

```python
from omegaconf import OmegaConf
cfg = OmegaConf.load("config.yaml")
```

---

### Issue 2: Configuration Validation Errors

**Error:**

```
omegaconf.errors.ValidationError: Value 'invalid' is not a valid choice
```

**Solution:**
Check your config against the schema:

```python
from ign_lidar.config.schema import IGNLiDARConfig
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")
# This will validate and show specific errors
validated_cfg = OmegaConf.to_object(cfg, IGNLiDARConfig)
```

---

### Issue 3: Feature Names Changed

**Error:**

```python
KeyError: 'eigenvalue_1'
```

**Solution:**
Feature names are now standardized. Check available features:

```python
computer = FeatureComputerFactory.create(backend='cpu')
print(computer.available_features)
```

---

### Issue 4: CLI Parameter Format

**Error:**

```bash
Error: no such option: --input-dir
```

**Solution:**
Use Hydra syntax without dashes:

```bash
# Old: --input-dir
# New: input_dir=
ign-lidar-hd process input_dir=data/
```

---

## 📖 Additional Resources

### Documentation

- **Main README**: Overview and quick start
- **API Reference**: Detailed API documentation
- **Examples Guide**: `examples/EXAMPLES_README_v2.md`
- **Architecture**: `CODEBASE_ANALYSIS_CONSOLIDATION.md`

### Configuration Templates

- **Basic**: `examples/config_complete.yaml`
- **GPU**: `examples/config_gpu_processing.yaml`
- **Training**: `examples/config_lod3_training.yaml`
- **Memory**: `examples/config_lod3_training_memory_optimized.yaml`

### Scripts

- **Config Converter**: `scripts/convert_config_to_hydra.py`
- **Output Comparison**: `scripts/compare_outputs.py`
- **Feature Checker**: `scripts/check_features.py`

---

## 🎯 Migration Checklist

Use this checklist to track your migration progress:

### Configuration

- [ ] Convert YAML configs to Hydra format
- [ ] Test config loading with OmegaConf
- [ ] Validate config structure
- [ ] Update config file paths in scripts

### Code

- [ ] Replace `PipelineConfig` with `OmegaConf`
- [ ] Update `LiDARProcessor` instantiation
- [ ] Migrate feature computation to factory pattern
- [ ] Update import statements

### CLI

- [ ] Update process commands
- [ ] Update download commands
- [ ] Update verify commands
- [ ] Test all CLI workflows

### Testing

- [ ] Run unit tests
- [ ] Run integration tests
- [ ] Compare outputs with v2.4.x
- [ ] Verify feature counts

### Documentation

- [ ] Update README files
- [ ] Update code comments
- [ ] Update example scripts
- [ ] Document custom changes

---

## 💡 Best Practices

### 1. Use Config Files

Prefer config files over programmatic configuration:

```python
# Good: Config file
cfg = OmegaConf.load("config.yaml")

# Avoid: Hardcoded values
cfg = OmegaConf.create({'input_dir': 'data/', ...})
```

### 2. Override Parameters Sparingly

Override only what's necessary:

```bash
# Good: Override specific values
ign-lidar-hd process --config-file config.yaml processor.use_gpu=true

# Avoid: Override everything
ign-lidar-hd process input_dir=... output_dir=... processor.patch_size=... # too many
```

### 3. Validate Configs Early

Catch errors before processing:

```python
from ign_lidar.config.schema import IGNLiDARConfig
from omegaconf import OmegaConf

cfg = OmegaConf.load("config.yaml")
validated = OmegaConf.to_object(cfg, IGNLiDARConfig)  # Raises error if invalid
```

---

## 🆘 Getting Help

### Questions?

- **GitHub Issues**: Report bugs or ask questions
- **Documentation**: Check the docs/ directory
- **Examples**: See examples/ for working configs

### Found a Bug?

1. Check if it's documented in CHANGELOG.md
2. Search existing GitHub issues
3. Create new issue with reproduction steps

---

**Version:** 2.5.0  
**Last Updated:** October 13, 2025  
**Feedback:** Please report issues or suggestions on GitHub
