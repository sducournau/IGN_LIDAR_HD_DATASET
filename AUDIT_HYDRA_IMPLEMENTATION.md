# 🔍 Project Audit & Hydra Implementation Plan

**Date**: October 7, 2025  
**Version**: v2.0.0-alpha  
**Status**: Architecture Modernization

---

## 📊 Current State Analysis

### ✅ Strengths

1. **Modular Architecture**

   - Clear separation of concerns (`processor.py`, `features.py`, `utils.py`)
   - GPU/CPU dual implementation for features
   - Tile stitching already implemented (`tile_stitcher.py`)
   - Multi-architecture support (`datasets/`, `formatters/`)

2. **Advanced Features**

   - RGB augmentation from IGN orthophotos
   - Infrared + NDVI support
   - Architectural style encoding
   - Memory-aware processing
   - GPU chunking for large files

3. **Well-Documented**
   - Comprehensive docstrings
   - Multiple documentation files
   - Configuration examples

### ⚠️ Issues & Technical Debt

1. **Configuration Management** ❌

   - Mixed config systems: `config.py` (dict) + `pipeline_config.py` (YAML)
   - Hardcoded defaults scattered across files
   - No environment-specific configs
   - No config validation framework
   - No override mechanism

2. **CLI Architecture** ❌

   - Using `argparse` (verbose, limited)
   - 1863 lines in `cli.py` (too large)
   - Mixed concerns (CLI logic + business logic)
   - No command composition
   - Difficult to test

3. **Pipeline Flow** ⚠️

   - Legacy dual-command workflow (enrich → patch)
   - New unified workflow exists but not fully integrated
   - Missing clear entrypoint for v2.0 architecture

4. **Testing** ⚠️

   - Test files exist but incomplete coverage
   - No integration tests for Hydra configs
   - No config validation tests

5. **Dependency Management** ⚠️
   - Using both `setup.py` and `pyproject.toml`
   - GPU dependencies not clearly separated
   - Missing Hydra in dependencies

---

## 🎯 Hydra Implementation Strategy

### Why Hydra?

- **Hierarchical Configs**: Compose configs from multiple files
- **Override System**: Command-line overrides with dot notation
- **Multi-run**: Easy parameter sweeps for experiments
- **Type Safety**: Structured configs with dataclasses
- **Validation**: Automatic config validation
- **Extensibility**: Plugin system for custom resolvers

### Architecture Changes

```
Current:
┌─────────────────────────────────────────────┐
│ cli.py (argparse) → config.py (dict)       │
│           ↓                                 │
│     processor.py                            │
└─────────────────────────────────────────────┘

New (Hydra):
┌─────────────────────────────────────────────┐
│ configs/                                    │
│   ├── config.yaml (base)                    │
│   ├── processor/                            │
│   │   ├── default.yaml                      │
│   │   ├── gpu.yaml                          │
│   │   └── cpu.yaml                          │
│   ├── features/                             │
│   │   ├── minimal.yaml                      │
│   │   ├── full.yaml                         │
│   │   └── pointnet.yaml                     │
│   └── experiment/                           │
│       ├── buildings.yaml                    │
│       └── vegetation.yaml                   │
│                                             │
│ ign_lidar/                                 │
│   ├── config/                               │
│   │   ├── __init__.py                       │
│   │   ├── schema.py (dataclasses)           │
│   │   └── defaults.py                       │
│   └── cli/                                  │
│       ├── __init__.py                       │
│       ├── main.py (Hydra app)               │
│       ├── process.py                        │
│       ├── enrich.py                         │
│       └── download.py                       │
└─────────────────────────────────────────────┘
```

---

## 📋 Implementation Roadmap

### Phase 1: Setup & Core Infrastructure (Week 1)

#### 1.1 Add Hydra Dependencies

```toml
# pyproject.toml
dependencies = [
    ...
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
]
```

#### 1.2 Create Config Schema

```python
# ign_lidar/config/schema.py
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import MISSING

@dataclass
class ProcessorConfig:
    """Processor configuration."""
    lod_level: str = "LOD2"
    use_gpu: bool = False
    num_workers: int = 4
    patch_size: float = 150.0
    patch_overlap: float = 0.1
    num_points: int = 16384

@dataclass
class FeaturesConfig:
    """Feature computation configuration."""
    mode: str = "full"
    k_neighbors: int = 20
    include_extra: bool = False
    use_rgb: bool = False
    use_infrared: bool = False

@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""
    enabled: bool = False
    sor_k: int = 12
    sor_std: float = 2.0
    ror_radius: float = 1.0
    ror_neighbors: int = 4

@dataclass
class StitchingConfig:
    """Tile stitching configuration."""
    enabled: bool = False
    buffer_size: float = 10.0
    auto_detect_neighbors: bool = True

@dataclass
class IGNLiDARConfig:
    """Root configuration."""
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)

    # I/O
    input_dir: str = MISSING
    output_dir: str = MISSING

    # Hydra
    defaults: List[str] = field(default_factory=lambda: ["_self_"])
```

#### 1.3 Create Base Config Files

```yaml
# configs/config.yaml
defaults:
  - processor: default
  - features: full
  - preprocess: default
  - stitching: default
  - _self_

# Paths (required)
input_dir: ??? # Must be provided
output_dir: ??? # Must be provided

# Hydra configuration
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

```yaml
# configs/processor/default.yaml
lod_level: LOD2
use_gpu: false
num_workers: 4
patch_size: 150.0
patch_overlap: 0.1
num_points: 16384
augment: false
num_augmentations: 3
```

```yaml
# configs/processor/gpu.yaml
# @package _global_
defaults:
  - default

use_gpu: true
num_workers: 8 # More workers for GPU
```

```yaml
# configs/features/minimal.yaml
mode: minimal
k_neighbors: 10
include_extra: false
use_rgb: false
use_infrared: false
```

```yaml
# configs/features/full.yaml
mode: full
k_neighbors: 20
include_extra: true
use_rgb: true
use_infrared: true
```

```yaml
# configs/features/pointnet.yaml
# Optimized for PointNet++
mode: full
k_neighbors: 10
include_extra: true
use_rgb: true
use_infrared: false
sampling_method: fps
normalize_xyz: true
normalize_features: true
```

### Phase 2: Refactor CLI (Week 2)

#### 2.1 Create Hydra-based CLI

```python
# ign_lidar/cli/main.py
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging

from ..processor import LiDARProcessor
from ..config.schema import IGNLiDARConfig

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def process(cfg: DictConfig) -> None:
    """
    Process LiDAR tiles with Hydra configuration.

    Examples:
        # Basic usage
        python -m ign_lidar.cli.main input_dir=data/raw output_dir=data/patches

        # GPU processing
        python -m ign_lidar.cli.main processor=gpu input_dir=data/raw output_dir=data/patches

        # PointNet++ optimized
        python -m ign_lidar.cli.main features=pointnet processor=gpu \\
            input_dir=data/raw output_dir=data/patches

        # Multi-run (parameter sweep)
        python -m ign_lidar.cli.main -m processor.num_points=4096,8192,16384 \\
            input_dir=data/raw output_dir=data/patches
    """
    # Validate config
    schema = OmegaConf.structured(IGNLiDARConfig)
    cfg = OmegaConf.merge(schema, cfg)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Initialize processor
    processor = LiDARProcessor(
        lod_level=cfg.processor.lod_level,
        use_gpu=cfg.processor.use_gpu,
        patch_size=cfg.processor.patch_size,
        patch_overlap=cfg.processor.patch_overlap,
        num_points=cfg.processor.num_points,
        k_neighbors=cfg.features.k_neighbors,
        include_extra_features=cfg.features.include_extra,
        include_rgb=cfg.features.use_rgb,
        preprocess=cfg.preprocess.enabled,
        preprocess_config=OmegaConf.to_container(cfg.preprocess),
        use_stitching=cfg.stitching.enabled,
        buffer_size=cfg.stitching.buffer_size,
    )

    # Process
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    total_patches = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        num_workers=cfg.processor.num_workers
    )

    logger.info(f"✅ Processed {total_patches:,} patches")

if __name__ == "__main__":
    process()
```

#### 2.2 Create Command Modules

```python
# ign_lidar/cli/commands.py
"""Modular commands for IGN LiDAR HD."""
import hydra
from omegaconf import DictConfig
from pathlib import Path

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def download(cfg: DictConfig) -> None:
    """Download LiDAR tiles from IGN."""
    from ..downloader import IGNLiDARDownloader
    # Implementation
    pass

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def enrich(cfg: DictConfig) -> None:
    """Enrich LAZ files with features (legacy command)."""
    # Implementation
    pass

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def verify(cfg: DictConfig) -> None:
    """Verify dataset quality."""
    from ..verification import FeatureVerifier
    # Implementation
    pass
```

### Phase 3: Experiment Configs (Week 3)

#### 3.1 Create Experiment Presets

```yaml
# configs/experiment/buildings_lod2.yaml
# @package _global_
defaults:
  - override /processor: default
  - override /features: minimal
  - override /preprocess: default

processor:
  lod_level: LOD2
  num_points: 8192

features:
  use_rgb: false
  include_extra: true # Building-specific features

preprocess:
  enabled: true
```

```yaml
# configs/experiment/vegetation_ndvi.yaml
# @package _global_
defaults:
  - override /processor: gpu
  - override /features: full

processor:
  num_points: 16384

features:
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
```

```yaml
# configs/experiment/pointnet_training.yaml
# @package _global_
defaults:
  - override /processor: gpu
  - override /features: pointnet
  - override /stitching: enabled

processor:
  num_points: 16384
  augment: true
  num_augmentations: 5

stitching:
  enabled: true
  buffer_size: 10.0
```

### Phase 4: Testing & Documentation (Week 4)

#### 4.1 Config Tests

```python
# tests/test_hydra_config.py
import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from ign_lidar.config.schema import IGNLiDARConfig

def test_default_config():
    """Test default configuration loads."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        assert cfg.processor.lod_level == "LOD2"

def test_gpu_override():
    """Test GPU configuration override."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["processor=gpu"])
        assert cfg.processor.use_gpu is True

def test_experiment_config():
    """Test experiment preset."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["experiment=buildings_lod2"])
        assert cfg.processor.lod_level == "LOD2"
        assert cfg.processor.num_points == 8192
```

#### 4.2 Migration Guide

````markdown
# Migration Guide: v1.7.7 → v2.0.0

## Breaking Changes

### CLI Commands

**Before (v1.7.7):**

```bash
ign-lidar-hd process --input-dir data/raw --output data/patches \\
    --num-points 16384 --use-gpu --add-rgb
```
````

**After (v2.0.0):**

```bash
python -m ign_lidar.cli.main processor=gpu features=full \\
    input_dir=data/raw output_dir=data/patches
```

### Configuration Files

**Before:**

```yaml
# pipeline.yaml
global:
  use_gpu: true

process:
  num_points: 16384
```

**After:**

```yaml
# configs/config.yaml
defaults:
  - processor: gpu
  - features: full
```

````

---

## 🚀 Quick Start (After Implementation)

### Basic Usage
```bash
# Process with defaults
python -m ign_lidar.cli.main input_dir=data/raw output_dir=data/patches

# GPU processing
python -m ign_lidar.cli.main processor=gpu input_dir=data/raw output_dir=data/patches

# PointNet++ training dataset
python -m ign_lidar.cli.main experiment=pointnet_training \\
    input_dir=data/raw output_dir=data/patches
````

### Parameter Overrides

```bash
# Override specific values
python -m ign_lidar.cli.main processor.num_points=32768 \\
    features.k_neighbors=30 input_dir=data/raw output_dir=data/patches

# Multi-run experiment
python -m ign_lidar.cli.main -m processor.num_points=4096,8192,16384 \\
    input_dir=data/raw output_dir=data/patches
```

### Using Config Files

```bash
# Custom config file
python -m ign_lidar.cli.main --config-name my_config \\
    input_dir=data/raw output_dir=data/patches

# Config in different directory
python -m ign_lidar.cli.main --config-path /path/to/configs \\
    --config-name config input_dir=data/raw output_dir=data/patches
```

---

## 📊 Performance Optimizations

### 1. Memory Management

```yaml
# configs/performance/memory_optimized.yaml
processor:
  num_workers: 2 # Reduce for large tiles
  batch_size: auto # Auto-detect based on available RAM

features:
  chunk_size: 10_000_000 # Process in chunks
```

### 2. GPU Optimization

```yaml
# configs/performance/gpu_optimized.yaml
processor:
  use_gpu: true
  num_workers: 8
  prefetch_factor: 4

features:
  gpu_batch_size: 1_000_000
  use_gpu_chunked: true
```

### 3. Speed Priority

```yaml
# configs/performance/fast.yaml
processor:
  num_workers: -1 # Use all CPU cores

features:
  mode: minimal
  k_neighbors: 10

preprocess:
  enabled: false

stitching:
  enabled: false
```

---

## 📈 Benefits Summary

| Aspect                  | Before         | After           | Improvement     |
| ----------------------- | -------------- | --------------- | --------------- |
| **Config Complexity**   | Mixed systems  | Unified Hydra   | ✅ 70% simpler  |
| **CLI Verbosity**       | Long commands  | Compose configs | ✅ 50% shorter  |
| **Experiment Tracking** | Manual logs    | Auto-tracked    | ✅ Built-in     |
| **Reproducibility**     | Difficult      | Easy            | ✅ Config saved |
| **Parameter Sweeps**    | Manual loops   | Multi-run       | ✅ One command  |
| **Type Safety**         | Runtime errors | Compile-time    | ✅ Validated    |
| **Extensibility**       | Hardcoded      | Plugin system   | ✅ Modular      |

---

## 🎯 Next Steps

1. [ ] Implement Phase 1 (Config schema + base configs)
2. [ ] Implement Phase 2 (Hydra CLI)
3. [ ] Implement Phase 3 (Experiment presets)
4. [ ] Implement Phase 4 (Tests + docs)
5. [ ] Migrate existing YAML configs
6. [ ] Update documentation
7. [ ] Create migration script for v1.7.7 users
8. [ ] Release v2.0.0

---

**Maintainer**: @sducournau  
**Status**: Ready for Implementation  
**Estimated Time**: 4 weeks
