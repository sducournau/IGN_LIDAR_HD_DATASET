# Configuration System Harmonization: Implementation Plan v4.0

**Date:** November 28, 2025  
**Version:** 3.1.0 → 4.0.0  
**Status:** Implementation Roadmap  
**Owner:** IGN LiDAR HD Core Team

---

## 🎯 Executive Summary

This document provides a **concrete, actionable implementation plan** to harmonize the IGN LiDAR HD configuration system from the current fragmented state (v3.1/v3.2/v5.1) to a unified v4.0 architecture.

### Key Objectives

1. **Eliminate 3 parallel config systems** → Single unified approach
2. **Remove deprecated code** → Clean up `schema.py` (415 lines)
3. **Standardize naming** → Consistent terminology across Python/YAML
4. **Unify documentation** → Single source of truth
5. **Provide migration tooling** → Zero-friction upgrade path

### Success Metrics

| Metric                    | Current              | Target v4.0           |
| ------------------------- | -------------------- | --------------------- |
| Config approaches         | 3 parallel systems   | 1 unified system      |
| Lines of config code      | ~1,800 lines         | ~1,200 lines (-33%)   |
| Config loading time       | ~200ms               | <50ms (-75%)          |
| Parameter inconsistencies | 12+ naming conflicts | 0 conflicts           |
| Documentation pages       | 5+ overlapping docs  | 1 comprehensive guide |
| User confusion score      | High (3 systems)     | Low (1 system)        |

---

## 📊 Current State Analysis

### Configuration Architecture Inventory

#### 1. Python Configuration Modules (`ign_lidar/config/`)

| File                   | Status        | Lines | Purpose                   | Action      |
| ---------------------- | ------------- | ----- | ------------------------- | ----------- |
| `schema.py`            | ⚠️ DEPRECATED | 415   | Old v3.1 Hydra config     | **DELETE**  |
| `schema_simplified.py` | ⚠️ DEPRECATED | ~300  | Interim simplified config | **DELETE**  |
| `config.py`            | ✅ ACTIVE     | 598   | Modern v3.2+ config       | **ENHANCE** |
| `building_config.py`   | ✅ ACTIVE     | 387   | LOD3 building config      | **KEEP**    |
| `preset_loader.py`     | ✅ ACTIVE     | 466   | Preset loading system     | **KEEP**    |
| `validator.py`         | ✅ ACTIVE     | ~200  | Config validation         | **ENHANCE** |

**Total:** ~2,366 lines → Target: ~1,651 lines (-30%)

#### 2. YAML Configuration Structure (`ign_lidar/configs/`)

```
configs/
├── base.yaml                    # 436 lines - v5.1 structure
├── base/                        # 6 modular files
│   ├── processor.yaml           # Nested under "processor:"
│   ├── features.yaml            # Nested under "features:"
│   ├── data_sources.yaml
│   ├── ground_truth.yaml
│   ├── output.yaml
│   └── monitoring.yaml
├── presets/                     # 7 preset files
│   ├── asprs_classification_gpu.yaml    # Uses "processor.lod_level"
│   ├── asprs_classification_cpu.yaml
│   ├── lod2_buildings.yaml
│   ├── lod3_detailed.yaml
│   ├── fast_preview.yaml
│   ├── minimal_debug.yaml
│   └── high_quality.yaml
├── hardware/                    # 5 hardware profiles
│   ├── gpu_rtx4090_24gb.yaml
│   ├── gpu_rtx4080_16gb.yaml
│   └── ...
└── advanced/                    # 5 specialized configs
```

#### 3. Example Configurations (`examples/`)

| File                                 | Structure   | Status | Action              |
| ------------------------------------ | ----------- | ------ | ------------------- |
| `TEMPLATE_v3.2.yaml`                 | Flat (v3.2) | Active | **Migrate to v4.0** |
| `config_training_fast_50m_v3.2.yaml` | Flat (v3.2) | Active | **Migrate to v4.0** |
| `config_asprs_production.yaml`       | Mixed       | Active | **Migrate to v4.0** |
| `config_multi_scale_adaptive.yaml`   | Mixed       | Active | **Migrate to v4.0** |

### Key Inconsistencies Identified

#### Naming Conflicts

| Concept             | Python `Config`        | YAML `base.yaml`            | v3.2 Example           | **v4.0 Standard**                 |
| ------------------- | ---------------------- | --------------------------- | ---------------------- | --------------------------------- |
| Classification mode | `mode`                 | `processor.lod_level`       | `mode`                 | **`mode`** (top-level)            |
| Feature selection   | `features.feature_set` | `features.mode`             | `features.feature_set` | **`features.mode`**               |
| Processing output   | `processing_mode`      | `processor.processing_mode` | `processing_mode`      | **`processing_mode`** (top-level) |
| GPU usage           | `use_gpu`              | `processor.use_gpu`         | `use_gpu`              | **`use_gpu`** (top-level)         |

#### Structure Conflicts

**v5.1 YAML (nested):**

```yaml
processor:
  lod_level: "LOD2"
  use_gpu: true
  processing_mode: "patches_only"
```

**v3.2 YAML (flat):**

```yaml
mode: "lod2"
use_gpu: true
processing_mode: "patches_only"
```

**v4.0 TARGET (unified flat):**

```yaml
mode: lod2 # Lowercase, top-level
use_gpu: true # Top-level
processing_mode: patches_only
```

---

## 🏗️ v4.0 Unified Architecture

### Design Principles

1. **Flat over Nested:** Top-level parameters for common settings
2. **Consistency:** Same names in Python/YAML/CLI
3. **Simplicity:** Essential params visible, advanced params nested
4. **Backward Compatible:** Migration tools for seamless upgrade
5. **Type Safety:** Python dataclasses + validation

### Proposed Structure

#### Python Configuration (`config.py`)

```python
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List

@dataclass
class FeatureConfig:
    """Feature computation configuration."""
    mode: Literal["minimal", "standard", "full", "custom"] = "standard"
    k_neighbors: int = 30
    search_radius: Optional[float] = None
    use_rgb: bool = False
    use_nir: bool = False
    compute_ndvi: bool = False
    multi_scale: bool = False
    scales: Optional[List[str]] = None

@dataclass
class OptimizationsConfig:
    """Phase 4 optimizations (v3.9+)."""
    enabled: bool = True

    # Async I/O: +12-14% performance
    async_io_enabled: bool = True
    async_workers: int = 2
    tile_cache_size: int = 3

    # Batch processing: +25-30% performance
    batch_processing_enabled: bool = True
    batch_size: int = 4

    # GPU pooling: +8.5% performance
    gpu_pooling_enabled: bool = True
    gpu_pool_max_size_gb: float = 4.0

    print_stats: bool = True

@dataclass
class AdvancedConfig:
    """Advanced options for expert users."""
    preprocessing: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Dict[str, Any]] = None
    classification: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    reclassification: Optional[Dict[str, Any]] = None

@dataclass
class Config:
    """
    IGN LiDAR HD v4.0 Configuration.

    Unified configuration class replacing schema.py and schema_simplified.py.
    Supports YAML loading, presets, and CLI overrides.

    Quick Start:
        >>> # Use preset
        >>> config = Config.preset('asprs_production')
        >>> config.input_dir = '/data/tiles'
        >>>
        >>> # Load from YAML
        >>> config = Config.from_yaml('my_config.yaml')
        >>>
        >>> # Auto-configure
        >>> config = Config.from_environment(
        ...     input_dir='/data/tiles',
        ...     output_dir='/data/output'
        ... )
    """

    # ============================================================================
    # REQUIRED PARAMETERS
    # ============================================================================
    input_dir: str = MISSING
    output_dir: str = MISSING

    # ============================================================================
    # ESSENTIAL PARAMETERS (top-level, commonly modified)
    # ============================================================================
    mode: Literal["asprs", "lod2", "lod3"] = "lod2"
    processing_mode: Literal["patches_only", "both", "enriched_only", "reclassify_only"] = "patches_only"
    use_gpu: bool = False
    num_workers: int = 4

    # Patch configuration
    patch_size: float = 150.0
    num_points: int = 16384
    patch_overlap: float = 0.1

    # Architecture
    architecture: Literal["pointnet++", "hybrid", "octree", "transformer", "sparse_conv", "multi"] = "pointnet++"

    # ============================================================================
    # NESTED CONFIGURATIONS
    # ============================================================================
    features: FeatureConfig = field(default_factory=FeatureConfig)
    optimizations: OptimizationsConfig = field(default_factory=OptimizationsConfig)
    advanced: Optional[AdvancedConfig] = None

    # ============================================================================
    # CLASS METHODS
    # ============================================================================

    @classmethod
    def preset(cls, name: str, **overrides) -> "Config":
        """Load a preset configuration."""
        # Implementation...

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "Config":
        """Load configuration from YAML file."""
        # Implementation...

    @classmethod
    def from_environment(cls, **overrides) -> "Config":
        """Auto-configure from hardware environment."""
        # Implementation...

    @classmethod
    def from_legacy_schema(cls, legacy_config: "IGNLiDARConfig") -> "Config":
        """Convert legacy v3.1 schema config to v4.0."""
        # Migration logic...

    def validate(self) -> List[str]:
        """Validate configuration and return errors."""
        # Validation logic...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Serialization...

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        # Save logic...
```

#### YAML Configuration (`base.yaml` v4.0)

```yaml
# ============================================================================
# IGN LiDAR HD - Base Configuration v4.0
# ============================================================================
config_version: "4.0.0"
config_name: "base"

# ============================================================================
# REQUIRED PARAMETERS
# ============================================================================
input_dir: null # Required
output_dir: null # Required

# ============================================================================
# ESSENTIAL PARAMETERS (top-level, flat structure)
# ============================================================================

# Classification mode: asprs, lod2, lod3
mode: lod2

# Processing output: patches_only, both, enriched_only, reclassify_only
processing_mode: patches_only

# Hardware
use_gpu: false
num_workers: 4

# Patch configuration
patch_size: 150.0
num_points: 16384
patch_overlap: 0.1

# Architecture
architecture: pointnet++

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
features:
  mode: standard # minimal, standard, full, custom
  k_neighbors: 30
  search_radius: null
  use_rgb: false
  use_nir: false
  compute_ndvi: false
  multi_scale: false
  scales: null

# ============================================================================
# OPTIMIZATIONS (v3.9+)
# ============================================================================
optimizations:
  enabled: true

  # Async I/O: +12-14% performance
  async_io_enabled: true
  async_workers: 2
  tile_cache_size: 3

  # Batch processing: +25-30% performance
  batch_processing_enabled: true
  batch_size: 4

  # GPU pooling: +8.5% performance
  gpu_pooling_enabled: true
  gpu_pool_max_size_gb: 4.0

  print_stats: true

# ============================================================================
# ADVANCED CONFIGURATION (optional, nested)
# ============================================================================
advanced:
  preprocessing:
    remove_outliers: true
    outlier_method: statistical
    outlier_std_threshold: 3.0

  ground_truth:
    enabled: true
    fuzzy_boundary_enabled: true
    fuzzy_boundary_outer: 2.5

  classification:
    confidence_threshold: 0.7
    min_points_per_class: 10

  performance:
    gpu_batch_size: 1000000
    gpu_memory_target: 0.85
    gpu_streams: 4
```

---

## 📋 Implementation Phases

### Phase 1: Python Config Consolidation (Week 1-2)

#### 1.1 Enhance `config.py` with v4.0 Structure

**File:** `ign_lidar/config/config.py`

**Actions:**

- ✅ Add `OptimizationsConfig` dataclass
- ✅ Rename `FeatureConfig.feature_set` → `FeatureConfig.mode`
- ✅ Add comprehensive docstrings (Google style)
- ✅ Add `from_legacy_schema()` migration method
- ✅ Add enhanced validation logic

**Code Changes:**

```python
# Add OptimizationsConfig
@dataclass
class OptimizationsConfig:
    """Phase 4 optimizations configuration."""
    enabled: bool = True
    async_io_enabled: bool = True
    async_workers: int = 2
    tile_cache_size: int = 3
    batch_processing_enabled: bool = True
    batch_size: int = 4
    gpu_pooling_enabled: bool = True
    gpu_pool_max_size_gb: float = 4.0
    print_stats: bool = True

# Update Config class
@dataclass
class Config:
    # ... existing fields ...

    # Add optimizations
    optimizations: OptimizationsConfig = field(default_factory=OptimizationsConfig)

    # Add migration method
    @classmethod
    def from_legacy_schema(cls, legacy_config: "IGNLiDARConfig") -> "Config":
        """
        Convert legacy v3.1 schema config to v4.0 Config.

        This method enables backward compatibility with old configs.
        """
        return cls(
            input_dir=legacy_config.input_dir,
            output_dir=legacy_config.output_dir,
            mode=legacy_config.processor.lod_level.lower(),
            processing_mode=legacy_config.processor.processing_mode,
            use_gpu=legacy_config.processor.use_gpu,
            num_workers=legacy_config.processor.num_workers,
            patch_size=legacy_config.processor.patch_size,
            num_points=legacy_config.processor.num_points,
            patch_overlap=legacy_config.processor.patch_overlap,
            architecture=legacy_config.processor.architecture,
            features=FeatureConfig(
                mode=_map_feature_mode(legacy_config.features.mode),
                k_neighbors=legacy_config.features.k_neighbors,
                # ... rest of mapping
            ),
        )
```

#### 1.2 Mark `schema.py` for Final Deprecation

**File:** `ign_lidar/config/schema.py`

**Actions:**

- ⚠️ Add loud deprecation warning (v3.9)
- 📝 Update deprecation message with migration instructions
- 🔗 Link to migration tool
- ⏰ Schedule removal in v4.0

**Code Changes:**

```python
"""
DEPRECATED: This module will be REMOVED in v4.0.0 (Q1 2026)

Use ign_lidar.config.Config instead.

Migration:
    # Automatic migration
    $ ign-lidar-hd migrate-config old_config.yaml

    # Manual migration
    from ign_lidar.config import Config
    config = Config.preset('lod2_buildings')

See: https://sducournau.github.io/IGN_LIDAR_HD_DATASET/guides/migration-v3-to-v4/
"""

import warnings

warnings.warn(
    "\n" + "=" * 80 + "\n"
    "DEPRECATION WARNING: ign_lidar.config.schema\n"
    "=" * 80 + "\n"
    "This module is DEPRECATED and will be REMOVED in v4.0.0.\n"
    "\n"
    "ACTION REQUIRED:\n"
    "  1. Run: ign-lidar-hd migrate-config your_config.yaml\n"
    "  2. Update imports: from ign_lidar.config import Config\n"
    "  3. See: https://sducournau.github.io/.../migration-v3-to-v4/\n"
    "\n"
    "Timeline: v3.9 (now) → v4.0 (removal in Q1 2026)\n"
    "=" * 80,
    DeprecationWarning,
    stacklevel=2
)

# Rest of schema.py code...
```

#### 1.3 Update All Imports

**Files to Update:**

- `ign_lidar/__init__.py` (line 103)
- `ign_lidar/cli/hydra_runner.py` (line 280)
- `ign_lidar/core/processor_core.py` (line 89)

**Search Pattern:**

```bash
grep -r "from.*config.schema import" ign_lidar/
grep -r "from.*config.*import.*ProcessorConfig" ign_lidar/
```

**Migration:**

```python
# OLD (v3.1)
from ign_lidar.config.schema import ProcessorConfig, FeaturesConfig, IGNLiDARConfig

# NEW (v4.0)
from ign_lidar.config import Config

# If legacy config received
if isinstance(config, IGNLiDARConfig):
    config = Config.from_legacy_schema(config)
```

---

### Phase 2: YAML Harmonization (Week 3-4)

#### 2.1 Update `base.yaml` to v4.0 Structure

**File:** `ign_lidar/configs/base.yaml`

**Changes:**

1. Flatten `processor.*` to top-level
2. Rename `processor.lod_level` → `mode` (lowercase)
3. Rename `features.mode` → `features.mode` (keep, but clarify)
4. Add `optimizations` section
5. Update to v4.0.0 version marker

**Before (v5.1):**

```yaml
config_version: "5.1.0"

processor:
  lod_level: "LOD2"
  processing_mode: "patches_only"
  use_gpu: true

features:
  mode: "lod2"
```

**After (v4.0):**

```yaml
config_version: "4.0.0"

# Top-level (flattened)
mode: lod2 # Lowercase
processing_mode: patches_only
use_gpu: true

features:
  mode: standard # Clarified: feature computation mode
```

#### 2.2 Update All 7 Preset Files

**Files:**

- `presets/asprs_classification_gpu.yaml`
- `presets/asprs_classification_cpu.yaml`
- `presets/lod2_buildings.yaml`
- `presets/lod3_detailed.yaml`
- `presets/fast_preview.yaml`
- `presets/minimal_debug.yaml`
- `presets/high_quality.yaml`

**Template for Preset Update:**

```yaml
# ============================================================================
# IGN LiDAR HD - Preset: {NAME}
# ============================================================================
defaults:
  - ../base
  - _self_

config_version: "4.0.0"
config_name: "{name}"

# ============================================================================
# OVERRIDES (flat structure, no nesting)
# ============================================================================
mode: { asprs|lod2|lod3 }
processing_mode: { patches_only|enriched_only|both }
use_gpu: { true|false }

features:
  mode: { minimal|standard|full }
  k_neighbors: { value }
  # ... other feature overrides

optimizations:
  enabled: true
  # ... optimization overrides
# ... advanced overrides if needed
```

#### 2.3 Update Example Configs

**Files:**

- `examples/TEMPLATE_v3.2.yaml` → `examples/TEMPLATE_v4.0.yaml`
- `examples/config_training_fast_50m_v3.2.yaml` → `examples/config_training_fast_50m_v4.0.yaml`
- `examples/config_asprs_production.yaml` (update in-place)
- `examples/config_multi_scale_adaptive.yaml` (update in-place)

---

### Phase 3: Migration Tooling (Week 5-6)

#### 3.1 Create `ConfigMigrator` Class

**File:** `ign_lidar/config/migration.py` (NEW)

```python
"""
Configuration migration utilities for v3.x → v4.0 upgrade.

Provides automatic migration of:
- v3.1 nested configs (schema.py)
- v3.2 flat configs
- v5.1 YAML configs
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import logging

from .config import Config, FeatureConfig, OptimizationsConfig

logger = logging.getLogger(__name__)


class ConfigMigrator:
    """Migrate configuration files between versions."""

    VERSION_MARKERS = {
        "3.1": ["IGNLiDARConfig", "processor.lod_level"],
        "3.2": ["mode", "feature_set"],
        "5.1": ["config_version: 5", "processor.lod_level"]
    }

    def detect_version(self, config_path: str) -> str:
        """
        Detect configuration file version.

        Args:
            config_path: Path to config file

        Returns:
            Version string ("3.1", "3.2", "5.1", or "unknown")
        """
        with open(config_path) as f:
            content = f.read()
            config_dict = yaml.safe_load(content)

        # Check explicit version marker
        if "config_version" in config_dict:
            version = config_dict["config_version"]
            if version.startswith("5"):
                return "5.1"
            elif version.startswith("4"):
                return "4.0"

        # Heuristic detection
        if "processor" in config_dict:
            if "lod_level" in config_dict.get("processor", {}):
                return "5.1" if "config_version" in config_dict else "3.1"

        if "mode" in config_dict and "processor" not in config_dict:
            return "3.2"

        return "unknown"

    def migrate(
        self,
        config_path: str,
        target_version: str = "4.0.0",
        output_path: Optional[str] = None
    ) -> Config:
        """
        Migrate configuration file to target version.

        Args:
            config_path: Path to source config file
            target_version: Target version (default: "4.0.0")
            output_path: Output path (default: config_path + ".v4.yaml")

        Returns:
            Migrated Config object

        Raises:
            ValueError: If source version is unsupported
        """
        source_version = self.detect_version(config_path)
        logger.info(f"Detected config version: {source_version}")

        if source_version == "4.0":
            logger.info("Config is already v4.0, no migration needed")
            return Config.from_yaml(config_path)

        # Dispatch to version-specific migrator
        if source_version == "3.1":
            config = self._migrate_from_v3_1(config_path)
        elif source_version == "3.2":
            config = self._migrate_from_v3_2(config_path)
        elif source_version == "5.1":
            config = self._migrate_from_v5_1(config_path)
        else:
            raise ValueError(f"Unsupported config version: {source_version}")

        # Validate migrated config
        errors = config.validate()
        if errors:
            logger.warning(f"Validation warnings: {errors}")

        # Save if output path specified
        if output_path:
            config.to_yaml(output_path)
            logger.info(f"Migrated config saved to: {output_path}")

        return config

    def _migrate_from_v3_1(self, config_path: str) -> Config:
        """Migrate from v3.1 nested structure."""
        with open(config_path) as f:
            old_config = yaml.safe_load(f)

        processor = old_config.get("processor", {})
        features = old_config.get("features", {})

        # Build v4.0 config
        return Config(
            input_dir=old_config.get("input_dir"),
            output_dir=old_config.get("output_dir"),
            mode=processor.get("lod_level", "LOD2").lower(),
            processing_mode=processor.get("processing_mode", "patches_only"),
            use_gpu=processor.get("use_gpu", False),
            num_workers=processor.get("num_workers", 4),
            patch_size=processor.get("patch_size", 150.0),
            num_points=processor.get("num_points", 16384),
            patch_overlap=processor.get("patch_overlap", 0.1),
            architecture=processor.get("architecture", "pointnet++"),
            features=FeatureConfig(
                mode=self._map_feature_mode(features.get("mode", "lod2")),
                k_neighbors=features.get("k_neighbors", 30),
                use_rgb=features.get("use_rgb", False),
                use_nir=features.get("use_infrared", False),
                compute_ndvi=features.get("compute_ndvi", False),
            ),
            optimizations=OptimizationsConfig(
                enabled=processor.get("enable_optimizations", True),
                async_io_enabled=processor.get("enable_async_io", True),
                batch_processing_enabled=processor.get("enable_batch_processing", True),
            )
        )

    def _migrate_from_v3_2(self, config_path: str) -> Config:
        """Migrate from v3.2 flat structure."""
        # Simpler migration - mostly 1:1 mapping
        return Config.from_yaml(config_path)

    def _migrate_from_v5_1(self, config_path: str) -> Config:
        """Migrate from v5.1 YAML structure."""
        with open(config_path) as f:
            old_config = yaml.safe_load(f)

        processor = old_config.get("processor", {})
        features = old_config.get("features", {})

        return Config(
            input_dir=old_config.get("input_dir"),
            output_dir=old_config.get("output_dir"),
            mode=processor.get("lod_level", "ASPRS").lower(),
            processing_mode=processor.get("processing_mode", "patches_only"),
            use_gpu=processor.get("use_gpu", False),
            num_workers=old_config.get("num_workers", 4),
            patch_size=old_config.get("patch_size", 150.0),
            num_points=old_config.get("num_points", 16384),
            features=FeatureConfig(
                mode=self._map_feature_mode(features.get("mode", "standard")),
                k_neighbors=features.get("k_neighbors", 30),
                use_rgb=features.get("use_rgb", False),
                use_nir=features.get("use_nir", False),
                compute_ndvi=features.get("compute_ndvi", False),
            ),
        )

    def _map_feature_mode(self, old_mode: str) -> str:
        """Map old feature mode names to v4.0."""
        mapping = {
            "lod2": "standard",
            "lod3": "full",
            "asprs_classes": "full",
            "minimal": "minimal",
            "standard": "standard",
            "full": "full",
        }
        return mapping.get(old_mode.lower(), "standard")

    def generate_migration_report(self, config_path: str) -> Dict[str, Any]:
        """Generate detailed migration report."""
        source_version = self.detect_version(config_path)

        report = {
            "source_version": source_version,
            "target_version": "4.0.0",
            "changes": [],
            "warnings": [],
            "breaking_changes": []
        }

        if source_version == "3.1":
            report["changes"].append("Flattened nested processor/features structure")
            report["changes"].append("Renamed lod_level → mode (lowercase)")
            report["breaking_changes"].append("IGNLiDARConfig → Config")

        if source_version in ["3.1", "5.1"]:
            report["changes"].append("Renamed features.mode for clarity")

        return report
```

#### 3.2 Create CLI Migration Command

**File:** `ign_lidar/cli/commands/migrate_config.py` (UPDATE)

```python
"""
CLI command for configuration migration.

Automatically migrates v3.1/v3.2/v5.1 configs to v4.0.
"""

import click
from pathlib import Path
import logging

from ...config.migration import ConfigMigrator

logger = logging.getLogger(__name__)


@click.command(name="migrate-config")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path (default: <input>.v4.yaml)"
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate migrated config"
)
@click.option(
    "--report/--no-report",
    default=False,
    help="Generate detailed migration report"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be migrated without writing file"
)
def migrate_config(config_file: str, output: str, validate: bool, report: bool, dry_run: bool):
    """
    Migrate configuration files from v3.x to v4.0.

    Automatically detects source version and performs appropriate migration.

    Examples:
        # Basic migration
        ign-lidar-hd migrate-config old_config.yaml

        # Specify output path
        ign-lidar-hd migrate-config old_config.yaml -o new_config.yaml

        # Dry run to preview changes
        ign-lidar-hd migrate-config old_config.yaml --dry-run

        # Generate detailed migration report
        ign-lidar-hd migrate-config old_config.yaml --report
    """
    migrator = ConfigMigrator()

    try:
        # Detect version
        source_version = migrator.detect_version(config_file)
        click.echo(f"✓ Detected config version: {source_version}")

        if source_version == "4.0":
            click.echo("✓ Config is already v4.0, no migration needed")
            return

        # Generate report if requested
        if report:
            migration_report = migrator.generate_migration_report(config_file)
            click.echo("\n" + "=" * 70)
            click.echo("MIGRATION REPORT")
            click.echo("=" * 70)
            click.echo(f"Source Version: {migration_report['source_version']}")
            click.echo(f"Target Version: {migration_report['target_version']}")
            click.echo("\nChanges:")
            for change in migration_report['changes']:
                click.echo(f"  • {change}")
            if migration_report['breaking_changes']:
                click.echo("\n⚠️  Breaking Changes:")
                for change in migration_report['breaking_changes']:
                    click.echo(f"  • {change}")
            click.echo("=" * 70 + "\n")

        # Perform migration
        if not dry_run:
            output_path = output or f"{config_file}.v4.yaml"
            migrated_config = migrator.migrate(config_file, output_path=output_path)

            click.echo(f"✓ Migrated config saved to: {output_path}")

            # Validate if requested
            if validate:
                errors = migrated_config.validate()
                if errors:
                    click.echo("\n⚠️  Validation warnings:", err=True)
                    for error in errors:
                        click.echo(f"  • {error}", err=True)
                else:
                    click.echo("✓ Config validated successfully")
        else:
            click.echo("\n[DRY RUN] Migration preview - no files written")
            migrated_config = migrator.migrate(config_file)
            click.echo("\nMigrated config preview:")
            click.echo(yaml.dump(migrated_config.to_dict(), sort_keys=False))

    except Exception as e:
        click.echo(f"✗ Migration failed: {e}", err=True)
        raise click.Abort()
```

---

### Phase 4: Documentation Consolidation (Week 7)

#### 4.1 Create Unified Configuration Guide

**File:** `docs/docs/guides/configuration/index.md` (NEW)

**Structure:**

````markdown
# Configuration System

## Overview

IGN LiDAR HD v4.0 uses a unified configuration system that combines Python dataclasses and YAML files.

### Key Features

- **Single Source of Truth:** One Config class, one base.yaml
- **Preset System:** Pre-configured workflows for common tasks
- **Type Safety:** Python dataclasses with validation
- **Backward Compatible:** Automatic migration from v3.x

## Quick Start

[5-minute quick start guide]

## Configuration Methods

### 1. Using Presets (Recommended)

```python
from ign_lidar.config import Config

config = Config.preset('asprs_production')
config.input_dir = '/data/tiles'
config.output_dir = '/data/output'
```
````

### 2. Loading from YAML

```python
config = Config.from_yaml('my_config.yaml')
```

### 3. Auto-Configuration

```python
config = Config.from_environment(
    input_dir='/data/tiles',
    output_dir='/data/output'
)
```

## Parameter Reference

[Link to complete reference]

## Migration from v3.x

[Link to migration guide]

````

#### 4.2 Create Migration Guide

**File:** `docs/docs/guides/configuration/migration-v3-to-v4.md` (NEW)

```markdown
# Migration Guide: v3.x → v4.0

## Overview

This guide helps you migrate from v3.1, v3.2, or v5.1 to the unified v4.0 configuration system.

## Automatic Migration (Recommended)

```bash
# Migrate your config file
ign-lidar-hd migrate-config old_config.yaml

# Output: old_config.yaml.v4.yaml
````

## What's Changed?

### 1. Naming Changes

| Old (v3.1/v5.1)       | New (v4.0)      | Location  |
| --------------------- | --------------- | --------- |
| `processor.lod_level` | `mode`          | Top-level |
| `features.mode`       | `features.mode` | Nested    |
| `processor.use_gpu`   | `use_gpu`       | Top-level |

### 2. Structure Changes

**Before (v5.1 - nested):**

```yaml
processor:
  lod_level: "LOD2"
  use_gpu: true
```

**After (v4.0 - flat):**

```yaml
mode: lod2
use_gpu: true
```

### 3. Python API Changes

**Before (v3.1 - deprecated):**

```python
from ign_lidar.config.schema import ProcessorConfig
config = ProcessorConfig(lod_level='LOD2')
```

**After (v4.0 - current):**

```python
from ign_lidar.config import Config
config = Config(mode='lod2')
```

## Manual Migration Steps

[Detailed step-by-step instructions]

## Common Issues

[FAQ and troubleshooting]

````

#### 4.3 Archive Old Documentation

**Action:** Move old README files to archive

```bash
# Create archive directory
mkdir -p ign_lidar/configs/archive/

# Move old docs
mv ign_lidar/configs/README.md ign_lidar/configs/archive/README_v5.1.md
mv ign_lidar/configs/README_V5.1.md ign_lidar/configs/archive/README_V5.1_original.md
mv ign_lidar/config/README.md ign_lidar/config/archive/README_v3.2.md
````

---

### Phase 5: Testing & Validation (Week 8)

#### 5.1 Unit Tests

**File:** `tests/config/test_migration.py` (NEW)

```python
"""
Tests for configuration migration v3.x → v4.0
"""

import pytest
from pathlib import Path
import yaml

from ign_lidar.config import Config
from ign_lidar.config.migration import ConfigMigrator


class TestConfigMigrator:
    """Test configuration migration functionality."""

    @pytest.fixture
    def migrator(self):
        return ConfigMigrator()

    @pytest.fixture
    def v3_1_config(self, tmp_path):
        """Sample v3.1 config."""
        config = {
            "input_dir": "/data/tiles",
            "output_dir": "/data/output",
            "processor": {
                "lod_level": "LOD2",
                "use_gpu": True,
                "processing_mode": "patches_only"
            },
            "features": {
                "mode": "lod2",
                "k_neighbors": 30
            }
        }
        config_file = tmp_path / "config_v3.1.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        return config_file

    @pytest.fixture
    def v5_1_config(self, tmp_path):
        """Sample v5.1 config."""
        config = {
            "config_version": "5.1.0",
            "processor": {
                "lod_level": "ASPRS",
                "use_gpu": False
            },
            "features": {
                "mode": "asprs_classes"
            }
        }
        config_file = tmp_path / "config_v5.1.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        return config_file

    def test_detect_version_v3_1(self, migrator, v3_1_config):
        """Test detection of v3.1 config."""
        version = migrator.detect_version(str(v3_1_config))
        assert version == "3.1"

    def test_detect_version_v5_1(self, migrator, v5_1_config):
        """Test detection of v5.1 config."""
        version = migrator.detect_version(str(v5_1_config))
        assert version == "5.1"

    def test_migrate_v3_1_to_v4_0(self, migrator, v3_1_config):
        """Test migration from v3.1 to v4.0."""
        config = migrator.migrate(str(v3_1_config))

        # Check structure
        assert isinstance(config, Config)
        assert config.mode == "lod2"
        assert config.use_gpu is True
        assert config.processing_mode == "patches_only"
        assert config.features.mode == "standard"
        assert config.features.k_neighbors == 30

    def test_migrate_v5_1_to_v4_0(self, migrator, v5_1_config):
        """Test migration from v5.1 to v4.0."""
        config = migrator.migrate(str(v5_1_config))

        assert config.mode == "asprs"
        assert config.use_gpu is False
        assert config.features.mode == "full"

    def test_migration_preserves_paths(self, migrator, v3_1_config):
        """Test that input/output paths are preserved."""
        config = migrator.migrate(str(v3_1_config))
        assert config.input_dir == "/data/tiles"
        assert config.output_dir == "/data/output"

    def test_migration_report(self, migrator, v3_1_config):
        """Test migration report generation."""
        report = migrator.generate_migration_report(str(v3_1_config))

        assert report["source_version"] == "3.1"
        assert report["target_version"] == "4.0.0"
        assert len(report["changes"]) > 0
        assert len(report["breaking_changes"]) > 0
```

#### 5.2 Integration Tests

**File:** `tests/integration/test_config_loading.py` (UPDATE)

```python
"""
Integration tests for v4.0 config loading across all methods.
"""

import pytest
from ign_lidar.config import Config


class TestConfigLoadingV4:
    """Test all v4.0 config loading methods."""

    def test_preset_loading(self):
        """Test loading presets."""
        config = Config.preset('asprs_production')
        assert config.mode == "asprs"
        assert isinstance(config, Config)

    def test_yaml_loading_v4(self, tmp_path):
        """Test loading v4.0 YAML files."""
        yaml_content = """
        config_version: "4.0.0"
        input_dir: /data/tiles
        output_dir: /data/output
        mode: lod2
        use_gpu: true
        """
        config_file = tmp_path / "config_v4.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.mode == "lod2"
        assert config.use_gpu is True

    def test_environment_config(self):
        """Test auto-configuration from environment."""
        config = Config.from_environment(
            input_dir="/data/tiles",
            output_dir="/data/output"
        )
        assert config.input_dir == "/data/tiles"
        assert config.output_dir == "/data/output"
```

---

## 📊 Rollout Strategy

### Timeline

```
Week 1-2: Phase 1 - Python Config Consolidation
Week 3-4: Phase 2 - YAML Harmonization
Week 5-6: Phase 3 - Migration Tooling
Week 7:   Phase 4 - Documentation
Week 8:   Phase 5 - Testing & Validation
Week 9:   Beta Release (v4.0.0-beta)
Week 10:  User Feedback & Fixes
Week 11:  Release Candidate (v4.0.0-rc)
Week 12:  Final Release (v4.0.0)
```

### Release Plan

#### v3.9 (Pre-release, Week 9)

**Purpose:** Deprecation warnings + migration tooling

**Contents:**

- ✅ All v4.0 features available
- ⚠️ Loud deprecation warnings for `schema.py`
- 🛠️ `migrate-config` CLI command
- 📚 Migration guide published
- ⏰ schema.py still present but deprecated

**User Action:**

- Install v3.9
- Run `ign-lidar-hd migrate-config` on old configs
- Test migrated configs with v3.9 (backward compatible)
- Report any migration issues

#### v4.0.0-beta (Week 9)

**Purpose:** Feature-complete beta for early adopters

**Contents:**

- ✅ Complete v4.0 config system
- ✅ Migration tooling
- ✅ New documentation
- ⏰ schema.py still present (deprecated)

**User Action:**

- Install beta: `pip install ign-lidar-hd==4.0.0b1`
- Test with migrated configs
- Report bugs

#### v4.0.0-rc (Week 11)

**Purpose:** Release candidate

**Contents:**

- ✅ All beta feedback incorporated
- ✅ Final documentation review
- ⏰ Last version with schema.py

#### v4.0.0 (Week 12)

**Purpose:** Stable release

**Contents:**

- ✅ `schema.py` REMOVED
- ✅ `schema_simplified.py` REMOVED
- ✅ Unified config system only
- ✅ Complete documentation

---

## 🎯 Success Criteria

### Quantitative Metrics

| Metric                     | Target | Measure               |
| -------------------------- | ------ | --------------------- |
| Config loading time        | <50ms  | Performance benchmark |
| Lines of code              | <1,200 | Line count            |
| Test coverage              | >95%   | pytest --cov          |
| Migration success rate     | >99%   | User reports          |
| Documentation completeness | 100%   | Manual review         |

### Qualitative Metrics

- ✅ Zero user reports of "config confusion"
- ✅ All presets load without errors
- ✅ All example configs work out-of-the-box
- ✅ Migration tool handles all known configs
- ✅ Documentation is clear and comprehensive

---

## 🚀 Next Steps

### Immediate (This Week)

1. ✅ **Review this plan** with team
2. ⬜ **Create GitHub milestone** for v4.0
3. ⬜ **Create issues** for each phase
4. ⬜ **Assign developers** to phases
5. ⬜ **Set up v4.0-dev branch**

### Short-Term (Weeks 1-2)

1. ⬜ **Start Phase 1:** Python config consolidation
2. ⬜ **Set up CI** for v4.0 branch
3. ⬜ **Draft user announcement** for v3.9/v4.0

### Medium-Term (Weeks 3-8)

1. ⬜ **Complete Phases 2-5**
2. ⬜ **Beta testing** with select users
3. ⬜ **Iterate** based on feedback

### Long-Term (Weeks 9-12)

1. ⬜ **Release v4.0.0**
2. ⬜ **Announce** on GitHub/PyPI/docs
3. ⬜ **Monitor** user adoption
4. ⬜ **Support** migration questions

---

## 📝 Appendix

### A. Complete Parameter Mapping

| v3.1/v5.1 Parameter         | v4.0 Parameter         | Notes            |
| --------------------------- | ---------------------- | ---------------- |
| `processor.lod_level`       | `mode`                 | Lowercase values |
| `processor.use_gpu`         | `use_gpu`              | Top-level        |
| `processor.processing_mode` | `processing_mode`      | Top-level        |
| `processor.num_workers`     | `num_workers`          | Top-level        |
| `processor.patch_size`      | `patch_size`           | Top-level        |
| `processor.num_points`      | `num_points`           | Top-level        |
| `features.mode`             | `features.mode`        | Still nested     |
| `features.k_neighbors`      | `features.k_neighbors` | Still nested     |
| `features.use_rgb`          | `features.use_rgb`     | Still nested     |
| `features.use_infrared`     | `features.use_nir`     | Renamed          |

### B. Breaking Changes Checklist

- [ ] `schema.py` removed
- [ ] `schema_simplified.py` removed
- [ ] `lod_level` → `mode` (lowercase)
- [ ] `use_infrared` → `use_nir`
- [ ] `feature_set` → `mode` (in FeatureConfig)
- [ ] Nested `processor.*` → top-level

### C. Resources

- **GitHub Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Documentation:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **PyPI:** https://pypi.org/project/ign-lidar-hd/

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Status:** Ready for Implementation  
**Owner:** IGN LiDAR HD Core Team
