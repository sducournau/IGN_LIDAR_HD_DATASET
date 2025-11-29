# GitHub Issues for Configuration v4.0 Harmonization

**Purpose:** Issue templates ready for GitHub milestone creation  
**Date:** November 28, 2025  
**Milestone:** v4.0.0 Configuration Harmonization  
**Total Issues:** 25

---

## 🏷️ Issue Labels

Create these labels first:

- `config-harmonization` (purple)
- `breaking-change` (red)
- `migration` (orange)
- `documentation` (blue)
- `v4.0` (green)

---

## 📦 Phase 1: Python Config Consolidation (Week 1-2)

### Issue #1: Add OptimizationsConfig to config.py

**Title:** Add OptimizationsConfig dataclass for Phase 4 optimizations

**Labels:** `config-harmonization`, `v4.0`, `enhancement`

**Description:**
```markdown
## Goal
Add `OptimizationsConfig` dataclass to `ign_lidar/config/config.py` to consolidate Phase 4 optimization parameters.

## Current State
Phase 4 optimization parameters are scattered as individual fields in `Config` class:
- `enable_optimizations`
- `enable_async_io`, `async_workers`, `tile_cache_size`
- `enable_batch_processing`, `batch_size`
- `enable_gpu_pooling`, `gpu_pool_max_size_gb`

## Proposed Changes

1. Create new `OptimizationsConfig` dataclass:
```python
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
```

2. Add to `Config` class:
```python
optimizations: OptimizationsConfig = field(default_factory=OptimizationsConfig)
```

3. Update all references to old parameter names

## Acceptance Criteria
- [ ] `OptimizationsConfig` class created with all parameters
- [ ] Added to `Config` class as nested field
- [ ] Backward compatibility maintained (old params still work with deprecation)
- [ ] Unit tests pass
- [ ] Documentation updated

## Related Files
- `ign_lidar/config/config.py`
- `ign_lidar/core/processor.py` (update references)
- `tests/test_config.py`

## Estimated Time
3-4 hours
```

---

### Issue #2: Rename FeatureConfig.feature_set → mode

**Title:** Standardize feature parameter naming: feature_set → mode

**Labels:** `config-harmonization`, `v4.0`, `breaking-change`

**Description:**
```markdown
## Goal
Rename `FeatureConfig.feature_set` to `FeatureConfig.mode` for consistency with YAML and overall naming scheme.

## Rationale
Current inconsistency:
- Python: `features.feature_set`
- YAML: `features.mode`

This causes confusion. We should use `mode` everywhere.

## Changes Required

1. In `ign_lidar/config/config.py`:
```python
@dataclass
class FeatureConfig:
    mode: Literal["minimal", "standard", "full", "custom"] = "standard"  # was: feature_set
```

2. Add deprecation handling:
```python
def __post_init__(self):
    # Backward compatibility
    if hasattr(self, 'feature_set') and self.feature_set:
        warnings.warn(
            "feature_set is deprecated, use mode instead",
            DeprecationWarning
        )
        self.mode = self.feature_set
```

3. Update all references:
- Grep search: `feature_set`
- Update: ~10-15 files

## Acceptance Criteria
- [ ] Parameter renamed to `mode`
- [ ] Deprecation warning for old `feature_set` parameter
- [ ] All internal references updated
- [ ] Tests updated and passing
- [ ] CHANGELOG entry added

## Breaking Change
⚠️ This is a breaking change in v4.0. Mark as such in release notes.

## Estimated Time
4-5 hours
```

---

### Issue #3: Add Config.from_legacy_schema() migration method

**Title:** Implement Config.from_legacy_schema() for v3.1 migration

**Labels:** `config-harmonization`, `v4.0`, `migration`

**Description:**
```markdown
## Goal
Add class method to convert legacy v3.1 `IGNLiDARConfig` to v4.0 `Config`.

## Implementation

Add to `Config` class:

```python
@classmethod
def from_legacy_schema(cls, legacy_config: "IGNLiDARConfig") -> "Config":
    """
    Convert legacy v3.1 schema config to v4.0 Config.
    
    Enables backward compatibility with old schema.py configs.
    
    Args:
        legacy_config: Legacy IGNLiDARConfig instance
        
    Returns:
        Migrated Config instance
        
    Example:
        >>> from ign_lidar.config.schema import IGNLiDARConfig
        >>> legacy = IGNLiDARConfig(...)
        >>> config = Config.from_legacy_schema(legacy)
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
            mode=_map_legacy_feature_mode(legacy_config.features.mode),
            k_neighbors=legacy_config.features.k_neighbors,
            use_rgb=legacy_config.features.use_rgb,
            use_nir=legacy_config.features.use_infrared,
            compute_ndvi=legacy_config.features.compute_ndvi,
        ),
    )
```

## Acceptance Criteria
- [ ] Method implemented with full parameter mapping
- [ ] Helper function `_map_legacy_feature_mode()` created
- [ ] Unit tests for migration
- [ ] Test with real v3.1 configs
- [ ] Documentation with examples

## Test Cases
- LOD2 config → v4.0
- LOD3 config → v4.0
- ASPRS config → v4.0
- Custom config with advanced options

## Estimated Time
5-6 hours
```

---

### Issue #4: Add comprehensive docstrings to Config class

**Title:** Document Config class with comprehensive Google-style docstrings

**Labels:** `config-harmonization`, `v4.0`, `documentation`

**Description:**
```markdown
## Goal
Add comprehensive documentation to all Config class parameters and methods.

## Requirements

Each parameter needs:
- Type annotation
- Default value
- Description
- Valid values/range
- Examples
- Notes/warnings (if applicable)

## Example Format

```python
input_dir: str = MISSING
"""
Input directory containing LAZ/LAS files.

**Required.** Must be specified via CLI, YAML, or code.

Type:
    str (path)

Examples:
    - `/data/lidar/tiles`
    - `/mnt/storage/ign_lidar/raw`
    - `s3://bucket/tiles` (if S3 support enabled)

Notes:
    Directory must exist and contain .laz or .las files.
    Subdirectories are scanned recursively.

See Also:
    - output_dir: Corresponding output directory
    - User Guide: https://sducournau.github.io/.../quickstart/
"""
```

## Scope
- All `Config` parameters (~15 top-level)
- All `FeatureConfig` parameters (~8)
- All `OptimizationsConfig` parameters (~9)
- All class methods

## Acceptance Criteria
- [ ] Every parameter documented
- [ ] Every method documented
- [ ] Class docstring with quick start examples
- [ ] Sphinx/mkdocs can generate API docs
- [ ] No missing docstrings (checked by linter)

## Estimated Time
6-8 hours
```

---

### Issue #5: Update imports to remove schema.py dependencies

**Title:** Remove schema.py imports across codebase

**Labels:** `config-harmonization`, `v4.0`, `breaking-change`

**Description:**
```markdown
## Goal
Update all files that import from `schema.py` to use new `Config` instead.

## Files to Update

Found via grep:
```bash
grep -r "from.*config.schema import" ign_lidar/
```

Results:
1. `ign_lidar/__init__.py` (line 103)
2. `ign_lidar/cli/hydra_runner.py` (line 280)
3. `ign_lidar/core/processor_core.py` (line 89)

## Changes

### 1. `ign_lidar/__init__.py`

**Before:**
```python
from .config.schema import (
    IGNLiDARConfig,
    ProcessorConfig,
    FeaturesConfig,
)
```

**After:**
```python
# Legacy imports for backward compatibility (deprecated)
try:
    from .config.schema import (
        IGNLiDARConfig,
        ProcessorConfig,
        FeaturesConfig,
    )
except ImportError:
    # schema.py removed in v4.0
    pass
```

### 2. `cli/hydra_runner.py`

Add conversion:
```python
if isinstance(config, IGNLiDARConfig):
    config = Config.from_legacy_schema(config)
```

### 3. `core/processor_core.py`

Same conversion pattern.

## Acceptance Criteria
- [ ] All imports updated
- [ ] Backward compatibility maintained (v3.9)
- [ ] Tests pass with both old and new configs
- [ ] Deprecation warnings shown

## Estimated Time
3-4 hours
```

---

### Issue #6: Add final deprecation warning to schema.py

**Title:** Add loud deprecation warning to schema.py (v3.9)

**Labels:** `config-harmonization`, `v4.0`, `migration`

**Description:**
```markdown
## Goal
Add highly visible deprecation warning to `schema.py` that will appear in v3.9.

## Implementation

Update module docstring and add warning at import time:

```python
"""
DEPRECATED: This module will be REMOVED in v4.0.0 (Q1 2026)

⚠️  ACTION REQUIRED  ⚠️

This configuration system is deprecated. You MUST migrate to v4.0 before
the v4.0.0 release.

MIGRATION STEPS:
    1. Run migration tool:
       $ ign-lidar-hd migrate-config your_config.yaml
    
    2. Update Python imports:
       # OLD (deprecated)
       from ign_lidar.config.schema import ProcessorConfig
       
       # NEW (v4.0)
       from ign_lidar.config import Config
    
    3. Test migrated config with v3.9 (fully compatible)
    
    4. Upgrade to v4.0 when ready

TIMELINE:
    - v3.9 (now): Deprecation warnings, migration tool available
    - v4.0 (Q1 2026): schema.py REMOVED

HELP:
    - Migration guide: https://sducournau.github.io/.../migration-v3-to-v4/
    - GitHub issues: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
"""

import warnings

warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATION WARNING: ign_lidar.config.schema  ⚠️\n"
    "=" * 80 + "\n"
    "This module will be REMOVED in v4.0.0 (Q1 2026)\n"
    "\n"
    "ACTION REQUIRED:\n"
    "  1. Run: ign-lidar-hd migrate-config your_config.yaml\n"
    "  2. Update imports: from ign_lidar.config import Config\n"
    "  3. See: https://sducournau.github.io/.../migration-v3-to-v4/\n"
    "\n"
    "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
```

## Acceptance Criteria
- [ ] Warning displays on any import
- [ ] Clear migration instructions
- [ ] Links to documentation
- [ ] Timeline specified

## Estimated Time
1-2 hours
```

---

## 📦 Phase 2: YAML Harmonization (Week 3-4)

### Issue #7: Update base.yaml to v4.0 structure

**Title:** Flatten and harmonize base.yaml configuration

**Labels:** `config-harmonization`, `v4.0`, `breaking-change`

**Description:**
```markdown
## Goal
Update `ign_lidar/configs/base.yaml` from v5.1 nested structure to v4.0 flat structure.

## Changes

### Structure Changes

**Before (v5.1 - nested):**
```yaml
config_version: "5.1.0"

processor:
  lod_level: "ASPRS"
  processing_mode: "enriched_only"
  use_gpu: true
  num_workers: 4
  
features:
  mode: "asprs_classes"
```

**After (v4.0 - flat):**
```yaml
config_version: "4.0.0"

# Top-level essential parameters
mode: asprs  # lowercase
processing_mode: enriched_only
use_gpu: true
num_workers: 4

features:
  mode: full  # clarified naming
```

### Naming Changes
- `processor.lod_level: "ASPRS"` → `mode: asprs` (lowercase)
- `features.mode: "asprs_classes"` → `features.mode: full`

### New Section
Add optimizations section:
```yaml
optimizations:
  enabled: true
  async_io_enabled: true
  async_workers: 2
  tile_cache_size: 3
  batch_processing_enabled: true
  batch_size: 4
  gpu_pooling_enabled: true
  gpu_pool_max_size_gb: 4.0
  print_stats: true
```

## Acceptance Criteria
- [ ] base.yaml updated to v4.0 structure
- [ ] All parameters flattened appropriately
- [ ] optimizations section added
- [ ] Version marker updated to 4.0.0
- [ ] Hydra can still compose configs
- [ ] Tests pass

## Estimated Time
3-4 hours
```

---

### Issue #8: Update preset files to v4.0 (batch)

**Title:** Migrate all 7 preset files to v4.0 structure

**Labels:** `config-harmonization`, `v4.0`

**Description:**
```markdown
## Goal
Update all preset configuration files to v4.0 flat structure.

## Files to Update

1. `presets/asprs_classification_gpu.yaml`
2. `presets/asprs_classification_cpu.yaml`
3. `presets/lod2_buildings.yaml`
4. `presets/lod3_detailed.yaml`
5. `presets/fast_preview.yaml`
6. `presets/minimal_debug.yaml`
7. `presets/high_quality.yaml`

## Template

Each preset should follow this structure:

```yaml
# ============================================================================
# IGN LiDAR HD - Preset: {NAME}
# ============================================================================
defaults:
  - ../base
  - _self_

config_version: "4.0.0"
config_name: "{preset_name}"

# ============================================================================
# OVERRIDES (flat structure)
# ============================================================================
mode: {asprs|lod2|lod3}
processing_mode: {patches_only|enriched_only|both}
use_gpu: {true|false}

features:
  mode: {minimal|standard|full}
  k_neighbors: {value}
  
# ... additional overrides
```

## Acceptance Criteria
- [ ] All 7 presets updated
- [ ] Hydra composition works
- [ ] Preset loading tests pass
- [ ] Each preset tested individually

## Checklist
- [ ] asprs_classification_gpu.yaml
- [ ] asprs_classification_cpu.yaml
- [ ] lod2_buildings.yaml
- [ ] lod3_detailed.yaml
- [ ] fast_preview.yaml
- [ ] minimal_debug.yaml
- [ ] high_quality.yaml

## Estimated Time
4-5 hours (batch update)
```

---

### Issue #9: Update example configs to v4.0

**Title:** Migrate example configurations to v4.0 structure

**Labels:** `config-harmonization`, `v4.0`, `documentation`

**Description:**
```markdown
## Goal
Update all example configuration files in `examples/` directory to v4.0.

## Files

1. `TEMPLATE_v3.2.yaml` → `TEMPLATE_v4.0.yaml`
2. `config_training_fast_50m_v3.2.yaml` → `config_training_fast_50m_v4.0.yaml`
3. `config_asprs_production.yaml` (update in place)
4. `config_multi_scale_adaptive.yaml` (update in place)

## Actions

1. Create new v4.0 versions
2. Keep old versions for reference
3. Update README to point to v4.0 versions
4. Add migration notice to old files

## Acceptance Criteria
- [ ] All examples updated
- [ ] Old examples preserved with deprecation notice
- [ ] README updated
- [ ] Examples tested and working

## Estimated Time
3-4 hours
```

---

## 📦 Phase 3: Migration Tooling (Week 5-6)

### Issue #10: Implement ConfigMigrator class

**Title:** Create ConfigMigrator for automatic config migration

**Labels:** `config-harmonization`, `v4.0`, `migration`

**Description:**
```markdown
## Goal
Implement `ConfigMigrator` class in new `ign_lidar/config/migration.py` module.

## Functionality

1. **Version Detection**
   - Detect config version (3.1, 3.2, 5.1, 4.0)
   - Heuristic detection if no version marker

2. **Migration**
   - v3.1 → v4.0
   - v3.2 → v4.0
   - v5.1 → v4.0

3. **Validation**
   - Validate migrated config
   - Report warnings/errors

4. **Reporting**
   - Generate migration report
   - Show what changed

## Implementation

See implementation plan document for full code.

Key methods:
- `detect_version(config_path) -> str`
- `migrate(config_path, target_version) -> Config`
- `_migrate_from_v3_1(config_path) -> Config`
- `_migrate_from_v3_2(config_path) -> Config`
- `_migrate_from_v5_1(config_path) -> Config`
- `generate_migration_report(config_path) -> Dict`

## Acceptance Criteria
- [ ] All methods implemented
- [ ] Version detection works for all versions
- [ ] Migration successful for test configs
- [ ] Validation integrated
- [ ] Report generation works
- [ ] Unit tests (>90% coverage)

## Test Cases
- Detect v3.1 config
- Detect v3.2 config
- Detect v5.1 config
- Migrate LOD2 config
- Migrate ASPRS config
- Migrate config with all options
- Handle missing parameters
- Handle invalid configs

## Estimated Time
8-10 hours
```

---

### Issue #11: Create migrate-config CLI command

**Title:** Implement ign-lidar-hd migrate-config CLI command

**Labels:** `config-harmonization`, `v4.0`, `migration`, `cli`

**Description:**
```markdown
## Goal
Create user-friendly CLI command for config migration.

## Usage

```bash
# Basic migration
ign-lidar-hd migrate-config old_config.yaml

# Specify output
ign-lidar-hd migrate-config old_config.yaml -o new_config.yaml

# Dry run (preview)
ign-lidar-hd migrate-config old_config.yaml --dry-run

# With migration report
ign-lidar-hd migrate-config old_config.yaml --report
```

## Features

- Automatic version detection
- Validation of migrated config
- Dry-run mode
- Detailed migration report
- Progress feedback
- Error handling

## Implementation

Update `ign_lidar/cli/commands/migrate_config.py`:

```python
@click.command(name="migrate-config")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path())
@click.option("--validate/--no-validate", default=True)
@click.option("--report/--no-report", default=False)
@click.option("--dry-run", is_flag=True)
def migrate_config(config_file, output, validate, report, dry_run):
    """Migrate configuration from v3.x to v4.0."""
    # Implementation...
```

## Acceptance Criteria
- [ ] Command registered in CLI
- [ ] All options work
- [ ] User-friendly output
- [ ] Error messages helpful
- [ ] Help text comprehensive
- [ ] Tests for CLI command

## Estimated Time
4-5 hours
```

---

### Issue #12: Write migration tests

**Title:** Comprehensive test suite for config migration

**Labels:** `config-harmonization`, `v4.0`, `testing`

**Description:**
```markdown
## Goal
Create comprehensive test suite for migration functionality.

## Test Files

1. `tests/config/test_migration.py` - ConfigMigrator tests
2. `tests/cli/test_migrate_config.py` - CLI command tests
3. `tests/integration/test_migration_e2e.py` - End-to-end tests

## Test Coverage

### Unit Tests (ConfigMigrator)
- [ ] Version detection (all versions)
- [ ] Migration v3.1 → v4.0
- [ ] Migration v3.2 → v4.0
- [ ] Migration v5.1 → v4.0
- [ ] Parameter mapping
- [ ] Feature mode mapping
- [ ] Report generation
- [ ] Error handling

### CLI Tests
- [ ] Basic migration
- [ ] Output path
- [ ] Dry run
- [ ] Report generation
- [ ] Validation
- [ ] Error cases

### Integration Tests
- [ ] Migrate and load config
- [ ] Use migrated config in pipeline
- [ ] Batch migration
- [ ] Real config files

## Test Data

Create fixtures for each version:
- `fixtures/config_v3.1.yaml`
- `fixtures/config_v3.2.yaml`
- `fixtures/config_v5.1.yaml`

## Acceptance Criteria
- [ ] >90% test coverage
- [ ] All version combinations tested
- [ ] Edge cases covered
- [ ] Integration tests pass
- [ ] CI passes

## Estimated Time
6-8 hours
```

---

## 📦 Phase 4: Documentation (Week 7)

### Issue #13: Create unified configuration guide

**Title:** Write comprehensive v4.0 configuration guide

**Labels:** `config-harmonization`, `v4.0`, `documentation`

**Description:**
```markdown
## Goal
Create single, comprehensive configuration guide replacing 5+ scattered docs.

## Structure

```
docs/docs/guides/configuration/
├── index.md                    # Main guide
├── quickstart.md               # 5-min quick start
├── reference.md                # Complete parameter reference
├── presets.md                  # Preset catalog
├── advanced.md                 # Advanced features
├── migration-v3-to-v4.md       # Migration guide
└── examples/
    ├── basic-usage.md
    ├── gpu-optimization.md
    └── multi-scale.md
```

## Content Requirements

### index.md
- Overview of config system
- Design principles
- Configuration methods (presets, YAML, Python)
- Quick examples
- Links to other guides

### quickstart.md
- 5-minute tutorial
- Minimal working example
- Common use cases

### reference.md
- Every parameter documented
- Type, default, valid values
- Examples for each
- Notes and warnings

### presets.md
- Catalog of all presets
- When to use each
- Performance characteristics
- Customization examples

### advanced.md
- Advanced features
- Custom configurations
- Performance tuning
- Troubleshooting

### migration-v3-to-v4.md
- Why migrate
- Automatic migration
- Manual migration
- Breaking changes
- FAQ

## Acceptance Criteria
- [ ] All guides written
- [ ] Cross-references work
- [ ] Code examples tested
- [ ] Docusaurus builds
- [ ] Review complete

## Estimated Time
12-15 hours
```

---

### Issue #14: Create migration guide

**Title:** Write detailed migration guide (v3.x → v4.0)

**Labels:** `config-harmonization`, `v4.0`, `documentation`, `migration`

**Description:**
```markdown
## Goal
Detailed guide for users migrating from v3.x to v4.0.

## Sections

1. **Overview**
   - Why migrate
   - What's changed
   - Timeline

2. **Automatic Migration** (Recommended)
   - Using CLI tool
   - Step-by-step
   - Validation

3. **Manual Migration**
   - Parameter mapping table
   - Structure changes
   - Code examples

4. **Breaking Changes**
   - Complete list
   - Impact assessment
   - Mitigation strategies

5. **Python API Changes**
   - Import changes
   - Class changes
   - Method changes

6. **YAML Changes**
   - Structure changes
   - Naming changes
   - Examples

7. **FAQ**
   - Common issues
   - Troubleshooting
   - Getting help

## Acceptance Criteria
- [ ] All sections complete
- [ ] Examples tested
- [ ] FAQ covers common issues
- [ ] Links work
- [ ] Review complete

## Estimated Time
6-8 hours
```

---

### Issue #15: Update inline documentation

**Title:** Add docstrings to all config classes and methods

**Labels:** `config-harmonization`, `v4.0`, `documentation`

**Description:**
```markdown
## Goal
Comprehensive inline documentation for config module.

## Files

- `ign_lidar/config/config.py`
- `ign_lidar/config/migration.py`
- `ign_lidar/config/validator.py`
- `ign_lidar/config/preset_loader.py`

## Requirements

- Google-style docstrings
- Every parameter documented
- Every method documented
- Type hints complete
- Examples included

## Acceptance Criteria
- [ ] No missing docstrings
- [ ] Sphinx can generate API docs
- [ ] Examples work
- [ ] Links valid

## Estimated Time
8-10 hours
```

---

### Issue #16: Archive old configuration docs

**Title:** Archive legacy configuration documentation

**Labels:** `config-harmonization`, `v4.0`, `documentation`

**Description:**
```markdown
## Goal
Archive old configuration docs without deleting them.

## Actions

```bash
# Create archive directories
mkdir -p ign_lidar/config/archive/
mkdir -p ign_lidar/configs/archive/

# Move old docs
mv ign_lidar/config/README.md ign_lidar/config/archive/README_v3.2.md
mv ign_lidar/configs/README.md ign_lidar/configs/archive/README_v5.1.md
mv ign_lidar/configs/README_V5.1.md ign_lidar/configs/archive/README_V5.1_original.md
```

## Create Archive README

Add `archive/README.md`:

```markdown
# Archived Configuration Documentation

This directory contains configuration documentation from previous versions.

**Current Documentation:** See `/docs/docs/guides/configuration/`

## Archived Versions

- `README_v3.2.md` - v3.2 configuration guide
- `README_v5.1.md` - v5.1 configuration guide
- `README_V5.1_original.md` - Original v5.1 guide

These are preserved for reference only. All new projects should use v4.0.

For migration help, see: https://sducournau.github.io/.../migration-v3-to-v4/
```

## Acceptance Criteria
- [ ] Old docs archived
- [ ] Archive README created
- [ ] Links updated
- [ ] No docs lost

## Estimated Time
1-2 hours
```

---

## 📦 Phase 5: Testing & Validation (Week 8)

### Issue #17: Write comprehensive unit tests for v4.0

**Title:** Unit test suite for Config class and migration

**Labels:** `config-harmonization`, `v4.0`, `testing`

**Description:**
```markdown
## Goal
Comprehensive unit tests for all v4.0 configuration functionality.

## Test Files

1. `tests/config/test_config_v4.py` - Config class tests
2. `tests/config/test_feature_config.py` - FeatureConfig tests
3. `tests/config/test_optimizations_config.py` - OptimizationsConfig tests
4. `tests/config/test_migration.py` - Migration tests

## Coverage Target

>95% line coverage

## Test Categories

### Config Class Tests
- [ ] Initialization with defaults
- [ ] Initialization with overrides
- [ ] Preset loading
- [ ] YAML loading
- [ ] Environment config
- [ ] Legacy schema conversion
- [ ] Validation
- [ ] Serialization (to_dict, to_yaml)

### FeatureConfig Tests
- [ ] All parameter combinations
- [ ] Validation (ndvi requires rgb+nir)
- [ ] Multi-scale config
- [ ] Feature list generation

### OptimizationsConfig Tests
- [ ] All optimization combinations
- [ ] Enable/disable master switch
- [ ] Parameter bounds

### Migration Tests
- [ ] Version detection
- [ ] v3.1 migration
- [ ] v3.2 migration
- [ ] v5.1 migration
- [ ] Parameter mapping
- [ ] Report generation

## Acceptance Criteria
- [ ] >95% coverage
- [ ] All edge cases covered
- [ ] Fixtures for each version
- [ ] Fast execution (<5s)
- [ ] CI passes

## Estimated Time
8-10 hours
```

---

### Issue #18: Integration tests for v4.0 configs

**Title:** End-to-end integration tests with v4.0 configs

**Labels:** `config-harmonization`, `v4.0`, `testing`, `integration`

**Description:**
```markdown
## Goal
Verify v4.0 configs work in full processing pipeline.

## Test Scenarios

1. **Preset Usage**
   - Load each preset
   - Run mini processing task
   - Verify output

2. **YAML Loading**
   - Load v4.0 YAML
   - Run processing
   - Verify output

3. **Migrated Configs**
   - Migrate v3.1 config
   - Use in pipeline
   - Compare results with original

4. **All Modes**
   - ASPRS mode
   - LOD2 mode
   - LOD3 mode

5. **Hardware Variations**
   - CPU only
   - GPU enabled
   - Multi-worker

## Acceptance Criteria
- [ ] All presets tested
- [ ] All modes tested
- [ ] CPU and GPU tested
- [ ] Results verified
- [ ] Performance benchmarked

## Estimated Time
10-12 hours
```

---

### Issue #19: Validate all preset configurations

**Title:** Test and validate all 7 presets with real data

**Labels:** `config-harmonization`, `v4.0`, `testing`, `validation`

**Description:**
```markdown
## Goal
Ensure all presets work correctly with real LiDAR data.

## Test Matrix

| Preset | Hardware | Dataset | Status |
|--------|----------|---------|--------|
| asprs_classification_gpu | RTX 4080 | IGN Sample | ⬜ |
| asprs_classification_cpu | CPU | IGN Sample | ⬜ |
| lod2_buildings | Mixed | Building Dense | ⬜ |
| lod3_detailed | GPU | Heritage | ⬜ |
| fast_preview | CPU | Quick Test | ⬜ |
| minimal_debug | CPU | Debug Data | ⬜ |
| high_quality | GPU | Full Quality | ⬜ |

## Validation Criteria

For each preset:
- [ ] Loads without errors
- [ ] Processes test tile
- [ ] Produces expected output
- [ ] Performance within bounds
- [ ] Memory usage acceptable

## Deliverables

- Test report per preset
- Performance benchmarks
- Known issues documented
- Recommendations for users

## Estimated Time
8-10 hours
```

---

## 🚢 Release Preparation

### Issue #20: Create v3.9 pre-release

**Title:** Package v3.9 with deprecation warnings and migration tools

**Labels:** `config-harmonization`, `v4.0`, `release`

**Description:**
```markdown
## Goal
Release v3.9 as pre-release with full v4.0 features + backward compatibility.

## Contents

- ✅ All v4.0 Config features
- ⚠️ Deprecation warnings for schema.py
- 🛠️ migrate-config CLI tool
- 📚 Migration documentation
- ⏰ schema.py still present

## Checklist

- [ ] All Phase 1-3 issues complete
- [ ] Tests pass
- [ ] Documentation published
- [ ] CHANGELOG updated
- [ ] Version bumped to 3.9.0
- [ ] PyPI release notes prepared
- [ ] GitHub release created
- [ ] Announcement prepared

## Timeline

Target: End of Week 8

## Estimated Time
4-6 hours (packaging + release)
```

---

### Issue #21: Beta testing v4.0

**Title:** Beta test v4.0 with select users

**Labels:** `config-harmonization`, `v4.0`, `testing`, `beta`

**Description:**
```markdown
## Goal
Get feedback from beta testers before final release.

## Beta Testers

- Internal team members
- 3-5 external users (volunteers)
- CI/CD pipelines

## Test Period

2 weeks

## Feedback Areas

1. Migration experience
2. Documentation clarity
3. Breaking changes impact
4. Performance
5. Bugs/issues

## Deliverables

- Beta feedback report
- Bug fixes
- Documentation improvements
- Final adjustments

## Estimated Time
Ongoing during Week 9-10
```

---

### Issue #22: Prepare v4.0 release notes

**Title:** Write comprehensive v4.0 release notes

**Labels:** `config-harmonization`, `v4.0`, `documentation`, `release`

**Description:**
```markdown
## Goal
Comprehensive release notes for v4.0.

## Sections

1. **Overview**
   - Major changes
   - Why v4.0
   - Benefits

2. **Breaking Changes**
   - Complete list
   - Migration guide link
   - Upgrade timeline

3. **New Features**
   - Unified config system
   - Migration tooling
   - Better documentation

4. **Improvements**
   - Performance
   - Usability
   - Developer experience

5. **Deprecations**
   - What's removed
   - Alternatives

6. **Migration Guide**
   - Quick steps
   - Link to full guide

7. **Known Issues**
   - Any caveats
   - Workarounds

8. **Contributors**
   - Thank contributors

## Acceptance Criteria
- [ ] All sections complete
- [ ] Accurate and clear
- [ ] Links work
- [ ] Review approved

## Estimated Time
4-5 hours
```

---

### Issue #23: Final v4.0 release

**Title:** Release v4.0.0 stable

**Labels:** `config-harmonization`, `v4.0`, `release`

**Description:**
```markdown
## Goal
Stable release of v4.0 configuration harmonization.

## Pre-release Checklist

- [ ] All issues resolved
- [ ] Beta feedback addressed
- [ ] Tests pass (100%)
- [ ] Documentation complete
- [ ] Performance validated
- [ ] Release notes ready

## Release Steps

1. Final version bump (4.0.0)
2. Update CHANGELOG
3. Tag release in git
4. Build package
5. Upload to PyPI
6. Create GitHub release
7. Update documentation
8. Announce release

## Post-release

- Monitor for issues
- Support users
- Quick fixes if needed

## Timeline

Target: End of Week 12

## Estimated Time
4-6 hours
```

---

## 📈 Tracking & Monitoring

### Issue #24: Create v4.0 project board

**Title:** Set up GitHub project board for v4.0 tracking

**Labels:** `config-harmonization`, `v4.0`, `project-management`

**Description:**
```markdown
## Goal
Visual tracking of v4.0 progress.

## Columns

1. **Backlog** - Not started
2. **In Progress** - Actively working
3. **Review** - Needs code review
4. **Testing** - In QA
5. **Done** - Completed

## Setup

1. Create project: "Configuration v4.0 Harmonization"
2. Add all issues to project
3. Set up automation:
   - New issue → Backlog
   - Assigned → In Progress
   - PR opened → Review
   - PR merged → Testing
   - Issue closed → Done

## Milestones

- [ ] Phase 1 Complete (Week 2)
- [ ] Phase 2 Complete (Week 4)
- [ ] Phase 3 Complete (Week 6)
- [ ] Phase 4 Complete (Week 7)
- [ ] Phase 5 Complete (Week 8)
- [ ] v3.9 Released (Week 9)
- [ ] v4.0 Released (Week 12)

## Estimated Time
2-3 hours
```

---

### Issue #25: Weekly progress reports

**Title:** Weekly progress updates for v4.0

**Labels:** `config-harmonization`, `v4.0`, `project-management`

**Description:**
```markdown
## Goal
Regular progress updates to track v4.0 development.

## Format

**Week X Progress Report**

### Completed
- List completed issues
- Key achievements

### In Progress
- Current work
- Blockers

### Upcoming
- Next week's plan
- Priorities

### Metrics
- Issues closed: X/25
- Test coverage: X%
- Documentation: X%

## Frequency

Weekly (every Friday)

## Distribution

- GitHub discussions
- Team meeting
- Stakeholders

## Estimated Time
1 hour/week
```

---

## 📊 Summary

**Total Issues:** 25  
**Estimated Time:** ~150-180 hours (4-5 person-weeks)  
**Timeline:** 12 weeks  

**Breakdown by Phase:**
- Phase 1 (Python): 6 issues, ~25 hours
- Phase 2 (YAML): 3 issues, ~12 hours
- Phase 3 (Migration): 3 issues, ~22 hours
- Phase 4 (Docs): 4 issues, ~35 hours
- Phase 5 (Testing): 3 issues, ~30 hours
- Release: 6 issues, ~20 hours

**Priority Order:**
1. Issues #1-6 (Phase 1 - blocking)
2. Issues #7-9 (Phase 2 - blocking)
3. Issues #10-12 (Phase 3 - critical for users)
4. Issues #13-16 (Phase 4 - documentation)
5. Issues #17-19 (Phase 5 - validation)
6. Issues #20-25 (Release preparation)

