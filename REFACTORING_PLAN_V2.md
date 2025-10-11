# IGN LiDAR HD v2.3.0 - Refactoring & Optimization Plan

**Date:** October 11, 2025  
**Current Version:** 2.2.2  
**Target Version:** 2.3.0

---

## Executive Summary

This plan addresses:

1. **✅ THREE PROCESSING MODES** - Clear, explicit processing modes
2. **✅ CUSTOM CONFIG FILE SUPPORT** - Load YAML configs from any path
3. **🔧 AUGMENTATION REFACTORING** - Simplify and consolidate augmentation
4. **🔧 PIPELINE VERIFICATION** - Ensure all modes work correctly
5. **🔧 CODE CONSOLIDATION** - Remove redundancy and improve maintainability

---

## Part 1: Processing Modes Refactoring

### Current State: Implicit Modes ⚠️

Currently, processing modes are controlled by two flags:

- `save_enriched_laz: bool` - Save enriched LAZ files
- `only_enriched_laz: bool` - Skip patch creation

This creates **implicit behavior** that's confusing:

```python
# Mode 1: Patches only (default)
save_enriched_laz=False, only_enriched_laz=False

# Mode 2: Patches + Enriched LAZ
save_enriched_laz=True, only_enriched_laz=False

# Mode 3: Enriched LAZ only
save_enriched_laz=True, only_enriched_laz=True  # Redundant!
```

### Proposed: Explicit Processing Modes ✨

**Add an enum-based mode parameter:**

```python
from enum import Enum

class ProcessingMode(Enum):
    """Processing modes for LiDAR tile processing."""
    PATCHES_ONLY = "patches_only"           # Mode 1: Create patches (default)
    PATCHES_AND_ENRICHED = "both"           # Mode 2: Patches + enriched LAZ
    ENRICHED_ONLY = "enriched_only"         # Mode 3: Only enriched LAZ
```

### Implementation

#### File: `ign_lidar/core/processor.py`

Add at top of file:

```python
from enum import Enum
from typing import Literal

# Processing modes
ProcessingMode = Literal["patches_only", "both", "enriched_only"]
```

Modify `__init__` signature:

```python
def __init__(
    self,
    lod_level: str = 'LOD2',
    processing_mode: ProcessingMode = "patches_only",  # NEW
    # ... other params
    save_enriched_laz: bool = None,  # DEPRECATED - use processing_mode
    only_enriched_laz: bool = None,  # DEPRECATED - use processing_mode
    # ... rest
):
    """
    Initialize processor.

    Args:
        lod_level: 'LOD2' or 'LOD3'
        processing_mode: Processing mode:
            - "patches_only": Create patches only (default, fastest)
            - "both": Create patches AND enriched LAZ files
            - "enriched_only": Only create enriched LAZ (no patches)
        save_enriched_laz: DEPRECATED - use processing_mode instead
        only_enriched_laz: DEPRECATED - use processing_mode instead
    """

    # Handle backward compatibility
    if save_enriched_laz is not None or only_enriched_laz is not None:
        logger.warning(
            "save_enriched_laz and only_enriched_laz are deprecated. "
            "Use processing_mode parameter instead."
        )
        # Map old flags to new mode
        if only_enriched_laz:
            processing_mode = "enriched_only"
        elif save_enriched_laz:
            processing_mode = "both"
        else:
            processing_mode = "patches_only"

    # Set processing mode
    self.processing_mode = processing_mode

    # Derive flags from mode (for internal use)
    self.save_enriched_laz = processing_mode in ["both", "enriched_only"]
    self.only_enriched_laz = processing_mode == "enriched_only"

    # Log mode
    logger.info(f"Processing mode: {processing_mode}")
```

#### File: `ign_lidar/config/schema.py`

Update OutputConfig:

```python
@dataclass
class OutputConfig:
    """
    Configuration for output formats and saving.

    Attributes:
        format: Output format for patches ('npz', 'hdf5', 'torch', 'laz', 'all')
        processing_mode: Processing mode:
            - "patches_only": Create patches only (default)
            - "both": Create patches + enriched LAZ files
            - "enriched_only": Only create enriched LAZ files (no patches)
        save_stats: Save processing statistics
        save_metadata: Save patch metadata
        compression: Compression level (0-9, None for no compression)

        # Deprecated fields (backward compatibility)
        save_enriched_laz: Use processing_mode instead
        only_enriched_laz: Use processing_mode instead
    """
    format: Literal["npz", "hdf5", "torch", "laz", "all"] = "npz"

    # NEW: Explicit processing mode
    processing_mode: Literal["patches_only", "both", "enriched_only"] = "patches_only"

    # Standard output options
    save_stats: bool = True
    save_metadata: bool = True
    compression: Optional[int] = None

    # DEPRECATED: Backward compatibility
    save_enriched_laz: Optional[bool] = None
    only_enriched_laz: Optional[bool] = None

    def __post_init__(self):
        """Handle backward compatibility and validation."""
        # Handle deprecated flags
        if self.save_enriched_laz is not None or self.only_enriched_laz is not None:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "save_enriched_laz and only_enriched_laz are deprecated. "
                "Use processing_mode instead."
            )

            # Map old flags to new mode
            if self.only_enriched_laz:
                self.processing_mode = "enriched_only"
            elif self.save_enriched_laz:
                self.processing_mode = "both"
            else:
                self.processing_mode = "patches_only"
```

#### YAML Configurations

Update all config files to use new mode:

**Old way (still works):**

```yaml
output:
  format: npz
  save_enriched_laz: true
  only_enriched_laz: true
```

**New way (recommended):**

```yaml
output:
  format: npz
  processing_mode: enriched_only # or "both" or "patches_only"
```

---

## Part 2: Custom Config File Support

### Implementation

#### File: `ign_lidar/cli/commands/process.py`

Add new function:

```python
def load_config_from_file(
    config_file: Optional[str] = None,
    overrides: Optional[list] = None
) -> DictConfig:
    """
    Load configuration from custom file or package defaults.

    Args:
        config_file: Path to custom YAML file (optional)
        overrides: CLI overrides to apply

    Returns:
        Composed Hydra configuration
    """
    if config_file:
        # Load from custom file
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        import yaml
        with open(config_path, 'r') as f:
            custom_cfg = yaml.safe_load(f)

        # Get package config dir for defaults
        package_config_dir = get_config_dir()

        # Clear Hydra
        GlobalHydra.instance().clear()

        # Initialize with package defaults
        with initialize_config_dir(config_dir=package_config_dir, version_base=None):
            # Start with base config
            cfg = compose(config_name="config")

            # Merge custom config
            custom_omega = OmegaConf.create(custom_cfg)
            cfg = OmegaConf.merge(cfg, custom_omega)

            # Apply CLI overrides (highest priority)
            if overrides:
                override_cfg = OmegaConf.from_dotlist(overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)

            return cfg
    else:
        # Use package configs with overrides
        return load_hydra_config(overrides)
```

Update command:

```python
@click.command()
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Path to custom YAML config file')
@click.option('--show-config', is_flag=True,
              help='Show composed configuration and exit')
@click.argument('overrides', nargs=-1)
def process_command(config_file, show_config, overrides):
    """
    Process LiDAR tiles.

    Examples:
        # Use package defaults
        ign-lidar-hd process input_dir=data/raw output_dir=data/out

        # Use custom config
        ign-lidar-hd process --config-file my_config.yaml

        # Custom config + overrides
        ign-lidar-hd process -c my_config.yaml processor.use_gpu=true

        # Show config without processing
        ign-lidar-hd process -c my_config.yaml --show-config
    """
    try:
        cfg = load_config_from_file(config_file, list(overrides))

        if show_config:
            click.echo("=" * 70)
            click.echo("Composed Configuration:")
            click.echo("=" * 70)
            click.echo(OmegaConf.to_yaml(cfg))
            return

        process_lidar(cfg)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.ClickException(str(e))
```

---

## Part 3: Augmentation Refactoring

### Current State: Scattered Augmentation Logic

**Problem:** Augmentation code is spread across multiple locations:

1. `preprocessing/utils.py::augment_raw_points()` - Point-level augmentation
2. `preprocessing/utils.py::augment_patch()` - Patch-level augmentation
3. `datasets/augmentation.py::PatchAugmentation` - Class-based augmentation
4. Inline augmentation in `processor.py`

### Proposed: Unified Augmentation System

**Consolidate into a single, clear module:**

```
ign_lidar/
  augmentation/              # NEW module
    __init__.py
    core.py                  # Core augmentation functions
    transforms.py            # Individual transforms
    pipeline.py              # Augmentation pipeline
```

#### File: `ign_lidar/augmentation/core.py`

```python
"""
Core augmentation functions for IGN LiDAR HD.

Unified augmentation system with clear interfaces.
"""

import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class AugmentationConfig:
    """Configuration for augmentation pipeline."""

    def __init__(
        self,
        rotation: bool = True,
        rotation_max_angle: float = 180.0,
        scaling: bool = True,
        scaling_range: tuple = (0.9, 1.1),
        jitter: bool = True,
        jitter_sigma: float = 0.01,
        jitter_clip: float = 0.05,
        translation: bool = False,
        translation_max: float = 0.5,
        dropout: bool = False,
        dropout_ratio: float = 0.1,
        feature_noise: bool = False,
        feature_noise_sigma: float = 0.01
    ):
        """
        Initialize augmentation configuration.

        Args:
            rotation: Enable random rotation around Z axis
            rotation_max_angle: Max rotation angle (degrees)
            scaling: Enable random scaling
            scaling_range: (min_scale, max_scale)
            jitter: Enable random jitter (Gaussian noise)
            jitter_sigma: Standard deviation for jitter
            jitter_clip: Clip jitter to this range
            translation: Enable random translation
            translation_max: Max translation offset (meters)
            dropout: Enable random point dropout
            dropout_ratio: Ratio of points to drop (0-1)
            feature_noise: Enable feature noise
            feature_noise_sigma: Std dev for feature noise
        """
        self.rotation = rotation
        self.rotation_max_angle = rotation_max_angle
        self.scaling = scaling
        self.scaling_range = scaling_range
        self.jitter = jitter
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.translation = translation
        self.translation_max = translation_max
        self.dropout = dropout
        self.dropout_ratio = dropout_ratio
        self.feature_noise = feature_noise
        self.feature_noise_sigma = feature_noise_sigma

    @classmethod
    def default(cls):
        """Default augmentation config (conservative)."""
        return cls(
            rotation=True,
            scaling=True,
            jitter=True,
            translation=False,
            dropout=False,
            feature_noise=False
        )

    @classmethod
    def aggressive(cls):
        """Aggressive augmentation config (more variety)."""
        return cls(
            rotation=True,
            rotation_max_angle=180.0,
            scaling=True,
            scaling_range=(0.8, 1.2),
            jitter=True,
            jitter_sigma=0.02,
            translation=True,
            translation_max=1.0,
            dropout=True,
            dropout_ratio=0.1,
            feature_noise=True
        )


def augment_patch(
    patch: Dict[str, np.ndarray],
    config: Optional[AugmentationConfig] = None
) -> Dict[str, np.ndarray]:
    """
    Apply augmentation to a patch.

    This is the MAIN augmentation function that should be used
    throughout the codebase.

    Args:
        patch: Patch dictionary with 'points' and optional features
        config: Augmentation configuration (None = defaults)

    Returns:
        Augmented patch

    Example:
        >>> config = AugmentationConfig(rotation=True, scaling=True)
        >>> augmented = augment_patch(patch, config)
    """
    if config is None:
        config = AugmentationConfig.default()

    # Copy to avoid modifying original
    aug_patch = {k: v.copy() if isinstance(v, np.ndarray) else v
                 for k, v in patch.items()}

    if 'points' not in aug_patch:
        logger.warning("No 'points' in patch, skipping augmentation")
        return aug_patch

    points = aug_patch['points']

    # Apply geometric transformations
    if config.rotation:
        points = _rotate_z(points, config.rotation_max_angle)

    if config.scaling:
        points = _scale(points, config.scaling_range)

    if config.translation:
        points = _translate(points, config.translation_max)

    if config.jitter:
        points = _jitter(points, config.jitter_sigma, config.jitter_clip)

    # Apply dropout (affects all arrays)
    if config.dropout:
        points, mask = _dropout(points, config.dropout_ratio)
        aug_patch = _apply_mask(aug_patch, mask)

    aug_patch['points'] = points

    # Feature noise
    if config.feature_noise and 'features' in aug_patch:
        aug_patch['features'] = _add_noise(
            aug_patch['features'],
            config.feature_noise_sigma
        )

    return aug_patch


# ============================================================================
# Individual Transform Functions (private)
# ============================================================================

def _rotate_z(points: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotate around Z axis."""
    angle = np.random.uniform(-max_angle, max_angle)
    angle_rad = np.deg2rad(angle)

    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return points @ R.T


def _scale(points: np.ndarray, scale_range: tuple) -> np.ndarray:
    """Uniform scaling."""
    scale = np.random.uniform(*scale_range)
    return points * scale


def _translate(points: np.ndarray, max_offset: float) -> np.ndarray:
    """Random translation."""
    offset = np.random.uniform(-max_offset, max_offset, size=(3,)).astype(np.float32)
    return points + offset


def _jitter(points: np.ndarray, sigma: float, clip: float) -> np.ndarray:
    """Gaussian noise."""
    noise = np.random.normal(0, sigma, size=points.shape).astype(np.float32)
    noise = np.clip(noise, -clip, clip)
    return points + noise


def _dropout(points: np.ndarray, ratio: float) -> tuple:
    """Random point dropout."""
    n = len(points)
    n_keep = int(n * (1 - ratio))
    indices = np.random.choice(n, size=n_keep, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    return points[mask], mask


def _apply_mask(patch: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Apply mask to all arrays in patch."""
    for key, value in patch.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            patch[key] = value[mask]
    return patch


def _add_noise(features: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to features."""
    noise = np.random.normal(0, sigma, size=features.shape).astype(np.float32)
    return features + noise
```

#### Deprecation of Old Functions

In `preprocessing/utils.py`, add deprecation warnings:

```python
def augment_raw_points(...):
    """DEPRECATED: Use augmentation.core.augment_patch instead."""
    import warnings
    warnings.warn(
        "augment_raw_points is deprecated. "
        "Use ign_lidar.augmentation.core.augment_patch instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... keep for backward compatibility
```

#### Update Processor to Use New Augmentation

In `processor.py`:

```python
from ..augmentation.core import augment_patch, AugmentationConfig

# In __init__:
if self.augment:
    self.augmentation_config = AugmentationConfig.default()
else:
    self.augmentation_config = None

# When augmenting:
if self.augment:
    augmented_patch = augment_patch(patch, self.augmentation_config)
```

---

## Part 4: Pipeline Verification

### Verification Script

Create `tests/test_processing_modes.py`:

```python
"""
Test all three processing modes to ensure they work correctly.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from ign_lidar.core.processor import LiDARProcessor


@pytest.fixture
def test_data_dir():
    """Fixture providing test LAZ file."""
    # Assume we have a small test LAZ file
    test_dir = Path(__file__).parent / "test_data"
    if not test_dir.exists():
        pytest.skip("Test data not available")
    return test_dir


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_mode_patches_only(test_data_dir, output_dir):
    """Test Mode 1: Patches only."""
    processor = LiDARProcessor(
        lod_level='LOD2',
        processing_mode='patches_only',
        num_points=4096,
        patch_size=100.0
    )

    laz_file = list(test_data_dir.glob("*.laz"))[0]
    num_patches = processor.process_tile(laz_file, output_dir)

    # Should create patches
    assert num_patches > 0
    patches = list(output_dir.glob("*.npz"))
    assert len(patches) > 0

    # Should NOT create enriched LAZ
    enriched_files = list(output_dir.glob("*_enriched.laz"))
    assert len(enriched_files) == 0


def test_mode_both(test_data_dir, output_dir):
    """Test Mode 2: Patches + Enriched LAZ."""
    processor = LiDARProcessor(
        lod_level='LOD2',
        processing_mode='both',
        num_points=4096,
        patch_size=100.0
    )

    laz_file = list(test_data_dir.glob("*.laz"))[0]
    num_patches = processor.process_tile(laz_file, output_dir)

    # Should create patches
    assert num_patches > 0
    patches = list(output_dir.glob("*.npz"))
    assert len(patches) > 0

    # Should ALSO create enriched LAZ
    enriched_files = list(output_dir.glob("*_enriched.laz"))
    assert len(enriched_files) == 1


def test_mode_enriched_only(test_data_dir, output_dir):
    """Test Mode 3: Enriched LAZ only."""
    processor = LiDARProcessor(
        lod_level='LOD2',
        processing_mode='enriched_only',
        num_points=4096,
        patch_size=100.0
    )

    laz_file = list(test_data_dir.glob("*.laz"))[0]
    num_patches = processor.process_tile(laz_file, output_dir)

    # Should NOT create patches
    assert num_patches == 0
    patches = list(output_dir.glob("*.npz"))
    assert len(patches) == 0

    # Should create enriched LAZ
    enriched_files = list(output_dir.glob("*_enriched.laz"))
    assert len(enriched_files) == 1


def test_backward_compatibility(test_data_dir, output_dir):
    """Test old flags still work."""
    # Old way: save_enriched_laz=True, only_enriched_laz=False
    processor = LiDARProcessor(
        lod_level='LOD2',
        save_enriched_laz=True,
        only_enriched_laz=False,
        num_points=4096
    )

    # Should map to "both" mode
    assert processor.processing_mode == "both"

    # Old way: only_enriched_laz=True
    processor2 = LiDARProcessor(
        lod_level='LOD2',
        only_enriched_laz=True,
        num_points=4096
    )

    # Should map to "enriched_only" mode
    assert processor2.processing_mode == "enriched_only"


def test_augmentation(test_data_dir, output_dir):
    """Test augmentation pipeline."""
    from ign_lidar.augmentation.core import augment_patch, AugmentationConfig

    # Create test patch
    import numpy as np
    patch = {
        'points': np.random.randn(1000, 3).astype(np.float32),
        'features': np.random.randn(1000, 10).astype(np.float32)
    }

    # Test default config
    config = AugmentationConfig.default()
    augmented = augment_patch(patch, config)

    assert 'points' in augmented
    assert augmented['points'].shape == (1000, 3)

    # Points should be different (augmented)
    assert not np.allclose(patch['points'], augmented['points'])
```

---

## Part 5: Documentation Updates

### User Guide

Create `docs/docs/guides/processing-modes.md`:

```markdown
# Processing Modes

IGN LiDAR HD supports three processing modes for different use cases.

## Mode 1: Patches Only (Default)

**Use case:** Training deep learning models

Creates ML-ready patches suitable for PointNet++, transformers, etc.

**Command:**
\`\`\`bash
ign-lidar-hd process \\
input_dir=data/raw \\
output_dir=data/patches \\
output.processing_mode=patches_only
\`\`\`

**Output:**

- `tile_patch_0001.npz`
- `tile_patch_0002.npz`
- ...

## Mode 2: Patches + Enriched LAZ

**Use case:** Training models + GIS analysis

Creates both ML patches and feature-enriched LAZ files for QGIS.

**Command:**
\`\`\`bash
ign-lidar-hd process \\
input_dir=data/raw \\
output_dir=data/out \\
output.processing_mode=both
\`\`\`

**Output:**

- `tile_patch_0001.npz` (ML patches)
- `tile_enriched.laz` (QGIS-ready)

## Mode 3: Enriched LAZ Only

**Use case:** GIS analysis only (fastest)

Only creates enriched LAZ files with computed features.

**Command:**
\`\`\`bash
ign-lidar-hd process \\
input_dir=data/raw \\
output_dir=data/enriched \\
output.processing_mode=enriched_only
\`\`\`

**Output:**

- `tile_enriched.laz` (with normals, curvature, RGB, etc.)

## Comparison

| Mode            | Patches | Enriched LAZ | Speed   | Use Case    |
| --------------- | ------- | ------------ | ------- | ----------- |
| `patches_only`  | ✅      | ❌           | Fast    | ML training |
| `both`          | ✅      | ✅           | Slower  | ML + GIS    |
| `enriched_only` | ❌      | ✅           | Fastest | GIS only    |
```

---

## Implementation Checklist

### Phase 1: Processing Modes (Week 1)

- [ ] Add `ProcessingMode` type to `processor.py`
- [ ] Update `LiDARProcessor.__init__()` with mode parameter
- [ ] Add backward compatibility for old flags
- [ ] Update `OutputConfig` in `schema.py`
- [ ] Update all YAML configs to use new mode
- [ ] Test all three modes manually
- [ ] Write automated tests

### Phase 2: Custom Config (Week 1)

- [ ] Add `load_config_from_file()` function
- [ ] Update `process_command()` with `--config-file` option
- [ ] Add `--show-config` option
- [ ] Test with custom YAML files
- [ ] Document config precedence

### Phase 3: Augmentation (Week 2)

- [ ] Create `ign_lidar/augmentation/` module
- [ ] Implement `core.py` with unified functions
- [ ] Add `AugmentationConfig` class
- [ ] Update `processor.py` to use new augmentation
- [ ] Add deprecation warnings to old functions
- [ ] Test augmentation pipeline
- [ ] Update documentation

### Phase 4: Testing & Docs (Week 2)

- [ ] Create `test_processing_modes.py`
- [ ] Create `test_augmentation.py`
- [ ] Run full integration tests
- [ ] Write user guide for processing modes
- [ ] Update API documentation
- [ ] Update examples

### Phase 5: Cleanup (Week 3)

- [ ] Archive deprecated functions (keep for compatibility)
- [ ] Update CHANGELOG.md
- [ ] Bump version to 2.3.0
- [ ] Create migration guide for v2.2.x → v2.3.0

---

## Migration Guide (v2.2.x → v2.3.0)

### For Users

**Old way (still works):**

```yaml
output:
  save_enriched_laz: true
  only_enriched_laz: false
```

**New way (recommended):**

```yaml
output:
  processing_mode: both
```

### For Developers

**Old augmentation:**

```python
from ign_lidar.preprocessing.utils import augment_patch
augmented = augment_patch(patch)
```

**New augmentation:**

```python
from ign_lidar.augmentation.core import augment_patch, AugmentationConfig

config = AugmentationConfig(rotation=True, scaling=True)
augmented = augment_patch(patch, config)
```

---

## Release Notes Draft (v2.3.0)

### ✨ New Features

**Explicit Processing Modes**

- Added `processing_mode` parameter with three clear modes:
  - `patches_only`: ML patches only (default)
  - `both`: Patches + enriched LAZ
  - `enriched_only`: Enriched LAZ only
- Old `save_enriched_laz`/`only_enriched_laz` still supported (deprecated)

**Custom Config File Support**

- New `--config-file` option to load configs from any path
- Clear configuration precedence: defaults < file < overrides
- New `--show-config` to preview composed configuration

**Unified Augmentation System**

- New `ign_lidar.augmentation` module with clean API
- `AugmentationConfig` class for easy configuration
- Default and aggressive presets
- Backward compatible

### 🔧 Improvements

- Better error messages for mode configuration
- Clearer logging of processing mode
- Simplified codebase with reduced redundancy

### ⚠️ Deprecations

- `save_enriched_laz` → use `processing_mode` instead
- `only_enriched_laz` → use `processing_mode` instead
- `preprocessing.utils.augment_*` → use `augmentation.core` instead

### 📚 Documentation

- New processing modes guide
- Custom config examples
- Updated API reference
- Migration guide from v2.2.x

---

**End of Refactoring Plan**
