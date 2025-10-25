# IGN LiDAR HD - Harmonization & Simplification Implementation Plan

**Status:** Ready for Implementation  
**Target Version:** v3.2.0  
**Timeline:** 2-3 weeks  
**Related:** [CLASSIFICATION_CONFIG_AUDIT.md](../CLASSIFICATION_CONFIG_AUDIT.md)

---

## Overview

This document outlines the concrete implementation steps for harmonizing the classification module and simplifying the configuration system based on the comprehensive audit.

## Goals

- ✅ **54% code reduction** (6,355 → 2,900 LOC)
- ✅ **87% reduction in config complexity** (118 → 15 params)
- ✅ **Unified classifier API** (5 → 1 interface)
- ✅ **Improved user experience** (clearer docs, fewer choices)
- ✅ **Maintain backward compatibility** (gradual migration)

---

## Phase 1: Configuration System Simplification (Week 1)

### Step 1.1: Create Unified Config Schema

**File:** `ign_lidar/config/config.py` (new)

```python
"""
Unified Configuration Schema for IGN LiDAR HD v3.2+

This replaces both schema.py and schema_simplified.py with a single,
intuitive configuration system.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any
from pathlib import Path

@dataclass
class Config:
    """
    Main configuration for IGN LiDAR HD processing.

    Quick Start:
        >>> config = Config.preset('asprs_production')
        >>> config.input_dir = '/data/tiles'
        >>> config.output_dir = '/data/output'

    Expert Mode:
        >>> config = Config(
        ...     mode='lod2',
        ...     use_gpu=True,
        ...     advanced=AdvancedConfig(...)
        ... )
    """

    # ===========================================
    # ESSENTIAL PARAMETERS (Top-level)
    # Most users only need to set these 8 params
    # ===========================================

    # Paths (required)
    input_dir: str = field(metadata={'required': True})
    output_dir: str = field(metadata={'required': True})

    # Processing mode
    mode: Literal['asprs', 'lod2', 'lod3'] = 'lod2'

    # Hardware
    use_gpu: bool = False
    num_workers: int = 4

    # Patch configuration
    patch_size: float = 150.0
    num_points: int = 16384

    # Output mode
    processing_mode: Literal['patches_only', 'both', 'enriched_only'] = 'patches_only'

    # ===========================================
    # COMMON PARAMETERS (Feature-level)
    # Frequently adjusted by users
    # ===========================================

    features: 'FeatureConfig' = field(default_factory=lambda: FeatureConfig())

    # ===========================================
    # OPTIONAL PARAMETERS (Nested)
    # Rarely modified - for experts only
    # ===========================================

    advanced: Optional['AdvancedConfig'] = None

    @classmethod
    def preset(cls, name: str, **overrides) -> 'Config':
        """
        Load a preset configuration.

        Available presets:
        - 'asprs_production': ASPRS classification for production
        - 'lod2_buildings': LOD2 building detection
        - 'lod3_detailed': LOD3 detailed classification
        - 'gpu_optimized': GPU-accelerated processing
        - 'minimal_fast': Minimal features for quick processing

        Args:
            name: Preset name
            **overrides: Override any preset parameters

        Returns:
            Config instance with preset values

        Example:
            >>> config = Config.preset('asprs_production',
            ...                        num_workers=8,
            ...                        use_gpu=True)
        """
        presets = _load_presets()
        if name not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")

        preset_config = presets[name]
        preset_config.update(overrides)
        return cls(**preset_config)

    @classmethod
    def from_environment(cls, input_dir: str, output_dir: str, **overrides) -> 'Config':
        """
        Auto-configure based on system environment.

        Auto-detects:
        - GPU availability
        - CPU count
        - Available memory
        - Input data characteristics

        Args:
            input_dir: Input tile directory
            output_dir: Output directory
            **overrides: Override any auto-detected parameters

        Returns:
            Config instance with auto-detected values
        """
        import os
        from ign_lidar.core.gpu_context import GPU_AVAILABLE

        auto_config = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'use_gpu': GPU_AVAILABLE,
            'num_workers': min(os.cpu_count() or 4, 16),
        }

        auto_config.update(overrides)
        return cls(**auto_config)


@dataclass
class FeatureConfig:
    """
    Feature computation configuration.

    This replaces the old FeaturesConfig with simpler options.
    """

    # Feature set selection (replaces 'mode' + 50 individual flags)
    feature_set: Literal['minimal', 'standard', 'full'] = 'standard'

    # Geometric features
    k_neighbors: int = 30
    search_radius: Optional[float] = None  # Auto-calculated if None

    # Spectral features
    use_rgb: bool = False
    use_nir: bool = False
    compute_ndvi: bool = False

    # Multi-scale (simplified from 20+ params to 3)
    multi_scale: bool = False
    scales: Optional[List[str]] = None  # e.g., ['fine', 'medium', 'coarse']

    @property
    def feature_list(self) -> List[str]:
        """Get actual feature list based on feature_set."""
        from ign_lidar.features import get_feature_list_for_mode
        return get_feature_list_for_mode(self.feature_set)


@dataclass
class AdvancedConfig:
    """
    Advanced configuration for expert users.

    Most users should not need to touch these parameters.
    Provides fine-grained control over processing behavior.
    """

    # Classification
    classification: Optional['ClassificationConfig'] = None

    # Ground truth
    ground_truth: Optional['GroundTruthConfig'] = None

    # Preprocessing
    preprocessing: Optional['PreprocessConfig'] = None

    # Performance tuning
    performance: Optional['PerformanceConfig'] = None

    # Reclassification
    reclassification: Optional['ReclassificationConfig'] = None


# ... (Additional nested configs for advanced users)

def _load_presets() -> Dict[str, Dict[str, Any]]:
    """Load preset configurations."""
    return {
        'asprs_production': {
            'mode': 'asprs',
            'features': FeatureConfig(feature_set='standard', k_neighbors=30),
            'processing_mode': 'both',
        },
        'lod2_buildings': {
            'mode': 'lod2',
            'features': FeatureConfig(feature_set='standard', k_neighbors=30),
            'processing_mode': 'patches_only',
        },
        'lod3_detailed': {
            'mode': 'lod3',
            'features': FeatureConfig(feature_set='full', k_neighbors=40),
            'processing_mode': 'both',
        },
        'gpu_optimized': {
            'use_gpu': True,
            'num_workers': 1,
            'features': FeatureConfig(feature_set='standard'),
        },
        'minimal_fast': {
            'mode': 'asprs',
            'features': FeatureConfig(feature_set='minimal', k_neighbors=20),
            'processing_mode': 'patches_only',
        },
    }
```

### Step 1.2: Deprecate Old Schemas

**File:** `ign_lidar/config/schema.py`

Add deprecation warning at top:

```python
"""
DEPRECATED: This module is deprecated in favor of ign_lidar.config.config

Migration:
    # Old
    from ign_lidar.config.schema import ProcessorConfig, FeaturesConfig

    # New
    from ign_lidar.config import Config

Please see MIGRATION_v3.1_to_v3.2.md for details.

This module will be removed in v4.0.0.
"""

import warnings
warnings.warn(
    "ign_lidar.config.schema is deprecated. Use ign_lidar.config.config instead.",
    DeprecationWarning,
    stacklevel=2
)

# Keep old classes for backward compatibility
from .schema_legacy import ProcessorConfig, FeaturesConfig, PreprocessConfig
```

### Step 1.3: Update Config Loader

**File:** `ign_lidar/config/__init__.py`

```python
"""Configuration system for IGN LiDAR HD."""

# New unified configuration (v3.2+)
from .config import Config, FeatureConfig, AdvancedConfig

# Legacy imports (deprecated, will be removed in v4.0)
try:
    from .schema import ProcessorConfig, FeaturesConfig, PreprocessConfig
except ImportError:
    pass

__all__ = [
    'Config',
    'FeatureConfig',
    'AdvancedConfig',
]
```

### Step 1.4: Create Migration Tool

**File:** `ign_lidar/cli/commands/migrate_config.py`

```python
"""
Command-line tool to migrate old configs to new format.

Usage:
    ign-lidar migrate-config old_config.yaml --output new_config.yaml
"""

import click
import yaml
from pathlib import Path

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--dry-run', is_flag=True, help='Show changes without writing')
def migrate_config(input_file, output, dry_run):
    """Migrate old configuration format to v3.2 format."""

    # Load old config
    with open(input_file) as f:
        old_config = yaml.safe_load(f)

    # Convert to new format
    new_config = _convert_config(old_config)

    # Display changes
    click.echo("Configuration migration:")
    click.echo(f"  Old: {len(_flatten_dict(old_config))} parameters")
    click.echo(f"  New: {len(_flatten_dict(new_config))} parameters")
    click.echo(f"  Simplified: {_calculate_simplification(old_config, new_config)}%")

    if dry_run:
        click.echo("\nNew configuration (dry-run):")
        click.echo(yaml.dump(new_config, default_flow_style=False))
    else:
        output_path = Path(output or input_file.replace('.yaml', '_v3.2.yaml'))
        with open(output_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        click.echo(f"\n✅ Migrated configuration written to: {output_path}")

def _convert_config(old_config: dict) -> dict:
    """Convert old config format to new format."""
    new_config = {
        'input_dir': old_config.get('input_dir'),
        'output_dir': old_config.get('output_dir'),
        'mode': old_config.get('processor', {}).get('lod_level', 'lod2').lower(),
        'use_gpu': old_config.get('processor', {}).get('use_gpu', False),
        'num_workers': old_config.get('processor', {}).get('num_workers', 4),
    }

    # Convert features section
    old_features = old_config.get('features', {})
    new_config['features'] = {
        'feature_set': _map_feature_mode(old_features.get('mode', 'full')),
        'k_neighbors': old_features.get('k_neighbors', 30),
        'use_rgb': old_features.get('use_rgb', False),
        'use_nir': old_features.get('use_infrared', False),
        'compute_ndvi': old_features.get('compute_ndvi', False),
    }

    # Preserve advanced settings if present
    if _has_advanced_settings(old_config):
        new_config['advanced'] = _extract_advanced_settings(old_config)

    return new_config

def _map_feature_mode(old_mode: str) -> str:
    """Map old feature mode to new feature_set."""
    mapping = {
        'minimal': 'minimal',
        'full': 'full',
        'lod2': 'standard',
        'lod3': 'full',
        'asprs_classes': 'standard',
        'custom': 'standard',
    }
    return mapping.get(old_mode, 'standard')
```

---

## Phase 2: Classifier Interface Harmonization (Week 2)

### Step 2.1: Create Base Classifier

**File:** `ign_lidar/core/classification/base.py` (new)

```python
"""
Base classifier interface for all classification modules.

This establishes a unified API that all classifiers must follow,
ensuring consistency and making it easier to swap implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import numpy as np
import geopandas as gpd

@dataclass
class ClassificationResult:
    """
    Unified result object returned by all classifiers.

    Attributes:
        labels: Classification labels [N]
        confidence: Confidence scores [N], range [0, 1]
        metadata: Additional information about classification
    """
    labels: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)

        stats = {
            'total_points': total,
            'num_classes': len(unique),
            'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
            'class_percentages': {
                int(c): float(cnt) / total * 100
                for c, cnt in zip(unique, counts)
            },
        }

        if self.confidence is not None:
            stats['avg_confidence'] = float(np.mean(self.confidence))
            stats['min_confidence'] = float(np.min(self.confidence))
            stats['low_confidence_count'] = int(np.sum(self.confidence < 0.5))

        return stats


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers.

    All classifiers must implement the classify() method with this signature.
    This ensures a consistent API across the codebase.
    """

    @abstractmethod
    def classify(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[Union[gpd.GeoDataFrame, Dict[str, Any]]] = None,
        **kwargs
    ) -> ClassificationResult:
        """
        Classify point cloud.

        Args:
            points: Point cloud [N, 3] with XYZ coordinates
            features: Dictionary of feature arrays
            ground_truth: Optional ground truth data (BD TOPO, cadastre, etc.)
            **kwargs: Classifier-specific parameters

        Returns:
            ClassificationResult with labels, confidence, and metadata

        Raises:
            ValueError: If input data is invalid
            ProcessingError: If classification fails
        """
        pass

    def validate_inputs(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray]
    ) -> None:
        """
        Validate input data.

        Raises:
            ValueError: If inputs are invalid
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points [N, 3], got {points.shape}")

        n_points = len(points)
        for name, feat in features.items():
            if len(feat) != n_points:
                raise ValueError(
                    f"Feature '{name}' has {len(feat)} values, "
                    f"expected {n_points} (matching points)"
                )
```

### Step 2.2: Refactor UnifiedClassifier

**File:** `ign_lidar/core/classification/unified_classifier.py`

Update to inherit from BaseClassifier:

```python
from .base import BaseClassifier, ClassificationResult

class UnifiedClassifier(BaseClassifier):
    """
    Unified classifier implementing the standard interface.

    v3.2+: Now inherits from BaseClassifier for consistent API.
    """

    def __init__(self, strategy='adaptive', **kwargs):
        """Initialize classifier."""
        self.strategy = strategy
        self.config = UnifiedClassifierConfig(**kwargs)

    def classify(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[gpd.GeoDataFrame] = None,
        **kwargs
    ) -> ClassificationResult:
        """
        Classify points using unified interface.

        This method now conforms to the BaseClassifier interface.
        """
        # Validate inputs
        self.validate_inputs(points, features)

        # Call internal implementation
        labels, confidence = self._classify_internal(points, features, ground_truth, **kwargs)

        # Return unified result
        return ClassificationResult(
            labels=labels,
            confidence=confidence,
            metadata={
                'strategy': self.strategy,
                'num_points': len(points),
            }
        )

    def _classify_internal(self, points, features, ground_truth, **kwargs):
        """Internal classification logic (unchanged)."""
        # ... existing implementation ...
        pass
```

### Step 2.3: Create Unified Facade

**File:** `ign_lidar/core/classification/__init__.py`

```python
"""
Classification module with unified interface.

v3.2+ Changes:
- Single Classifier class as main entry point
- All classifiers follow BaseClassifier interface
- Backward compatibility via aliases
"""

# New unified interface (v3.2+)
from .base import BaseClassifier, ClassificationResult
from .unified_classifier import UnifiedClassifier

# Main classifier (use this!)
Classifier = UnifiedClassifier  # Unified entry point

# Specialized classifiers (follow BaseClassifier interface)
from .hierarchical_classifier import HierarchicalClassifier
from .parcel_classifier import ParcelClassifier
from .building import AdaptiveBuildingClassifier

# Backward compatibility (deprecated)
import warnings

def _deprecated_import(old_name, new_name):
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

__all__ = [
    'Classifier',  # ← Use this!
    'ClassificationResult',
    'BaseClassifier',

    # Specialized (advanced users)
    'UnifiedClassifier',
    'HierarchicalClassifier',
    'ParcelClassifier',
    'AdaptiveBuildingClassifier',
]
```

---

## Phase 3: Testing & Documentation (Week 3)

### Step 3.1: Regression Tests

**File:** `tests/test_unified_api.py`

```python
"""
Test unified classifier API.

Ensures all classifiers follow BaseClassifier interface and
produce consistent results.
"""

import pytest
import numpy as np
from ign_lidar.core.classification import (
    Classifier,
    UnifiedClassifier,
    HierarchicalClassifier,
    ClassificationResult,
)

@pytest.fixture
def sample_data():
    """Sample point cloud and features."""
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100
    features = {
        'planarity': np.random.rand(n_points),
        'verticality': np.random.rand(n_points),
        'height': np.random.rand(n_points) * 10,
    }
    return points, features

def test_classifier_interface(sample_data):
    """Test that Classifier follows BaseClassifier interface."""
    points, features = sample_data

    classifier = Classifier()
    result = classifier.classify(points, features)

    # Check result type
    assert isinstance(result, ClassificationResult)
    assert result.labels.shape == (len(points),)

    # Check statistics
    stats = result.get_statistics()
    assert 'total_points' in stats
    assert stats['total_points'] == len(points)

def test_all_classifiers_unified_interface(sample_data):
    """Test that all classifiers follow same interface."""
    points, features = sample_data

    classifiers = [
        Classifier(),
        UnifiedClassifier(),
        HierarchicalClassifier(),
    ]

    for clf in classifiers:
        result = clf.classify(points, features)
        assert isinstance(result, ClassificationResult)
        assert len(result.labels) == len(points)

def test_config_migration():
    """Test that old configs still work."""
    # Old config format
    old_config = {
        'processor': {'lod_level': 'LOD2'},
        'features': {'mode': 'lod2'}
    }

    # Should be able to load via migration
    from ign_lidar.config import Config
    new_config = Config.from_dict(old_config)  # Auto-migrates

    assert new_config.mode == 'lod2'
```

### Step 3.2: Documentation Updates

**File:** `docs/docs/guides/unified-configuration.md` (new)

````markdown
# Unified Configuration Guide (v3.2+)

## Overview

IGN LiDAR HD v3.2 introduces a simplified configuration system that replaces
the complex dual-schema approach with a single, intuitive interface.

## Quick Start

### 1. Using Presets (Recommended)

```python
from ign_lidar import Config, LiDARProcessor

# Load a preset
config = Config.preset('asprs_production')

# Set your paths
config.input_dir = '/data/tiles'
config.output_dir = '/data/output'

# Run processing
processor = LiDARProcessor(config)
processor.process()
```
````

### 2. Auto-Configuration

```python
# Let IGN LiDAR HD detect optimal settings
config = Config.from_environment(
    input_dir='/data/tiles',
    output_dir='/data/output'
)
# Auto-detects: GPU, CPU count, memory, etc.
```

### 3. Manual Configuration

```python
config = Config(
    input_dir='/data/tiles',
    output_dir='/data/output',
    mode='lod2',
    use_gpu=True,
    num_workers=8
)
```

## Available Presets

| Preset             | Description             | Best For             |
| ------------------ | ----------------------- | -------------------- |
| `asprs_production` | ASPRS classification    | Production pipelines |
| `lod2_buildings`   | LOD2 building detection | Building modeling    |
| `lod3_detailed`    | LOD3 detailed elements  | Architectural detail |
| `gpu_optimized`    | GPU-accelerated         | Large datasets       |
| `minimal_fast`     | Minimal features        | Quick testing        |

## Migration from v3.1

See [Migration Guide](migration-v3.1-to-v3.2.md) for details.

**TL;DR:**

```python
# Old (v3.1)
from ign_lidar.config.schema import ProcessorConfig, FeaturesConfig

processor_config = ProcessorConfig(lod_level='LOD2')
features_config = FeaturesConfig(mode='lod2')

# New (v3.2)
from ign_lidar import Config

config = Config.preset('lod2_buildings')
```

## Advanced Usage

For expert users who need fine-grained control:

```python
from ign_lidar import Config, AdvancedConfig

config = Config(
    mode='lod3',
    advanced=AdvancedConfig(
        classification=ClassificationConfig(...),
        performance=PerformanceConfig(...),
    )
)
```

```

---

## Timeline & Milestones

### Week 1: Configuration (Days 1-7)

- [ ] **Day 1-2:** Create `config.py` with unified Config class
- [ ] **Day 3:** Implement preset system
- [ ] **Day 4:** Create migration tool (`migrate-config` command)
- [ ] **Day 5:** Update `__init__.py` exports
- [ ] **Day 6:** Add deprecation warnings to old schemas
- [ ] **Day 7:** Test config loading and validation

### Week 2: Classifiers (Days 8-14)

- [ ] **Day 8:** Create `base.py` with BaseClassifier
- [ ] **Day 9-10:** Refactor UnifiedClassifier to inherit from BaseClassifier
- [ ] **Day 11:** Refactor HierarchicalClassifier
- [ ] **Day 12:** Refactor ParcelClassifier
- [ ] **Day 13:** Update building classifiers
- [ ] **Day 14:** Create Classifier facade

### Week 3: Testing & Docs (Days 15-21)

- [ ] **Day 15-16:** Write regression tests
- [ ] **Day 17-18:** Update documentation
- [ ] **Day 19:** Create migration guide
- [ ] **Day 20:** Code review and fixes
- [ ] **Day 21:** Release v3.2.0

---

## Success Criteria

### Code Metrics

- [ ] Configuration LOC reduced from 755 to < 300
- [ ] All classifiers inherit from BaseClassifier
- [ ] No regression in test coverage (maintain 85%+)
- [ ] All existing tests pass

### User Experience

- [ ] New user can run processing in < 5 lines of code
- [ ] Config errors have clear, actionable messages
- [ ] Documentation has < 3 getting started examples (not 10+)
- [ ] Migration tool successfully converts all example configs

### Performance

- [ ] No performance regression (< 2% slower acceptable)
- [ ] Startup time improved by 20%+ (fewer imports)
- [ ] Memory usage unchanged or improved

---

## Rollback Plan

If critical issues discovered:

1. **Immediate:** Revert the release commit
2. **Quick:** Release v3.1.1 with old behavior
3. **Communication:** GitHub issue explaining rollback
4. **Analysis:** Review what went wrong, create fix plan
5. **Retry:** Re-release as v3.2.1 after fixes

---

## Questions & Decisions

### Q1: Should we remove old schemas immediately or keep for v3.x?

**Decision:** Keep with deprecation warnings for v3.2-3.9, remove in v4.0.

**Rationale:** Gives users time to migrate without breaking existing code.

### Q2: Should presets be in YAML files or Python code?

**Decision:** Python code (in `config.py`).

**Rationale:** Easier to maintain, type-safe, can include logic.

### Q3: How to handle advanced users who need full control?

**Decision:** `AdvancedConfig` dataclass with nested configs.

**Rationale:** Keeps simple cases simple, complex cases possible.

---

## Related Documents

- [CLASSIFICATION_CONFIG_AUDIT.md](../CLASSIFICATION_CONFIG_AUDIT.md) - Full audit findings
- [MIGRATION_v3.1_to_v3.2.md](migration-v3.1-to-v3.2.md) - Migration guide (to be created)
- [API_REFERENCE.md](api-reference.md) - API documentation (to be updated)

---

**Status:** Ready for Implementation
**Approved by:** (Pending review)
**Start Date:** TBD
```
