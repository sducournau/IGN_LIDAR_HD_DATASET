# Implementation Plan: Custom Config File Support

**Priority:** CRITICAL  
**Estimated Effort:** 1-2 days  
**Impact:** Enables users to load custom YAML config files

---

## 1. Enhanced Process Command

### File: `ign_lidar/cli/commands/process.py`

Add the following function and modify the command:

```python
def load_hydra_config_from_file(
    config_file: str,
    overrides: Optional[list] = None
) -> DictConfig:
    """
    Load Hydra configuration from a custom YAML file.

    This allows users to provide their own config files instead of
    using only the built-in presets.

    Args:
        config_file: Path to custom YAML config file
        overrides: List of CLI overrides to apply on top

    Returns:
        Composed Hydra configuration

    Example:
        >>> cfg = load_hydra_config_from_file(
        ...     "my_custom_config.yaml",
        ...     ["processor.use_gpu=true"]
        ... )
    """
    from pathlib import Path
    import yaml

    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load the custom config
    with open(config_file, 'r') as f:
        custom_config = yaml.safe_load(f)

    # Get the package config directory for defaults
    package_config_dir = get_config_dir()

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra with package config directory
    with initialize_config_dir(config_dir=package_config_dir, version_base=None):
        # Start with base config
        cfg = compose(config_name="config", overrides=overrides or [])

        # Merge custom config on top
        custom_omega = OmegaConf.create(custom_config)
        cfg = OmegaConf.merge(cfg, custom_omega)

        # Apply CLI overrides last (highest priority)
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)

        # Handle output shorthand
        if hasattr(cfg, 'output') and isinstance(cfg.output, str):
            output_mode = cfg.output
            cfg.output = OmegaConf.create({
                "format": "npz",
                "save_enriched_laz": output_mode in ['both', 'enriched_only'],
                "only_enriched_laz": output_mode == 'enriched_only',
                "save_stats": True,
                "save_metadata": output_mode != 'enriched_only',
                "compression": None
            })

        return cfg


@click.command()
@click.option(
    '--config-file', '-c',
    type=click.Path(exists=True),
    help='Path to custom YAML config file (optional)'
)
@click.option(
    '--show-config',
    is_flag=True,
    help='Show the composed configuration and exit'
)
@click.argument('overrides', nargs=-1)
def process_command(config_file, show_config, overrides):
    """
    Process LiDAR tiles to create training patches.

    OVERRIDES: Hydra configuration overrides in key=value format

    Examples:
        # Use built-in config with overrides
        ign-lidar-hd process input_dir=data/raw output_dir=data/patches

        # Use custom config file
        ign-lidar-hd process --config-file my_config.yaml

        # Use custom config + overrides
        ign-lidar-hd process -c my_config.yaml processor.use_gpu=true

        # Show composed config without processing
        ign-lidar-hd process -c my_config.yaml --show-config
    """
    try:
        # Load configuration
        if config_file:
            logger.info(f"Loading configuration from: {config_file}")
            cfg = load_hydra_config_from_file(config_file, list(overrides))
        else:
            cfg = load_hydra_config(list(overrides))

        # Show config and exit if requested
        if show_config:
            click.echo("=" * 70)
            click.echo("Composed Configuration:")
            click.echo("=" * 70)
            click.echo(OmegaConf.to_yaml(cfg))
            return

        # Process
        process_lidar(cfg)

    except Exception as e:
        logger.error(f"Error: {e}")
        if logger.getEffectiveLevel() <= logging.DEBUG:
            raise
        raise click.ClickException(str(e))
```

---

## 2. Configuration Precedence

The configuration is composed in the following order (lowest to highest priority):

```
1. Package defaults (configs/config.yaml + sub-configs)
   ↓
2. Custom config file (--config-file)
   ↓
3. CLI overrides (key=value arguments)
```

### Example:

```yaml
# my_custom_config.yaml
processor:
  use_gpu: true
  num_workers: 8
  patch_size: 200.0

features:
  mode: full
  k_neighbors: 30
  use_rgb: true

input_dir: /data/lidar/raw
output_dir: /data/lidar/patches
```

**Command:**

```bash
ign-lidar-hd process --config-file my_custom_config.yaml processor.num_workers=16
```

**Result:**

- `processor.use_gpu = true` (from custom file)
- `processor.num_workers = 16` (from CLI override)
- `processor.patch_size = 200.0` (from custom file)
- Other settings from package defaults

---

## 3. Custom Config File Examples

### Example 1: GPU Processing

```yaml
# configs/custom_gpu_processing.yaml
processor:
  use_gpu: true
  num_workers: 1
  batch_size: 1
  pin_memory: true
  patch_size: 150.0
  num_points: 32768

features:
  mode: full
  k_neighbors: 30
  use_rgb: true
  use_infrared: true
  compute_ndvi: true
  gpu_batch_size: 1000000
  use_gpu_chunked: true

preprocess:
  enabled: false

stitching:
  enabled: false

output:
  save_enriched_laz: true
  only_enriched_laz: true

input_dir: /data/lidar/raw
output_dir: /data/lidar/enriched
```

**Usage:**

```bash
ign-lidar-hd process -c configs/custom_gpu_processing.yaml
```

### Example 2: Training Dataset Creation

```yaml
# configs/training_dataset.yaml
processor:
  use_gpu: false
  num_workers: 4
  patch_size: 100.0
  num_points: 16384
  augment: true
  num_augmentations: 5

features:
  mode: full
  k_neighbors: 20
  include_extra: true
  use_rgb: true
  normalize_xyz: true
  normalize_features: true

preprocess:
  enabled: true
  sor_k: 12
  sor_std: 2.0

stitching:
  enabled: true
  buffer_size: 10.0
  auto_detect_neighbors: true

output:
  format: npz
  save_enriched_laz: false
  save_stats: true
  save_metadata: true

input_dir: /data/lidar/raw
output_dir: /data/lidar/training_patches
```

**Usage:**

```bash
ign-lidar-hd process -c configs/training_dataset.yaml
```

### Example 3: Quick Enrichment

```yaml
# configs/quick_enrich.yaml
processor:
  use_gpu: true
  num_workers: 1

features:
  mode: minimal
  k_neighbors: 20
  use_rgb: true

preprocess:
  enabled: false

stitching:
  enabled: false

output:
  save_enriched_laz: true
  only_enriched_laz: true

input_dir: ??? # Required from CLI
output_dir: ??? # Required from CLI
```

**Usage:**

```bash
ign-lidar-hd process -c configs/quick_enrich.yaml \
  input_dir=/data/raw \
  output_dir=/data/enriched
```

---

## 4. Consolidated Memory Module

### File: `ign_lidar/core/memory.py` (merged from memory_manager.py + memory_utils.py)

```python
"""
Memory management utilities for IGN LiDAR HD processing.

Provides memory monitoring, estimation, and automatic batch sizing
to prevent OOM errors during large-scale processing.
"""

import psutil
import logging
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryInfo:
    """Memory usage information."""
    total: int
    available: int
    used: int
    percent: float

    def to_gb(self, value: int) -> float:
        """Convert bytes to GB."""
        return value / (1024 ** 3)

    def __str__(self) -> str:
        return (
            f"Memory: {self.to_gb(self.used):.2f}/{self.to_gb(self.total):.2f} GB "
            f"({self.percent:.1f}% used, {self.to_gb(self.available):.2f} GB available)"
        )


class MemoryManager:
    """
    Manages memory usage and provides automatic batch sizing.

    Monitors system memory and adjusts processing parameters
    to prevent out-of-memory errors.
    """

    def __init__(self, safety_margin: float = 0.2):
        """
        Initialize memory manager.

        Args:
            safety_margin: Safety margin (0-1) to leave available
        """
        self.safety_margin = safety_margin

    def get_memory_info(self) -> MemoryInfo:
        """Get current memory usage information."""
        mem = psutil.virtual_memory()
        return MemoryInfo(
            total=mem.total,
            available=mem.available,
            used=mem.used,
            percent=mem.percent
        )

    def get_available_memory(self) -> int:
        """Get available memory in bytes."""
        mem = psutil.virtual_memory()
        return mem.available

    def estimate_point_cloud_memory(
        self,
        num_points: int,
        num_features: int = 3,
        dtype: str = 'float32'
    ) -> int:
        """
        Estimate memory required for point cloud.

        Args:
            num_points: Number of points
            num_features: Number of features per point
            dtype: Data type ('float32' or 'float64')

        Returns:
            Estimated memory in bytes
        """
        bytes_per_value = 4 if dtype == 'float32' else 8
        base_memory = num_points * num_features * bytes_per_value

        # Add overhead for intermediate arrays (factor of 2)
        return base_memory * 2

    def calculate_optimal_batch_size(
        self,
        num_points: int,
        num_features: int = 3,
        dtype: str = 'float32',
        min_batch: int = 100,
        max_batch: Optional[int] = None
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            num_points: Total number of points to process
            num_features: Number of features per point
            dtype: Data type
            min_batch: Minimum batch size
            max_batch: Maximum batch size (optional)

        Returns:
            Optimal batch size
        """
        available = self.get_available_memory()
        usable = int(available * (1 - self.safety_margin))

        # Estimate memory per point
        memory_per_point = self.estimate_point_cloud_memory(1, num_features, dtype)

        # Calculate batch size
        batch_size = usable // memory_per_point
        batch_size = max(batch_size, min_batch)

        if max_batch is not None:
            batch_size = min(batch_size, max_batch)

        batch_size = min(batch_size, num_points)

        logger.debug(
            f"Calculated batch size: {batch_size:,} points "
            f"(available memory: {usable / (1024**3):.2f} GB)"
        )

        return batch_size

    def check_memory_available(self, required_gb: float) -> bool:
        """
        Check if sufficient memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory available
        """
        available_gb = self.get_available_memory() / (1024 ** 3)
        return available_gb >= required_gb

    def wait_for_memory(
        self,
        required_gb: float,
        timeout: int = 300,
        check_interval: int = 5
    ) -> bool:
        """
        Wait for sufficient memory to become available.

        Args:
            required_gb: Required memory in GB
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds

        Returns:
            True if memory became available, False if timeout
        """
        import time

        elapsed = 0
        while elapsed < timeout:
            if self.check_memory_available(required_gb):
                return True

            logger.info(
                f"Waiting for memory: {required_gb:.2f} GB required, "
                f"{self.get_available_memory() / (1024**3):.2f} GB available"
            )

            time.sleep(check_interval)
            elapsed += check_interval

        return False

    def log_memory_usage(self, prefix: str = "") -> None:
        """Log current memory usage."""
        info = self.get_memory_info()
        logger.info(f"{prefix}{info}")


# Utility functions (maintain backward compatibility)

def get_available_memory() -> int:
    """Get available memory in bytes."""
    return MemoryManager().get_available_memory()


def estimate_memory_usage(num_points: int, num_features: int = 3) -> int:
    """Estimate memory usage for point cloud."""
    return MemoryManager().estimate_point_cloud_memory(num_points, num_features)


def calculate_batch_size(
    total_points: int,
    available_memory: Optional[int] = None,
    safety_factor: float = 0.8
) -> int:
    """Calculate optimal batch size (backward compatibility)."""
    if available_memory is None:
        available_memory = get_available_memory()

    manager = MemoryManager(safety_margin=1.0 - safety_factor)
    return manager.calculate_optimal_batch_size(total_points)
```

### Migration Guide:

**Old imports:**

```python
from ign_lidar.core.memory_manager import MemoryManager
from ign_lidar.core.memory_utils import get_available_memory, estimate_memory_usage
```

**New imports:**

```python
from ign_lidar.core.memory import MemoryManager, get_available_memory, estimate_memory_usage
```

---

## 5. Testing Plan

### Test 1: Custom Config File Loading

```python
# tests/test_custom_config.py
import pytest
from pathlib import Path
import yaml
from ign_lidar.cli.commands.process import load_hydra_config_from_file


def test_load_custom_config_file(tmp_path):
    """Test loading configuration from custom file."""
    config_file = tmp_path / "test_config.yaml"

    config_content = {
        "processor": {
            "use_gpu": True,
            "num_workers": 8
        },
        "features": {
            "mode": "full",
            "k_neighbors": 30
        },
        "input_dir": "/test/input",
        "output_dir": "/test/output"
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    cfg = load_hydra_config_from_file(str(config_file))

    assert cfg.processor.use_gpu == True
    assert cfg.processor.num_workers == 8
    assert cfg.features.mode == "full"
    assert cfg.features.k_neighbors == 30


def test_custom_config_with_overrides(tmp_path):
    """Test custom config + CLI overrides."""
    config_file = tmp_path / "test_config.yaml"

    config_content = {
        "processor": {
            "use_gpu": False,
            "num_workers": 4
        },
        "input_dir": "/test/input",
        "output_dir": "/test/output"
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)

    # Overrides should take precedence
    cfg = load_hydra_config_from_file(
        str(config_file),
        ["processor.use_gpu=true", "processor.num_workers=16"]
    )

    assert cfg.processor.use_gpu == True  # Overridden
    assert cfg.processor.num_workers == 16  # Overridden


def test_config_file_not_found():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_hydra_config_from_file("nonexistent.yaml")
```

### Test 2: Memory Module Consolidation

```python
# tests/test_memory.py
import pytest
from ign_lidar.core.memory import MemoryManager, get_available_memory, estimate_memory_usage


def test_memory_manager():
    """Test MemoryManager basic functionality."""
    manager = MemoryManager()

    info = manager.get_memory_info()
    assert info.total > 0
    assert info.available > 0
    assert 0 <= info.percent <= 100


def test_batch_size_calculation():
    """Test optimal batch size calculation."""
    manager = MemoryManager()

    batch_size = manager.calculate_optimal_batch_size(
        num_points=1_000_000,
        num_features=3,
        min_batch=100
    )

    assert batch_size >= 100
    assert batch_size <= 1_000_000


def test_backward_compatibility():
    """Test backward compatibility of utility functions."""
    available = get_available_memory()
    assert available > 0

    estimated = estimate_memory_usage(10000, 3)
    assert estimated > 0
```

---

## 6. Documentation Updates

### File: `docs/docs/guides/configuration.md`

Add new section:

````markdown
## Loading Custom Configuration Files

You can create your own configuration files and load them with the `--config-file` option:

### Create a Custom Config

```yaml
# my_config.yaml
processor:
  use_gpu: true
  num_workers: 8
  patch_size: 150.0

features:
  mode: full
  k_neighbors: 30
  use_rgb: true

input_dir: /data/lidar/raw
output_dir: /data/lidar/patches
```
````

### Load and Process

```bash
ign-lidar-hd process --config-file my_config.yaml
```

### Configuration Precedence

Settings are applied in this order (highest priority last):

1. **Package defaults** - Built-in configs from `ign_lidar/configs/`
2. **Custom config file** - Your file specified with `--config-file`
3. **CLI overrides** - Command-line `key=value` arguments

Example:

```bash
# Custom file sets num_workers=8, override to 16
ign-lidar-hd process -c my_config.yaml processor.num_workers=16
```

### Show Composed Config

Preview the final configuration without processing:

```bash
ign-lidar-hd process -c my_config.yaml --show-config
```

```

---

## 7. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Add `load_hydra_config_from_file()` function
- [ ] Modify `process_command()` to accept `--config-file`
- [ ] Add `--show-config` option
- [ ] Update help text and docstrings

### Phase 2: Memory Consolidation
- [ ] Create new `ign_lidar/core/memory.py`
- [ ] Merge code from `memory_manager.py` and `memory_utils.py`
- [ ] Add backward compatibility imports
- [ ] Update all imports in codebase
- [ ] Deprecation warnings in old files

### Phase 3: Testing
- [ ] Write tests for custom config loading
- [ ] Write tests for config precedence
- [ ] Write tests for memory module
- [ ] Test backward compatibility
- [ ] Integration tests

### Phase 4: Documentation
- [ ] Update configuration guide
- [ ] Add example custom configs
- [ ] Update CLI reference
- [ ] Add migration guide

### Phase 5: Cleanup
- [ ] Mark old memory files as deprecated
- [ ] Update CHANGELOG
- [ ] Bump version to 2.3.0

---

## 8. Release Notes Draft

### Version 2.3.0 - Custom Configuration Support

#### ✨ New Features

**Custom Config File Loading**
- Added `--config-file` option to process command
- Load configurations from any YAML file
- Clear precedence: defaults < custom file < CLI overrides
- New `--show-config` option to preview composed configuration

**Consolidated Memory Management**
- Merged `memory_manager.py` and `memory_utils.py` into `core/memory.py`
- Improved API with `MemoryInfo` dataclass
- Backward compatibility maintained

#### 📚 Documentation
- New configuration guide with custom config examples
- Updated CLI reference
- Added migration guide

#### 🔧 Improvements
- Better error messages for config loading
- Enhanced logging for memory operations
- Type hints improvements

#### ⚠️ Deprecations
- `core/memory_manager.py` - use `core/memory.py` instead
- `core/memory_utils.py` - use `core/memory.py` instead

---

**End of Implementation Plan**
```
