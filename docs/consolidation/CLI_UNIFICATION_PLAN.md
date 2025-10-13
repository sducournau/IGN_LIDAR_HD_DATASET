# CLI Unification Status & Plan

**Date:** October 13, 2025  
**Phase:** 2.3 - Update CLI to Use Hydra  
**Status:** Analysis Complete, Implementation Ready

---

## 📊 Current CLI Architecture

### Structure Overview

```
ign_lidar/cli/
├── __init__.py
├── main.py              # Click-based entry point (hybrid)
├── hydra_main.py        # Pure Hydra entry point
└── commands/            # Click command modules
    ├── __init__.py
    ├── process.py       # Main processing
    ├── download.py      # Dataset download
    ├── verify.py        # Verification
    ├── batch_convert.py # Format conversion
    └── info.py          # Information display
```

### Current Entry Points

1. **Click CLI** (`main.py`):

   ```bash
   ign-lidar-hd process --config-file config.yaml
   ign-lidar-hd download --position 650000 6860000 --radius 5000
   ign-lidar-hd verify output_dir=patches/
   ```

2. **Hydra CLI** (`hydra_main.py`):

   ```bash
   python -m ign_lidar.cli.hydra_main input_dir=data/ output_dir=patches/
   ```

3. **Hybrid Detection** (`main.py:120-130`):
   ```python
   if '--config-path' in sys.argv or '--config-name' in sys.argv:
       hydra_process_entry()  # Use Hydra
   else:
       cli()  # Use Click
   ```

---

## 🎯 Target Architecture (Phase 2.3)

### Goal: Unified Hydra-based CLI

**Approach:** Keep Click as user-friendly interface layer, use Hydra as configuration engine underneath.

### Why Hybrid Instead of Pure Hydra?

1. **Better UX**: Click provides cleaner help messages and command structure
2. **Backward Compatibility**: Existing scripts using Click syntax continue to work
3. **Progressive Migration**: Users can gradually adopt Hydra overrides
4. **Best of Both**: Click for commands, Hydra for configuration

### New Architecture

```
ign_lidar/cli/
├── __init__.py
├── main.py              # Unified entry point (Click + Hydra)
├── hydra_runner.py      # NEW: Programmatic Hydra wrapper
└── commands/            # Updated commands using Hydra configs
    ├── __init__.py
    ├── process.py       # Uses hydra_runner
    ├── download.py      # Uses hydra_runner
    ├── verify.py        # Uses hydra_runner
    ├── batch_convert.py # Uses hydra_runner
    └── info.py          # Pure Click (no config needed)
```

---

## 📝 Implementation Plan

### Step 1: Create Hydra Runner Utility

**File:** `ign_lidar/cli/hydra_runner.py`

**Purpose:** Programmatically use Hydra without decorators

```python
"""
Hydra runner utility for programmatic Hydra usage.

Enables using Hydra configuration system within Click commands.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


class HydraRunner:
    """Programmatic Hydra configuration loader."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Hydra runner.

        Args:
            config_path: Path to config directory (defaults to package configs/)
        """
        if config_path is None:
            # Default to package configs
            config_path = Path(__file__).parent.parent / "configs"

        self.config_path = config_path
        self._initialized = False

    def load_config(
        self,
        config_name: str = "config",
        overrides: Optional[List[str]] = None,
        config_file: Optional[Path] = None
    ) -> DictConfig:
        """
        Load Hydra configuration with overrides.

        Args:
            config_name: Base config name (without .yaml)
            overrides: List of Hydra overrides (e.g., ["processor.use_gpu=true"])
            config_file: Specific config file to load

        Returns:
            Loaded and composed configuration
        """
        if config_file:
            # Load specific file
            cfg = OmegaConf.load(config_file)

            # Apply overrides
            if overrides:
                override_cfg = OmegaConf.from_dotlist(overrides)
                cfg = OmegaConf.merge(cfg, override_cfg)

            return cfg
        else:
            # Use Hydra compose
            with initialize_config_dir(
                config_dir=str(self.config_path.absolute()),
                version_base=None
            ):
                cfg = compose(
                    config_name=config_name,
                    overrides=overrides or []
                )

            return cfg
```

---

### Step 2: Update Command Modules

#### process.py

**Before:**

```python
@click.command()
@click.option('--config-file', type=click.Path(exists=True))
@click.option('--input-dir', type=click.Path(exists=True))
def process_command(config_file, input_dir, ...):
    # Complex parameter handling
    # Manual config loading
    pass
```

**After:**

```python
@click.command()
@click.option('--config-file', '-c', type=click.Path(exists=True))
@click.argument('overrides', nargs=-1)
def process_command(config_file, overrides):
    """Process LiDAR tiles with Hydra configuration."""
    from ..hydra_runner import HydraRunner

    # Load config with overrides
    runner = HydraRunner()
    cfg = runner.load_config(
        config_file=config_file if config_file else None,
        overrides=list(overrides)
    )

    # Run processing
    from ..hydra_main import process_lidar
    process_lidar(cfg)
```

---

### Step 3: Simplify main.py

**Remove:** Hybrid detection logic  
**Keep:** Click group structure  
**Add:** Hydra override support

```python
@click.group()
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def cli(ctx, verbose):
    """IGN LiDAR HD - Unified CLI"""
    from .logging_config import setup_logging
    setup_logging(level="DEBUG" if verbose else "INFO")
```

---

### Step 4: Deprecate Pure Hydra Entry Point

**File:** `hydra_main.py`

**Add warning:**

```python
"""
.. deprecated:: 2.4.4
    Direct use of hydra_main is deprecated.
    Use the unified CLI instead:

    OLD: python -m ign_lidar.cli.hydra_main input_dir=data/
    NEW: ign-lidar-hd process input_dir=data/

    This module will be removed in v2.5.0.
"""

import warnings

warnings.warn(
    "Direct use of hydra_main is deprecated. "
    "Use 'ign-lidar-hd process' instead. "
    "See MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2
)
```

---

## 🧪 Testing Strategy

### Test Cases

1. **Config File Loading**

   ```bash
   ign-lidar-hd process --config-file examples/config_complete.yaml
   ```

2. **Hydra Overrides**

   ```bash
   ign-lidar-hd process --config-file config.yaml \
     processor.use_gpu=true \
     features.k_neighbors=30
   ```

3. **Pure Overrides (No Config File)**

   ```bash
   ign-lidar-hd process \
     input_dir=data/ \
     output_dir=patches/ \
     processor.patch_size=100.0
   ```

4. **Mixed Parameters**

   ```bash
   ign-lidar-hd process \
     --config-file config.yaml \
     input_dir=data/ \
     processor.use_gpu=true
   ```

5. **Backward Compatibility**
   ```bash
   # Old syntax should still work
   python -m ign_lidar.cli.hydra_main input_dir=data/
   ```

---

## 📊 Benefits of This Approach

### User Experience

- ✅ Familiar Click command structure
- ✅ Clean help messages
- ✅ Progressive adoption of Hydra features
- ✅ Both syntaxes work during transition

### Developer Experience

- ✅ Single configuration system (Hydra)
- ✅ Type-safe configs with OmegaConf
- ✅ Easy to test (programmatic Hydra)
- ✅ Reduced code duplication

### Migration Path

- ✅ Backward compatible
- ✅ Deprecation warnings guide users
- ✅ Migration guide available
- ✅ Can remove old code in v2.5.0

---

## 🚀 Implementation Checklist

### Phase 2.3.1: Core Infrastructure (2 hours)

- [ ] Create `hydra_runner.py` with HydraRunner class
- [ ] Add unit tests for HydraRunner
- [ ] Document usage patterns

### Phase 2.3.2: Update Commands (2 hours)

- [ ] Update `commands/process.py` to use HydraRunner
- [ ] Update `commands/download.py` (if needed)
- [ ] Update `commands/verify.py` (if needed)
- [ ] Keep `commands/info.py` as-is (no config needed)

### Phase 2.3.3: Simplify main.py (1 hour)

- [ ] Remove hybrid detection logic
- [ ] Clean up entry points
- [ ] Update help messages
- [ ] Test all commands

### Phase 2.3.4: Deprecate Old Patterns (1 hour)

- [ ] Add deprecation warning to `hydra_main.py`
- [ ] Update documentation
- [ ] Add to MIGRATION_GUIDE.md
- [ ] Test deprecation warnings

### Phase 2.3.5: Testing & Documentation (1 hour)

- [ ] Integration tests for all commands
- [ ] Update CLI documentation
- [ ] Update example scripts
- [ ] Verify backward compatibility

**Total Estimated Time:** 7 hours

---

## 📋 CLI Command Reference

### After Implementation

All commands use unified syntax:

```bash
# Process command
ign-lidar-hd process [OPTIONS] [OVERRIDES...]

# Download command
ign-lidar-hd download [OPTIONS] [OVERRIDES...]

# Verify command
ign-lidar-hd verify [OPTIONS] [OVERRIDES...]

# Info command
ign-lidar-hd info

# Batch convert command
ign-lidar-hd batch-convert [OPTIONS] [OVERRIDES...]
```

### Options (Common)

- `--config-file, -c`: Load base configuration from file
- `--verbose, -v`: Enable verbose output
- `--help`: Show help message

### Overrides

Any Hydra-style override:

- `input_dir=path/to/data`
- `processor.use_gpu=true`
- `features.k_neighbors=30`
- `processor.patch_size=100.0`

---

## 🎓 Design Decisions

### Why Not Pure Hydra CLI?

**Considered:**

```python
@hydra.main(...)
def main(cfg):
    if cfg.command == 'process':
        process(cfg)
    elif cfg.command == 'download':
        download(cfg)
```

**Issues:**

- Less intuitive help messages
- Harder to discover commands
- Non-standard CLI UX
- Requires config for everything

### Why Programmatic Hydra?

**Advantages:**

- Flexible integration with Click
- Better error messages
- Easier testing
- Can use different configs per command

**Implementation:**

```python
# Instead of @hydra.main decorator
from hydra import compose, initialize_config_dir

with initialize_config_dir(...):
    cfg = compose(config_name="config", overrides=overrides)
```

---

## 🔄 Migration Examples

### Example 1: Basic Processing

**Old (v2.4.x):**

```bash
python -m ign_lidar.cli.hydra_main input_dir=data/ output_dir=patches/
```

**New (v2.5.0):**

```bash
ign-lidar-hd process input_dir=data/ output_dir=patches/
```

### Example 2: Config File

**Old:**

```bash
ign-lidar-hd process --config-file config.yaml --input data/ --output patches/
```

**New:**

```bash
ign-lidar-hd process --config-file config.yaml input_dir=data/ output_dir=patches/
```

### Example 3: GPU Processing

**Old:**

```bash
ign-lidar-hd process --config-file config.yaml --use-gpu
```

**New:**

```bash
ign-lidar-hd process --config-file config.yaml processor.use_gpu=true
```

---

## 📈 Success Metrics

### Completion Criteria

- ✅ All commands use Hydra config system
- ✅ Click interface preserved for UX
- ✅ Backward compatibility maintained
- ✅ Tests pass for all commands
- ✅ Documentation updated

### Code Quality

- Reduce CLI code by ~30%
- Single configuration system
- Better error handling
- Improved testability

---

**Status:** Ready for Implementation  
**Priority:** HIGH (Phase 2.3)  
**Estimated Effort:** 7 hours  
**Next Step:** Create `hydra_runner.py`
