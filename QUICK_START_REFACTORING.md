# Quick Start - Refactoring IGN LiDAR HD

**Goal:** Improve package with 3 clear processing modes and custom config support  
**Time:** 3 weeks  
**Version:** 2.2.2 → 2.3.0

---

## 🎯 The Problem

### Current Issues:
```python
# Confusing! What does this do?
save_enriched_laz=True
only_enriched_laz=False

# Can't do this!
ign-lidar-hd process --config-file my_config.yaml  ❌
```

---

## ✨ The Solution

### 1. Three Explicit Processing Modes

```yaml
# Mode 1: Patches only (ML training)
output:
  processing_mode: patches_only

# Mode 2: Both (ML + GIS)
output:
  processing_mode: both

# Mode 3: Enriched LAZ only (GIS analysis)
output:
  processing_mode: enriched_only
```

### 2. Custom Config File Support

```bash
# Load your own config
ign-lidar-hd process --config-file my_project.yaml

# Preview before running
ign-lidar-hd process -c my_project.yaml --show-config

# Combine with overrides
ign-lidar-hd process -c my_project.yaml processor.use_gpu=true
```

### 3. Unified Augmentation

```python
from ign_lidar.augmentation.core import augment_patch, AugmentationConfig

# Simple!
config = AugmentationConfig.default()
augmented = augment_patch(patch, config)
```

---

## 📋 Implementation Steps

### Week 1: Core Features

**Day 1-2: Processing Modes**
```python
# File: ign_lidar/core/processor.py
class LiDARProcessor:
    def __init__(
        self,
        processing_mode: Literal["patches_only", "both", "enriched_only"] = "patches_only",
        # ...
    ):
        self.processing_mode = processing_mode
```

**Day 3-4: Custom Config**
```python
# File: ign_lidar/cli/commands/process.py
@click.option('--config-file', '-c', type=click.Path(exists=True))
def process_command(config_file, overrides):
    if config_file:
        cfg = load_config_from_file(config_file, overrides)
    else:
        cfg = load_hydra_config(overrides)
```

**Day 5: Test**
```bash
# Test all modes
python -m pytest tests/test_processing_modes.py -v
```

### Week 2: Refactoring

**Day 1-3: Augmentation Module**
```
ign_lidar/
  augmentation/          # NEW
    __init__.py
    core.py              # Unified functions
```

**Day 4-5: Memory Consolidation**
```
ign_lidar/core/
  memory.py              # Merged from memory_manager + memory_utils
```

### Week 3: Documentation & Release

**Day 1-2: Tests**
- Integration tests
- Backward compatibility tests
- Performance benchmarks

**Day 3-4: Documentation**
- User guide for processing modes
- Custom config examples
- Migration guide

**Day 5: Release v2.3.0**
- Update CHANGELOG
- Create release notes
- Tag and publish

---

## 🧪 Quick Test

After implementing, test with:

```bash
# Create test config
cat > test_config.yaml << 'YAML'
processor:
  use_gpu: false
  num_workers: 4

output:
  processing_mode: enriched_only

input_dir: data/raw
output_dir: data/enriched
YAML

# Run with custom config
ign-lidar-hd process --config-file test_config.yaml

# Should create only enriched LAZ files, no patches
ls data/enriched/
# Expected: tile_enriched.laz (no .npz files)
```

---

## 📚 Full Documentation

For complete details, see:

1. **AUDIT_SUMMARY.md** - Overview and action plan
2. **PACKAGE_AUDIT_REPORT.md** - Detailed analysis
3. **REFACTORING_PLAN_V2.md** - Complete implementation guide

---

## ✅ Success Checklist

- [ ] Three processing modes work correctly
- [ ] Custom config file loading works
- [ ] `--show-config` option added
- [ ] Unified augmentation module created
- [ ] All tests pass (100% coverage on new code)
- [ ] Documentation updated
- [ ] Backward compatibility maintained
- [ ] Version bumped to 2.3.0

---

## 🚀 Start Now!

```bash
# 1. Read the full plan
cat REFACTORING_PLAN_V2.md

# 2. Create test file first (TDD)
touch tests/test_processing_modes.py

# 3. Implement ProcessingMode
vim ign_lidar/core/processor.py

# 4. Update config schema
vim ign_lidar/config/schema.py

# Let's go! 🎉
```

---

*See REFACTORING_PLAN_V2.md for complete implementation details*
