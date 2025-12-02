# Version 4.0.0 Breaking Changes - Quick Reference

**Target Release:** Q2 2025  
**Current Version:** 3.6.3  
**Last Updated:** December 2, 2025

This document provides a **quick reference** for all breaking changes in v4.0.0 with migration examples.

---

## 🚨 Summary of Breaking Changes

| Category | Changes | Impact | Migration Path |
|----------|---------|--------|----------------|
| **Configuration** | v3.x structure removed | 🔴 HIGH | Automatic tool available |
| **Feature APIs** | FeatureComputer removed | 🟡 MEDIUM | Use FeatureOrchestrator |
| **Normal Functions** | Deprecated functions removed | 🟡 MEDIUM | Use canonical API |
| **Import Paths** | Legacy paths removed | 🟢 LOW | Update imports |
| **GPU APIs** | Centralized management required | 🟡 MEDIUM | Use GPUManager |
| **Python Version** | 3.8 dropped | 🟢 LOW | Upgrade to 3.9+ |

---

## 1️⃣ Configuration Changes (HIGH IMPACT)

### ❌ REMOVED: v3.x Nested Configuration

```yaml
# ❌ NO LONGER WORKS IN v4.0
processor:
  lod_level: LOD2
  use_gpu: true
  processing_mode: patches_only
features:
  feature_set: standard
  k_neighbors: 30
```

### ✅ REQUIRED: v4.0 Flat Configuration

```yaml
# ✅ CORRECT FOR v4.0
mode: lod2              # Flat, top-level
use_gpu: true           # Flat, top-level
processing_mode: patches_only

features:
  mode: standard        # Renamed from feature_set
  k_neighbors: 30
```

### 🔧 Migration Tool (Automatic)

```bash
# Automatically migrate your config
ign-lidar migrate-config your_config.yaml

# Preview changes first
ign-lidar migrate-config your_config.yaml --dry-run --verbose

# Batch migration
ign-lidar migrate-config configs/ --batch
```

### 🐍 Python API Changes

```python
# ❌ REMOVED IN v4.0
from ign_lidar.config.schema import IGNLiDARConfig
config = IGNLiDARConfig(...)

# ❌ REMOVED IN v4.0
config = Config(processor={'lod_level': 'LOD2', 'use_gpu': True})

# ✅ CORRECT FOR v4.0
from ign_lidar import Config

# Option 1: Use presets (recommended)
config = Config.preset('lod2_buildings')

# Option 2: Flat structure
config = Config(
    mode='lod2',
    use_gpu=True,
    processing_mode='patches_only'
)
```

### 📋 Key Renames

| v3.x | v4.0 | Notes |
|------|------|-------|
| `processor.lod_level: "LOD2"` | `mode: lod2` | Lowercase, top-level |
| `features.feature_set` | `features.mode` | Consistent naming |
| `features.use_infrared` | `features.use_nir` | Standard terminology |
| `processor.use_gpu` | `use_gpu` | Top-level |
| `processor.num_workers` | `num_workers` | Top-level |

---

## 2️⃣ FeatureComputer Removal (MEDIUM IMPACT)

### ❌ REMOVED: FeatureComputer Class

```python
# ❌ NO LONGER WORKS IN v4.0
from ign_lidar.features import FeatureComputer

computer = FeatureComputer(use_gpu=True)
features = computer.compute_features(
    points=points,
    k_neighbors=30
)
```

### ✅ MIGRATION: Use FeatureOrchestrator

```python
# ✅ CORRECT FOR v4.0
from ign_lidar.features import FeatureOrchestrator
from ign_lidar import Config

config = Config(use_gpu=True)
orchestrator = FeatureOrchestrator(config=config)

features = orchestrator.compute_features(
    points=points,
    k_neighbors=30
)
```

### 📦 Alternative: Use High-Level API

```python
# ✅ EVEN SIMPLER (recommended for most users)
from ign_lidar import LiDARProcessor, Config

config = Config.preset('lod2_buildings')
processor = LiDARProcessor(config)

# Features computed automatically during processing
processor.process_tile('tile.laz')
```

---

## 3️⃣ Normal Computation Functions (MEDIUM IMPACT)

### ❌ REMOVED: Deprecated Functions

```python
# ❌ NO LONGER WORKS IN v4.0
from ign_lidar.features.numba_accelerated import (
    compute_normals_from_eigenvectors,
    compute_normals_from_eigenvectors_numpy,
    compute_normals_from_eigenvectors_numba
)

normals = compute_normals_from_eigenvectors(eigenvectors)
```

### ✅ MIGRATION: Canonical Normal API

```python
# ✅ CORRECT FOR v4.0 (CPU)
from ign_lidar.features.compute.normals import compute_normals

normals = compute_normals(
    points=points,
    k_neighbors=30,
    method='fast'  # or 'accurate'
)
```

```python
# ✅ CORRECT FOR v4.0 (GPU - automatic via orchestrator)
from ign_lidar.features import FeatureOrchestrator
from ign_lidar import Config

config = Config(use_gpu=True)
orchestrator = FeatureOrchestrator(config)

# Normals computed automatically with optimal method
features = orchestrator.compute_features(points)
normals = features['normals']
```

### 🔄 Method Options

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| `'fast'` | ⚡⚡⚡ | ⭐⭐ | Quick previews, large datasets |
| `'accurate'` | ⚡⚡ | ⭐⭐⭐ | Production quality |
| `'gpu'` (auto) | ⚡⚡⚡⚡ | ⭐⭐⭐ | GPU available |

---

## 4️⃣ Import Path Changes (LOW IMPACT)

### ❌ REMOVED: Legacy Import Paths

```python
# ❌ NO LONGER WORKS IN v4.0
from ign_lidar.features.core import compute_normals
from ign_lidar.features.core import compute_curvature
from ign_lidar.features.core import extract_geometric_features
```

### ✅ MIGRATION: Canonical Import Paths

```python
# ✅ CORRECT FOR v4.0
from ign_lidar.features.compute import compute_normals
from ign_lidar.features.compute import compute_curvature
from ign_lidar.features.compute import extract_geometric_features

# OR use top-level imports (recommended)
from ign_lidar.features import (
    compute_normals,
    compute_curvature,
    extract_geometric_features
)
```

### 📚 Import Reference

```python
# Top-level imports (RECOMMENDED)
from ign_lidar import (
    Config,                    # Configuration
    LiDARProcessor,           # Main processor
    FeatureOrchestrator,      # Feature computation
    BaseClassifier,           # Classification
)

# Module-specific imports
from ign_lidar.features.compute import (
    compute_normals,
    compute_curvature,
    compute_eigenvalue_features,
)

from ign_lidar.preprocessing import (
    preprocess_point_cloud,
    radius_outlier_removal,
    statistical_outlier_removal,
)
```

---

## 5️⃣ GPU API Changes (MEDIUM IMPACT)

### ❌ REMOVED: Scattered GPU Checks

```python
# ❌ NO LONGER WORKS IN v4.0
from ign_lidar.optimization.gpu_wrapper import check_gpu_available

if check_gpu_available():
    # Use GPU

# ❌ NO LONGER WORKS IN v4.0
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# ❌ NO LONGER WORKS IN v4.0
import cupy as cp
mempool = cp.get_default_memory_pool()
```

### ✅ MIGRATION: Centralized GPUManager

```python
# ✅ CORRECT FOR v4.0
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()

# Check availability
if gpu.gpu_available:
    # Use GPU
    cp = gpu.get_cupy()  # Get CuPy if needed
    
    # Memory operations
    mempool = gpu.get_memory_pool()
    memory_info = gpu.get_memory_info()
    
    # Batch transfers (efficient)
    gpu_arrays = gpu.batch_upload(arr1, arr2, arr3)
    cpu_arrays = gpu.batch_download(*gpu_arrays)
```

### 💡 Best Practice: Use Config

```python
# ✅ RECOMMENDED: Let config handle GPU management
from ign_lidar import Config, LiDARProcessor

config = Config.preset('lod2_buildings')
config.use_gpu = True  # GPU auto-detected and used if available

processor = LiDARProcessor(config)
# GPU usage is automatic and optimized
```

---

## 6️⃣ Python Version Changes (LOW IMPACT)

### ❌ DROPPED: Python 3.8

```bash
# ❌ NO LONGER SUPPORTED IN v4.0
python3.8 -m pip install ign-lidar-hd==4.0.0
# ERROR: Requires Python >=3.9
```

### ✅ REQUIRED: Python 3.9+

```bash
# ✅ CORRECT FOR v4.0
python3.9 -m pip install ign-lidar-hd==4.0.0
python3.10 -m pip install ign-lidar-hd==4.0.0
python3.11 -m pip install ign-lidar-hd==4.0.0
python3.12 -m pip install ign-lidar-hd==4.0.0
```

### 🔄 Upgrade Guide

```bash
# Check current Python version
python --version

# If < 3.9, install newer Python
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9

# macOS with Homebrew
brew install python@3.9

# Windows: Download from python.org

# Create new virtual environment
python3.9 -m venv venv_v4
source venv_v4/bin/activate  # Linux/Mac
# or
venv_v4\Scripts\activate  # Windows

# Install v4.0
pip install ign-lidar-hd==4.0.0
```

---

## 📋 Complete Migration Checklist

### Before Upgrading

- [ ] **Backup your configs** - Copy all YAML configs
- [ ] **Backup your code** - Commit all changes to git
- [ ] **Review warnings** - Run v3.7.0 and check deprecation warnings
- [ ] **Test migration** - Try migration tool on copy of configs
- [ ] **Read docs** - Review full migration guide

### Upgrade Steps

1. **Upgrade Python** (if needed)
   ```bash
   python --version  # Ensure >=3.9
   ```

2. **Install v3.7.0 first** (transition release)
   ```bash
   pip install ign-lidar-hd==3.7.0
   ```

3. **Review deprecation warnings**
   ```bash
   ign-lidar process --config your_config.yaml
   # Note all warnings
   ```

4. **Migrate configuration**
   ```bash
   ign-lidar migrate-config your_config.yaml --output config_v4.yaml
   ```

5. **Update code imports**
   - Replace `FeatureComputer` → `FeatureOrchestrator`
   - Replace deprecated normal functions → canonical API
   - Replace legacy imports → new paths
   - Replace GPU checks → `GPUManager`

6. **Test with v3.7.0**
   ```bash
   ign-lidar process --config config_v4.yaml --dry-run
   ```

7. **Upgrade to v4.0**
   ```bash
   pip install --upgrade ign-lidar-hd==4.0.0
   ```

8. **Validate installation**
   ```bash
   ign-lidar info  # Check system info
   ign-lidar validate-config config_v4.yaml
   ```

9. **Run tests**
   ```bash
   ign-lidar process --config config_v4.yaml
   ```

### After Upgrading

- [ ] **Verify outputs** - Compare with v3.x results
- [ ] **Check performance** - Should be equal or better
- [ ] **Update documentation** - Note any behavior changes
- [ ] **Report issues** - If you find problems

---

## 🆘 Common Issues & Solutions

### Issue: "Config validation failed"

**Problem:** Using v3.x config structure with v4.0

**Solution:**
```bash
# Use migration tool
ign-lidar migrate-config old_config.yaml
```

### Issue: "ModuleNotFoundError: FeatureComputer"

**Problem:** Code uses deprecated FeatureComputer

**Solution:**
```python
# Replace
from ign_lidar.features import FeatureComputer

# With
from ign_lidar.features import FeatureOrchestrator
```

### Issue: "ImportError: cannot import compute_normals_from_eigenvectors"

**Problem:** Using deprecated normal functions

**Solution:**
```python
# Replace
from ign_lidar.features.numba_accelerated import compute_normals_from_eigenvectors

# With
from ign_lidar.features.compute.normals import compute_normals
```

### Issue: "Python version not supported"

**Problem:** Using Python 3.8

**Solution:**
```bash
# Upgrade to Python 3.9+
python3.9 -m venv venv
source venv/bin/activate
pip install ign-lidar-hd==4.0.0
```

### Issue: "Performance regression"

**Problem:** v4.0 slower than v3.x

**Solution:**
1. Check GPU is actually being used (`ign-lidar info`)
2. Verify config has `use_gpu: true`
3. Check CUDA/CuPy installation
4. Report issue with benchmarks

---

## 📞 Getting Help

### Documentation
- **Migration Guide:** [docs/migration-guide-v4.md](docs/docs/migration-guide-v4.md)
- **Configuration Guide:** [docs/configuration-guide-v4.md](docs/docs/configuration-guide-v4.md)
- **API Reference:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

### Community Support
- **GitHub Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **GitHub Discussions:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/discussions
- **Email:** simon.ducournau@gmail.com

### Reporting Problems
When reporting migration issues, please include:
- Current version (v3.x)
- Python version
- Error message (full traceback)
- Config file (sanitized)
- Code snippet (minimal example)

---

**Document Version:** 1.0  
**Last Updated:** December 2, 2025  
**Status:** 📋 DRAFT - Ready for Review
