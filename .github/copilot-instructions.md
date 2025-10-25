# GitHub Copilot Instructions for IGN LiDAR HD Dataset

## Project Overview

This is the **IGN LiDAR HD Processing Library** - a Python library for processing French IGN LiDAR HD data into machine learning-ready datasets with building Level of Detail (LOD) classification support.

**Version:** 3.0.0  
**Language:** Python 3.8+  
**Type:** Data Processing & ML Pipeline Library

## Core Technologies

- **LiDAR Processing:** laspy, lazrs, NumPy, SciPy
- **ML/Scientific:** scikit-learn, NumPy, PyTorch (optional)
- **GPU Acceleration:** CuPy, RAPIDS cuML, FAISS (optional)
- **Configuration:** Hydra, OmegaConf
- **Geospatial:** Shapely, GeoPandas, Rasterio, Rtree
- **Testing:** pytest

## Project Architecture

### Module Structure

```
ign_lidar/
├── core/              # Core processing orchestration
│   ├── processor.py         # Main LiDARProcessor class
│   ├── classification/      # Classification logic
│   ├── memory.py           # Memory management
│   ├── performance.py      # Performance monitoring
│   └── tile_stitcher.py    # Tile stitching
│
├── features/          # Feature computation
│   ├── orchestrator.py     # FeatureOrchestrator (unified API)
│   ├── feature_computer.py # Feature computation engine
│   ├── compute/            # Low-level compute functions
│   ├── strategies.py       # Strategy pattern for CPU/GPU
│   └── mode_selector.py    # Automatic mode selection
│
├── preprocessing/     # Data preprocessing
│   ├── outliers.py        # Outlier removal
│   └── augmentation.py    # RGB/NIR augmentation
│
├── io/                # Input/Output operations
│   ├── laz.py            # LAZ file handling
│   ├── metadata.py       # Metadata management
│   └── wfs_ground_truth.py # WFS ground truth fetching
│
├── config/            # Configuration management
│   ├── schema.py         # Config schema (Hydra)
│   └── defaults.py       # Default configurations
│
├── datasets/          # PyTorch datasets
│   └── multi_arch_dataset.py
│
└── cli/               # Command-line interface
    └── main.py
```

### Key Design Patterns

1. **Strategy Pattern:** CPU/GPU feature computation (`strategy_cpu.py`, `strategy_gpu.py`, `strategy_gpu_chunked.py`)
2. **Factory Pattern:** Optimization factory for adaptive processing
3. **Orchestrator Pattern:** `FeatureOrchestrator` unifies feature management
4. **Configuration Pattern:** Hydra-based hierarchical configuration

## Coding Standards

### Code Modification Rules (CRITICAL)

**ALWAYS follow these rules when making changes:**

1. **Modify Existing Files First:** Never create new files without first checking if the functionality can be added to or upgraded in existing files
2. **Update Before Create:** Always update and upgrade existing implementations before creating new ones
3. **Avoid Duplication:** Search for similar functionality in the codebase and extend/refactor it rather than duplicating
4. **No Redundant Prefixes:** Avoid using redundant prefixes like "unified", "enhanced", "new", "improved" in function/class names - just name things clearly by their purpose
5. **Refactor Over Rewrite:** When improving code, refactor the existing implementation rather than creating a parallel "v2" or "enhanced" version

**Examples of what NOT to do:**

- ❌ Creating `unified_feature_computer.py` when `feature_computer.py` exists
- ❌ Adding `enhanced_process_tile()` when `process_tile()` can be improved
- ❌ Creating `new_classifier.py` alongside `classifier.py`

**Examples of what to do:**

- ✅ Update `feature_computer.py` with new capabilities
- ✅ Refactor `process_tile()` to handle new cases
- ✅ Extend `classifier.py` with additional methods

### Python Style

- **PEP 8 compliance** with 88-character line length (Black formatter)
- **Type hints:** Use comprehensive type annotations (Python 3.8+ syntax)
- **Docstrings:** Google-style docstrings for all public functions/classes
- **Imports:** Organized in order: stdlib, third-party, local
- **Error handling:** Use custom exceptions from `core.error_handler`

### Example Code Style

```python
from typing import Dict, Optional, Union, Any
import numpy as np
from pathlib import Path

def compute_geometric_features(
    points: np.ndarray,
    k_neighbors: int = 30,
    search_radius: float = 3.0,
    use_gpu: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute geometric features for point cloud.

    Args:
        points: Point cloud array [N, 3] with XYZ coordinates
        k_neighbors: Number of neighbors for local features
        search_radius: Search radius in meters
        use_gpu: Whether to use GPU acceleration

    Returns:
        Dictionary with feature arrays (normals, curvature, etc.)

    Raises:
        ValueError: If points array is invalid
        GPUMemoryError: If GPU runs out of memory
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [N, 3] array, got {points.shape}")

    # Implementation...
    return features
```

### Naming Conventions

- **Classes:** `PascalCase` (e.g., `LiDARProcessor`, `FeatureOrchestrator`)
- **Functions/Methods:** `snake_case` (e.g., `compute_features`, `process_tile`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `ASPRS_CLASS_NAMES`, `LOD2_CLASSES`)
- **Private:** Prefix with `_` (e.g., `_process_tile_core`, `_validate_config`)

## Key Concepts

### 1. LOD (Level of Detail) Classification

- **LOD2:** Simplified building classification (12 features, 15 classes)
- **LOD3:** Detailed architectural classification (38 features, 30+ classes)
- **ASPRS:** American Society for Photogrammetry standard codes

### 2. Feature Modes

```python
from ign_lidar.features import FeatureMode

# Available modes:
# - MINIMAL: ~8 features (ultra-fast)
# - LOD2: ~12 features (essential for buildings)
# - LOD3: ~38 features (complete geometric description)
# - ASPRS_CLASSES: ~25 features (ASPRS classification)
# - FULL: All available features
# - CUSTOM: User-defined selection
```

### 3. Processing Modes

- **patches_only:** Create training patches only (default)
- **both:** Create patches + enriched LAZ tiles
- **enriched_only:** Create enriched LAZ tiles only

### 4. GPU Acceleration

Three computation strategies:

- **CPU:** Standard scikit-learn (`strategy_cpu.py`)
- **GPU:** CuPy/cuML for full dataset (`strategy_gpu.py`)
- **GPU_CHUNKED:** Batch processing for large datasets (`strategy_gpu_chunked.py`)

Automatic mode selection via `mode_selector.py`.

### 5. Configuration System (v3.0)

Hydra-based hierarchical configuration:

```yaml
# Example config structure
input_dir: /data/tiles
output_dir: /data/output

processor:
  lod_level: LOD2
  processing_mode: patches_only
  use_gpu: true
  patch_size: 150.0
  num_points: 16384

features:
  mode: lod2
  k_neighbors: 30
  search_radius: 3.0

data_sources:
  bd_topo:
    buildings: true
    roads: true
    vegetation: false
```

## Common Tasks & Patterns

### Adding a New Feature

1. Add computation in `features/compute/`
2. Update `FeatureOrchestrator` to include feature
3. Add to appropriate feature mode in `feature_modes.py`
4. Update tests in `tests/test_feature_*.py`
5. Document in `docs/docs/features/`

### GPU/CPU Compatibility

Always provide CPU fallback:

```python
def compute_feature(points: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Compute feature with GPU/CPU support."""
    if use_gpu and GPU_AVAILABLE:
        try:
            import cupy as cp
            points_gpu = cp.asarray(points)
            result = _compute_gpu(points_gpu)
            return cp.asnumpy(result)
        except Exception as e:
            logger.warning(f"GPU failed, falling back to CPU: {e}")
            # Fall through to CPU

    # CPU implementation
    return _compute_cpu(points)
```

### Memory Management

Use `AdaptiveMemoryManager` for large datasets:

```python
from ign_lidar.core import AdaptiveMemoryManager

memory_manager = AdaptiveMemoryManager(
    available_memory_gb=32,
    safety_margin=0.2
)

# Check before processing
if memory_manager.check_available_memory(required_gb=5.0):
    result = process_large_dataset(data)
```

### Error Handling

Use custom exceptions:

```python
from ign_lidar.core.error_handler import (
    ProcessingError,
    GPUMemoryError,
    FileProcessingError
)

try:
    result = process_tile(tile_path)
except GPUMemoryError:
    logger.warning("GPU OOM, falling back to CPU")
    result = process_tile_cpu(tile_path)
except FileProcessingError as e:
    logger.error(f"Failed to process {tile_path}: {e}")
    raise
```

## Testing Guidelines

### Test Structure

```
tests/
├── test_core_*.py        # Core functionality tests
├── test_feature_*.py     # Feature computation tests
├── test_gpu_*.py        # GPU-specific tests
├── test_integration_*.py # Integration tests
└── test_modules/        # Module-specific tests
```

### Test Markers

```python
import pytest

@pytest.mark.unit
def test_compute_normals():
    """Unit test for normal computation."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Integration test for full pipeline."""
    pass

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_features():
    """GPU-specific test."""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Slow test for large datasets."""
    pass
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/ -v -m unit

# Skip integration tests
pytest tests/ -v -m "not integration"

# With coverage
pytest tests/ -v --cov=ign_lidar --cov-report=html
```

## Performance Optimization

### Key Performance Areas

1. **GPU Utilization:** Aim for >80% GPU usage when enabled
2. **Memory Efficiency:** Use chunked processing for large datasets
3. **Parallel Processing:** Multi-worker for CPU operations (avoid with GPU)
4. **Caching:** Cache KD-trees, ground truth data, RGB tiles

### Performance Monitoring

```python
from ign_lidar.core import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_phase("feature_computation")
# ... processing ...
monitor.end_phase("feature_computation")

stats = monitor.get_summary()
print(f"Feature computation: {stats['feature_computation']['duration']:.2f}s")
```

## Documentation

### Where to Document

- **API docs:** Docstrings in code (Google style)
- **User guides:** `docs/docs/guides/`
- **Architecture:** `docs/docs/architecture.md`
- **Release notes:** `docs/docs/release-notes/`
- **Examples:** `examples/` with YAML configs

### Docstring Template

```python
def process_function(param1: type, param2: type) -> return_type:
    """
    One-line summary of the function.

    Longer description explaining what the function does,
    its purpose, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Example:
        >>> result = process_function(data, config)
        >>> print(result.shape)
        (1000, 3)

    Note:
        Any additional notes or warnings.

    See Also:
        related_function: Related functionality
    """
```

## Common Pitfalls to Avoid

1. **GPU + Multiprocessing:** Don't use GPU with `num_workers > 1` (CUDA context issues)
2. **Memory Leaks:** Always call `gc.collect()` after large operations
3. **Tile Boundaries:** Features at boundaries may have artifacts (use stitching)
4. **Classification Recalculation:** Recompute classification-dependent features after ground truth
5. **Config Validation:** Always validate config before processing (use `ConfigValidator`)

## Version Compatibility

### Backward Compatibility

The library maintains backward compatibility:

```python
# v3.0+ (new, recommended)
from ign_lidar import LiDARProcessor
processor = LiDARProcessor(config_path="config.yaml")

# v2.x (legacy, still supported with deprecation warnings)
from ign_lidar.processor import LiDARProcessor  # Deprecated
processor = LiDARProcessor(lod_level="LOD2", use_gpu=True)
```

### Deprecation Warnings

Add deprecation warnings for legacy code:

```python
import warnings

def legacy_function():
    warnings.warn(
        "legacy_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

## Dependencies

### Core Dependencies

- numpy>=1.21.0
- laspy>=2.3.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- numba>=0.56.0
- hydra-core>=1.3.0
- omegaconf>=2.3.0

### Optional Dependencies

- **GPU:** cupy-cuda11x or cupy-cuda12x, cuml, cuspatial
- **PyTorch:** torch>=2.0.0
- **FAISS:** faiss-gpu or faiss-cpu
- **Geospatial:** shapely>=2.0.0, geopandas>=0.12.0

## Contact & Resources

- **GitHub:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET
- **Documentation:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **PyPI:** https://pypi.org/project/ign-lidar-hd/
- **Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues

## Key Files to Reference

When working on this project, always reference:

- `ign_lidar/core/processor.py` - Main processing logic
- `ign_lidar/features/orchestrator.py` - Feature management
- `ign_lidar/classification_schema.py` - Classification definitions
- `ign_lidar/config/schema.py` - Configuration schema
- `examples/*.yaml` - Example configurations
- `tests/conftest.py` - Test fixtures and utilities

---

**Remember:** This is a production library used for processing large LiDAR datasets. Prioritize:

1. **Correctness** over speed
2. **Memory efficiency** for large datasets
3. **GPU/CPU compatibility** with fallbacks
4. **Clear error messages** for users
5. **Comprehensive testing** before merging
