# IGN LiDAR Package Reorganization Summary

## Overview

The `ign_lidar` package has been reorganized to improve modularity, maintainability, and separation of concerns. Legacy functionality has been moved to appropriate subdirectories while maintaining backward compatibility.

## Files Moved

### Core Processing Utilities → `ign_lidar/core/`

- `memory_manager.py` → `ign_lidar/core/memory_manager.py`
  - `AdaptiveMemoryManager` class for real-time memory monitoring
  - `MemoryConfig` dataclass for memory configuration
- `memory_utils.py` → `ign_lidar/core/memory_utils.py`
  - Memory management utilities for CLI commands
  - System memory analysis functions
- `performance_monitor.py` → `ign_lidar/core/performance_monitor.py`
  - Real-time performance monitoring for GPU/CPU
  - `PerformanceMonitor` and `PerformanceSnapshot` classes
- `verification.py` → `ign_lidar/core/verification.py`
  - Feature verification and artifact detection
  - `FeatureVerifier` and `FeatureStats` classes
- `error_handler.py` → `ign_lidar/core/error_handler.py`
  - Enhanced error handling with recovery suggestions
  - Various error classes: `ProcessingError`, `GPUMemoryError`, etc.

### Feature Utilities → `ign_lidar/features/`

- `architectural_styles.py` → `ign_lidar/features/architectural_styles.py`
  - Architectural style classification system
  - `ARCHITECTURAL_STYLES`, `STYLE_NAME_TO_ID` mappings
  - Style encoding functions

### Dataset Management → `ign_lidar/datasets/`

- `strategic_locations.py` → `ign_lidar/datasets/strategic_locations.py`
  - Strategic location database for AI training datasets
  - `STRATEGIC_LOCATIONS` dictionary with heritage sites and building types
- `tile_list.py` → `ign_lidar/datasets/tile_list.py`
  - Working list of 50 IGN LiDAR HD tiles with real WFS data
  - `WORKING_TILES`, `TileInfo` class, and tile filtering functions

### Input/Output Utilities → `ign_lidar/io/`

- `metadata.py` → `ign_lidar/io/metadata.py`
  - Metadata management for IGN LiDAR HD processing
  - `MetadataManager` class for stats.json file handling
- `qgis_converter.py` → `ign_lidar/io/qgis_converter.py`
  - QGIS compatibility converter for LAZ files
  - `simplify_for_qgis` function for format conversion

### Preprocessing Utilities → `ign_lidar/preprocessing/`

- `utils.py` → `ign_lidar/preprocessing/utils.py`
  - Patch extraction and data augmentation utilities
  - `augment_raw_points`, `extract_patches`, `save_patch` functions
- `tile_analyzer.py` → `ign_lidar/preprocessing/tile_analyzer.py`
  - LiDAR tile analysis for optimal processing parameters
  - `analyze_tile` function for automatic parameter detection

## Files Remaining at Root Level

The following files remain at the root level as they are core to the package:

- `__init__.py` - Package initialization and exports
- `classes.py` - LOD classification constants (LOD2_CLASSES, LOD3_CLASSES)
- `downloader.py` - IGN LiDAR data downloading (main user-facing component)

## Backward Compatibility

All moved functionality is still available through the main `ign_lidar` module via backward compatibility imports:

```python
# These still work (legacy imports)
from ign_lidar import AdaptiveMemoryManager
from ign_lidar import ARCHITECTURAL_STYLES
from ign_lidar import STRATEGIC_LOCATIONS
from ign_lidar import analyze_tile
```

## New Recommended Import Patterns

### Core Processing

```python
from ign_lidar.core import (
    LiDARProcessor,
    AdaptiveMemoryManager,
    PerformanceMonitor,
    FeatureVerifier
)
```

### Feature Extraction

```python
from ign_lidar.features import (
    compute_normals,
    compute_curvature,
    ARCHITECTURAL_STYLES
)
```

### Preprocessing

```python
from ign_lidar.preprocessing import (
    statistical_outlier_removal,
    augment_raw_points,
    analyze_tile
)
```

### Dataset Management

```python
from ign_lidar.datasets import (
    IGNLiDARMultiArchDataset,  # requires PyTorch
    STRATEGIC_LOCATIONS,
    WORKING_TILES,
    TileInfo,
    get_tiles_by_environment
)
```

### Input/Output Operations

```python
from ign_lidar.io import (
    MetadataManager,
    simplify_for_qgis
)
```

## Updated Files

The following files were updated to reflect the new import paths:

- `ign_lidar/__init__.py` - Added backward compatibility imports
- `ign_lidar/core/__init__.py` - Exports for core modules
- `ign_lidar/features/__init__.py` - Added architectural styles exports
- `ign_lidar/datasets/__init__.py` - Added strategic locations exports
- `ign_lidar/preprocessing/__init__.py` - Added utils and analyzer exports
- `ign_lidar/core/processor.py` - Updated import paths
- `ign_lidar/downloader.py` - Updated strategic_locations import
- `ign_lidar/cli/hydra_main.py` - Updated verification import
- `ign_lidar/features/features_gpu_chunked.py` - Updated memory_manager import
- `scripts/cleanup_old_files.py` - Updated file paths

## Benefits of Reorganization

1. **Improved Modularity**: Related functionality is now grouped together
2. **Cleaner Root Level**: Root directory only contains essential package files (3 files vs 11+ previously)
3. **Better Separation of Concerns**: Core, features, preprocessing, datasets, and IO are clearly separated
4. **Maintained Compatibility**: Existing code continues to work without changes
5. **Future Extensibility**: Easy to add new modules in appropriate subdirectories
6. **Professional Structure**: Follows Python packaging best practices

## Final Package Structure

```
ign_lidar/
├── __init__.py                    # Main package exports
├── classes.py                     # LOD classification schemas
├── downloader.py                  # Main data downloader
├── cli/                          # Command-line interface
├── config/                       # Hydra configuration
├── core/                         # Core processing logic
│   ├── processor.py              # Main LiDAR processor
│   ├── memory_manager.py         # Adaptive memory management
│   ├── performance_monitor.py    # Real-time monitoring
│   ├── verification.py           # Feature verification
│   └── error_handler.py          # Enhanced error handling
├── datasets/                     # Dataset management
│   ├── strategic_locations.py    # Location database
│   ├── tile_list.py              # Working tiles
│   └── multi_arch_dataset.py     # PyTorch datasets
├── features/                     # Feature extraction
│   ├── features.py               # CPU feature computation
│   ├── features_gpu.py           # GPU acceleration
│   └── architectural_styles.py   # Style classification
├── io/                          # Input/Output operations
│   ├── metadata.py              # Metadata management
│   ├── qgis_converter.py        # QGIS compatibility
│   └── formatters/              # Data formatters
└── preprocessing/               # Data preprocessing
    ├── preprocessing.py         # Core preprocessing
    ├── utils.py                 # Patch utilities
    └── tile_analyzer.py         # Analysis tools
```

## Migration Guide

For new code, use the modular imports from subdirectories. Existing code will continue to work but may show deprecation warnings in future versions recommending the new import paths.
