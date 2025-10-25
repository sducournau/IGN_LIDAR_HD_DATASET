# Codebase Structure

## Top-Level Structure
```
IGN_LIDAR_HD_DATASET/
├── ign_lidar/          # Main package
├── tests/              # Test suite
├── examples/           # Example configs and demos
├── docs/               # Documentation (Docusaurus)
├── scripts/            # Utility scripts
├── data/               # Data directory (cache, test data)
├── conda-recipe/       # Conda packaging
├── validation_test_versailles/  # Validation data
└── pyproject.toml      # Project configuration
```

## Main Package: ign_lidar/

### Core Processing (`core/`)
- `processor.py` - Main LiDARProcessor class (entry point)
- `classification/` - Classification logic and rules framework
- `memory.py` - Memory management
- `performance.py` - Performance monitoring
- `tile_stitcher.py` - Tile stitching operations

### Feature Computation (`features/`)
- `orchestrator.py` - FeatureOrchestrator (unified API)
- `feature_computer.py` - Feature computation engine
- `compute/` - Low-level compute functions
- `strategies.py` - Strategy pattern for CPU/GPU
- `mode_selector.py` - Automatic mode selection

### Preprocessing (`preprocessing/`)
- `outliers.py` - Outlier removal
- `augmentation.py` - RGB/NIR augmentation

### I/O Operations (`io/`)
- `laz.py` - LAZ file handling
- `metadata.py` - Metadata management
- `wfs_ground_truth.py` - WFS ground truth fetching

### Configuration (`config/`)
- `schema.py` - Config schema (Hydra)
- `defaults.py` - Default configurations

### Datasets (`datasets/`)
- `multi_arch_dataset.py` - PyTorch datasets

### CLI (`cli/`)
- `main.py` - Command-line interface

### Top-Level Files
- `__init__.py` - Package initialization
- `classification_schema.py` - Classification definitions
- `downloader.py` - Data download utilities

## Key Design Patterns
1. **Strategy Pattern:** CPU/GPU feature computation
2. **Factory Pattern:** Optimization factory for adaptive processing
3. **Orchestrator Pattern:** FeatureOrchestrator unifies feature management
4. **Configuration Pattern:** Hydra-based hierarchical configuration
