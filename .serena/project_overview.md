# IGN LiDAR HD Dataset - Project Overview

## Project Identity

- **Name:** IGN LiDAR HD Processing Library
- **Version:** 3.0.0 (3.1.0 in development)
- **Type:** Python Library for LiDAR Data Processing & ML
- **License:** MIT
- **Author:** imagodata (Simon Ducournau)
- **Repository:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET

## Purpose

Transform French IGN LiDAR HD point clouds into machine learning-ready datasets with:

- Building LOD (Level of Detail) classification
- Rich geometric feature extraction
- GPU-accelerated processing
- Multi-modal data augmentation (RGB, NIR, NDVI)
- ASPRS standard classification support

## Project Structure

```
IGN_LIDAR_HD_DATASET/
├── ign_lidar/              # Main package
│   ├── core/              # Core processing (processor, memory, performance)
│   ├── features/          # Feature computation (orchestrator, strategies)
│   ├── preprocessing/     # Data preprocessing (outliers, augmentation)
│   ├── io/               # I/O operations (LAZ, metadata, ground truth)
│   ├── config/           # Configuration management (Hydra)
│   ├── datasets/         # PyTorch datasets
│   ├── cli/              # Command-line interface
│   └── classification_schema.py  # Classification definitions
├── tests/                 # Test suite
├── examples/              # Example YAML configurations
├── docs/                  # Documentation (Docusaurus)
├── scripts/               # Utility scripts
└── conda-recipe/          # Conda packaging
```

## Core Technologies

### Python Stack

- **Python:** 3.8+
- **Core:** NumPy, SciPy, scikit-learn, laspy, lazrs
- **Config:** Hydra, OmegaConf
- **Geospatial:** Shapely, GeoPandas, Rasterio, Rtree
- **ML (optional):** PyTorch, h5py

### GPU Acceleration (Optional)

- **CUDA:** 12.0+
- **CuPy:** GPU array operations
- **RAPIDS cuML:** GPU-accelerated ML algorithms
- **FAISS:** Ultra-fast k-NN search (50-100× speedup)

## Key Architectural Patterns

### 1. Strategy Pattern for GPU/CPU

Three processing strategies in `features/strategies/`:

- `strategy_cpu.py` - Standard CPU processing (scikit-learn)
- `strategy_gpu.py` - Full GPU processing (CuPy/cuML)
- `strategy_gpu_chunked.py` - Chunked GPU for large datasets

Automatic selection via `mode_selector.py`.

### 2. Orchestrator Pattern

`FeatureOrchestrator` (v3.0+) replaces legacy `FeatureManager`:

- Unified API for feature computation
- Automatic GPU/CPU/GPU_CHUNKED mode selection
- Memory-aware batch sizing
- Strategy pattern for extensibility

### 3. Configuration System

Hydra-based hierarchical configuration (v3.0):

- Base configs with inheritance
- Hardware profiles (gpu_rtx4080, cpu_high, etc.)
- Task presets (asprs_classification, lod2_buildings, etc.)
- Validation before processing

### 4. Factory Pattern

`OptimizationFactory` for adaptive processing:

- Hardware detection
- Memory profiling
- Automatic parameter tuning

## Classification System

### LOD Levels

1. **LOD2** (Simplified):

   - 12 geometric features
   - 15 building classes
   - Fast processing (~15s per 1M points)
   - Use case: Quick prototyping, baseline models

2. **LOD3** (Detailed):

   - 38 geometric features
   - 30+ architectural classes
   - Detailed processing (~45s per 1M points)
   - Use case: Architectural analysis, research

3. **ASPRS** (Standard):
   - American Society for Photogrammetry standards
   - Codes 1-43 (ground, vegetation, buildings, water, etc.)
   - Integration with IGN BD TOPO® ground truth

### Feature Modes

Defined in `features/feature_modes.py`:

- **MINIMAL** - ~8 features (ultra-fast)
- **LOD2** - ~12 features (essential)
- **LOD3** - ~38 features (complete)
- **ASPRS_CLASSES** - ~25 features (classification)
- **FULL** - All available features
- **CUSTOM** - User-defined selection

## Processing Pipeline

### Main Entry Point

`core/processor.py::LiDARProcessor` - orchestrates entire pipeline:

1. **Load** - Read LAZ tile with laspy
2. **Preprocess** - Outlier removal, normalization (optional)
3. **Compute Features** - via FeatureOrchestrator
4. **Ground Truth** - BD TOPO® classification (optional)
5. **Augmentation** - RGB/NIR from IGN services (optional)
6. **Patch Extraction** - Create training patches
7. **Save** - NPZ/HDF5/PyTorch/LAZ formats

### Processing Modes

Controlled by `processor.processing_mode`:

- `patches_only` - Training patches only (default)
- `both` - Patches + enriched LAZ tiles
- `enriched_only` - Enriched LAZ tiles only

## Feature Computation

### Geometric Features (35-45 total)

Computed in `features/compute/`:

- **Normals** - Surface orientation (nx, ny, nz)
- **Curvature** - Principal curvatures (mean, gaussian)
- **Eigenvalues** - PCA eigenvalues/vectors (λ1, λ2, λ3)
- **Shape descriptors** - Planarity, linearity, sphericity, omnivariance
- **Height features** - Absolute height, height above ground
- **Building scores** - Wall score, roof score, building probability
- **Density features** - Point density, local density
- **Architectural features** - Corner detection, edge detection

### Spectral Features (Optional)

- **RGB** - From IGN orthophotos via WMS
- **NIR** - Near-infrared from IRC service
- **NDVI** - Normalized Difference Vegetation Index

### Architectural Style (Optional)

From `features/architectural_styles.py`:

- 13 architectural styles (Haussmannian, Modern, etc.)
- Multi-scale analysis (tile, building, patch)
- Encoding options: constant, info, one_hot, embedding

## Ground Truth Integration

### BD TOPO® (IGN Vector Database)

Integration in `io/wfs_ground_truth.py`:

- Buildings (BATIMENT)
- Roads (TRONCON_DE_ROUTE)
- Vegetation (ZONE_DE_VEGETATION)
- Water (SURFACE_D_EAU, COURS_D_EAU)
- Railways (TRONCON_DE_VOIE_FERREE)
- Sports facilities (TERRAIN_DE_SPORT)
- Cemeteries (CIMETIERE)
- Power lines (LIGNE_ELECTRIQUE)
- Parking (AIRE_DE_TRAFIC)
- Bridges (PONT)

### Cadastre (Land Registry)

Integration for parcel-level classification:

- Building footprints
- Parcel boundaries
- Land use classification

### Ground Truth Optimizer

`core/classification/ground_truth_optimizer.py`:

- Automatic CPU/GPU/GPU_CHUNKED selection
- Spatial indexing (R-tree, cuSpatial)
- Batch processing for large datasets
- Geometric rule-based filtering

## Memory Management

### Adaptive Memory Manager

`core/memory.py::AdaptiveMemoryManager`:

- Runtime memory profiling
- Automatic batch size adjustment
- Garbage collection strategies
- OOM prevention

### Chunked Processing

For datasets >10M points:

- Automatic chunking
- GPU memory monitoring
- Batch size optimization
- Progressive processing with progress bars

## Performance Characteristics

### Benchmarks (v3.0)

- **GPU acceleration:** 16× faster feature computation
- **Ground truth:** 10× faster with GPU
- **Overall pipeline:** 8× speedup (80min → 10min per large tile)
- **GPU utilization:** >80% (vs 17% in legacy configs)

### Optimization Strategies

1. **KD-tree caching** - Reuse neighbor searches
2. **Ground truth caching** - Avoid repeated WFS queries
3. **Skip existing** - Resume interrupted workflows (~1800× faster)
4. **Parallel workers** - Multi-core CPU processing (avoid with GPU)
5. **Numba JIT** - Just-in-time compilation for hot paths

## Testing Strategy

### Test Organization

```
tests/
├── test_core_*.py        # Core functionality (processor, memory)
├── test_feature_*.py     # Feature computation
├── test_gpu_*.py        # GPU-specific tests
├── test_integration_*.py # End-to-end pipelines
├── test_rules_*.py      # Classification rules
└── test_modules/        # Module-specific tests
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Slow integration tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.slow` - Long-running tests

### Test Data

- `validation_test_versailles/` - Real Versailles LiDAR tiles
- `tests/test_modules/` - Synthetic test data
- `examples/` - Example configurations

## Configuration System (v3.0)

### 3-Tier Architecture

1. **Base config** - Complete defaults (`base_complete.yaml`)
2. **Hardware profiles** - GPU/CPU optimization (`hardware/*.yaml`)
3. **Task presets** - Feature sets & workflows (`task/*.yaml`)

Result: 97% smaller config files (430 lines → 15 lines)

### Available Profiles

**Hardware:**

- `gpu_rtx4080` - 16GB VRAM, 5M batch, 8 workers
- `gpu_rtx3080` - 10GB VRAM, 3M batch, 6 workers
- `cpu_high` - 64GB RAM, 2M batch, 8 workers
- `cpu_standard` - 32GB RAM, 1M batch, 4 workers

**Tasks:**

- `asprs_classification` - Full ASPRS with BD TOPO®
- `lod2_buildings` - Fast building detection
- `lod3_architecture` - Detailed architectural analysis
- `quick_enrich` - Minimal features for testing

### Config Validation

`core/classification/config_validator.py`:

- Schema validation
- Range checks
- Enum validation
- Path verification
- Early error detection

## CLI Interface

### Main Command

```bash
ign-lidar-hd process --config-file config.yaml
```

### Subcommands (v2.x legacy)

- `enrich` - Compute features only
- `patch` - Extract patches from enriched tiles
- `download` - Download IGN tiles by bbox
- `validate-config` - Validate configuration file
- `list-profiles` - Show available hardware profiles
- `list-presets` - Show available task presets

### Hydra Overrides

```bash
ign-lidar-hd process \
  --config-name my_config \
  processor.use_gpu=true \
  processor.gpu_batch_size=5000000 \
  input_dir=/data/tiles
```

## Output Formats

### Training Datasets

- **NPZ** - NumPy arrays (recommended for ML)
- **HDF5** - Hierarchical data format
- **PyTorch** - `.pt` files for PyTorch
- **Multi-format** - Combined (e.g., `npz,laz`)

### Enriched Point Clouds

- **LAZ** - Compressed LAS with extra dimensions
  - All features as extra attributes
  - Compatible with CloudCompare, QGIS
  - Warning: May show boundary artifacts in tile mode

### Metadata

`io/metadata.py::MetadataManager`:

- Processing parameters
- Feature names & counts
- Tile information
- Performance metrics
- Version tracking

## Known Issues & Pitfalls

### 1. GPU + Multiprocessing

**Problem:** CUDA context cannot be shared across processes
**Solution:** Use `num_workers=1` with GPU or disable GPU for multiprocessing

### 2. Tile Boundary Artifacts

**Problem:** Features at tile edges may be incorrect
**Solution:** Use tile stitching (`use_stitching=true`) or patches_only mode

### 3. Classification Recalculation

**Problem:** Ground truth changes classification, affecting dependent features
**Solution:** Recompute classification-dependent features after ground truth

### 4. Memory Leaks

**Problem:** Large datasets can cause memory accumulation
**Solution:** Call `gc.collect()` periodically, use chunked processing

### 5. Config Compatibility

**Problem:** v2.x configs don't work with v3.0
**Solution:** Use migration guide, legacy params still supported with warnings

## Development Workflow

### Setup

```bash
git clone https://github.com/sducournau/IGN_LIDAR_HD_DATASET
cd IGN_LIDAR_HD_DATASET
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v                    # All tests
pytest tests/ -m unit              # Unit tests only
pytest tests/ -m "not integration" # Skip slow tests
pytest tests/ --cov=ign_lidar      # With coverage
```

### Code Quality

```bash
black ign_lidar/           # Format code
flake8 ign_lidar/          # Lint code
mypy ign_lidar/            # Type check
```

### Building Docs

```bash
cd docs
npm install
npm run start              # Development server
npm run build              # Production build
```

## Migration Notes

### v2.x → v3.0

**Major changes:**

1. `FeatureManager` → `FeatureOrchestrator`
2. Manual GPU config → Automatic mode selection
3. Flat config → Hierarchical Hydra config
4. 50+ CLI params → 10 CLI params + YAML
5. `classes.py` + `asprs_classes.py` → `classification_schema.py`

**Backward compatibility:**

- Legacy imports work with deprecation warnings
- Old config format still supported
- CLI v2.x commands still functional

### v3.0 → v3.1 (Current)

**In development:**

- Rules framework for extensible classification
- Enhanced rule-based classification with confidence
- Improved documentation
- Performance optimizations

## Documentation

### Online Docs

https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

### Key Documentation Files

- `DOCUMENTATION.md` - Central navigation
- `README.md` - Main project readme
- `CHANGELOG.md` - Version history
- `MIGRATION_GUIDE.md` - v2 → v3 migration
- `docs/` - Docusaurus documentation site

### Documentation Categories

- **Guides:** Getting started, configuration, features
- **API Reference:** Python API, CLI, config schema
- **Architecture:** System design, patterns
- **Release Notes:** Version history, changes
- **Examples:** Sample configs, workflows

## Key Contributors & Contact

- **Maintainer:** Simon Ducournau (imagodata)
- **Email:** simon.ducournau@gmail.com
- **Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Discussions:** GitHub Discussions

## Citation

```bibtex
@software{ign_lidar_hd,
  author       = {Ducournau, Simon},
  title        = {IGN LiDAR HD Processing Library},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/sducournau/IGN_LIDAR_HD_DATASET},
  version      = {3.0.0}
}
```
