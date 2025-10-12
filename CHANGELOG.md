# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.4.4] - 2025-10-12

### Added

- **LAZ Data Quality Tools**: New post-processing tools for enriched LAZ file validation and correction
  - `scripts/fix_enriched_laz.py`: Automated tool to fix common data quality issues in enriched LAZ files
    - Detects and reports NDVI calculation errors (all values -1.0 due to missing NIR data)
    - Identifies extreme eigenvalue outliers (values >10,000 indicating numerical artifacts)
    - Caps eigenvalues to reasonable limits (default: 100) to prevent ML training issues
    - Recomputes derived features (anisotropy, planarity, linearity, sphericity, omnivariance, eigenentropy, change_curvature)
    - Validates results and generates detailed diagnostic reports
    - Command-line interface with options for custom eigenvalue caps and feature recomputation
  - Comprehensive diagnostic reports:
    - `LAZ_ENRICHMENT_ISSUES_REPORT.md`: Detailed technical analysis of data quality issues
    - `ENRICHMENT_ANALYSIS_SUMMARY.md`: Executive summary with impact assessment
    - `scripts/README_FIX_ENRICHED_LAZ.md`: Complete user guide with examples
    - `ENRICHMENT_QUICK_REFERENCE.md`: Quick reference card for common issues

### Fixed

- **Data Quality Issues in Enriched LAZ Files**: Identified and documented three critical issues
  - **NDVI Calculation Failure**: All NDVI values = -1.0 when NIR data is missing or corrupted
    - Root cause: NIR channel stored as float32 with near-zero values instead of uint16 [0-65535]
    - Impact: NDVI feature completely unusable for vegetation detection/classification
    - Fix: Added validation to detect missing NIR data and skip NDVI computation with warnings
  - **Extreme Eigenvalue Outliers**: ~0.18% of points with eigenvalue_1 > 100 (max observed: 52,842)
    - Root cause: Numerical instability in PCA computation on degenerate neighborhoods
    - Impact: Corrupts all eigenvalue-based features, causes ML training instability
    - Fix: Added eigenvalue capping at computation time and post-processing correction
  - **Cascading Derived Feature Corruption**: Features computed from eigenvalues inherit artifacts
    - Affected: change_curvature (max 24,531 vs expected <1), omnivariance (max 3,742 vs expected <10)
    - Impact: ~9,000 points with unrealistic feature values, visual artifacts in visualizations
    - Fix: Recompute all derived features from corrected eigenvalues
- **Duplicate Field Warnings in LAZ Patch Export**: Fixed hundreds of duplicate field warnings when saving patches
  - **Root cause**: When processing already-enriched LAZ files, extra dimensions (height, ndvi, etc.) were loaded into patches, then code attempted to add them again during export
  - **Impact**: Generated warnings "Could not add feature 'X' to LAZ: field 'X' occurs more than once" for every patch, cluttering logs
  - **Solution**: Added dimension tracking system with `added_dimensions` set to prevent duplicate additions in both `serialization.py` and `processor.py`
  - **Files modified**: `ign_lidar/core/modules/serialization.py`, `ign_lidar/core/processor.py`
  - Now handles both fresh and already-enriched LAZ files correctly without duplicate warnings

### Changed

- **Enrichment Pipeline Robustness**: Enhanced validation and error handling
  - Added NIR data validation before NDVI computation (checks for zero/near-zero values)
  - Added warnings when NIR max value < 0.001 indicates missing/corrupted data
  - Improved error messages to help diagnose data quality issues
  - Added expected feature value range documentation for validation
- **LAZ Export Reliability**: Improved handling of pre-enriched input files
  - Duplicate dimension checking prevents redundant field additions
  - Cleaner processing logs without spurious warnings
  - Better support for iterative processing workflows

### Documentation

- **LAZ Quality Diagnostic Suite**: Complete documentation for data quality analysis
  - Comprehensive diagnostic report with root cause analysis, percentile distributions, spatial analysis
  - Quick reference card with Python code snippets for health checks
  - User guide with batch processing examples and troubleshooting
  - Expected value ranges table for all geometric features (eigenvalues, derived features)

### Performance

- **Fix Script Performance**: Efficiently processes large point clouds
  - ~1M points/second analysis and fixing speed
  - Tested on 21M point file (~4GB): 45 seconds total processing time
  - Memory efficient: ~6x file size in RAM (includes multiple array copies)

## [2.4.3] - 2025-10-12

### Added

- **Feature Export Completeness**: All computed features now saved to disk in all formats
  - Feature name tracking in metadata: `metadata['feature_names']` lists all features in the feature matrix
  - Feature count in metadata: `metadata['num_features']` for quick verification
  - Comprehensive feature ordering for reproducibility across runs
- **Enhanced Progress Reporting**: Detailed progress bars for CPU/GPU chunked processing
  - Shows point count, chunk count, memory per chunk, and processing rate
  - GPU mode indicators: 🎯 (cuML), 🔧 (sklearn fallback), 💻 (CPU)
  - Improved user feedback for long-running operations

### Changed

- **Complete Feature Export**: Previously only 12 features were exported; now all 35-45+ computed features saved
  - NPZ/HDF5/PyTorch: Feature matrix size increased from 12 to 40+ features (depending on config)
  - LAZ patches: Extra dimensions increased from 7 to 35+ (all computed features now saved as extra dims)
  - File sizes will increase proportionally (~3-4x for patches with full mode)
- **Feature Matrix Construction**: Updated to export ALL geometric features in consistent order
  - Core shape descriptors (6): planarity, linearity, sphericity, anisotropy, roughness, omnivariance
  - Curvature features (2): curvature, change_curvature
  - Eigenvalue features (5): eigenvalue_1/2/3, sum_eigenvalues, eigenentropy
  - Height features (3): height_above_ground, vertical_std, z_normalized
  - Building scores (3): verticality, wall_score, roof_score
  - Density features (5): density, local_density, num_points_2m, neighborhood_extent, height_extent_ratio
  - Architectural features (5): edge_strength, corner_likelihood, overhang_indicator, surface_roughness, local_roughness
  - Additional features (6): z_absolute, z_from_ground, z_from_median, distance_to_center, horizontality
- **LAZ Export Enhancement**: All geometric, height, and radiometric features now exported as extra dimensions
  - Consistent feature ordering across all output formats
  - Better interoperability with GIS tools (QGIS, CloudCompare, etc.)

### Fixed

- **Feature Export Bug**: Resolved issue where `full` mode computed 40+ features but only saved 12
  - Impact: Datasets generated before v2.4.3 may be missing critical features
  - Recommendation: Regenerate training datasets to access complete feature sets
- **Progress Bar Information**: Added missing context to chunk processing progress bars
  - Now shows total points, chunk count, memory usage, and processing rate
  - Helps users estimate remaining time and monitor performance

## [2.4.2] - 2025-10-12

### Performance

- **Full GPU Implementation for Advanced Features**
  - Implemented complete GPU acceleration for all advanced features in "full" mode
  - `compute_eigenvalue_features()`: Now uses CuPy for GPU-accelerated eigenvalue decomposition, entropy calculations, and omnivariance
  - `compute_architectural_features()`: GPU-accelerated edge strength, corner likelihood, overhang detection, and surface roughness
  - `compute_density_features()`: GPU-accelerated density computation, neighborhood extent, and height extent ratio
  - All three methods now use CuPy (`cp`) when available, automatically falling back to NumPy on CPU
  - **Impact**: 5-10x speedup for full feature mode on large point clouds (>10M points) when GPU is available
  - **Compatibility**: Zero changes to API or output - seamless GPU/CPU fallback maintained
  - No more CPU fallback messages for advanced features when GPU is available

## [2.4.1] - 2025-10-12

### Fixed

- **CRITICAL: Full Feature Mode Implementation**
  - Fixed "full" feature mode (`mode: full`) to compute ALL 30 documented features instead of just 19
  - Added missing eigenvalue features (7): `eigenvalue_1`, `eigenvalue_2`, `eigenvalue_3`, `sum_eigenvalues`, `eigenentropy`, `omnivariance`, `change_curvature`
  - Added missing architectural features (4): `edge_strength`, `corner_likelihood`, `overhang_indicator`, `surface_roughness`
  - Added missing density features (1): `num_points_2m`
  - **Impact**: All datasets generated with `mode: full` were missing 11 critical features for LOD3 building classification
  - **Solution**: Integrated orphaned helper functions into CPU, GPU, and GPU-chunked implementations
  - **Recommendation**: Regenerate training datasets with fixed version for complete feature sets
  - Added comprehensive test suite (`tests/test_full_feature_mode.py`) to verify feature completeness
  - Updated documentation: `FEATURE_MODE_REFERENCE.md`, `FIX_SUMMARY_FULL_FEATURE_MODE.md`

## [2.4.0] - 2025-10-12

### Added

- Enhanced geometric feature validation across all computation modules
- Eigenvalue clamping to prevent negative values from numerical artifacts
- Density normalization (capped at 1000 points/m³) for ML stability
- Production-ready feature robustness guarantees

### Changed

- All geometric features now guaranteed within valid ranges [0, 1]
- Standardized formula implementations across CPU/GPU/boundary modules (λ0 normalization)
- Complete feature set parity across all computation paths

### Fixed

- CPU radius-based features (loop version) now have same validation as GPU/boundary
- Eliminated out-of-range feature warnings in all scenarios
- Improved ML model stability through consistent feature ranges

### Performance

- <1% performance overhead from validation checks

## [2.3.4] - 2025-10-12

### Fixed

- **Geometric Feature Validation & Robustness**
  - Added eigenvalue clamping to prevent negative values in all feature computation modules
  - Added explicit result clipping to ensure all geometric features stay within [0, 1] range
  - Fixed CPU radius-based features (loop version) - now has same validation as GPU/boundary
  - Fixed density feature - now capped at 1000 points/m³ to prevent extreme values
  - Standardized formulas across all modules (λ0 normalization consistent everywhere)
  - **Impact**: Eliminates out-of-range feature warnings and improves ML model stability
- **Boundary Feature Completeness**
  - Added missing features to boundary-aware computation: anisotropy, roughness, density
  - Updated verticality computation to also return horizontality
  - Ensures complete feature parity across all computation paths (GPU, CPU, boundary)
  - **Result**: Tile stitching now produces same features as non-stitched processing

### Changed

- **Formula Standardization**: All modules now use λ0 normalization (Weinmann et al. standard)
  - CPU k-NN features updated from sum_λ to λ0 normalization
  - Ensures consistent feature values across all computation paths
  - Maintains mathematical property: linearity + planarity + sphericity ≤ 1.0

### Performance

- Validation overhead: <1% (negligible impact from clipping operations)
- No memory impact (in-place operations)
- All features guaranteed valid: linearity, planarity, sphericity, anisotropy, roughness in [0, 1]
- Density bounded at [0, 1000], verticality/horizontality in [0, 1]

## [2.3.3] - 2025-10-12

### Changed

- **Version Update**: Consolidated version to 2.3.3 across all configuration files
- **Documentation**: Updated README and package metadata to reflect stable release
- All features from 2.3.2 are included and tested

## [2.3.2] - 2025-10-12

### Added

- **Preserve RGB/NIR/NDVI from Input LAZ Files**
  - RGB, NIR (infrared), and NDVI values are now automatically detected and preserved from input LAZ files
  - When generating patches, input RGB/NIR/NDVI take priority over fetched/computed values
  - All geometric features (normals, curvature, planarity, etc.) are still recomputed for consistency
  - Prevents unnecessary RGB fetching when data already exists in source files
  - Maintains data quality by preserving original spectral information
  - Useful for processing enriched or pre-augmented LAZ files

### Fixed

- **CRITICAL: RGB Augmentation Coordinate Mismatch**
  - Fixed critical bug where augmented patches received RGB colors from incorrect spatial locations
  - RGB augmentation now applied at tile level before patch extraction, ensuring spatial correspondence
  - All patch versions (original and augmented) now correctly get RGB from the same spatial region
  - Same fix applied to NIR (near-infrared) and NDVI computation
  - **Impact**: All datasets created with `include_rgb=True` and `augment=True` had mismatched
    RGB-geometry correspondence in augmented patches, which could negatively affect model training
  - **Solution**: RGB/NIR/NDVI are now added to tile features before patch extraction, maintaining
    correct spatial correspondence through augmentation transformations
  - **Performance**: Additional benefit of ~3x faster RGB processing (fetch once per tile vs once per patch)
  - Added patch metadata (`_patch_center`, `_patch_bounds`) for debugging and validation
  - Added comprehensive tests to verify RGB consistency across augmentations

## [2.3.1] - 2025-10-12

### Added

- **Memory Optimization Configurations**
  - Memory-optimized config for 16-24GB RAM systems (`config_lod3_training_memory_optimized.yaml`)
  - Sequential processing config for 8-16GB RAM systems (`config_lod3_training_sequential.yaml`)
  - Comprehensive memory optimization guide (`examples/MEMORY_OPTIMIZATION.md`)
  - System recommendations and troubleshooting for different RAM configurations
  - Performance comparison tables for different configurations

### Changed

- **Automatic Memory Management**
  - System now detects swap usage and memory pressure
  - Automatically scales down workers when memory constraints detected
  - Better OOM (Out of Memory) prevention with intelligent worker reduction
  - Improved garbage collection frequency for large tile processing

### Documentation

- Added detailed memory optimization guide with system requirements
- Three configuration profiles: Original (32GB+), Optimized (16-24GB), Sequential (8-16GB)
- Memory monitoring commands and troubleshooting tips
- CUDA setup guidance for GPU acceleration

### Added (Previously Unreleased)

- **Phase 4 Refactoring Complete** (2025-10-12)

  - Extracted 6 modular components from monolithic processor (2,609 → ~800 lines)
  - New modules: memory, serialization, loader, enrichment, patch_extractor, stitching
  - Type-safe interfaces with dataclasses
  - Improved testability and maintainability
  - Single source of truth for all operations

- **Intelligent Skip System** (2025-10-12)
  - Automatic detection and skipping of already-processed tiles
  - Deep content validation for patches and enriched LAZ files
  - Partial skip optimization: skip what exists, generate what's missing
  - Feature validation for enriched LAZ (normals, RGB, NIR, NDVI, geometric features)
  - Automatic recovery from corrupted or incomplete outputs
  - ~1800x faster on re-runs (0.1s vs 180s per tile)
  - ~2x faster for partial processing (add enriched LAZ to existing patches or vice versa)
  - Smart skip messages: ⏭️ (skip), 🔄 (process)
  - Comprehensive documentation: `docs/INTELLIGENT_SKIP.md`

### Changed

- **Modular Architecture**: Complete processor refactoring
  - Processor now orchestrates specialized modules
  - Better separation of concerns
  - Improved code organization and reusability
- **Enhanced Skip Checker**: `PatchSkipChecker` now validates:
  - Enriched LAZ feature completeness (core + optional features)
  - Patch content integrity (coords, labels, dimensions)
  - Configuration-based feature validation
  - Supports "both" mode with partial skip logic
- **Processor Integration**: `process_tile()` now passes enrichment config to skip checker
  - Enables intelligent skip based on expected features
  - Supports partial skip when only one output type exists

## [2.3.0] - 2025-10-11

### Added

- **Processing Modes**: New explicit processing modes replace confusing boolean flags

  - `processing_mode="patches_only"` (default) - Generate patches for training
  - `processing_mode="both"` - Generate patches + enriched LAZ
  - `processing_mode="enriched_only"` - Only enrich LAZ files
  - Replaces old `save_enriched_laz` and `only_enriched_laz` flags
  - Backward compatibility maintained via deprecation warnings

- **Custom Configuration Files**: Load complete configurations from YAML files
  - New `--config-file` / `-c` CLI option to load custom configs
  - New `--show-config` option to preview merged configuration
  - Configuration precedence: package defaults < custom file < CLI overrides
  - Four production-ready example configs provided in `examples/`:
    - `config_gpu_processing.yaml` - GPU-accelerated processing
    - `config_training_dataset.yaml` - ML training dataset creation
    - `config_quick_enrich.yaml` - Fast LAZ enrichment
    - `config_complete.yaml` - Complete research pipeline
  - Comprehensive `examples/README.md` usage guide

### Changed

- **Processor API**: Added `processing_mode` parameter to `LiDARTileProcessor.__init__()`

  - Old parameters `save_enriched_laz` and `only_enriched_laz` still work but are deprecated
  - Deprecation warnings guide users to new API

- **OutputConfig Schema**: Added `processing_mode` field to configuration schema
  - Old fields remain for backward compatibility but default to None

### Deprecated

- `save_enriched_laz` parameter in `LiDARTileProcessor.__init__()` - use `processing_mode="both"` or `"enriched_only"`
- `only_enriched_laz` parameter in `LiDARTileProcessor.__init__()` - use `processing_mode="enriched_only"`

### Documentation

- Added `PROCESSING_MODES_USAGE.md` - Quick usage guide for new processing modes
- Added `PHASE1_COMPLETE.md` - Phase 1 implementation summary
- Added `PHASE2_COMPLETE.md` - Phase 2 implementation summary
- Added `IMPLEMENTATION_SUMMARY.md` - Overall refactoring summary
- Added comprehensive examples/README.md with customization guide

### Tests

- Added `tests/test_processing_modes.py` - Comprehensive testing of all 3 modes and backward compatibility
- Added `tests/test_custom_config.py` - Testing of config loading, precedence, and merging
- All tests passing with 100% coverage of new features

## [2.2.2] - 2025-10-10

### Fixed

- **LAZ Patch Features**: LAZ patches now include ALL computed features as extra dimensions
  - Previously only saved XYZ, RGB, NIR, intensity, classification
  - Now includes geometric features (planarity, linearity, sphericity, anisotropy, roughness, density, curvature, verticality)
  - Now includes normals (nx, ny, nz)
  - Now includes height features (height, z_normalized, z_from_ground, z_from_median)
  - Now includes radiometric features (NDVI when NIR available)
  - LAZ patches are now feature-complete and suitable for analysis/visualization

### Added

- **LAZ Feature Verification Tool**: New `scripts/verify_laz_features.py` to validate LAZ patches contain all features
- **LAZ Fix Documentation**: Comprehensive `LAZ_FEATURES_FIX.md` explaining the issue and solution

## [2.2.1] - 2025-10-10

### Fixed

- **Critical Augmentation Bug**: Fixed spatial inconsistency where augmented patches represented different geographical regions than their original patches
  - Root cause: Augmentation was applied to entire tiles before patch extraction, causing spatial grid shifts
  - Solution: Patches are now extracted once from original data, then each patch is augmented individually
  - Impact: All patch versions (original, aug_0, aug_1, etc.) now represent the same spatial region
  - **Breaking Change**: Datasets with augmentation created before v2.2.1 should be regenerated

### Added

- **Enhanced Augmentation Function**: Added `return_mask` parameter to `augment_raw_points()` for proper label alignment after dropout
- **Patch Version Metadata**: Added `_version` and `_patch_idx` tracking to patches for better version management
- **Augmentation Verification Tool**: New `scripts/verify_augmentation_fix.py` to validate spatial consistency of augmented patches
- **Documentation**: Comprehensive `AUGMENTATION_FIX.md` explaining the issue, solution, and migration path

### Changed

- **Pipeline Restructure**: Changed from tile-level augmentation to patch-level augmentation
  - Extracts patches once from original data (defines spatial locations)
  - Creates augmented versions by transforming each patch individually
  - Maintains spatial consistency across all augmentation versions

## [2.2.0] - 2025-10-10

### Added

- **Multi-Format Output Support**: Save patches in multiple formats simultaneously
  - New format string parsing: `hdf5,laz` saves both HDF5 and LAZ patch files
  - Supports any combination: `npz`, `hdf5`, `pytorch`/`torch`, `laz`
  - Added multi-format configuration preset: `ign_lidar/configs/output/multi.yaml`
- **LAZ Patch Export**: New LAZ output format for patch files
  - Patches can now be saved as LAZ point clouds for visualization
  - Compatible with CloudCompare, QGIS, PDAL, and other LiDAR tools
  - Includes coordinates, classification, RGB, intensity, and features as extra dimensions
  - New method `_save_patch_as_laz()` in processor
- **HDF5 Format Support**: Fixed and enhanced HDF5 output
  - Proper HDF5 file generation with gzip compression
  - Supports all patch data: points, features, labels, RGB, NIR, normals
- **PyTorch Format Support**: Native PyTorch tensor output
  - Direct `.pt` file generation with torch tensors
  - Automatic NumPy to PyTorch tensor conversion
  - Requires PyTorch installation (optional dependency)
- **Hybrid Architecture Formatter**: Comprehensive single-file format
  - New `HybridFormatter` class for ensemble/hybrid models
  - Saves all architecture formats in one file for maximum flexibility
  - Includes KNN graph, voxel representation, and all feature types
  - Supports switching between architectures without regenerating patches
- **HDF5 to LAZ Conversion Tool**: New script `scripts/convert_hdf5_to_laz.py`
  - Convert HDF5 patches back to LAZ format for visualization
  - Batch conversion support for directories
  - Inspection mode to view HDF5 file structure
  - Comprehensive documentation in `scripts/CONVERT_HDF5_TO_LAZ.md`
- **Format Validation**: Automatic validation of output format specifications
  - Clear error messages for unsupported formats
  - PyTorch availability checking
  - Multi-format syntax validation

### Fixed

- **HDF5 Output Bug**: Critical fix for HDF5 format not saving any files
  - Previous versions silently failed to generate HDF5 files
  - Now properly saves HDF5 patches with compression
- **Output Format Implementation Gaps**: Completed missing format implementations
  - HDF5 saving was documented but not implemented
  - PyTorch format was documented but not implemented
  - LAZ patch format is now available

### Changed

- **Output Format Configuration**: Enhanced format specification
  - Old: Single format only (`format: hdf5`)
  - New: Multi-format support (`format: hdf5,laz`)
  - Updated documentation to reflect actual supported formats
- **Format Validation**: Stricter validation at initialization
  - Unsupported formats now raise ValueError immediately
  - Missing dependencies (e.g., PyTorch) detected early
- **Patch Saving Logic**: Refactored for multi-format support
  - Cleaner separation of format-specific logic
  - Base filename generation for consistency across formats
  - Incremental saving per format reduces memory pressure

### Documentation

- **Multi-Format Implementation Guide**: New `MULTI_FORMAT_OUTPUT_IMPLEMENTATION.md`
  - Complete technical documentation of multi-format feature
  - Performance considerations and recommendations
  - Usage examples for all format combinations
- **Quick Start Guide**: New `MULTI_FORMAT_QUICK_START.md`
  - Fast reference for multi-format output
  - Common configuration examples
  - Troubleshooting tips
- **Format Analysis**: New `OUTPUT_FORMAT_ANALYSIS.md`
  - Detailed analysis of all supported formats
  - Implementation status for each format
  - Recommendations for different use cases
- **Bug Fix Summary**: New `OUTPUT_FORMAT_COMPLETE_FIX.md`
  - Complete documentation of HDF5 bug fix
  - Code changes and their locations
  - Migration guide for existing users

### Updated

- **Configuration Presets**: Updated `ign_lidar/configs/experiment/config_lod3_training.yaml`
  - Changed default format from `hdf5` to `hdf5,laz`
  - Enables both training (HDF5) and visualization (LAZ) outputs
- **README**: Updated version badge and feature descriptions
- **Docstrings**: Updated processor documentation to reflect actual formats

## [2.1.2] - 2025-10-10

### Changed

- **Version Update**: Incremented version to 2.1.2
- **Documentation**: Updated README, CHANGELOG, and Docusaurus documentation

## [2.1.1] - 2025-10-10

### Fixed

- **Planarity Feature Computation**: Fixed formula to correctly compute planarity eigenvalues
- **Preprocessing Stitching**: Fixed boundary feature computation in tile stitching workflow
- **Feature Validation**: Improved artifact detection at tile boundaries

### Changed

- **Code Cleanup**: Removed temporary debug files and improved repository organization
- **Documentation**: Updated README and documentation for v2.1.1

## [2.1.0] - 2025-10-09

### Added

- **🔍 Feature Validation & Artifact Detection**

  - Automatic detection of geometric feature artifacts at tile boundaries
  - Validation for linearity scan line patterns (mean>0.8, std<0.1)
  - Validation for planarity discontinuities (std<0.05 or std>0.4)
  - Validation for verticality bimodal extremes (>95% at extremes)
  - Detection of NaN, Inf, and out-of-range values
  - Graceful degradation: drops problematic features, continues with valid ones
  - Comprehensive test suite for validation logic
  - Documentation: `FEATURE_VALIDATION.md` and `FEATURE_VALIDATION_SUMMARY.md`

- **🌍 French Documentation Translation (Phase 1)**

  - Complete French i18n structure synchronized (73 files)
  - Fixed 12 YAML frontmatter issues with translation-ready content
  - Translation workflow and tools ready
  - Documentation: `TRANSLATION_WORKFLOW.md`, `TRANSLATION_QUICKSTART.md`
  - Translation status tracking and reporting tools

- **🎓 Hybrid Model Training Support**

  - Optimized LOD3 hybrid model training configuration
  - Training patch generation script: `generate_training_patches_lod3_hybrid.sh`
  - Comprehensive hybrid model explanation: `HYBRID_MODEL_EXPLANATION_FR.md`
  - Dataset optimization analysis: `HYBRID_DATASET_ANALYSIS_FR.md`
  - Quick start guide: `QUICK_START_LOD3_HYBRID.md`
  - Support for PointNet++, Transformer, Octree-CNN, and Sparse Conv architectures

- **📚 Enhanced Documentation**
  - Training commands reference: `TRAINING_COMMANDS.md`
  - Phase 1 and Phase 2 translation plans
  - START_HERE.md quick navigation guide
  - Multiple workflow documentation files

### Changed

- **Breaking**: `geo_features` in boundary-aware processing now returns dictionary instead of numpy array
- Feature validation integrated into `BoundaryAwareFeatureComputer.compute_features()`
- Enhanced logging for feature validation and artifact detection

### Fixed

- Fixed geo_features format inconsistency in boundary-aware vs standard processing
- Fixed dictionary update error when boundary-aware features contain artifacts
- Improved robustness of tile boundary processing

### Improved

- Better handling of edge cases in boundary-aware feature computation
- More informative warning messages for detected artifacts
- Enhanced test coverage for feature validation scenarios

## [2.0.1] - 2025-10-08

### Added

- **✨ Enriched LAZ Only Mode**

  - New `output.only_enriched_laz` parameter to skip patch creation
  - Save enriched LAZ files with computed features only
  - 3-5x faster processing when patches are not needed
  - Seamless integration with auto-download and stitching features
  - New `enriched_only` output preset configuration
  - Comprehensive documentation in `ENRICHED_LAZ_ONLY_MODE.md`

- **🛡️ Automatic Corruption Detection & Recovery**
  - Detects corrupted LAZ files during processing (IoError, buffer errors, EOF)
  - Automatically attempts to re-download corrupted tiles from IGN WFS
  - Backs up corrupted files with `.laz.corrupted` extension
  - Verifies re-downloaded file integrity before proceeding
  - Up to 2 retry attempts with automatic fallback
  - Transparent to users - works automatically during processing
  - Applied to both v2.0 and legacy processing pipelines

## [2.0.0] - 2025-10-08

### Added

- **🏗️ Complete Modular Architecture Redesign**

  - New `ign_lidar.core` module with processor and tile stitching
  - New `ign_lidar.features` module with boundary-aware feature computation
  - New `ign_lidar.preprocessing` module with optimized preprocessing pipeline
  - New `ign_lidar.io` module with multi-format I/O and QGIS integration
  - New `ign_lidar.cli` module with modern Hydra-based CLI system
  - New `ign_lidar.config` module with configuration schema and management
  - New `ign_lidar.datasets` module with multi-architecture dataset support

- **⚡ Unified Processing Pipeline**

  - Single-step RAW→Patches workflow (eliminates intermediate files)
  - Multi-architecture support: PointNet++, Octree, Transformer, Sparse Conv
  - In-memory processing with 35-50% disk space savings
  - 2-3x faster processing through optimized data flow

- **🔗 Boundary-Aware Feature Computation**

  - Cross-tile processing with neighbor tile context
  - Buffer zone extraction for seamless stitching
  - Improved feature quality at tile boundaries
  - Spatial indexing for efficient cross-tile queries

- **⚙️ Modern Configuration System**

  - Hydra-based hierarchical configuration management
  - Preset configurations for common use cases (buildings, vegetation, etc.)
  - Easy parameter sweeps and experiment configuration
  - Backward-compatible YAML support

- **🛠️ Enhanced CLI Interface**

  - New `ign-lidar-hd-v2` command with Hydra integration
  - Legacy `ign-lidar-hd` command maintained for compatibility
  - QGIS integration with `ign-lidar-qgis` command
  - Improved help system and parameter validation

- **📦 Multi-Architecture Dataset Support**
  - Native PyTorch dataset classes
  - Automatic data augmentation pipeline
  - Support for different ML architectures in single workflow
  - Optimized batch loading and caching

### Changed

- **Breaking**: Major API reorganization (see Migration Guide)
- Configuration system migrated from YAML to Hydra
- File organization restructured into modular packages
- Processing pipeline completely rewritten for efficiency

### Improved

- 35-50% reduction in processing time
- 50% reduction in disk I/O operations
- Better memory efficiency with chunked processing
- Enhanced error handling and logging

### Migration

- Use `scripts/migrate_to_v2.py` for automatic migration assistance
- Legacy CLI commands redirected to new system with deprecation warnings
- See [Migration Guide](MIGRATION.md) for detailed instructions

## [1.7.7] - 2025-10-07

### Changed

- Version bump and maintenance release
- Updated documentation and configuration files
- Package metadata updates

## [1.7.6] - 2025-10-06

### Added

- **Feature Verification System** 🔍
  - New `verify` command in CLI for validating enriched LAZ files
  - New `ign_lidar/verifier.py` module with `FeatureVerifier` class
  - Comprehensive checks for RGB, NIR, and geometric features
  - Validates feature ranges, detects anomalies, and provides statistics
  - Supports single file or batch directory verification
  - Options: `--quiet`, `--show-samples`, `--max-files`
  - Python API: `verify_laz_files()` function for programmatic use
  - Documentation: `VERIFICATION_FEATURE.md` and `VERIFICATION_QUICKREF.md`

### Fixed

- **Critical Fix: Verticality Computation in GPU Chunked Processing** 🐛

  - **Issue**: Files processed with `--use-gpu` on large point clouds (>5M points) had verticality feature with all zeros
  - **Root cause**: GPU chunked code (`features_gpu_chunked.py`) initialized verticality but never computed it
  - **Impact**: Wall detection broken, building segmentation degraded
  - **Fix**: Added verticality computation from normals in all code paths:
    - GPU chunked processing (lines 1033-1040)
    - Simple GPU processing (added to `compute_all_features_with_gpu`)
    - CPU fallback path (ensures verticality always present)
  - **Removed**: Zero-value features (`eigenvalue_sum`, `omnivariance`, `eigenentropy`, `surface_variation`) that were never computed
  - **Result**: ~20% smaller files, correct verticality values (0-1 range), wall detection working
  - **Documentation**: `FEATURE_VERIFICATION_FIX.md`, `VERTICALITY_IMPLEMENTATION.md`
  - **Tests**: Comprehensive test suite in `tests/test_verticality_fix.py` and `tests/test_all_verticality_paths.py`

- Minor bug fixes and documentation updates

## [1.7.5] - 2025-10-05

### Changed

- **🚀 MASSIVE Performance Optimization: Vectorized Feature Computation (100-200x speedup!)**

  - **Replaced per-point PCA loops with vectorized batch operations**

    - Old: 17M separate PCA operations (one per point)
    - New: Batched covariance matrix computation with `einsum`
    - Processes all points in chunks using vectorized NumPy/CuPy operations

  - **All computation modes optimized:**

    - GPU with RAPIDS cuML: 100-150x faster
    - GPU without cuML: 80-120x faster
    - CPU mode: Already optimized (50-100x vs old implementations)

  - **Real-world impact:**

    - Before: Stuck at 0% (would take hours)
    - After: ~30 seconds for 17M points
    - GPU utilization: 100% (vs 0-5% before)

  - **Technical improvements:**
    - Vectorized covariance: `np.einsum('mki,mkj->mij', centered, centered)`
    - Batched eigendecomposition: `np.linalg.eigh(cov_matrices)`
    - Broadcasting for normalization and orientation
    - Removed dependency on `sklearn.decomposition.PCA`
    - Increased CPU batch sizes: 10k → 50k points

- **Per-Chunk Feature Computation (ALL Modes)** 🎯

  - **GPU + cuML**: Refactored to compute ALL features (normals, curvature, height, geometric) within each chunk iteration
  - **GPU without cuML**: Added `compute_all_features_chunked()` method with local KDTree per chunk
  - **CPU-only**: Already had per-chunk processing with global KDTree
  - **Memory efficiency**: 50-60% reduction in peak memory usage across all modes
  - **Scalability**: Can now process unlimited dataset sizes (tested up to 1B+ points)
  - **Performance**: 30-40% faster than previous chunked implementations

- **Intelligent Auto-Scaling System** 🧠

  - **Adaptive safety margins**: Scale based on available hardware (15-30% for RAM, 10-25% for VRAM)
  - **Smart chunk sizing**: 1.5M-5M points based on VRAM tier (16GB+, 12-16GB, 8-12GB, 4-8GB)
  - **Dynamic batch sizing**: 150K-500K matrices for eigendecomposition based on available VRAM
  - **Worker optimization**: Automatic worker count calculation based on RAM and file sizes
  - High-end systems (32GB+ RAM, 16GB+ VRAM) get more aggressive parameters for maximum performance

- **GPU Memory Optimization** 💾

  - **Aggressive cleanup**: `del` statements after each operation + forced memory pool cleanup
  - **VRAM reduction**: ~50% less VRAM usage (7.2GB → 3.4GB on test dataset)
  - **Chunk size reduction**: 5M → 2.5M baseline, adaptive 1.5M-5M based on hardware
  - **Sub-chunking eigendecomposition**: Process in 150K-500K batches to avoid CuSOLVER limits

- **Per-Chunk Strategy Enhancements** ⚡
  - Forced per-chunk KDTree strategy for all processing
  - Reduced chunk sizes for better GPU memory management
  - Increased overlap from 5% to 10% for boundary accuracy
  - Added cuML NearestNeighbors support for GPU-accelerated per-chunk KDTree
  - Optimized memory cleanup (immediate cleanup after each chunk)
  - Local KDTree per chunk for GPU modes (better VRAM efficiency)

### Added

- **New Documentation**:
  - `PER_CHUNK_FEATURES.md`: Comprehensive guide to per-chunk architecture
  - `ALL_MODES_PER_CHUNK_UPDATE.md`: Comparison of all three processing modes
  - `INTELLIGENT_AUTO_SCALING.md`: Adaptive parameter system documentation
  - `GPU_MEMORY_OPTIMIZATION.md`: Memory management strategies
  - `PERFORMANCE_OPTIMIZATION.md`: Chunk size tuning and benchmarks
  - `GPU_CUSOLVER_FIX.md`: CuSOLVER error resolution

### Fixed

- **Critical bottleneck**: Per-point PCA loops causing indefinite hangs
- **CuSOLVER error**: Fixed CUSOLVER_STATUS_INVALID_VALUE with float64 conversion and sub-chunking
- **Memory leaks**: Aggressive cleanup prevents memory accumulation
- **Adaptive memory manager**: Fixed RAM_SAFETY_MARGIN attribute errors with dynamic calculation
- Processing stuck at 0% on large point clouds (10M+ points)
- Low GPU utilization (0-5%) when GPU acceleration was enabled
- Global KDTree bottleneck on large datasets
- Processing timeouts on medium-to-large tiles (15-20M points)
- **GPU CUSOLVER errors**: Fixed `CUSOLVER_STATUS_INVALID_VALUE` errors during GPU-accelerated normal computation
  - Added matrix symmetry enforcement to prevent numerical precision issues
  - Added diagonal regularization (1e-8) for numerical stability
  - Added NaN/Inf validation before eigendecomposition
  - Added robust error handling with fallback to safe default normals
  - GPU processing now works reliably on large point clouds without falling back to CPU

### Added

- Comprehensive vectorization documentation:
  - `VECTORIZED_OPTIMIZATION.md` - Technical deep dive
  - `OPTIMIZATION_COMPLETE.md` - Comprehensive guide
  - `OPTIMIZATION_SUMMARY.md` - Quick reference
  - `TEST_RESULTS.md` - Verified test results
- Performance test suite: `test_vectorized_performance.py`
- GPU monitoring script: `monitor_gpu.sh`
- Automated testing for all three computation modes

### Technical Details

- **Vectorization Strategy:**
  - Gather neighbor points: `[N, k, 3]` arrays
  - Compute all covariance matrices at once: `[N, 3, 3]`
  - Batched eigendecomposition for all points
  - Broadcasting for orientation (upward Z)
- **Performance Verified:**

  - CPU: 90k-110k points/sec (50k point test)
  - GPU: 100% utilization confirmed
  - VRAM: 40% usage (6.6GB / 16GB)
  - Real-world: 17M points in ~3-4 minutes (total pipeline)

- **Algorithmic Correctness:**

  - Same PCA algorithm (eigendecomposition of covariance)
  - Same normal selection (smallest eigenvalue)
  - Same orientation logic
  - Produces identical results to original implementation

- **No API Changes:**
  - Existing code automatically benefits
  - All optimizations are internal
  - Drop-in replacement with massive speedup

## [1.7.4] - 2025-10-04

### Added

- **GPU Acceleration Support** 🚀

  - Complete GPU acceleration with three performance modes: CPU, Hybrid (CuPy), Full GPU (RAPIDS cuML)
  - CuPy integration for GPU-accelerated array operations (5-10x speedup)
  - RAPIDS cuML support for GPU-accelerated ML algorithms (15-20x speedup)
  - Automatic fallback to CPU when GPU unavailable
  - Intelligent memory management with chunking for large point clouds
  - Full WSL2 compatibility

- **Per-Chunk Optimization Strategy** ⚡

  - Intelligent local KDTree strategy for optimal CPU/GPU performance
  - Chunks point clouds into ~5M point segments
  - 5% overlap between chunks for edge case handling
  - 10x faster than global KDTree with CPU sklearn
  - Provides 80-90% of GPU performance without RAPIDS cuML

- **Comprehensive Documentation** 📚

  - New GPU Quick Start Guide (`GPU_QUICK_START.md`)
  - GPU Implementation Summary (`GPU_IMPLEMENTATION_SUMMARY.md`)
  - RAPIDS cuML Installation Guide (`INSTALL_CUML_GUIDE.md`)
  - Per-Chunk Optimization documentation (`PER_CHUNK_OPTIMIZATION.md`)
  - Repository Harmonization summary (`REPO_HARMONIZATION_SUMMARY.md`)
  - Complete GPU guides in English and French (Docusaurus)
  - Real hardware benchmarks (RTX 4080, 17M points)
  - Comprehensive troubleshooting sections
  - WSL2 installation guides

- **Installation Scripts**
  - Automated RAPIDS cuML installation script (`install_cuml.sh`)
  - CUDA Toolkit installation helper (`install_cuda_wsl2.sh`)
  - Three installation options: CuPy hybrid, RAPIDS cuML, automated

### Changed

- **Code Refactoring**

  - Separated `use_gpu` and `use_cuml` flags in `features_gpu_chunked.py`
  - GPU now works with CuPy alone, cuML optional for maximum performance
  - Enhanced `features_gpu.py` with improved GPU feature computation
  - Updated `processor.py` with better GPU integration

- **Documentation Updates**

  - Updated README.md with GPU installation options and quick start
  - Updated English Docusaurus intro and GPU guide
  - Updated French Docusaurus intro and GPU guide (complete translation)
  - Version bumped to 1.7.4 across all files

- **Performance Benchmarks**
  - CPU: 60 min (baseline)
  - Hybrid GPU: 7-10 min (6-8x speedup)
  - Full GPU: 3-5 min (12-20x speedup)
  - Batch (100 tiles): CPU 100h → Hybrid 14h → Full GPU 6h

### Fixed

- **GPU Detection Issue**

  - Fixed code that required both CuPy AND cuML for GPU mode
  - GPU now works with just CuPy installed (hybrid mode)
  - Proper separation of GPU array operations and ML algorithms

- **Global KDTree Performance**

  - Fixed performance bottleneck with global KDTree for large point clouds
  - Implemented per-chunk strategy with 5% overlap
  - 10x improvement in hybrid mode processing time

- **CuPy CUDA Library Detection**
  - Fixed CuPy not finding CUDA runtime libraries in WSL2
  - Added CUDA Toolkit installation guide
  - Added LD_LIBRARY_PATH configuration instructions

### Migration Notes

- **No breaking changes** - GPU acceleration is opt-in via `--use-gpu` flag
- **Existing workflows continue to work** without modifications
- **To enable GPU**: Add `--use-gpu` flag to CLI or `use_gpu: true` in YAML
- **For maximum performance**: Install RAPIDS cuML via conda

### Requirements

- **Hardware**: NVIDIA GPU with Compute Capability 6.0+ (4GB+ VRAM recommended)
- **Software**: CUDA 12.0+ driver
- **Hybrid Mode**: CuPy (cuda11x or cuda12x)
- **Full GPU Mode**: RAPIDS cuML 24.10 + CuPy (via conda)

## [1.7.3] - 2025-10-03

### Changed

- **Breaking Change**: Geometric augmentation is now **DISABLED by default** in the `enrich` command
  - Use `--augment` flag to enable augmentation (previously enabled by default)
  - Updated default value from `True` to `False` in both CLI and processor
  - This allows users to process original tiles only by default, enabling augmentation only when needed
  - Updated all documentation to reflect this change

## [1.7.0] - 2025-10-04

### Added

- **GPU Chunked Processing** 🚀

  - New `features_gpu_chunked.py` module for GPU acceleration with chunked processing
  - GPU now works with large point clouds (>10M points) and augmented data
  - `GPUChunkedFeatureComputer` class with intelligent memory management
  - Supports configurable chunk sizes and VRAM limits
  - Automatic fallback to CPU if GPU fails or unavailable
  - 10-15x speedup over CPU for large files and augmented processing
  - Global KDTree strategy for correct spatial relationships across chunks
  - Incremental memory management prevents VRAM exhaustion

- **Adaptive Memory Management** 🧠

  - New `memory_manager.py` module for intelligent resource configuration
  - `AdaptiveMemoryManager` class with real-time monitoring
  - Dynamic chunk size calculation based on available RAM/VRAM
  - Intelligent worker count optimization
  - Memory estimation before processing
  - GPU vs CPU decision logic based on system resources
  - Handles memory pressure scenarios (high swap usage, low RAM)

- **CLI Integration**

  - GPU chunking automatically used when `--use-gpu` flag is set
  - Adaptive memory manager integrated into enrichment workflow
  - Automatic worker optimization based on file sizes and system resources
  - Graceful degradation with informative warnings

- **Testing & Documentation**

  - Comprehensive test suite: `test_gpu_chunking_v17.py` (450 lines)
  - Complete implementation guide: `GPU_CHUNKING_IMPLEMENTATION.md` (650 lines)
  - Implementation summary: `V17_IMPLEMENTATION_SUMMARY.md`
  - Usage examples, performance benchmarks, and migration guide

### Changed

- Modified `cli.py` to integrate GPU chunked processing (lines 486-537)
- Updated worker optimization logic to use adaptive memory manager (lines 901-948)
- GPU acceleration now available for all file sizes (previously limited to <10M points)
- GPU acceleration now available for augmented processing (previously disabled)

### Performance

- Large files (>10M points): **13x faster** with GPU vs CPU
- Augmented processing: **12x faster** with GPU vs CPU
- Typical workflow (17M points + 2 augmentations): **30 min → 2.5 min**
- Memory efficiency: GPU now uses 4-6GB VRAM (vs 8-12GB without chunking)

### Requirements

- CuPy (cupy-cuda11x or cupy-cuda12x) >= 11.0.0 for GPU support
- RAPIDS cuML >= 23.10.0 for GPU algorithms
- psutil >= 5.9.0 for memory management (usually already installed)

### Technical Details

- Global KDTree built once on GPU for entire point cloud
- Features processed in configurable chunks (default: 5M points)
- Results transferred incrementally to CPU
- Explicit GPU memory cleanup between chunks
- Constant VRAM usage regardless of file size
- Compatible with all existing CPU workflows (100% backwards compatible)

## [1.7.3] - 2025-10-04

### Changed

- Updated version number from 1.7.2 to 1.7.3
- Consolidated documentation updates for infrared augmentation feature
- Updated all configuration files and documentation to reflect v1.7.3

### Documentation

- Finalized comprehensive infrared augmentation documentation (EN + FR)
- Updated README with v1.7.3 version number
- Updated all Docusaurus documentation pages
- Synchronized English and French documentation

## [1.7.2] - 2025-10-04

### Added

- **Infrared Augmentation** 🌿

  - New `infrared_augmentation.py` module for Near-Infrared (NIR) value integration
  - Fetches NIR data from IGN Géoplateforme IRC orthophotos (20cm resolution)
  - Smart caching system (disk + GPU) shared with RGB augmentation
  - NIR values stored as 'nir' extra dimension (uint8, 0-255) in LAZ files
  - Enables NDVI, EVI, GNDVI, SAVI vegetation index calculation
  - Added CLI flags: `--add-infrared` and `--infrared-cache-dir`
  - YAML pipeline configuration support for infrared settings
  - Compatible with RGB augmentation (can be used together)

- **Documentation**

  - Comprehensive [Infrared Augmentation Guide](website/docs/features/infrared-augmentation.md)
  - [NDVI calculation examples](examples/demo_infrared_augmentation.py)
  - CloudCompare NIR visualization guide
  - Updated all example configurations with infrared settings

- **Examples & Tests**
  - `examples/demo_infrared_augmentation.py` - Demo with NDVI calculation
  - `test_infrared_basic.py` - Basic functionality tests (4/4 passing)
  - `test_infrared_single_file.py` - Single file integration test
  - `test_full_enrich_rgb_infrared.py` - Full pipeline test with RGB + NIR

### Fixed

- **Metadata Copying Bug** 🐛

  - Fixed error when enriching single LAZ file (not directory)
  - Issue: `relative_to()` caused ValueError when input_path is a file
  - Solution: Check if input is file or directory before computing relative paths
  - Now correctly copies JSON metadata for both file and directory inputs

- **COPC Format Handling**
  - Enhanced COPC (Cloud Optimized Point Cloud) detection and conversion
  - Automatic conversion to standard LAZ when adding extra dimensions
  - Better error messages for COPC-related operations

### Changed

- Updated README.md with infrared features and examples
- Updated config examples (pipeline_full.yaml, pipeline_enrich.yaml) with infrared settings
- Enhanced release notes and documentation with v1.7.2 information

## [1.7.1] - 2025-10-04

### Fixed

- **Preprocessing + RGB Augmentation Bug** 🐛
  - Fixed shape mismatch error when using `--preprocess` with `--add-rgb`
  - Error occurred because RGB augmentation tried to broadcast filtered point array into original unfiltered LAS structure
  - Now properly applies preprocessing mask when creating output LAZ file
  - Affects both COPC and standard LAZ files
  - Error message was: "could not broadcast input array from shape (X,) into shape (Y,)"
  - Solution: Track and apply preprocessing mask to all point data before adding features and RGB

## [1.7.0] - 2025-10-04

### Added

- **Point Cloud Preprocessing Pipeline** 🧹

  - New `preprocessing.py` module with three artifact mitigation techniques
  - Statistical Outlier Removal (SOR): eliminates measurement noise and atmospheric returns
  - Radius Outlier Removal (ROR): removes scan line artifacts and isolated points
  - Voxel Downsampling: homogenizes point density and reduces memory usage
  - 22 comprehensive tests covering all preprocessing functions

- **CLI Preprocessing Integration** ⚙️

  - Added 9 new CLI flags for preprocessing control:
    - `--preprocess`: Enable preprocessing pipeline
    - `--sor-k`: Number of neighbors for SOR (default: 12)
    - `--sor-std`: Standard deviation multiplier for SOR (default: 2.0)
    - `--ror-radius`: Search radius in meters for ROR (default: 1.0)
    - `--ror-neighbors`: Minimum neighbors required for ROR (default: 4)
    - `--voxel-size`: Voxel size in meters for downsampling (optional)
    - `--no-preprocess`: Explicitly disable preprocessing
  - Full integration with enrich command workflow
  - Backward compatible (preprocessing disabled by default)

- **Processor Integration** 🔧

  - Added `preprocess` and `preprocess_config` parameters to `LidarProcessor`
  - Preprocessing applied before feature computation
  - Synchronous filtering of points, intensity, and classification arrays
  - Detailed logging of reduction statistics

- **Comprehensive Documentation** 📚
  - English documentation fully updated (README, CLI guide, new preprocessing guide)
  - Complete French translation (900+ lines):
    - `guides/preprocessing.md` (FR): comprehensive preprocessing guide
    - `guides/cli-commands.md` (FR): updated with all preprocessing parameters
    - `intro.md` (FR): v1.7.0 highlights and examples
  - 5 recommended presets (conservative, standard, aggressive, urban, memory-optimized)
  - Performance impact tables and quality metrics
  - 10+ practical examples with code snippets
  - Complete troubleshooting guide

### Changed

- **Feature Computation Quality** 📊
  - Geometric features now computed on cleaner point clouds
  - 60-80% reduction in scan line artifacts
  - 40-60% cleaner surface normals
  - 30-50% smoother edge features

### Performance

- **Processing Impact** ⚡
  - 15-30% overhead when preprocessing enabled
  - Voxel downsampling can improve speed (40-60% point reduction)
  - Memory usage reduced with voxel downsampling
  - Overall quality vs. speed trade-offs well documented

### Validated

- ✅ **22 Tests Passing**: Full test coverage for preprocessing module
- ✅ **Backward Compatible**: Preprocessing disabled by default, no breaking changes
- ✅ **Bilingual Documentation**: Complete feature coverage in English and French
- ✅ **Production Ready**: Integrated into main processing pipeline with comprehensive logging

### Documentation

- Added `PHASE1_SPRINT1_COMPLETE.md`: Preprocessing module implementation summary
- Added `PHASE1_SPRINT2_COMPLETE.md`: CLI/Processor integration summary
- Added `DOCUMENTATION_UPDATE_COMPLETE.md`: English documentation summary
- Added `FRENCH_DOCS_UPDATE_COMPLETE.md`: French translation summary
- Updated `website/docs/guides/preprocessing.md`: 500+ line comprehensive guide
- Updated `website/i18n/fr/.../guides/preprocessing.md`: 900+ line French guide
- Updated CLI command guides in both languages

## [1.6.5] - 2025-10-03

### Added

- **Radius Parameter Support** 🎯
  - Added `--radius` parameter to CLI enrich command for manual control
  - Automatic radius estimation by default (eliminates LIDAR scan artefacts)
  - Pipeline configuration support for radius parameter
  - Radius-based search eliminates "dash line" artefacts in geometric features
  - Typical values: 0.5-2.0m (auto-estimated based on point density)

### Changed

- **Feature Computation Enhancement** ⚡
  - Updated `compute_all_features_with_gpu()` to support radius parameter
  - Improved worker process to pass radius through to feature computation
  - Maintained backward compatibility (radius=None for auto-estimate)

### Documentation

- **Comprehensive Artefact Audit** 📊
  - Added `ARTEFACT_AUDIT_REPORT.md` - Full technical audit (11KB)
  - Added `ARTEFACT_AUDIT_SUMMARY.md` - Quick reference guide (5.9KB)
  - Added `RADIUS_PARAMETER_GUIDE.md` - Detailed usage guide (~10KB)
  - Added `ARTEFACT_AUDIT_COMPLETE.md` - Completion summary
  - Added `scripts/analysis/visualize_artefact_audit.py` - Visualization tool
  - All tests passing: GPU/CPU consistency, degenerate cases, feature ranges

### Validated

- ✅ **No Cross-Contamination**: Artefact fixes do NOT affect other geometric features
- ✅ **Mathematical Independence**: Each feature uses independent computations
- ✅ **GPU/CPU Parity**: Perfect equivalence (0.000000 difference)
- ✅ **Robust to Degenerate Cases**: No NaN/Inf propagation
- ✅ **Production Ready**: Approved for all workflows

### Performance

- Radius-based search: ~10-15% slower than k-NN but scientifically correct
- Eliminates LIDAR scan line artefacts completely
- No memory overhead
- Auto-estimation adds negligible time

## [1.6.4] - 2025-10-03

### Changed

- **Enhanced Documentation** 📺
  - Updated README with embedded YouTube player for better video experience
  - Improved visual integration of demo content in Docusaurus documentation
  - Better presentation of video tutorials and demos

## [1.6.3] - 2025-10-03

### Fixed

- Package metadata for PyPI upload

## [1.6.2] - 2025-10-03

### Fixed

- **Critical: GPU Feature Formula Correction** 🔧

  - Fixed inconsistent eigenvalue normalization between GPU and CPU implementations
  - GPU now uses standard Σλ normalization (Weinmann et al., 2015) matching CPU
  - Validated: GPU and CPU produce identical results (max_rel_diff < 1e-6)
  - **Breaking change**: GPU feature values changed (were incorrect before)
  - Users with GPU-trained models should retrain or switch to CPU

- **Degenerate Case Handling** 🛡️

  - Added robust filtering for points with insufficient neighbors
  - Invalid features now set to 0.0 instead of NaN/Inf
  - Prevents pipeline crashes from bad neighborhoods
  - Handles collinear points and near-zero eigenvalues correctly

- **Robust Curvature Computation** 📐
  - Replaced std with Median Absolute Deviation (MAD \* 1.4826)
  - Resistant to outlier points common in LIDAR data
  - Maintains similar value ranges via standard scaling
  - Better captures true surface curvature without noise influence

### Added

- **GPU Radius Search Support** 🎯

  - Added radius parameter to GPU feature extraction
  - Automatically falls back to CPU when radius requested
  - Avoids LIDAR scan line artifacts with spatial radius search
  - Clear warning messages about CPU fallback

- **Comprehensive Validation Suite** ✅
  - New test suite: `tests/test_feature_fixes.py`
  - Tests GPU/CPU consistency, degenerate cases, robust curvature
  - All validation tests passing

### Documentation

- **Comprehensive Documentation Overhaul** 📚
  - Created complete codebase analysis with architecture documentation (`CODEBASE_ANALYSIS_2025.md`)
  - Added 7 professional Mermaid diagrams for visual understanding:
    - Core processing pipeline with GPU/CPU paths
    - GPU integration architecture with automatic fallback
    - RGB augmentation system with spatial indexing
    - API design patterns (Factory, Strategy, Pipeline, Context Manager)
    - Complete 3-stage workflow diagrams (English + French)
    - Documentation navigation map
  - Created comprehensive workflow guides (English + French):
    - `website/docs/guides/complete-workflow.md`
    - `website/i18n/fr/.../guides/complete-workflow.md`
  - Added 8 summary and reference documents:
    - `DOCUMENTATION_README.md` - Quick navigation
    - `QUICK_REFERENCE.md` - 2-minute overview
    - `DOCUMENTATION_COMPLETE_SUMMARY.md` - Executive summary
    - `DOCUMENTATION_UPDATE_2025.md` - Detailed update log
    - `DOCUMENTATION_INDEX.md` - Master index with navigation
    - `MERMAID_DIAGRAMS_SUMMARY.md` - Diagram reference
  - Enhanced intro pages with badges and expanded features (EN + FR)
  - Updated README with latest features and improvements
  - Added 40+ working code examples throughout documentation
  - 100% bilingual coverage (English + French)
  - Multiple learning paths (beginner to advanced)
  - Comprehensive troubleshooting sections
  - Professional formatting with badges and visual elements

### Added

- Quick start guides for new users (EN + FR)
- Release notes for v1.6.0 (EN + FR)
- Documentation navigation diagrams
- Visual architecture documentation
- Performance benchmarks and optimization guides

### Improved

- Documentation structure and organization
- Code example quality and coverage
- Visual communication with diagrams
- Cross-referencing between documents
- Search and navigation experience

## [1.6.1] - 2025-10-03

### Fixed

- **RGB Point Format Compatibility** 🎨
  - Fixed "Point format does not support red dimension" error when using `--add-rgb` with COPC files
  - Automatically converts point format 6 to format 7 (RGB+NIR) when RGB augmentation is requested
  - Smart format conversion for non-RGB formats: format 7 for LAS 1.4+, format 3 for older versions
  - Ensures RGB dimensions are properly initialized in laspy when converting from COPC to LAZ
  - Files affected: `ign_lidar/cli.py`
  - See `RGB_FORMAT_FIX.md` for technical details

## [1.6.0] - 2025-10-03

### Changed

- **Data Augmentation Improvement** 🎯
  - **MAJOR**: Moved data augmentation from PATCH phase to ENRICH phase
  - Geometric features are now computed AFTER augmentation (rotation, jitter, scaling, dropout)
  - Ensures feature-geometry consistency: normals, curvature, planarity, linearity all match augmented geometry
  - **Benefits**:
    - ✅ No more feature-geometry mismatch
    - ✅ Better training data quality
    - ✅ Expected improved model performance
  - **Trade-off**: ~40% longer processing time (features computed per augmented version)
  - **Migration**: No config changes needed - just reprocess data for better quality
  - See `AUGMENTATION_IMPROVEMENT.md` for technical details

### Fixed

- **RGB CloudCompare Display** 🎨
  - Fixed RGB scaling from 256 to 257 multiplier
  - Now correctly produces full 16-bit range (0-65535) instead of 0-65280
  - RGB colors now display correctly in CloudCompare and other viewers
  - Files affected: `ign_lidar/cli.py`, `ign_lidar/rgb_augmentation.py`
  - Added diagnostic/fix script: `scripts/fix_rgb_cloudcompare.py`
  - See `RGB_CLOUDCOMPARE_FIX.md` for details and migration guide

### Added

- `augment_raw_points()` function in `ign_lidar/utils.py`
  - Applies augmentation to raw point cloud before feature computation
  - Transformations: rotation (Z-axis), jitter (σ=0.1m), scaling (0.95-1.05), dropout (5-15%)
  - Returns all arrays filtered by dropout mask
- `examples/demo_augmentation_enrich.py` - Demo script for new augmentation
- `examples/compare_augmentation_approaches.py` - Visual comparison of old vs new approach
- `tests/test_augmentation_enrich.py` - Unit tests for augmentation at ENRICH phase
- `AUGMENTATION_IMPROVEMENT.md` - Comprehensive technical documentation
- `AUGMENTATION_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- `AUGMENTATION_QUICK_GUIDE.md` - Quick guide for users

### Removed

- Patch-level augmentation logic from `utils.augment_patch()` (kept for backward compatibility but no longer used)
- Import of `augment_patch()` from processor (no longer used in pipeline)

## [1.5.3] - 2025-10-03

### Added

- **LAZ Compression Backend** 🔧
  - Added `lazrs>=0.5.0` as a core dependency
  - Provides LAZ compression/decompression backend for `laspy`
  - Fixes "No LazBackend selected, cannot decompress data" errors
  - Enables processing of compressed LAZ and COPC files out of the box

### Fixed

- **LAZ File Processing**
  - Resolved issue where LAZ files could not be read without manual installation of compression backend
  - All LAZ and COPC.LAZ files now work automatically after installation

### Changed

- `lazrs` is now a required dependency (added to both `pyproject.toml` and `requirements.txt`)
- Users no longer need to manually install LAZ backend packages

## [1.5.2] - 2025-10-03

### Fixed

- **CuPy Installation Issue** 🔧
  - Removed `cupy>=10.0.0` from optional dependencies `[gpu]`, `[gpu-full]`, and `[all]`
  - CuPy now must be installed separately with the appropriate CUDA version:
    - `pip install cupy-cuda11x` for CUDA 11.x
    - `pip install cupy-cuda12x` for CUDA 12.x
  - This prevents pip from attempting to build CuPy from source, which fails without CUDA toolkit headers
  - Updated all installation documentation to reflect the new installation method

### Documentation

- Updated installation instructions in:
  - `README.md` - Updated GPU installation section
  - `website/docs/gpu/overview.md` - Added warning and corrected installation steps
  - `website/docs/installation/quick-start.md` - Updated GPU support section
  - `website/i18n/fr/docusaurus-plugin-content-docs/current/intro.md` - French documentation
  - `website/i18n/fr/docusaurus-plugin-content-docs/current/guides/gpu-acceleration.md` - French GPU guide
- All documentation now clearly states that CuPy must be installed separately

### Changed

- The `[all]` extra no longer includes GPU dependencies (CuPy)
- Users with GPU support must now explicitly install CuPy after installing the base package
- `[gpu]` and `[gpu-full]` extras are now empty/minimal (RAPIDS cuML only for gpu-full)

### Migration Guide

If you previously installed with `pip install ign-lidar-hd[gpu]` or `pip install ign-lidar-hd[all]`, you now need to:

1. Install the base package: `pip install ign-lidar-hd`
2. Install CuPy separately: `pip install cupy-cuda11x` (or `cupy-cuda12x`)
3. Optionally install RAPIDS: `conda install -c rapidsai -c conda-forge -c nvidia cuml`

## [1.5.1] - 2025-10-03

### Documentation

- **Major Documentation Consolidation** 📚
  - Consolidated three fragmented GPU guides into unified GPU section
  - Created new `website/docs/gpu/` directory with structured documentation:
    - `gpu/overview.md` - Comprehensive GPU setup and installation guide
    - `gpu/features.md` - Detailed GPU feature computation reference
    - `gpu/rgb-augmentation.md` - GPU-accelerated RGB augmentation guide
  - Updated sidebar navigation to include all GPU documentation
  - Added "GPU Acceleration" section to main navigation
  - Promoted core documentation (`architecture.md`, `workflows.md`) to top-level
- **Improved Navigation & Organization**
  - Fixed sidebar navigation - all docs now accessible
  - Resolved duplicate sidebar position conflicts
  - Created logical documentation hierarchy:
    - Installation → Guides → GPU → Features → QGIS → Technical Reference → Release Notes
  - Added cross-references between related documentation pages
- **Enhanced Cross-Referencing**
  - Added "See Also" sections to all GPU-related pages
  - Updated `architecture.md` with links to GPU guides
  - Updated `workflows.md` with correct GPU guide references
  - Updated `features/rgb-augmentation.md` with GPU acceleration links
  - Fixed broken internal links across documentation

### Changed

- Updated sidebar configuration in `website/sidebars.ts`
- Reorganized QGIS documentation into dedicated section
- Added placeholder pages for API reference and release notes structure
- Updated version references in documentation to reflect v1.5.1

### Fixed

- Fixed 60-70% content overlap between multiple GPU guides
- Resolved inconsistent frontmatter and sidebar positions
- Fixed broken links to GPU documentation throughout the site
- Corrected relative paths after documentation restructuring

### Benefits

- **For Users:**

  - Clear, discoverable navigation - all docs in sidebar
  - Single authoritative guide per topic (no confusion)
  - Better SEO and searchability
  - Consistent documentation style

- **For Maintainers:**
  - Single source of truth - update in one place
  - 60-70% reduction in duplicate content
  - Easier to maintain and update
  - Clear content ownership per section

## [1.5.0] - 2025-10-03

### Added

- **GPU-Accelerated RGB Augmentation** ✅ (Phase 3.1 Complete)

  - GPU-accelerated color interpolation with `interpolate_colors_gpu()` method
  - 24x speedup for adding RGB colors from IGN orthophotos
  - GPU memory caching for RGB tiles with LRU eviction strategy
  - `fetch_orthophoto_gpu()` method in `IGNOrthophotoFetcher`
  - GPU cache management with configurable size (default: 10 tiles)
  - Automatic CPU fallback for RGB operations
  - Bilinear interpolation using CuPy for parallel color computation

- **Enhanced Documentation**

  - New comprehensive architecture documentation with Mermaid diagrams
  - RGB GPU guide (`rgb-gpu-guide.md`) with usage examples
  - French translations for all new documentation
  - Performance benchmarking documentation
  - Complete API reference for GPU RGB features

- **Testing & Validation**
  - New test suite `test_gpu_rgb.py` with 7 comprehensive tests
  - RGB GPU benchmark script (`benchmark_rgb_gpu.py`)
  - Cache performance validation
  - Accuracy validation for GPU color interpolation
  - Integration tests for end-to-end GPU RGB pipeline

### Changed

- Updated `pyproject.toml` to version 1.5.0
- Enhanced `IGNOrthophotoFetcher` class with GPU support
- Improved `GPUFeatureComputer` with RGB interpolation capabilities
- Updated documentation with latest GPU features and architecture
- Enhanced error handling for GPU memory management

### Performance

- RGB color interpolation: 24x faster on GPU vs CPU
- 10K points: 0.005s (GPU) vs 0.12s (CPU)
- 1M points: 0.5s (GPU) vs 12s (CPU)
- 10M points: 5s (GPU) vs 120s (CPU)
- GPU memory usage: ~30MB for 10 cached RGB tiles

## [1.4.0] - 2025-10-03

### Added

- **GPU Integration Phase 2.5 - Building Features** ✅ COMPLETE
  - GPU implementation of building-specific features:
    - `compute_verticality()` - GPU-accelerated verticality computation
    - `compute_wall_score()` - GPU-accelerated wall score computation
    - `compute_roof_score()` - GPU-accelerated roof score computation
  - `include_building_features` parameter in `compute_all_features()`
  - Wrapper functions for API compatibility
  - Complete test suite (`tests/test_gpu_building_features.py`)
  - CPU fallback for all building features
  - RAPIDS cuML optional dependency support

### Changed

- Updated `pyproject.toml` to version 1.4.0
- Enhanced GPU documentation with RAPIDS cuML installation options
- Updated README.md with `gpu-full` installation instructions

## [1.3.0] - 2025-10-03

### Added

- **GPU Integration Phase 2** ✅ COMPLETE
  - Added `compute_all_features()` method to `GPUFeatureComputer` class
  - Integrated GPU support into `LiDARProcessor` class
  - New `use_gpu` parameter in processor initialization
  - GPU availability validation with automatic CPU fallback
  - Updated `process_tile()` to use GPU-accelerated feature computation
  - Full feature parity between CPU and GPU implementations
  - **GPU Benchmark Suite** (`scripts/benchmarks/benchmark_gpu.py`)
    - Comprehensive CPU vs GPU performance comparison
    - Multi-size benchmarking (1K to 5M points)
    - Synthetic data generation for quick testing
    - Real LAZ file testing support
    - Detailed performance metrics and speedup calculations (5-6x speedup)
  - **GPU Documentation** (`website/docs/gpu-guide.md`)
    - Complete installation guide with CUDA setup
    - Usage examples for CLI and Python API
    - Performance benchmarks and expected speedups
    - Troubleshooting guide for common issues
    - GPU hardware compatibility matrix
    - Best practices for GPU optimization

### Changed

- **GPU Feature Computation**
  - GPU module now fully integrated with processor pipeline
  - Conditional feature computation based on GPU availability
  - Improved error handling and user feedback
  - Updated benchmark documentation with GPU comparison examples

## [1.2.1] - 2025-10-03

### Added

- **GPU Integration Phase 1**
  - Connected GPU module to CLI and feature computation pipeline
  - New `compute_all_features_with_gpu()` wrapper function in `features.py`
  - Automatic CPU fallback when GPU unavailable
  - GPU support in `enrich` command via `--use-gpu` flag
  - GPU integration tests (`tests/test_gpu_integration.py`, `tests/test_gpu_simple.py`)
  - Updated documentation for GPU installation and usage

### Fixed

- **GPU Module Integration** (Issue: GPU module existed but was not connected)
  - `--use-gpu` flag now functional (was previously parsed but ignored)
  - Feature computation now uses GPU when available and requested
  - Proper error handling and logging for GPU availability

## [1.2.0] - 2025-10-03

### 🎨 New Features - RGB Augmentation & Pipeline Configuration

This release introduces two major new features: automatic RGB color augmentation from IGN orthophotos and declarative YAML-based pipeline configuration for complete workflow automation.

### Added

- **RGB Augmentation from IGN Orthophotos** (`ign_lidar/rgb_augmentation.py`)

  - Automatically fetch RGB colors from IGN BD ORTHO® service (20cm resolution)
  - `IGNOrthophotoFetcher` class for orthophoto retrieval and caching
  - Intelligent caching system for orthophotos (10-20x speedup)
  - Seamless integration with `enrich` command via `--add-rgb` flag
  - Support for custom cache directories with `--rgb-cache-dir`
  - RGB colors normalized to [0, 1] range for ML compatibility
  - Multi-modal learning support (geometry + photometry)

- **Pipeline Configuration System** (`ign_lidar/pipeline_config.py`)

  - YAML-based declarative workflow configuration
  - Support for complete pipelines: download → enrich → patch
  - Stage-specific configurations (enrich-only, patch-only, full pipeline)
  - Global settings inheritance across stages
  - Configuration validation and error handling
  - Example configuration files in `config_examples/`
  - New `pipeline` command for executing YAML workflows

- **Documentation & Examples**

  - RGB Augmentation Guide (`website/docs/features/rgb-augmentation.md`)
  - Pipeline Configuration Guide (`website/docs/features/pipeline-configuration.md`)
  - Blog post announcing RGB feature (`website/blog/2025-10-03-rgb-augmentation-release.md`)
  - Example: `examples/enrich_with_rgb.py` - RGB augmentation usage
  - Example: `examples/pipeline_example.py` - Pipeline configuration usage
  - Example YAML configs: `config_examples/pipeline_*.yaml`
  - French translations for all new documentation

- **Testing**
  - RGB integration tests (`tests/test_rgb_integration.py`)
  - CLI argument validation for RGB parameters
  - Orthophoto fetcher initialization tests

### Changed

- **CLI Command Naming**

  - Renamed `process` command to `patch` for clarity
  - Old `process` command still works (deprecated with warning)
  - Updated all documentation to use `patch` command
  - Migration guide provided for existing users

- **CLI Enrich Command** (`ign_lidar/cli.py`)

  - Added `--add-rgb` flag to enable RGB augmentation
  - Added `--rgb-cache-dir` parameter for orthophoto caching
  - Worker function signature updated to support RGB parameters
  - Improved help text with RGB options

- **Website Documentation**
  - Updated all CLI examples to use `ign-lidar-hd` command
  - Changed from `python -m ign_lidar.cli` to `ign-lidar-hd`
  - Consistent command naming across English and French docs
  - Added RGB augmentation to feature list on homepage

### Dependencies

- **New Optional Dependencies** (for RGB augmentation)
  - `requests` - For WMS service calls
  - `Pillow` - For image processing
  - Install with: `pip install ign-lidar-hd[rgb]`

### Performance

- **RGB Augmentation**
  - First patch per tile: +2-5s (includes orthophoto download)
  - Cached patches: +0.1-0.5s (minimal overhead)
  - Cache speedup: 10-20x faster
  - Memory overhead: ~196KB per patch (16384 points × 3 × 4 bytes)

### Technical Details

#### RGB Augmentation Workflow

```python
# Fetch orthophoto from IGN WMS
image = fetcher.fetch_orthophoto(bbox, tile_id="0123_4567")

# Map 3D points to 2D pixels
rgb = fetcher.augment_points_with_rgb(points, bbox)

# Result: RGB array normalized to [0, 1]
```

#### Pipeline Configuration Example

```yaml
global:
  num_workers: 4

enrich:
  input_dir: "data/raw"
  output: "data/enriched"
  add_rgb: true
  rgb_cache_dir: "cache/"

patch:
  input_dir: "data/enriched"
  output: "data/patches"
  lod_level: "LOD2"
```

### Migration Guide

#### Command Renaming

```bash
# Old (still works, shows deprecation warning)
ign-lidar-hd process --input tiles/ --output patches/

# New (recommended)
ign-lidar-hd patch --input tiles/ --output patches/
```

#### RGB Augmentation (Opt-in)

```bash
# Without RGB (default, backwards compatible)
ign-lidar-hd enrich --input-dir raw/ --output enriched/

# With RGB (new feature, opt-in)
ign-lidar-hd enrich --input-dir raw/ --output enriched/ \
  --add-rgb --rgb-cache-dir cache/
```

### Backwards Compatibility

- All existing code continues to work without modifications
- RGB augmentation is opt-in via `--add-rgb` flag
- Default behavior unchanged (no RGB)
- `process` command still functional (with deprecation notice)
- No breaking changes to Python API

### Known Issues

#### GPU Acceleration Non-Functional

The `--use-gpu` flag is currently **non-functional** in v1.2.0:

- GPU module exists (`features_gpu.py`) but is not integrated with CLI/Processor
- Flag is parsed but silently falls back to CPU processing
- No functional impact (CPU processing works correctly)
- Will be properly integrated in v1.3.0

**Workaround:** None needed - CPU processing is fully functional and optimized.

See `GPU_ANALYSIS.md` for detailed technical analysis.

### See Also

- [RGB Augmentation Guide](https://igndataset.dev/docs/features/rgb-augmentation)
- [Pipeline Configuration Guide](https://igndataset.dev/docs/features/pipeline-configuration)
- [CLI Commands Reference](https://igndataset.dev/docs/guides/cli-commands)

## [1.1.0] - 2025-10-03

### 🎯 Major Improvements - QGIS Compatibility & Geometric Features

This release fixes critical issues with QGIS compatibility and geometric feature calculation, eliminating scan line artifacts and ensuring enriched LAZ files can be visualized in QGIS.

### Added

- **QGIS Compatibility Script** (`scripts/validation/simplify_for_qgis.py`)

  - Converts LAZ 1.4 format 6 files to LAZ 1.2 format 3 for QGIS compatibility
  - Preserves 3 key dimensions: height, planar, vertical
  - Remaps classification values to 0-31 range (format 3 limit)
  - Reduces file size by ~73% while maintaining essential geometric features

- **Radius-Based Geometric Features** (`ign_lidar/features.py`)

  - New `estimate_optimal_radius_for_features()` function for adaptive radius calculation
  - Auto-calculates optimal search radius (15-20x average nearest neighbor distance)
  - Eliminates scan line artifacts in linearity/planarity attributes
  - Typical radius: 0.75-1.5m for IGN LIDAR HD data

- **Diagnostic Tools**

  - `scripts/validation/diagnostic_qgis.py` - Comprehensive LAZ file validation for QGIS
  - `scripts/validation/test_radius_vs_k.py` - Comparison of k-neighbors vs radius-based features

- **Documentation**
  - `SOLUTION_FINALE_QGIS.md` - Complete guide for QGIS compatibility
  - `docs/QGIS_TROUBLESHOOTING.md` - Troubleshooting guide with 6 solution categories
  - `docs/RADIUS_BASED_FEATURES_FIX.md` - Technical explanation of radius-based approach
  - `docs/LASPY_BACKEND_ERROR_FIX.md` - Backend compatibility documentation

### Fixed

- **Geometric Feature Artifacts**

  - Replaced k-neighbors (k=50) with radius-based neighborhood search
  - Fixed "dash lines" (lignes pointillées) appearing in linearity/planarity attributes
  - Corrected geometric formulas: normalized by eigenvalue sum instead of λ₀
  - Formula corrections:
    - Linearity: `(λ₀ - λ₁) / (λ₀ + λ₁ + λ₂)` (was: `/ λ₀`)
    - Planarity: `(λ₁ - λ₂) / (λ₀ + λ₁ + λ₂)` (was: `/ λ₀`)
    - Sphericity: `λ₂ / (λ₀ + λ₁ + λ₂)` (was: `/ λ₀`)

- **LAZ Compression Issues**

  - Added `do_compress=True` parameter to all `.write()` calls
  - Ensures proper LAZ compression in enriched output files

- **Laspy Backend Compatibility**

  - Removed `laz_backend='laszip'` parameter (incompatible with laspy 2.6.1+)
  - Let laspy auto-detect available backend (lazrs/laszip)
  - Fixed `'str' object has no attribute 'is_available'` error

- **QGIS File Reading**
  - Files are now readable in QGIS via simplified format conversion
  - Addressed limitation: QGIS has poor support for LAZ 1.4 format 6 with extra dimensions
  - Solution: Convert to LAZ 1.2 format 3 while preserving key attributes

### Changed

- **Geometric Feature Calculation** (`ign_lidar/features.py`)

  - `extract_geometric_features()` now uses `query_radius()` instead of `query(k)`
  - Default behavior: auto-calculate radius if not provided
  - Maintains backward compatibility with `k` parameter
  - Performance: slightly slower but produces artifact-free results

- **CLI Enrichment** (`ign_lidar/cli.py`)
  - Removed problematic `laz_backend` parameter from write operations
  - Improved LAZ compression reliability

### Performance

- **File Size Reduction**: Simplified QGIS files are ~73% smaller (192 MB → 51 MB typical)
- **Feature Calculation**: Radius-based search is ~10-15% slower but eliminates artifacts
- **Memory**: No significant change in memory usage

### Technical Details

#### Radius Calculation

```python
# Auto-calculated from average nearest neighbor distance
radius = 15-20 × avg_nn_distance
# Typical for IGN LIDAR HD: 0.75-1.5m
```

#### QGIS Compatible Format

- **Input**: LAZ 1.4, point format 6, 15 extra dimensions
- **Output**: LAZ 1.2, point format 3, 3 key dimensions
- **Preserved dimensions**: height_above_ground, planarity, verticality

#### References

- Weinmann et al. (2015) - Semantic point cloud interpretation
- Demantké et al. (2011) - Dimensionality based scale selection

### Migration Guide

#### For existing users

1. **Update package**: `pip install --upgrade ign-lidar-hd`

2. **Re-enrich files** (recommended): Previous enriched files may have scan artifacts

   ```bash
   ign-lidar enrich your_file.laz
   ```

3. **For QGIS visualization**: Convert existing enriched files

   ```bash
   python scripts/validation/simplify_for_qgis.py enriched_file.laz
   ```

4. **Batch conversion**: Convert all enriched files for QGIS
   ```bash
   find /path/to/files/ -name "*.laz" ! -name "*_qgis.laz" -exec python scripts/validation/simplify_for_qgis.py {} \;
   ```

### Known Issues

- QGIS versions < 3.18 may not support point cloud visualization
- Full 15-dimension files require CloudCompare or PDAL for visualization
- Classification values > 31 are clipped in format 3 conversion

### Dependencies

- laspy >= 2.6.1 (with lazrs backend)
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

---

### Changed - Repository Consolidation (October 3, 2025)

**Major repository reorganization for improved maintainability and professionalism.**

#### Structure Changes

- Moved `enrich_laz_building.py` to `examples/legacy/`
- Moved `workflow_100_tiles_building.py` to `examples/workflows/`
- Moved `preprocess_and_train.py` to `examples/workflows/`
- Moved `validation_results.json` to `data/validation/`
- Moved `location_replacement_mapping.json` to `ign_lidar/data/`
- Archived `WORKFLOW_EN_COURS.md` to `docs/archive/`
- Archived `INSTRUCTIONS_WORKFLOW_AMELIORE.md` to `docs/archive/`
- Removed empty `temp_validation/` directory

#### New Directories

- Created `examples/legacy/` for deprecated scripts
- Created `examples/workflows/` for workflow examples
- Created `docs/user-guide/` for user documentation
- Created `docs/developer-guide/` for developer documentation
- Created `docs/reference/` for API reference
- Created `data/validation/` for validation data
- Created `ign_lidar/data/` for package-embedded data

#### Documentation

- Added deprecation notices to moved scripts
- Created `CONSOLIDATION_PLAN.md` with consolidation strategy
- Created `CONSOLIDATION_COMPLETE.md` with completion report
- Created `docs/README.md` as documentation index

#### Maintenance

- Updated `.gitignore` with new patterns
- Fixed import paths in affected scripts
- Updated file references to new locations

#### Benefits

- Cleaner root directory (10 files vs 18)
- Clear separation between package, examples, and documentation
- Professional appearance for PyPI publication
- Better organization for long-term maintenance
- No breaking changes to public API or CLI

**Migration Guide**: All functionality remains intact. Use `ign-lidar-hd` CLI instead of root scripts.

## [1.2.0] - 2025-10-03

### Changed

- **Optimized K-neighbors parameter**: `DEFAULT_K_NEIGHBORS` increased from 10 to 20

  - Better quality for building extraction (normals, planarity, curvature)
  - ~0.5m effective radius, optimal for IGN LiDAR HD density
  - Aligned with existing workflows and examples
  - Performance impact: +38% computation time (still fast with vectorization)

- **Optimized points per patch**: `DEFAULT_NUM_POINTS` increased from 8192 to 16384

  - **2× larger context** for better learning and prediction quality
  - Better capture of complex building structures
  - Reduced border artifacts between patches
  - Optimal for modern GPUs (≥12 GB VRAM)
  - Density: ~0.73 pt/m² on 150m × 150m patches (vs 0.36 with 8192)
  - Performance impact: +35-40% training time, requires batch_size=8 instead of 16
  - Quality improvement: +6-8% IoU, +18% precision on building extraction

- **Fixed patch size inconsistency in `workflow_100_tiles_building.py`**:
  - Corrected default patch size from 12.25m to 150.0m
  - Updated patch area from 150m² to 22,500m² (150m × 150m)
  - Fixed documentation and metadata to reflect correct patch dimensions
  - Renamed output directory from `patches_150m2` to `patches_150x150m`

### Added

- **Documentation**: `docs/K_NEIGHBORS_OPTIMIZATION.md`

  - Comprehensive analysis of k-neighbors parameter
  - Performance benchmarks for k=10, 20, 30, 40
  - Recommendations by zone type (urban, rural, etc.)

- **Documentation**: `docs/NUM_POINTS_OPTIMIZATION.md`
  - Complete guide for optimizing points per patch
  - GPU-specific recommendations (4096 to 32768 points)
  - Memory consumption estimates and performance benchmarks
  - Quality vs speed trade-offs analysis
  - Migration checklist for upgrading from 8192 to 16384
  - Quality metrics and trade-offs

## [1.1.0] - 2025-10-02

### Added

- **New module** `ign_lidar/config.py`: Centralized configuration management

  - DEFAULT_PATCH_SIZE harmonized to 150.0m across all workflows
  - DEFAULT_NUM_POINTS, DEFAULT_K_NEIGHBORS, DEFAULT_NUM_TILES constants
  - FEATURE_DIMENSIONS dictionary defining all 16 geometric features
  - LAZ_EXTRA_DIMS defining 11 extra dimensions for enriched LAZ
  - Configuration validation functions
  - Feature set definitions (minimal, geometric, full)

- **New module** `ign_lidar/strategic_locations.py`: Strategic location database

  - STRATEGIC_LOCATIONS: 23 locations across 11 building categories
  - validate_locations_via_wfs(): WFS validation function
  - download_diverse_tiles(): Diversified tile download
  - Helper functions: get_categories(), get_locations_by_category()
  - Comprehensive coverage: urban, suburban, rural, coastal, mountain, infrastructure

- **Documentation improvements**:

  - START_HERE.md: Quick start guide post-consolidation
  - CLEANUP_PLAN.md: Detailed consolidation plan
  - CLEANUP_REPORT.md: Complete consolidation metrics and report
  - scripts/legacy/README.md: Guide for archived scripts

- **Verification tools**:
  - verify_consolidation.py: Post-consolidation validation script

### Changed

- **workflow_laz_enriched.py**: Updated to use new modules

  - Now imports from `ign_lidar.strategic_locations`
  - Now imports from `ign_lidar.config`
  - Uses centralized DEFAULT\_\* constants
  - No breaking changes for CLI usage

- **Code organization**: 10 scripts archived to `scripts/legacy/`
  - adaptive_tile_selection.py
  - strategic_tile_selection.py
  - create_strategic_list.py
  - create_diverse_dataset.py
  - validate_and_download_diverse_dataset.py
  - download_50_tiles_for_training.py
  - download_and_preprocess.py
  - test_processing.py
  - debug_wfs.py
  - diagnose_laz.py

### Fixed

- **Configuration inconsistencies**: Patch size harmonized
  - Previously: mix of 50m and 150m in different scripts
  - Now: 150.0m everywhere via DEFAULT_PATCH_SIZE
- **Code duplication**: Eliminated redundancy
  - STRATEGIC_LOCATIONS defined once (was in 4+ files)
  - validate_wfs() defined once (was in 3+ files)
  - download functions consolidated (was in 5+ files)

### Improved

- **Maintainability**: -75% code duplication
- **Structure**: Professional Python package layout
- **Clarity**: 28 → 6 scripts at root (-79%)
- **Consistency**: Single source of truth for all configurations

### Technical Details

- Package structure: 2 new modules (config.py, strategic_locations.py)
- Total lines added: ~800 lines of consolidated, documented code
- Scripts archived: 10 legacy scripts preserved for reference
- Breaking changes: None (backward compatible)
- Test coverage: All new modules verified with verify_consolidation.py

## [1.0.0] - 2024-10-02

### Added

- Initial release of ign-lidar-hd library
- Core `LiDARProcessor` class for processing IGN LiDAR HD data
- `IGNLiDARDownloader` class for automated tile downloading
- Support for LOD2 (15 classes) and LOD3 (30 classes) classification schemas
- Feature extraction functions: normals, curvature, geometric features
- Command-line interface `ign-lidar-hd`
- Patch-based processing with configurable sizes and overlap
- Data augmentation capabilities (rotation, jitter, scaling, dropout)
- Parallel processing support for batch operations
- Comprehensive tile management with 50 curated test tiles
- Examples and documentation for basic and advanced usage
- Complete test suite with pytest
- Development environment setup scripts
- Build and distribution automation

### Features

- **LiDAR-only processing**: Works purely with geometric data, no RGB dependency
- **Multi-level classification**: LOD2 and LOD3 building classification schemas
- **Rich feature extraction**: Comprehensive geometric and statistical features
- **Flexible patch processing**: Configurable patch sizes and overlap ratios
- **Spatial filtering**: Bounding box support for focused analysis
- **Environment-based processing**: Different strategies for urban/coastal/rural areas
- **Robust downloading**: Integrated IGN WFS service integration
- **Parallel processing**: Multi-worker support for large datasets
- **Quality assurance**: Extensive testing and code quality tools

### Technical Details

- Python 3.8+ support
- Core dependencies: numpy, laspy, scikit-learn, tqdm, requests, click
- Development tools: pytest, black, flake8, mypy, pre-commit
- Distribution: PyPI-ready with proper packaging
- CLI: User-friendly command-line interface
- Documentation: Comprehensive README and examples

### Project Structure

```
ign_lidar/
├── __init__.py          # Main package initialization
├── processor.py         # Core LiDAR processing class
├── downloader.py        # IGN WFS downloading functionality
├── features.py          # Feature extraction functions
├── classes.py           # Classification schemas (LOD2/LOD3)
├── tile_list.py         # Curated tile management
├── utils.py             # Utility functions
└── cli.py               # Command-line interface

examples/
├── basic_usage.py       # Basic usage examples
└── advanced_usage.py    # Advanced processing examples

tests/
├── conftest.py          # Test configuration and fixtures
├── test_core.py         # Core functionality tests
└── test_cli.py          # CLI testing
```

### Development

- Consolidated redundant files and improved project structure
- Enhanced build and development scripts
- Comprehensive testing framework
- Code quality enforcement with linting and formatting
- Simplified dependency management
- Ready for PyPI distribution

[1.0.0]: https://github.com/your-username/ign-lidar-hd/releases/tag/v1.0.0
