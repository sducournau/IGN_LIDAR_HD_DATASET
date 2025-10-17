# IGN LiDAR HD - Base Configuration Files v4.0

This directory contains base configuration files for the IGN LiDAR HD processing pipeline. These files provide modular, reusable configuration components that can be mixed and matched to create custom processing workflows.

## Overview

The base configuration system uses Hydra's composition mechanism to allow flexible configuration inheritance and overrides. Each base file focuses on a specific aspect of the processing pipeline.

## Available Base Configurations

### Core Processing Components

- **`features.yaml`** - Feature computation and extraction settings
- **`classification.yaml`** - Classification rules and algorithms
- **`preprocessing.yaml`** - Point cloud preprocessing and cleaning
- **`ground_truth.yaml`** - External data source integration
- **`output.yaml`** - Output formats and validation
- **`stitching.yaml`** - Tile stitching and seamless merging

### System Configuration

- **`performance.yaml`** - Hardware optimization and resource management
- **`data_sources.yaml`** - External data source configuration

## Usage

### Basic Usage

Include base configurations in your custom config using the `defaults` list:

```yaml
# my_custom_config.yaml
defaults:
  - base/features
  - base/classification
  - base/output
  - _self_

# Override specific settings
features:
  mode: "asprs_classes"
  k_neighbors: 30

classification:
  use_ground_truth: true
  use_geometric_rules: true
```

### Selective Configuration

Include only the components you need:

```yaml
# minimal_config.yaml
defaults:
  - base/features
  - base/output
  - _self_

# Minimal feature set
features:
  mode: "minimal"
  compute_normals: true
  compute_planarity: true
```

### Full Pipeline Configuration

For complete processing with all components:

```yaml
# full_pipeline_config.yaml
defaults:
  - base/preprocessing
  - base/features
  - base/ground_truth
  - base/classification
  - base/stitching
  - base/output
  - base/performance
  - _self_

# Enable everything
preprocessing:
  enabled: true

ground_truth:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
      water: true
```

## Configuration Hierarchy

The system follows this hierarchy (later entries override earlier ones):

1. Base configurations (in order listed)
2. Main configuration file
3. Command-line overrides

Example:

```bash
ign-lidar-hd process \
  --config-file my_config.yaml \
  features.k_neighbors=50 \
  classification.use_ground_truth=false
```

## Component Details

### Features (`base/features.yaml`)

Controls feature computation for classification:

- **Geometric features**: normals, planarity, curvature, height above ground
- **Spectral features**: RGB, NIR, NDVI
- **Advanced features**: multiscale, architectural, statistical
- **Performance**: GPU acceleration, memory management

Key parameters:

- `mode`: Feature computation mode (minimal, asprs_classes, lod2, lod3, full)
- `k_neighbors`: Number of neighbors for feature computation
- `use_rgb`: Enable RGB spectral features

### Classification (`base/classification.yaml`)

Defines classification rules and algorithms:

- **Methods**: Ground truth, geometric rules, ML, NDVI
- **Thresholds**: Height, planarity, density thresholds per class
- **Post-processing**: Morphological filters, size filtering, smoothing

Key parameters:

- `use_ground_truth`: Use external reference data
- `use_geometric_rules`: Use geometric classification rules
- `height_low_vegetation`: Low vegetation height threshold

### Preprocessing (`base/preprocessing.yaml`)

Point cloud cleaning and preparation:

- **Outlier removal**: Statistical and radius-based
- **Noise filtering**: Bilateral, Gaussian, median filters
- **Data cleaning**: Classification code normalization
- **Validation**: Quality control and statistics

Key parameters:

- `enabled`: Enable preprocessing pipeline
- `statistical_outlier_removal.std_threshold`: Outlier detection sensitivity

### Ground Truth (`base/ground_truth.yaml`)

External data source integration:

- **BD TOPO**: Buildings, roads, water, vegetation features
- **Cadastre**: Building outlines and parcels
- **Other sources**: BD Forêt, RPG, orthophotos
- **Integration**: Priority-based conflict resolution

Key parameters:

- `bd_topo.enabled`: Enable BD TOPO integration
- `bd_topo.features.*`: Enable specific feature types

### Output (`base/output.yaml`)

Output format and validation settings:

- **Formats**: LAZ, LAS, HDF5, PLY, XYZ
- **Content**: Enriched clouds, training patches, metadata
- **Validation**: Format validation, quality control
- **Organization**: File naming and directory structure

Key parameters:

- `format`: Primary output format
- `save_enriched`: Save enriched point clouds
- `validation.enabled`: Enable output validation

### Stitching (`base/stitching.yaml`)

Tile stitching and seamless merging:

- **Buffer management**: Overlap handling and edge processing
- **Harmonization**: Classification and feature harmonization
- **Quality control**: Density and consistency checks
- **Performance**: GPU acceleration and parallel processing

Key parameters:

- `enabled`: Enable tile stitching
- `buffer_size`: Buffer size around tile edges
- `auto_detect_neighbors`: Automatic neighbor detection

### Performance (`base/performance.yaml`)

Hardware optimization and resource management:

- **GPU settings**: Memory management, batch sizes, CUDA streams
- **CPU configuration**: Worker processes and threading
- **Memory management**: Chunking and garbage collection
- **Monitoring**: Performance metrics and profiling

Key parameters:

- `gpu.batch_size`: GPU batch size for processing
- `memory.chunk_size`: Chunk size for large datasets
- `auto_tuning.enabled`: Enable automatic performance tuning

### Data Sources (`base/data_sources.yaml`)

External data source configuration:

- **BD TOPO features**: Individual feature toggles (flat structure)
- **Orthophotos**: RGB and NIR imagery
- **Other sources**: Cadastre, BD Forêt, RPG
- **Fetching**: Connection settings and caching

Key parameters:

- `bd_topo_buildings`: Enable BD TOPO building data
- `orthophoto_rgb`: Enable RGB orthophoto fetching
- `fetching.parallel_requests`: Number of parallel data requests

## Best Practices

### 1. Start with Minimal Configuration

Begin with minimal base configurations and add components as needed:

```yaml
defaults:
  - base/features
  - base/output
```

### 2. Use Appropriate Feature Modes

Choose the right feature mode for your use case:

- `minimal`: Basic processing, fastest
- `asprs_classes`: ASPRS classification optimized
- `lod2`: Building modeling optimized
- `lod3`: Detailed architectural modeling
- `full`: All available features

### 3. Enable Components Gradually

Enable complex components only when needed:

- Start with geometric classification
- Add ground truth data when available
- Enable preprocessing for noisy data
- Use stitching for large continuous areas

### 4. Monitor Performance

Use performance monitoring to optimize settings:

```yaml
defaults:
  - base/performance

performance:
  monitoring:
    enabled: true
    save_performance_metrics: true
```

### 5. Validate Outputs

Always enable output validation for production:

```yaml
defaults:
  - base/output

output:
  validation:
    enabled: true
    validate_classifications: true
```

## Migration from v3.x

The v4.0 base configurations use a flatter structure for better usability:

**v3.x (nested)**:

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
```

**v4.0 (flat)**:

```yaml
data_sources:
  bd_topo_buildings: true
  bd_topo_roads: true
```

The system maintains backward compatibility with v3.x configurations.

## Examples

See the `presets/` directory for complete examples using these base configurations:

- `asprs_classification.yaml` - ASPRS standard classification
- `gpu_optimized.yaml` - GPU-optimized processing
- `minimal.yaml` - Minimal processing for testing
- `ground_truth_training.yaml` - ML training data generation

## Support

For issues or questions about base configurations:

1. Check the main configuration documentation
2. Review preset examples
3. Enable debug logging: `monitoring.log_level: DEBUG`
4. Use configuration validation: `validation.strict_validation: true`
