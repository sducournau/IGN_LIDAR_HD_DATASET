---
sidebar_position: 1
title: CLI Enrich Command
description: Command-line interface for enriching LiDAR data with building features
keywords: [cli, command-line, enrich, features, processing]
---

# CLI Enrich Command Reference

The `ign-lidar enrich` command adds building component features to LiDAR point clouds, enhancing them with geometric and architectural information.

## Syntax

```bash
ign-lidar enrich [OPTIONS] INPUT_PATH OUTPUT_PATH
```

## Basic Usage

### Enrich a Single File

```bash
ign-lidar enrich input.las enriched_output.las
```

### Enrich with RGB Data

```bash
ign-lidar enrich --rgb-path orthophoto.tif input.las output.las
```

### Batch Processing

```bash
ign-lidar enrich --batch tiles/*.las --output-dir enriched/
```

## Command Options

### Input/Output Options

#### `INPUT_PATH` (required)

Path to input LAS/LAZ file or directory for batch processing.

```bash
# Single file
ign-lidar enrich tile_001.las output.las

# Directory (batch mode)
ign-lidar enrich tiles/ --output-dir enriched/
```

#### `OUTPUT_PATH` (required for single files)

Output path for enriched point cloud.

#### `--output-dir, -o` (batch mode)

Output directory for batch processing.

```bash
ign-lidar enrich tiles/ --output-dir /path/to/enriched/
```

#### `--output-format`

Output file format.

**Options:** `las`, `laz`, `ply`, `h5`
**Default:** `las`

```bash
ign-lidar enrich input.las output.h5 --output-format h5
```

### Processing Options

#### `--features, -f`

Specify which features to extract.

**Options:**

- `geometric`: Basic geometric features (planarity, linearity, etc.)
- `architectural`: Building-specific features (walls, roofs, etc.)
- `all`: All available features
- `custom`: User-defined feature set

```bash
# Extract only geometric features
ign-lidar enrich --features geometric input.las output.las

# Extract all features
ign-lidar enrich --features all input.las output.las

# Custom feature set
ign-lidar enrich --features planarity,height,normal_z input.las output.las
```

#### `--neighborhood-size, -n`

Neighborhood radius for feature computation (meters).

**Default:** `1.0`

```bash
ign-lidar enrich --neighborhood-size 2.0 input.las output.las
```

#### `--min-building-points`

Minimum points required to classify as building.

**Default:** `50`

```bash
ign-lidar enrich --min-building-points 100 input.las output.las
```

### RGB Integration Options

#### `--rgb-path, -r`

Path to orthophoto for color augmentation.

```bash
ign-lidar enrich --rgb-path orthophoto.tif input.las output.las
```

#### `--rgb-interpolation`

Interpolation method for RGB assignment.

**Options:** `nearest`, `bilinear`, `bicubic`
**Default:** `bilinear`

```bash
ign-lidar enrich --rgb-path ortho.tif --rgb-interpolation bicubic input.las output.las
```

#### `--rgb-bands`

Specify which bands to extract from orthophoto.

**Default:** `1,2,3` (RGB)

```bash
# Include infrared band
ign-lidar enrich --rgb-path ortho.tif --rgb-bands 1,2,3,4 input.las output.las
```

### Performance Options

#### `--gpu, -g`

Enable GPU acceleration.

```bash
ign-lidar enrich --gpu input.las output.las
```

#### `--gpu-memory-fraction`

Fraction of GPU memory to use.

**Default:** `0.7`

```bash
ign-lidar enrich --gpu --gpu-memory-fraction 0.9 input.las output.las
```

#### `--batch-size, -b`

Processing batch size for memory management.

**Default:** `100000`

```bash
ign-lidar enrich --batch-size 50000 input.las output.las
```

#### `--num-workers, -w`

Number of parallel workers for processing.

**Default:** Number of CPU cores

```bash
ign-lidar enrich --num-workers 8 input.las output.las
```

### Quality Control Options

#### `--validate`

Perform validation checks on output.

```bash
ign-lidar enrich --validate input.las output.las
```

#### `--quality-report`

Generate quality assessment report.

```bash
ign-lidar enrich --quality-report report.json input.las output.las
```

#### `--preserve-classification`

Keep original point classifications.

```bash
ign-lidar enrich --preserve-classification input.las output.las
```

### Architectural Analysis Options

#### `--architectural-style`

Specify architectural style for enhanced analysis.

**Options:** `haussmanian`, `traditional`, `contemporary`, `industrial`

```bash
ign-lidar enrich --architectural-style haussmanian input.las output.las
```

#### `--region`

Geographic region for style adaptation.

**Options:** `ile_de_france`, `provence`, `brittany`, `alsace`

```bash
ign-lidar enrich --region ile_de_france input.las output.las
```

#### `--building-type`

Expected building types in dataset.

**Options:** `residential`, `commercial`, `industrial`, `mixed`

```bash
ign-lidar enrich --building-type residential input.las output.las
```

## Configuration File

Use a YAML configuration file for complex processing scenarios.

### Configuration Example

```yaml
# enrich_config.yaml
processing:
  features: ["geometric", "architectural"]
  neighborhood_size: 1.5
  min_building_points: 75

rgb:
  enabled: true
  interpolation: "bilinear"
  bands: [1, 2, 3]

performance:
  use_gpu: true
  gpu_memory_fraction: 0.8
  batch_size: 75000
  num_workers: 6

architectural:
  style: "haussmanian"
  region: "ile_de_france"
  building_type: "residential"

quality:
  validate: true
  generate_report: true
```

### Using Configuration File

```bash
ign-lidar enrich --config enrich_config.yaml input.las output.las
```

## Advanced Examples

### High-Quality Building Analysis

```bash
ign-lidar enrich \
  --features all \
  --rgb-path orthophoto.tif \
  --architectural-style haussmanian \
  --region ile_de_france \
  --neighborhood-size 1.5 \
  --gpu \
  --validate \
  --quality-report quality.json \
  input.las output.las
```

### Batch Processing with GPU

```bash
ign-lidar enrich \
  --batch tiles/*.las \
  --output-dir enriched/ \
  --features geometric,architectural \
  --gpu \
  --batch-size 200000 \
  --num-workers 4
```

### Memory-Optimized Processing

```bash
ign-lidar enrich \
  --batch-size 25000 \
  --gpu-memory-fraction 0.5 \
  --num-workers 2 \
  input_large.las output.las
```

## Output Information

### Added Point Fields

The enrich command adds the following fields to point clouds:

| Field Name                         | Type  | Description                                |
| ---------------------------------- | ----- | ------------------------------------------ |
| `planarity`                        | float | Surface planarity (0-1)                    |
| `linearity`                        | float | Linear structure strength (0-1)            |
| `sphericity`                       | float | 3D compactness (0-1)                       |
| `height_above_ground`              | float | Normalized height (meters)                 |
| `building_component`               | uint8 | Component class (0=ground, 1=wall, 2=roof) |
| `architectural_style`              | uint8 | Detected architectural style               |
| `normal_x`, `normal_y`, `normal_z` | float | Surface normal vectors                     |

### RGB Fields (when enabled)

| Field Name             | Type   | Description                  |
| ---------------------- | ------ | ---------------------------- |
| `red`, `green`, `blue` | uint16 | RGB color values             |
| `infrared`             | uint16 | Near-infrared (if available) |
| `material_class`       | uint8  | Material classification      |

## Error Handling

### Common Error Messages

#### "GPU memory insufficient"

**Solution:** Reduce batch size or GPU memory fraction

```bash
ign-lidar enrich --gpu-memory-fraction 0.5 --batch-size 50000 input.las output.las
```

#### "RGB file not found"

**Solution:** Check orthophoto path and file permissions

```bash
ls -la /path/to/orthophoto.tif
```

#### "Insufficient points for feature extraction"

**Solution:** Lower minimum building points threshold

```bash
ign-lidar enrich --min-building-points 25 input.las output.las
```

### Debugging Options

#### `--verbose, -v`

Enable detailed logging output.

```bash
ign-lidar enrich --verbose input.las output.las
```

#### `--debug`

Enable debug mode with extensive logging.

```bash
ign-lidar enrich --debug input.las output.las
```

#### `--log-file`

Save logs to file.

```bash
ign-lidar enrich --log-file enrich.log input.las output.las
```

## Performance Benchmarks

### Processing Times (approximate)

| Points | Features  | GPU  | CPU   | Speedup |
| ------ | --------- | ---- | ----- | ------- |
| 1M     | Geometric | 30s  | 4min  | 8x      |
| 1M     | All + RGB | 45s  | 8min  | 11x     |
| 10M    | Geometric | 3min | 35min | 12x     |
| 10M    | All + RGB | 5min | 75min | 15x     |

### Memory Usage

- **CPU Mode**: ~6GB RAM per 10M points
- **GPU Mode**: ~3GB GPU + 2GB RAM per 10M points
- **Batch Processing**: Configurable memory footprint

## Related Commands

- [`ign-lidar download`](./cli-download) - Download IGN LiDAR tiles
- [`ign-lidar patch`](./cli-patch) - Extract ML training patches
- [`ign-lidar qgis`](./cli-qgis) - QGIS integration tools

## See Also

- [Processing Guide](../guides/preprocessing)
- [Features API](../api/features)
- [Performance Optimization](../guides/performance)
- [Troubleshooting](../guides/troubleshooting)
