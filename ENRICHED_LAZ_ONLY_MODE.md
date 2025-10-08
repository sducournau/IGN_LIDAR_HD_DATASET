# Enriched LAZ Only Mode

## Overview

The **Enriched LAZ Only** mode allows you to compute and save enriched LAZ files with advanced features (normals, curvature, RGB, NDVI, etc.) **without creating patches**. This is useful for:

- **Feature enrichment**: Add computed geometric and radiometric features to raw LAZ files
- **Preprocessing**: Prepare enriched point clouds for other workflows
- **Data exploration**: Analyze features without patch-based processing
- **Storage efficiency**: Skip intermediate patch creation when only enriched LAZ is needed

## Configuration

### Quick Start

Use the command line with `output.only_enriched_laz=true`:

```bash
ign-lidar-hd process \
  input_dir="/path/to/raw_tiles" \
  output_dir="/path/to/enriched" \
  output=enriched_only \
  processor=gpu \
  features=full \
  preprocess=aggressive \
  stitching=auto_download \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  num_workers=8 \
  verbose=true
```

### Using Configuration Presets

The system includes an `enriched_only` output preset:

```bash
ign-lidar-hd process \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output" \
  output=enriched_only \
  features=full
```

### Manual Configuration

Set these parameters:

```bash
output.save_enriched_laz=true \
output.only_enriched_laz=true
```

## Features in Enriched LAZ Files

Enriched LAZ files include all original point attributes plus:

### Geometric Features

- **Normals** (`normal_x`, `normal_y`, `normal_z`): Surface normal vectors
- **Curvature** (`curvature`): Local surface curvature
- **Height** (relative to ground): Computed from DEM

### Radiometric Features (optional)

- **RGB** (`red`, `green`, `blue`): From IGN orthophotos (if `features.use_rgb=true`)
- **NIR** (`near_infrared`): From IRC imagery (if `features.use_infrared=true`)
- **NDVI**: Vegetation index (if `features.compute_ndvi=true` and NIR available)

### Original Attributes (preserved)

- XYZ coordinates
- Intensity
- Return number
- Classification
- All other LAS attributes

## Auto-Download Missing Neighbors

Combine enriched LAZ mode with auto-download for seamless boundary processing:

```bash
ign-lidar-hd process \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/enriched" \
  output.only_enriched_laz=true \
  output.save_enriched_laz=true \
  stitching=auto_download \
  features=full
```

This configuration will:

1. ✅ Auto-detect missing neighbor tiles
2. ✅ Download them from IGN WFS if needed
3. ✅ Use neighbors for accurate boundary features
4. ✅ Save enriched LAZ with high-quality features
5. ❌ Skip patch creation

## Output Structure

```
output_dir/
├── enriched/
│   ├── 0891_6248_enriched.laz  # Enriched LAZ with features
│   ├── 0891_6249_enriched.laz
│   └── ...
├── config.yaml  # Processing configuration
└── stats.json   # Processing statistics (optional)
```

## Performance

**Enriched LAZ Only mode is ~3-5x faster** than full patch processing because:

- ✅ No patch extraction overhead
- ✅ No patch-level feature normalization
- ✅ No multiple output format conversions
- ✅ Single-pass processing

Example processing time for 1 km² tile:

- **Full patches mode**: ~120 seconds
- **Enriched LAZ only**: ~25 seconds

## Use Cases

### 1. Feature-Enriched Archive

Create a library of feature-rich LAZ files for future use:

```bash
ign-lidar-hd process \
  input_dir="raw_archive/" \
  output_dir="enriched_archive/" \
  output.only_enriched_laz=true \
  features=full \
  features.use_rgb=true \
  stitching.auto_download_neighbors=true
```

### 2. Preprocessing for Custom Workflows

Enrich LAZ files before using them in custom pipelines:

```bash
ign-lidar-hd process \
  input_dir="project_tiles/" \
  output_dir="project_enriched/" \
  output.only_enriched_laz=true \
  preprocess=aggressive \
  features.k_neighbors=30
```

### 3. Boundary-Aware Feature Computation

Ensure high-quality features at tile boundaries:

```bash
ign-lidar-hd process \
  input_dir="edge_tiles/" \
  output_dir="edge_enriched/" \
  output.only_enriched_laz=true \
  stitching.enabled=true \
  stitching.buffer_size=20.0 \
  stitching.auto_download_neighbors=true
```

## Configuration Reference

### Output Parameters

| Parameter                  | Type | Default | Description                             |
| -------------------------- | ---- | ------- | --------------------------------------- |
| `output.save_enriched_laz` | bool | false   | Save enriched LAZ files                 |
| `output.only_enriched_laz` | bool | false   | Skip patch creation (only enriched LAZ) |
| `output.save_stats`        | bool | true    | Save processing statistics              |
| `output.save_metadata`     | bool | true    | Save metadata (only for patches)        |

### Stitching Parameters (for boundary quality)

| Parameter                           | Type  | Default | Description                |
| ----------------------------------- | ----- | ------- | -------------------------- |
| `stitching.enabled`                 | bool  | false   | Enable tile stitching      |
| `stitching.buffer_size`             | float | 10.0    | Buffer zone in meters      |
| `stitching.auto_detect_neighbors`   | bool  | true    | Auto-detect neighbor tiles |
| `stitching.auto_download_neighbors` | bool  | false   | Download missing neighbors |

### Feature Parameters

| Parameter               | Type | Default | Description                      |
| ----------------------- | ---- | ------- | -------------------------------- |
| `features.mode`         | str  | "full"  | Feature computation mode         |
| `features.k_neighbors`  | int  | 20      | Number of neighbors for features |
| `features.use_rgb`      | bool | false   | Include RGB from orthophotos     |
| `features.use_infrared` | bool | false   | Include NIR from IRC             |
| `features.compute_ndvi` | bool | false   | Compute NDVI vegetation index    |

## Notes

- **Automatic Enablement**: Setting `only_enriched_laz=true` automatically enables `save_enriched_laz=true`
- **Performance**: Use GPU acceleration (`processor=gpu`) for faster feature computation
- **Storage**: Enriched LAZ files are ~1.5-2x larger than raw LAZ due to extra dimensions
- **Compatibility**: Enriched LAZ files can be opened in CloudCompare, QGIS, and other LAZ tools

## Examples

### Minimal (CPU, basic features)

```bash
ign-lidar-hd process \
  input_dir="tiles/" \
  output_dir="enriched/" \
  output.only_enriched_laz=true
```

### Full (GPU, all features, auto-download)

```bash
ign-lidar-hd process \
  input_dir="tiles/" \
  output_dir="enriched/" \
  output=enriched_only \
  processor=gpu \
  features=full \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  stitching=auto_download \
  preprocess=aggressive \
  num_workers=8
```

### Custom (specific features, no preprocessing)

```bash
ign-lidar-hd process \
  input_dir="tiles/" \
  output_dir="enriched/" \
  output.only_enriched_laz=true \
  features.k_neighbors=30 \
  features.use_rgb=true \
  preprocess.enabled=false \
  stitching.enabled=true \
  stitching.buffer_size=15.0
```
