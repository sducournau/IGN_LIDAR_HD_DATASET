---
sidebar_position: 2
title: CLI Patch Command
description: Generate training patches from processed LiDAR data
keywords: [cli, patches, machine-learning, training, dataset]
---

# CLI Patch Command Reference

The `ign-lidar patch` command generates machine learning training patches from processed LiDAR point clouds.

## Syntax

```bash
ign-lidar patch [OPTIONS] INPUT_PATH OUTPUT_PATH
```

## Basic Usage

### Generate Standard Patches

```bash
ign-lidar patch enriched_data.las training_patches.h5
```

### Custom Patch Size

```bash
ign-lidar patch --patch-size 64 --overlap 0.3 input.las patches.h5
```

## Command Options

### Patch Generation

#### `--patch-size`

Size of square patches in points.
**Default:** `32`

#### `--overlap`

Overlap between adjacent patches (0.0-1.0).
**Default:** `0.5`

#### `--min-points`

Minimum points required per patch.
**Default:** `100`

### Output Options

#### `--output-format`

Output format for patches.
**Options:** `h5`, `npz`, `pkl`
**Default:** `h5`

#### `--augmentation`

Enable data augmentation.

### Processing Options

#### `--batch-size`

Processing batch size.
**Default:** `10000`

#### `--num-workers`

Number of parallel workers.

## Examples

### Basic Patch Generation

```bash
ign-lidar patch --patch-size 48 --overlap 0.4 input.las patches.h5
```

### High-Quality Training Data

```bash
ign-lidar patch \
  --patch-size 64 \
  --overlap 0.3 \
  --min-points 200 \
  --augmentation \
  input.las training_data.h5
```

## Related Commands

- [`ign-lidar enrich`](./cli-enrich.md) - Enrich point clouds
- [`ign-lidar download`](./cli-download.md) - Download tiles
