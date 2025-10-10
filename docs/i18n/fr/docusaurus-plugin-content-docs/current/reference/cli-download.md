---
sidebar_position: 3
title: CLI Download Command
description: Download IGN LiDAR HD tiles from official servers
keywords: [cli, download, tiles, ign, lidar]
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->



# CLI Download Command Reference

The `ign-lidar download` command retrieves LiDAR HD tiles from IGN's official servers.

## Syntax

```bash
ign-lidar download [OPTIONS] TILE_IDS OUTPUT_DIR
```

## Basic Usage

### Download Single Tile

```bash
ign-lidar download C_3945-6730_2022 ./tiles/
```

### Download Multiple Tiles

```bash
ign-lidar download C_3945-6730_2022 C_3945-6735_2022 ./tiles/
```

### Download from List

```bash
ign-lidar download --from-file tile_list.txt ./tiles/
```

## Command Options

### Input Options

#### `TILE_IDS` (required)

One or more tile identifiers to download.

#### `--from-file, -f`

Read tile IDs from text file (one per line).

#### `--bbox`

Download all tiles within bounding box.
Format: `xmin,ymin,xmax,ymax`

### Output Options

#### `OUTPUT_DIR` (required)

Directory to save downloaded tiles.

#### `--format`

Download format.
**Options:** `laz`, `las`
**Default:** `laz`

### Download Options

#### `--overwrite`

Overwrite existing files.

#### `--verify`

Verify downloaded files.

#### `--parallel, -p`

Number of parallel downloads.
**Default:** `4`

## Examples

### Download by Bounding Box

```bash
ign-lidar download --bbox 3945000,6730000,3950000,6735000 ./tiles/
```

### Parallel Downloads with Verification

```bash
ign-lidar download --parallel 8 --verify --from-file tiles.txt ./data/
```

## Related Commands

- [`ign-lidar enrich`](./cli-enrich) - Enrich downloaded tiles
- [`ign-lidar patch`](./cli-patch) - Generate training patches
