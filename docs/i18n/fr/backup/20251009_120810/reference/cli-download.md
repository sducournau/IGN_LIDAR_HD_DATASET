---
# 🇫🇷 Traduisez les champs title et description ci-dessous
sidebar_position: 3
title: CLI Download Command
description: Download IGN LiDAR HD tiles from official servers
keywords: [cli, download, tiles, ign, lidar]
---

# CLI Téléchargement Command Reference

The `ign-lidar download` command retrieves LiDAR HD tiles from IGN's official servers.

## Syntax

```bash
ign-lidar download [OPTIONS] TILE_IDS OUTPUT_DIR
```

## Utilisation de base

### Téléchargement Single Tile

```bash
ign-lidar download C_3945-6730_2022 ./tiles/
```

### Téléchargement Multiple Tiles

```bash
ign-lidar download C_3945-6730_2022 C_3945-6735_2022 ./tiles/
```

### Téléchargement from List

```bash
ign-lidar download --from-file tile_list.txt ./tiles/
```

## Command Options

### Entrée Options

#### `TILE_IDS` (required)

One or more tile identifiers to download.

#### `--from-file, -f`

Read tile IDs from text file (one per line).

#### `--bbox`

Téléchargement all tiles within bounding box.
Format: `xmin,ymin,xmax,ymax`

### Sortie Options

#### `OUTPUT_DIR` (required)

Répertoire pour sauvegarder downloaded tiles.

#### `--format`

Téléchargement format.
**Options:** `laz`, `las`
**Default:** `laz`

### Téléchargement Options

#### `--overwrite`

Overwrite existing files.

#### `--verify`

Verify downloaded files.

#### `--parallel, -p`

Number of parallel downloads.
**Default:** `4`

## Exemples

### Téléchargement by Bounding Box

```bash
ign-lidar download --bbox 3945000,6730000,3950000,6735000 ./tiles/
```

### Parallel Téléchargements with Verification

```bash
ign-lidar download --parallel 8 --verify --from-file tiles.txt ./data/
```

## Related Commands

- [`ign-lidar enrich`](./cli-enrich) - Enrichissement downloaded tiles
- [`ign-lidar patch`](./cli-patch) - Generate training patches
