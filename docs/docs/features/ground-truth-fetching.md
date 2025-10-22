# Ground Truth Fetching from IGN BD TOPO®

This guide explains how to use the WFS ground truth fetching functionality to generate labeled training patches from IGN's BD TOPO® vector data.

## Overview

The ground truth fetching system retrieves vector data from IGN's WFS (Web Feature Service) for:

- **Building footprints** (emprise de bâtiment)
- **Road polygons** generated from centerlines + width attributes (largeur)
- **Water surfaces** (surface hydrographique)
- **Vegetation zones** (zone de végétation)
- **Other topographic features**

This data is used to label LiDAR point clouds for supervised machine learning training.

## Installation

Additional dependencies required:

```bash
pip install shapely geopandas
```

## Quick Start

### Command Line Interface

Generate patches with ground truth labels:

```bash
# Basic usage
ign-lidar-hd ground-truth data/tile_0650_6860.laz data/patches_gt

# Customize features
ign-lidar-hd ground-truth data/tile.laz data/patches \
    --patch-size 100 \
    --no-water \
    --no-vegetation \
    --cache-dir cache/gt

# Save ground truth vectors
ign-lidar-hd ground-truth data/tile.laz data/patches \
    --save-ground-truth \
    --include-roads \
    --include-buildings
```

### Python API

```python
from ign_lidar.io.wfs_ground_truth import (
    IGNGroundTruthFetcher,
    fetch_ground_truth_for_tile,
    generate_patches_with_ground_truth
)
from ign_lidar.core.modules.loader import load_laz_file

# Load LiDAR tile
lidar_data = load_laz_file("data/tile_0650_6860.laz")
points = lidar_data.points

# Compute bounding box
tile_bbox = (
    points[:, 0].min(),
    points[:, 1].min(),
    points[:, 0].max(),
    points[:, 1].max()
)

# Fetch ground truth
fetcher = IGNGroundTruthFetcher(cache_dir="cache/ground_truth")
ground_truth = fetcher.fetch_all_features(
    bbox=tile_bbox,
    include_roads=True,
    include_buildings=True,
    include_water=True,
    include_vegetation=True
)

# Label points
labels = fetcher.label_points_with_ground_truth(
    points=points,
    ground_truth_features=ground_truth
)
```

## Features

### 1. Building Footprints

Fetches building polygons from BD TOPO®:

```python
buildings = fetcher.fetch_buildings(bbox)
# Returns GeoDataFrame with:
# - geometry: Polygon
# - hauteur: Building height (if available)
# - nature: Building type
```

### 2. Road Polygons from Width

Generates road polygons from centerlines using width attributes:

```python
roads = fetcher.fetch_roads_with_polygons(
    bbox,
    default_width=4.0  # Default width if attribute missing
)
# Returns GeoDataFrame with:
# - geometry: Road polygon (buffered from centerline)
# - width_m: Road width in meters
# - nature: Road type (route, autoroute, etc.)
# - importance: Road importance
# - original_geometry: Original centerline
```

**How it works:**

1. Fetches road centerlines from `BDTOPO_V3:troncon_de_route`
2. Reads `largeur` or `largeur_de_chaussee` attribute
3. Buffers centerline by `width / 2` on each side
4. Returns polygons representing actual road surface

### 3. Water Surfaces

```python
water = fetcher.fetch_water_surfaces(bbox)
# Rivers, lakes, ponds, etc.
```

### 4. Vegetation Zones

```python
vegetation = fetcher.fetch_vegetation_zones(bbox)
# Forest, parks, green spaces
```

### 5. Combined Fetching

```python
all_features = fetcher.fetch_all_features(
    bbox=tile_bbox,
    include_roads=True,
    include_buildings=True,
    include_water=True,
    include_vegetation=True
)
# Returns dict: {'buildings': GeoDataFrame, 'roads': GeoDataFrame, ...}
```

## Point Cloud Labeling

### Label Assignment

Points are labeled based on spatial intersection with ground truth polygons:

```python
labels = fetcher.label_points_with_ground_truth(
    points=points,  # [N, 3] XYZ coordinates
    ground_truth_features=ground_truth,
    label_priority=['buildings', 'roads', 'water', 'vegetation']
)
# Returns [N] array with labels:
# 0: unlabeled/ground
# 1: building
# 2: road
# 3: water
# 4: vegetation
```

### Priority Order

When multiple features overlap (e.g., building on road), the `label_priority` list determines which label is assigned. Higher priority = later in list.

Example: `['roads', 'buildings']` means buildings override roads.

## Complete Workflow

### From Tile to Labeled Patches

```python
from pathlib import Path
from ign_lidar.io.wfs_ground_truth import generate_patches_with_ground_truth
from ign_lidar.core.modules.loader import load_laz_file
from ign_lidar.core.modules.saver import save_patch_npz

# Load tile
tile_file = Path("data/raw/tile_0650_6860.laz")
lidar_data = load_laz_file(tile_file)
points = lidar_data.points

# Compute features
features = {
    'classification': lidar_data.classification,
    'intensity': lidar_data.intensity
}

# Compute bounding box
tile_bbox = (
    points[:, 0].min(),
    points[:, 1].min(),
    points[:, 0].max(),
    points[:, 1].max()
)

# Generate patches with ground truth
patches = generate_patches_with_ground_truth(
    points=points,
    features=features,
    tile_bbox=tile_bbox,
    patch_size=150.0,
    cache_dir=Path("cache/ground_truth")
)

# Save patches
output_dir = Path("data/patches_with_gt")
output_dir.mkdir(parents=True, exist_ok=True)

for i, patch in enumerate(patches):
    patch_file = output_dir / f"patch_{i:04d}.npz"
    save_patch_npz(patch, patch_file)

print(f"Generated {len(patches)} patches with ground truth labels")
```

## Configuration

### WFS Service Configuration

```python
from ign_lidar.io.wfs_ground_truth import IGNWFSConfig

config = IGNWFSConfig()
config.WFS_URL = "https://data.geopf.fr/wfs"
config.BUILDINGS_LAYER = "BDTOPO_V3:batiment"
config.ROADS_LAYER = "BDTOPO_V3:troncon_de_route"
config.MAX_FEATURES = 10000

fetcher = IGNGroundTruthFetcher(config=config)
```

### Caching

Ground truth data is cached to avoid redundant WFS requests:

```python
fetcher = IGNGroundTruthFetcher(cache_dir=Path("cache/ground_truth"))

# First call: fetches from WFS
buildings = fetcher.fetch_buildings(bbox, use_cache=True)

# Second call: loads from cache (fast)
buildings = fetcher.fetch_buildings(bbox, use_cache=True)

# Force refresh
buildings = fetcher.fetch_buildings(bbox, use_cache=False)
```

## Examples

### Example 1: Road Width Analysis

```python
# Fetch roads with polygons
roads = fetcher.fetch_roads_with_polygons(bbox)

# Analyze road widths
print(f"Average width: {roads['width_m'].mean():.2f}m")
print(f"Total road area: {roads['geometry'].area.sum():.2f} m²")

# Filter by width
wide_roads = roads[roads['width_m'] > 10.0]
print(f"Roads wider than 10m: {len(wide_roads)}")

# Save for visualization
roads.to_file("roads_with_width.geojson", driver='GeoJSON')
```

### Example 2: Building Height Statistics

```python
# Fetch buildings
buildings = fetcher.fetch_buildings(bbox)

# Check for height attribute
if 'hauteur' in buildings.columns:
    heights = buildings['hauteur'].dropna()
    print(f"Buildings with height data: {len(heights)}")
    print(f"Average height: {heights.mean():.1f}m")
    print(f"Max height: {heights.max():.1f}m")

    # Filter tall buildings
    tall_buildings = buildings[buildings['hauteur'] > 20.0]
    print(f"Buildings > 20m: {len(tall_buildings)}")
```

### Example 3: Multi-Tile Batch Processing

```python
from pathlib import Path
import multiprocessing as mp

def process_tile(tile_file):
    """Process single tile with ground truth."""
    try:
        # Load tile
        lidar_data = load_laz_file(tile_file)
        points = lidar_data.points

        # Compute bbox
        tile_bbox = (
            points[:, 0].min(),
            points[:, 1].min(),
            points[:, 0].max(),
            points[:, 1].max()
        )

        # Generate patches
        patches = generate_patches_with_ground_truth(
            points=points,
            features={'classification': lidar_data.classification},
            tile_bbox=tile_bbox,
            cache_dir=Path("cache/ground_truth")
        )

        # Save patches
        output_dir = Path("data/patches_gt") / tile_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, patch in enumerate(patches):
            save_patch_npz(patch, output_dir / f"patch_{i:04d}.npz")

        return len(patches)

    except Exception as e:
        print(f"Error processing {tile_file}: {e}")
        return 0

# Process multiple tiles in parallel
tile_files = list(Path("data/raw").glob("*.laz"))
with mp.Pool(processes=4) as pool:
    results = pool.map(process_tile, tile_files)

print(f"Total patches generated: {sum(results)}")
```

## Troubleshooting

### No Features Found

If no ground truth features are returned:

1. Check bounding box coordinates are in Lambert 93 (EPSG:2154)
2. Verify bbox covers valid French territory
3. Check WFS service is accessible: https://data.geopf.fr/wfs
4. Try a smaller area (WFS has feature limits)

### Missing Width Attribute

If roads don't have width attributes:

```python
roads = fetcher.fetch_roads_with_polygons(
    bbox,
    default_width=4.0  # Used when largeur is missing
)
```

### Memory Issues

For large areas:

1. Process in smaller tiles
2. Reduce `MAX_FEATURES` limit
3. Disable caching: `use_cache=False`
4. Process tiles in batches

## API Reference

### IGNGroundTruthFetcher

Main class for fetching ground truth data.

```python
fetcher = IGNGroundTruthFetcher(
    cache_dir=None,  # Optional cache directory
    config=None      # Optional WFS configuration
)
```

**Methods:**

- `fetch_buildings(bbox, use_cache=True)` - Fetch building footprints
- `fetch_roads_with_polygons(bbox, use_cache=True, default_width=4.0)` - Fetch road polygons
- `fetch_water_surfaces(bbox, use_cache=True)` - Fetch water bodies
- `fetch_vegetation_zones(bbox, use_cache=True)` - Fetch vegetation areas
- `fetch_all_features(bbox, ...)` - Fetch all available features
- `label_points_with_ground_truth(points, ground_truth_features, label_priority)` - Label points
- `save_ground_truth(features, output_dir, bbox)` - Save to disk

### Helper Functions

```python
# Fetch ground truth for a tile
features = fetch_ground_truth_for_tile(
    tile_bbox,
    cache_dir=None,
    include_roads=True,
    include_buildings=True,
    include_water=True,
    include_vegetation=True
)

# Complete workflow
patches = generate_patches_with_ground_truth(
    points,
    features,
    tile_bbox,
    patch_size=150.0,
    cache_dir=None
)
```

## Data Sources

All data comes from **IGN BD TOPO®** via the Géoplateforme WFS service:

- **Service**: https://data.geopf.fr/wfs
- **Documentation**: https://geoservices.ign.fr/documentation/services/services-geoplateforme
- **License**: Licence Ouverte / Open License
- **CRS**: EPSG:2154 (Lambert 93)
- **Coverage**: Metropolitan France

## Citation

If you use this functionality in research, please cite:

```
IGN LiDAR HD Dataset
Ground Truth Fetching from BD TOPO®
https://github.com/sducournau/IGN_LIDAR_HD_DATASET
```

## See Also

- [Complete Workflow Guide](../guides/complete-workflow.md)
- [Ground Truth Classification](../features/ground-truth-classification.md)
- [RGB Augmentation](../features/rgb-augmentation.md)
- [IGN Géoplateforme Documentation](https://geoservices.ign.fr/)
