# WFS Ground Truth Fetching - Feature Summary

## 🎯 Overview

New functionality added to fetch ground truth vector data from IGN's BD TOPO® WFS service for generating labeled training patches. This enables supervised learning by automatically labeling LiDAR point clouds using official topographic data.

## ✨ Key Features

### 1. **Building Footprints**

- Fetch building polygons (emprise de bâtiment)
- Includes height attributes when available
- Building type classification (residential, industrial, etc.)

### 2. **Road Polygons from Width** 🛣️

- Fetches road centerlines with width attributes (largeur)
- **Automatically generates road surface polygons** by buffering centerlines
- Configurable default width for roads without attributes
- Road type and importance classification

### 3. **Water Surfaces** 💧

- Rivers, lakes, ponds
- Surface hydrographique from BD TOPO®

### 4. **Vegetation Zones** 🌳

- Forest, parks, green spaces
- Zone de végétation classification

### 5. **Point Cloud Labeling**

- Spatial intersection-based labeling
- Configurable label priority for overlapping features
- Batch processing support

## 📦 New Files Added

```
ign_lidar/
├── io/
│   ├── wfs_ground_truth.py          # Main WFS fetcher implementation
│   └── __init__.py                  # Updated exports
├── cli/
│   └── commands/
│       ├── ground_truth.py          # CLI command
│       └── __init__.py              # Updated exports

examples/
└── ground_truth_fetching_example.py # Usage examples

docs/docs/features/
└── ground-truth-fetching.md         # Complete documentation
```

## 🚀 Quick Start

### Command Line

```bash
# Install dependencies
pip install shapely geopandas

# Generate patches with ground truth
ign-lidar-hd ground-truth data/tile_0650_6860.laz data/patches_gt

# Customize features
ign-lidar-hd ground-truth data/tile.laz data/patches \
    --patch-size 100 \
    --include-roads \
    --include-buildings \
    --no-water \
    --cache-dir cache/gt
```

### Python API

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

# Initialize fetcher
fetcher = IGNGroundTruthFetcher(cache_dir="cache/ground_truth")

# Fetch ground truth
bbox = (650000, 6860000, 651000, 6861000)  # Lambert 93
ground_truth = fetcher.fetch_all_features(bbox)

# Buildings
buildings = fetcher.fetch_buildings(bbox)

# Roads with polygons (generated from width!)
roads = fetcher.fetch_roads_with_polygons(bbox, default_width=4.0)
print(f"Average road width: {roads['width_m'].mean():.2f}m")

# Label points
labels = fetcher.label_points_with_ground_truth(
    points=points,
    ground_truth_features=ground_truth
)
```

## 🔧 Technical Details

### Road Polygon Generation

The system fetches road **centerlines** from BD TOPO® and generates **polygon surfaces**:

1. Fetch road centerline (LineString) from `BDTOPO_V3:troncon_de_route`
2. Read `largeur` or `largeur_de_chaussee` attribute (road width in meters)
3. Buffer centerline by `width / 2` on each side
4. Return polygon representing actual road surface

**Example:**

- Centerline: 100m long road
- Width attribute: 8 meters
- Generated polygon: 100m × 8m surface (800 m²)

### Label Assignment

Points labeled by spatial intersection:

```
Label Values:
  0: unlabeled/ground
  1: building
  2: road
  3: water
  4: vegetation
```

Priority order handles overlaps (e.g., building on road).

### Caching

WFS responses cached to disk to avoid redundant requests:

- First request: fetches from IGN WFS (slow)
- Subsequent requests: loads from cache (fast)
- Cache organized by bbox hash

## 📊 Use Cases

### 1. **Supervised Learning**

Generate labeled training data for semantic segmentation:

```python
patches = generate_patches_with_ground_truth(
    points=points,
    features=features,
    tile_bbox=tile_bbox,
    patch_size=150.0
)
```

### 2. **Road Network Analysis**

Analyze road widths and surface areas:

```python
roads = fetcher.fetch_roads_with_polygons(bbox)
total_road_area = roads['geometry'].area.sum()
avg_width = roads['width_m'].mean()
```

### 3. **Building Height Extraction**

Extract building heights for 3D reconstruction:

```python
buildings = fetcher.fetch_buildings(bbox)
tall_buildings = buildings[buildings['hauteur'] > 20.0]
```

### 4. **Multi-Scale Training**

Combine with existing multi-scale workflow:

```python
# 50m patches with ground truth
patches_50m = generate_patches_with_ground_truth(
    points, features, bbox, patch_size=50.0
)

# 150m patches with ground truth
patches_150m = generate_patches_with_ground_truth(
    points, features, bbox, patch_size=150.0
)
```

## 🔗 Integration with Existing Features

Works seamlessly with:

- ✅ RGB augmentation from orthophotos
- ✅ Geometric feature computation
- ✅ Multi-scale patch extraction
- ✅ Data augmentation pipeline
- ✅ Multiple ML architectures (PointNet++, Transformers, etc.)

## 🌐 Data Source

**IGN BD TOPO®** via Géoplateforme:

- Service: https://data.geopf.fr/wfs
- License: Licence Ouverte / Open License
- CRS: EPSG:2154 (Lambert 93)
- Coverage: Metropolitan France

## 📚 Documentation

Full documentation in:

- `docs/docs/features/ground-truth-fetching.md`
- `examples/ground_truth_fetching_example.py`

## 🐛 Testing

The functionality has been tested with:

- Paris region tiles
- Versailles tile (0650_6860)
- Various building densities
- Different road types (autoroutes, routes, chemins)

## 🔜 Future Enhancements

Potential additions:

- Railway track labeling
- Sports ground detection
- Tree canopy vs. understory vegetation
- Terrain elevation from DTM
- Building facade orientation
- Parking area detection

## 📞 Support

For issues or questions:

1. Check documentation: `docs/docs/features/ground-truth-fetching.md`
2. Run examples: `python examples/ground_truth_fetching_example.py`
3. Open GitHub issue with details

## ✅ Summary

This feature enables **automatic ground truth labeling** of LiDAR point clouds using official IGN vector data, with special emphasis on **road polygon generation from width attributes**. It seamlessly integrates with the existing pipeline for supervised learning workflows.

**Key Innovation:** Automatic conversion of road centerlines + width → road surface polygons! 🎉
