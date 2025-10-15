# Summary: WFS Ground Truth Integration

## What Was Added

Integrated **IGN BD TOPO® WFS service** into the codebase to fetch ground truth vector data for automatic point cloud labeling. This enables supervised learning by providing semantic labels from official topographic data.

## Key Innovation

**Automatic road polygon generation from centerlines + width attributes:**

- BD TOPO® provides road centerlines (LineString) with `largeur` (width) attribute
- System automatically buffers centerlines to generate road surface polygons
- Result: accurate road surface representation for point cloud labeling

## Files Created/Modified

### New Files

1. **`ign_lidar/io/wfs_ground_truth.py`** (585 lines)

   - `IGNGroundTruthFetcher` class - main WFS fetcher
   - `IGNWFSConfig` - WFS service configuration
   - Helper functions for tile processing
   - Point cloud labeling logic

2. **`ign_lidar/cli/commands/ground_truth.py`** (150 lines)

   - CLI command: `ign-lidar-hd ground-truth`
   - Configurable feature selection
   - Batch processing support

3. **`examples/ground_truth_fetching_example.py`** (330 lines)

   - 4 complete usage examples
   - Road polygon generation demo
   - Multi-tile batch processing

4. **`docs/docs/features/ground-truth-fetching.md`** (450 lines)

   - Complete documentation
   - API reference
   - Troubleshooting guide

5. **`WFS_GROUND_TRUTH_FEATURE.md`** (210 lines)
   - Feature summary
   - Quick start guide
   - Technical details

### Modified Files

1. **`ign_lidar/io/__init__.py`**

   - Added exports for WFS ground truth functionality
   - Conditional import (requires shapely/geopandas)

2. **`ign_lidar/cli/commands/__init__.py`**

   - Registered `ground_truth_command`

3. **`ign_lidar/cli/main.py`**

   - Added ground-truth command to CLI

4. **`requirements.txt`**
   - Added `shapely>=2.0.0`
   - Added `geopandas>=0.12.0`

## Features Provided

### 1. Vector Data Fetching

- **Buildings**: Footprints with height/type attributes
- **Roads**: Centerlines → surface polygons using width
- **Water**: Rivers, lakes, ponds
- **Vegetation**: Forest, parks, green spaces

### 2. Point Cloud Labeling

- Spatial intersection-based labeling
- Configurable label priority (overlaps)
- Label values: 0=ground, 1=building, 2=road, 3=water, 4=vegetation

### 3. Patch Generation

- Complete workflow: tile → ground truth → labeled patches
- Integration with existing patch extraction
- Caching to avoid redundant WFS requests

## Usage

### Command Line

```bash
# Install dependencies
pip install shapely geopandas

# Generate patches with ground truth
ign-lidar-hd ground-truth data/tile.laz data/patches_gt \
    --patch-size 150 \
    --include-roads \
    --include-buildings \
    --cache-dir cache/gt
```

### Python API

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

# Fetch ground truth
fetcher = IGNGroundTruthFetcher(cache_dir="cache")
bbox = (650000, 6860000, 651000, 6861000)  # Lambert 93

# Roads with polygon generation
roads = fetcher.fetch_roads_with_polygons(bbox)
print(f"Average width: {roads['width_m'].mean():.2f}m")

# Buildings
buildings = fetcher.fetch_buildings(bbox)

# All features
ground_truth = fetcher.fetch_all_features(bbox)

# Label points
labels = fetcher.label_points_with_ground_truth(
    points=points,
    ground_truth_features=ground_truth
)
```

## Technical Details

### Road Polygon Generation Algorithm

1. Fetch road centerline (LineString) from `BDTOPO_V3:troncon_de_route`
2. Extract width from `largeur` or `largeur_de_chaussee` attribute
3. Buffer centerline by `width / 2` meters on each side
4. Return Polygon representing road surface

### WFS Service

- **URL**: https://data.geopf.fr/wfs
- **CRS**: EPSG:2154 (Lambert 93)
- **Format**: GeoJSON
- **License**: Licence Ouverte / Open License
- **Coverage**: Metropolitan France

### Caching Strategy

- Ground truth data cached by bbox hash
- Avoids redundant WFS requests
- Configurable cache directory
- Can force refresh with `use_cache=False`

## Integration

Works seamlessly with existing features:

- ✅ RGB augmentation from orthophotos
- ✅ Infrared (NIR) augmentation
- ✅ Geometric feature computation
- ✅ Multi-scale patch extraction (50m, 100m, 150m)
- ✅ Data augmentation pipeline
- ✅ Multiple architectures (PointNet++, Transformers, etc.)

## Use Cases

1. **Supervised Semantic Segmentation**: Train models with ground truth labels
2. **Road Network Analysis**: Compute road widths and surface areas
3. **Building Detection**: Label building points with height attributes
4. **Multi-Class Classification**: Buildings, roads, water, vegetation
5. **Dataset Validation**: Compare predicted vs. ground truth labels

## Dependencies

**Required:**

- `shapely>=2.0.0` - Geometric operations
- `geopandas>=0.12.0` - GeoDataFrame support

**Installation:**

```bash
pip install shapely geopandas
```

## Testing

Tested with:

- ✅ Paris region (dense urban)
- ✅ Versailles (mixed urban/parks)
- ✅ Various road types (autoroutes, routes, chemins)
- ✅ Multiple tile sizes (50m-150m patches)
- ✅ Batch processing (multiple tiles)

## Documentation

Complete documentation available:

- `docs/docs/features/ground-truth-fetching.md` - Full guide
- `examples/ground_truth_fetching_example.py` - Working examples
- `WFS_GROUND_TRUTH_FEATURE.md` - Feature summary

## Future Enhancements

Potential additions:

- Railway track polygons (similar to roads)
- Sports ground labeling
- Tree canopy classification
- Building facade orientation
- Terrain elevation from DTM
- Parking area detection

## Summary Statistics

- **Total lines of code added**: ~1,700
- **New Python modules**: 2
- **Documentation pages**: 2
- **Example scripts**: 1
- **CLI commands**: 1
- **New dependencies**: 2

This feature enables **automatic supervised learning** by fetching official ground truth data from IGN and labeling point clouds with semantic classes. The key innovation is **automatic road surface polygon generation from centerlines + width attributes**.
