# 🚂🌲 Railways and BD Forêt® Integration

## Summary

Advanced classification module now supports:

1. **Railway (Rail) Classification** from IGN BD TOPO® (ASPRS code 10)
2. **Precise Forest Type Classification** from IGN BD Forêt® V2

---

## 🚂 Railway Classification

### Features

- **Data Source**: IGN BD TOPO® V3 - `BDTOPO_V3:troncon_de_voie_ferree` layer
- **ASPRS Code**: 10 (Rail)
- **Intelligent Buffering**: Similar to roads, based on track width and count

### Railway Attributes

| Attribute      | Type    | Description                      | Default |
| -------------- | ------- | -------------------------------- | ------- |
| `nombre_voies` | Integer | Number of tracks (1, 2, 3+)      | 1       |
| `largeur`      | Float   | Width per track (m)              | 3.5m    |
| `electrifie`   | String  | Electrified (OUI/NON)            | -       |
| `nature`       | String  | Track type (principale, service) | -       |

### Buffer Calculation

```python
# Single track: buffer = 3.5m / 2 = 1.75m radius
# Double track: buffer = (3.5m × 2) / 2 = 3.5m radius
# Triple track: buffer = (3.5m × 3) / 2 = 5.25m radius
```

### Implementation

**In `wfs_ground_truth.py`:**

```python
def fetch_railways_with_polygons(
    self,
    bbox: Tuple[float, float, float, float],
    width_fallback: float = 3.5
) -> Optional['gpd.GeoDataFrame']:
    """
    Fetch railway centerlines and convert to polygons using buffering.

    Default width: 3.5m for single track
    Multi-track: width × nombre_voies
    """
```

**In `advanced_classification.py`:**

```python
def _classify_railways_with_buffer(
    self,
    labels: np.ndarray,
    point_geoms: List[Point],
    railways_gdf: 'gpd.GeoDataFrame',
    asprs_class: int
) -> np.ndarray:
    """
    Classify railway points with intelligent buffering.
    Logs statistics: width range, track counts, points per railway.
    """
```

### Usage

```python
from ign_lidar.core.modules.advanced_classification import classify_with_all_features
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

# Initialize fetcher
fetcher = IGNGroundTruthFetcher(cache_dir="cache")

# Classify with railways enabled
labels, forest_attrs = classify_with_all_features(
    points=points,
    ground_truth_fetcher=fetcher,
    bbox=bbox,
    include_railways=True,  # 🚂 Enable railways
    ndvi=ndvi,
    height=height
)
```

### Example Output Logs

```
Processing railways: 23 features
  Using intelligent railway buffers (tolerance=0.5m)
  Railway widths: 3.5m - 10.5m (avg: 5.2m)
  Classified 8,450 railway points from 23 railways
  Avg points per railway: 367
  Track counts: [1, 2, 3] (single, double, etc.)
```

---

## 🌲 BD Forêt® V2 Integration

### Features

- **Data Source**: IGN BD Forêt® V2 - `BDFORET_V2:formation_vegetale` layer
- **Forest Types**: Coniferous, Deciduous, Mixed, Open/Closed
- **Species Information**: Up to 3 tree species per formation with coverage rates
- **Height Estimation**: Type-based tree height estimates (5-25m)

### Forest Type Codes (TFV)

| Code Pattern | Type            | Description                              |
| ------------ | --------------- | ---------------------------------------- |
| `FF1-*`      | Coniferous      | Closed coniferous forest (>75% conifers) |
| `FF2-*`      | Mixed           | Closed mixed forest (40-75% each type)   |
| `FF3-*`      | Deciduous       | Closed deciduous forest (>75% deciduous) |
| `FO1-*`      | Open Coniferous | Open coniferous forest                   |
| `FO2-*`      | Open Mixed      | Open mixed forest                        |
| `FO3-*`      | Open Deciduous  | Open deciduous forest                    |

### Forest Attributes

BD Forêt® provides rich attributes for each forest formation:

```python
{
    'code_tfv': 'FF3-00',              # Formation code
    'lib_tfv': 'Forêt fermée feuillus', # Human-readable type
    'essence_1': 'Chêne',               # Dominant species
    'taux_1': 60,                       # 60% coverage
    'essence_2': 'Hêtre',               # Secondary species
    'taux_2': 30,                       # 30% coverage
    'essence_3': 'Charme',              # Tertiary species
    'taux_3': 10,                       # 10% coverage
    'densite': 'Fermée',                # Density: Fermée/Ouverte
    'structure': 'Mature'               # Structure: Mature/Jeune
}
```

### Returned Forest Attributes

The `classify_with_all_features` function returns a dictionary with per-point attributes:

```python
labels, forest_attrs = classify_with_all_features(...)

# forest_attrs structure:
{
    'forest_type': ['coniferous', 'deciduous', 'mixed', ...],  # [N] Forest type
    'primary_species': ['Chêne', 'Sapin', ...],                # [N] Dominant species
    'species_rate': [60, 75, ...],                             # [N] Coverage % of primary species
    'density': ['closed', 'open', ...],                        # [N] Density category
    'structure': ['mature', 'young', ...],                     # [N] Forest structure
    'estimated_height': [15.0, 20.0, ...]                      # [N] Estimated tree height (m)
}
```

### Height Estimation by Forest Type

| Forest Type         | Estimated Height Range |
| ------------------- | ---------------------- |
| Coniferous (closed) | 15-25m                 |
| Deciduous (closed)  | 12-20m                 |
| Mixed (closed)      | 10-18m                 |
| Open forests        | 5-12m                  |
| Young forests       | 5-10m                  |

### Implementation

**New Module: `io/bd_foret.py`**

Key classes and methods:

1. **`BDForetFetcher`** - Main fetcher class
   - `fetch_forest_polygons()` - Retrieve forest formations via WFS
   - `label_points_with_forest_type()` - Assign forest types to vegetation points
2. **`ForestType`** - Type classification helper
   - `from_code_tfv()` - Extract forest type from TFV code
   - `estimate_height()` - Get height range for forest type

**Integration in `advanced_classification.py`:**

```python
# In classify_with_all_features()
if include_forest and bd_foret_fetcher and bbox:
    logger.info("Refining vegetation classification with BD Forêt® V2...")

    # Fetch forest polygons
    forest_gdf = bd_foret_fetcher.fetch_forest_polygons(bbox)

    # Label vegetation points (ASPRS 3, 4, 5) with forest types
    forest_attributes = bd_foret_fetcher.label_points_with_forest_type(
        points=points,
        labels=labels,
        forest_gdf=forest_gdf
    )
```

### Usage Example

```python
from ign_lidar.core.modules.advanced_classification import classify_with_all_features
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.io.bd_foret import BDForetFetcher

# Initialize fetchers
gt_fetcher = IGNGroundTruthFetcher(cache_dir="cache/ground_truth")
forest_fetcher = BDForetFetcher(cache_dir="cache/bd_foret")

# Classify with both railways and forest types
labels, forest_attrs = classify_with_all_features(
    points=points,
    ground_truth_fetcher=gt_fetcher,
    bd_foret_fetcher=forest_fetcher,
    bbox=bbox,
    ndvi=ndvi,
    height=height,
    include_railways=True,    # 🚂 Railway classification
    include_forest=True       # 🌲 BD Forêt® forest types
)

# Analyze forest types
if forest_attrs:
    import pandas as pd
    from collections import Counter

    # Vegetation mask (ASPRS 3, 4, 5)
    veg_mask = np.isin(labels, [3, 4, 5])

    # Count forest types
    forest_types = [ft for ft in forest_attrs['forest_type'] if ft != 'unknown']
    type_counts = Counter(forest_types)

    print("Forest type distribution:")
    for ftype, count in type_counts.most_common():
        pct = 100 * count / len(forest_types)
        print(f"  {ftype:20s}: {count:8,} ({pct:5.1f}%)")

    # Top tree species
    species = [s for s in forest_attrs['primary_species'] if s != 'unknown']
    species_counts = Counter(species)

    print("\nTop 5 tree species:")
    for species_name, count in species_counts.most_common(5):
        print(f"  {species_name:20s}: {count:8,} points")
```

### Example Output

```
Refining vegetation classification with BD Forêt® V2...
  Fetching forest polygons...
  Found 142 forest formations
  Labeling 1,335,700 vegetation points...
  Labeled 1,203,450 vegetation points with forest types
    coniferous: 456,200 points
    mixed: 387,100 points
    deciduous: 360,150 points

Forest type distribution:
  coniferous          :  456,200 ( 37.9%)
  mixed               :  387,100 ( 32.2%)
  deciduous           :  360,150 ( 29.9%)

Top 5 tree species:
  Sapin               :  234,500 points
  Chêne               :  198,300 points
  Hêtre               :  145,200 points
  Épicéa              :  112,800 points
  Douglas             :   98,600 points
```

---

## 🎯 Complete Workflow Example

### Full Pipeline with Railways and BD Forêt®

```python
from pathlib import Path
import numpy as np
import laspy

from ign_lidar.core.modules.advanced_classification import classify_with_all_features
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.io.bd_foret import BDForetFetcher
from ign_lidar.features.geometric import compute_geometric_features
from ign_lidar.core.modules.enrichment import compute_ndvi

# 1. Load LAZ file
las = laspy.read("input.laz")
points = np.vstack([las.x, las.y, las.z]).T
bbox = (las.header.x_min, las.header.y_min,
        las.header.x_max, las.header.y_max)

# 2. Compute features
rgb = np.vstack([las.red, las.green, las.blue]).T / 65535.0
nir = np.array(las.nir) / 65535.0
ndvi = compute_ndvi(rgb, nir)

height, normals, planarity, curvature = compute_geometric_features(
    points, k_neighbors=20
)

# 3. Initialize fetchers
gt_fetcher = IGNGroundTruthFetcher(cache_dir="cache/gt")
forest_fetcher = BDForetFetcher(cache_dir="cache/forest")

# 4. Advanced classification
labels, forest_attrs = classify_with_all_features(
    points=points,
    ground_truth_fetcher=gt_fetcher,
    bd_foret_fetcher=forest_fetcher,
    bbox=bbox,
    ndvi=ndvi,
    height=height,
    normals=normals,
    planarity=planarity,
    curvature=curvature,
    include_railways=True,     # 🚂
    include_forest=True,       # 🌲
    road_buffer_tolerance=0.5,
    ndvi_veg_threshold=0.35
)

# 5. Save results
las.classification = labels

# Save forest height as extra dimension
if forest_attrs:
    las.add_extra_dim(laspy.ExtraBytesParams(
        name='forest_height',
        type=np.float32,
        description='BD Forêt estimated height'
    ))
    las.forest_height = np.array(forest_attrs['estimated_height'], dtype=np.float32)

las.write("output_classified.laz")

# 6. Generate statistics
print(f"\nClassification complete: {len(points):,} points")
print(f"\nASPRS distribution:")
for code in [1, 2, 3, 4, 5, 6, 9, 10, 11]:
    count = (labels == code).sum()
    if count > 0:
        pct = 100 * count / len(points)
        print(f"  {code:2d}: {count:10,} ({pct:5.1f}%)")

if forest_attrs:
    from collections import Counter
    print(f"\nForest types:")
    type_counts = Counter(forest_attrs['forest_type'])
    for ftype, count in type_counts.most_common():
        if ftype != 'unknown':
            print(f"  {ftype:20s}: {count:8,}")
```

---

## 📊 Performance Considerations

### Memory Usage

- **Railways**: Minimal overhead (typically 10-50 railway features per tile)
- **BD Forêt®**: Moderate overhead (50-200 forest polygons per tile)
- **Forest Attributes**: ~40 bytes per vegetation point

### Processing Time

Approximate processing time for 1M points:

| Operation              | Time        | Notes                               |
| ---------------------- | ----------- | ----------------------------------- |
| Railway fetching       | 1-2s        | WFS query + buffering               |
| Railway classification | 2-5s        | Spatial containment checks          |
| BD Forêt® fetching     | 2-4s        | WFS query                           |
| Forest labeling        | 5-10s       | Spatial join + attribute extraction |
| **Total overhead**     | **~10-20s** | On top of base classification       |

### Optimization Tips

1. **Cache WFS Results**: Both fetchers use caching by default
2. **Limit Spatial Queries**: Use tight bounding boxes
3. **Skip Empty Areas**: Check if features exist before labeling
4. **Batch Processing**: Fetch ground truth once for multiple tiles

---

## 🔧 Configuration Options

### Railway Configuration

```python
# In fetch_railways_with_polygons()
width_fallback=3.5  # Default track width when attribute missing

# In classify_with_all_features()
include_railways=True  # Enable/disable railway classification
```

### BD Forêt® Configuration

```python
# In classify_with_all_features()
include_forest=True  # Enable/disable forest type refinement

# Forest type classification only affects ASPRS codes 3, 4, 5 (vegetation)
# Does not override ground truth buildings or other non-vegetation classes
```

---

## 📚 Related Documentation

- **Advanced Classification Guide**: `docs/ADVANCED_CLASSIFICATION_GUIDE.md`
- **WFS Ground Truth Module**: `ign_lidar/io/wfs_ground_truth.py`
- **BD Forêt® Module**: `ign_lidar/io/bd_foret.py`
- **Example Script**: `examples/example_advanced_classification.py`

---

## ✅ Testing

### Test Railway Classification

```bash
# Find a tile with railways
python scripts/select_optimal_tiles.py \
    --region "Île-de-France" \
    --features railways \
    --output tiles_with_railways.json

# Process with railways enabled
python examples/example_advanced_classification.py \
    --input data/test_tile.laz \
    --output data/test_classified.laz \
    --include-railways
```

### Test BD Forêt® Classification

```bash
# Find a forested tile
python scripts/select_optimal_tiles.py \
    --region "Vosges" \
    --features forest \
    --min-forest-coverage 0.5 \
    --output forested_tiles.json

# Process with BD Forêt®
python examples/example_advanced_classification.py \
    --input data/forest_tile.laz \
    --output data/forest_classified.laz \
    --include-forest
```

---

## 🎉 Benefits

### Railways

- **Accurate Track Detection**: Follows real-world track layouts from official data
- **Multi-track Support**: Handles single, double, triple track sections
- **Standardized Code**: Uses ASPRS code 10 for interoperability

### BD Forêt®

- **Precise Vegetation Types**: Move beyond simple height-based classification
- **Species Information**: Know what trees are in your point cloud
- **Height Estimates**: Get realistic tree height ranges by forest type
- **Ecological Context**: Understand forest structure (closed/open, mature/young)

---

## 🚀 Future Enhancements

Potential additions:

1. **More BD TOPO® Features**:

   - Bridges (`pont`)
   - Power lines (`ligne_electrique`)
   - Sports facilities (`terrain_de_sport`)

2. **BD Forêt® Extensions**:

   - Save species attributes as extra LAZ dimensions
   - Forest age/maturity classification
   - Integration with height models for canopy analysis

3. **Advanced Railway Features**:
   - Platform detection at stations
   - Catenary wire classification (electrified lines)
   - Railway embankments and cuttings

---

**Author**: Classification Enhancement  
**Date**: October 15, 2025  
**Version**: 1.0
