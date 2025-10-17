---
sidebar_position: 3
title: BD TOPO Integration
description: IGN BD TOPO® data source integration for ground truth classification
---

# BD TOPO® Integration Reference

Complete guide to integrating IGN's BD TOPO® V3 database for authoritative ground truth classification.

---

## 🎯 Overview

BD TOPO® is France's national topographic database maintained by IGN, providing:

- **Authoritative vector data** - Official French topographic information
- **High accuracy** - Survey-grade geometric precision
- **Rich attributes** - Detailed feature classification
- **National coverage** - Complete coverage of France
- **Regular updates** - Continuous maintenance and updates

**This library integrates BD TOPO® V3** through WFS (Web Feature Service) for real-time ground truth fetching.

---

## 📊 Available Layers

### Core Layers (Always Available)

| Layer Name     | BD TOPO® Table                     | ASPRS Class | Description         |
| -------------- | ---------------------------------- | ----------- | ------------------- |
| **Buildings**  | `BDTOPO_V3:batiment`               | 6           | Building footprints |
| **Roads**      | `BDTOPO_V3:troncon_de_route`       | 11          | Road segments       |
| **Railways**   | `BDTOPO_V3:troncon_de_voie_ferree` | 10          | Railway tracks      |
| **Water**      | `BDTOPO_V3:surface_hydrographique` | 9           | Water surfaces      |
| **Vegetation** | `BDTOPO_V3:zone_de_vegetation`     | 3-5         | Vegetation zones    |

### Extended Layers (Optional)

| Layer Name            | BD TOPO® Table                      | ASPRS Class | Description           |
| --------------------- | ----------------------------------- | ----------- | --------------------- |
| **Sports Facilities** | `BDTOPO_V3:terrain_de_sport`        | 91          | Sports grounds        |
| **Cemeteries**        | `BDTOPO_V3:cimetiere`               | 90          | Cemetery areas        |
| **Power Lines**       | `BDTOPO_V3:ligne_electrique`        | 92          | Power line corridors  |
| **Constructions**     | `BDTOPO_V3:construction_surfacique` | 6           | Surface constructions |
| **Reservoirs**        | `BDTOPO_V3:reservoir`               | 84          | Water tanks           |

### Deprecated Layers (Not Available in V3)

| Layer Name  | Status     | Notes                             |
| ----------- | ---------- | --------------------------------- |
| **Bridges** | ❌ Removed | No longer available in BD TOPO V3 |
| **Parking** | ❌ Removed | No longer available in BD TOPO V3 |

---

## ⚙️ Configuration

### V5 Complete Configuration

```yaml
# config.yaml (V5)
data_sources:
  bd_topo:
    # Enable/disable BD TOPO integration
    enabled: true

    # Features to fetch
    features:
      buildings: true # Building footprints
      roads: true # Road network
      railways: false # Railway tracks (optional)
      water: true # Water surfaces
      vegetation: true # Vegetation zones
      sports_facilities: false # Sports grounds (optional)
      cemeteries: false # Cemeteries (optional)
      power_lines: false # Power lines (optional)

    # WFS Service Configuration
    wfs_url: "https://data.geopf.fr/wfs"
    wfs_version: "2.0.0"
    output_format: "application/json"
    max_features: 10000 # Features per WFS request
    timeout: 30 # Request timeout (seconds)

    # Cache Configuration (V5 - auto-uses input folder)
    cache_enabled: true
    cache_dir: null # null = {input_dir}/cache/ground_truth
    use_global_cache: false # Use global cache instead

    # Performance
    use_gpu: true # GPU-accelerated point-in-polygon
    parallel_fetch: true # Fetch layers in parallel
    simplify_geometries: false # Simplify complex polygons
    simplify_tolerance: 0.5 # Simplification tolerance (meters)

    # Road-specific settings
    road_buffer: 2.5 # Buffer around road centerlines (meters)
    road_min_width: 2.0 # Minimum road width
    road_max_width: 50.0 # Maximum road width

    # Priority for overlapping features
    feature_priority:
      buildings: 1 # Highest priority
      roads: 2
      railways: 2
      water: 3
      vegetation: 4 # Lowest priority
```

### Minimal Configuration

```yaml
# Minimal BD TOPO setup
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
```

---

## 🔄 Feature Attribute Mapping

### Buildings (batiment)

BD TOPO® attributes mapped to ASPRS classes:

```python
BUILDING_TYPE_MAPPING = {
    'nature': {
        'Indifférencié': 50,      # Residential
        'Commercial': 51,          # Commercial
        'Industriel': 52,          # Industrial
        'Religieux': 53,           # Religious
        'Administratif': 54,       # Public
        'Agricole': 55,            # Agricultural
        'Sportif': 56,             # Sports
        'Remarquable': 57,         # Historic
    },
    'default': 6  # Standard building class
}
```

**Key Attributes**:

- `nature` - Building type
- `hauteur` - Building height (meters)
- `nombre_de_niveaux` - Number of floors
- `leger` - Lightweight construction (boolean)

### Roads (troncon_de_route)

Road segments with width information for polygon generation:

```python
ROAD_TYPE_MAPPING = {
    'nature': {
        'Autoroute': 32,           # Motorway
        'Route à 1 chaussée': 33,  # Single carriageway
        'Route à 2 chaussées': 34, # Dual carriageway
        'Bretelle': 35,            # Ramp
        'Chemin': 36,              # Path
        'Sentier': 38,             # Trail
        'Piste cyclable': 39,      # Cycle path
    },
    'default': 11  # Standard road surface
}
```

**Key Attributes**:

- `nature` - Road type
- `largeur` - Road width (meters)
- `pos_sol` - Position relative to ground (0=ground, 1=bridge, -1=tunnel)
- `importance` - Road importance (1-5)

### Water (surface_hydrographique)

Water surfaces:

```python
WATER_TYPE_MAPPING = {
    'nature': {
        'Cours d\'eau': 80,        # River
        'Plan d\'eau': 81,         # Lake
        'Réservoir': 84,           # Reservoir
        'Canal': 83,               # Canal
    },
    'default': 9  # Standard water class
}
```

### Vegetation (zone_de_vegetation)

Vegetation zones classified by type and height:

```python
VEGETATION_MAPPING = {
    'nature': {
        'Forêt fermée': 74,        # Forest
        'Forêt ouverte': 74,       # Open forest
        'Haie': 73,                # Hedge
        'Verger': 76,              # Orchard
        'Vigne': 75,               # Vineyard
        'Lande': 71,               # Shrubland
    },
    'height': {
        '< 0.5m': 3,   # Low vegetation
        '0.5-2m': 4,   # Medium vegetation
        '> 2m': 5,     # High vegetation
    }
}
```

---

## 💻 Python API

### Basic Integration

```python
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
import numpy as np

# Initialize fetcher
fetcher = IGNGroundTruthFetcher(
    cache_dir=None,  # Auto-use input_dir/cache/ground_truth
    verbose=True
)

# Compute bounding box from LiDAR tile
bbox = (
    points[:, 0].min(),  # xmin
    points[:, 1].min(),  # ymin
    points[:, 0].max(),  # xmax
    points[:, 1].max()   # ymax
)

# Fetch BD TOPO features
ground_truth = fetcher.fetch_all_features(
    bbox=bbox,
    include_buildings=True,
    include_roads=True,
    include_water=True,
    include_vegetation=True
)

print(f"Fetched {len(ground_truth['buildings'])} buildings")
print(f"Fetched {len(ground_truth['roads'])} road segments")
print(f"Fetched {len(ground_truth['water'])} water surfaces")
```

### Feature-Specific Fetching

```python
# Fetch only buildings
buildings = fetcher.fetch_buildings(bbox=bbox)
print(f"Building types: {buildings['nature'].unique()}")
print(f"Height range: {buildings['hauteur'].min():.1f}m - {buildings['hauteur'].max():.1f}m")

# Fetch roads with attributes
roads = fetcher.fetch_roads(bbox=bbox)
print(f"Road types: {roads['nature'].unique()}")
print(f"Width range: {roads['largeur'].min():.1f}m - {roads['largeur'].max():.1f}m")

# Generate road polygons from centerlines
road_polygons = fetcher.generate_road_polygons(
    roads_gdf=roads,
    buffer_distance=2.5  # meters
)
```

### Classification with BD TOPO

```python
# Classify points using BD TOPO ground truth
labels = fetcher.label_points_with_ground_truth(
    points=points,
    ground_truth_features=ground_truth,
    priority_order={
        'buildings': 1,
        'roads': 2,
        'water': 3,
        'vegetation': 4
    }
)

# Check classification coverage
unique, counts = np.unique(labels, return_counts=True)
for cls, count in zip(unique, counts):
    pct = 100 * count / len(labels)
    print(f"Class {cls}: {count:8d} points ({pct:5.2f}%)")
```

---

## 🎨 Advanced Features

### Parallel Layer Fetching

Fetch multiple layers simultaneously for faster processing:

```python
from concurrent.futures import ThreadPoolExecutor

def fetch_layer(layer_name, bbox):
    """Fetch a single layer."""
    if layer_name == 'buildings':
        return fetcher.fetch_buildings(bbox)
    elif layer_name == 'roads':
        return fetcher.fetch_roads(bbox)
    elif layer_name == 'water':
        return fetcher.fetch_water(bbox)
    # ... etc

# Parallel fetch
layers = ['buildings', 'roads', 'water', 'vegetation']
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(
        lambda layer: fetch_layer(layer, bbox),
        layers
    ))

buildings, roads, water, vegetation = results
```

### Geometry Simplification

Reduce polygon complexity for faster processing:

```python
from shapely.geometry import shape
from shapely.ops import unary_union

# Simplify building footprints
buildings_simplified = buildings.copy()
buildings_simplified['geometry'] = buildings_simplified['geometry'].simplify(
    tolerance=0.5,  # meters
    preserve_topology=True
)

# Merge adjacent buildings
building_clusters = unary_union(buildings_simplified['geometry'])
```

### Custom Buffering for Linear Features

```python
# Variable road buffering based on width attribute
def buffer_roads_variable(roads_gdf):
    """Buffer roads based on their width attribute."""
    buffered = []
    for idx, road in roads_gdf.iterrows():
        width = road.get('largeur', 3.0)  # Default 3m if missing
        buffer_dist = width / 2.0
        buffered_geom = road.geometry.buffer(buffer_dist)
        buffered.append(buffered_geom)

    roads_gdf = roads_gdf.copy()
    roads_gdf['geometry'] = buffered
    return roads_gdf

roads_buffered = buffer_roads_variable(roads)
```

---

## 📈 Performance Optimization

### WFS Request Optimization

```yaml
data_sources:
  bd_topo:
    # Optimize WFS requests
    max_features: 10000 # Increase for fewer requests
    timeout: 60 # Increase for complex geometries

    # Enable parallel fetching
    parallel_fetch: true
    max_workers: 4
```

### Cache Strategy

```python
# Pre-fetch and cache for entire region
from pathlib import Path

def prefetch_region(tiles, cache_dir):
    """Pre-fetch BD TOPO data for all tiles."""
    fetcher = IGNGroundTruthFetcher(cache_dir=cache_dir)

    for tile_path in tiles:
        # Load tile to get bbox
        las = laspy.read(tile_path)
        bbox = (las.x.min(), las.y.min(), las.x.max(), las.y.max())

        # Fetch and cache
        print(f"Prefetching {tile_path.name}...")
        fetcher.fetch_all_features(bbox=bbox, use_cache=True)

# Pre-fetch once, then use cache for all processing
tiles = list(Path("data/tiles").glob("*.laz"))
prefetch_region(tiles, cache_dir="data/cache/ground_truth")
```

### GPU Acceleration

```python
from ign_lidar.io.ground_truth_optimizer import GroundTruthOptimizer

# Use GPU for point-in-polygon classification
optimizer = GroundTruthOptimizer(
    use_gpu=True,
    gpu_batch_size=8_000_000
)

labels_gpu = optimizer.classify_with_ground_truth(
    points=points,
    ground_truth_features=ground_truth
)

# Performance: 10-20x faster than CPU for large tiles
```

---

## 🔍 Troubleshooting

### Common Issues

#### WFS Service Errors

```python
# Handle connection errors gracefully
from requests.exceptions import RequestException
import time

def fetch_with_retry(fetcher, bbox, max_retries=3):
    """Fetch with automatic retry."""
    for attempt in range(max_retries):
        try:
            return fetcher.fetch_all_features(bbox=bbox)
        except RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

#### Empty Results

```python
# Check if bbox is valid
def validate_bbox(bbox):
    """Validate bounding box."""
    xmin, ymin, xmax, ymax = bbox

    # Check order
    assert xmin < xmax, "xmin must be < xmax"
    assert ymin < ymax, "ymin must be < ymax"

    # Check CRS (Lambert 93)
    assert 0 < xmin < 1_300_000, "Invalid Lambert 93 X coordinate"
    assert 6_000_000 < ymin < 7_200_000, "Invalid Lambert 93 Y coordinate"

    # Check size
    width = xmax - xmin
    height = ymax - ymin
    assert width < 10_000, f"Bbox too large: {width}m wide"
    assert height < 10_000, f"Bbox too large: {height}m tall"

    return True

validate_bbox(bbox)
```

#### Attribute Errors

```python
# Handle missing attributes gracefully
def safe_get_attribute(feature, attribute, default=None):
    """Safely get feature attribute."""
    return feature.get(attribute, default)

# Example
building_height = safe_get_attribute(building, 'hauteur', 10.0)
road_width = safe_get_attribute(road, 'largeur', 3.0)
```

---

## 📚 Data Quality Considerations

### Coverage

- **Urban areas**: Excellent coverage and detail
- **Rural areas**: Good coverage, less detail
- **Remote areas**: Basic coverage
- **Update frequency**: Varies by region (annual to multi-year)

### Accuracy

- **Planimetric**: ±50cm to ±5m (depending on source)
- **Altimetric**: ±1m to ±5m
- **Attributes**: Generally reliable but may have missing values
- **Classification**: High quality but may need validation

### Limitations

1. **Temporal Mismatch**: BD TOPO may not match LiDAR acquisition date
2. **Generalization**: Some features simplified/generalized
3. **Missing Features**: Not all features always captured
4. **Attribute Completeness**: Some attributes may be null
5. **Geometric Precision**: Varies by feature type and source

---

## 💡 Best Practices

1. **Always cache** - WFS requests are slow, caching is essential
2. **Validate geometries** - Check for invalid/self-intersecting polygons
3. **Handle missing attributes** - Not all features have all attributes
4. **Use appropriate buffers** - Different features need different buffers
5. **Priority matters** - Set sensible priorities for overlapping features
6. **Monitor WFS quotas** - IGN may have rate limits
7. **Update cache periodically** - BD TOPO is regularly updated
8. **Validate results** - Visual QC recommended for critical applications

---

## 📖 See Also

- [Ground Truth Classification](../features/ground-truth-classification.md)
- [ASPRS Classification Reference](./asprs-classification.md)
- [Classification Workflow](./classification-workflow.md)
- [IGN Géoportail Documentation](https://geoservices.ign.fr/documentation)

---

## 📝 External Resources

- **BD TOPO® Documentation**: https://geoservices.ign.fr/bdtopo
- **WFS Service**: https://data.geopf.fr/wfs
- **WFS Capabilities**: https://data.geopf.fr/wfs?SERVICE=WFS&REQUEST=GetCapabilities
- **Layer Descriptions**: https://geoservices.ign.fr/documentation/donnees/vecteur/bdtopo

---

**Data Source**: IGN BD TOPO® V3  
**Updated**: October 17, 2025 - V5 Configuration
