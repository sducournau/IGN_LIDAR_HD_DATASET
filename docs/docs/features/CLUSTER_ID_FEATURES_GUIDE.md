# Building Cluster ID and Parcel Cluster ID Features Guide

**Configuration Version:** 6.3.1  
**Date:** October 25, 2025  
**Status:** ✅ Already Enabled in `asprs_complete.yaml`

---

## Overview

The **Building Cluster ID** and **Parcel Cluster ID** features are advanced object identification features that assign unique identifiers to points based on their association with specific buildings or cadastral parcels. These features are essential for:

- **Object-based analysis:** Group points by building or parcel
- **Instance segmentation:** Separate individual building instances
- **Property analysis:** Link LiDAR data to cadastral information
- **Change detection:** Track changes to specific buildings/parcels over time

---

## ✅ Current Configuration Status

Your `asprs_complete.yaml` configuration **already has these features enabled**:

### 1. Feature Computation (Lines 106-109)

```yaml
features:
  # Cluster ID features (building and parcel identification)
  compute_cluster_id: true # Spatial clustering for object identification
  compute_building_cluster_id: true # Building-specific cluster IDs from BD TOPO
  compute_parcel_cluster_id: true # Cadastral parcel cluster IDs
```

### 2. Ground Truth Assignment (Lines 296-297)

```yaml
ground_truth:
  bd_topo:
    # Cluster ID assignment
    assign_building_cluster_ids: true # Assign unique ID per building polygon
    assign_parcel_cluster_ids: true # Assign unique ID per cadastral parcel
```

### 3. Output Saving (Lines 499-501)

```yaml
output:
  extra_dims:
    # Cluster IDs (object identification)
    - cluster_id # General spatial cluster ID
    - building_cluster_id # Building-specific ID from BD TOPO
    - parcel_cluster_id # Cadastral parcel ID
```

---

## 🔍 Feature Descriptions

### 1. `cluster_id` - General Spatial Clustering

- **Type:** Integer (int32)
- **Range:** 0 to N (number of detected clusters)
- **Method:** DBSCAN or similar spatial clustering
- **Purpose:** Groups nearby points into spatial clusters regardless of class
- **Use case:** Detect connected objects, segment point clouds into instances

**Example values:**

- `0` - Noise/unassigned points
- `1, 2, 3, ...` - Individual cluster IDs

### 2. `building_cluster_id` - Building Instance IDs

- **Type:** Integer (int32)
- **Range:** 0 to N (number of buildings in BD TOPO)
- **Method:** Spatial intersection with BD TOPO building polygons
- **Purpose:** Assign unique ID to each building from ground truth
- **Use case:** Building-level analysis, facade extraction, architectural studies

**Example values:**

- `0` - Not inside any building polygon
- `1001, 1002, 1003, ...` - Unique building IDs from BD TOPO

**Key features:**

- Links points to official building registry (BD TOPO)
- Preserves building identity across tiles
- Enables building-level statistics and analysis

### 3. `parcel_cluster_id` - Cadastral Parcel IDs

- **Type:** Integer (int32)
- **Range:** 0 to N (number of parcels)
- **Method:** Spatial intersection with cadastral parcel polygons
- **Source:** French cadastre (PARCELLE.shp)
- **Purpose:** Link points to land ownership parcels
- **Use case:** Property analysis, urban planning, land use studies

**Example values:**

- `0` - Not inside any cadastral parcel
- `2001, 2002, 2003, ...` - Unique parcel IDs from cadastre

**Key features:**

- Links points to official land registry
- Enables property-level aggregation
- Useful for legal/administrative boundaries

---

## 📊 How Cluster IDs Work

### Processing Pipeline

```
1. Load Point Cloud
   ↓
2. Compute Features (geometric, spectral, height)
   ↓
3. Load Ground Truth (BD TOPO + Cadastre)
   ↓
4. Spatial Clustering (compute cluster_id)
   ↓
5. Building Assignment (compute building_cluster_id)
   ├─ Check if point falls inside building polygon (with buffer)
   ├─ Assign building ID from BD TOPO
   └─ Use 3D bounding box if extrude_3d=true
   ↓
6. Parcel Assignment (compute parcel_cluster_id)
   ├─ Check if point falls inside parcel polygon
   └─ Assign parcel ID from cadastre
   ↓
7. Save to LAZ (cluster IDs as extra dimensions)
```

### Assignment Logic

**Building Cluster ID:**

```python
# For each point (x, y, z):
for building in bd_topo_buildings:
    if point_inside_building_polygon(x, y, building):
        # Check vertical extent if 3D extrusion enabled
        if extrude_3d and (z_min <= z <= z_max):
            building_cluster_id = building.id
        else:
            building_cluster_id = building.id
        break
else:
    building_cluster_id = 0  # Not in any building
```

**Parcel Cluster ID:**

```python
# For each point (x, y):
for parcel in cadastre_parcels:
    if point_inside_parcel_polygon(x, y, parcel):
        parcel_cluster_id = parcel.id
        break
else:
    parcel_cluster_id = 0  # Not in any parcel
```

---

## 🎯 Use Cases & Applications

### 1. Building-Level Analysis

**Extract all points for a specific building:**

```python
import laspy

# Read LAZ file with cluster IDs
las = laspy.read("output.laz")
building_id = 1001

# Filter points by building ID
mask = las.building_cluster_id == building_id
building_points = las.xyz[mask]
building_classification = las.classification[mask]

print(f"Building {building_id}: {len(building_points)} points")
print(f"Classes: {np.unique(building_classification)}")
```

**Calculate building statistics:**

```python
# Group by building ID
unique_buildings = np.unique(las.building_cluster_id)
unique_buildings = unique_buildings[unique_buildings > 0]  # Exclude 0

for bldg_id in unique_buildings:
    mask = las.building_cluster_id == bldg_id
    points = las.xyz[mask]

    # Compute building metrics
    height = points[:, 2].max() - points[:, 2].min()
    volume_estimate = len(points) * 0.01  # Assuming 10cm point spacing

    print(f"Building {bldg_id}: {height:.1f}m tall, ~{volume_estimate:.0f}m³")
```

### 2. Parcel-Level Aggregation

**Aggregate points by cadastral parcel:**

```python
# Group by parcel
unique_parcels = np.unique(las.parcel_cluster_id)
unique_parcels = unique_parcels[unique_parcels > 0]

parcel_stats = []
for parcel_id in unique_parcels:
    mask = las.parcel_cluster_id == parcel_id
    points = las.xyz[mask]
    classes = las.classification[mask]

    # Count class distribution per parcel
    stats = {
        'parcel_id': parcel_id,
        'n_points': len(points),
        'n_buildings': np.sum(classes == 6),  # Building class
        'n_vegetation': np.sum((classes >= 3) & (classes <= 5)),  # Veg classes
        'n_ground': np.sum(classes == 2),  # Ground class
        'mean_height': points[:, 2].mean(),
    }
    parcel_stats.append(stats)

import pandas as pd
df = pd.DataFrame(parcel_stats)
print(df.describe())
```

### 3. Instance Segmentation

**Separate individual building instances:**

```python
from sklearn.cluster import DBSCAN

# Get all building points
building_mask = las.classification == 6
building_points = las.xyz[building_mask]
building_ids = las.building_cluster_id[building_mask]

# Separate by building cluster ID
for bldg_id in np.unique(building_ids):
    if bldg_id == 0:
        continue

    instance_mask = building_ids == bldg_id
    instance_points = building_points[instance_mask]

    # Save individual building
    # ... process or save instance_points
```

### 4. Change Detection

**Track changes to specific buildings over time:**

```python
# Load two time periods
las_t1 = laspy.read("tile_2023.laz")
las_t2 = laspy.read("tile_2025.laz")

# Compare building 1001 between time periods
bldg_id = 1001
mask_t1 = las_t1.building_cluster_id == bldg_id
mask_t2 = las_t2.building_cluster_id == bldg_id

n_points_t1 = np.sum(mask_t1)
n_points_t2 = np.sum(mask_t2)

height_t1 = las_t1.xyz[mask_t1][:, 2].max() - las_t1.xyz[mask_t1][:, 2].min()
height_t2 = las_t2.xyz[mask_t2][:, 2].max() - las_t2.xyz[mask_t2][:, 2].min()

print(f"Building {bldg_id} changes:")
print(f"  Points: {n_points_t1} → {n_points_t2} ({n_points_t2-n_points_t1:+d})")
print(f"  Height: {height_t1:.1f}m → {height_t2:.1f}m ({height_t2-height_t1:+.1f}m)")
```

---

## 🔧 Configuration Options

### Advanced Tuning

If you need to adjust cluster ID computation, here are the relevant parameters:

#### Spatial Clustering (`cluster_id`)

```yaml
reclassification:
  use_clustering: true # Enable spatial clustering
  spatial_cluster_eps: 0.5 # 50cm clustering radius (DBSCAN epsilon)
  min_cluster_size: 10 # Min 10 points per cluster
```

- **`spatial_cluster_eps`**: Distance threshold for grouping points
  - Smaller (0.2-0.4m): Tighter clusters, more instances
  - Larger (0.5-1.0m): Looser clusters, merged instances
- **`min_cluster_size`**: Minimum points to form a cluster
  - Smaller (5-10): More small objects detected
  - Larger (20-50): Only large objects, filter noise

#### Building Cluster ID

```yaml
ground_truth:
  bd_topo:
    features:
      buildings:
        buffer_distance: 0.8 # Tolerance for building boundaries (m)
        extrude_3d: true # Use 3D bounding boxes
        adaptive_buffer_max: 6.0 # Max search distance for facades
```

- **`buffer_distance`**: Tolerance for point-polygon intersection
  - Accounts for building alignment errors in BD TOPO
  - Captures points slightly outside footprint
- **`extrude_3d`**: Enable 3D bounding box checking
  - Checks both horizontal (XY) and vertical (Z) containment
  - More accurate for multi-story buildings

#### Parcel Cluster ID

```yaml
ground_truth:
  bd_topo:
    cadastre:
      enabled: true # Enable parcel integration
      path: "./data/ground_truth/cadastre/"
      parcel_file: "PARCELLE.shp" # Shapefile with parcel polygons
```

- **`enabled`**: Master switch for cadastre integration
- **`path`**: Directory containing cadastral shapefiles
- **`parcel_file`**: Name of parcel polygon shapefile

---

## 📁 Required Data Files

### BD TOPO (Buildings)

**File:** `BATIMENT.shp`  
**Location:** `./data/ground_truth/BDTOPO/`  
**Required fields:**

- Geometry (Polygon)
- `ID` or `OBJECTID` (unique building identifier)
- `hauteur` (building height, optional)

**Download:**

```bash
# From IGN Géoplateforme
# https://geoservices.ign.fr/bdtopo
```

### Cadastre (Parcels)

**File:** `PARCELLE.shp`  
**Location:** `./data/ground_truth/cadastre/`  
**Required fields:**

- Geometry (Polygon)
- `IDU` or `ID` (unique parcel identifier)
- `SECTION`, `NUMERO` (cadastral reference)

**Download:**

```bash
# From data.gouv.fr or cadastre.gouv.fr
# https://cadastre.data.gouv.fr/
```

---

## ⚡ Performance Considerations

### Memory Usage

Cluster IDs add minimal memory overhead:

- **Per point:** +12 bytes (3 × int32)
  - `cluster_id`: 4 bytes
  - `building_cluster_id`: 4 bytes
  - `parcel_cluster_id`: 4 bytes
- **18M point tile:** ~216 MB additional memory
- **LAZ file size:** +2-5 MB (compressed)

### Processing Time

Cluster ID computation adds ~10-30 seconds per tile:

- **Spatial clustering (`cluster_id`):** ~5-10s (DBSCAN)
- **Building assignment:** ~10-15s (spatial index lookups)
- **Parcel assignment:** ~5-10s (spatial index lookups)

**Optimization:**

- Uses STRtree spatial index for fast polygon lookups
- Cached ground truth data reduces repeated loading
- Parallel processing for large datasets

---

## 🐛 Troubleshooting

### Issue: All cluster IDs are 0

**Symptoms:**

- `building_cluster_id` and `parcel_cluster_id` are all 0
- No points assigned to buildings/parcels

**Solutions:**

1. **Check ground truth files exist:**

   ```bash
   ls ./data/ground_truth/BDTOPO/BATIMENT.shp
   ls ./data/ground_truth/cadastre/PARCELLE.shp
   ```

2. **Verify coordinate systems match:**

   - Point cloud: Lambert-93 (EPSG:2154)
   - BD TOPO: Lambert-93 (EPSG:2154)
   - Cadastre: Lambert-93 (EPSG:2154)

3. **Check bounding box overlap:**

   ```python
   import geopandas as gpd

   # Load ground truth
   buildings = gpd.read_file("./data/ground_truth/BDTOPO/BATIMENT.shp")

   # Check extent
   print(f"Buildings extent: {buildings.total_bounds}")
   print(f"Tile extent: {tile_bbox}")
   ```

4. **Increase buffer distance:**
   ```yaml
   buffer_distance: 1.0 # Increase from 0.8m
   ```

### Issue: Too many points assigned to buildings

**Symptoms:**

- Non-building points have building_cluster_id > 0
- Over-classification

**Solutions:**

1. **Enable 3D extrusion:**

   ```yaml
   extrude_3d: true # Check vertical extent
   ```

2. **Reduce buffer distance:**

   ```yaml
   buffer_distance: 0.5 # Reduce from 0.8m
   ```

3. **Use adaptive buffers:**
   ```yaml
   enable_adaptive_buffer: true
   adaptive_buffer_max: 4.0 # Reduce max buffer
   ```

### Issue: Slow processing

**Symptoms:**

- Ground truth assignment takes >1 minute per tile

**Solutions:**

1. **Enable spatial indexing:**

   ```yaml
   use_spatial_index: true # Should be default
   ```

2. **Enable caching:**

   ```yaml
   cache_enabled: true
   cache_dir: "./cache/ground_truth"
   ```

3. **Simplify ground truth polygons:**

   ```python
   import geopandas as gpd

   buildings = gpd.read_file("BATIMENT.shp")
   buildings['geometry'] = buildings.geometry.simplify(0.1)  # Simplify to 10cm
   buildings.to_file("BATIMENT_simplified.shp")
   ```

---

## 📈 Validation

### Check Cluster ID Distribution

```python
import laspy
import numpy as np

las = laspy.read("output.laz")

# Check cluster_id
print("Cluster ID Statistics:")
print(f"  Min: {las.cluster_id.min()}")
print(f"  Max: {las.cluster_id.max()}")
print(f"  Unique clusters: {len(np.unique(las.cluster_id))}")
print(f"  Unassigned (0): {np.sum(las.cluster_id == 0)}")

# Check building_cluster_id
print("\nBuilding Cluster ID Statistics:")
print(f"  Min: {las.building_cluster_id.min()}")
print(f"  Max: {las.building_cluster_id.max()}")
print(f"  Unique buildings: {len(np.unique(las.building_cluster_id[las.building_cluster_id > 0]))}")
print(f"  Assigned to buildings: {np.sum(las.building_cluster_id > 0)} ({100*np.sum(las.building_cluster_id > 0)/len(las):.1f}%)")

# Check parcel_cluster_id
print("\nParcel Cluster ID Statistics:")
print(f"  Min: {las.parcel_cluster_id.min()}")
print(f"  Max: {las.parcel_cluster_id.max()}")
print(f"  Unique parcels: {len(np.unique(las.parcel_cluster_id[las.parcel_cluster_id > 0]))}")
print(f"  Assigned to parcels: {np.sum(las.parcel_cluster_id > 0)} ({100*np.sum(las.parcel_cluster_id > 0)/len(las):.1f}%)")
```

### Visualize Cluster IDs

```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load LAZ
las = laspy.read("output.laz")
points = las.xyz
building_ids = las.building_cluster_id

# Create color map for building IDs
unique_ids = np.unique(building_ids[building_ids > 0])
cmap = plt.get_cmap('tab20')
colors = np.zeros((len(points), 3))

for i, bid in enumerate(unique_ids):
    mask = building_ids == bid
    colors[mask] = cmap(i % 20)[:3]

# Visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
```

---

## 🎓 Summary

Your configuration is **already optimized** for cluster ID features:

✅ **Enabled:**

- `compute_cluster_id` - Spatial clustering
- `compute_building_cluster_id` - BD TOPO buildings
- `compute_parcel_cluster_id` - Cadastral parcels

✅ **Configured:**

- Ground truth assignment from BD TOPO + Cadastre
- 3D building bounding boxes with adaptive buffers
- Output saving as extra dimensions in LAZ

✅ **Optimized:**

- STRtree spatial indexing for fast lookups
- Caching for repeated queries
- Adaptive buffers for better building capture

**Next Steps:**

1. Ensure ground truth files are available:

   - `./data/ground_truth/BDTOPO/BATIMENT.shp`
   - `./data/ground_truth/cadastre/PARCELLE.shp`

2. Run processing:

   ```bash
   ign-lidar-hd process \
     -c examples/production/asprs_complete.yaml \
     input_dir="/data/lidar/tiles" \
     output_dir="/data/output"
   ```

3. Verify cluster IDs in output:
   ```python
   import laspy
   las = laspy.read("output/enriched/*.laz")
   print(f"Cluster IDs available: {las.point_format.extra_dimension_names}")
   ```

---

**Version:** 6.3.1  
**Last Updated:** October 25, 2025  
**Documentation:** See codebase docs for implementation details
