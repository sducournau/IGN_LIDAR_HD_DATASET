# Cadastral Parcel Clustering - Configuration Guide

## Overview

The `config_asprs_cadastre_foret.yaml` configuration includes **cadastral parcel clustering**, which assigns a unique numerical cluster ID to each cadastral parcel. This allows you to easily group and analyze LiDAR points by cadastral parcel.

## 📊 What Gets Added to Your LAZ Files

Each point in the enriched LAZ file will have **two cadastre-related fields**:

1. **`parcel_id`** (String): The official 19-character cadastral parcel identifier

   - Example: `"000AB0123000AA0001"`
   - Unique for each parcel in France
   - Links to official cadastral database

2. **`parcel_cluster`** (uint32): A numerical cluster ID (1-based)
   - Example: `1`, `2`, `3`, `4`, ...
   - Sequential numbering for each unique parcel in the tile
   - Easy to use for segmentation and visualization
   - Allows efficient grouping of points

## 🎯 Use Cases

### 1. **Visualization in CloudCompare**

```
- Open enriched LAZ file
- Display → Scalar Fields → parcel_cluster
- Edit → Colors → Color Scale → Random colors
- Result: Each parcel displayed in a different color
```

### 2. **Segmentation by Parcel**

```
- Filter points by parcel_cluster value
- Extract all points from parcel #5: parcel_cluster == 5
- Compare multiple parcels: parcel_cluster IN [1, 3, 7, 12]
```

### 3. **Analysis per Parcel**

- Calculate volume per parcel
- Compute average height per parcel
- Analyze ASPRS class distribution per parcel
- Count points per parcel (point density)

### 4. **Agricultural/Forest Analysis**

When combined with RPG and BD Forêt:

- Parcel #1: Wheat field (crop_code: "BLE")
- Parcel #2: Pine forest (primary_species: "Pinus")
- Parcel #3: Building (ASPRS class 6)

## 📁 Output Files

### LAZ Files

Located in: `/mnt/d/ign/preprocessed/asprs_cadastre_foret/enriched_tiles/`

Each LAZ contains:

- Standard dimensions (X, Y, Z, RGB, NIR, Classification, etc.)
- **`parcel_id`**: String identifier
- **`parcel_cluster`**: Numerical cluster ID
- `forest_type`, `primary_species`, `estimated_height` (from BD Forêt)
- `crop_code`, `crop_category` (from RPG)

### Parcel Statistics (GeoJSON/CSV)

Located in: `/mnt/d/ign/parcel_stats/asprs_cadastre_foret/`

Contains per-parcel statistics:

- **Cluster Mapping**: `parcel_id` → `parcel_cluster` lookup table
- **ASPRS Classes**: Distribution of point classes per parcel
- **Point Density**: Number of points per parcel
- **Crop Types**: Agricultural crop information (if RPG enabled)
- **Forest Types**: Forest species and types (if BD Forêt enabled)

Example GeoJSON structure:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "parcel_id": "000AB0123000AA0001",
        "parcel_cluster": 1,
        "point_count": 15423,
        "point_density": 12.3,
        "asprs_classes": {
          "2": 8234,  // Ground
          "5": 7189   // High vegetation
        },
        "crop_code": "BLE",
        "crop_category": "cereals",
        "forest_type": null
      },
      "geometry": { ... }
    }
  ]
}
```

## ⚙️ Configuration Options

### Enable Clustering

```yaml
cadastre:
  enabled: true
  create_cluster_ids: true # ✅ Enable numerical cluster IDs
  cluster_id_start: 1 # Start numbering at 1
  cluster_id_field: "parcel_cluster" # Field name in LAZ
  assign_cluster_to_points: true # Assign to all points in parcel
```

### Export Options

```yaml
export_parcel_stats:
  enabled: true
  format: "geojson" # or "csv" or "both"
  include_cluster_mapping: true # Export parcel_id → cluster_id table
```

## 🚀 Running the Pipeline

```bash
# Process tiles with cadastral clustering
ign-lidar-hd process --config configs/multiscale/config_asprs_cadastre_foret.yaml

# Check output
ls /mnt/d/ign/preprocessed/asprs_cadastre_foret/enriched_tiles/*.laz

# Check parcel statistics
ls /mnt/d/ign/parcel_stats/asprs_cadastre_foret/*.geojson
```

## 📈 Performance Notes

- **Clustering overhead**: Minimal (~2-3% processing time increase)
- **Storage overhead**: +4 bytes per point for `parcel_cluster` (uint32)
- **Memory usage**: Efficient - clusters computed on-the-fly during enrichment
- **Cache**: Cadastral data is cached for reuse across tiles

## 🔍 Querying Parcels

### Python Example

```python
import laspy

# Load enriched LAZ
las = laspy.read("enriched_tile.laz")

# Get parcel cluster IDs
parcel_clusters = las.parcel_cluster

# Get all points from parcel #5
parcel_5_points = las.points[parcel_clusters == 5]

# Count points per parcel
import numpy as np
unique_parcels, counts = np.unique(parcel_clusters, return_counts=True)
for parcel_id, count in zip(unique_parcels, counts):
    print(f"Parcel {parcel_id}: {count} points")
```

### PDAL Example

```json
{
  "pipeline": [
    "input.laz",
    {
      "type": "filters.expression",
      "expression": "parcel_cluster == 5"
    },
    "parcel_5_only.laz"
  ]
}
```

## 🗺️ Integration with Other Data Sources

The parcel clustering works seamlessly with:

- **BD Forêt**: Forest type per parcel
- **RPG**: Crop type per parcel
- **BD TOPO**: Infrastructure per parcel
- **ASPRS Classes**: Point classification per parcel

Example combined query:

```python
# Find all parcels with wheat crops and high vegetation
wheat_parcels = las.points[
    (las.crop_code == "BLE") &
    (las.Classification == 5)  # High vegetation
]
```

## 📚 Related Documentation

- Main config: `configs/multiscale/config_asprs_cadastre_foret.yaml`
- Processing guide: `docs/MULTISCALE_QUICKSTART.md`
- Data sources: `docs/DATA_SOURCES.md`
- ASPRS classification: `docs/CLASSIFICATION_QUICK_REFERENCE.md`

## 🤝 Support

For questions about cadastral clustering:

1. Check the configuration file comments
2. Review parcel statistics output files
3. Inspect LAZ files in CloudCompare
4. See example scripts in `examples/`
