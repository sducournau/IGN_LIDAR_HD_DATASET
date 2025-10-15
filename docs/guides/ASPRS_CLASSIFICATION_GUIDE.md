# ASPRS LAS 1.4 Classification Guide

This guide explains the ASPRS LAS 1.4 classification system implemented in the IGN LiDAR HD Dataset processing library.

## Overview

The library now supports three classification modes:

1. **ASPRS Standard** (`asprs_standard`): Uses only standard ASPRS codes 0-31 for maximum compatibility
2. **ASPRS Extended** (`asprs_extended`): Uses ASPRS codes + extended codes 32-255 for detailed French topographic classification
3. **LOD2/LOD3** (`lod2`, `lod3`): Training-focused building classification schemas

## ASPRS LAS 1.4 Standard Classifications (0-31)

Based on the official ASPRS LAS Specification Version 1.4 - R15.

| Code | Classification            | Description                                          |
| ---- | ------------------------- | ---------------------------------------------------- |
| 0    | Created, Never Classified | Points that have been created but not yet classified |
| 1    | Unclassified              | Points with no classification assigned               |
| 2    | Ground                    | Bare earth and terrain                               |
| 3    | Low Vegetation            | Vegetation < 0.5m                                    |
| 4    | Medium Vegetation         | Vegetation 0.5m - 2.0m                               |
| 5    | High Vegetation           | Vegetation > 2.0m (trees)                            |
| 6    | Building                  | Building structures                                  |
| 7    | Low Point (Noise)         | Noise points below ground                            |
| 9    | Water                     | Water surfaces                                       |
| 10   | Rail                      | Railway tracks and infrastructure                    |
| 11   | Road Surface              | Roads, streets, and paved surfaces                   |
| 13   | Wire - Guard (Shield)     | Shield wires on transmission lines                   |
| 14   | Wire - Conductor (Phase)  | Conductor wires on transmission lines                |
| 15   | Transmission Tower        | Electrical transmission towers                       |
| 16   | Wire-Structure Connector  | Insulators connecting wires to structures            |
| 17   | Bridge Deck               | Bridge surfaces                                      |
| 18   | High Noise                | Noise points above ground                            |
| 19   | Overhead Structure        | Overhead structures (canopies, etc.)                 |
| 20   | Ignored Ground            | Ground points near breaklines (ignored in DTM)       |
| 21   | Snow                      | Snow-covered surfaces                                |
| 22   | Temporal Exclusion        | Points excluded due to temporal criteria             |

## Extended Classifications for French Topography (32-255)

These extended codes are specific to IGN BD TOPO® data and provide detailed classification for French infrastructure and topography.

### Road Types (32-49)

| Code | Classification   | BD TOPO® Nature            |
| ---- | ---------------- | -------------------------- |
| 32   | Motorway         | Autoroute, Quasi-autoroute |
| 33   | Primary Road     | Route à 2 chaussées        |
| 34   | Secondary Road   | Route à 1 chaussée         |
| 35   | Tertiary Road    | Route empierrée            |
| 36   | Residential Road | (Inferred from context)    |
| 37   | Service Road     | Chemin, Bretelle           |
| 38   | Pedestrian Zone  | Place, Sentier, Escalier   |
| 39   | Cycleway         | Piste cyclable             |
| 40   | Parking          | Parking                    |
| 41   | Road Bridge      | Pont routier               |
| 42   | Road Tunnel      | Tunnel routier             |
| 43   | Roundabout       | Rond-point                 |

### Building Types (50-69)

| Code | Classification        | BD TOPO® Nature                         |
| ---- | --------------------- | --------------------------------------- |
| 50   | Residential Building  | Résidentiel, Immeuble, Maison           |
| 51   | Commercial Building   | Commercial et services                  |
| 52   | Industrial Building   | Industriel                              |
| 53   | Religious Building    | Religieux, Chapelle, Église, Cathédrale |
| 54   | Public Building       | Enseignement, Santé, Gare, Transport    |
| 55   | Agricultural Building | Agricole, Serre                         |
| 56   | Sports Building       | Sportif                                 |
| 57   | Historic Building     | Monument, Château, Fort, Remarquable    |
| 58   | Roof                  | Detected roof surfaces                  |
| 59   | Wall                  | Building walls                          |
| 60   | Facade                | Building facades                        |
| 61   | Chimney               | Chimneys                                |
| 62   | Balcony               | Balconies                               |

### Vegetation Types (70-79)

| Code | Classification | BD TOPO® Nature         |
| ---- | -------------- | ----------------------- |
| 70   | Tree           | Arbre, Peupleraie       |
| 71   | Bush           | Lande ligneuse          |
| 72   | Grass          | Low grass and meadows   |
| 73   | Hedge          | Haie                    |
| 74   | Forest         | Bois, Forêt (all types) |
| 75   | Vineyard       | Vigne                   |
| 76   | Orchard        | Verger                  |

### Water Types (80-89)

| Code | Classification | BD TOPO® Nature          |
| ---- | -------------- | ------------------------ |
| 80   | River          | Cours d'eau              |
| 81   | Lake           | Lac, Plan d'eau          |
| 82   | Pond           | Étang, Bassin, Réservoir |
| 83   | Canal          | Canal                    |
| 84   | Fountain       | Fontaine                 |
| 85   | Swimming Pool  | Pool structures          |

### Infrastructure (90-109)

| Code | Classification   | Description              |
| ---- | ---------------- | ------------------------ |
| 90   | Railway Track    | Voie ferrée              |
| 91   | Railway Platform | Quai de gare             |
| 92   | Railway Bridge   | Pont ferroviaire         |
| 93   | Railway Tunnel   | Tunnel ferroviaire       |
| 94   | Power Line       | Ligne électrique         |
| 95   | Power Pylon      | Pylône électrique        |
| 96   | Antenna          | Radio/TV antennas        |
| 97   | Street Light     | Lampadaire               |
| 98   | Traffic Sign     | Panneau de signalisation |
| 99   | Fence            | Clôture                  |
| 100  | Standalone Wall  | Mur indépendant          |

### Urban Furniture (110-119)

| Code | Classification | Description |
| ---- | -------------- | ----------- |
| 110  | Bench          | Banc        |
| 111  | Bin            | Poubelle    |
| 112  | Shelter        | Abri        |
| 113  | Bollard        | Borne       |
| 114  | Barrier        | Barrière    |

### Terrain Types (120-129)

| Code | Classification | Description |
| ---- | -------------- | ----------- |
| 120  | Bare Terrain   | Sol nu      |
| 121  | Gravel         | Gravier     |
| 122  | Sand           | Sable       |
| 123  | Rock           | Roche       |
| 124  | Cliff          | Falaise     |
| 125  | Quarry         | Carrière    |

### Vehicles (130-139)

| Code | Classification | Description |
| ---- | -------------- | ----------- |
| 130  | Car            | Voiture     |
| 131  | Truck          | Camion      |
| 132  | Bus            | Bus         |
| 133  | Train          | Train       |
| 134  | Boat           | Bateau      |
| 135  | Aircraft       | Avion       |

## Configuration

### Enabling ASPRS Classification

In your experiment configuration YAML file:

```yaml
ground_truth:
  enabled: true
  update_classification: true

  # Classification mode
  classification_mode: asprs_extended # or asprs_standard, lod2, lod3

  # Preserve BD TOPO® nature attributes for detailed classification
  preserve_building_nature: true
  preserve_road_nature: true
  preserve_water_nature: true

  # Features to fetch
  fetch_buildings: true
  fetch_roads: true
  fetch_water: true
  fetch_vegetation: true
  fetch_railways: true
```

### Classification Mode Options

- **`asprs_standard`**: Maximum compatibility, uses only codes 0-31

  - Best for: General LiDAR workflows, compatibility with other software
  - Building → 6, Road → 11, Water → 9, etc.

- **`asprs_extended`**: Detailed French topography

  - Best for: Detailed urban analysis, infrastructure mapping
  - Residential building → 50, Motorway → 32, Lake → 81, etc.

- **`lod2`**: Building-focused training with 15 classes

  - Best for: LOD2 building reconstruction training
  - Includes: walls, roof types, context classes

- **`lod3`**: Detailed building training with 30 classes
  - Best for: LOD3 detailed building reconstruction
  - Includes: windows, doors, architectural details

## Usage Examples

### Python API

```python
from ign_lidar import (
    ASPRSClass,
    ClassificationMode,
    get_classification_for_building,
    get_classification_for_road,
    get_class_name,
)

# Get classification for a building
building_class = get_classification_for_building(
    nature="Résidentiel",
    mode=ClassificationMode.ASPRS_EXTENDED
)
print(f"Classification: {building_class} - {get_class_name(building_class)}")
# Output: Classification: 50 - Residential Building

# Get classification for a road
road_class = get_classification_for_road(
    nature="Autoroute",
    mode=ClassificationMode.ASPRS_EXTENDED
)
print(f"Classification: {road_class} - {get_class_name(road_class)}")
# Output: Classification: 32 - Motorway

# Standard mode for compatibility
building_class_std = get_classification_for_building(
    nature="Résidentiel",
    mode=ClassificationMode.ASPRS_STANDARD
)
print(f"Standard classification: {building_class_std}")
# Output: Standard classification: 6
```

### Reading Classified LAZ Files

```python
import laspy
import numpy as np
from ign_lidar import get_class_name

# Read LAZ file
las = laspy.read("enriched_tile.laz")

# Get classification codes
classifications = las.classification

# Get unique classes and their counts
unique_classes, counts = np.unique(classifications, return_counts=True)

print("Classification distribution:")
for cls, count in zip(unique_classes, counts):
    print(f"  {cls:3d} - {get_class_name(cls):30s}: {count:10,} points")
```

## Output

The enriched LAZ files will contain:

1. **Updated Classification Field**: ASPRS LAS 1.4 compliant classification codes
2. **Original LAS Fields**: XYZ, intensity, return number, etc.
3. **RGB/NIR**: Preserved from input or fetched
4. **Extra Dimensions**: Computed features (normals, NDVI, architectural style, etc.)

### Example Output Format

```
Point Format: 7 (RGB) or 8 (RGB + NIR)
Classification: ASPRS LAS 1.4 codes (uint8)
Extra Dimensions:
  - normal_x, normal_y, normal_z (float32)
  - ndvi (float32)
  - architectural_style (uint8)
  - planarity, linearity, etc. (float32)
```

## Best Practices

1. **Use `asprs_standard` for maximum compatibility** with other LiDAR software
2. **Use `asprs_extended` for detailed French infrastructure analysis**
3. **Use `lod2`/`lod3` modes for training building reconstruction models**
4. **Enable nature preservation** (`preserve_*_nature: true`) for detailed classification
5. **Verify classification distribution** after processing to ensure expected results

## References

- ASPRS LAS Specification 1.4 - R15: https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities
- IGN BD TOPO® Documentation: https://geoservices.ign.fr/bdtopo
- IGN LiDAR HD Dataset: https://geoservices.ign.fr/lidarhd

## See Also

- `ign_lidar/asprs_classes.py` - Full classification implementation
- `ign_lidar/classes.py` - LOD2/LOD3 training schemas
- `ign_lidar/io/wfs_ground_truth.py` - Ground truth fetching and labeling
- `configs/experiment/classify_enriched_tiles.yaml` - Example configuration
