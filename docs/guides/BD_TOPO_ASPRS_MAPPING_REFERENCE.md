# BD TOPO® to ASPRS Mapping - Quick Reference

**Date**: October 16, 2025  
**Version**: 2.0  
**Status**: ✅ Complete

---

## 📊 Complete Mapping Table

### Primary Feature Types

| BD TOPO® Layer           | Feature Type           | ASPRS Code | Code Name  | Priority |
| ------------------------ | ---------------------- | ---------- | ---------- | -------- |
| `batiment`               | Bâtiments              | **6**      | Building   | Highest  |
| `pont`                   | Ponts                  | **17**     | Bridge     | High     |
| `troncon_de_route`       | Routes                 | **11**     | Road       | High     |
| `troncon_de_voie_ferree` | Voies ferrées          | **10**     | Rail       | Medium   |
| `ligne_electrique`       | Lignes électriques     | **43**     | Power Line | Medium   |
| `terrain_de_sport`       | Équipements sportifs   | **41**     | Sports     | Low      |
| `parking`                | Aires de stationnement | **40**     | Parking    | Low      |
| `cimetiere`              | Cimetières             | **42**     | Cemetery   | Low      |
| `surface_hydrographique` | Surfaces d'eau         | **9**      | Water      | Medium   |
| `zone_de_vegetation`     | Végétation             | **3-5**    | Vegetation | Lowest   |

---

## 🚂 Railways (Code 10)

**BD TOPO® Layer**: `troncon_de_voie_ferree`  
**ASPRS Code**: 10 (Rail)  
**All Types Map to Same Code**

### Nature Types

- `Principale` → Main railway line
- `LGV` → High-speed line (TGV)
- `Voie de service` → Service track
- `Voie ferrée` → Generic railway
- `Tramway` → Tram line
- `Métro` → Metro/subway line

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_railway

code = get_classification_for_railway("LGV")  # Returns: 10
code = get_classification_for_railway("Tramway")  # Returns: 10
code = get_classification_for_railway()  # Returns: 10 (default)
```

---

## 🏃 Sports Facilities (Code 41)

**BD TOPO® Layer**: `terrain_de_sport`  
**ASPRS Code**: 41 (Sports Facility)  
**Extended Code** (not in ASPRS standard)

### Nature Types

- `Terrain de football` → Soccer/football field
- `Terrain de rugby` → Rugby field
- `Terrain de tennis` → Tennis court
- `Terrain de basketball` → Basketball court
- `Terrain de handball` → Handball court
- `Terrain de volleyball` → Volleyball court
- `Piste d'athlétisme` → Athletics track
- `Stade` → Stadium
- `Terrain multisports` → Multi-sport field
- `Terrain de golf` → Golf course
- `Piscine` → Swimming pool
- `Skatepark` → Skate park
- `Terrain de pétanque` → Pétanque court
- `Terrain de sport` → Generic sports field

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_sports, ClassificationMode

# Extended mode (default)
code = get_classification_for_sports("Terrain de football")  # Returns: 41

# Standard ASPRS mode (no extended codes)
code = get_classification_for_sports(
    "Terrain de football",
    mode=ClassificationMode.ASPRS_STANDARD
)  # Returns: 1 (Unclassified)
```

---

## ⚰️ Cemeteries (Code 42)

**BD TOPO® Layer**: `cimetiere`  
**ASPRS Code**: 42 (Cemetery)  
**Extended Code** (not in ASPRS standard)

### Nature Types

- `Cimetière` → Cemetery
- `Cimetière militaire` → Military cemetery
- `Cimetière communal` → Municipal cemetery
- `Cimetière paroissial` → Parish cemetery
- `Ossuaire` → Ossuary
- `Columbarium` → Columbarium

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_cemetery, ClassificationMode

# Extended mode (default)
code = get_classification_for_cemetery("Cimetière militaire")  # Returns: 42

# Standard ASPRS mode
code = get_classification_for_cemetery(
    "Cimetière",
    mode=ClassificationMode.ASPRS_STANDARD
)  # Returns: 1 (Unclassified)
```

---

## ⚡ Power Lines (Code 43)

**BD TOPO® Layer**: `ligne_electrique`  
**ASPRS Code**: 43 (Power Line)  
**Extended Code** (not in ASPRS standard)

### Nature Types

- `Ligne électrique` → Power line (generic)
- `Ligne haute tension` → High voltage line
- `Ligne moyenne tension` → Medium voltage line
- `Ligne basse tension` → Low voltage line
- `Aérienne` → Overhead line
- `Souterraine` → Underground line (corridor)

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_power_line, ClassificationMode

# Extended mode (default)
code = get_classification_for_power_line("Ligne haute tension")  # Returns: 43

# Standard ASPRS mode
code = get_classification_for_power_line(
    "Ligne électrique",
    mode=ClassificationMode.ASPRS_STANDARD
)  # Returns: 1 (Unclassified)
```

**Note**: Power lines are buffered by 2.0m by default to create corridors for point classification.

---

## 🅿️ Parking Areas (Code 40)

**BD TOPO® Layer**: `parking`  
**ASPRS Code**: 40 (Parking)  
**Extended Code** (not in ASPRS standard)

### Nature Types

- `Parking` → Parking area (generic)
- `Parking souterrain` → Underground parking
- `Parking aérien` → Surface parking
- `Parking couvert` → Covered parking
- `Aire de stationnement` → Parking area
- `Place de parking` → Parking space
- `Parc de stationnement` → Parking lot

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_parking, ClassificationMode

# Extended mode (default)
code = get_classification_for_parking("Parking souterrain")  # Returns: 40

# Standard ASPRS mode
code = get_classification_for_parking(
    "Parking",
    mode=ClassificationMode.ASPRS_STANDARD
)  # Returns: 1 (Unclassified)
```

---

## 🌉 Bridges (Code 17)

**BD TOPO® Layer**: `pont`  
**ASPRS Code**: 17 (Bridge Deck)  
**Standard ASPRS Code**

### Nature Types

- `Pont` → Bridge
- `Viaduc` → Viaduct
- `Passerelle` → Footbridge
- `Pont-route` → Road bridge
- `Pont ferroviaire` → Railway bridge
- `Aqueduc` → Aqueduct

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_bridge

# Always returns 17 (standard ASPRS code for bridges)
code = get_classification_for_bridge("Viaduc")  # Returns: 17
code = get_classification_for_bridge("Passerelle")  # Returns: 17
code = get_classification_for_bridge()  # Returns: 17 (default)
```

**Note**: Bridge is a standard ASPRS code (17), so it works in both standard and extended modes.

---

## 🚗 Roads (Code 11)

**BD TOPO® Layer**: `troncon_de_route`  
**ASPRS Code**: 11 (Road Surface) - standard mode  
**ASPRS Codes**: 32-43 (Road types) - extended mode

### Nature Types (Extended Mode)

- `Autoroute` → 32 (Motorway)
- `Quasi-autoroute` → 32 (Motorway)
- `Route à 2 chaussées` → 33 (Primary Road)
- `Route à 1 chaussée` → 34 (Secondary Road)
- `Route empierrée` → 35 (Tertiary Road)
- `Chemin` → 37 (Service Road)
- `Bretelle` → 37 (Service Road)
- `Rond-point` → 43 (Roundabout)
- `Place` → 38 (Pedestrian Zone)
- `Sentier` → 38 (Pedestrian)
- `Escalier` → 38 (Pedestrian)
- `Piste cyclable` → 39 (Cycleway)
- `Parking` → 40 (Parking)

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_road, ClassificationMode

# Extended mode - detailed road types
code = get_classification_for_road("Autoroute", ClassificationMode.ASPRS_EXTENDED)  # Returns: 32

# Standard mode - all roads use code 11
code = get_classification_for_road("Autoroute", ClassificationMode.ASPRS_STANDARD)  # Returns: 11
```

---

## 💧 Water (Code 9)

**BD TOPO® Layer**: `surface_hydrographique`  
**ASPRS Code**: 9 (Water) - standard mode  
**ASPRS Codes**: 80-85 (Water types) - extended mode

### Nature Types (Extended Mode)

- `Cours d'eau` → 80 (River)
- `Plan d'eau` → 81 (Lake)
- `Étang` → 82 (Pond)
- `Lac` → 81 (Lake)
- `Canal` → 83 (Canal)
- `Bassin` → 82 (Pond)
- `Réservoir` → 82 (Pond)

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_water, ClassificationMode

# Extended mode - detailed water types
code = get_classification_for_water("Lac", ClassificationMode.ASPRS_EXTENDED)  # Returns: 81

# Standard mode - all water uses code 9
code = get_classification_for_water("Lac", ClassificationMode.ASPRS_STANDARD)  # Returns: 9
```

---

## 🏢 Buildings (Code 6)

**BD TOPO® Layer**: `batiment`  
**ASPRS Code**: 6 (Building) - standard mode  
**ASPRS Codes**: 50-57 (Building types) - extended mode

### Nature Types (Extended Mode)

- `Indifférencié` → 6 (Generic Building)
- `Résidentiel` → 50 (Residential)
- `Immeuble` → 50 (Residential)
- `Maison` → 50 (Residential)
- `Commercial` → 51 (Commercial)
- `Commercial et services` → 51 (Commercial)
- `Industriel` → 52 (Industrial)
- `Serre` → 55 (Agricultural)
- `Religieux` → 53 (Religious)
- `Sportif` → 56 (Sports)
- `Agricole` → 55 (Agricultural)
- `Monument` → 57 (Historic)
- `Château` → 57 (Historic)

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_building, ClassificationMode

# Extended mode - detailed building types
code = get_classification_for_building("Château", ClassificationMode.ASPRS_EXTENDED)  # Returns: 57

# Standard mode - all buildings use code 6
code = get_classification_for_building("Château", ClassificationMode.ASPRS_STANDARD)  # Returns: 6
```

---

## 🌳 Vegetation (Codes 3-5)

**BD TOPO® Layer**: `zone_de_vegetation`  
**ASPRS Codes**: 3-5 (by height) - standard mode  
**ASPRS Codes**: 70-76 (Vegetation types) - extended mode

### Height-Based Classification (Standard Mode)

- Height < 0.5m → 3 (Low Vegetation)
- 0.5m ≤ Height < 2.0m → 4 (Medium Vegetation)
- Height ≥ 2.0m → 5 (High Vegetation)

### Nature Types (Extended Mode)

- `Arbre` → 70 (Tree)
- `Haie` → 73 (Hedge)
- `Bois` → 74 (Forest)
- `Forêt fermée de feuillus` → 74 (Forest)
- `Forêt fermée de conifères` → 74 (Forest)
- `Forêt fermée mixte` → 74 (Forest)
- `Forêt ouverte` → 74 (Forest)
- `Lande ligneuse` → 71 (Bush)
- `Verger` → 76 (Orchard)
- `Vigne` → 75 (Vineyard)
- `Peupleraie` → 70 (Tree)

### Usage

```python
from ign_lidar.asprs_classes import get_classification_for_vegetation, ClassificationMode

# Extended mode with nature attribute
code = get_classification_for_vegetation("Vigne", height=2.5, mode=ClassificationMode.ASPRS_EXTENDED)  # Returns: 75

# Standard mode with height
code = get_classification_for_vegetation(height=2.5, mode=ClassificationMode.ASPRS_STANDARD)  # Returns: 5

# Fallback when both nature and height are missing
code = get_classification_for_vegetation()  # Returns: 4 (Medium Vegetation)
```

---

## 🔄 Processing Order & Priority

Features are applied in priority order (lowest to highest):

1. **Vegetation** (3-5) - Lowest priority
2. **Water** (9)
3. **Cemeteries** (42)
4. **Parking** (40)
5. **Sports** (41)
6. **Power Lines** (43)
7. **Railways** (10)
8. **Roads** (11)
9. **Bridges** (17)
10. **Buildings** (6) - Highest priority

**Note**: Higher priority features overwrite lower priority ones when geometries overlap.

---

## 📦 Python API

### Import All Mappings

```python
from ign_lidar.asprs_classes import (
    # Classification functions
    get_classification_for_railway,
    get_classification_for_sports,
    get_classification_for_cemetery,
    get_classification_for_power_line,
    get_classification_for_parking,
    get_classification_for_bridge,
    get_classification_for_building,
    get_classification_for_road,
    get_classification_for_water,
    get_classification_for_vegetation,

    # Mapping dictionaries
    RAILWAY_NATURE_TO_ASPRS,
    SPORTS_NATURE_TO_ASPRS,
    CEMETERY_NATURE_TO_ASPRS,
    POWER_LINE_NATURE_TO_ASPRS,
    PARKING_NATURE_TO_ASPRS,
    BRIDGE_NATURE_TO_ASPRS,
    BUILDING_NATURE_TO_ASPRS,
    ROAD_NATURE_TO_ASPRS,
    WATER_NATURE_TO_ASPRS,
    VEGETATION_NATURE_TO_ASPRS,

    # Utilities
    ClassificationMode,
    get_class_name,
    get_class_color,
)
```

### Direct Dictionary Access

```python
# Get ASPRS code directly from mapping
code = RAILWAY_NATURE_TO_ASPRS.get("LGV", 10)  # Returns: 10
code = SPORTS_NATURE_TO_ASPRS.get("Stade", 41)  # Returns: 41
code = CEMETERY_NATURE_TO_ASPRS.get("Cimetière militaire", 42)  # Returns: 42
```

---

## ⚙️ Configuration Examples

### Enable All BD TOPO® Features

```yaml
data_sources:
  bd_topo:
    enabled: true
    cache_enabled: true
    features:
      buildings: true # Code 6
      roads: true # Code 11
      railways: true # Code 10
      water: true # Code 9
      vegetation: true # Codes 3-5
      bridges: true # Code 17
      parking: true # Code 40
      sports: true # Code 41
      cemeteries: true # Code 42
      power_lines: true # Code 43
```

### Enable Only Transport Features

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      roads: true # Code 11
      railways: true # Code 10
      bridges: true # Code 17

      # Disable everything else
      buildings: false
      water: false
      vegetation: false
      parking: false
      sports: false
      cemeteries: false
      power_lines: false
```

### Enable Only Extended Features

```yaml
data_sources:
  bd_topo:
    enabled: true
    features:
      # Standard features (disabled)
      buildings: false
      roads: false
      water: false
      vegetation: false

      # Extended features (enabled)
      railways: true # Code 10
      bridges: true # Code 17
      parking: true # Code 40
      sports: true # Code 41
      cemeteries: true # Code 42
      power_lines: true # Code 43
```

---

## 🎨 Visualization Colors

Default RGB colors for each ASPRS code:

| Code | Feature    | Color        | RGB           |
| ---- | ---------- | ------------ | ------------- |
| 6    | Building   | Red          | (255, 0, 0)   |
| 9    | Water      | Blue         | (0, 0, 255)   |
| 10   | Rail       | Purple       | (128, 0, 128) |
| 11   | Road       | Black        | (0, 0, 0)     |
| 17   | Bridge     | Saddle Brown | (139, 69, 19) |
| 40   | Parking    | Dark Gray    | (64, 64, 64)  |
| 41   | Sports     | Dark Gray    | (64, 64, 64)  |
| 42   | Cemetery   | Dark Gray    | (64, 64, 64)  |
| 43   | Power Line | Dark Gray    | (64, 64, 64)  |

```python
from ign_lidar.asprs_classes import get_class_color

color = get_class_color(10)  # Returns: (128, 0, 128) - Purple for railways
color = get_class_color(41)  # Returns: (64, 64, 64) - Dark gray for sports
```

---

## 📊 Statistics Example

```python
import laspy
import numpy as np
from collections import Counter
from ign_lidar.asprs_classes import get_class_name

# Read enriched LAZ file
las = laspy.read("enriched_tile.laz")

# Count points per classification
unique, counts = np.unique(las.classification, return_counts=True)

print("Classification Distribution:")
print(f"{'Code':<6} {'Name':<20} {'Points':>12} {'Percent':>8}")
print("-" * 50)

total_points = len(las.classification)
for code, count in zip(unique, counts):
    name = get_class_name(code)
    percent = 100.0 * count / total_points
    print(f"{code:<6} {name:<20} {count:>12,} {percent:>7.2f}%")
```

Example output:

```
Classification Distribution:
Code   Name                      Points  Percent
--------------------------------------------------
1      Unclassified             145,800     5.91%
2      Ground                   124,300     5.05%
3      Low Vegetation           234,100     9.51%
5      High Vegetation          567,200    23.04%
6      Building                 189,400     7.69%
9      Water                     23,100     0.94%
10     Rail                      12,400     0.50%
11     Road Surface              89,300     3.63%
17     Bridge Deck                5,600     0.23%
40     Parking                    8,200     0.33%
41     Sports Facility            4,300     0.17%
42     Cemetery                   2,100     0.09%
43     Power Line                 3,800     0.15%
```

---

## 🔗 Related Documentation

- [Complete Implementation Report](../updates/BD_TOPO_ASPRS_MAPPING_UPDATE.md)
- [BD TOPO RPG Integration](../reports/BD_TOPO_RPG_INTEGRATION.md)
- [ASPRS Implementation Summary](ASPRS_IMPLEMENTATION_SUMMARY.md)
- [Advanced Classification Guide](../ADVANCED_CLASSIFICATION_GUIDE.md)

---

**Last Updated**: October 16, 2025
