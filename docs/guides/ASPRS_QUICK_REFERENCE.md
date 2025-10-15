# ASPRS LAS 1.4 Quick Reference

## Standard Classifications (0-31)

| Code | Name              | Code | Name               |
| ---- | ----------------- | ---- | ------------------ |
| 0    | Never Classified  | 11   | Road Surface       |
| 1    | Unclassified      | 13   | Wire - Guard       |
| 2    | **Ground**        | 14   | Wire - Conductor   |
| 3    | Low Vegetation    | 15   | Transmission Tower |
| 4    | Medium Vegetation | 16   | Wire Connector     |
| 5    | High Vegetation   | 17   | Bridge Deck        |
| 6    | **Building**      | 18   | High Noise         |
| 7    | Low Point (Noise) | 19   | Overhead Structure |
| 9    | **Water**         | 20   | Ignored Ground     |
| 10   | Rail              | 21   | Snow               |

## Extended Classifications for French Topography

### Roads (32-43)

- 32: Motorway | 33: Primary Road | 34: Secondary Road | 35: Tertiary Road
- 36: Residential | 37: Service Road | 38: Pedestrian | 39: Cycleway
- 40: Parking | 41: Road Bridge | 42: Road Tunnel | 43: Roundabout

### Buildings (50-62)

- 50: Residential | 51: Commercial | 52: Industrial | 53: Religious
- 54: Public | 55: Agricultural | 56: Sports | 57: Historic
- 58: Roof | 59: Wall | 60: Facade | 61: Chimney | 62: Balcony

### Vegetation (70-76)

- 70: Tree | 71: Bush | 72: Grass | 73: Hedge
- 74: Forest | 75: Vineyard | 76: Orchard

### Water (80-85)

- 80: River | 81: Lake | 82: Pond | 83: Canal
- 84: Fountain | 85: Swimming Pool

### Infrastructure (90-100)

- 90: Railway Track | 91: Railway Platform | 92: Railway Bridge
- 94: Power Line | 95: Power Pylon | 97: Street Light
- 98: Traffic Sign | 99: Fence | 100: Wall

### Urban (110-114)

- 110: Bench | 111: Bin | 112: Shelter | 113: Bollard | 114: Barrier

### Terrain (120-125)

- 120: Bare | 121: Gravel | 122: Sand | 123: Rock | 124: Cliff | 125: Quarry

### Vehicles (130-135)

- 130: Car | 131: Truck | 132: Bus | 133: Train | 134: Boat | 135: Aircraft

## Configuration Modes

```yaml
ground_truth:
  classification_mode:
    asprs_extended # Options:
    # asprs_standard - Codes 0-31 only (max compatibility)
    # asprs_extended - Codes 0-255 (detailed French classification)
    # lod2 - 15 building classes for training
    # lod3 - 30 building classes for training

  preserve_building_nature: true # Use BD TOPO® building types
  preserve_road_nature: true # Use BD TOPO® road types
  preserve_water_nature: true # Use BD TOPO® water types
```

## Python Usage

```python
from ign_lidar import ASPRSClass, get_classification_for_building

# Get classification code
code = get_classification_for_building("Résidentiel", "asprs_extended")
print(code)  # 50

# Use enum
print(ASPRSClass.BUILDING_RESIDENTIAL)  # 50
print(ASPRSClass.ROAD_MOTORWAY)  # 32
```

## BD TOPO® Nature → ASPRS Mapping

### Buildings

- "Résidentiel" → 50 (Residential)
- "Commercial" → 51 (Commercial)
- "Industriel" → 52 (Industrial)
- "Religieux" → 53 (Religious)
- "Sportif" → 56 (Sports)

### Roads

- "Autoroute" → 32 (Motorway)
- "Route à 2 chaussées" → 33 (Primary)
- "Route à 1 chaussée" → 34 (Secondary)
- "Piste cyclable" → 39 (Cycleway)
- "Parking" → 40 (Parking)

### Vegetation

- "Forêt" → 74 (Forest)
- "Vigne" → 75 (Vineyard)
- "Verger" → 76 (Orchard)
- "Haie" → 73 (Hedge)

### Water

- "Cours d'eau" → 80 (River)
- "Lac" → 81 (Lake)
- "Canal" → 83 (Canal)
- "Étang" → 82 (Pond)
