---
sidebar_position: 6
title: Architectural Style API
description: API reference for architectural style detection and classification
keywords: [api, architecture, styles, classification, features]
---

# Architectural Style API Reference

The architectural style API provides functions to detect, classify, and encode architectural styles for both entire tiles and individual point cloud patches.

## Overview

The architectural style system supports:

- **13 distinct architectural styles** (Classical, Gothic, Renaissance, Haussmannian, Modern, etc.)
- **Tile-level style detection** from location metadata
- **Patch-level style inference** from geometry and building features
- **Multiple encoding formats** for ML training (constant, one-hot, multi-hot)
- **Multi-style detection** with confidence weights

## Core Functions

### get_tile_architectural_style

Get architectural style information for an entire tile.

```python
from ign_lidar import get_tile_architectural_style

style_info = get_tile_architectural_style(
    tile_name=None,
    tile_bbox=None,
    location_info=None,
    encoding="info"
)
```

**Parameters:**

- `tile_name` (str, optional): Tile filename (e.g., "HD_LIDARHD_FXX_0650_6860_PTS_C_LAMB93_IGN69")
- `tile_bbox` (tuple, optional): Tile bounding box (xmin, ymin, xmax, ymax)
- `location_info` (dict, optional): Location metadata with keys:
  - `location_name` (str): Name of location
  - `category` (str): Location category (e.g., "urban_dense", "heritage_palace")
  - `characteristics` (List[str]): Architectural characteristics
- `encoding` (str): Return format
  - `"info"`: Full style information dict (default)
  - `"id"`: Just the dominant style ID (int)
  - `"name"`: Just the dominant style name (str)

**Returns:**

Dict with structure:

```python
{
    "dominant_style": {
        "style_id": 5,
        "style_name": "haussmann",
        "weight": 1.0
    },
    "all_styles": [
        {"style_id": 5, "style_name": "haussmann", "weight": 0.7},
        {"style_id": 2, "style_name": "gothic", "weight": 0.3}
    ],
    "location_name": "paris_marais",
    "category": "urban_dense",
    "characteristics": ["architecture_haussmannienne", "hotels_particuliers"],
    "confidence": 0.9  # 0-1 confidence score
}
```

**Examples:**

```python
# From location metadata
location_info = {
    "location_name": "versailles_chateau",
    "category": "heritage_palace",
    "characteristics": ["chateau_royal", "architecture_classique"]
}

# Get full information
style = get_tile_architectural_style(location_info=location_info)
print(style["dominant_style"]["style_name"])  # "classical"
print(style["confidence"])  # 0.9

# Get just the ID
style_id = get_tile_architectural_style(
    location_info=location_info,
    encoding="id"
)
print(style_id)  # 1

# Get just the name
style_name = get_tile_architectural_style(
    location_info=location_info,
    encoding="name"
)
print(style_name)  # "classical"
```

---

### get_patch_architectural_style

Get architectural style information for a point cloud patch.

```python
from ign_lidar import get_patch_architectural_style

patch_style = get_patch_architectural_style(
    points,
    classification=None,
    tile_style_info=None,
    building_features=None,
    encoding="info"
)
```

**Parameters:**

- `points` (np.ndarray): Point cloud array [N, 3] with XYZ coordinates
- `classification` (np.ndarray, optional): Point classification codes [N]
- `tile_style_info` (dict, optional): Style info from parent tile (from `get_tile_architectural_style`)
- `building_features` (dict, optional): Extracted building features:
  - `roof_slope_mean` (float): Average roof slope in degrees
  - `wall_thickness_mean` (float): Average wall thickness in meters
  - `window_to_wall_ratio` (float): Ratio of window to wall area (0-1)
  - `geometric_regularity` (float): Regularity score (0-1)
  - `building_height` (float): Height in meters
  - `footprint_area` (float): Building footprint area in m²
- `encoding` (str): Return format
  - `"info"`: Full style information dict
  - `"id"`: Dominant style ID
  - `"name"`: Dominant style name
  - `"constant"`: Array [N] with style ID for all points
  - `"onehot"`: Array [N, 13] one-hot encoding
  - `"multihot"`: Array [N, 13] multi-hot with weights

**Returns:**

Dict (if encoding="info") or np.ndarray (if encoding is array format)

**Examples:**

```python
import numpy as np

# Create point cloud
points = np.random.rand(10000, 3) * 100
classification = np.full(10000, 6)  # Building points

# 1. Inherit style from parent tile
tile_style = get_tile_architectural_style(
    location_info={"category": "urban_dense",
                   "characteristics": ["architecture_haussmannienne"]}
)

patch_style = get_patch_architectural_style(
    points=points,
    classification=classification,
    tile_style_info=tile_style,
    encoding="constant"
)
# Returns array of shape (10000,) with style ID 5 (Haussmannian)

# 2. Infer from building features
building_features = {
    "roof_slope_mean": 38.0,  # degrees
    "wall_thickness_mean": 0.55,  # meters
    "building_height": 18.5,  # meters
    "geometric_regularity": 0.88,  # high regularity
}

style_info = get_patch_architectural_style(
    points=points,
    building_features=building_features,
    encoding="info"
)
print(style_info["dominant_style"]["style_name"])  # "haussmann"
print(style_info["confidence"])  # 0.6

# 3. One-hot encoding for ML training
onehot_features = get_patch_architectural_style(
    points=points,
    tile_style_info=tile_style,
    encoding="onehot"
)
# Returns array of shape (10000, 13)
```

---

### compute_architectural_style_features

Convenience wrapper for computing architectural style features suitable for ML training.

```python
from ign_lidar import compute_architectural_style_features

features = compute_architectural_style_features(
    points,
    classification=None,
    tile_style_info=None,
    building_features=None,
    encoding="constant"
)
```

**Parameters:** Same as `get_patch_architectural_style`

**Returns:** np.ndarray - Always returns a numpy array (not dict)

**Examples:**

```python
# Constant encoding (single value per point)
style_features = compute_architectural_style_features(
    points=points,
    tile_style_info=tile_style,
    encoding="constant"
)
# shape: (N,), dtype: int32, values: style ID

# One-hot encoding (13-dimensional)
style_features = compute_architectural_style_features(
    points=points,
    tile_style_info=tile_style,
    encoding="onehot"
)
# shape: (N, 13), dtype: float32

# Multi-hot encoding (mixed styles with weights)
style_features = compute_architectural_style_features(
    points=points,
    tile_style_info=mixed_tile_style,  # Has multiple styles
    encoding="multihot"
)
# shape: (N, 13), dtype: float32, multiple non-zero values
```

---

## Helper Functions

### get_architectural_style_id

Get architectural style ID from characteristics or category.

```python
from ign_lidar.features import get_architectural_style_id

style_id = get_architectural_style_id(
    characteristics=["architecture_haussmannienne", "toitures_zinc"],
    category="urban_dense"
)
# Returns: 5 (Haussmannian)
```

---

### get_style_name

Get style name from style ID.

```python
from ign_lidar.features import get_style_name

name = get_style_name(5)
# Returns: "haussmann"
```

---

### infer_multi_styles_from_characteristics

Infer multiple architectural styles with weights from characteristics.

```python
from ign_lidar.features import infer_multi_styles_from_characteristics

styles = infer_multi_styles_from_characteristics(
    characteristics=["architecture_haussmannienne", "architecture_gothique"],
    default_weights={5: 0.7, 2: 0.3}  # Optional custom weights
)

# Returns:
# [
#     {"style_id": 5, "style_name": "haussmann", "weight": 0.7},
#     {"style_id": 2, "style_name": "gothic", "weight": 0.3}
# ]
```

---

## Available Architectural Styles

| Style ID | Style Name  | Description                             |
| -------- | ----------- | --------------------------------------- |
| 0        | unknown     | Unknown or unclassified                 |
| 1        | classical   | Classical/Traditional French            |
| 2        | gothic      | Gothic (medieval churches, cathedrals)  |
| 3        | renaissance | Renaissance (châteaux, palaces)         |
| 4        | baroque     | Baroque ornate style                    |
| 5        | haussmann   | Haussmannian (Paris-style buildings)    |
| 6        | modern      | Modern/Contemporary (20th-21st century) |
| 7        | industrial  | Industrial buildings and warehouses     |
| 8        | vernacular  | Vernacular/Local traditional rural      |
| 9        | art_deco    | Art Deco style (1920s-1940s)            |
| 10       | brutalist   | Brutalist concrete architecture         |
| 11       | glass_steel | Modern glass and steel buildings        |
| 12       | fortress    | Military fortifications and fortresses  |

Access via:

```python
from ign_lidar import ARCHITECTURAL_STYLES, STYLE_NAME_TO_ID

# Dict: ID -> name
print(ARCHITECTURAL_STYLES[5])  # "haussmann"

# Dict: name -> ID
print(STYLE_NAME_TO_ID["haussmann"])  # 5
```

---

## Feature-Based Style Inference Rules

When `building_features` are provided, the system uses these heuristics:

### Traditional Rural (vernacular)

- Roof slope > 45°
- Wall thickness > 0.6m
- Height < 12m

### Haussmannian

- Roof slope: 25-45°
- Height: 15-25m
- Geometric regularity > 0.8

### Modern Glass/Steel

- Window-to-wall ratio > 0.6
- Height > 20m

### Contemporary Modern

- Geometric regularity < 0.5
- Height: 10-30m

### Industrial

- Footprint area > 1000m²
- Height < 15m
- High regularity (> 0.9)

### Gothic

- Roof slope > 55°
- Wall thickness > 0.8m
- Height > 15m

### Renaissance/Classical

- Regularity > 0.85
- Height: 12-20m
- Roof slope: 35-50°

---

## Integration with Processing Pipeline

### In Feature Orchestrator

```python
from ign_lidar.features import FeatureOrchestrator

# Enable architectural style features
config = {
    "processor": {
        "include_architectural_style": True,
        "style_encoding": "constant"  # or "onehot"
    }
}

orchestrator = FeatureOrchestrator(config)

# Compute features (including architectural style)
features = orchestrator.compute_features(tile_data)

# Access style features
style_feature = features.get("architectural_style")
```

### In Dataset Creation

```python
from ign_lidar.datasets import create_training_dataset

# Style features are automatically included if location metadata is available
dataset_config = {
    "features": {
        "include_architectural_style": True,
        "style_encoding": "onehot"
    }
}
```

---

## Performance Considerations

### Memory Usage

- **Constant encoding**: ~4 bytes per point (int32)
- **One-hot encoding**: ~52 bytes per point (13 × float32)
- **Multi-hot encoding**: ~52 bytes per point (13 × float32)

For a 1M point cloud:

- Constant: ~4 MB
- One-hot/Multi-hot: ~52 MB

### Processing Time

- Tile-level style lookup: < 1ms
- Patch-level inference (with features): ~5-10ms
- Patch-level inference (without features): ~2-5ms
- Encoding to arrays: ~10-50ms depending on size

### Recommendations

1. **For ML training**: Use `constant` or `onehot` encoding depending on model architecture
2. **For analysis**: Use `"info"` encoding to get full details with confidence scores
3. **For mixed styles**: Use `multihot` encoding to preserve style distributions
4. **For large datasets**: Prefer `constant` encoding to save memory

---

## Error Handling

```python
# Handle missing location info gracefully
style = get_tile_architectural_style(
    location_info=None,
    encoding="info"
)
# Returns: {"dominant_style": {"style_id": 0, "style_name": "unknown", ...}}

# Invalid encoding raises ValueError
try:
    get_patch_architectural_style(
        points=points,
        encoding="invalid_encoding"
    )
except ValueError as e:
    print(f"Error: {e}")
```

---

## Complete Workflow Example

```python
import numpy as np
from ign_lidar import (
    get_tile_architectural_style,
    get_patch_architectural_style,
    compute_architectural_style_features
)

# 1. Get tile style from location metadata
tile_location = {
    "location_name": "versailles_chateau",
    "category": "heritage_palace",
    "characteristics": ["chateau_royal", "architecture_classique"]
}

tile_style = get_tile_architectural_style(
    location_info=tile_location,
    encoding="info"
)
print(f"Tile style: {tile_style['dominant_style']['style_name']}")
print(f"Confidence: {tile_style['confidence']:.2f}")

# 2. Load point cloud (example)
points = np.random.rand(50000, 3) * 100
classification = np.random.choice([2, 6, 9], size=50000)

# 3. Extract building features (simplified example)
building_mask = classification == 6
building_points = points[building_mask]

building_features = {
    "roof_slope_mean": 40.0,
    "wall_thickness_mean": 0.6,
    "building_height": np.ptp(building_points[:, 2]),
    "geometric_regularity": 0.85,
    "footprint_area": 400.0
}

# 4. Get patch style with feature refinement
patch_style = get_patch_architectural_style(
    points=points,
    classification=classification,
    tile_style_info=tile_style,
    building_features=building_features,
    encoding="info"
)
print(f"Patch style: {patch_style['dominant_style']['style_name']}")
print(f"Confidence: {patch_style['confidence']:.2f}")

# 5. Generate ML features
ml_features = compute_architectural_style_features(
    points=points,
    classification=classification,
    tile_style_info=tile_style,
    building_features=building_features,
    encoding="onehot"
)
print(f"ML feature shape: {ml_features.shape}")  # (50000, 13)
```

---

## See Also

- [Architectural Styles Guide](../features/architectural-styles.md) - Detailed style descriptions
- [Feature Extraction API](./features.md) - Other geometric features
- [Ground Truth Classification](../features/ground-truth-classification.md) - Using architectural features
- [Feature Modes Reference](../features/feature-modes.md) - LOD2/LOD3 feature sets
