# Architectural Style Detection Examples

This directory contains comprehensive examples demonstrating the architectural style detection and classification system for IGN LiDAR HD data.

## Quick Start

Run all examples:

```bash
python examples/example_architectural_styles.py
```

## What's Included

The example script demonstrates 7 different use cases:

### 1. **Tile Style from Location Info**

Get architectural style for an entire tile based on location metadata (category, characteristics).

```python
location_info = {
    "location_name": "versailles_chateau",
    "category": "heritage_palace",
    "characteristics": ["chateau_royal", "architecture_classique"]
}
style = get_tile_architectural_style(location_info=location_info)
# Returns: classical (ID: 1) with 0.90 confidence
```

### 2. **Multiple Architectural Styles**

Detect and weight multiple styles in mixed historical areas.

```python
# Paris Marais - mixed Haussmannian and Gothic
marais_info = {
    "characteristics": ["architecture_haussmannienne", "hotels_particuliers", "architecture_gothique"]
}
style = get_tile_architectural_style(location_info=marais_info)
# Returns: haussmann (50%), gothic (50%)
```

### 3. **Patch Style Inheritance**

Point cloud patches inherit style from parent tiles with optional refinement.

```python
# Get tile style first
tile_style = get_tile_architectural_style(location_info=tile_info)

# Apply to patch
patch_style = get_patch_architectural_style(
    points=points,
    classification=classification,
    tile_style_info=tile_style
)
```

### 4. **Style Inference from Building Features**

Automatically infer architectural style from extracted geometric features.

```python
building_features = {
    "roof_slope_mean": 38.0,  # degrees
    "wall_thickness_mean": 0.55,  # meters
    "building_height": 18.5,  # meters
    "geometric_regularity": 0.88
}

style = get_patch_architectural_style(
    points=points,
    building_features=building_features
)
# Returns: haussmann (inferred from features)
```

### 5. **ML Training Features**

Generate different encoding formats suitable for machine learning:

- **Constant encoding**: Single style ID per point (int32)
- **One-hot encoding**: 13-dimensional binary vector
- **Multi-hot encoding**: 13-dimensional weighted vector

```python
# Constant: array of shape (N,)
constant = compute_architectural_style_features(
    points, tile_style_info=tile_style, encoding="constant"
)

# One-hot: array of shape (N, 13)
onehot = compute_architectural_style_features(
    points, tile_style_info=tile_style, encoding="onehot"
)

# Multi-hot: array of shape (N, 13) with weights
multihot = compute_architectural_style_features(
    points, tile_style_info=mixed_style, encoding="multihot"
)
```

### 6. **Style Inference from Characteristics**

Test various characteristic combinations:

```python
characteristics = ["cathedrale_gothique", "architecture_medievale"]
styles = infer_multi_styles_from_characteristics(characteristics)
# Returns: gothic style with full metadata
```

### 7. **Complete Style Reference**

Display all 13 available architectural styles with descriptions.

## Expected Output

The script produces detailed output for each example:

```
======================================================================
Example 1: Get Tile Architectural Style from Location Info
======================================================================

Location: versailles_chateau
Category: heritage_palace
Characteristics: ['chateau_royal', 'architecture_classique', 'toitures_complexes']

Dominant Style:
  ID: 1
  Name: classical
  Weight: 1.0

Confidence: 0.90

Simple ID retrieval: 1
Simple name retrieval: classical
```

## Available Architectural Styles

| ID  | Name        | Description                             |
| --- | ----------- | --------------------------------------- |
| 0   | unknown     | Unknown or unclassified                 |
| 1   | classical   | Classical/Traditional French            |
| 2   | gothic      | Gothic (medieval churches, cathedrals)  |
| 3   | renaissance | Renaissance (châteaux, palaces)         |
| 4   | baroque     | Baroque ornate style                    |
| 5   | haussmann   | Haussmannian (Paris-style buildings)    |
| 6   | modern      | Modern/Contemporary (20th-21st century) |
| 7   | industrial  | Industrial buildings and warehouses     |
| 8   | vernacular  | Vernacular/Local traditional rural      |
| 9   | art_deco    | Art Deco style (1920s-1940s)            |
| 10  | brutalist   | Brutalist concrete architecture         |
| 11  | glass_steel | Modern glass and steel buildings        |
| 12  | fortress    | Military fortifications and fortresses  |

## Key Functions Demonstrated

### Tile-Level Functions

```python
from ign_lidar import get_tile_architectural_style

# Get full info
style = get_tile_architectural_style(location_info=info, encoding="info")

# Get just ID
style_id = get_tile_architectural_style(location_info=info, encoding="id")

# Get just name
style_name = get_tile_architectural_style(location_info=info, encoding="name")
```

### Patch-Level Functions

```python
from ign_lidar import get_patch_architectural_style

# From tile inheritance
patch_style = get_patch_architectural_style(
    points=points,
    tile_style_info=tile_style,
    encoding="info"
)

# From building features
patch_style = get_patch_architectural_style(
    points=points,
    building_features=features,
    encoding="constant"  # Returns array
)

# From point cloud analysis
patch_style = get_patch_architectural_style(
    points=points,
    classification=classification,
    encoding="onehot"  # Returns one-hot array
)
```

### ML Training Features

```python
from ign_lidar import compute_architectural_style_features

# Always returns numpy array (not dict)
features = compute_architectural_style_features(
    points=points,
    tile_style_info=tile_style,
    encoding="constant"  # or "onehot", "multihot"
)
```

## Style Inference Rules

The system uses these heuristics when building features are provided:

**Traditional Rural (vernacular)**

- Roof slope > 45°, Wall thickness > 0.6m, Height < 12m

**Haussmannian**

- Roof slope: 25-45°, Height: 15-25m, Regularity > 0.8

**Modern Glass/Steel**

- Window ratio > 0.6, Height > 20m

**Contemporary Modern**

- Regularity < 0.5, Height: 10-30m

**Industrial**

- Footprint > 1000m², Height < 15m, High regularity

**Gothic**

- Roof slope > 55°, Wall thickness > 0.8m, Height > 15m

**Renaissance/Classical**

- Regularity > 0.85, Height: 12-20m, Roof slope: 35-50°

## Integration with Processing Pipeline

### In Configuration Files

```yaml
processor:
  include_architectural_style: true
  style_encoding: "constant" # or "onehot", "multihot"
```

### In Python Code

```python
from ign_lidar.features import FeatureOrchestrator

config = {
    "processor": {
        "include_architectural_style": True,
        "style_encoding": "constant"
    }
}

orchestrator = FeatureOrchestrator(config)
features = orchestrator.compute_features(tile_data)

# Access style features
style_feature = features.get("architectural_style")
```

## Performance Notes

**Memory Usage per 1M Points:**

- Constant encoding: ~4 MB
- One-hot encoding: ~52 MB
- Multi-hot encoding: ~52 MB

**Processing Time:**

- Tile-level style lookup: < 1ms
- Patch-level inference (with features): ~5-10ms
- Encoding to arrays: ~10-50ms

## Use Cases

1. **Urban Planning Analysis**: Identify architectural heritage areas
2. **Building Classification**: Enhance LOD3 models with style information
3. **Historical Studies**: Map architectural evolution over time
4. **ML Training**: Add architectural context as features
5. **3D Reconstruction**: Guide reconstruction based on architectural rules
6. **Heritage Conservation**: Prioritize buildings by architectural significance

## Related Documentation

- [Architectural Styles Guide](../docs/docs/features/architectural-styles.md)
- [Architectural Style API Reference](../docs/docs/api/architectural-style-api.md)
- [Feature Extraction](../docs/docs/guides/feature-extraction.md)
- [Dataset Creation](../docs/docs/guides/dataset-creation.md)

## Troubleshooting

**Issue**: Style returns "unknown" (ID: 0)

- **Solution**: Provide location_info with category or characteristics

**Issue**: Low confidence scores

- **Solution**: Add more characteristics or building features

**Issue**: Style doesn't match visual inspection

- **Solution**: Provide building_features to refine detection

**Issue**: Memory issues with onehot encoding

- **Solution**: Use constant encoding instead (13x smaller)

## Next Steps

After running this example, try:

1. Integrating style detection into your processing pipeline
2. Creating a multi-style dataset with architectural annotations
3. Training a model that uses architectural style as a feature
4. Analyzing architectural distributions across regions

## Support

For questions or issues:

- GitHub Issues: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- Documentation: See `docs/` directory
- Examples: See other files in `examples/` directory
