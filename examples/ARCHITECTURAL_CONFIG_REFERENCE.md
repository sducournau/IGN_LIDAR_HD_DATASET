# Architectural Style Configuration Reference

This guide explains how to configure architectural style detection and classification in IGN LiDAR HD processing.

## Quick Start

### Enable Architectural Styles in Any Config

Add these lines to your configuration file:

```yaml
processor:
  include_architectural_style: true
  style_encoding: constant # constant, onehot, or multihot

features:
  include_architectural_style: true
  style_encoding: constant
  style_from_building_features: false
```

### Use Pre-configured Templates

**For Training:**

```bash
ign-lidar-hd process experiment=architectural_training \
  input_dir=/path/to/tiles output_dir=/path/to/patches
```

**For Analysis:**

```bash
ign-lidar-hd process experiment=architectural_analysis \
  input_dir=/path/to/tiles output_dir=/path/to/enriched
```

**For Heritage Sites:**

```bash
ign-lidar-hd process experiment=architectural_heritage \
  input_dir=/path/to/heritage output_dir=/path/to/output
```

---

## Configuration Options

### Processor Section

```yaml
processor:
  # Enable architectural style features
  include_architectural_style: true # true or false

  # Style encoding format
  style_encoding: constant # constant, onehot, or multihot

  # Other standard options...
  lod_level: LOD3
  use_gpu: true
```

### Features Section

```yaml
features:
  # Enable architectural style features
  include_architectural_style: true # true or false

  # Style encoding format
  style_encoding: constant # constant, onehot, or multihot

  # Infer/refine style from building geometry
  style_from_building_features: false # true or false

  # Other standard options...
  mode: full
  k_neighbors: 30
```

---

## Encoding Options

### 1. Constant Encoding (Recommended)

**Memory**: ~4 bytes per point  
**Output**: Single integer (0-12) per point  
**Use Case**: Memory-efficient, general purpose

```yaml
style_encoding: constant
```

**Example Output:**

```python
architectural_style: array([5, 5, 5, ..., 5], dtype=int32)  # Haussmannian
```

### 2. One-Hot Encoding

**Memory**: ~52 bytes per point (13 × float32)  
**Output**: 13-dimensional binary vector  
**Use Case**: Neural network training, classification tasks

```yaml
style_encoding: onehot
```

**Example Output:**

```python
architectural_style: array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Haussmannian
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            ...], dtype=float32)
```

### 3. Multi-Hot Encoding

**Memory**: ~52 bytes per point (13 × float32)  
**Output**: 13-dimensional weighted vector  
**Use Case**: Mixed styles, uncertainty modeling

```yaml
style_encoding: multihot
```

**Example Output:**

```python
architectural_style: array([[0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0],  # 70% Haussmannian
                            [0, 0, 0.3, 0, 0, 0.7, 0, 0, 0, 0, 0, 0, 0],  # 30% Gothic
                            ...], dtype=float32)
```

---

## Style Detection Methods

### 1. From Location Metadata (Highest Priority)

Automatically detected if tiles are from known strategic locations:

```python
# Automatic - based on STRATEGIC_LOCATIONS database
# Examples:
#   - versailles_chateau → classical (ID: 1)
#   - paris_haussmann → haussmann (ID: 5)
#   - reims_cathedrale → gothic (ID: 2)
```

**Confidence**: 0.7-0.9

### 2. From Building Features (Medium Priority)

Enable building feature analysis:

```yaml
features:
  style_from_building_features: true
```

Uses geometric features:

- Roof slope (degrees)
- Wall thickness (meters)
- Window-to-wall ratio
- Geometric regularity (0-1)
- Building height (meters)
- Footprint area (m²)

**Confidence**: 0.5-0.7

### 3. From Point Cloud Geometry (Lowest Priority)

Automatic fallback when no metadata available:

- Analyzes height distributions
- Building density patterns
- Basic geometric rules

**Confidence**: 0.3-0.5

---

## Architectural Style Classes

| ID  | Name        | Description                           | Typical Characteristics                 |
| --- | ----------- | ------------------------------------- | --------------------------------------- |
| 0   | unknown     | Unknown or unclassified               | Default when no info available          |
| 1   | classical   | Classical/Traditional French          | Symmetry, proportions                   |
| 2   | gothic      | Gothic medieval                       | Steep roofs (>55°), thick walls, tall   |
| 3   | renaissance | Renaissance châteaux                  | Regular, 12-20m height, 35-50° roof     |
| 4   | baroque     | Baroque ornate                        | Ornamental details                      |
| 5   | haussmann   | Haussmannian Paris-style              | 15-25m height, 25-45° roof, regular     |
| 6   | modern      | Modern/Contemporary 20th-21st century | Irregular geometry, diverse materials   |
| 7   | industrial  | Industrial buildings                  | Large footprint (>1000m²), low, simple  |
| 8   | vernacular  | Traditional rural                     | Steep roofs (>45°), thick walls (>0.6m) |
| 9   | art_deco    | Art Deco 1920s-1940s                  | Stepped facades, vertical emphasis      |
| 10  | brutalist   | Brutalist concrete                    | Raw concrete, geometric                 |
| 11  | glass_steel | Modern glass/steel                    | High window ratio (>60%), tall          |
| 12  | fortress    | Military fortifications               | Defensive features, thick walls         |

---

## Configuration Templates

### For ML Training (Neural Networks)

```yaml
processor:
  lod_level: LOD3
  patch_size: 100.0
  num_points: 32768
  augment: true
  include_architectural_style: true
  style_encoding: onehot # One-hot for neural networks

features:
  mode: full
  include_architectural_style: true
  style_encoding: onehot
  style_from_building_features: true
  normalize_xyz: true
  normalize_features: true

output:
  format: npz,laz
  processing_mode: patches_only
```

### For GIS Analysis

```yaml
processor:
  lod_level: LOD3
  include_architectural_style: true
  style_encoding: constant # Memory-efficient

features:
  mode: full
  include_architectural_style: true
  style_encoding: constant
  style_from_building_features: true

output:
  format: laz
  processing_mode: enriched_only
  save_stats: true # Includes style distribution
```

### For Heritage Documentation

```yaml
processor:
  lod_level: LOD3
  patch_size: 100.0
  include_architectural_style: true
  style_encoding: onehot # Detailed classification

features:
  mode: full
  k_neighbors: 30
  search_radius: 1.5
  include_architectural_style: true
  style_encoding: onehot
  style_from_building_features: true

output:
  format: npz,laz
  save_metadata: true # Detailed style metadata
  save_stats: true
```

---

## Performance Considerations

### Memory Usage (per 1M points)

| Encoding  | Memory | Recommendation             |
| --------- | ------ | -------------------------- |
| Constant  | ~4 MB  | ✅ Default, most efficient |
| One-hot   | ~52 MB | For ML training only       |
| Multi-hot | ~52 MB | For mixed styles only      |

### Processing Time Impact

- Style lookup: < 1ms (negligible)
- Building feature inference: +5-10ms per patch
- Encoding: +10-50ms depending on size

**Total overhead**: < 5% of total processing time

### Recommendations

✅ **Use `constant` encoding** for:

- Large datasets
- GIS visualization
- General analysis
- Memory-constrained systems

✅ **Use `onehot` encoding** for:

- Neural network training
- Deep learning models
- Style classification tasks

✅ **Use `multihot` encoding** for:

- Mixed architectural areas
- Uncertainty modeling
- Research applications

---

## Output Files

### NPZ Files (Training)

```python
import numpy as np

data = np.load('patch_0001.npz')

# Standard features
points = data['points']          # (N, 3)
classification = data['classification']  # (N,)
features = data['features']      # (N, F)

# Architectural style
style = data['architectural_style']
# Constant: (N,) with values 0-12
# One-hot: (N, 13) with binary values
# Multi-hot: (N, 13) with weighted values
```

### LAZ Files (Visualization)

The `architectural_style` is added as an extra dimension:

```python
import laspy

las = laspy.read('tile_enriched.laz')

# Access architectural style
style = las.architectural_style  # Integer values 0-12

# Visualize in QGIS/CloudCompare
# Color by 'architectural_style' dimension
```

### Metadata JSON

```json
{
  "tile_name": "LHD_FXX_0650_6860",
  "location": "versailles_chateau",
  "architectural_style": {
    "dominant_style": {
      "id": 1,
      "name": "classical",
      "weight": 0.85
    },
    "all_styles": [
      { "id": 1, "name": "classical", "weight": 0.85 },
      { "id": 3, "name": "renaissance", "weight": 0.15 }
    ],
    "confidence": 0.9,
    "distribution": {
      "classical": 42500,
      "renaissance": 7500
    }
  }
}
```

---

## Command-Line Overrides

Override style settings from command line:

```bash
# Enable architectural styles
ign-lidar-hd process config.yaml \
  processor.include_architectural_style=true

# Change encoding
ign-lidar-hd process config.yaml \
  processor.style_encoding=onehot \
  features.style_encoding=onehot

# Enable building feature inference
ign-lidar-hd process config.yaml \
  features.style_from_building_features=true

# Combine multiple overrides
ign-lidar-hd process config.yaml \
  processor.include_architectural_style=true \
  processor.style_encoding=constant \
  features.style_from_building_features=true
```

---

## Troubleshooting

### Issue: All styles return "unknown" (ID: 0)

**Cause**: No location metadata available

**Solution**:

1. Use tiles from strategic locations (STRATEGIC_LOCATIONS)
2. Enable building feature inference:
   ```yaml
   features:
     style_from_building_features: true
   ```

### Issue: Low confidence scores

**Cause**: Insufficient information for classification

**Solution**:

- Process known heritage sites with metadata
- Enable `style_from_building_features: true`
- Use tiles with good building coverage

### Issue: High memory usage with one-hot encoding

**Cause**: One-hot encoding is 13x larger than constant

**Solution**: Use constant encoding instead:

```yaml
style_encoding: constant
```

### Issue: Style doesn't match visual inspection

**Cause**: Incorrect inference from features

**Solution**:

- Provide location metadata
- Check building feature extraction quality
- Manually verify STRATEGIC_LOCATIONS mappings

---

## Related Documentation

- [Architectural Style API](../docs/docs/api/architectural-style-api.md)
- [Architectural Styles Guide](../docs/docs/features/architectural-styles.md)
- [Example Script](./example_architectural_styles.py)
- [Implementation Summary](../ARCHITECTURAL_STYLES_IMPLEMENTATION.md)

---

## Examples

See the following config files:

- `config_architectural_training.yaml` - ML training with styles
- `config_architectural_analysis.yaml` - GIS analysis with styles
- `ign_lidar/configs/experiment/architectural_heritage.yaml` - Heritage processing
- `ign_lidar/configs/features/architectural.yaml` - Features config
- `ign_lidar/configs/processor/heritage.yaml` - Heritage processor
