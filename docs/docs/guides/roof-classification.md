# Roof Type Classification (LOD3)

**Version:** 3.1.0  
**Status:** ✅ Stable

## Overview

The Roof Type Classification system automatically detects and classifies building roofs into geometric types (flat, gabled, hipped, complex) and identifies architectural details (ridges, edges, dormers) for LOD3 building models.

## Features

### Roof Types

- **Flat Roofs** - Horizontal surfaces with low slope angles (< 15°)
- **Gabled Roofs** - Two-plane pitched roofs (classic "A" shape)
- **Hipped Roofs** - Three or four-plane roofs with slopes on all sides
- **Complex Roofs** - Multi-plane roofs with 5+ segments

### Architectural Details

- **Ridge Lines** - Peak lines where roof planes meet (high curvature points)
- **Roof Edges** - Perimeter boundary of the roof (eaves, gables)
- **Dormers** - Vertical protrusions with windows in sloped roofs

## Quick Start

### 1. Configuration

Create a configuration file with roof classification enabled:

```yaml
# config.yaml
processor:
  lod_level: LOD3 # Required for roof features

features:
  mode: lod3 # Includes normals, verticality, curvature

classification:
  building_facade:
    enable_roof_classification: true
    roof_flat_threshold: 15.0
    roof_pitched_threshold: 20.0
```

### 2. Run Processing

```bash
ign-lidar process --config config.yaml
```

### 3. Inspect Results

Output LAZ files will contain these new classification codes:

| Code | Class Name            | Description               |
| ---- | --------------------- | ------------------------- |
| 63   | BUILDING_ROOF_FLAT    | Flat roof surfaces        |
| 64   | BUILDING_ROOF_GABLED  | Gabled/pitched roofs      |
| 65   | BUILDING_ROOF_HIPPED  | Hipped roofs              |
| 66   | BUILDING_ROOF_COMPLEX | Complex multi-plane roofs |
| 67   | BUILDING_ROOF_RIDGE   | Ridge lines               |
| 68   | BUILDING_ROOF_EDGE    | Roof edges/eaves          |
| 69   | BUILDING_DORMER       | Dormer windows            |

## Configuration Parameters

### Essential Parameters

```yaml
classification:
  building_facade:
    # Enable/disable roof classification
    enable_roof_classification: true

    # Slope thresholds (degrees)
    roof_flat_threshold: 15.0 # Max slope for flat roofs
    roof_pitched_threshold: 20.0 # Min slope for pitched roofs
```

### Advanced Parameters (Future)

These parameters are currently hardcoded but will be exposed in future versions:

- `roof_dbscan_eps`: DBSCAN epsilon for plane clustering (default: 0.15)
- `roof_min_plane_points`: Minimum points per roof plane (default: 50)
- `roof_ridge_curvature`: Minimum curvature for ridges (default: 0.5)
- `roof_dormer_verticality`: Minimum verticality for dormers (default: 0.5)

## Algorithm Overview

### 1. Roof Point Identification

Identifies roof points using verticality (measure of how horizontal a surface is):

```python
# Points with low verticality are ~horizontal (potential roof)
roof_mask = verticality < 0.3
```

### 2. Plane Segmentation

Clusters roof points into planar segments using DBSCAN on normal vectors:

```python
# Group points with similar normals
clustering = DBSCAN(eps=0.15, min_samples=50)
labels = clustering.fit_predict(normals)
```

### 3. Type Classification

Classifies roof type based on number and orientation of segments:

- **1 segment** → FLAT (if slope < 15°)
- **2 segments** → GABLED
- **3-4 segments** → HIPPED
- **5+ segments** → COMPLEX

### 4. Detail Detection

- **Ridges**: High curvature points at plane intersections
- **Edges**: Convex hull perimeter of roof footprint
- **Dormers**: Small vertical clusters within roof area

## Usage Examples

### Example 1: Basic Roof Detection

```yaml
# Minimal configuration for roof classification
processor:
  lod_level: LOD3

features:
  mode: lod3

classification:
  building_facade:
    enable_roof_classification: true
```

### Example 2: Adjust Sensitivity

```yaml
# Classify more roofs as "flat" (increase threshold)
classification:
  building_facade:
    enable_roof_classification: true
    roof_flat_threshold: 20.0 # Default: 15.0
```

### Example 3: Production Configuration

See the complete example:  
`examples/production/asprs_roof_detection.yaml`

## Performance

### Typical Processing Times

For a building with 50,000 points:

- Roof identification: ~50ms
- Plane segmentation: ~100ms
- Type classification: ~10ms
- Detail detection: ~120ms
- **Total overhead: ~280ms (~10-15% of building classification)**

### Optimization Tips

1. **Use LOD3 mode** - Includes all required features
2. **Enable GPU** - Accelerates feature computation
3. **Cache features** - Reuse computed normals/verticality
4. **Adjust thresholds** - Fine-tune for your dataset

## Output Statistics

The processor logs detailed statistics about roof classification:

```
INFO: Building classification completed
INFO:   ✓ Roofs classified: 45
INFO:     - flat: 18
INFO:     - gabled: 15
INFO:     - hipped: 8
INFO:     - complex: 4
```

## Troubleshooting

### Problem: No roofs classified

**Symptoms:**

- `Roofs classified: 0` in logs
- All roofs still have class 58 (BUILDING_ROOF)

**Solutions:**

1. Check if roof classification is enabled:

   ```yaml
   classification:
     building_facade:
       enable_roof_classification: true
   ```

2. Verify LOD3 mode (includes required features):

   ```yaml
   features:
     mode: lod3
   ```

3. Check logs for errors:
   ```bash
   grep "roof" logs/processing.log
   ```

### Problem: Too many "complex" roofs

**Symptoms:**

- Most roofs classified as class 66 (COMPLEX)
- Expected more flat or gabled roofs

**Solutions:**

Increase the flat roof threshold to classify more roofs as flat:

```yaml
classification:
  building_facade:
    roof_flat_threshold: 20.0 # Default: 15.0
```

### Problem: Missing ridge lines

**Symptoms:**

- No points with class 67 (BUILDING_ROOF_RIDGE)
- Ridge lines not detected on pitched roofs

**Solutions:**

1. Ensure curvature features are computed:

   ```yaml
   features:
     compute_curvature: true
   ```

2. Check if LOD3 mode is enabled (includes curvature):
   ```yaml
   features:
     mode: lod3
   ```

### Problem: False dormer detections

**Symptoms:**

- Too many points classified as class 69 (DORMER)
- Noise mistaken for dormers

**Solutions:**

This requires code adjustment (future config parameter):

```python
# In roof_classifier.py, increase verticality threshold
dormer_min_verticality = 0.6  # Default: 0.5
```

## API Reference

### RoofTypeClassifier

```python
from ign_lidar.core.classification.building import RoofTypeClassifier

classifier = RoofTypeClassifier(
    flat_threshold=15.0,
    pitched_threshold=20.0,
    dbscan_eps=0.15,
    min_plane_points=50,
    ridge_curvature=0.5,
    dormer_min_verticality=0.5
)

result = classifier.classify_roof(
    points=building_points,     # [N, 3] array
    features={                  # Feature dictionary
        'normals': normals,
        'verticality': verticality,
        'curvature': curvature,
        'planarity': planarity
    },
    labels=current_labels       # Current classifications
)

# Access results
print(f"Type: {result.roof_type}")           # RoofType enum
print(f"Confidence: {result.confidence}")    # 0-1 score
print(f"Segments: {len(result.segments)}")   # Number of planes
```

### RoofClassificationResult

```python
@dataclass
class RoofClassificationResult:
    roof_type: RoofType              # FLAT, GABLED, HIPPED, COMPLEX
    confidence: float                # Classification confidence (0-1)
    segments: List[RoofSegment]      # Detected roof planes
    ridge_lines: np.ndarray          # Indices of ridge points
    edge_points: np.ndarray          # Indices of edge points
    dormer_points: np.ndarray        # Indices of dormer points
    num_roof_points: int             # Total roof points
    processing_time: float           # Time taken (seconds)
```

## Validation

### Test Suite

The roof classifier includes comprehensive tests:

```bash
# Run roof classification tests
pytest tests/test_roof_classifier.py -v

# Run integration tests
pytest tests/test_building_classification.py -v -k roof
```

**Test Coverage:** 20+ tests covering all classification types and edge cases

### Visual Inspection

Use CloudCompare or similar tools to visually inspect classifications:

1. Load output LAZ file
2. Color by classification
3. Check roof types (classes 63-66)
4. Verify details (classes 67-69)

## Limitations

### Current Limitations

1. **No overhang detection** - Roof overhangs not distinguished from edges
2. **Simple dormer logic** - May miss complex dormer shapes
3. **No chimney detection** - Chimneys not separately classified (Phase 2.2)
4. **Fixed thresholds** - Some parameters not yet configurable

### Planned Improvements (v3.2)

- [ ] Chimney and superstructure detection (Phase 2.2)
- [ ] Balcony and overhang detection (Phase 2.3)
- [ ] Configurable detection thresholds
- [ ] Machine learning-based type classification
- [ ] Support for curved/non-planar roofs

## References

### Scientific Background

1. **Awrangjeb, M. & Fraser, C. (2014)**  
   "Automatic Segmentation of Raw LiDAR Data for Building Roof Extraction"

2. **Tarsha-Kurdi, F. et al. (2007)**  
   "Hough-Transform and Extended RANSAC Algorithms for Automatic Detection of 3D Building Roof Planes"

3. **Demantké, J. et al. (2011)**  
   "Dimensionality Based Scale Selection in 3D LiDAR Point Clouds"

### Related Documentation

- [Feature Computation Guide](../features/overview.md)
- [Building Classification Guide](./building-classification.md)
- [Configuration Reference](../configuration/reference.md)
- [ASPRS Classification Codes](../reference/asprs-codes.md)

## Support

For issues or questions:

- **GitHub Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Documentation:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **Email:** [maintainer email]

---

**Last Updated:** 2025-01-XX  
**Version:** 3.1.0
