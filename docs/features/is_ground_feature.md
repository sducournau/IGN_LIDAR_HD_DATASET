# is_ground Feature with DTM Augmentation

The `is_ground` feature is a binary indicator (0/1) that identifies ground points in LiDAR point clouds, with full support for DTM (Digital Terrain Model) augmentation.

## Overview

**Feature Name:** `is_ground`  
**Type:** Binary (int8)  
**Values:**

- `1` = Ground point (ASPRS class 2)
- `0` = Non-ground point

**Added in:** Version 3.1.0  
**Module:** `ign_lidar.features.compute.is_ground`

## Key Features

✅ **Binary Ground Indicator:** Simple 0/1 classification of ground vs non-ground  
✅ **DTM-Aware:** Detects and handles synthetic ground points from DTM augmentation  
✅ **Multi-Source:** Combines natural LiDAR ground + DTM-augmented synthetic ground  
✅ **Configurable:** Option to include or exclude synthetic ground points  
✅ **Statistics:** Detailed logging of ground coverage and DTM contribution  
✅ **Efficient:** O(N) time complexity, minimal memory overhead

## Use Cases

### 1. Height Computation

The primary use case is to identify ground reference points for accurate height computation:

```python
from ign_lidar.features.compute import compute_is_ground, compute_height_above_ground

# Identify ground points
is_ground = compute_is_ground(classification)

# Use ground points to compute accurate heights
ground_points = points[is_ground.astype(bool)]
heights = compute_height_above_ground(points, classification)
```

**Benefits:**

- More accurate ground reference in areas with sparse ground coverage
- Better under vegetation canopy where LiDAR can't penetrate
- Improved under buildings where ground is occluded

### 2. Classification Validation

Assess ground/non-ground separation quality:

```python
from ign_lidar.features.compute import compute_is_ground_with_stats

# Get is_ground with statistics
is_ground, stats = compute_is_ground_with_stats(
    classification,
    synthetic_flags,
    verbose=True
)

print(f"Ground coverage: {stats['ground_percentage']:.1f}%")
print(f"DTM contribution: {stats['synthetic_percentage']:.1f}%")
```

**Metrics:**

- Total ground points and percentage
- Natural vs synthetic ground distribution
- Coverage gaps identification

### 3. Machine Learning Feature

Use as binary input feature for ML models:

```python
# Add is_ground to feature array
features['is_ground'] = compute_is_ground(classification)

# Helps models learn:
# - Ground/non-ground boundary
# - Low object classification (curbs, low walls)
# - Terrain vs structure separation
```

### 4. Ground Coverage Analysis

Identify areas needing DTM augmentation:

```python
from ign_lidar.features.compute import identify_ground_gaps

# Find areas with sparse ground coverage
gap_mask, gap_stats = identify_ground_gaps(
    points,
    is_ground,
    grid_size=10.0,
    min_density_threshold=0.5
)

print(f"Found {gap_stats['n_gap_cells']} cells needing augmentation")
```

## Configuration

### Enable is_ground Feature

Add to your YAML configuration:

```yaml
features:
  mode: "asprs_classes" # or "lod2", "lod3", "full"
  compute_is_ground: true # Enable is_ground computation

  # Synthetic ground handling
  include_synthetic_ground: true # Include DTM-augmented points
```

### DTM Augmentation Setup

To use DTM augmentation with is_ground:

```yaml
processor:
  augment_ground: true # Enable DTM augmentation

dtm:
  enabled: true
  resolution: 1.0 # 1m resolution

  augmentation:
    strategy: "intelligent" # Add ground where needed
    spacing: 2.0 # Synthetic point spacing (meters)

    # Areas to augment
    augment_vegetation: true # Under trees/vegetation
    augment_buildings: true # Under building footprints
    augment_gaps: true # Coverage gaps
```

## DTM Augmentation Workflow

The is_ground feature integrates seamlessly with DTM augmentation:

```
1. LiDAR Processing
   ├─ Points classified (ASPRS codes)
   └─ Natural ground points identified (class 2)

2. DTM Augmentation (if enabled)
   ├─ Fetch RGE ALTI DTM for tile area
   ├─ Generate synthetic ground points in gaps
   ├─ Validate synthetic points
   └─ Mark with synthetic_flags

3. is_ground Computation
   ├─ Natural ground: classification == 2
   ├─ Synthetic ground: synthetic_flags == True (if included)
   └─ Combined: is_ground = 1 for all ground
```

## API Reference

### compute_is_ground()

```python
def compute_is_ground(
    classification: np.ndarray,
    synthetic_flags: Optional[np.ndarray] = None,
    ground_class: int = 2,
    include_synthetic: bool = True
) -> np.ndarray
```

**Parameters:**

- `classification` (np.ndarray): ASPRS classification codes, shape (N,)
- `synthetic_flags` (np.ndarray, optional): Boolean array marking synthetic points from DTM
- `ground_class` (int): ASPRS ground class code (default: 2)
- `include_synthetic` (bool): Include synthetic ground points (default: True)

**Returns:**

- `is_ground` (np.ndarray): Binary ground indicator, shape (N,), dtype int8

**Example:**

```python
>>> classification = np.array([2, 2, 6, 3, 2, 5])
>>> is_ground = compute_is_ground(classification)
>>> print(is_ground)
[1 1 0 0 1 0]
```

### compute_is_ground_with_stats()

```python
def compute_is_ground_with_stats(
    classification: np.ndarray,
    synthetic_flags: Optional[np.ndarray] = None,
    ground_class: int = 2,
    include_synthetic: bool = True,
    verbose: bool = True
) -> tuple[np.ndarray, dict]
```

**Returns:**

- `is_ground` (np.ndarray): Binary ground indicator
- `stats` (dict): Statistics with keys:
  - `total_points`: Total number of points
  - `natural_ground`: Natural LiDAR ground points
  - `synthetic_ground`: DTM-augmented ground points
  - `total_ground`: Total ground points
  - `non_ground`: Non-ground points
  - `ground_percentage`: Percentage of ground points
  - `synthetic_percentage`: Percentage of ground that is synthetic

**Example:**

```python
>>> is_ground, stats = compute_is_ground_with_stats(
...     classification,
...     synthetic_flags,
...     verbose=True
... )
=== Ground Point Statistics ===
  Total points: 1,000,000
  Natural ground: 120,000
  Synthetic ground (DTM): 30,000
  Total ground: 150,000 (15.0%)
  Non-ground: 850,000
  DTM contribution: 20.0% of ground
```

### compute_ground_density()

```python
def compute_ground_density(
    points: np.ndarray,
    is_ground: np.ndarray,
    grid_size: float = 10.0
) -> tuple[np.ndarray, float]
```

Compute spatial density of ground points on a grid.

**Parameters:**

- `points` (np.ndarray): Point cloud, shape (N, 3)
- `is_ground` (np.ndarray): Binary ground indicator
- `grid_size` (float): Grid cell size in meters

**Returns:**

- `density_map` (np.ndarray): 2D grid with ground point density (points/m²)
- `mean_density` (float): Mean ground density

### identify_ground_gaps()

```python
def identify_ground_gaps(
    points: np.ndarray,
    is_ground: np.ndarray,
    grid_size: float = 10.0,
    min_density_threshold: float = 0.5
) -> tuple[np.ndarray, dict]
```

Identify spatial gaps in ground coverage (areas needing DTM augmentation).

**Returns:**

- `gap_mask` (np.ndarray): Boolean mask for points in sparse-ground areas
- `gap_stats` (dict): Statistics about identified gaps

## Performance

- **Computation:** O(N) time complexity, simple boolean operation
- **Memory:** Minimal overhead (int8 array = 1 byte per point)
- **Typical tile:** <1ms for 1M points

## Integration with FeatureOrchestrator

The is_ground feature is automatically computed when:

1. Feature mode includes it (asprs_classes, lod2, lod3, full)
2. `compute_is_ground: true` in config

```python
from ign_lidar.features.orchestrator import FeatureOrchestrator

# Initialize with config
orchestrator = FeatureOrchestrator(config)

# Compute features (includes is_ground)
tile_data = {
    'points': points,
    'classification': classification,
    'synthetic_flags': synthetic_flags,  # Optional
    'intensity': intensity,
    'return_number': return_number,
}

features = orchestrator.compute_features(tile_data)

# is_ground is now in features dict
print(f"Ground points: {np.sum(features['is_ground'])}")
```

## Logging Output

When is_ground is computed, you'll see:

```
  ✓ is_ground feature: 150,000 ground points (15.0%) | 30,000 from DTM (20.0%)
```

Or without DTM:

```
  ✓ is_ground feature: 120,000 ground points (12.0%)
```

## Comparison: With vs Without DTM

### Without DTM Augmentation

```python
# Sparse ground in forests/buildings
is_ground, stats = compute_is_ground_with_stats(classification)
# Ground coverage: 8-12% (many gaps under vegetation)
```

### With DTM Augmentation

```python
# DTM fills gaps
is_ground, stats = compute_is_ground_with_stats(
    classification,
    synthetic_flags
)
# Ground coverage: 15-20% (synthetic points fill gaps)
# DTM contribution: 20-40% of total ground
```

**Impact:**

- ✅ Better ground reference for height computation
- ✅ More accurate vegetation height estimation
- ✅ Reduced bias from ground coverage gaps
- ✅ Improved low-object classification

## Feature Modes

The is_ground feature is included in:

- ✅ **ASPRS_CLASSES** mode (ASPRS classification)
- ✅ **LOD2_SIMPLIFIED** mode (building detection)
- ✅ **LOD3_FULL** mode (detailed modeling)
- ✅ **FULL** mode (all features)
- ❌ **MINIMAL** mode (ultra-fast, minimal features)

## Best Practices

1. **Enable DTM Augmentation:** For better ground coverage, especially in forested areas
2. **Validate Synthetic Points:** Use `validate_against_neighbors: true` in DTM config
3. **Check Statistics:** Review ground coverage and DTM contribution in logs
4. **Exclude When Needed:** Set `include_synthetic_ground: false` if analyzing natural LiDAR only
5. **Use with Height Features:** Combine with `height_above_ground` for best results

## Troubleshooting

### Problem: Low ground percentage (<5%)

**Solution:** Enable DTM augmentation

```yaml
processor:
  augment_ground: true
```

### Problem: Too many synthetic points

**Solution:** Adjust augmentation strategy

```yaml
dtm:
  augmentation:
    strategy: "gaps" # Only fill gaps, don't add everywhere
    min_spacing_to_existing: 2.0 # Increase spacing
```

### Problem: Need natural ground only

**Solution:** Exclude synthetic points

```yaml
features:
  include_synthetic_ground: false
```

## Related Features

- **height_above_ground:** Uses ground points for reference
- **height:** Z-normalized height
- **classification:** ASPRS classification codes
- **verticality:** Vertical surface detection (non-ground)
- **planarity:** Flat surface detection (ground/roofs)

## References

- RGE ALTI® DTM: 1m resolution digital terrain model from IGN
- ASPRS LAS 1.4 Specification: Standard classification codes
- DTM Augmentation Module: `ign_lidar.core.classification.dtm_augmentation`
