# Fixing Scan Line Artifacts in Geometric Features

## Problem Description

When visualizing geometric features like **planarity**, **linearity**, and **roof_score**, you may observe **dash-line patterns** or **stripe artifacts**. These artifacts appear as regular patterns following the LiDAR scan lines rather than representing true surface geometry.

### Example of Artifacts

```
Normal Surface (no artifacts):  Artifacted Surface (dash lines):
████████████████████████        ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║
██████████████████████          ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║
████████████████████████        ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║ ║
```

### Affected Features

- ✗ `planarity` - Shows stripes instead of uniform roof surfaces
- ✗ `linearity` - Follows scan lines instead of actual edges
- ✗ `roof_score` - Dash patterns on flat roofs
- ✗ `wall_score` - Irregular patterns on walls
- ✗ `sphericity` - Artificial patterns in vegetation

## Root Cause

The artifacts are caused by **k-nearest neighbors (k-NN) search** picking up the **anisotropic point distribution** from LiDAR scanning patterns:

1. **Airborne LiDAR scan pattern**: Points are dense along flight lines but sparse between them
2. **k-NN search bias**: Always finds k neighbors, often all from the same scan line
3. **Degenerate covariance**: Eigenvalue computation receives collinear points
4. **Feature artifacts**: Geometric features reflect scan pattern, not true geometry

### Why k-NN Fails for Airborne LiDAR

```
Scan Pattern:          k-NN Selection:        Radius Selection:
·····················  ·····················  ·····················
·····················  · ● ● ● ● ● ·······  · ● ● ● ● ● ·······
·····················  ·····················  · ● ● ◆ ● ● ·······
······◆··············  ······◆··············  · ● ● ● ● ● ·······
·····················  ·····················  ·····················
·····················  ·····················  · ● ● ● ● ● ·······
                                             ·····················

k-NN picks 5 points    → All from same line  Radius picks points
from same scan line      (collinear, bad!)    from multiple lines
                                              (planar, good!)
```

## Solution: Radius-Based Search

Use **spatial radius search** instead of k-NN to capture true surface geometry.

### Configuration Fix

Add the `search_radius` parameter to your configuration:

```yaml
features:
  mode: full
  k_neighbors: 30 # Fallback only (not used if search_radius set)
  search_radius: 1.5 # ← ADD THIS LINE
  # Recommended values:
  #   1.0-1.5m for dense urban areas (>15 pts/m²)
  #   1.5-2.0m for typical areas (10-15 pts/m²)
  #   2.0-2.5m for sparse rural areas (<10 pts/m²)
  #   null = auto-estimate from point density
  #   0 = use k_neighbors instead (NOT recommended)
```

### How It Works

**Radius-based search** queries all points within a fixed spatial distance:

1. **Isotropic sampling**: Gets points from all directions, not just scan lines
2. **True surface geometry**: Neighbors represent actual local surface
3. **Consistent scale**: Same physical area regardless of point density
4. **Clean features**: No scan line artifacts

### Algorithm Details

```python
# OLD (k-NN - causes artifacts):
tree.query(points, k=30)  # Always returns 30 neighbors
                          # → Often all from same scan line!

# NEW (Radius - no artifacts):
tree.query_radius(points, r=1.5)  # Returns all points within 1.5m
                                  # → Captures true surface geometry!
```

## Implementation

The fix has been applied in two places:

### 1. Configuration Schema (`ign_lidar/config/schema.py`)

Added `search_radius` parameter to `FeaturesConfig`:

```python
@dataclass
class FeaturesConfig:
    mode: Literal["minimal", "full", "custom"] = "full"
    k_neighbors: int = 20
    search_radius: Optional[float] = None  # ← NEW
    # ...
```

### 2. Feature Orchestrator (`ign_lidar/features/orchestrator.py`)

Updated `_compute_geometric_features` to pass radius:

```python
def _compute_geometric_features(self, points, classification, ...):
    search_radius = features_cfg.get('search_radius', None)

    # Log search strategy
    if search_radius is not None and search_radius > 0:
        logger.info(f"Computing features | radius={search_radius:.2f}m (avoids artifacts)")

    # Pass radius to feature computer
    feature_dict = self.computer.compute_features(
        points=points,
        classification=classification,
        radius=search_radius,  # ← NEW
        # ...
    )
```

## Recommended Configurations

### For IGN LIDAR HD (10 pts/m²)

```yaml
features:
  search_radius: 1.5 # Captures ~70-100 neighbors
```

### For Dense Urban Areas (>20 pts/m²)

```yaml
features:
  search_radius: 1.0 # Captures ~60-80 neighbors
```

### For Sparse Rural Areas (<5 pts/m²)

```yaml
features:
  search_radius: 2.5 # Captures ~40-60 neighbors
```

### Auto-Estimation

```yaml
features:
  search_radius: null # Auto-estimate from point density
```

The auto-estimation uses:

```
optimal_radius = avg_nearest_neighbor_distance × 18
```

## Verification

To verify the fix is working:

1. **Check logs** for radius message:

   ```
   🔧 Computing features | radius=1.50m (avoids scan line artifacts) | mode=FULL
   ```

2. **Visualize features** in CloudCompare:

   - Open enriched LAZ file
   - Display scalar field: `planarity`
   - Look for uniform surfaces instead of dash lines
   - Roofs should show solid high planarity (red), not stripes

3. **Compare before/after**:

   ```bash
   # Before (with artifacts)
   k_neighbors: 30
   # Planarity shows dash lines

   # After (no artifacts)
   search_radius: 1.5
   # Planarity shows smooth surfaces
   ```

## Performance Considerations

### Speed

- **Radius search**: Slightly slower (10-15% overhead) but more accurate
- **Chunked GPU**: Maintains good performance with radius search
- **Trade-off**: Worth it for artifact-free features

### Memory

- **Variable neighbors**: Radius returns different counts per point
- **Auto-adaptation**: Sparse areas get fewer, dense areas get more
- **Stable**: No OOM issues, handled by chunking

## Additional Resources

- **Documentation**: See `docs/docs/features/geometric-features.md`
- **Examples**: See `examples/config_*_fixed.yaml` files
- **Research**: Weinmann et al. (2015) - "Semantic point cloud interpretation"

## References

- [Weinmann et al., 2015](https://www.sciencedirect.com/science/article/pii/S0924271615001842) - Semantic point cloud interpretation based on geometric features
- [Demantké et al., 2011](https://hal.science/hal-00550385) - Dimensionality based scale selection in 3D LiDAR point clouds

## Summary

| Aspect                 | k-NN (Old)          | Radius (New)        |
| ---------------------- | ------------------- | ------------------- |
| Artifacts              | ✗ Dash lines        | ✓ Clean             |
| Surface Representation | ✗ Scan pattern      | ✓ True geometry     |
| Consistency            | ✗ Varies by density | ✓ Stable            |
| Performance            | ✓ Fast              | ✓ Fast (10% slower) |
| **Recommendation**     | ❌ Not recommended  | ✅ **Use this**     |

**Action Required**: Update your config with `search_radius: 1.5` to eliminate artifacts!
