# ARTIFACT ANALYSIS REPORT - Patch 0300

## Versailles Dataset - Dash Line Artifacts

### 📊 PROBLEM CONFIRMED

Analysis of patch 0300 shows **strong spatial artifacts** in the Y-direction:

**Affected Features:**

- ✅ **Planarity**: CV = 0.2889 (28.9% variation) ⚠️ HIGH
- ✅ **Roof_score**: CV = 0.3085 (30.8% variation) ⚠️ HIGH
- ⚠️ **Linearity**: CV = 0.1630 (16.3% variation) - Borderline

**Coefficient of Variation (CV) Analysis:**

- **X-direction**: Normal variation (CV < 0.13)
- **Y-direction**: Abnormal striping (CV > 0.28)

### 🔍 ROOT CAUSE ANALYSIS

The **dash line artifacts in Y-direction** are caused by:

1. **Patch Boundary Effects**:

   - Patches are created with sharp boundaries at 50m × 50m tiles
   - Points near Y-boundaries have truncated neighborhoods
   - Feature computation at edges uses incomplete spatial context

2. **Neighborhood Truncation**:

   - `search_radius: 1.5m` is used for feature computation
   - At patch edges, the search sphere is cut off
   - This creates systematically different feature values along patch boundaries

3. **No Buffer Zone**:
   - Config shows: `stitching: enabled: false`
   - No overlap buffering between adjacent patches
   - Features computed independently per patch without spatial context

### 📈 DETECTION METHOD

**Spatial Grid Analysis (20 bins):**

```
Planarity - Y-direction:
  Mean across bins: 0.3274
  Std across bins: 0.0946
  CV: 0.2889  ← 28.9% variation indicates striping!

Roof_score - Y-direction:
  Mean across bins: 0.2692
  Std across bins: 0.0831
  CV: 0.3085  ← 30.8% variation indicates striping!
```

### ✅ RECOMMENDED FIXES

#### Option 1: Enable Overlap Processing (BEST)

```yaml
preprocessing:
  patch_overlap: 0.25 # Increase from 0.15 to 0.25
  trim_edge_features: true # NEW: Trim edge features
  edge_trim_distance: 2.0 # Remove features within 2m of patch boundary
```

#### Option 2: Increase Search Radius at Boundaries

```yaml
features:
  search_radius: 2.5 # Increase from 1.5 to 2.5m
  boundary_aware_features: true # NEW: Use expanded search at edges
  edge_expansion_factor: 1.5 # Expand search 1.5x at boundaries
```

#### Option 3: Post-Processing Smoothing

```python
def smooth_boundary_artifacts(features, patch_coords, boundary_width=2.0):
    """
    Smooth features near patch boundaries to remove dash line artifacts.

    Args:
        features: Feature dict with planarity, linearity, roof_score
        patch_coords: XY coordinates of points
        boundary_width: Distance from edge to smooth (meters)
    """
    # Find patch boundaries
    x_min, x_max = patch_coords[:, 0].min(), patch_coords[:, 0].max()
    y_min, y_max = patch_coords[:, 1].min(), patch_coords[:, 1].max()

    # Identify edge points
    edge_mask = (
        (patch_coords[:, 0] < x_min + boundary_width) |
        (patch_coords[:, 0] > x_max - boundary_width) |
        (patch_coords[:, 1] < y_min + boundary_width) |
        (patch_coords[:, 1] > y_max - boundary_width)
    )

    # Smooth edge features using k-NN interpolation from interior
    for feat_name in ['planarity', 'linearity', 'roof_score']:
        if feat_name in features:
            # Apply Gaussian smoothing to edge features
            features[feat_name][edge_mask] = gaussian_smooth(
                features[feat_name],
                patch_coords,
                edge_mask,
                sigma=1.0
            )

    return features
```

#### Option 4: Use Boundary-Aware Feature Computer

```python
from ign_lidar.features.features_boundary import BoundaryAwareFeatureComputer

computer = BoundaryAwareFeatureComputer(
    k_neighbors=30,
    search_radius=1.5,
    compute_planarity=True,
    compute_density=True,
    compute_architectural=True
)

# Compute features with buffer points from adjacent patches
features = computer.compute_features(
    core_points=patch_points,
    buffer_points=neighbor_points,  # Points from adjacent patches
    tile_bounds=(x_min, y_min, x_max, y_max)
)
```

### 🎯 IMMEDIATE ACTION

**Quick Fix for Current Dataset:**
Run post-processing to identify and flag boundary points:

```bash
# Mark boundary points in existing files
python scripts/mark_boundary_artifacts.py \
    --input /mnt/c/Users/Simon/ign/versailles/output/*.laz \
    --boundary_width 2.0 \
    --add_flag "boundary_artifact"
```

**For Future Processing:**
Update config to use boundary-aware features:

```yaml
features:
  search_radius: 2.0 # Increased for better boundary handling
  boundary_aware: true

preprocessing:
  patch_overlap: 0.25
  trim_edges: true
  edge_trim_distance: 2.0
```

### 📝 NOTES

- Artifacts are **spatial patterns**, not value outliers
- Features are mathematically valid but **geometrically biased** at boundaries
- This is a **common issue** in patch-based point cloud processing
- The fix requires either:
  1. Larger context windows (overlap + buffer)
  2. Boundary trimming (lose some data)
  3. Post-processing smoothing (less accurate)

### 🔗 RELATED FILES

- Config: `examples/config_lod3_training_50m_versailles.yaml`
- Boundary computer: `ign_lidar/features/features_boundary.py`
- Feature modes: `ign_lidar/features/feature_modes.py`
- Detection script: `scripts/check_spatial_artifacts_patch300.py`

---

**Analysis Date:** October 14, 2025
**Analyst:** GitHub Copilot
**Patch:** 0300 (24,576 points, 50m × 50m)
**Dataset:** Versailles IGN LIDAR HD
