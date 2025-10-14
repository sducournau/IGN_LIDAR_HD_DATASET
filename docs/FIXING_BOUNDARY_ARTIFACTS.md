# Fixing Dash Line Artifacts in IGN LIDAR HD Features

This guide explains how to identify and fix the spatial striping artifacts found in patch 0300 and similar patches.

## Problem Description

**Symptoms:**

- Dash lines or banding visible in planarity, roof_score, and linearity features
- High coefficient of variation (CV > 0.20) in Y-direction spatial analysis
- Features show systematic variation across patch boundaries

**Root Cause:**

- Patch boundaries create sharp cutoffs at 50m × 50m tiles
- Feature computation with `search_radius: 1.5m` has truncated neighborhoods at edges
- Points near boundaries use incomplete spatial context
- No buffering between adjacent patches

## Quick Diagnosis

### Check for Artifacts in Your Data

```bash
# Run spatial analysis on a patch
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
python scripts/check_spatial_artifacts_patch300.py
```

**Look for:**

- Y-direction CV > 0.20 → Strong artifacts
- Y-direction CV > 0.15 → Moderate artifacts
- Y-direction CV < 0.10 → Clean data

### Visualize Artifacts

```bash
# Create heatmap visualization
python scripts/visualize_artifacts.py \
    --input /mnt/c/Users/Simon/ign/versailles/output/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_hybrid_patch_0300.laz \
    --features planarity,roof_score,linearity \
    --output artifact_viz_patch300.png
```

## Solutions

### Option 1: Reprocess with Fixed Configuration (RECOMMENDED)

Use the fixed configuration that addresses boundary effects:

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

ign-lidar-hd process \
    --config-file examples/config_lod3_training_50m_versailles_fixed.yaml
```

**Key Changes:**

- `search_radius: 2.5m` (increased from 1.5m)
- `patch_overlap: 0.25` (increased from 0.15)
- `edge_trim_distance: 2.5m` (new - removes unreliable edge features)

**Trade-offs:**

- ✅ Clean features with consistent spatial context
- ✅ No visible artifacts
- ⚠️ Slightly smaller effective patch size (45m × 45m after trimming)
- ⚠️ ~15% longer processing time

### Option 2: Post-Process Existing Files

If you can't reprocess, fix existing LAZ files:

#### A. Flag Boundary Points

```bash
# Add boundary flags to user_data field
python scripts/fix_boundary_artifacts.py \
    --input "/mnt/c/Users/Simon/ign/versailles/output/*.laz" \
    --output /mnt/c/Users/Simon/ign/versailles/output_fixed/ \
    --method flag \
    --boundary_width 2.5
```

Then filter during training:

```python
# In your training script
mask = (user_data != 255)  # Exclude flagged boundary points
clean_features = features[mask]
clean_labels = labels[mask]
```

#### B. Smooth Boundary Features

```bash
# Smooth features at boundaries using k-NN interpolation
python scripts/fix_boundary_artifacts.py \
    --input "/mnt/c/Users/Simon/ign/versailles/output/*.laz" \
    --output /mnt/c/Users/Simon/ign/versailles/output_smoothed/ \
    --method smooth \
    --boundary_width 2.5 \
    --sigma 1.0
```

#### C. Remove Boundary Points

```bash
# Completely remove points near boundaries
python scripts/fix_boundary_artifacts.py \
    --input "/mnt/c/Users/Simon/ign/versailles/output/*.laz" \
    --output /mnt/c/Users/Simon/ign/versailles/output_trimmed/ \
    --method remove \
    --boundary_width 2.5
```

## Validation

After applying fixes, verify the improvements:

```bash
# Check spatial statistics again
python scripts/check_spatial_artifacts_patch300.py

# Visualize the fixed data
python scripts/visualize_artifacts.py \
    --input /mnt/c/Users/Simon/ign/versailles/output_fixed/LHD_FXX_0635_6857_PTS_C_LAMB93_IGN69_hybrid_patch_0300.laz \
    --features planarity,roof_score,linearity \
    --output fixed_viz_patch300.png
```

**Expected Results:**

- Planarity Y-direction CV: < 0.15 (down from 0.289)
- Roof_score Y-direction CV: < 0.15 (down from 0.309)
- Linearity Y-direction CV: < 0.10 (down from 0.163)

## Understanding the Metrics

### Coefficient of Variation (CV)

CV measures how much features vary across spatial bins:

```
CV = std(bin_means) / mean(bin_means)
```

**Interpretation:**

- CV < 0.10: Excellent uniformity
- CV < 0.15: Good uniformity
- CV > 0.20: Artifacts present ⚠️
- CV > 0.30: Strong artifacts ❌

### Why Y-Direction?

The artifacts appear in Y-direction because:

1. Patches are tiled in a regular grid
2. Feature computation happens independently per patch
3. Boundary effects accumulate along patch edges
4. Y-direction typically aligns with flight lines, amplifying the effect

## Configuration Reference

### Current (Artifact-Prone) Config

```yaml
features:
  search_radius: 1.5 # Too small for edges

processor:
  patch_overlap: 0.15 # Insufficient for 1.5m search radius

preprocess:
  trim_edges: false # No edge trimming
```

### Fixed Config

```yaml
features:
  search_radius: 2.5 # Larger for complete neighborhoods
  boundary_aware: true # Special edge handling
  edge_expansion_factor: 1.5 # Expand search at boundaries

processor:
  patch_overlap: 0.25 # 50m × 25% = 12.5m overlap

preprocess:
  trim_edges: true
  edge_trim_distance: 2.5 # Remove 2.5m from each edge
```

## Technical Details

### Search Radius Requirements

For a point at distance `d` from patch edge:

- Needs `search_radius ≤ d` for complete neighborhood
- At `d = 0` (exact edge), neighborhood is 50% truncated
- At `d = search_radius`, neighborhood is complete

**Rule of thumb:**

```
edge_trim_distance ≥ search_radius
patch_overlap ≥ 2 × search_radius
```

### Boundary Detection

Points are flagged as boundary if:

```python
boundary_mask = (
    (x < x_min + boundary_width) |
    (x > x_max - boundary_width) |
    (y < y_min + boundary_width) |
    (y > y_max - boundary_width)
)
```

For 50m patches with 2.5m boundary width:

- Interior region: 45m × 45m (81% of points)
- Boundary region: ~19% of points

## References

- **Analysis Report**: `ARTIFACT_REPORT_PATCH300.md`
- **Detection Script**: `scripts/check_spatial_artifacts_patch300.py`
- **Fix Script**: `scripts/fix_boundary_artifacts.py`
- **Visualization**: `scripts/visualize_artifacts.py`
- **Fixed Config**: `examples/config_lod3_training_50m_versailles_fixed.yaml`

## Support

If artifacts persist after applying these fixes:

1. Check that `search_radius` is properly set
2. Verify `patch_overlap >= 2 × search_radius`
3. Increase `edge_trim_distance` to match `search_radius`
4. Consider using larger patches (100m) to reduce edge ratio

For questions: See `ARTIFACT_REPORT_PATCH300.md` for detailed analysis.
