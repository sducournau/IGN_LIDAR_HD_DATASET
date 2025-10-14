# Scan Line Artifacts Report - Patch 300

## Executive Summary

**Problem**: Dash-line artifacts detected in eigenvalue-based features (planarity, roof_score, linearity) in patch 300 and likely other patches.

**Severity**: SEVERE (CV_y > 0.15 for all three features)

**Root Cause**: LiDAR scan line patterns causing systematic spatial variation in feature values, despite uniform point density.

---

## Detailed Analysis

### 1. Artifact Metrics (Patch 300)

| Feature        | CV_y  | Range/Mean | Status      |
| -------------- | ----- | ---------- | ----------- |
| **Planarity**  | 0.278 | 1.12       | ⚠️ SEVERE   |
| **Roof Score** | 0.301 | 1.31       | ⚠️ SEVERE   |
| **Linearity**  | 0.156 | 0.60       | ⚠️ MODERATE |

- CV_y > 0.15 indicates significant stripe patterns
- Values vary by 2-3x across spatial bins despite similar geometry

### 2. Pattern Visualization

```
Y-direction bins (1-20) showing planarity variation:
Bins 4,10,17: HIGH (0.47-0.58) ***
Bins 5,9,15,18: LOW (0.21-0.22) ***
Pattern repeats ~every 2-3 bins → scan line spacing
```

### 3. Diagnostic Results

✓ **Point density is uniform** (CV=0.08) → Not a density issue
✓ **24,576 points in 50m×50m patch** → Good coverage
⚠️ **Return number distribution shows clustering** (CV_y up to 1.31 for return 3)
⚠️ **Spatial bins show 2-3x variation** in feature means

### 4. Root Causes Identified

1. **LiDAR Scan Geometry**

   - Airborne LiDAR scans in parallel lines (flight path)
   - Points within same scan line have different local neighborhoods than between lines
   - Scan line spacing: ~2.5-5m (inferred from artifact pattern)

2. **Fixed Radius Neighborhood Search**

   - Current: `tree.query_radius(points, r=radius)`
   - Same radius for all points ignores anisotropic point distribution
   - Neighborhoods crossing scan lines get different geometry than within-line

3. **Eigenvalue Computation Sensitivity**
   - PCA on neighborhood point cloud
   - Scan line gaps → biased covariance matrix
   - Small changes in neighborhood → large changes in eigenvalues

---

## Solution Strategy

### Approach 1: Scan-Line-Aware Neighborhood Search (RECOMMENDED)

**Concept**: Detect scan lines and use anisotropic search radii

```python
def compute_scan_line_aware_features(points, radius=2.0):
    """
    Compute geometric features with scan line correction.

    Strategy:
    1. Detect scan lines using point spacing analysis
    2. Use elliptical search (longer along scan line, shorter across)
    3. Weight neighbors by distance and scan line compatibility
    """
    # Detect scan direction
    scan_direction = detect_scan_lines(points)

    # Anisotropic search
    for i, pt in enumerate(points):
        # Search with ellipse aligned to scan direction
        neighbors = find_anisotropic_neighbors(
            pt,
            radius_along=radius * 1.5,  # Along scan line
            radius_across=radius * 0.75, # Across scan lines
            scan_dir=scan_direction[i]
        )

        # Compute features...
```

### Approach 2: Post-Processing Spatial Smoothing (QUICK FIX)

**Concept**: Apply median filter to remove stripe patterns

```python
def fix_scan_line_artifacts(features, coords, window_size=5):
    """
    Remove scan line artifacts using spatial median filtering.

    For each feature:
    1. Divide space into grid cells
    2. Apply 2D median filter across cells
    3. Interpolate smoothed values back to points
    """
    target_features = ['planarity', 'linearity', 'roof_score']

    for feat_name in target_features:
        # Grid-based median filtering
        smoothed = spatial_median_filter(
            features[feat_name],
            coords,
            window_size=window_size,
            direction='y'  # Perpendicular to scan lines
        )
        features[feat_name] = smoothed
```

### Approach 3: Adaptive Radius Selection

**Concept**: Use different radii based on local point pattern

```python
def adaptive_radius_search(point, tree, min_radius=1.5, max_radius=3.0):
    """
    Adaptively select radius to ensure consistent neighborhood quality.

    Target: 20-50 neighbors with good spatial distribution
    """
    for radius in np.linspace(min_radius, max_radius, 10):
        neighbors = tree.query_radius([point], r=radius)[0]

        # Check neighborhood quality
        if 20 <= len(neighbors) <= 50:
            # Compute isotropy metric
            isotropy = compute_isotropy(points[neighbors])
            if isotropy > 0.6:  # Reasonably isotropic
                return radius, neighbors

    # Fallback
    return max_radius, tree.query_radius([point], r=max_radius)[0]
```

---

## Implementation Priority

### Phase 1: Quick Fix (Post-Processing) - **IMMEDIATE**

1. Create script to apply spatial median filter to existing LAZ files
2. Focus on patch 300 and other SEVERE cases
3. Validate that artifacts are reduced

**Files to modify**:

- `scripts/fix_boundary_artifacts.py` → Add scan line fixing
- Test on patch 300

### Phase 2: Core Algorithm Fix (Medium-term) - **RECOMMENDED**

1. Modify `compute_local_geometric_features()` in `ign_lidar/features/features.py`
2. Add scan line detection
3. Implement anisotropic neighborhood search
4. Test on full dataset

**Files to modify**:

- `ign_lidar/features/features.py` lines 800-920
- `ign_lidar/features/features_gpu.py` (GPU version)

### Phase 3: Advanced Solution (Long-term) - **OPTIMAL**

1. Pre-process raw LAS files to detect scan lines
2. Store scan line metadata
3. Use metadata during feature computation
4. Validate on multiple datasets

---

## Expected Results

After applying fixes:

- CV_y should drop below 0.10 (acceptable variation)
- Range/Mean should be < 0.50
- Visual inspection should show smooth feature gradients
- No visible stripe patterns in visualization

---

## Testing Protocol

1. **Run batch analysis**:

   ```bash
   python scripts/batch_analyze_artifacts.py \
       --input /mnt/c/Users/Simon/ign/versailles/output/*.laz \
       --output artifact_report_before.csv
   ```

2. **Apply fix**:

   ```bash
   python scripts/fix_scan_line_artifacts.py \
       --input /mnt/c/Users/Simon/ign/versailles/output/*.laz \
       --output /mnt/c/Users/Simon/ign/versailles/output_fixed/
   ```

3. **Re-analyze**:

   ```bash
   python scripts/batch_analyze_artifacts.py \
       --input /mnt/c/Users/Simon/ign/versailles/output_fixed/*.laz \
       --output artifact_report_after.csv
   ```

4. **Compare metrics**:
   - Check CV_y reduction
   - Visual inspection in CloudCompare
   - Training performance impact

---

## Related Issues

- Similar to NDVI boundary artifacts (already fixed)
- May affect other eigenvalue-based features
- Important for training dataset quality

## References

- `ARTIFACT_REPORT_PATCH300.md`
- `batch_analyze_artifacts.py` output
- `fix_boundary_artifacts.py` (template for fix)
