# Dash Line Artifacts - Detection & Fix Summary

## Problem Confirmed ✓

Analysis of patch 0300 shows **spatial striping artifacts**:
- **Planarity**: 28.9% Y-direction variation
- **Roof_score**: 30.8% Y-direction variation  
- **Linearity**: 16.3% Y-direction variation

## Available Tools

### 1. Detection & Analysis
```bash
# Spatial analysis
python scripts/check_spatial_artifacts_patch300.py

# Visual inspection  
python scripts/visualize_artifacts.py \
    --input output/patch_0300.laz \
    --features planarity,roof_score,linearity
```

### 2. Fixing Options

#### Option A: Reprocess (Best Quality)
```bash
ign-lidar-hd process \
    --config-file examples/config_lod3_training_50m_versailles_fixed.yaml
```

#### Option B: Post-Process Existing Files
```bash
# Flag boundary points
python scripts/fix_boundary_artifacts.py \
    --input "output/*.laz" \
    --output output_fixed/ \
    --method flag --boundary_width 2.5

# OR smooth features
--method smooth

# OR remove boundary points
--method remove
```

## Files Created

| File | Purpose |
|------|---------|
| `ARTIFACT_REPORT_PATCH300.md` | Detailed analysis report |
| `docs/FIXING_BOUNDARY_ARTIFACTS.md` | Complete fixing guide |
| `scripts/check_spatial_artifacts_patch300.py` | Spatial CV analysis |
| `scripts/visualize_artifacts.py` | Artifact visualization |
| `scripts/fix_boundary_artifacts.py` | Post-processing fix |
| `examples/config_lod3_training_50m_versailles_fixed.yaml` | Fixed config |

## Quick Reference

**Root Cause:** Patch boundaries + insufficient overlap + truncated search radius

**Fix Formula:**
```
patch_overlap >= 2 × search_radius
edge_trim_distance >= search_radius
```

**Recommended Settings:**
- `search_radius: 2.5m` (was 1.5m)
- `patch_overlap: 0.25` (was 0.15)
- `edge_trim_distance: 2.5m` (new)

**Expected Results After Fix:**
- All CV values < 0.15
- No visible dash lines
- ~5% loss of edge points (acceptable trade-off)

## Next Steps

1. **Validate Current Data:**
   ```bash
   python scripts/check_spatial_artifacts_patch300.py
   ```

2. **Choose Fix Method:**
   - Reprocess if possible (best quality)
   - Post-process if reprocessing not feasible

3. **Verify Fix:**
   ```bash
   # Run analysis on fixed data
   python scripts/check_spatial_artifacts_patch300.py
   ```

4. **Visual Comparison:**
   ```bash
   python scripts/visualize_artifacts.py --input fixed_patch.laz
   ```

## Support

See `docs/FIXING_BOUNDARY_ARTIFACTS.md` for complete guide.
