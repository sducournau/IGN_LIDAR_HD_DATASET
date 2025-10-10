# LAZ Output Update Summary

## Overview

Updated the processor's LAZ saving functionality (`_save_patch_as_laz()`) to match the enhanced conversion script capabilities with proper LAMB93 coordinates and consistent feature naming.

## Changes Made

### 1. **LAMB93 Coordinate Restoration** ✅

The processor now restores proper LAMB93 coordinates when saving LAZ files:

```python
# Extract tile coordinates from filename (e.g., LHD_FXX_0649_6863_...)
tile_x = int(parts[2])  # 0649
tile_y = int(parts[3])  # 6863
tile_center_x = tile_x * 1000 + 500  # 649500 meters
tile_center_y = tile_y * 1000 + 500  # 6863500 meters

# Restore original coordinates
coords[:, 0] = coords[:, 0] + centroid[0] + tile_center_x
coords[:, 1] = coords[:, 1] + centroid[1] + tile_center_y
coords[:, 2] = coords[:, 2] + centroid[2]
```

**Result**: LAZ files now have real-world LAMB93 coordinates instead of normalized coordinates centered around 0.

### 2. **Consistent Feature Naming** ✅

Changed normal vector naming to match the conversion script:

**Before**: `nx`, `ny`, `nz`  
**After**: `normal_x`, `normal_y`, `normal_z`

**Rationale**: Consistency between conversion script and processor output. Both now use the same naming convention.

### 3. **NIR as Extra Dimension** ✅

Added NIR as an extra dimension for point formats that don't support it natively:

```python
# Add NIR as extra dimension if not already included in point format
if point_format != 8 and 'nir' in original_patch and original_patch['nir'] is not None:
    las.add_extra_dim(laspy.ExtraBytesParams(
        name='nir',
        type=np.float32,
        description="Near Infrared reflectance (normalized 0-1)"
    ))
    las.nir = original_patch['nir'].astype(np.float32)
```

**Result**: NIR is now included in all LAZ exports, either as standard field (format 8) or extra dimension (formats 3, 6).

### 4. **Point Format Selection** ✅

Improved point format selection logic:

```python
has_rgb = 'rgb' in arch_data
has_nir = 'nir' in original_patch and original_patch['nir'] is not None
point_format = 8 if (has_rgb and has_nir) else (3 if has_rgb else 6)
```

- **Format 8**: RGB + NIR (most feature-rich)
- **Format 3**: RGB only (maximum CloudCompare compatibility)
- **Format 6**: No RGB/NIR (basic format)

## Complete Feature List

The processor's LAZ output now includes:

### Standard Fields

- **Coordinates**: X, Y, Z (LAMB93 + IGN69)
- **Intensity**: Normalized 0-65535
- **RGB**: Red, Green, Blue (0-65535)
- **NIR**: Near Infrared (format 8 standard field, or extra dimension)
- **Classification**: Point classification code
- **Return Number**: Lidar return information

### Extra Dimensions (Computed Features)

#### Geometric Features (8 features)

- `planarity`: Planar surface characteristic (0-1)
- `linearity`: Linear structure characteristic (0-1)
- `sphericity`: Spherical/volumetric characteristic (0-1)
- `anisotropy`: Directional variation (0-1)
- `roughness`: Surface roughness measure
- `density`: Local point density
- `curvature`: Surface curvature
- `verticality`: Vertical alignment (0-1)

#### Normal Vectors (3 components)

- `normal_x`: X component of surface normal
- `normal_y`: Y component of surface normal
- `normal_z`: Z component of surface normal

#### Height Features (4 features)

- `height`: Absolute height above reference
- `z_normalized`: Normalized Z coordinate
- `z_from_ground`: Height above ground
- `z_from_median`: Height relative to median

#### Radiometric Features (2 features)

- `nir`: Near Infrared reflectance (0-1)
- `ndvi`: Normalized Difference Vegetation Index (-1 to 1)

**Total**: 17+ extra dimensions (depending on data availability)

## Consistency Verification

Both the conversion script and processor now:
✅ Use LAMB93 coordinates with tile center restoration  
✅ Use `normal_x`, `normal_y`, `normal_z` naming  
✅ Include NIR as extra dimension when not in standard fields  
✅ Include all geometric features  
✅ Include all radiometric features  
✅ Use LAS 1.4 format  
✅ Support LAZ compression

## CloudCompare Compatibility

The updated LAZ files are fully compatible with CloudCompare:

- ✅ Proper LAMB93 coordinates (visible on map)
- ✅ All standard RGB fields
- ✅ All extra dimensions accessible via scalar field menu
- ✅ Normals visible for visualization
- ✅ Geometric features for analysis
- ✅ NIR and NDVI for vegetation studies

## Testing

To verify the updates work correctly:

```bash
# Process a sample tile
python -m ign_lidar.cli.process \
    --architecture lod3_hybrid \
    --output outputs/test_laz_update \
    --save_format laz \
    --config your_config.yaml

# Check LAZ file in CloudCompare
# Should show:
# - Real-world LAMB93 coordinates (X≈649xxx, Y≈6863xxx)
# - 17+ extra dimensions in scalar field menu
# - normal_x, normal_y, normal_z (not nx, ny, nz)
```

## Impact

### Before Updates

- LAZ files had normalized coordinates (centered around 0)
- Inconsistent feature naming between conversion and processor
- NIR missing from format 3 files
- CloudCompare couldn't display files on map

### After Updates

- LAZ files have proper LAMB93 coordinates
- Consistent feature naming across all exports
- NIR included in all formats
- Full CloudCompare compatibility with map display
- All computed features accessible for visualization and analysis

## File Locations

- **Processor**: `ign_lidar/core/processor.py` (method `_save_patch_as_laz()`)
- **Conversion Script**: `scripts/convert_npz_to_laz_with_coords.py`
- **Documentation**:
  - `LAZ_FEATURES_GUIDE.md` (comprehensive guide)
  - `LAZ_FEATURES_QUICKREF.md` (quick reference)
  - `LAZ_IMPLEMENTATION_SUMMARY.md` (technical details)
  - `LAZ_OUTPUT_UPDATE_SUMMARY.md` (this file)

## Next Steps

1. **Test the updated processor** with your training data
2. **Regenerate LAZ files** from existing patches to get updated format
3. **Verify in CloudCompare** that all features are present
4. **Compare with conversion script output** to ensure consistency

## Notes

- The update is backward compatible - old LAZ files will still work
- Coordinate restoration requires filename to follow naming convention: `LHD_FXX_XXXX_YYYY_*`
- If metadata doesn't contain centroid, only tile offset is applied
- All feature additions are error-handled - missing features won't cause failures

---

**Date**: 2025-01-10  
**Version**: 2.0+ (post-LAZ enhancement)  
**Status**: ✅ Complete - Ready for testing
