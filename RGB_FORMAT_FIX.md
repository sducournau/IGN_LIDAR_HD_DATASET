# RGB Point Format Fix

## Issue

When running `ign-lidar-hd enrich` with `--add-rgb` flag on COPC files, the following error occurred:

```
⚠️  Could not add RGB colors: Point format <PointFormat(6, 60 bytes of extra dims)> does not support red dimension
```

## Root Cause

When converting from COPC (Cloud Optimized Point Cloud) to standard LAZ format:

1. A new `LasHeader` was created with `point_format=6`
2. Point format 6 theoretically supports RGB in the LAS 1.4 specification
3. However, when creating a fresh `LasData` object with a new header using format 6, laspy doesn't properly initialize the RGB dimensions
4. Attempting to set `las_out.red`, `las_out.green`, `las_out.blue` failed because the dimensions weren't available

## Solution

Modified the COPC-to-LAZ conversion logic in `ign_lidar/cli.py` to:

1. **For COPC files with RGB augmentation**: Convert point format 6 to format 7

   - Format 7 includes RGB + NIR and properly initializes RGB dimensions
   - This ensures RGB fields are available when needed

2. **For standard LAZ files without RGB support**: Convert to appropriate RGB-enabled format
   - Format 7 for LAS 1.4+ (version.minor >= 4)
   - Format 3 for older LAS versions
   - Only converts if RGB augmentation is requested and format doesn't already support RGB

## Technical Details

### Point Format Support for RGB

- **LAS 1.2-1.3**: Formats 2, 3, 5 support RGB
- **LAS 1.4**: Formats 6, 7, 8, 10 support RGB
- **Format 7** is the recommended choice for LAS 1.4 with RGB

### Code Changes

The fix adds smart format detection and conversion:

```python
# For COPC files
if add_rgb and target_format == 6:
    target_format = 7  # Format 7 has native RGB support

# For standard LAZ files without RGB support
rgb_formats = [2, 3, 5, 7, 8, 10]
if add_rgb and las.header.point_format.id not in rgb_formats:
    target_format = 7 if las.header.version.minor >= 4 else 3
```

## Testing

Run the same command that previously failed:

```bash
ign-lidar-hd enrich \
  --input-dir /mnt/c/Users/Simon/ign/raw_tiles \
  --output /mnt/c/Users/Simon/ign/pre_tiles \
  --num-workers 4 \
  --mode building \
  --add-rgb
```

Expected behavior:

- COPC files are converted from format 6 to format 7
- RGB colors are successfully added to all points
- No "does not support red dimension" errors

## Impact

- **Backward compatible**: Only converts formats when RGB augmentation is requested
- **Minimal overhead**: Format conversion happens during COPC-to-LAZ conversion (already creating new header)
- **Quality**: Format 7 is the most feature-rich LAS 1.4 format with RGB support

## Related Files

- `ign_lidar/cli.py` - Main fix location (lines 263-330)
- `ign_lidar/rgb_augmentation.py` - RGB color fetching logic (unchanged)

## Date

Fixed: October 3, 2025
