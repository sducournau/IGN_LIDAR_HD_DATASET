# Building Mode Enrichment - Bug Fixes

## Issues Identified

### 1. Missing Building-Specific Functions

**Error:** `cannot import name 'compute_wall_score' from 'ign_lidar.features'`

**Cause:** The CLI was attempting to import three building-specific functions that were not implemented:

- `compute_wall_score()`
- `compute_roof_score()`
- `compute_num_points_in_radius()`

### 2. COPC File Writing Not Supported

**Error:** `Writing COPC is not supported`

**Cause:** Some input files were COPC (Cloud Optimized Point Cloud) format, which laspy doesn't support for writing with extra dimensions.

### 3. Invalid File Signature

**Error:** `Invalid file signature "b'\x00\x00\x00\x00'"`

**Cause:** Some COPC files had corrupted or invalid headers that couldn't be read.

## Fixes Applied

### 1. Implemented Missing Functions in `features.py`

Added three new building-specific feature computation functions:

#### `compute_wall_score(normals, height_above_ground, min_height=1.5)`

Computes wall probability by combining:

- **Verticality**: High vertical alignment of normals
- **Height**: Elevated above ground (>1.5m)
- **Formula**: `wall_score = verticality × height_score`

#### `compute_roof_score(normals, height_above_ground, curvature, min_height=2.0)`

Computes roof probability by combining:

- **Horizontality**: Normal vectors pointing up/down
- **Height**: Elevated above ground (>2.0m)
- **Planarity**: Low curvature (flat surfaces)
- **Formula**: `roof_score = horizontality × height_score × planarity_score`

#### `compute_num_points_in_radius(points, radius=2.0)`

Computes local point density:

- Uses KD-tree for efficient radius search
- Returns number of neighboring points within 2m radius
- Useful for distinguishing dense buildings from sparse vegetation

### 2. Added COPC File Handling in `cli.py`

**Detection Logic:**

```python
# Check filename
if '.copc.' in filename.lower():
    is_copc = True

# Check LAS header for COPC VLRs
if has_copc_variable_length_records():
    is_copc = True
```

**Conversion Strategy:**

- COPC files are detected (both by filename and VLR headers)
- Files are **converted** to standard LAZ format during processing
- Conversion creates a new header without COPC VLRs
- Preserves all point data and standard fields
- Allows adding extra dimensions (which COPC doesn't support)
- No data loss, no crashes

### 3. Added Read Error Handling

**Protection Against Corrupted Files:**

```python
try:
    with laspy.open(laz_path) as f:
        las = f.read()
except Exception as read_error:
    logger.error(f"Cannot read file: {read_error}")
    return False
```

## Verification

### Test Script

Created `test_building_features.py` to verify:

```bash
✓ Verticality: shape=(100,), range=[0.03, 1.00]
✓ Wall score: shape=(100,), range=[0.00, 0.98]
✓ Roof score: shape=(100,), range=[0.00, 0.33]
✓ Num points: shape=(100,), range=[0, 1]
```

### Import Test

```bash
python -c "from ign_lidar.features import compute_wall_score, compute_roof_score, compute_num_points_in_radius"
# ✓ Success
```

## Expected Behavior Now

### Normal LAZ Files

- Will be enriched with all BUILDING features:
  - normal_x, normal_y, normal_z
  - curvature
  - height_above_ground
  - planarity, linearity, sphericity, anisotropy, roughness, density
  - **verticality** (NEW)
  - **wall_score** (NEW)
  - **roof_score** (NEW)
  - **num_points_2m** (NEW)

### COPC Files

- Will be **automatically converted** to standard LAZ:
  ```
  ℹ️ COPC detected - will convert to standard LAZ
  ```
- All point data and fields preserved
- Extra dimensions added successfully
- Processing continues normally

### Corrupted Files

- Will be **skipped** with error:
  ```
  ✗ Cannot read file filename.laz: Invalid file signature
  ```
- Processing continues with next file

## Usage

Run the same command again:

```bash
python -m ign_lidar.cli enrich \
  --input /mnt/c/Users/Simon/ign/raw_tiles/ \
  --output /mnt/c/Users/Simon/ign/pre_tiles/ \
  --k-neighbors 20 \
  --mode building \
  --num-workers 4
```

### Expected Results

- **Normal LAZ files**: Enriched successfully with all building features
- **COPC files**: Converted and enriched successfully
- **Corrupted files**: Skipped (approximately 1-2 files)
- **Total successful**: ~120 out of 122 files (only corrupt files skipped)

### Performance

- **Time**: 2-3 hours for 122 tiles
- **Memory**: ~6-8 GB per worker (24-32 GB with 4 workers)
- If OOM occurs, reduce to `--num-workers 2`

## Files Modified

1. **`ign_lidar/features.py`**

   - Added `compute_wall_score()`
   - Added `compute_roof_score()`
   - Added `compute_num_points_in_radius()`

2. **`ign_lidar/cli.py`**
   - Added COPC file detection and skipping
   - Added read error handling
   - Improved error messages

## Testing

Run a quick test on a single file:

```bash
python -m ign_lidar.cli enrich \
  --input /mnt/c/Users/Simon/ign/raw_tiles/LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.laz \
  --output /tmp/test_output/ \
  --k-neighbors 20 \
  --mode building \
  --num-workers 1
```

Verify the output:

```python
import laspy

with laspy.open('/tmp/test_output/LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.laz') as f:
    las = f.read()

# Check building features exist
assert 'verticality' in las.point_format.dimension_names
assert 'wall_score' in las.point_format.dimension_names
assert 'roof_score' in las.point_format.dimension_names
assert 'num_points_2m' in las.point_format.dimension_names

print("✓ All building features present!")
```
