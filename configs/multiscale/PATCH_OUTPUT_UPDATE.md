# Patch Output Format Update

**Date:** October 15, 2025

## Summary

Updated LOD2 and LOD3 preprocessing configurations to output **both NPZ and LAZ formats** for training patches.

## Changes Made

### LOD2 Configuration (`config_unified_lod2_preprocessing.yaml`)

**Before:**

```yaml
output:
  format: npz # Training patches
  processing_mode: patches_only
```

**After:**

```yaml
output:
  format: [npz, laz] # Training patches in both NPZ and LAZ formats
  processing_mode: patches_only
  save_laz_patches: true # Also save patches as LAZ files
```

### LOD3 Configuration (`config_unified_lod3_preprocessing.yaml`)

**Before:**

```yaml
output:
  format: npz
  processing_mode: patches_only
```

**After:**

```yaml
output:
  format: [npz, laz] # Training patches in both NPZ and LAZ formats
  processing_mode: patches_only
  save_laz_patches: true # Also save patches as LAZ files
```

## Output Structure

### LOD2 Patches

```
/mnt/c/Users/Simon/ign/preprocessed/lod2/patches/
├── patch_0001.npz          # Training data (NumPy format)
├── patch_0001.laz          # Visualization (LAZ format)
├── patch_0002.npz
├── patch_0002.laz
└── ...
```

### LOD3 Patches

```
/mnt/c/Users/Simon/ign/preprocessed/lod3/patches/
├── patch_0001.npz          # Training data (NumPy format)
├── patch_0001.laz          # Visualization (LAZ format)
├── patch_0002.npz
├── patch_0002.laz
└── ...
```

## Benefits

### 1. **Visual Inspection**

- LAZ patches can be opened in CloudCompare, QGIS, or other point cloud viewers
- Verify patch quality and class distribution visually
- Debug preprocessing issues more easily

### 2. **Quality Control**

- Inspect boundary artifacts at patch edges
- Verify feature computation accuracy
- Check ground truth annotation quality

### 3. **Presentation & Documentation**

- Create visualizations for reports and papers
- Share example patches with collaborators
- Document preprocessing results

### 4. **Debugging**

- Compare NPZ features with LAZ point clouds
- Verify coordinate transformations
- Check feature normalization

### 5. **Data Provenance**

- LAZ files maintain original point cloud structure
- Can trace back to source tiles
- Easier to understand data lineage

## Storage Impact

### Disk Space Considerations

**Typical Patch Sizes:**

- NPZ file: ~2-5 MB (compressed features + metadata)
- LAZ file: ~3-8 MB (compressed point cloud)
- **Total per patch:** ~5-13 MB

**Example Storage Requirements:**

- 1000 patches = ~5-13 GB
- 10000 patches = ~50-130 GB

**Recommendation:**

- Keep LAZ patches during development and validation
- Archive or delete LAZ patches after confirming quality
- Always retain NPZ files for training

## Usage

### Generating Patches with Dual Output

```bash
# LOD2
ign-lidar-hd process \
    --config-file configs/multiscale/config_unified_lod2_preprocessing.yaml

# LOD3
ign-lidar-hd process \
    --config-file configs/multiscale/config_unified_lod3_preprocessing.yaml
```

### Viewing LAZ Patches

**CloudCompare:**

```bash
cloudcompare.CloudCompare /mnt/c/Users/Simon/ign/preprocessed/lod2/patches/patch_0001.laz
```

**QGIS:**

1. Open QGIS
2. Add Vector Layer → Point Cloud
3. Browse to patch LAZ file

**Python (laspy):**

```python
import laspy

# Read LAZ patch
las = laspy.read('patch_0001.laz')

# Access point cloud data
points = np.vstack([las.x, las.y, las.z]).T
colors = np.vstack([las.red, las.green, las.blue]).T
classification = las.classification

print(f"Points: {len(points)}")
print(f"Classes: {np.unique(classification)}")
```

### Training with NPZ Files

```python
import numpy as np

# Load NPZ patch for training
data = np.load('patch_0001.npz')

points = data['points']      # (N, 3) XYZ coordinates
features = data['features']  # (N, F) feature vectors
labels = data['labels']      # (N,) class labels
```

## Cleanup Strategy

### After Validation Phase

```bash
# Keep NPZ files, remove LAZ files to save space
find /mnt/c/Users/Simon/ign/preprocessed/lod2/patches -name "*.laz" -delete
find /mnt/c/Users/Simon/ign/preprocessed/lod3/patches -name "*.laz" -delete
```

### Archive LAZ Patches

```bash
# Archive LAZ patches for future reference
tar -czf lod2_patches_laz_backup.tar.gz \
    /mnt/c/Users/Simon/ign/preprocessed/lod2/patches/*.laz

tar -czf lod3_patches_laz_backup.tar.gz \
    /mnt/c/Users/Simon/ign/preprocessed/lod3/patches/*.laz

# Then delete originals
find /mnt/c/Users/Simon/ign/preprocessed/*/patches -name "*.laz" -delete
```

## Configuration Summary

| Aspect          | ASPRS                     | LOD2                  | LOD3                  |
| --------------- | ------------------------- | --------------------- | --------------------- |
| **Output Mode** | Enriched LAZ tiles        | Patches (dual format) | Patches (dual format) |
| **NPZ Format**  | ❌ No                     | ✅ Yes                | ✅ Yes                |
| **LAZ Format**  | ✅ Yes (full tiles)       | ✅ Yes (patches)      | ✅ Yes (patches)      |
| **Patch Size**  | N/A                       | 100m × 100m           | 50m × 50m             |
| **Use Case**    | Visualization & base data | Training + QC         | Training + QC         |

---

**Last Updated:** October 15, 2025
