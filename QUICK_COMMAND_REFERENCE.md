# Quick Command Reference - Enriched LAZ Only Mode

## Your Updated Command (Ready to Use)

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/enriched_laz_only" \
  output=enriched_only \
  processor=gpu \
  features=full \
  preprocess=aggressive \
  stitching=auto_download \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  num_workers=8 \
  verbose=true
```

## What This Does

✅ **Auto-downloads** missing neighbor tiles from IGN WFS  
✅ **Computes** geometric features (normals, curvature, height)  
✅ **Adds** RGB from IGN orthophotos  
✅ **Adds** NIR from IRC imagery  
✅ **Computes** NDVI vegetation index  
✅ **Saves** enriched LAZ files with all features  
❌ **Skips** patch creation (3-5x faster)  

## Output Location

Enriched LAZ files will be saved to:
```
/mnt/c/Users/Simon/ign/enriched_laz_only/enriched/
```

## Key Changes from Original

| Original | Updated | Why |
|----------|---------|-----|
| `output.save_enriched_laz=true` | `output=enriched_only` | Uses preset config |
| Creates patches + enriched LAZ | Only enriched LAZ | Faster, cleaner output |
| No auto-download | `stitching=auto_download` | Automatic neighbor tiles |
| `processor.augment=true` | Removed | No augmentation needed |
| `processor.num_augmentations=5` | Removed | No augmentation needed |

## Alternative: Manual Config

If you prefer manual configuration:

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/enriched_laz_only" \
  output.save_enriched_laz=true \
  output.only_enriched_laz=true \
  processor=gpu \
  features=full \
  preprocess=aggressive \
  stitching.enabled=true \
  stitching.auto_download_neighbors=true \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  num_workers=8 \
  verbose=true
```

## Performance Expectations

- **Speed**: ~3-5x faster than creating patches
- **Example**: 1 km² tile processes in ~25 seconds (vs ~120 seconds with patches)
- **Storage**: Single enriched LAZ per tile (vs dozens/hundreds of patches)

## Verify Installation

Before running, ensure you're in the correct environment:

```bash
conda activate ign_gpu
ign-lidar-hd --help
```
