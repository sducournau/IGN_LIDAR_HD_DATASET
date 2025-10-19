# LiDAR HD MNT as Default DTM Source

**Date**: January 2025  
**Status**: ✅ Implemented  
**Version**: 5.2+

## Overview

The DTM fetcher now uses **LiDAR HD MNT** as the default elevation source, providing the highest quality 1m resolution terrain data derived from LiDAR point clouds. This ensures optimal consistency when processing LiDAR HD datasets.

## What Changed

### 1. Default DTM Source

- **Previous**: RGE ALTI (5m or 1m resolution, mixed sources)
- **Current**: LiDAR HD MNT (1m resolution, pure LiDAR-derived)
- **Benefit**: Better quality and consistency for LiDAR processing workflows

### 2. Layer Selection

The fetcher now supports two WMS layers from IGN Géoplateforme:

| Layer                      | Resolution | Source                         | Coverage         | Use Case                        |
| -------------------------- | ---------- | ------------------------------ | ---------------- | ------------------------------- |
| **LiDAR HD MNT** (default) | 1m         | LiDAR point clouds             | Growing (France) | Best quality for LiDAR projects |
| **RGE ALTI** (fallback)    | 1m/5m      | Mixed (LiDAR + photogrammetry) | Complete France  | Broader coverage                |

### 3. API Changes

#### New Parameter

```python
RGEALTIFetcher(
    cache_dir=".cache/ign_lidar/rge_alti",
    resolution=1.0,
    use_wcs=True,  # Enables WMS (parameter name kept for compatibility)
    prefer_lidar_hd=True  # NEW: Use LiDAR HD MNT (default: True)
)
```

#### Automatic Layer Selection

```python
# Uses LiDAR HD MNT by default
fetcher = RGEALTIFetcher(cache_dir=".cache", prefer_lidar_hd=True)

# Fall back to RGE ALTI if needed (e.g., for areas without LiDAR HD coverage)
fetcher = RGEALTIFetcher(cache_dir=".cache", prefer_lidar_hd=False)
```

## Usage Examples

### Standard Configuration (Automatic)

```yaml
# config.yaml
preprocessing:
  ground_augmentation:
    enabled: true
    dtm_source: "ign_rge_alti" # Automatically uses LiDAR HD MNT
    cache_dir: ".cache/ign_lidar/rge_alti"
```

### Explicit Layer Selection

```python
from ign_lidar.io import RGEALTIFetcher

# Best quality - LiDAR HD MNT (default)
fetcher = RGEALTIFetcher(
    cache_dir=".cache/dtm",
    resolution=1.0,
    prefer_lidar_hd=True  # Default, can be omitted
)

# Broader coverage - RGE ALTI
fetcher = RGEALTIFetcher(
    cache_dir=".cache/dtm",
    resolution=1.0,
    prefer_lidar_hd=False  # Use if LiDAR HD not available in your area
)

# Fetch DTM for bounding box
dtm_array, transform = fetcher.fetch_dtm_for_bbox(
    bbox=(x_min, y_min, x_max, y_max),
    crs="EPSG:2154"
)
```

## Technical Details

### WMS Layer IDs

- **LiDAR HD MNT**: `IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.SHADOW`
- **RGE ALTI**: `ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES`

### WMS Endpoint

- **URL**: `https://data.geopf.fr/wms-r/wms`
- **Version**: WMS 1.3.0
- **Format**: `image/geotiff` (with LZW compression)
- **CRS**: EPSG:2154 (Lambert-93)

### Caching Strategy

- DTM tiles are automatically cached to avoid repeated downloads
- Cache key: `bbox_resolution_layer` (e.g., `dtm_650000_6860000_651000_6861000_1.0_lidar_hd.tif`)
- Cache location: `{cache_dir}/` (configurable)
- Automatic reuse on subsequent runs

### Performance Characteristics

- **First request**: ~2-5 seconds (WMS fetch + cache)
- **Cached request**: ~50-200ms (local file read)
- **Network usage**: Only on cache miss
- **Disk usage**: ~1-5 MB per 1km² tile (1m resolution)

## Migration Guide

### Existing Code

No changes needed! The new default is backward compatible:

```python
# This code works unchanged and now uses LiDAR HD MNT automatically
fetcher = RGEALTIFetcher(cache_dir=".cache")
dtm = fetcher.fetch_dtm_for_bbox(bbox, crs="EPSG:2154")
```

### Configuration Files

No changes needed! Existing configs automatically benefit:

```yaml
# Existing config - now uses LiDAR HD MNT by default
preprocessing:
  ground_augmentation:
    enabled: true
    dtm_source: "ign_rge_alti"
```

### Explicit RGE ALTI Selection

If you need RGE ALTI for specific reasons:

```python
# Option 1: Use prefer_lidar_hd parameter
fetcher = RGEALTIFetcher(cache_dir=".cache", prefer_lidar_hd=False)

# Option 2: Use local DTM files (bypasses WMS entirely)
fetcher = RGEALTIFetcher(local_dtm_dir="./dtm_files")
```

## Benefits

### 1. Quality Improvement

- **LiDAR HD MNT**: Derived directly from LiDAR point clouds (same data source as your input)
- **Consistency**: LiDAR-to-LiDAR matching eliminates source mismatches
- **Accuracy**: 1m resolution with high vertical accuracy

### 2. Processing Efficiency

- **Automatic caching**: No manual downloads needed
- **On-demand fetching**: Only downloads data for your processing extent
- **Fast reuse**: Cached tiles load in milliseconds

### 3. Coverage

- **Primary**: LiDAR HD MNT (growing coverage across France)
- **Fallback**: RGE ALTI (complete France coverage)
- **Flexibility**: Easy switch between sources with single parameter

## Logging

The fetcher now provides clear logging about which layer is being used:

```
INFO - DTM Fetcher initialized: LiDAR HD MNT (1m, best quality), resolution=1.0m, WMS=enabled, local_dir=None
INFO - DTM cache: /path/to/.cache/ign_lidar/rge_alti
INFO - Fetching DTM from IGN WMS: (650000, 6860000, 651000, 6861000) (1000x1000)
INFO - DTM fetched successfully: (1000, 1000) pixels, cached at dtm_650000_6860000_651000_6861000_1.0_lidar_hd.tif
```

## Troubleshooting

### Issue: LiDAR HD MNT not available in my area

**Solution**: Set `prefer_lidar_hd=False` to use RGE ALTI as fallback

```python
fetcher = RGEALTIFetcher(cache_dir=".cache", prefer_lidar_hd=False)
```

### Issue: WMS request fails

**Check**:

1. Internet connectivity
2. IGN Géoplateforme service status: https://data.geopf.fr
3. Verify your bbox coordinates are in EPSG:2154 (Lambert-93)
4. Check logs for specific error messages

### Issue: Need to force cache refresh

**Solution**: Delete cached tile and re-run

```bash
rm .cache/ign_lidar/rge_alti/dtm_*.tif
```

## References

- **IGN Géoplateforme**: https://data.geopf.fr
- **WMS Service**: https://data.geopf.fr/wms-r/wms
- **LiDAR HD Product Info**: https://geoservices.ign.fr/lidarhd
- **RGE ALTI Product Info**: https://geoservices.ign.fr/rgealti
- **Migration Notice**: [RGE_ALTI_MIGRATION_NOTICE.md](./RGE_ALTI_MIGRATION_NOTICE.md)
- **WMS Caching Details**: [RGE_ALTI_WMS_CACHING.md](./RGE_ALTI_WMS_CACHING.md)

## See Also

- Ground augmentation configuration: `examples/config_asprs_bdtopo_cadastre_optimized.yaml`
- WMS implementation: `ign_lidar/io/rge_alti_fetcher.py`
- Demo scripts: `examples/demo_variable_object_filtering.py`
