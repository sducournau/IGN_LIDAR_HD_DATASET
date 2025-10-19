# RGE ALTI WMS Caching Implementation

**Date:** October 19, 2025  
**Solution:** WMS-based DTM fetching with automatic caching

## Overview

Following IGN's migration to Géoplateforme, the WCS (Web Coverage Service) for programmatic DTM download is no longer available. This implementation uses **WMS (Web Map Service)** as an alternative, with automatic GeoTIFF caching for efficient repeated access.

## How It Works

### 1. Data Source Priority

The `RGEALTIFetcher` attempts to fetch DTM data in this order:

1. **Cache** (fastest) - Checks for previously downloaded tiles
2. **Local files** - Searches local DTM directory if specified
3. **WMS** - Downloads from IGN Géoplateforme and caches result

### 2. WMS GetMap Request

Instead of WCS GetCoverage, we use WMS GetMap:

```python
# WMS GetMap parameters
params = {
    'SERVICE': 'WMS',
    'VERSION': '1.3.0',
    'REQUEST': 'GetMap',
    'LAYERS': 'ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES',
    'FORMAT': 'image/geotiff',
    'BBOX': f'{minx},{miny},{maxx},{maxy}',
    'WIDTH': pixel_width,
    'HEIGHT': pixel_height,
    'CRS': 'EPSG:2154'
}
```

### 3. Automatic Caching

- Downloaded GeoTIFF is automatically saved to cache directory
- Cache files are named by bbox: `rge_alti_{crs}_{minx}_{miny}_{maxx}_{maxy}.tif`
- Subsequent requests for same bbox use cached file (no network request)

## Usage

### Basic Usage (with WMS caching)

```python
from ign_lidar.io.rge_alti_fetcher import RGEALTIFetcher

# Enable WMS with caching (default behavior)
fetcher = RGEALTIFetcher(
    cache_dir=".cache/ign_lidar/rge_alti",  # Where to cache tiles
    use_wcs=True,  # Actually enables WMS (parameter name kept for compatibility)
    resolution=1.0
)

# First call: Downloads from WMS and caches
bbox = (635000.0, 6856000.0, 636000.0, 6857000.0)
dtm_data = fetcher.fetch_dtm_for_bbox(bbox, crs="EPSG:2154")

# Second call: Uses cached file (instant)
dtm_data = fetcher.fetch_dtm_for_bbox(bbox, crs="EPSG:2154")
```

### With Local Files

```python
# Prefer local files, fallback to WMS with caching
fetcher = RGEALTIFetcher(
    local_dtm_dir="/path/to/rge_alti_tiles",
    cache_dir=".cache/ign_lidar/rge_alti",
    use_wcs=True,  # WMS fallback if local file not found
    resolution=1.0
)
```

### Cache-Only Mode

```python
# Only use cache (no WMS downloads)
fetcher = RGEALTIFetcher(
    cache_dir=".cache/ign_lidar/rge_alti",
    use_wcs=False,  # Disable WMS
    resolution=1.0
)
```

## Advantages of WMS Caching

### ✅ Benefits

1. **No manual downloads** - Automatic on-demand fetching
2. **Efficient caching** - Downloaded once, used many times
3. **Works immediately** - No need to pre-download large datasets
4. **Bandwidth friendly** - Only fetches what you need
5. **Transparent** - Same API as before

### ⚠️ Considerations

1. **Image size limits** - WMS has max image dimensions (~2048x2048)
   - Large areas are automatically downsampled
   - For best quality, use smaller bboxes or local files
2. **Network dependency** - First fetch requires internet
3. **Quality** - WMS may apply rendering/compression
   - For critical applications, use local GeoTIFF files

## Performance Comparison

| Method          | First Access     | Subsequent Access | Data Quality    |
| --------------- | ---------------- | ----------------- | --------------- |
| **WMS + Cache** | ~2-5 seconds     | <0.1 seconds      | Good (rendered) |
| **Local files** | ~0.1 seconds     | ~0.1 seconds      | Best (raw)      |
| **WCS (old)**   | ❌ Not available | ❌ Not available  | N/A             |

## Migration from WCS

### Old Code (WCS - broken)

```python
# This no longer works
fetcher = RGEALTIFetcher(use_wcs=True, api_key="pratique")
```

### New Code (WMS - working)

```python
# Same parameter name, but uses WMS internally
fetcher = RGEALTIFetcher(
    use_wcs=True,  # Actually enables WMS
    cache_dir=".cache/rge_alti"  # Recommended for caching
)
```

**No code changes needed!** The `use_wcs` parameter now enables WMS instead.

## Cache Management

### Cache Location

Default: `.cache/ign_lidar/rge_alti/`

Files are named: `rge_alti_2154_635000_6856000_636000_6857000.tif`

### Cache Size

- Each 1km² tile at 1m resolution ≈ 4-8 MB (compressed)
- 100 tiles ≈ 400-800 MB
- Cache grows as you process different areas

### Clearing Cache

```bash
# Clear all cached tiles
rm -rf .cache/ign_lidar/rge_alti/*.tif

# Clear specific bbox
rm .cache/ign_lidar/rge_alti/rge_alti_2154_635000_6856000_636000_6857000.tif
```

### Programmatic Cache Management

```python
from pathlib import Path
import shutil

# Clear entire cache
cache_dir = Path(".cache/ign_lidar/rge_alti")
if cache_dir.exists():
    shutil.rmtree(cache_dir)

# Or selectively remove old files
import time
max_age_days = 30
now = time.time()

for cache_file in cache_dir.glob("*.tif"):
    age_days = (now - cache_file.stat().st_mtime) / 86400
    if age_days > max_age_days:
        cache_file.unlink()
        print(f"Removed old cache file: {cache_file.name}")
```

## Configuration Examples

### In YAML Config

```yaml
rge_alti:
  cache_dir: ".cache/ign_lidar/rge_alti"
  use_wcs: true # Enables WMS
  resolution: 1.0
  local_dtm_dir: null # Optional: path to local files
```

### Environment-Specific

```python
import os

# Development: Use WMS with caching
if os.getenv('ENV') == 'development':
    fetcher = RGEALTIFetcher(
        cache_dir=".cache/rge_alti",
        use_wcs=True
    )

# Production: Use local files for reliability
else:
    fetcher = RGEALTIFetcher(
        local_dtm_dir="/data/rge_alti",
        cache_dir="/var/cache/rge_alti",
        use_wcs=False  # Disable WMS fallback
    )
```

## Troubleshooting

### Issue: WMS request fails

```
[WARNING] WMS fetch failed: HTTPError 400
```

**Causes:**

- Invalid bbox or CRS
- Image size too large
- Network connectivity issues

**Solutions:**

- Reduce bbox size
- Check CRS is EPSG:2154 (Lambert-93)
- Verify network connectivity to data.geopf.fr

### Issue: Cache not being used

```
[INFO] Fetching DTM from IGN WMS: ...  # Every time
```

**Causes:**

- Cache directory not specified
- Cache directory not writable
- Different bbox each time

**Solutions:**

- Set `cache_dir` parameter
- Check directory permissions
- Use consistent bbox values

### Issue: Low quality DTM

**Cause:** WMS rendering/compression

**Solution:** Download high-quality GeoTIFF files:

1. Visit https://geoservices.ign.fr/telechargement
2. Download RGE ALTI tiles
3. Use `local_dtm_dir` parameter

## Technical Details

### WMS Endpoint

```
URL: https://data.geopf.fr/wms-r/wms
Layer: ELEVATION.ELEVATIONGRIDCOVERAGE.HIGHRES
Format: image/geotiff
CRS: EPSG:2154 (Lambert-93)
```

### Pixel Size Calculation

```python
width = int((maxx - minx) / resolution)
height = int((maxy - miny) / resolution)

# Example: 1km² at 1m resolution
# width = 1000 / 1.0 = 1000 pixels
# height = 1000 / 1.0 = 1000 pixels
```

### Cache File Format

- **Format:** GeoTIFF (LZW compressed)
- **Bands:** 1 (elevation)
- **Data type:** Float32
- **NoData:** -9999.0
- **Metadata:** Geotransform, CRS, bounds

## Best Practices

1. **Enable caching** - Always set `cache_dir` for better performance
2. **Reasonable tile sizes** - Keep bbox < 2km² for best WMS quality
3. **Local files for production** - Download tiles for critical applications
4. **Monitor cache size** - Implement cleanup for long-running applications
5. **Network retry** - WMS failures are transient; consider retry logic

## Future Enhancements

Potential improvements:

- [ ] Tile-based caching (standard tile grid)
- [ ] Asynchronous downloads
- [ ] Cache expiration/LRU eviction
- [ ] Multi-threaded tile fetching
- [ ] Alternative data sources (EU-DEM, etc.)

## Resources

- **IGN Géoplateforme:** https://geoservices.ign.fr
- **WMS Documentation:** https://data.geopf.fr/wms-r/wms?REQUEST=GetCapabilities
- **RGE ALTI Info:** https://geoservices.ign.fr/rgealti
- **Download Portal:** https://geoservices.ign.fr/telechargement

---

_Last updated: October 19, 2025_
