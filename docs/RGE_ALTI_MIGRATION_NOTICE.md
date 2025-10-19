# RGE ALTI Service Migration Notice

**Date:** October 19, 2025  
**Impact:** High - WCS service discontinued

## Summary

IGN has migrated from the old Géoservices platform to the new **Géoplateforme** (`data.geopf.fr`). As part of this migration, the **WCS (Web Coverage Service)** for programmatic DTM/DEM download is **no longer available**.

## What Changed

### Old Service (Deprecated)

- **Endpoint:** `https://wxs.ign.fr/altimetrie/geoportail/r/wcs`
- **Protocol:** WCS 2.0.1
- **Status:** ❌ No longer functional
- **Error:** DNS resolution failure / service not available

### New Géoplateforme

- **Base URL:** `https://data.geopf.fr/`
- **Available Services:**
  - ✅ WMTS (Web Map Tile Service) - visualization only
  - ✅ WMS (Web Map Service) - images only
  - ✅ WFS (Web Feature Service) - vector data only
  - ❌ **WCS NOT AVAILABLE** - no programmatic DTM download

## Impact on RGE ALTI Fetcher

The `RGEALTIFetcher` class previously supported three modes:

1. ✅ **Local files** - Still works
2. ✅ **Cache** - Still works
3. ❌ **WCS download** - No longer works

## Migration Guide

### For Users

#### Option 1: Download RGE ALTI Files Manually (Recommended)

1. Visit IGN download portal: https://geoservices.ign.fr/telechargement
2. Navigate to **Altimetry** → **RGE ALTI**
3. Download the tiles covering your area of interest
4. Store them in a local directory
5. Use the `local_dtm_dir` parameter:

```python
from ign_lidar.io.rge_alti_fetcher import RGEALTIFetcher

fetcher = RGEALTIFetcher(
    local_dtm_dir="/path/to/rge_alti_files",
    cache_dir=".cache/ign_lidar/rge_alti",
    resolution=1.0
)
```

#### Option 2: Use Cache from Previous Downloads

If you previously used WCS and have cached files:

```python
fetcher = RGEALTIFetcher(
    cache_dir=".cache/ign_lidar/rge_alti",  # Existing cache
    resolution=1.0
)
```

### For Developers

#### Update Configuration

```yaml
# Old configuration (deprecated)
rge_alti:
  use_wcs: true
  api_key: "your_key"

# New configuration (recommended)
rge_alti:
  local_dtm_dir: "/path/to/rge_alti_tiles"
  cache_dir: ".cache/rge_alti"
  resolution: 1.0
```

#### Code Updates

```python
# Old code (no longer works)
fetcher = RGEALTIFetcher(use_wcs=True, api_key="pratique")

# New code (works)
fetcher = RGEALTIFetcher(local_dtm_dir="/path/to/dtm_files")
```

## Alternative Data Sources

### 1. LiDAR HD MNT (Preferred for LiDAR projects)

- Higher resolution (1m from LiDAR)
- Better quality for recent areas
- Download: https://geoservices.ign.fr/lidarhd

### 2. BD ALTI

- Lower resolution (25m)
- Complete national coverage
- Download: https://geoservices.ign.fr/bdalti

### 3. EU-DEM / Copernicus

- European coverage
- Free access
- Lower resolution (25m-30m)

## Error Messages

### Common Errors After Migration

```
[WARNING] WCS fetch failed: Failed to resolve 'wxs.ign.fr'
```

**Cause:** Old WCS endpoint no longer exists  
**Solution:** Download files manually and use `local_dtm_dir`

```
[WARNING] No local DTM directory specified. RGE ALTI augmentation will not work.
```

**Cause:** No DTM source configured  
**Solution:** Set `local_dtm_dir` parameter

## Resources

- **Géoplateforme Documentation:** https://geoservices.ign.fr/documentation
- **Altimetry Services:** https://geoservices.ign.fr/services-web-experts-altimetrie
- **Download Portal:** https://geoservices.ign.fr/telechargement
- **Contact IGN:** https://geoservices.ign.fr/contact

## Timeline

- **Before October 2025:** WCS service operational
- **October 2025:** Migration to Géoplateforme, WCS discontinued
- **After October 2025:** Use local files only

## FAQ

**Q: Will WCS be restored?**  
A: No indication from IGN. The new architecture focuses on visualization services (WMTS/WMS) rather than data download services (WCS).

**Q: Can I automate DTM download?**  
A: Not via WCS. Consider using:

- WFS to identify tile extents
- Direct HTTP download of pre-tiled files
- Contact IGN for bulk download options

**Q: What about API access?**  
A: The new Géoplateforme has API keys for visualization services, but not for WCS since the service doesn't exist.

## Support

For issues related to:

- **IGN services:** https://geoservices.ign.fr/contact
- **This package:** Open an issue on GitHub

---

_Last updated: October 19, 2025_
