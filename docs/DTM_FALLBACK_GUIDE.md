# DTM Fallback Strategy - LiDAR HD MNT → RGE ALTI

**Date:** October 19, 2025  
**Version:** 5.2.1  
**Feature:** Automatic fallback between DTM sources

## Overview

When fetching digital terrain models (DTM/MNT) from IGN Géoplateforme, the system now implements an automatic fallback strategy to maximize data availability:

1. **Primary:** LiDAR HD MNT (1m resolution, best quality)
2. **Fallback:** RGE ALTI (1m-5m resolution, broader coverage)

This ensures that ground augmentation continues even when the preferred LiDAR HD MNT service is temporarily unavailable.

## Why Fallback is Needed

### LiDAR HD MNT Advantages

- **Highest quality:** Derived directly from LiDAR point clouds
- **1m resolution:** Best detail for terrain analysis
- **Perfect consistency:** Same data source as input point clouds
- **Preferred for LiDAR projects:** Ensures data coherence

### Common Issues

- **Service interruptions:** WMS may return 502 Bad Gateway errors
- **Coverage gaps:** LiDAR HD MNT not available for all areas
- **Maintenance windows:** Temporary service unavailability
- **High load:** Service may be slow during peak usage

### RGE ALTI Fallback Benefits

- **Broader coverage:** Available for all of France
- **Stable service:** Well-established infrastructure
- **Good quality:** 1m-5m resolution depending on area
- **Reliable:** Less prone to interruptions

## How It Works

### Automatic Fallback Sequence

```
1. Check cache (local stored tiles)
   └─ Found? → Use cached data
   └─ Not found? → Continue to step 2

2. Check local files (pre-downloaded tiles)
   └─ Found? → Use local data + cache for future
   └─ Not found? → Continue to step 3

3. Try WMS download - LiDAR HD MNT (primary)
   └─ Success? → Use data + cache for future
   └─ Failed? → Continue to step 4

4. Try WMS download - RGE ALTI (fallback)
   └─ Success? → Use data + cache for future
   └─ Failed? → Log error + skip ground augmentation
```

### What You'll See in Logs

#### Successful Primary Fetch (LiDAR HD MNT)

```
2025-10-19 23:45:39 - [INFO] Fetching DTM from IGN WMS (LiDAR HD MNT): (635000.0, 6856000.0, 636000.0, 6857000.0) (1000x1000)
2025-10-19 23:45:41 - [INFO] ✅ Successfully fetched DTM using LiDAR HD MNT
2025-10-19 23:45:41 - [INFO] ✅ Added 1,234,567 synthetic ground points from DTM
```

#### Successful Fallback (RGE ALTI)

```
2025-10-19 23:45:39 - [INFO] Fetching DTM from IGN WMS (LiDAR HD MNT): (635000.0, 6856000.0, 636000.0, 6857000.0) (1000x1000)
2025-10-19 23:45:41 - [WARNING] WMS fetch failed for LiDAR HD MNT: 502 Server Error: Bad Gateway
2025-10-19 23:45:41 - [INFO] Fetching DTM from IGN WMS (RGE ALTI (fallback)): (635000.0, 6856000.0, 636000.0, 6857000.0) (1000x1000)
2025-10-19 23:45:43 - [INFO] ✅ Successfully fetched DTM using RGE ALTI (fallback)
2025-10-19 23:45:43 - [INFO] ✅ Added 1,234,567 synthetic ground points from DTM
```

#### Complete Failure (All Sources)

```
2025-10-19 23:45:39 - [INFO] Fetching DTM from IGN WMS (LiDAR HD MNT): (635000.0, 6856000.0, 636000.0, 6857000.0) (1000x1000)
2025-10-19 23:45:41 - [WARNING] WMS fetch failed for LiDAR HD MNT: 502 Server Error: Bad Gateway
2025-10-19 23:45:41 - [INFO] Fetching DTM from IGN WMS (RGE ALTI (fallback)): (635000.0, 6856000.0, 636000.0, 6857000.0) (1000x1000)
2025-10-19 23:45:43 - [WARNING] WMS fetch failed for RGE ALTI (fallback): 502 Server Error: Bad Gateway
2025-10-19 23:45:43 - [ERROR] Failed to fetch DTM from all WMS layers for bbox (635000.0, 6856000.0, 636000.0, 6857000.0)
2025-10-19 23:45:43 - [WARNING] Failed to fetch DTM from all sources (cache, local, WMS)
2025-10-19 23:45:43 - [WARNING] Skipping ground augmentation - using existing LiDAR ground points only
2025-10-19 23:45:43 - [INFO] 💡 Tip: Check your internet connection or consider pre-downloading DTM tiles
2025-10-19 23:45:43 - [INFO] ℹ️  No ground points added (sufficient existing coverage)
```

## Configuration

### Default Settings (Recommended)

```yaml
data_sources:
  rge_alti:
    enabled: true
    use_wcs: true # Enables WMS download with fallback
    resolution: 1.0
    cache_enabled: true # Reduces WMS requests
```

The system automatically uses LiDAR HD MNT first, then falls back to RGE ALTI.

### Prefer RGE ALTI (Broader Coverage)

If you want to prioritize RGE ALTI over LiDAR HD MNT (e.g., for areas with limited LiDAR coverage):

```python
# In code (advanced users only)
fetcher = RGEALTIFetcher(
    prefer_lidar_hd=False,  # Try RGE ALTI first
    cache_dir=cache_dir,
    resolution=1.0
)
```

### Disable Fallback (Force Specific Source)

**Not recommended** - reduces reliability. Only use if you need specific data source:

```python
# This requires custom code modification
# Contact support if you need this feature
```

## Best Practices

### 1. Enable Caching (Essential)

Always enable caching to reduce WMS requests and improve reliability:

```yaml
data_sources:
  rge_alti:
    cache_enabled: true
    cache_dir: null # Auto-set to {input_dir}/cache/rge_alti
    cache_ttl_days: 90 # DTM data rarely changes
```

**Benefits:**

- Faster processing (5-10× speedup on repeated runs)
- Reduced network dependency
- Protection against service interruptions
- Lower server load

### 2. Pre-download DTM Tiles (Optional)

For large-scale processing or areas with poor connectivity, pre-download DTM tiles:

```bash
# Download DTM tiles for your area of interest
# (Contact IGN or use their download tools)
```

Configure local directory:

```yaml
data_sources:
  rge_alti:
    use_local: true
    local_dtm_dir: "/path/to/downloaded/tiles"
```

### 3. Monitor Fallback Usage

Check logs to understand which DTM source is being used:

```bash
# Count fallback occurrences in logs
grep "fallback" pipeline.log | wc -l

# Show successful LiDAR HD MNT fetches
grep "Successfully fetched DTM using LiDAR HD MNT" pipeline.log | wc -l

# Show successful RGE ALTI fallbacks
grep "Successfully fetched DTM using RGE ALTI" pipeline.log | wc -l
```

### 4. Handle Processing Interruptions

If WMS services are down for extended periods:

1. **Wait and retry:** Services usually recover within hours
2. **Use cached data:** Previous runs may have cached DTM for your area
3. **Pre-download:** Get DTM tiles from IGN for offline use
4. **Continue without augmentation:** Processing will work with existing ground points

## Quality Considerations

### LiDAR HD MNT vs RGE ALTI

| Aspect          | LiDAR HD MNT            | RGE ALTI                |
| --------------- | ----------------------- | ----------------------- |
| **Resolution**  | 1m (always)             | 1m-5m (varies)          |
| **Source**      | LiDAR point clouds      | Mixed (LiDAR + other)   |
| **Quality**     | Highest                 | Good                    |
| **Coverage**    | Limited (newer areas)   | Complete (all France)   |
| **Consistency** | Perfect with input data | Good                    |
| **Use case**    | Best for LiDAR projects | Fallback/broad coverage |

### When Fallback is Used

The fallback mechanism ensures **continuity** rather than perfection:

- ✅ **Still accurate:** RGE ALTI provides good quality DTM (typically ±0.15-0.30m)
- ✅ **Better than nothing:** Far superior to skipping ground augmentation
- ✅ **Transparent:** Logs clearly indicate which source was used
- ℹ️ **Quality tracking:** Metadata includes DTM source for analysis

### Output Metadata

DTM source is tracked in output files:

```json
{
  "dtm_source": "RGE ALTI (fallback)",
  "dtm_resolution": 1.0,
  "synthetic_ground_points": 1234567,
  "augmentation_strategy": "intelligent"
}
```

## Troubleshooting

### Issue: Both WMS sources fail

**Symptoms:**

```
[ERROR] Failed to fetch DTM from all WMS layers
[WARNING] Skipping ground augmentation
```

**Causes:**

- No internet connection
- IGN services down for maintenance
- Firewall blocking HTTPS requests
- Network timeout

**Solutions:**

1. Check internet connection: `curl https://data.geopf.fr/wms-r/wms`
2. Wait and retry (services usually recover quickly)
3. Use cached data from previous runs
4. Pre-download DTM tiles for offline use

### Issue: Slow WMS downloads

**Symptoms:**

- Long wait times (>60 seconds per tile)
- Frequent timeouts

**Solutions:**

1. **Enable caching** (first fetch is slow, subsequent ones are instant)
2. **Reduce resolution** (e.g., `resolution: 5.0` for faster downloads)
3. **Process during off-peak hours** (evenings/weekends)
4. **Pre-download tiles** for large-scale processing

### Issue: Inconsistent results (LiDAR HD vs RGE ALTI)

**Symptoms:**

- Different number of synthetic points on repeated runs
- Slight height differences

**Explanation:**

- This is **expected** when fallback occurs on different runs
- Caching prevents this (same source used consistently)

**Solutions:**

1. **Enable caching** (enforces consistent source)
2. **Pre-download tiles** (guarantees same data)
3. **Accept variation** (typically <5% difference in synthetic points)

## API Reference

### RGEALTIFetcher.**init**()

```python
def __init__(
    self,
    cache_dir: Optional[str] = None,
    resolution: float = 1.0,
    use_wcs: bool = True,  # Actually enables WMS (name kept for compatibility)
    local_dtm_dir: Optional[str] = None,
    api_key: Optional[str] = None,  # Legacy, not needed for WMS
    prefer_lidar_hd: bool = True  # Try LiDAR HD MNT first
)
```

**Parameters:**

- `prefer_lidar_hd`: If `True` (default), tries LiDAR HD MNT first, then RGE ALTI fallback
- `use_wcs`: If `True` (default), enables WMS downloads with automatic fallback

### Metadata in Output

```python
metadata = {
    'transform': ...,
    'crs': 'EPSG:2154',
    'resolution': (1.0, 1.0),
    'bounds': (minx, miny, maxx, maxy),
    'nodata': -9999.0,
    'source': 'LiDAR HD MNT'  # or 'RGE ALTI (fallback)'
}
```

## Examples

### Example 1: Standard Processing (Automatic Fallback)

```yaml
# config.yaml
data_sources:
  rge_alti:
    enabled: true
    use_wcs: true
    cache_enabled: true

ground_truth:
  rge_alti:
    augment_ground: true
    augmentation_strategy: "intelligent"
```

**Result:** Automatically uses best available DTM source.

### Example 2: Check DTM Source in Output

```python
import laspy
import json

# Read LAZ file
las = laspy.read("output/enriched.laz")

# Check metadata
with open("output/metadata.json") as f:
    metadata = json.load(f)
    dtm_source = metadata.get("dtm_source", "unknown")
    print(f"DTM source: {dtm_source}")

# Check for synthetic ground points
if hasattr(las, 'is_synthetic_ground'):
    n_synthetic = las.is_synthetic_ground.sum()
    print(f"Synthetic ground points: {n_synthetic:,}")
```

### Example 3: Monitor Fallback Rate

```python
import json
import glob
from pathlib import Path

# Analyze multiple tiles
results = {"lidar_hd": 0, "rge_alti": 0, "failed": 0}

for metadata_file in glob.glob("output/*/metadata.json"):
    with open(metadata_file) as f:
        meta = json.load(f)
        source = meta.get("dtm_source", "unknown")

        if "LiDAR HD" in source:
            results["lidar_hd"] += 1
        elif "RGE ALTI" in source or "fallback" in source.lower():
            results["rge_alti"] += 1
        else:
            results["failed"] += 1

total = sum(results.values())
print(f"DTM Source Statistics ({total} tiles):")
print(f"  LiDAR HD MNT:    {results['lidar_hd']:3d} ({100*results['lidar_hd']/total:.1f}%)")
print(f"  RGE ALTI:        {results['rge_alti']:3d} ({100*results['rge_alti']/total:.1f}%)")
print(f"  Failed/Skipped:  {results['failed']:3d} ({100*results['failed']/total:.1f}%)")
```

## Performance Impact

### Network Overhead

| Scenario                   | LiDAR HD MNT  | RGE ALTI Fallback | Total Time |
| -------------------------- | ------------- | ----------------- | ---------- |
| **Cache hit**              | 0s            | 0s                | <1s        |
| **Cache miss (success)**   | 1-3s          | 0s (not tried)    | 1-3s       |
| **Cache miss (fallback)**  | 3-5s (failed) | 2-4s (success)    | 5-9s       |
| **Cache miss (both fail)** | 3-5s (failed) | 3-5s (failed)     | 6-10s      |

**Recommendation:** Enable caching to minimize network overhead.

### Processing Impact

- **No impact on classification speed** (DTM fetched once per tile)
- **Slight increase in total time** if fallback occurs (5-10 seconds)
- **Cached runs identical** (no performance difference)

## Future Improvements

### Planned Enhancements

1. **Retry logic:** Automatic retry with exponential backoff
2. **Parallel fallback:** Try both sources simultaneously
3. **Quality validation:** Automatic quality checks on fetched DTM
4. **Smart caching:** Predictive pre-fetching for adjacent tiles
5. **Hybrid mode:** Blend LiDAR HD and RGE ALTI in overlap areas

### Community Contributions

We welcome contributions to improve DTM fallback:

- **Additional DTM sources:** Support for other elevation datasets
- **Improved error handling:** Better recovery strategies
- **Performance optimizations:** Faster downloads, better caching
- **Documentation:** Share your experiences and best practices

## Support

### Getting Help

- **GitHub Issues:** Report bugs or request features
- **Documentation:** See other guides in `docs/` folder
- **Community:** Share experiences with other users

### Reporting Issues

When reporting DTM-related issues, include:

1. **Log excerpt:** Relevant WMS request/response logs
2. **Configuration:** Your `data_sources.rge_alti` config
3. **Network:** `curl https://data.geopf.fr/wms-r/wms` output
4. **Bounding box:** Coordinates where issue occurred

## Related Documentation

- **[LIDAR_HD_MNT_DEFAULT.md](LIDAR_HD_MNT_DEFAULT.md):** Main DTM integration guide
- **[RGE_ALTI_IMPLEMENTATION_COMPLETE.md](RGE_ALTI_IMPLEMENTATION_COMPLETE.md):** Full RGE ALTI implementation details
- **[CONFIGURATION_UPDATES_V5.1.md](CONFIGURATION_UPDATES_V5.1.md):** Configuration reference
- **[QUICK_START_DEVELOPER.md](QUICK_START_DEVELOPER.md):** Developer setup guide

---

**Version History:**

- **v1.0 (Oct 19, 2025):** Initial documentation for automatic DTM fallback feature
