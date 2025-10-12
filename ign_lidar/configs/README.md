# Configuration Files Guide

## Overview

This directory contains Hydra configuration files for the IGN LiDAR HD processing pipeline. Configurations are organized by component and can be composed using Hydra's defaults mechanism.

## Directory Structure

```
configs/
├── config.yaml                 # Root configuration
├── processor/                  # Processing configurations
│   ├── default.yaml           # CPU processing
│   ├── gpu.yaml               # GPU processing
│   ├── cpu_fast.yaml          # Fast CPU processing
│   └── memory_constrained.yaml # Low memory mode
├── features/                   # Feature extraction configs
│   ├── full.yaml              # All features
│   ├── minimal.yaml           # Basic features
│   ├── buildings.yaml         # Building detection
│   ├── vegetation.yaml        # Vegetation analysis
│   └── pointnet.yaml          # PointNet features
├── stitching/                  # Tile stitching configs
│   ├── disabled.yaml          # No stitching
│   ├── enabled.yaml           # Basic stitching
│   ├── enhanced.yaml          # Advanced stitching
│   ├── advanced.yaml          # Research-grade stitching
│   └── auto_download.yaml     # 🆕 Auto-download neighbors
├── preprocess/                 # Preprocessing configs
├── output/                     # Output format configs
└── experiment/                 # Full experiment configs
    ├── fast.yaml
    ├── pointnet_training.yaml
    ├── boundary_aware_autodownload.yaml  # 🆕 With downloads
    └── boundary_aware_offline.yaml       # 🆕 Without downloads
```

## Tile Stitching Configurations

### 🆕 New Feature: Auto-Download Neighbors

The stitching system now supports **automatic downloading of missing adjacent tiles** from IGN WFS service.

#### Configuration Files

##### 1. `stitching/disabled.yaml`

```yaml
enabled: false
```

**Use when:** No boundary processing needed, fastest processing

##### 2. `stitching/enabled.yaml`

```yaml
enabled: true
buffer_size: 10.0
auto_detect_neighbors: true
auto_download_neighbors: false # Default: disabled
cache_enabled: true
```

**Use when:** Basic boundary processing with local tiles only

##### 3. `stitching/enhanced.yaml`

```yaml
enabled: true
buffer_size: 15.0
auto_detect_neighbors: true
auto_download_neighbors: false # Can be overridden
adaptive_buffer: true
boundary_smoothing: true
parallel_loading: true
# ... many advanced options
```

**Use when:** Production processing with advanced quality settings

##### 4. `stitching/auto_download.yaml` 🆕

```yaml
enabled: true
buffer_size: 15.0
auto_detect_neighbors: true
auto_download_neighbors: true # ⚡ Downloads enabled!
validate_tiles: true
download_max_concurrent: 2
# ... download-specific settings
```

**Use when:**

- Processing tiles without pre-downloaded neighbors
- Automatic recovery from corrupted tiles
- Exploratory analysis

## Usage Examples

### Example 1: Basic Processing (No Stitching)

```bash
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  stitching=disabled
```

### Example 2: Boundary-Aware with Local Tiles

```bash
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  stitching=enhanced \
  processor.use_stitching=true
```

### Example 3: Auto-Download Missing Neighbors 🆕

```bash
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  stitching=auto_download \
  processor.use_stitching=true
```

### Example 4: Override Auto-Download on Enhanced Config

```bash
python -m ign_lidar.cli.process \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output \
  stitching=enhanced \
  stitching.auto_download_neighbors=true  # Override to enable
```

### Example 5: Use Experiment Config

```bash
python -m ign_lidar.cli.process \
  --config-name boundary_aware_autodownload \
  input_dir=/path/to/tiles \
  output_dir=/path/to/output
```

## Configuration Override Examples

### Enable Auto-Download on Any Config

```bash
# Add to any command:
stitching.auto_download_neighbors=true
```

### Adjust Buffer Size

```bash
# Larger buffer for better accuracy:
stitching.buffer_size=20.0
```

### Change Download Concurrency

```bash
# Download up to 4 tiles in parallel:
stitching.download_max_concurrent=4
```

### Disable Validation (Not Recommended)

```bash
stitching.validate_tiles=false
```

## Auto-Download Behavior

### When Enabled (`auto_download_neighbors: true`)

1. **Load core tile**
2. **Detect adjacent tiles** using bounding box checks
3. **For each expected neighbor position (N, S, E, W, NE, NW, SE, SW):**
   - Check if tile exists locally
   - If exists → Validate tile integrity
   - If valid → Use existing tile
   - If corrupted → Delete and download fresh copy
   - If missing → Query IGN WFS and download
4. **Validate all downloaded tiles**
5. **Extract buffer zones** (15m by default)
6. **Compute boundary-aware features**

### Validation Checks

Each tile (existing or downloaded) is validated:

- ✓ File exists
- ✓ File size > 1 MB
- ✓ Can be opened with laspy
- ✓ Contains points (not empty)
- ✓ Coordinates are valid (not NaN/zeros)

### Storage Requirements

- **Per tile:** ~100-300 MB
- **8 neighbors:** ~1-2 GB
- Downloaded tiles are **cached** and **reused** across processing runs
- Only **missing or corrupted** tiles are downloaded

### Network Requirements

- **Internet connection** required for WFS queries
- **IGN WFS service** must be accessible
- **Download speed:** Typically 1-2 MB/s
- **First tile:** 10-20 minutes (downloads neighbors)
- **Subsequent tiles:** <1 minute (reuses neighbors)

## Configuration Composition

### Using Defaults

```yaml
# In your config file:
defaults:
  - override /processor: gpu
  - override /features: full
  - override /stitching: auto_download # 🆕
  - override /output: default
```

### Programmatic Composition

```python
from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(config_path="configs"):
    cfg = compose(
        config_name="config",
        overrides=[
            "stitching=auto_download",
            "processor.use_gpu=true",
            "stitching.buffer_size=20.0"
        ]
    )
```

## Best Practices

### 1. For Production Pipelines

```yaml
# Use enhanced config, disable auto-download
stitching: enhanced
stitching.auto_download_neighbors: false
# Pre-download all needed tiles separately
# Then process offline
```

### 2. For Exploratory Analysis

```yaml
# Use auto-download for convenience
stitching: auto_download
stitching.buffer_size: 15.0
```

### 3. For Research/Testing

```yaml
# Use enhanced with custom settings
stitching: enhanced
stitching.auto_download_neighbors: true # Enable on demand
stitching.save_stitching_stats: true # Monitor performance
stitching.verbose_logging: true # Detailed logs
```

### 4. For Offline Processing

```yaml
# Disable auto-download explicitly
stitching: enhanced
stitching.auto_download_neighbors: false
```

## Troubleshooting

### Auto-Download Not Working?

**Check these settings:**

```bash
# Verify stitching is enabled:
stitching.enabled=true

# Verify auto-detection is enabled:
stitching.auto_detect_neighbors=true

# Verify auto-download is enabled:
stitching.auto_download_neighbors=true

# Verify processor uses stitching:
processor.use_stitching=true
```

### Downloads Failing?

1. **Check network connection**
2. **Verify IGN WFS is accessible**: https://data.geopf.fr/wfs
3. **Check logs** for specific error messages
4. **Try with fewer concurrent downloads**: `stitching.download_max_concurrent=1`

### Corrupted Tiles?

System will automatically:

- Detect corrupted tiles via validation
- Delete corrupted files
- Re-download fresh copies

Or manually delete and re-run:

```bash
rm /path/to/corrupted_tile.laz
# Re-run processing - will auto-download
```

## Related Documentation

- [Main Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- [Configuration Reference](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/api/configuration)
- [Hydra Documentation](https://hydra.cc/) - Configuration framework

## Configuration Schema

### Stitching Configuration

```yaml
enabled: bool # Enable tile stitching
buffer_size: float # Buffer zone width in meters
auto_detect_neighbors: bool # Auto-detect adjacent tiles
auto_download_neighbors: bool # 🆕 Download missing neighbors
validate_tiles: bool # Validate tile integrity
download_max_concurrent: int # Max concurrent downloads
cache_enabled: bool # Cache loaded tiles
parallel_loading: bool # Load neighbors in parallel
boundary_smoothing: bool # Smooth at boundaries
edge_artifact_removal: bool # Remove edge artifacts
compute_boundary_features: bool # Special boundary features
verbose_logging: bool # Detailed logging
save_stitching_stats: bool # Save statistics
```

## Version History

- **v2.4.2** (Oct 2025): Complete GPU acceleration for all advanced features
- **v2.3.0**: Processing modes and YAML configurations
- **v2.1**: Auto-download neighbors feature
- **v2.0**: Enhanced tile stitching with bbox-based detection

---

*Configuration guide for IGN LiDAR HD v2.4.2*
