# LAZ Enrichment Guide

## 🎯 Overview

This guide explains how to enrich IGN LiDAR HD LAZ files with geometric features for machine learning applications.

## 📋 Prerequisites

1. **Python Environment**: Activate the virtual environment

   ```bash
   cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_downloader
   source .venv/bin/activate
   ```

2. **Package Installation**: Verify the package is installed

   ```bash
   python -m ign_lidar.cli --help
   ```

3. **Input Data**: Have raw LAZ tiles ready (e.g., from IGN downloads)

## 🚀 Quick Start

### Basic Enrichment (Core Features)

```bash
python -m ign_lidar.cli enrich \
  --input-dir /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --num-workers 4
```

### Building Mode (Full Features)

```bash
python -m ign_lidar.cli enrich \
  --input-dir /path/to/raw_tiles/ \
  --output /path/to/enriched_tiles/ \
  --mode building \
  --num-workers 6
```

## 📊 Feature Modes

### Core Mode (Default)

Adds essential geometric features:

- Surface normals (nx, ny, nz)
- Curvature
- Planarity
- Verticality

**Use case**: General LiDAR processing, faster computation

### Building Mode

Adds comprehensive features for building classification:

- All core features
- Height above ground
- Local density
- Horizontality
- Slope
- Additional geometric descriptors

**Use case**: Building LOD classification, ML training datasets

## 🎛️ Command Options

### Required Arguments

- `--output`: Output directory for enriched LAZ files (required)

### Input Options (choose one)

- `--input`: Process a single LAZ file
- `--input-dir`: Process all LAZ files in a directory

### Performance Options

- `--num-workers`: Number of parallel workers (default: 1)
  - Recommended: Number of CPU cores - 1
  - Example: `--num-workers 6` for 8-core system
- `--use-gpu`: Enable GPU acceleration (requires CUDA)
  - Significantly faster for large point clouds
  - Requires `requirements_gpu.txt` dependencies

### Feature Options

- `--mode`: Feature computation mode
  - `core`: Basic geometric features (faster)
  - `building`: Full building features (recommended for ML)
- `--k-neighbors`: Number of neighbors for feature computation (default: 10)
  - Higher values: Smoother features, slower computation
  - Lower values: More detail, faster computation

### Behavior Options

- `--force`: Force re-enrichment even if output exists

  - By default, existing enriched files are skipped
  - Use this to reprocess with different parameters

- `--auto-convert-qgis`: Also create QGIS-compatible LAZ 1.2 files
  - Useful if you need to view in QGIS
  - Creates additional files with `_qgis` suffix

## 💡 Examples

### Example 1: Basic Workflow

```bash
# Step 1: Enrich with core features (fast)
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --num-workers 4

# Step 2: Process into ML patches
python -m ign_lidar.cli process \
  --input-dir enriched/ \
  --output patches/ \
  --lod-level LOD2 \
  --num-workers 4
```

### Example 2: Building Classification Workflow

```bash
# Enrich with full building features
python -m ign_lidar.cli enrich \
  --input-dir /mnt/c/Users/Simon/ign/raw_tiles/ \
  --output /mnt/c/Users/Simon/ign/pre_tiles/ \
  --mode building \
  --num-workers 6 \
  --k-neighbors 15
```

### Example 3: GPU-Accelerated Processing

```bash
# Install GPU dependencies first
pip install -r requirements_gpu.txt

# Run with GPU acceleration
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --mode building \
  --use-gpu
```

### Example 4: Single File Processing

```bash
# Process one file for testing
python -m ign_lidar.cli enrich \
  --input raw_tiles/LHD_FXX_0186_6834_PTS_C_LAMB93_IGN69.laz \
  --output enriched/ \
  --mode building
```

### Example 5: Resume Interrupted Processing

```bash
# First run: Processes 50 files, stops at file 30
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --num-workers 4

# Second run: Automatically skips files 1-30, continues from 31
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --num-workers 4
```

### Example 6: Force Reprocessing

```bash
# Reprocess all files with different parameters
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --mode building \
  --k-neighbors 20 \
  --force
```

### Example 7: QGIS Visualization

```bash
# Create both LAZ 1.4 (full features) and LAZ 1.2 (QGIS compatible)
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --mode building \
  --auto-convert-qgis
```

## 📈 Performance Tips

### Optimal Worker Count

```python
import multiprocessing
optimal_workers = max(1, multiprocessing.cpu_count() - 1)
print(f"Recommended: --num-workers {optimal_workers}")
```

### Memory Considerations

- **Large files (>2GB)**: Use fewer workers (2-4)
- **Small files (<500MB)**: Use more workers (6-8)
- **Monitor memory**: Use `htop` or `top` to watch usage

### Speed Comparison

| Configuration            | Speed | Quality   |
| ------------------------ | ----- | --------- |
| Core mode, 1 worker      | 1x    | Good      |
| Core mode, 4 workers     | 3.5x  | Good      |
| Building mode, 1 worker  | 0.7x  | Excellent |
| Building mode, 4 workers | 2.5x  | Excellent |
| Building mode, GPU       | 5-10x | Excellent |

## 🔍 Output Format

### Output Structure

```
enriched/
├── LHD_FXX_0186_6834_enriched.laz       # LAZ 1.4 with all features
├── LHD_FXX_0186_6835_enriched.laz
└── LHD_FXX_0186_6836_enriched.laz
```

### With QGIS Conversion

```
enriched/
├── LHD_FXX_0186_6834_enriched.laz       # Full features (LAZ 1.4)
├── LHD_FXX_0186_6834_enriched_qgis.laz  # QGIS compatible (LAZ 1.2)
├── LHD_FXX_0186_6835_enriched.laz
└── LHD_FXX_0186_6835_enriched_qgis.laz
```

### Features Added

**Core Mode:**

- `normal_x`, `normal_y`, `normal_z`: Surface normals
- `curvature`: Principal curvature
- `planarity`: Planarity measure
- `verticality`: Verticality measure

**Building Mode (includes all core + additional):**

- `height_above_ground`: Height from ground level
- `density`: Local point density
- `horizontality`: Horizontal surface measure
- `slope`: Surface slope angle

## ⏭️ Smart Skip Detection

The enrichment process automatically skips files that have already been enriched:

```
Processing: 100%|████████████████████| 100/100 files

Statistics:
  ⏭️  Skipped: 70 files (already enriched)
  ✅ Success: 30 files (newly enriched)
  ❌ Failed: 0 files
```

### Benefits

1. **Resume workflows**: Safe to re-run after interruptions
2. **Incremental processing**: Add new tiles without reprocessing old ones
3. **Time savings**: Skip detection is instant (no file reading required)

### Override Skip Detection

Use `--force` to reprocess all files:

```bash
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --force
```

## 🐛 Troubleshooting

### Command Not Found

**Error:** `zsh: command not found: ign-lidar-process`

**Solution:** Use Python module syntax:

```bash
python -m ign_lidar.cli enrich --help
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'ign_lidar'`

**Solution:** Reinstall the package:

```bash
pip install -e .
```

### Memory Errors

**Error:** `MemoryError` or system slowdown

**Solution:** Reduce number of workers:

```bash
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --num-workers 2
```

### GPU Not Available

**Error:** CUDA errors or GPU not found

**Solution:** Fall back to CPU:

```bash
# Remove --use-gpu flag
python -m ign_lidar.cli enrich \
  --input-dir raw_tiles/ \
  --output enriched/ \
  --num-workers 4
```

### Permission Errors

**Error:** `PermissionError: [Errno 13]`

**Solution:** Check directory permissions:

```bash
# Ensure output directory is writable
chmod -R u+w /path/to/output/
```

## 📊 Monitoring Progress

### Real-time Statistics

The enrichment process shows:

- Progress bar with percentage
- Current file being processed
- Estimated time remaining
- Success/skip/failure counts

### Example Output

```
Enriching LAZ files...
Processing: 45%|████████▎          | 45/100 [15:23<18:45, 0.49 files/s]
Current: LHD_FXX_0186_6879_enriched.laz

Statistics so far:
  ⏭️  Skipped: 20 files
  ✅ Success: 25 files
  ❌ Failed: 0 files
```

## 🔗 Next Steps

After enrichment, you can:

1. **Process into ML patches**:

   ```bash
   python -m ign_lidar.cli process \
     --input-dir enriched/ \
     --output patches/ \
     --lod-level LOD2
   ```

2. **Visualize in QGIS**: Use `--auto-convert-qgis` files

3. **Train ML models**: Use the enriched LAZ files or processed patches

4. **Extract custom features**: Use the Python API for advanced workflows

## 📚 Related Documentation

- [Smart Skip Features](docs/features/SMART_SKIP_SUMMARY.md)
- [Output Format Preferences](docs/features/OUTPUT_FORMAT_PREFERENCES.md)
- [Memory Optimization](docs/reference/MEMORY_OPTIMIZATION.md)
- [QGIS Integration](docs/guides/QUICK_START_QGIS.md)

---

**Last Updated:** October 3, 2025  
**Version:** 1.1.0  
**Status:** Production Ready
