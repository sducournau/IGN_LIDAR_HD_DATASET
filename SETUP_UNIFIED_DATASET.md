# Setting Up Your Unified Dataset

## Problem

Your pipeline failed because the unified dataset at `/mnt/d/ign/unified_dataset` is empty or improperly structured.

## Required Structure

```
unified_dataset/
├── asprs/
│   └── tiles/           # Raw ASPRS classified LiDAR tiles (.laz files)
├── lod2/
│   └── tiles/           # LOD2 building footprints/ground truth
└── lod3/
    └── tiles/           # LOD3 detailed building models
```

## Solution Options

### Option 1: Download Tiles from IGN (Recommended for New Data)

#### Step 1: Download Raw LiDAR Tiles

```bash
# Download ASPRS tiles for a specific region (e.g., Versailles)
ign-lidar-hd download \
    --bbox 639000,6846000,642000,6849000 \
    --output /mnt/d/ign/unified_dataset/asprs/tiles \
    --max-tiles 10
```

#### Step 2: Verify Downloads

```bash
# Check how many tiles were downloaded
ls -lh /mnt/d/ign/unified_dataset/asprs/tiles/*.laz | wc -l

# List first few tiles
ls /mnt/d/ign/unified_dataset/asprs/tiles/*.laz | head -5
```

#### Step 3: Run Analysis

```bash
# Analyze the unified dataset
python scripts/analyze_unified_dataset.py \
    --input /mnt/d/ign/unified_dataset \
    --output /mnt/d/ign/analysis_report.json
```

### Option 2: Use Existing Enriched Tiles (If You Already Have Processed Data)

If you already have preprocessed/enriched tiles (like in `/mnt/d/ign/enriched_tiles`), you can:

#### Create symlinks to avoid duplication:

```bash
# Copy enriched tiles back to unified dataset as "raw" tiles
# (They can be reprocessed or used as-is)
mkdir -p /mnt/d/ign/unified_dataset/asprs/tiles
cp /mnt/d/ign/enriched_tiles/*.laz /mnt/d/ign/unified_dataset/asprs/tiles/
```

### Option 3: Download Using the CLI Tool

```bash
# Interactive download - will prompt for region selection
ign-lidar-hd download-interactive \
    --output /mnt/d/ign/unified_dataset/asprs/tiles \
    --max-tiles 20
```

## Quick Start: Minimal Working Example

For testing, download just a few tiles:

```bash
# 1. Create structure
mkdir -p /mnt/d/ign/unified_dataset/{asprs,lod2,lod3}/tiles

# 2. Download 5 tiles from Versailles region (known good area)
ign-lidar-hd download \
    --bbox 639000,6846000,640000,6847000 \
    --output /mnt/d/ign/unified_dataset/asprs/tiles \
    --max-tiles 5

# 3. Verify
echo "Downloaded tiles:"
ls -lh /mnt/d/ign/unified_dataset/asprs/tiles/*.laz

# 4. Run the pipeline
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
./scripts/run_complete_pipeline.sh \
    --unified-dataset /mnt/d/ign/unified_dataset \
    --output-base /mnt/d/ign \
    --phases all \
    --gpu
```

## About LOD2 and LOD3

**Important**: LOD2 and LOD3 tiles are **ground truth labels**, not raw LiDAR:

- **ASPRS tiles**: Raw LiDAR point clouds (you download these)
- **LOD2 tiles**: Building footprints from BD TOPO (auto-generated during preprocessing)
- **LOD3 tiles**: Detailed building models (auto-generated during preprocessing)

**You only need to populate `asprs/tiles/`**. The LOD2/LOD3 data is generated automatically by the preprocessing step using BD TOPO (French topographic database).

## Troubleshooting

### "No tiles found in analysis"

- Make sure you have `.laz` files in `asprs/tiles/`
- Check file permissions: `chmod 644 /mnt/d/ign/unified_dataset/asprs/tiles/*.laz`

### "Directory not found"

- Verify the path exists: `ls -la /mnt/d/ign/unified_dataset/asprs/tiles`
- Check for typos in the path

### "Download failed"

- Check internet connection
- Verify IGN WFS service is available
- Try a different bbox (some areas may not have data)

### "Out of space"

- Each tile is ~100-500MB
- 100 tiles = ~50GB
- Make sure you have enough space on C: drive

## Next Steps

After populating the unified dataset:

1. ✅ Run analysis to verify: `python scripts/analyze_unified_dataset.py ...`
2. ✅ Run the complete pipeline: `./scripts/run_complete_pipeline.sh --phases all --gpu`
3. ✅ Monitor progress in `/mnt/d/ign/logs/`

## Alternative: Small Test Dataset

If you want to test quickly without downloading:

```bash
# Use the test integration data (very small)
./scripts/run_complete_pipeline.sh \
    --unified-dataset data/test_integration \
    --output-base /tmp/test_pipeline \
    --phases all
```
