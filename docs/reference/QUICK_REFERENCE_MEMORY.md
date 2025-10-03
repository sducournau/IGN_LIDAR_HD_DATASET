# Quick Reference: Memory-Optimized Enrichment with Directory Preservation

## TL;DR

Run your enrichment command as usual - the system will automatically handle memory management and preserve directory structure:

```bash
python -m ign_lidar.cli enrich \
  --input /mnt/c/Users/Simon/ign/raw_tiles/ \
  --output /mnt/c/Users/Simon/ign/pre_tiles/ \
  --k-neighbors 20 \
  --mode building \
  --num-workers 2
```

## What's New?

### 🛡️ Automatic Memory Protection

- ✅ Checks system memory before starting
- ✅ Detects swap usage and reduces workers
- ✅ Processes large files sequentially to prevent OOM
- ✅ Better memory estimation (150 bytes/point for building mode)
- ✅ Aggressive chunking for large point clouds

### 📁 Directory Structure Preservation

- ✅ Preserves all subdirectories from input
- ✅ Copies `stats.json` and other root-level metadata
- ✅ Copies per-file JSON metadata (`LHD_*.json`)
- ✅ Output directory mirrors input exactly

## Expected Behavior

### System Will Auto-Adjust:

**If swap usage > 50%:**

```
⚠️  High swap usage detected (65%)
⚠️  System is under memory pressure - reducing workers to 1
```

**If limited RAM:**

```
⚠️  Limited RAM (10.2GB available)
⚠️  Reducing workers from 4 to 2
```

**If large files detected:**

```
⚠️  Large files detected (max: 480MB)
⚠️  Reducing workers from 4 to 3 to prevent OOM
⚠️  Using sequential batching for large files to prevent memory issues
```

### Directory Structure:

**Before (Input):**

```
raw_tiles/
├── stats.json
├── coastal_residential/
│   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.laz
│   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.json
│   └── ...
├── urban_dense/
│   ├── LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.laz
│   └── ...
└── ...
```

**After (Output):**

```
pre_tiles/
├── stats.json                          ← COPIED
├── coastal_residential/                ← PRESERVED
│   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.laz  ← ENRICHED
│   ├── LHD_FXX_0331_6276_PTS_C_LAMB93_IGN69.json ← COPIED
│   └── ...
├── urban_dense/                        ← PRESERVED
│   ├── LHD_FXX_0889_6252_PTS_C_LAMB93_IGN69.laz  ← ENRICHED
│   └── ...
└── ...
```

## Validate Output

After enrichment completes, validate the structure:

```bash
python scripts/validation/validate_structure.py \
  /mnt/c/Users/Simon/ign/raw_tiles \
  /mnt/c/Users/Simon/ign/pre_tiles
```

This will check:

- ✓ All subdirectories preserved
- ✓ All metadata files copied
- ✓ All LAZ files processed
- ✓ Progress statistics

## Troubleshooting

### Still Getting OOM Errors?

**Option 1: Force single worker (safest)**

```bash
python -m ign_lidar.cli enrich \
  --input /mnt/c/Users/Simon/ign/raw_tiles/ \
  --output /mnt/c/Users/Simon/ign/pre_tiles/ \
  --k-neighbors 20 \
  --mode building \
  --num-workers 1
```

**Option 2: Free up swap space**

```bash
# Check current swap usage
free -h

# Clear swap (requires sudo)
sudo swapoff -a && sudo swapon -a
```

**Option 3: Process subdirectories individually**

```bash
# Process one category at a time
for dir in /mnt/c/Users/Simon/ign/raw_tiles/*/; do
  dirname=$(basename "$dir")
  python -m ign_lidar.cli enrich \
    --input "$dir" \
    --output "/mnt/c/Users/Simon/ign/pre_tiles/$dirname" \
    --k-neighbors 20 \
    --mode building \
    --num-workers 1
done

# Copy root-level metadata manually
cp /mnt/c/Users/Simon/ign/raw_tiles/*.json \
   /mnt/c/Users/Simon/ign/pre_tiles/
```

### Metadata Files Not Copied?

The system copies:

- All `.json` files in root directory
- All `.txt` files in root directory
- Per-file `.json` metadata (same name as LAZ file)

If metadata is in a different format or location, you may need to copy it manually:

```bash
# Copy all metadata
rsync -av --include='*/' --include='*.json' --include='*.txt' \
  --exclude='*.laz' \
  /mnt/c/Users/Simon/ign/raw_tiles/ \
  /mnt/c/Users/Simon/ign/pre_tiles/
```

## Memory Requirements

### Building Mode (with extra features)

| File Size | Points       | RAM per Worker | Max Workers (32GB) |
| --------- | ------------ | -------------- | ------------------ |
| < 100MB   | < 2M points  | ~300MB         | 8+                 |
| 100-200MB | 2-4M points  | ~600MB         | 4                  |
| 200-300MB | 4-6M points  | ~900MB         | 2-3                |
| 300-500MB | 6-10M points | ~1.5GB         | 1-2                |
| > 500MB   | > 10M points | ~3GB+          | 1                  |

### Core Mode (basic features only)

Uses ~50% less memory than building mode.

## Files Modified

1. **`ign_lidar/cli.py`**

   - Pre-flight memory checks
   - Directory structure preservation
   - Metadata file copying
   - Sequential batching logic
   - Improved memory estimation

2. **`ign_lidar/features.py`**

   - Better chunking in `compute_num_points_in_radius`
   - Explicit KDTree cleanup
   - Memory-efficient processing

3. **New: `scripts/validation/validate_structure.py`**

   - Validation tool for output structure

4. **New: `docs/MEMORY_OPTIMIZATION_V2.md`**
   - Full technical documentation

## Related Documentation

- **Full Details:** `docs/MEMORY_OPTIMIZATION_V2.md`
- **Previous Fixes:** `docs/MEMORY_OPTIMIZATION_FIXES.md`
- **Building Mode:** `docs/BUILDING_MODE_BUG_FIXES.md`
