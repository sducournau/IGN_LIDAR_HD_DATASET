# GPU Chunked Processing Support - Implementation Summary

## Overview

Added automatic GPU chunked processing support for large LiDAR tiles when both CuPy and RAPIDS cuML are available. The system now intelligently switches between standard GPU processing and chunked GPU processing based on point cloud size.

## Changes Made

### 1. Core Processor (`ign_lidar/core/processor.py`)

#### Added Parameters:

- `use_gpu_chunked` (bool, default=True): Enable chunked GPU processing for large tiles
- `gpu_batch_size` (int, default=1,000,000): Batch size for GPU processing

#### Logic Enhancement:

The processor now:

1. **Checks point cloud size** before feature computation
2. **Automatically switches** between processing modes:
   - **Small tiles (<5M points)**: Uses standard GPU processing with batch_size=250,000
   - **Large tiles (>5M points)**: Uses GPU chunked processing with batch_size=1,000,000 (configurable)
3. **Graceful fallback**: Falls back to standard GPU or CPU if chunked processing fails

### 2. CLI Integration (`ign_lidar/cli/hydra_main.py`)

#### Updated processor initialization:

```python
processor = LiDARProcessor(
    # ... existing params ...
    use_gpu=cfg.processor.use_gpu,
    use_gpu_chunked=cfg.features.use_gpu_chunked,  # ← NEW
    gpu_batch_size=cfg.features.gpu_batch_size,      # ← NEW
    # ... more params ...
)
```

### 3. Configuration Schema (`ign_lidar/config/schema.py`)

The configuration was already present in `FeaturesConfig`:

```python
@dataclass
class FeaturesConfig:
    # ... existing fields ...
    gpu_batch_size: int = 1_000_000
    use_gpu_chunked: bool = True
```

These settings are available in all feature config YAML files:

- `configs/features/full.yaml`
- `configs/features/minimal.yaml`
- `configs/features/buildings.yaml`
- `configs/features/vegetation.yaml`
- `configs/features/pointnet.yaml`

## Processing Flow

### When GPU is enabled (`processor=gpu`):

```
START
  ↓
Check: num_points > 5,000,000 AND use_gpu_chunked=True?
  ↓
YES → Try GPU Chunked Processing
  ↓     ├─ CuPy + cuML available?
  ↓     │   YES → 🚀 Use GPUChunkedFeatureComputer
  ↓     │         - chunk_size = gpu_batch_size (1M)
  ↓     │         - compute_all_features_chunked()
  ↓     │         - Log: "🚀 Using GPU chunked processing (15,692,509 points, batch_size=1,000,000)"
  ↓     │
  ↓     └─ CuML/CuPy missing?
  ↓           NO → Fall back to Standard GPU
  ↓
NO → Standard GPU Processing
  ↓   - Uses GPUFeatureComputer (features_gpu.py)
  ↓   - batch_size = 250,000
  ↓   - Log: "🚀 Mode GPU activé (batch_size=250000)"
  ↓
Error/Fallback → CPU Processing
  - Uses compute_all_features_optimized (CPU)
```

## Expected Output

### Before (your current output):

```
[INFO] Using GPU acceleration for feature computation
🚀 Mode GPU activé (batch_size=250000)
```

### After (with 15M point tile):

```
[INFO] Using GPU acceleration for feature computation
[INFO] 🚀 Using GPU chunked processing (15,692,509 points, batch_size=1,000,000)
✓ CuPy disponible - GPU activé
✓ RAPIDS cuML disponible - GPU algorithms enabled
🚀 GPU chunked mode enabled with RAPIDS cuML (chunk_size=1,000,000, VRAM limit=X.XGB / Y.YGB total)
```

## Configuration Example

Your command already has the correct configuration:

```bash
ign-lidar-hd process \
  input_dir="/mnt/c/Users/Simon/ign/raw_tiles/urban_dense" \
  output_dir="/mnt/c/Users/Simon/ign/enriched_laz_only" \
  output=enriched_only \
  processor=gpu \
  features=full \
  preprocess=aggressive \
  stitching=disabled \
  features.use_rgb=true \
  features.use_infrared=true \
  features.compute_ndvi=true \
  processor.num_workers=1 \
  verbose=true
```

The system will now automatically:

1. ✅ Detect that `features.use_gpu_chunked=true` (default in `full.yaml`)
2. ✅ Detect that `features.gpu_batch_size=1000000` (default)
3. ✅ Check if CuPy and cuML are available
4. ✅ Use GPU chunked processing for your 15M point tile

## Testing

Run the test script to verify configuration:

```bash
python test_gpu_chunked.py
```

## Benefits

### 1. **Memory Efficiency**

- Large tiles (>5M points) no longer risk VRAM exhaustion
- Processes in chunks while maintaining accuracy

### 2. **Performance**

- Small tiles: Fast standard GPU processing
- Large tiles: Efficient chunked processing
- No manual intervention needed

### 3. **Automatic Switching**

- System chooses optimal method based on point count
- Transparent to the user

### 4. **Backward Compatible**

- Existing configurations continue to work
- Can disable with `features.use_gpu_chunked=false`

## Troubleshooting

### If you still see "batch_size=250000"

This is normal for small tiles (<5M points). For your 15.6M point tile, you should see the chunked version.

### If chunked processing is not triggered

Check the log output carefully. The system will log:

- Whether GPU is available
- Whether CuML is available
- The actual processing mode selected
- Number of points in the tile

### Force chunked processing for testing

Override the threshold temporarily in code or use a custom config:

```yaml
# custom_config.yaml
features:
  use_gpu_chunked: true
  gpu_batch_size: 500000 # Smaller chunks for testing
```

## Files Modified

1. `ign_lidar/core/processor.py`
   - Added `use_gpu_chunked` and `gpu_batch_size` parameters
   - Enhanced feature computation logic with automatic mode selection
2. `ign_lidar/cli/hydra_main.py`

   - Pass GPU chunked parameters from config to processor

3. `test_gpu_chunked.py` (new)
   - Test script to verify configuration

## Next Steps

1. Run your command again and observe the log output
2. Verify that "GPU chunked processing" message appears for large tiles
3. Check processing speed and VRAM usage
4. Adjust `features.gpu_batch_size` if needed based on your GPU VRAM

---

**Status**: ✅ **READY TO TEST**

The code changes are complete and the package has been reinstalled. Your next processing run should automatically use GPU chunked processing when appropriate.
