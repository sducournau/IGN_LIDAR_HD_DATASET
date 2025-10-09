---
sidebar_position: 8
---

<!-- 🇫🇷 TRADUCTION FRANÇAISE REQUISE -->
<!-- Ce fichier est un modèle qui nécessite une traduction manuelle. -->
<!-- Veuillez traduire le contenu ci-dessous en conservant : -->
<!-- - Le frontmatter (métadonnées en haut) -->
<!-- - Les blocs de code (traduire uniquement les commentaires) -->
<!-- - Les liens et chemins de fichiers -->
<!-- - La structure Markdown -->



# Unified Processing Pipeline

IGN LiDAR HD v2.0+ introduces a **unified processing pipeline** that combines RAW LAZ preprocessing and patch extraction into a single, optimized workflow.

## Overview

### Old Pipeline (v1.x)

```
RAW LAZ → Enriched LAZ → Patches
  ↓          ↓            ↓
 Disk      Disk         Disk
```

**Issues:**

- Multiple disk I/O operations
- Intermediate files require storage
- Slower overall processing
- Manual two-step workflow

### New Pipeline (v2.0+)

```
RAW LAZ → [In-Memory Processing] → Patches
  ↓                                  ↓
 Disk                              Disk
```

**Benefits:**

- Single-step workflow
- In-memory processing
- 35-50% space savings (no intermediate files)
- 2-3x faster processing
- Automatic optimization

---

## Key Features

### 1. Single-Step Processing

Process RAW LAZ files directly to patches in one command:

```bash
# Using Hydra CLI
python -m ign_lidar.cli.hydra_app \
  input_dir=/path/to/raw_laz \
  output_dir=/path/to/patches

# Using legacy CLI
ign-lidar-hd enrich /path/to/raw_laz /path/to/patches
```

The pipeline automatically:

- Downloads missing files (if auto-download enabled)
- Preprocesses RAW LAZ (enrichment)
- Extracts patches
- Computes features
- Saves final output

### 2. In-Memory Processing

**No intermediate files written to disk** unless explicitly configured:

```yaml
# config.yaml
output:
  save_enriched_laz: false # Skip intermediate LAZ (default)
  only_enriched_laz: false # Only save enriched LAZ (optional)
```

**Memory Management:**

- Processes one tile at a time
- Configurable batch sizes for patches
- Automatic cleanup
- GPU memory pooling (if GPU enabled)

### 3. Space Savings

**Before (v1.x):**

```
RAW LAZ:      1.0 GB
Enriched LAZ: 1.5 GB  ← Intermediate file
Patches:      0.5 GB
─────────────────────
Total Disk:   3.0 GB
```

**After (v2.0+):**

```
RAW LAZ:      1.0 GB
Patches:      0.5 GB
─────────────────────
Total Disk:   1.5 GB (50% savings!)
```

### 4. Speed Improvements

**Processing Time Comparison:**

| Dataset Size | v1.x Pipeline | v2.0+ Unified | Speedup |
| ------------ | ------------- | ------------- | ------- |
| 1 tile       | 45s           | 20s           | 2.25x   |
| 10 tiles     | 8.5 min       | 3.5 min       | 2.4x    |
| 100 tiles    | 92 min        | 38 min        | 2.4x    |

:::tip Performance
The speedup comes from:

- Eliminating disk I/O for intermediate files
- In-memory data passing
- Optimized memory layout
- GPU acceleration (if available)
  :::

---

## Configuration

### Basic Configuration

```yaml
# Unified pipeline (default)
output:
  save_enriched_laz: false # Don't save intermediate LAZ
```

### Save Intermediate LAZ (Optional)

If you need enriched LAZ files:

```yaml
output:
  save_enriched_laz: true # Save both enriched LAZ and patches
  enriched_laz_dir: "/path/to/enriched"
```

### Enriched LAZ Only Mode

For 3-5x faster processing when you only need enriched LAZ:

```yaml
output:
  only_enriched_laz: true # Skip patch extraction
  save_enriched_laz: true
```

See [Enriched LAZ Only Mode](../features/enriched-laz-only.md) for details.

---

## Usage Examples

### Example 1: Default Unified Pipeline

Process RAW LAZ to patches directly:

```bash
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw_laz \
  output_dir=/data/output \
  preprocessing.num_neighbors=50
```

**Output:**

```
/data/output/
├── tile_001.laz          # Processed patches
├── tile_002.laz
└── tile_003.laz
```

### Example 2: With Intermediate LAZ

Save both enriched LAZ and patches:

```bash
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw_laz \
  output_dir=/data/output \
  output.save_enriched_laz=true \
  output.enriched_laz_dir=/data/enriched
```

**Output:**

```
/data/enriched/
├── tile_001.laz          # Enriched LAZ files
├── tile_002.laz
└── tile_003.laz

/data/output/
├── tile_001.laz          # Processed patches
├── tile_002.laz
└── tile_003.laz
```

### Example 3: With Tile Stitching

Combine multiple tiles in-memory before patch extraction:

```bash
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw_laz \
  output_dir=/data/output \
  stitching.enabled=true \
  stitching.pattern=3x3
```

**Workflow:**

```
RAW Tiles → [Stitch In-Memory] → [Extract Patches] → Output
tile_001.laz ─┐
tile_002.laz ─┼→ Combined Point Cloud → Patches → stitched.laz
tile_003.laz ─┘
```

### Example 4: GPU Accelerated

Enable GPU for faster feature computation:

```bash
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw_laz \
  output_dir=/data/output \
  gpu.enabled=true \
  gpu.use_cuml=true
```

---

## Performance Optimization

### 1. Memory Management

**Control memory usage:**

```yaml
preprocessing:
  batch_size: 50000 # Points per batch
  num_workers: 4 # CPU threads

gpu:
  chunk_size: 100000 # GPU chunk size
```

**Guidelines:**

- Larger batches = faster, more memory
- Smaller batches = slower, less memory
- Monitor with: `nvidia-smi` (GPU) or `htop` (CPU)

### 2. Disk I/O Optimization

**Use fast storage:**

- SSD preferred over HDD
- Local storage faster than network
- Temporary files on fast partition

**Example:**

```yaml
output:
  temp_dir: "/tmp" # Fast local storage
```

### 3. Parallel Processing

**Process multiple tiles concurrently:**

```python
from ign_lidar.core.processor import LiDARProcessor
from concurrent.futures import ProcessPoolExecutor

processor = LiDARProcessor(config)

def process_tile(tile_path):
    return processor.process_tile(tile_path)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_tile, tile_paths)
```

---

## Migration from v1.x

### Old Two-Step Workflow

```bash
# v1.x - Step 1: Enrich
ign-lidar-hd enrich /data/raw /data/enriched

# v1.x - Step 2: Extract patches
ign-lidar-hd extract /data/enriched /data/patches
```

### New Unified Workflow

```bash
# v2.0+ - Single step
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw \
  output_dir=/data/patches
```

### Configuration Migration

**Old (v1.x):**

```yaml
enrichment:
  num_neighbors: 50
  features: ["planarity", "linearity"]

extraction:
  patch_size: 50.0
  overlap: 10.0
```

**New (v2.0+):**

```yaml
preprocessing:
  num_neighbors: 50

features:
  enabled_features: ["planarity", "linearity"]

processing:
  patch_size: 50.0
  patch_overlap: 10.0
```

---

## API Usage

### Unified Processing

```python
from ign_lidar.core.processor import LiDARProcessor
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config.yaml")

# Create processor
processor = LiDARProcessor(config)

# Process single tile (RAW → Patches)
result = processor.process_tile(
    input_path="raw_tile.laz",
    output_path="output_patches.laz"
)

print(f"Processed {result.num_points} points")
print(f"Extracted {result.num_patches} patches")
```

### With Intermediate Output

```python
# Save enriched LAZ during processing
processor = LiDARProcessor(config)

result = processor.process_tile(
    input_path="raw_tile.laz",
    output_path="output_patches.laz",
    save_enriched=True,
    enriched_path="enriched_tile.laz"
)
```

### Batch Processing

```python
from pathlib import Path

input_dir = Path("/data/raw")
output_dir = Path("/data/output")

# Process all LAZ files
for laz_file in input_dir.glob("*.laz"):
    output_file = output_dir / laz_file.name

    processor.process_tile(
        input_path=str(laz_file),
        output_path=str(output_file)
    )
```

---

## Troubleshooting

### Out of Memory

**Issue:** Process crashes with memory errors

**Solutions:**

1. Reduce batch size:

   ```yaml
   preprocessing:
     batch_size: 25000 # Smaller batches
   ```

2. Process one tile at a time:

   ```python
   for tile in tiles:
       processor.process_tile(tile)
       # Memory released after each tile
   ```

3. Enable enriched LAZ saving (trades speed for memory):
   ```yaml
   output:
     save_enriched_laz: true
   ```

### Slow Processing

**Issue:** Processing slower than expected

**Solutions:**

1. Enable GPU acceleration:

   ```yaml
   gpu:
     enabled: true
   ```

2. Increase parallel workers:

   ```yaml
   preprocessing:
     num_workers: 8 # Match CPU cores
   ```

3. Check disk I/O bottleneck:
   ```bash
   iotop  # Monitor disk usage
   ```

### Incomplete Output

**Issue:** Some patches missing

**Solutions:**

1. Check logs for errors:

   ```yaml
   logging:
     level: DEBUG
   ```

2. Verify input files are valid:
   ```python
   import laspy
   las = laspy.read("input.laz")
   print(f"Points: {len(las.points)}")
   ```

---

## Best Practices

### 1. Start with Default Settings

```bash
# Use defaults for first run
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw \
  output_dir=/data/output
```

### 2. Monitor Resource Usage

```bash
# Terminal 1: Run processing
python -m ign_lidar.cli.hydra_app ...

# Terminal 2: Monitor resources
watch -n 1 nvidia-smi  # GPU
htop                    # CPU/Memory
iotop                   # Disk
```

### 3. Test on Small Dataset

```bash
# Test on 1-2 tiles first
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw \
  output_dir=/data/test \
  input.file_pattern="*_001.laz"
```

### 4. Use Configuration Files

```bash
# Save working configuration
python -m ign_lidar.cli.hydra_app \
  input_dir=/data/raw \
  output_dir=/data/output \
  --cfg job > working_config.yaml

# Reuse configuration
python -m ign_lidar.cli.hydra_app \
  --config-name=working_config
```

---

## Related Documentation

- [Hydra CLI Guide](hydra-cli.md) - CLI usage
- [Configuration System](configuration-system.md) - Configuration options
- [Enriched LAZ Only Mode](../features/enriched-laz-only.md) - Faster preprocessing
- [Tile Stitching](../features/tile-stitching.md) - Multi-tile processing
- [Performance Guide](performance.md) - Optimization tips

---

## Summary

The unified processing pipeline in v2.0+ offers:

✅ **Simplicity** - Single command for RAW to Patches  
✅ **Speed** - 2-3x faster than v1.x  
✅ **Efficiency** - 35-50% disk space savings  
✅ **Flexibility** - Multiple output modes  
✅ **Compatibility** - Works with existing workflows

**Recommended for all new projects!**
