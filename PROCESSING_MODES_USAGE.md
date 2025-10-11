# Quick Usage Guide - Processing Modes

**IGN LiDAR HD v2.3.0**

---

## 🚀 Quick Start

### Mode 1: Patches Only (ML Training)

Creates only ML-ready patches for training models:

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/patches \
  output.processing_mode=patches_only
```

**Output:** `tile_patch_0001.npz`, `tile_patch_0002.npz`, ...

---

### Mode 2: Both (ML + GIS)

Creates both patches AND enriched LAZ files:

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/both \
  output.processing_mode=both
```

**Output:**

- ML patches: `tile_patch_0001.npz`, ...
- GIS files: `tile_enriched.laz`

---

### Mode 3: Enriched LAZ Only (GIS Analysis)

Creates only enriched LAZ files (fastest for GIS workflows):

```bash
ign-lidar-hd process \
  input_dir=data/raw \
  output_dir=data/enriched \
  output.processing_mode=enriched_only
```

**Output:** `tile_enriched.laz` (with normals, curvature, RGB, etc.)

---

## 🐍 Python API

```python
from ign_lidar.core.processor import LiDARProcessor

# Mode 1: Patches only
processor = LiDARProcessor(
    lod_level='LOD2',
    processing_mode='patches_only',
    patch_size=150.0,
    num_points=16384
)
processor.process_directory('data/raw', 'data/patches')

# Mode 2: Both
processor = LiDARProcessor(
    lod_level='LOD2',
    processing_mode='both',
    patch_size=150.0,
    num_points=16384
)
processor.process_directory('data/raw', 'data/both')

# Mode 3: Enriched LAZ only
processor = LiDARProcessor(
    lod_level='LOD2',
    processing_mode='enriched_only',
    use_gpu=True
)
processor.process_directory('data/raw', 'data/enriched')
```

---

## 📝 YAML Configuration

Create `my_config.yaml`:

```yaml
processor:
  lod_level: LOD2
  use_gpu: false
  patch_size: 150.0
  num_points: 16384

output:
  processing_mode: both # Choose: patches_only, both, enriched_only
  format: npz

input_dir: data/raw
output_dir: data/processed
```

Then run:

```bash
ign-lidar-hd process experiment=my_config
```

---

## 🔄 Migration from v2.2.x

### Old Way (Deprecated but still works)

```python
# OLD - confusing flags
processor = LiDARProcessor(
    save_enriched_laz=True,
    only_enriched_laz=False
)
```

### New Way (Recommended)

```python
# NEW - clear and explicit
processor = LiDARProcessor(
    processing_mode='both'
)
```

---

## 📊 Comparison Table

| Mode            | Create Patches | Create LAZ | Speed   | Best For    |
| --------------- | -------------- | ---------- | ------- | ----------- |
| `patches_only`  | ✅             | ❌         | Fast    | ML training |
| `both`          | ✅             | ✅         | Slower  | ML + GIS    |
| `enriched_only` | ❌             | ✅         | Fastest | GIS only    |

---

## 💡 Tips

1. **For ML training only:** Use `patches_only` (default, fastest)
2. **For GIS analysis:** Use `enriched_only` (skips patch creation)
3. **For both workflows:** Use `both` (creates everything)

---

## 🐛 Troubleshooting

**Deprecation Warning?**

```
⚠️  'save_enriched_laz' and 'only_enriched_laz' are deprecated.
```

→ Update your code to use `processing_mode` instead!

**Invalid mode error?**
→ Valid modes: `patches_only`, `both`, `enriched_only`

---

_For more details, see PHASE1_COMPLETE.md_
