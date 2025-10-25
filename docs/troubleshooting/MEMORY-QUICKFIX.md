# Quick Fix: Out of Memory Crashes

## ⚡ Immediate Solution

**If your process is being killed with memory errors, use this:**

```bash
ign-lidar-hd process \
  -c examples/production/asprs_memory_optimized.yaml \
  input_dir="/your/tiles" \
  output_dir="/your/output"
```

This configuration uses **20-24GB peak memory** instead of 30-35GB.

---

## 🔧 What Changed in v6.3.2

### File: `ign_lidar/features/compute/multi_scale.py`

**Improved automatic chunking** to prevent OOM:

- Triggers chunking at 30% memory usage (was 50%)
- Uses smaller chunks (15% of available memory, was 20%)
- Caps chunk size at 3M points (was unlimited)
- More conservative fallback (2M points, was 5M)

### File: `examples/production/asprs_memory_optimized.yaml` (NEW)

**Memory-optimized configuration:**

- Disables multi-scale computation
- chunk_size: 2M (was 5M)
- gpu_batch_size: 15M (was 30M)
- DTM augmentation: 2m grid (was 1m) = 75% fewer points
- Strategy: "gaps" only (was "intelligent")

---

## 📊 Quick Comparison

| Config                      | Memory  | Speed  | Quality       | When to Use |
| --------------------------- | ------- | ------ | ------------- | ----------- |
| asprs_complete.yaml         | 30-35GB | Slower | Best (94-97%) | 64GB+ RAM   |
| asprs_memory_optimized.yaml | 20-24GB | Faster | Good (92-95%) | 28-32GB RAM |

---

## 🚀 Quick Configuration Tweaks

### If Still Running Out of Memory

Add these to your existing config:

```yaml
processor:
  chunk_size: 2_000_000 # Reduce from 5M
  gpu_batch_size: 15_000_000 # Reduce from 30M

features:
  multi_scale_computation: false # Disable multi-scale

ground_truth:
  rge_alti:
    augmentation_spacing: 2.0 # Increase from 1.0m
    augmentation_strategy: "gaps" # Change from "intelligent"
```

### For Maximum Speed

```yaml
features:
  multi_scale_computation: false
  compute_cluster_id: false
  compute_building_cluster_id: false
  compute_parcel_cluster_id: false

ground_truth:
  rge_alti:
    augment_ground: false # Disable DTM augmentation entirely
```

---

## ⚠️ Warning Signs

**Watch your logs for these patterns:**

```
⚠️  High memory usage: 99.7%   <- Danger!
[1] 80737 killed                <- OOM crash
```

**If you see 95%+ memory warnings:**

- Stop the process (Ctrl+C)
- Switch to asprs_memory_optimized.yaml
- Or reduce chunk_size further

---

## ✅ How to Verify the Fix

After the code changes in v6.3.2, you should see:

```
🔄 Auto-enabling chunked processing: chunk_size=1,234,567 (estimated 8500MB / 28000MB available)
📦 Processing in 18 chunks of ~1,234,567 points
```

If you see this, the automatic chunking is working!

---

## 📞 Still Having Issues?

1. Check your available memory:

   ```bash
   free -h
   ```

2. Try the most conservative config:

   ```bash
   ign-lidar-hd process \
     -c examples/production/asprs_memory_optimized.yaml \
     processor.chunk_size=1000000 \
     features.compute_cluster_id=false \
     ground_truth.rge_alti.augment_ground=false \
     input_dir="/your/tiles" \
     output_dir="/your/output"
   ```

3. Report issue with:
   - Your RAM amount
   - Tile point count
   - Full error log

---

## 🎯 Bottom Line

- **28-32GB RAM:** Use `asprs_memory_optimized.yaml`
- **32-48GB RAM:** Use `asprs_complete.yaml` with the v6.3.2 auto-chunking fix
- **64GB+ RAM:** Use `asprs_complete.yaml` (original)

The v6.3.2 code improvements make chunking automatic and more aggressive, so most users won't need to manually adjust configurations anymore.
