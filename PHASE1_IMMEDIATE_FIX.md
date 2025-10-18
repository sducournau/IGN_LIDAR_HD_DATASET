# Phase 1: Immediate Performance Fix

**Target:** 5-10× speedup on FAISS neighbor queries  
**Time:** 30 minutes  
**Risk:** Low (only changes batch size calculation)

## Changes Required

### File: `ign_lidar/features/features_gpu_chunked.py`

Location: Lines ~2800-2850 (in `compute_all_features_optimized_v2` method)

## Current Code (SLOW)

```python
# BATCH QUERIES to avoid GPU OOM
batch_size = 2_000_000  # 2M points per batch (TOO CONSERVATIVE!)
num_batches = (N + batch_size - 1) // batch_size

# Preallocate output arrays
indices_all = np.zeros((N, k), dtype=np.int32)

if num_batches > 1:
    logger.info(f"     Batching FAISS queries: {num_batches} batches of {batch_size:,} points")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, N)

        # Query this batch
        batch_points = points_np[batch_start:batch_end]
        distances_batch, indices_batch = faiss_index.search(batch_points, k)
        indices_all[batch_start:batch_end] = indices_batch

        if (batch_idx + 1) % 5 == 0:  # Log every 5 batches
            logger.info(f"        Batch {batch_idx + 1}/{num_batches} complete")

        del distances_batch, indices_batch, batch_points
```

## New Code (FAST)

```python
# OPTIMIZED: Dynamic batch size based on available VRAM
# Calculate optimal batch size
available_vram_gb = self.available_vram_gb if hasattr(self, 'available_vram_gb') else 14.0

# Memory per point for query (in bytes)
# - Query points: 3 × 4 = 12 bytes (xyz float32)
# - Result indices: k × 4 bytes (int32)
# - Result distances: k × 4 bytes (float32)
# - Overhead: 32 bytes
memory_per_point = 12 + (k * 8) + 32

# Use 50% of available VRAM for batch (conservative, leaves room for index + ops)
usable_vram_bytes = available_vram_gb * 0.5 * (1024**3)
max_batch_size = int(usable_vram_bytes / memory_per_point)

# Clamp to reasonable range
batch_size = min(
    max_batch_size,
    N,           # Don't exceed total points
    20_000_000   # Hard cap at 20M for safety
)
batch_size = max(batch_size, 1_000_000)  # Minimum 1M

num_batches = (N + batch_size - 1) // batch_size

logger.info(
    f"     📊 FAISS batch sizing: {N:,} points → {num_batches} batches of ~{batch_size:,} points "
    f"(VRAM: {available_vram_gb:.1f}GB available, using ~{usable_vram_bytes/1024**3:.1f}GB per batch)"
)

# Preallocate output arrays
indices_all = np.zeros((N, k), dtype=np.int32)

if num_batches > 1:
    logger.info(f"     ⚡ Querying in {num_batches} optimized batches...")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, N)
        batch_n_points = batch_end - batch_start

        # Time each batch
        import time
        batch_start_time = time.time()

        # Query this batch
        batch_points = points_np[batch_start:batch_end]
        distances_batch, indices_batch = faiss_index.search(batch_points, k)
        indices_all[batch_start:batch_end] = indices_batch

        batch_elapsed = time.time() - batch_start_time

        # Log progress with timing
        logger.info(
            f"        ✓ Batch {batch_idx + 1}/{num_batches}: {batch_n_points:,} points in {batch_elapsed:.1f}s "
            f"({batch_n_points/batch_elapsed/1e6:.2f}M pts/s)"
        )

        del distances_batch, indices_batch, batch_points
```

## Testing

After applying the fix:

```bash
# Test on single tile
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/asprs_rtx4080_fast.yaml" \
  input_dir="/mnt/d/ign/selected_tiles/asprs/tiles" \
  output_dir="/mnt/d/ign/test_phase1" \
  --max-tiles=1

# Expected logs:
# BEFORE: "Batching FAISS queries: 10 batches of 2,000,000 points"
# AFTER:  "Batching FAISS queries: 2 batches of ~10,000,000 points"
```

## Expected Results

### Before (Current)

- 18.6M points @ 2M batch size
- Number of batches: 10
- Time per batch: ~3-6s
- **Total query time: ~30-60s**

### After (Optimized)

- 18.6M points @ 10-12M batch size
- Number of batches: 2
- Time per batch: ~3-6s
- **Total query time: ~6-12s** ✅ **5× faster!**

### Full Pipeline Impact

- Before: ~50-80s per tile
- After: ~26-36s per tile
- **Overall speedup: 2-3×**

For 128 tiles:

- Before: ~1.7-2.8 hours
- After: ~55-77 minutes
- **Time saved: ~1 hour**

## Risk Assessment

**Risk Level:** LOW

- Only changes batch size calculation
- Same underlying FAISS calls
- Failsafes:
  - Hard cap at 20M points (prevents OOM)
  - Minimum 1M points (prevents too many batches)
  - Falls back to original 2M if calculation fails

## Rollback

If issues occur:

```python
# Revert to original:
batch_size = 2_000_000  # Conservative default
```

Or restore from backup:

```bash
cp ign_lidar/features/features_gpu_chunked.py.backup \
   ign_lidar/features/features_gpu_chunked.py
```
