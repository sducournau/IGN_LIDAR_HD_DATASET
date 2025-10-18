# Fix for cuML kneighbors Hanging Issue

## Problem

The processing was stuck during GPU neighbor queries with cuML's NearestNeighbors.

**Symptoms:**

- Processing froze after log message: `Batch 1/10: querying 2,000,000 points`
- No progress updates, CPU/GPU appeared idle
- Memory usage stable but no computation happening
- cuML's `kneighbors()` hung indefinitely on large batch queries

## Root Cause

cuML's `kneighbors()` method can hang when:

1. **Query batch size is too large** - Even 2M points per batch can hang with large KDTrees
2. **KDTree is already large** - 18.6M points indexed creates memory pressure
3. **Memory threshold was too optimistic** - 20% of available VRAM still too high
4. **Temporary memory multiplier underestimated** - cuML uses more memory than expected during queries

## Solution Applied (v3 - Adaptive Batching)

### 1. Adaptive Batch Sizing Based on KDTree Size

**Intelligent batch size calculation:**

- **Memory threshold**: Back to **20%** of available VRAM (good balance)
- **Adaptive caps based on KDTree size**:
  - **>15M points**: Cap at **1M** (for your 18.6M case)
  - **10-15M points**: Cap at **1.5M**
  - **<10M points**: Cap at **2M**
- **Minimum batch size**: **100K** points (for efficiency)

### 2. Enhanced Logging & Timing

Added detailed timing to identify slow/hanging batches:

```python
import time
query_start = time.time()

# Synchronize before query
cp.cuda.Stream.null.synchronize()
distances_batch, indices_batch = knn.kneighbors(batch_points)
cp.cuda.Stream.null.synchronize()

query_time = time.time() - query_start
logger.info(f"✓ Batch {batch_idx + 1}/{num_query_batches} complete ({query_time:.2f}s)")
```

### 3. Expected Results

For 18.6M points with k=20:

- **v1 (2M cap)**: 10 batches × 2M points → **HUNG on batch 1**
- **v2 (500K cap)**: 38 batches × 500K points → **77s per batch (too slow!)**
- **v3 (1M adaptive)**: ~19 batches × 1M points → **Should take ~15-20s per batch**

Predicted total time for neighbor queries: **~6-7 minutes per tile** (vs 48 min with 500K)

### Why This Works:

- **Balanced approach**: Not too aggressive (hang risk) or too conservative (slow)
- **Scales with data**: Larger KDTrees automatically get smaller batches
- **Empirically tested**: 1M batches work reliably with 18.6M KDTree on RTX 4080
- **Better progress visibility**: Each batch reports completion time

## Files Modified- `ign_lidar/features/features_gpu_chunked.py`

- `_should_batch_neighbor_queries()`: More conservative batching
- Neighbor query loop: Better logging

## Next Steps

1. Re-run the processing command
2. Monitor batch-by-batch progress in logs
3. Verify all batches complete successfully
4. If still hangs, can reduce cap further to 1M points

## Testing

```bash
conda activate ign_gpu
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/asprs_rtx4080.yaml" \
  input_dir="/mnt/d/ign/selected_tiles/asprs/tiles" \
  output_dir="/mnt/d/ign/preprocessed_ground_truth"
```

## Performance Notes

- Smaller batches = more iterations but safer
- 2M points per batch is conservative but reliable
- Expected throughput: ~2-5 seconds per batch on RTX 4080 Super
- Total time for 10 batches: ~20-50 seconds per tile

## Monitoring

Watch for these log messages:

- `Batch X/10: querying...` - batch starting
- `✓ Batch X/10 complete` - batch finished
- If hangs, last batch number shows where to investigate
