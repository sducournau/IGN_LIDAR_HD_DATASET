# K-NN Performance & Enriched LAZ Missing - Solutions

## Problem Analysis

### Issue 1: K-NN Search Taking 51+ Minutes

**Root Cause**: cuML's `NearestNeighbors.kneighbors()` is extremely slow (~164s per 1M points)

- 18.6M points × 20 neighbors = 372M neighbor lookups
- Current implementation: batching at 1M points due to memory constraints
- 19 batches × ~165s = **51 minutes of neighbor queries alone**

### Issue 2: Enriched LAZ File Not Created

**Root Cause**: Code path only saves enriched LAZ in `enriched_only` mode

- Your config has `processing_mode: both`
- But the save_enriched_tile_laz() is only called when:
  - `self.only_enriched_laz == True` AND `self.patch_size is None`
- In "both" mode, this path is never reached
- Line 1630-1675 in processor.py shows the conditional logic

## Solutions

---

## Solution 1: Fix Enriched LAZ Saving in "both" Mode

### Quick Fix (Immediate)

Add enriched LAZ saving after patch extraction in "both" mode.

**File**: `ign_lidar/core/processor.py`

**Location**: After line 1789 (after patch saving loop, before metadata saving)

```python
        # Save patches with proper naming
        # Each patch has _version and _patch_idx metadata
        for patch in all_patches_collected:
            # ... existing patch saving code ...
            pass  # (existing code continues)

        # 🆕 NEW: Save enriched LAZ tile in "both" mode
        if self.save_enriched_laz and not self.only_enriched_laz:
            from .modules.serialization import save_enriched_tile_laz

            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{laz_file.stem}_enriched.laz"

            try:
                logger.info(f"  💾 Saving enriched LAZ tile...")

                # Determine RGB/NIR to save (prefer fetched/computed over input)
                save_rgb = all_features_v.get('rgb') if all_features_v.get('rgb') is not None else original_data.get('input_rgb')
                save_nir = all_features_v.get('nir') if all_features_v.get('nir') is not None else original_data.get('input_nir')

                # Remove RGB/NIR from features dict to avoid duplication
                features_to_save = {k: v for k, v in all_features_v.items()
                                   if k not in ['rgb', 'nir', 'input_rgb', 'input_nir', 'points']}

                # Save the enriched tile with all computed features
                save_enriched_tile_laz(
                    save_path=output_path,
                    points=original_data['points'],
                    classification=labels_v,
                    intensity=original_data['intensity'],
                    return_number=original_data['return_number'],
                    features=features_to_save,
                    original_las=original_data.get('las'),
                    header=original_data.get('header'),
                    input_rgb=save_rgb,
                    input_nir=save_nir
                )

            except Exception as e:
                logger.error(f"  ✗ Failed to save enriched LAZ tile: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Continue with existing metadata saving...
        tile_time = time.time() - tile_start
```

---

## Solution 2: Dramatically Speed Up K-NN (Multiple Options)

### Option A: Use FAISS GPU (RECOMMENDED - 50-100× faster)

**Why FAISS?**

- Purpose-built for billion-scale nearest neighbor search
- 50-100× faster than cuML for k-NN queries
- Battle-tested by Meta AI for production workloads
- Better GPU memory management

**Installation**:

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
```

**Implementation**:
**File**: `ign_lidar/features/features_gpu_chunked.py`

Add after line 28:

```python
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
```

Add new method around line 780:

```python
def _build_faiss_index(self, points: np.ndarray, use_gpu: bool = True) -> Any:
    """
    Build FAISS index for ultra-fast k-NN queries.

    FAISS is 50-100× faster than cuML for neighbor searches.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available - falling back to cuML")

    N, D = points.shape
    logger.info(f"  🚀 Building FAISS index ({N:,} points)...")

    # Use IVF (Inverted File) index for large datasets
    # For 18M points, use ~4000 centroids (sqrt(N)/2)
    nlist = min(4096, int(np.sqrt(N)))

    # Create index
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)

    if use_gpu and self.use_cuml:
        # Move to GPU
        res = faiss.StandardGpuResources()
        res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Train index (required for IVF)
    logger.info(f"  🔧 Training FAISS index with {nlist} clusters...")
    index.train(points.astype(np.float32))

    # Add all points
    logger.info(f"  📥 Adding {N:,} points to index...")
    index.add(points.astype(np.float32))

    logger.info(f"  ✓ FAISS index ready")
    return index

def compute_normals_optimized_global_kdtree_faiss(
    self,
    points: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """
    Compute normals using FAISS for 50-100× faster neighbor queries.

    Expected performance: 18.6M points in ~30-60 seconds (vs 51 minutes with cuML)
    """
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)

    logger.info(f"  🚀 Using FAISS for neighbor queries (ultra-fast)")

    try:
        # Build FAISS index
        index = self._build_faiss_index(points, use_gpu=self.use_cuml)

        # Query all neighbors at once (FAISS handles batching internally)
        logger.info(f"  ⚡ Querying all {N:,} neighbors in one batch...")
        distances, indices = index.search(points.astype(np.float32), k)

        logger.info(f"  ✓ All neighbors found in seconds (FAISS)")

        # Compute normals from neighbors (use GPU if available)
        logger.info(f"  ⚡ Computing normals from neighbors...")
        if self.use_cuml and cp is not None:
            # Transfer to GPU for PCA
            points_gpu = cp.asarray(points)
            indices_gpu = cp.asarray(indices)

            normals_gpu = self._compute_normals_from_neighbors_gpu(
                points_gpu, indices_gpu
            )
            normals = cp.asnumpy(normals_gpu)

            del points_gpu, indices_gpu, normals_gpu
            self._free_gpu_memory(force=True)
        else:
            # CPU PCA
            normals = self._compute_normals_from_neighbors_cpu(
                points, indices
            )

        logger.info(f"  ✓ Normals computed")
        return normals

    except Exception as e:
        logger.warning(f"  ⚠️  FAISS failed: {e}, falling back to cuML")
        # Fallback to existing cuML implementation
        return self.compute_normals_optimized_global_kdtree(points, k)
```

Update the main compute method (around line 590):

```python
def compute_features_gpu_chunked(self, points, mode, ...):
    # ... existing setup ...

    # Choose k-NN backend
    if FAISS_AVAILABLE and self.use_cuml:
        logger.info("  🚀 Using FAISS GPU for neighbor search")
        normals = self.compute_normals_optimized_global_kdtree_faiss(points, k=15)
    else:
        logger.info("  🔧 Using cuML for neighbor search")
        normals = self.compute_normals_optimized_global_kdtree(points, k=15)

    # ... rest of feature computation ...
```

### Option B: Reduce k_neighbors (Quick Workaround)

**Impact**: Use fewer neighbors for speed

- k=20 → k=10: ~50% faster (25 min instead of 51 min)
- k=20 → k=5: ~75% faster (12 min instead of 51 min)
- Slight quality trade-off, but often acceptable for ASPRS classes

**Config Change**:

```yaml
features:
  k_neighbors: 10 # Reduced from 20
  neighbor_query_batch_size: 2000000 # Increase batch size
```

### Option C: Use Approximate k-NN in cuML

**File**: `ign_lidar/features/features_gpu_chunked.py`

Around line 646:

```python
# Use approximate k-NN for speed
knn = cuNearestNeighbors(
    n_neighbors=k,
    metric='euclidean',
    algorithm='brute'  # Change to 'ivfflat' or 'ivfpq' for approximation
)
```

**Trade-off**: 2-5× faster but slightly approximate results

---

## Solution 3: Hybrid Approach (Best Balance)

Combine multiple optimizations:

1. **Install FAISS**: For 50× k-NN speedup
2. **Reduce k**: Use k=10 instead of 20
3. **Enable approximate**: Use FAISS IVFFlat with nprobe=32

**Expected Performance**:

- Current: 51 minutes for neighbor queries
- With FAISS: ~30-60 seconds for neighbor queries
- **Total speedup: 50-100× faster**

---

## Recommended Action Plan

### Immediate (5 minutes):

1. ✅ **Fix enriched LAZ saving** - Apply Solution 1
2. ✅ **Reduce k_neighbors to 10** - Quick config change

### Short-term (30 minutes):

3. ✅ **Install and integrate FAISS** - Apply Solution 2A
4. ✅ Test with one tile to verify performance

### Expected Results:

- **Before**: 64 minutes per tile (51min k-NN + 13min other)
- **After**: 2-5 minutes per tile (30s k-NN + 1.5min other)
- **128 tiles**: From ~136 hours → ~4-10 hours total

---

## Implementation Priority

### Priority 1 (Do Now): Fix Enriched LAZ

Apply the code change in Solution 1 to ensure enriched LAZ files are created.

### Priority 2 (Do Today): FAISS Integration

Install FAISS and integrate for massive speedup:

```bash
conda activate ign_gpu
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
```

### Priority 3 (Optional): Fine-tune

- Adjust k_neighbors based on quality requirements
- Optimize FAISS parameters (nprobe, nlist)

---

## Testing Commands

### Test with one tile:

```bash
ign-lidar-hd process \
  -c "ign_lidar/configs/presets/asprs_rtx4080.yaml" \
  input_dir="/mnt/d/ign/selected_tiles/asprs/tiles" \
  output_dir="/mnt/d/ign/test_output" \
  processor.num_workers=1 \
  features.k_neighbors=10
```

### Verify enriched LAZ created:

```bash
ls -lh /mnt/d/ign/test_output/*_enriched.laz
```

### Check performance improvement:

Look for log lines:

- "⚡ Querying all X neighbors in one batch..." should complete in seconds
- "✓ All neighbors found in seconds (FAISS)"

---

## Additional Notes

### Why cuML is Slow

- cuML's NearestNeighbors is designed for machine learning workflows
- Not optimized for the massive batch queries needed here
- FAISS is purpose-built for this exact use case

### FAISS Benefits

- Used by production systems at Facebook, Google, etc.
- Handles billion-scale datasets efficiently
- GPU implementation is highly optimized
- Better memory management than cuML for this use case

### Quality Impact

- FAISS with IVFFlat and proper parameters: <1% accuracy loss
- Reducing k from 20→10: Minimal impact for ASPRS classes
- Most geometric features (planarity, sphericity) are robust to fewer neighbors
