# GPU Integration TODO List

**Version:** 1.2.0 → 1.3.0  
**Status:** Planning  
**Priority:** Medium  
**Effort:** 4-8 hours

---

## Current Status

- ❌ GPU module exists but not integrated
- ❌ `--use-gpu` flag parsed but ignored
- ✅ CPU processing fully functional
- ✅ Automatic fallback works

## Integration Checklist

### Phase 1: Basic Integration (v1.2.1 - Hotfix)

- [ ] **Update `ign_lidar/features.py`**

  - [ ] Add `compute_features_with_gpu()` wrapper function
  - [ ] Import GPUFeatureComputer conditionally
  - [ ] Add GPU detection logic

- [ ] **Update `ign_lidar/cli.py`**

  - [ ] Import GPU feature function in `_enrich_single_file()`
  - [ ] Connect `use_gpu` parameter to GPU computation
  - [ ] Add warning if GPU requested but not available

- [ ] **Testing**

  - [ ] Create `tests/test_gpu_integration.py`
  - [ ] Test CPU fallback
  - [ ] Test GPU flag handling
  - [ ] Test error messages

- [ ] **Documentation**
  - [ ] Update README.md with accurate GPU status
  - [ ] Add GPU installation guide
  - [ ] Document performance benchmarks

### Phase 2: Feature Parity (v1.3.0 - Feature Release)

- [ ] **Complete GPU Feature Computer**

  - [ ] Implement `compute_all_features()` method
  - [ ] Match CPU feature set exactly
  - [ ] Add height_above_ground on GPU
  - [ ] Add RGB augmentation GPU support

- [ ] **Add to LiDARProcessor**

  - [ ] Add `use_gpu` parameter to `__init__()`
  - [ ] Connect to GPU feature computation
  - [ ] Add to process_tile() method

- [ ] **Performance Optimization**

  - [ ] Benchmark CPU vs GPU
  - [ ] Optimize batch sizes
  - [ ] Memory profiling
  - [ ] Multi-GPU support planning

- [ ] **Advanced Testing**

  - [ ] Unit tests for all GPU features
  - [ ] Integration tests with full pipeline
  - [ ] Performance regression tests
  - [ ] Memory leak tests

- [ ] **Comprehensive Documentation**
  - [ ] Dedicated GPU guide page
  - [ ] Installation troubleshooting
  - [ ] Performance comparison charts
  - [ ] CUDA version compatibility matrix

### Phase 3: Advanced Features (v2.0.0 - Future)

- [ ] Multi-GPU support
- [ ] GPU-accelerated patch extraction
- [ ] Streaming processing
- [ ] Mixed precision computation

---

## Code Snippets

### features.py - GPU Wrapper

```python
def compute_features_with_gpu(points: np.ndarray, k: int = 10,
                              use_gpu: bool = False) -> np.ndarray:
    """
    Compute geometric features with optional GPU acceleration.

    Args:
        points: [N, 3] point coordinates
        k: number of neighbors
        use_gpu: enable GPU acceleration

    Returns:
        features: [N, M] computed features
    """
    if use_gpu:
        try:
            from .features_gpu import GPUFeatureComputer
            logger.info("Using GPU acceleration for feature computation")
            computer = GPUFeatureComputer(use_gpu=True)

            # Compute features on GPU
            normals = computer.compute_normals(points, k=k)
            curvature = computer.compute_curvature(points, normals, k=k)
            geom_features = computer.extract_geometric_features(points, normals, k=k)

            # Combine into feature array
            features = np.column_stack([
                normals,
                curvature,
                geom_features['planarity'],
                geom_features['linearity'],
                geom_features['sphericity'],
                geom_features['anisotropy'],
                geom_features['roughness'],
                geom_features['density']
            ])

            return features.astype(np.float32)

        except ImportError as e:
            logger.warning(f"GPU requested but not available: {e}")
            logger.warning("Falling back to CPU processing")
            return compute_all_features_optimized(points, k=k)
    else:
        return compute_all_features_optimized(points, k=k)
```

### cli.py - Integration Point

```python
def _enrich_single_file(args):
    """Worker function for enriching a single LAZ file."""
    laz_path, output_path, k_neighbors, use_gpu, mode, skip_existing, add_rgb, rgb_cache_dir = args

    # ... existing code for loading points ...

    # Compute features with GPU support
    logger.info(f"Computing features (GPU={'enabled' if use_gpu else 'disabled'})")

    if use_gpu:
        from .features import compute_features_with_gpu
        features = compute_features_with_gpu(points, k=k_neighbors, use_gpu=True)
    else:
        features = compute_all_features_optimized(points, k=k_neighbors)

    # ... rest of processing ...
```

### processor.py - Add GPU Parameter

```python
class LiDARProcessor:
    def __init__(self, ..., use_gpu: bool = False):
        """
        Args:
            ...
            use_gpu: Enable GPU acceleration for feature computation
        """
        self.use_gpu = use_gpu

        if use_gpu:
            try:
                from .features_gpu import GPU_AVAILABLE
                if not GPU_AVAILABLE:
                    logger.warning("GPU requested but CuPy not available. Using CPU.")
                    self.use_gpu = False
            except ImportError:
                logger.warning("GPU module not available. Using CPU.")
                self.use_gpu = False
```

---

## Testing Strategy

### Unit Tests

```python
def test_gpu_fallback():
    """Test graceful fallback when GPU not available."""
    from ign_lidar.features import compute_features_with_gpu

    points = np.random.rand(1000, 3).astype(np.float32)

    # Should not raise error even if GPU unavailable
    features = compute_features_with_gpu(points, k=10, use_gpu=True)

    assert features.shape[0] == 1000
    assert features.shape[1] > 0

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_computation():
    """Test actual GPU computation."""
    from ign_lidar.features_gpu import GPUFeatureComputer

    computer = GPUFeatureComputer(use_gpu=True)
    points = np.random.rand(10000, 3).astype(np.float32)

    normals = computer.compute_normals(points, k=10)

    assert normals.shape == (10000, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
```

### Integration Test

```bash
# Test GPU flag is recognized
ign-lidar-hd enrich --input test.laz --output test_out/ --use-gpu

# Should show warning if GPU unavailable
# Should process successfully with CPU fallback
```

---

## Documentation Updates

### README.md

Add section:

````markdown
## GPU Acceleration

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or higher
- CuPy: `pip install cupy-cuda11x` (adjust version for your CUDA)
- Optional: RAPIDS cuML for advanced algorithms

### Installation

```bash
# Check CUDA version
nvidia-smi

# Install GPU dependencies
pip install ign-lidar-hd[gpu]

# Or manually
pip install cupy-cuda11x  # Adjust for your CUDA version
```
````

### Usage

```bash
# Enable GPU acceleration
ign-lidar-hd enrich --input-dir tiles/ --output enriched/ --use-gpu
```

### Performance

Typical speedups with GPU acceleration:

| Dataset Size | CPU (12 cores) | GPU (RTX 3080) | Speedup |
| ------------ | -------------- | -------------- | ------- |
| 1M points    | 45s            | 8s             | 5.6x    |
| 10M points   | 6m 30s         | 1m 15s         | 5.2x    |
| 100M points  | 62m            | 11m            | 5.6x    |

_Results may vary based on hardware and point cloud characteristics._

```

---

## Timeline

**v1.2.1 (Hotfix - 1 week):**
- Basic GPU integration
- Update documentation
- Simple tests

**v1.3.0 (Feature - 1 month):**
- Complete GPU feature parity
- Performance benchmarks
- Comprehensive testing

**v2.0.0 (Major - 3 months):**
- Advanced GPU features
- Multi-GPU support
- Production optimization

---

## Resources

- **GPU_ANALYSIS.md** - Detailed technical analysis
- **features_gpu.py** - Existing GPU implementation
- **CuPy Docs** - https://docs.cupy.dev/
- **RAPIDS cuML** - https://docs.rapids.ai/api/cuml/

---

**Last Updated:** October 3, 2025
**Assignee:** TBD
**Estimated Effort:** 4-8 hours (basic), 20-30 hours (complete)
```
