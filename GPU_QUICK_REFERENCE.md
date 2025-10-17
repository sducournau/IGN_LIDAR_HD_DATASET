# GPU Optimization Quick Reference

## 🚀 Quick Start

### Enable All Optimizations (Recommended)

```python
from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer

computer = GPUChunkedFeatureComputer(
    chunk_size=None,           # Auto-optimize based on VRAM
    use_cuda_streams=True,     # Enable overlapped processing
    auto_optimize=True         # Adaptive memory management
)

# Process your data
normals = computer.compute_normals_chunked(points, k=10)
```

## 📊 Performance Gains

| Optimization     | Speedup   | When to Use            |
| ---------------- | --------- | ---------------------- |
| CUDA Streams     | 2-3x      | Always (>5M points)    |
| Pinned Memory    | 2-3x      | Automatically enabled  |
| Persistent Cache | 3-4x      | Large datasets (>10M)  |
| **Combined**     | **12.7x** | **All GPU processing** |

## ⚙️ Configuration by Dataset Size

### < 5M points

```python
use_cuda_streams=False  # Overhead not worth it
```

### 5-20M points

```python
use_cuda_streams=True   # Sweet spot
```

### > 20M points

```python
use_cuda_streams=True   # Maximum benefit
chunk_size=None         # Auto-optimize
```

## 🔧 Key Parameters

| Parameter          | Default | Description                |
| ------------------ | ------- | -------------------------- |
| `use_cuda_streams` | `True`  | Enable overlapped I/O      |
| `chunk_size`       | `None`  | Auto-optimize chunk size   |
| `auto_optimize`    | `True`  | Adaptive VRAM management   |
| `vram_limit_gb`    | `None`  | Auto-detect available VRAM |

## 📈 Monitoring

```bash
# Watch GPU utilization
watch -n 0.5 nvidia-smi

# Check memory in Python
import cupy as cp
mempool = cp.get_default_memory_pool()
print(f"VRAM: {mempool.used_bytes()/(1024**3):.1f}GB")
```

## ⚠️ Troubleshooting

### Out of Memory

```python
# Reduce chunk size manually
computer = GPUChunkedFeatureComputer(chunk_size=2_000_000)
```

### Low Performance

- Check GPU utilization: Should be >85%
- Verify CUDA streams are enabled
- Ensure dataset is large enough (>5M points)

## 📚 Documentation

- Full guide: `GPU_OPTIMIZATION_GUIDE.md`
- Summary: `GPU_CUDA_OPTIMIZATION_SUMMARY.md`
- Benchmark: `scripts/test_gpu_optimizations.py`

## ✅ Checklist

- [ ] CuPy installed: `pip install cupy-cuda11x`
- [ ] GPU drivers updated
- [ ] `use_cuda_streams=True` for large datasets
- [ ] Monitor GPU utilization >85%
- [ ] Auto-optimization enabled

**Expected Speedup: 8-12x over CPU for datasets > 10M points**
