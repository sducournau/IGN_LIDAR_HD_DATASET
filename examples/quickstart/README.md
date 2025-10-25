# 🚀 Quickstart Configurations

**Get started with IGN LiDAR HD in minutes!**

These configurations are designed for **first-time users** and **quick testing**. They work **offline** (no external data sources required) and are optimized for fast results.

---

## 📋 Quick Selection Guide

| Config                | Hardware  | Speed          | Features     | Best For                    |
| --------------------- | --------- | -------------- | ------------ | --------------------------- |
| **00_minimal.yaml**   | CPU       | ⚡⚡⚡ Fastest | ~8 features  | First test, exploration     |
| **01_cpu_basic.yaml** | CPU       | ⚡⚡ Fast      | ~12 features | No GPU, standard use        |
| **02_gpu_basic.yaml** | GPU 12GB+ | ⚡ Very Fast   | ~12 features | GPU available, speed needed |

---

## 🎯 Choose Your Configuration

### Option 1: 00_minimal.yaml (Fastest)

**Perfect for:**

- First-time users exploring the library
- Quick dataset testing
- Learning the workflow
- Limited computational resources

**What you get:**

- ✅ Essential geometric features (normals, curvature, height)
- ✅ Fastest processing time
- ✅ Works on any CPU
- ✅ No external dependencies

**Performance:** ~5-10 min per 18M point tile (CPU)

```bash
ign-lidar-hd process \
  -c examples/quickstart/00_minimal.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Output:**

- Training patches (HDF5): `output/patches/train_*.h5`
- Point count: 8,192 points per patch
- Features: ~8 essential features

---

### Option 2: 01_cpu_basic.yaml (Standard CPU)

**Perfect for:**

- Users without GPU
- LOD2 building classification
- Medium-sized datasets (5-20 tiles)
- Production use without GPU

**What you get:**

- ✅ LOD2 features (~12 features)
- ✅ Multi-scale artifact suppression (2 scales)
- ✅ Parallel CPU processing
- ✅ Better quality than minimal

**Performance:** ~8-12 min per 18M point tile (8-core CPU)

```bash
ign-lidar-hd process \
  -c examples/quickstart/01_cpu_basic.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Tuning for your CPU:**

```bash
# 4-core CPU
ign-lidar-hd process -c examples/quickstart/01_cpu_basic.yaml \
  processor.num_workers=2 \
  input_dir="/path/to/tiles"

# 16-core CPU
ign-lidar-hd process -c examples/quickstart/01_cpu_basic.yaml \
  processor.num_workers=8 \
  input_dir="/path/to/tiles"
```

**Output:**

- Training patches (HDF5): `output/patches/train_*.h5`
- Point count: 16,384 points per patch
- Features: ~12 LOD2 features
- Artifact reduction: 20-30% → 10-15%

---

### Option 3: 02_gpu_basic.yaml (GPU Accelerated)

**Perfect for:**

- Users with NVIDIA GPU (12GB+ VRAM)
- Need for speed (10-20x faster)
- Large datasets (10-50+ tiles)
- Production use with GPU

**What you get:**

- ✅ GPU acceleration (CUDA)
- ✅ LOD2 features (~12 features)
- ✅ Multi-scale artifact suppression
- ✅ 10-20x faster than CPU

**Requirements:**

- NVIDIA GPU with 12GB+ VRAM
- CUDA 11.8+ or CUDA 12.x
- CuPy installed: `pip install cupy-cuda11x` or `cupy-cuda12x`

**Performance:** ~30-60 sec per 18M point tile (RTX 3080+)

```bash
ign-lidar-hd process \
  -c examples/quickstart/02_gpu_basic.yaml \
  input_dir="/path/to/tiles" \
  output_dir="/path/to/output"
```

**Tuning for your GPU:**

```bash
# 12GB VRAM (RTX 3080 Ti, 3060 12GB)
ign-lidar-hd process -c examples/quickstart/02_gpu_basic.yaml \
  processor.gpu_batch_size=20000000 \
  processor.vram_limit_gb=10 \
  input_dir="/path/to/tiles"

# 16GB VRAM (RTX 4080, A5000)
ign-lidar-hd process -c examples/quickstart/02_gpu_basic.yaml \
  processor.gpu_batch_size=30000000 \
  processor.vram_limit_gb=14 \
  input_dir="/path/to/tiles"

# 24GB VRAM (RTX 4090, A6000)
ign-lidar-hd process -c examples/quickstart/02_gpu_basic.yaml \
  processor.gpu_batch_size=50000000 \
  processor.vram_limit_gb=22 \
  input_dir="/path/to/tiles"
```

**Output:**

- Training patches (HDF5): `output/patches/train_*.h5`
- Point count: 16,384 points per patch
- Features: ~12 LOD2 features
- Artifact reduction: 20-30% → 10-15%

---

## 📊 Feature Comparison

### 00_minimal.yaml (8 features)

- normals (nx, ny, nz)
- curvature
- height (Z)
- height_above_ground
- roughness
- density_local

### 01_cpu_basic.yaml & 02_gpu_basic.yaml (12 features)

All minimal features PLUS:

- planarity
- sphericity
- verticality
- linearity
- omnivariance
- anisotropy
- eigenentropy
- change_of_curvature
- Multi-scale weighted versions of key features

---

## 🔄 Next Steps

### After Processing

1. Check output directory:

   ```bash
   ls -lh output/patches/
   ```

2. Inspect a patch file:

   ```python
   import h5py
   with h5py.File('output/patches/train_0.h5', 'r') as f:
       print(f.keys())
       print(f['points'].shape)  # (N, num_features)
   ```

3. Train a model (see examples in docs):
   ```python
   from ign_lidar.datasets import MultiArchDataset
   dataset = MultiArchDataset('output/patches/', architecture='pointnet2')
   ```

### Upgrade to Production

Once you're comfortable with quickstart configs, move to:

- **Production configs** (`examples/production/`) for complete workflows
- **Advanced configs** (`examples/advanced/`) for custom scenarios

---

## 🆘 Troubleshooting

### Common Issues

**Problem:** "CUDA out of memory" (GPU config)  
**Solution:**

```bash
# Reduce batch size
ign-lidar-hd process -c examples/quickstart/02_gpu_basic.yaml \
  processor.gpu_batch_size=15000000 \
  processor.vram_limit_gb=8
```

**Problem:** Processing too slow (CPU config)  
**Solution:**

```bash
# Reduce num_workers or use minimal config
ign-lidar-hd process -c examples/quickstart/00_minimal.yaml
```

**Problem:** "No LAZ files found"  
**Solution:** Check your input_dir path is correct:

```bash
ls /path/to/your/laz/tiles/*.laz
```

**Problem:** Multiprocessing errors on Windows  
**Solution:** Set `num_workers=0`:

```bash
ign-lidar-hd process -c examples/quickstart/01_cpu_basic.yaml \
  processor.num_workers=0
```

---

## 📚 Related Documentation

- **[Production Configs](../production/README.md)** - Complete production workflows
- **[Advanced Configs](../advanced/README.md)** - Custom scenarios
- **[Main Documentation](../../docs/)** - Full library documentation
- **[Configuration Reference](../../docs/docs/configuration/)** - All config options

---

## 💡 Tips

1. **Start minimal:** Always test with `00_minimal.yaml` first on 1-2 tiles
2. **Know your hardware:** Check GPU memory before using GPU configs
3. **Iterate:** Start fast, then add features as needed
4. **Override paths:** Use command-line overrides instead of editing configs
5. **Monitor memory:** Watch memory usage with `htop` (CPU) or `nvidia-smi` (GPU)

---

**Need help?** Open an issue on [GitHub](https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues) or check the [documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/).

**Version:** 3.2.1  
**Last Updated:** October 25, 2025
