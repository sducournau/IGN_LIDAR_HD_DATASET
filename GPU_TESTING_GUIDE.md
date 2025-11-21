# GPU Testing Guide - IGN LiDAR HD

**Last Updated:** 2025-11-21

## ⚡ Critical: Always Use `ign_gpu` Environment

**All GPU-related development and testing MUST use the `ign_gpu` conda environment.**

### Why `ign_gpu`?

The `ign_gpu` environment contains:

- **CuPy** - GPU-accelerated NumPy replacement
- **RAPIDS cuML** - GPU machine learning
- **RAPIDS cuSpatial** - GPU spatial operations
- **FAISS-GPU** - GPU-accelerated similarity search

The base conda environment does **NOT** have these libraries.

---

## 🚀 Quick Start

### Activate Environment

```bash
# Activate for interactive work
conda activate ign_gpu

# Verify activation
conda env list  # Should show * next to ign_gpu
```

### Run Tests

```bash
# Method 1: After activation
conda activate ign_gpu
python -m pytest tests/test_gpu_*.py -v

# Method 2: Single command (no activation needed)
conda run -n ign_gpu python -m pytest tests/test_gpu_*.py -v
```

---

## 📋 GPU Test Suite

### All GPU Tests

```bash
conda run -n ign_gpu python -m pytest tests/test_gpu_*.py -v
```

### Individual Test Suites

```bash
# GPU eigenvalue optimization
conda run -n ign_gpu python -m pytest tests/test_gpu_eigenvalue_optimization.py -v

# FAISS index optimization
conda run -n ign_gpu python -m pytest tests/test_faiss_optimization.py -v

# Multi-scale GPU connection
conda run -n ign_gpu python -m pytest tests/test_multi_scale_gpu_connection.py -v
```

### With Coverage

```bash
conda run -n ign_gpu python -m pytest tests/test_gpu_*.py -v \
  --cov=ign_lidar.features.gpu_processor \
  --cov-report=html
```

---

## 🔧 GPU Benchmarks

### Performance Benchmarks

```bash
# GPU vs CPU benchmark
conda run -n ign_gpu python scripts/benchmark_gpu.py

# FAISS GPU optimization
conda run -n ign_gpu python scripts/benchmark_faiss_gpu_optimization.py

# Large-scale processing
conda run -n ign_gpu python scripts/benchmark_large_scale.py
```

---

## 🐛 Troubleshooting

### Check GPU Availability

```bash
conda run -n ign_gpu python -c "
import cupy as cp
print(f'GPU Available: {cp.cuda.is_available()}')
print(f'Device: {cp.cuda.Device().name}')
mem = cp.cuda.Device().mem_info
print(f'Memory: {mem[0]/1e9:.1f}/{mem[1]/1e9:.1f} GB free')
"
```

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'cupy'"

**Problem:** Not using `ign_gpu` environment

**Solution:**

```bash
conda activate ign_gpu
# OR
conda run -n ign_gpu python your_script.py
```

#### 2. "CUDA out of memory"

**Problem:** GPU memory exhausted

**Solution:**

- Use GPU chunked mode: `use_gpu_chunked=True`
- Reduce batch size: `gpu_batch_size=500000`
- Clear GPU memory: `cp.get_default_memory_pool().free_all_blocks()`

#### 3. "No CUDA-capable device detected"

**Problem:** No GPU or driver issues

**Solution:**

- Check NVIDIA drivers: `nvidia-smi`
- Verify CUDA installation
- Code will automatically fall back to CPU

---

## 📦 Environment Management

### View Environment Info

```bash
# List environments
conda env list

# Show ign_gpu packages
conda list -n ign_gpu

# Show only GPU packages
conda list -n ign_gpu | grep -E 'cupy|cuml|cuspatial|faiss-gpu'
```

### Update Environment

```bash
# Update from environment file
conda env update -n ign_gpu -f conda-recipe/environment_gpu.yml

# Install additional packages
conda activate ign_gpu
conda install -c conda-forge <package>
```

### Create New GPU Environment

```bash
# From environment file
conda env create -f conda-recipe/environment_gpu.yml

# Manually
conda create -n ign_gpu python=3.10
conda activate ign_gpu
conda install -c conda-forge cupy cudatoolkit=11.8
conda install -c rapidsai -c conda-forge cuml cuspatial
conda install -c conda-forge faiss-gpu
```

---

## 🔍 Verification Checklist

Before committing GPU code, verify:

- [ ] All GPU tests passing in `ign_gpu` environment
- [ ] CPU fallback working correctly
- [ ] Error messages are clear and actionable
- [ ] Memory usage is reasonable
- [ ] Performance improvement documented

### Verification Commands

```bash
# 1. Activate environment
conda activate ign_gpu

# 2. Run GPU tests
python -m pytest tests/test_gpu_*.py -v

# 3. Check GPU utilization
nvidia-smi

# 4. Benchmark performance
python scripts/benchmark_gpu.py

# 5. Memory profiling
python -m memory_profiler your_gpu_script.py
```

---

## 📚 Related Documentation

- **Copilot Instructions:** `.github/copilot-instructions.md` - GPU development rules
- **TODO List:** `TODO_OPTIMIZATIONS.md` - GPU optimization tasks
- **GPU Report:** `GPU_IMPLEMENTATION_REPORT.md` - Implementation details
- **Optimization Summary:** `GPU_OPTIMIZATION_SUMMARY.md` - Performance results

---

## 🎯 Best Practices

1. **Always specify environment:**

   ```bash
   conda run -n ign_gpu python script.py
   ```

2. **Check GPU in code:**

   ```python
   try:
       import cupy as cp
       gpu_available = cp.cuda.is_available()
   except ImportError:
       gpu_available = False
   ```

3. **Graceful fallback:**

   ```python
   if gpu_available:
       result = gpu_function(data)
   else:
       result = cpu_function(data)
   ```

4. **Clear error messages:**
   ```python
   if not gpu_available:
       logger.warning("GPU not available - install CuPy")
       logger.info("  conda activate ign_gpu")
   ```

---

## ✅ Quick Reference Card

| Task              | Command                                                                         |
| ----------------- | ------------------------------------------------------------------------------- |
| Activate GPU env  | `conda activate ign_gpu`                                                        |
| Run GPU tests     | `conda run -n ign_gpu pytest tests/test_gpu_*.py -v`                            |
| Check GPU         | `conda run -n ign_gpu python -c "import cupy; print(cupy.cuda.is_available())"` |
| GPU benchmark     | `conda run -n ign_gpu python scripts/benchmark_gpu.py`                          |
| View GPU memory   | `nvidia-smi`                                                                    |
| List GPU packages | `conda list -n ign_gpu \| grep -E 'cupy\|faiss'`                                |

---

**Remember: When in doubt, use `ign_gpu`! 🚀**
