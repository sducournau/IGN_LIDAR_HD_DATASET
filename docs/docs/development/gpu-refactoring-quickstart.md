---
sidebar_position: 1
title: GPU Refactoring Quick Start
description: Developer guide for implementing GPU-Core bridge module
tags: [development, gpu, refactoring, cupy, performance]
---

# Quick Start: GPU Refactoring Implementation

:::info For Developers
This guide is for developers working on Phase 1 GPU refactoring.  
**Estimated Time:** 1 week  
**Date:** October 2025
:::

---

## 🎯 Your Mission

Implement the GPU-Core Bridge module to eliminate code duplication while maintaining GPU performance.

**Why this matters:** Currently 71% of GPU feature code is duplicated. You're fixing that.

---

## 📋 Before You Start (Day 0)

### 1. Read These Documents (2 hours)

**Required reading:**

1. `AUDIT_SUMMARY.md` (15 min) - Understand the problem
2. `AUDIT_VISUAL_SUMMARY.md` (10 min) - See the architecture
3. `IMPLEMENTATION_GUIDE_GPU_BRIDGE.md` (30 min) - Your implementation guide

**Optional:**

- `AUDIT_GPU_REFACTORING_CORE_FEATURES.md` - Deep technical details

### 2. Set Up Environment (30 min)

```bash
# Clone repo if needed
cd /path/to/IGN_LIDAR_HD_DATASET

# Create feature branch
git checkout -b feature/gpu-core-bridge

# Verify GPU environment
python -c "import cupy as cp; print(f'CuPy: {cp.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"

# Install dependencies if needed
pip install cupy-cuda11x  # or cupy-cuda12x
pip install pytest pytest-benchmark

# Verify tests work
pytest tests/ -v
```

### 3. Understand the Codebase (1 hour)

```bash
# Key files to review
cat ign_lidar/features/core/__init__.py
cat ign_lidar/features/core/eigenvalues.py
cat ign_lidar/features/features_gpu_chunked.py | head -100
```

---

## 📅 Week 1 Schedule

### Day 1: Setup & Module Structure

**Morning (3 hours):**

- [ ] Create `ign_lidar/features/core/gpu_bridge.py`
- [ ] Add module docstring and imports
- [ ] Create `GPUCoreBridge` class skeleton
- [ ] Implement `__init__` method

**Afternoon (3 hours):**

- [ ] Implement `compute_eigenvalues_gpu()` method
- [ ] Implement CPU fallback `_compute_eigenvalues_cpu()`
- [ ] Add error handling
- [ ] Test basic functionality manually

**Code to write:** ~150 lines

**Reference:** `IMPLEMENTATION_GUIDE_GPU_BRIDGE.md` Step 1

---

### Day 2: GPU Implementation

**Morning (3 hours):**

- [ ] Implement `_compute_eigenvalues_batched_gpu()`
- [ ] Handle cuSOLVER batch size limits
- [ ] Add GPU memory management
- [ ] Add logging

**Afternoon (3 hours):**

- [ ] Implement `compute_eigenvalue_features_gpu()`
- [ ] Integrate with core module
- [ ] Add convenience function
- [ ] Test with small datasets

**Code to write:** ~200 lines

**Test manually:**

```python
from ign_lidar.features.core.gpu_bridge import GPUCoreBridge
import numpy as np

# Small test
points = np.random.rand(1000, 3).astype(np.float32)
neighbors = np.random.randint(0, 1000, size=(1000, 20))

bridge = GPUCoreBridge(use_gpu=True)
eigenvalues = bridge.compute_eigenvalues_gpu(points, neighbors)
print(f"Shape: {eigenvalues.shape}")  # Should be (1000, 3)
print(f"Sample: {eigenvalues[0]}")     # Should be 3 values
```

---

### Day 3: Testing Infrastructure

**Morning (3 hours):**

- [ ] Create `tests/test_gpu_bridge.py`
- [ ] Write test fixtures
- [ ] Implement basic unit tests
- [ ] Test CPU fallback

**Afternoon (3 hours):**

- [ ] Test GPU vs CPU consistency
- [ ] Test batching with large datasets
- [ ] Test error handling
- [ ] Test integration with core module

**Code to write:** ~400 lines

**Run tests:**

```bash
pytest tests/test_gpu_bridge.py -v
pytest tests/test_gpu_bridge.py::TestGPUCoreBridge -v
```

---

### Day 4: Performance & Validation

**Morning (3 hours):**

- [ ] Create `scripts/benchmark_gpu_bridge.py`
- [ ] Run benchmarks with different sizes
- [ ] Compare GPU vs CPU performance
- [ ] Verify speedup >= 8×

**Afternoon (3 hours):**

- [ ] Optimize if needed
- [ ] Test with real data
- [ ] Memory profiling
- [ ] Fix any issues

**Run benchmarks:**

```bash
python scripts/benchmark_gpu_bridge.py
python scripts/benchmark_gpu_bridge.py --sizes 10000 100000 500000
```

**Expected results:**

```
Dataset: 100,000 points, k=20
  CPU Time: 2.5s ± 0.1s
  GPU Time: 0.25s ± 0.02s
  Speedup: 10.0×
  ✅ Performance target met (>= 8×)
```

---

### Day 5: Documentation & Review

**Morning (2 hours):**

- [ ] Update `ign_lidar/features/core/__init__.py` exports
- [ ] Write docstrings for all functions
- [ ] Add usage examples
- [ ] Update CHANGELOG

**Afternoon (2 hours):**

- [ ] Code self-review
- [ ] Run full test suite
- [ ] Prepare pull request
- [ ] Document any issues

**Final checks:**

```bash
# All tests pass
pytest tests/ -v

# Benchmarks meet target
python scripts/benchmark_gpu_bridge.py

# Code quality
# (if using black/flake8)
black ign_lidar/features/core/gpu_bridge.py
flake8 ign_lidar/features/core/gpu_bridge.py
```

---

## 🔧 Implementation Tips

### GPU Memory Management

```python
# Always clean up GPU memory
def compute_something_gpu(self, data):
    data_gpu = cp.asarray(data)
    try:
        result_gpu = process(data_gpu)
        result = cp.asnumpy(result_gpu)
        return result
    finally:
        # Cleanup happens even if error
        del data_gpu
        if 'result_gpu' in locals():
            del result_gpu
```

### Batching Pattern

```python
# Standard batching pattern
batch_size = 500_000  # cuSOLVER limit
num_batches = (N + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, N)

    # Process batch
    batch_result = process_batch(data[start:end])
    results[start:end] = batch_result
```

### Testing Pattern

```python
# Always test GPU vs CPU consistency
def test_consistency():
    bridge_gpu = GPUCoreBridge(use_gpu=True)
    bridge_cpu = GPUCoreBridge(use_gpu=False)

    result_gpu = bridge_gpu.compute_eigenvalues_gpu(points, neighbors)
    result_cpu = bridge_cpu.compute_eigenvalues_gpu(points, neighbors)

    np.testing.assert_allclose(
        result_gpu, result_cpu,
        rtol=1e-5, atol=1e-7
    )
```

---

## 🐛 Common Issues & Solutions

### Issue 1: CuPy Import Error

```python
ImportError: No module named 'cupy'
```

**Solution:**

```bash
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

### Issue 2: cuSOLVER Batch Size Error

```
cupy._core.linalg.LinAlgError: cuSOLVER error
```

**Solution:** Implement batching for large datasets

```python
# Use max batch size of 500K
if N > 500_000:
    result = self._compute_eigenvalues_batched_gpu(data, N)
```

### Issue 3: GPU Out of Memory

```
cupy.cuda.memory.OutOfMemoryError
```

**Solution:** Reduce batch size or add cleanup

```python
# Add explicit cleanup
cp.get_default_memory_pool().free_all_blocks()
```

### Issue 4: Numerical Differences GPU vs CPU

```python
# Small differences are expected due to floating-point
# Use appropriate tolerances
np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
```

---

## ✅ Success Checklist

### Code Complete

- [ ] `gpu_bridge.py` created (~500 lines)
- [ ] All methods implemented
- [ ] Error handling added
- [ ] Logging configured

### Tests Complete

- [ ] Unit tests written (~400 lines)
- [ ] All tests passing
- [ ] GPU vs CPU consistency verified
- [ ] Edge cases covered

### Performance Validated

- [ ] Benchmarks run
- [ ] Speedup >= 8× confirmed
- [ ] Memory usage acceptable
- [ ] No performance regression

### Documentation Complete

- [ ] Docstrings for all functions
- [ ] Usage examples included
- [ ] Core module exports updated
- [ ] CHANGELOG updated

### Ready for Review

- [ ] Code self-reviewed
- [ ] All tests passing
- [ ] No linting errors
- [ ] PR prepared

---

## 📝 Daily Progress Template

Copy this for daily updates:

```markdown
## Day X Progress

**What I completed:**

-
-
- **What I learned:**

-
- **Issues encountered:**

-
- **Blockers:**

- **Tomorrow's plan:**

-
-
- **Time spent:** X hours
```

---

## 🆘 Getting Help

### Quick Questions

- Review `IMPLEMENTATION_GUIDE_GPU_BRIDGE.md`
- Check existing code in `features_gpu_chunked.py`
- Look at core module implementations

### Technical Issues

- Review `AUDIT_GPU_REFACTORING_CORE_FEATURES.md` Section 2
- Check error handling patterns in existing code
- Consult GPU optimization guide

### Architecture Questions

- Review `AUDIT_VISUAL_SUMMARY.md`
- Check data flow diagrams
- Review current vs. proposed architecture

---

## 🎓 Learning Resources

### CuPy Documentation

- Official docs: https://docs.cupy.dev/
- GPU arrays: Like NumPy but on GPU
- Key functions: `cp.asarray()`, `cp.asnumpy()`

### Eigenvalue Computation

- `np.linalg.eigvalsh()` - CPU version
- `cp.linalg.eigvalsh()` - GPU version
- Returns eigenvalues sorted ascending

### cuSOLVER Limits

- Maximum batch size: ~500K matrices
- For larger: implement batching
- Error: `CUSOLVER_STATUS_INVALID_VALUE`

---

## 🚀 After Phase 1

### If Successful

1. Request code review
2. Merge to main
3. Start Phase 2 (eigenvalue integration)

### If Issues Found

1. Document issues
2. Propose solutions
3. Adjust timeline if needed

### Metrics to Report

- Code written: ~XXX lines
- Tests written: ~XXX tests
- Test coverage: XX%
- GPU speedup: XX×
- Time spent: XX hours

---

## 📞 Contact

**Questions?**

- Technical Lead: [Name]
- Code Review: [Name]
- GPU Expert: [Name]

**Resources:**

- Project docs: `/docs/`
- Implementation guide: `IMPLEMENTATION_GUIDE_GPU_BRIDGE.md`
- Audit: `AUDIT_GPU_REFACTORING_CORE_FEATURES.md`

---

**Good luck! You're fixing 71% code duplication. This is important work! 🎉**

---

**Quick Links:**

- 📖 Full Guide: `IMPLEMENTATION_GUIDE_GPU_BRIDGE.md`
- 📊 Overview: `AUDIT_SUMMARY.md`
- ✅ Checklist: `AUDIT_CHECKLIST.md`
- 🎨 Diagrams: `AUDIT_VISUAL_SUMMARY.md`
