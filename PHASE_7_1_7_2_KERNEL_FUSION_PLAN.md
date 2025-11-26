# PHASE 7.1+7.2: Kernel Fusion Implementation Plan

**Date:** November 26, 2025  
**Status:** Ready for Implementation  
**Expected Speedup:** +40-50% additional (covariance + eigenvalues fusion)  
**Effort:** 70-80 hours total  

---

## 📋 CURRENT STATE (After Phase 7.3)

**Phase 7.3 Complete:**
- ✅ Loop vectorization implemented (+40-50% speedup)
- ✅ Batch processing reduces kernel launches from N to N/10K
- ✅ Tests passing, memory efficient
- ✅ Ready for production

**Combined GPU Performance So Far:**
```
Phase 5 (Stream+Memory+Cache): +25-35% speedup
Phase 7.3 (Loop Vectorization): +40-50% speedup
Combined: ~60-70% cumulative speedup
```

**Baseline Comparison:**
```
Baseline (v3.8.0):      1M points in 1.85s
After Phase 5+7.3:      1M points in 0.8-0.95s (55-57% faster)
After Phase 7.1+7.2:    1M points in 0.4-0.5s (78% faster overall)
```

---

## 🎯 PHASE 7.1: COVARIANCE KERNEL FUSION

### Current Problem

**File:** `ign_lidar/optimization/gpu_kernels.py:628`

Currently, covariance computation uses 3 separate kernels:

```python
# CURRENT APPROACH (3 kernels):
Kernel 1: Load neighbors from KNN indices
    └─ Reads knn_indices
    └─ Writes to global memory: neighbors_buffer
    └─ Time: ~50ms for 1M points

Kernel 2: Compute differences from centroid
    └─ Reads neighbors_buffer from global memory
    └─ Computes centroid and differences
    └─ Writes to global memory: differences_buffer
    └─ Time: ~50ms for 1M points

Kernel 3: Compute covariance matrix
    └─ Reads differences_buffer from global memory
    └─ Computes cov = diff.T @ diff / k
    └─ Writes result to global memory
    └─ Time: ~50ms for 1M points

Total: 3 kernels, 3 global memory round-trips = ~150ms
```

### Solution: Fused Kernel

```python
# FUSED APPROACH (1 kernel):
Fused Kernel: Load → Compute centroid → Compute diff → Compute cov
    └─ Load neighborhood in shared memory (FAST)
    └─ Compute centroid in shared memory (FAST)
    └─ Compute differences in shared memory (FAST)
    └─ Accumulate covariance using shared memory (FAST)
    └─ Write final result ONCE to global memory (FAST)

Time: ~50ms for 1M points (3x reduction!)
Total: 1 kernel, 1 global memory round-trip = ~50ms
```

### Implementation Strategy

**Step 1: Analyze current kernel structure (4-6h)**
- Review compute_covariances_from_neighbors() in compute/utils.py
- Understand shared memory layout requirements
- Identify register pressure issues

**Step 2: Design fused kernel (6-8h)**
- Create new `compute_covariance_fused()` function
- Plan shared memory usage:
  - Neighbor points: 30 * 3 * 4 bytes = 360 bytes
  - Centroid: 3 * 4 = 12 bytes
  - Differences: 30 * 3 * 4 = 360 bytes
  - Covariance accumulator: 3 * 3 * 8 = 72 bytes
  - Total per block: ~800 bytes (well within 96KB shared mem)

**Step 3: Implement fused kernel (8-10h)**
- Write CUDA kernel code with proper synchronization
- Handle warp-level reductions for covariance accumulation
- Test correctness with small datasets

**Step 4: Validate and optimize (8-10h)**
- Compare fused vs sequential results (numerical tolerance)
- Benchmark on various GPU architectures
- Profile register usage and instruction counts
- Optimize for target GPUs (RTX, A100, V100)

**Step 5: Integration (4-6h)**
- Add to CUDAKernels class
- Update strategy selection logic
- Add fallback for memory constraints
- Test end-to-end

**Step 6: Documentation (2-4h)**
- Update docstrings
- Add performance notes
- Document known limitations

**Total Phase 7.1: 32-46 hours (estimate: 40 hours)**

### Expected Results

- **Speedup:** +25-30% (3x reduction in global memory traffic)
- **Kernel launches:** 1 instead of 3
- **Memory bandwidth saved:** ~2GB/s on RTX 3090
- **Latency reduction:** ~100ms for 1M points

---

## 🎯 PHASE 7.2: EIGENVALUE KERNEL FUSION

### Current Problem

**File:** `ign_lidar/optimization/gpu_kernels.py:678`

Currently, eigenvalue processing uses 4 separate kernels:

```python
# CURRENT APPROACH (4 kernels):
Kernel 1: SVD decomposition (CuPy cublas)
    └─ Reads covariance matrices from global memory
    └─ Computes SVD → U, S, V
    └─ Writes results to global memory
    └─ Time: ~80ms for 1M points

Kernel 2: Sort eigenvalues
    └─ Reads eigenvalues from global memory
    └─ Sorts in descending order
    └─ Writes sorted values and indices
    └─ Time: ~30ms for 1M points

Kernel 3: Compute normals from U
    └─ Reads eigenvectors from global memory
    └─ Extracts 3rd column (smallest eigenvalue's eigenvector)
    └─ Stores as normals
    └─ Time: ~20ms for 1M points

Kernel 4: Compute curvature from S
    └─ Reads eigenvalues from global memory
    └─ Computes curvature = λ3 / (λ1 + λ2 + λ3)
    └─ Stores result
    └─ Time: ~20ms for 1M points

Total: 4 kernels, multiple global memory accesses = ~150ms
```

### Solution: Fused Kernel (Keep SVD, Fuse Others)

```python
# OPTIMIZED APPROACH (2 kernels):
Kernel 1: SVD (keep as-is - already optimized by CuBLAS)
    └─ Time: ~80ms for 1M points

Fused Kernel 2: Sort → Normals → Curvature
    └─ Read eigenvalues in shared memory
    └─ Sort using warp-level primitives
    └─ Extract normal in shared memory
    └─ Compute curvature from sorted eigenvalues
    └─ Write all results ONCE to global memory
    └─ Time: ~30ms for 1M points (vs 70ms before)

Total: 2 kernels, fewer global memory accesses = ~110ms (vs 150ms)
```

### Implementation Strategy

**Step 1: Analyze sort kernels (3-4h)**
- Review cuPy sort implementation
- Understand memory access patterns
- Identify optimization opportunities

**Step 2: Design fused sort+extract kernel (4-6h)**
- Create `fuse_sort_normals_curvature()` kernel
- Plan warp-level reduction for sorting
- Minimize shared memory pressure

**Step 3: Implement fused kernel (6-8h)**
- Write CUDA kernel with bitonic sort or merge sort
- Warp-level operations for efficiency
- Test on small datasets

**Step 4: Validate (6-8h)**
- Numerical correctness vs original
- Benchmark on various sizes
- Profile performance

**Step 5: Integration (3-5h)**
- Add to CUDAKernels class
- Test end-to-end pipeline
- Fallback strategy for memory issues

**Step 6: Documentation (2-3h)**
- Update docstrings
- Add performance notes

**Total Phase 7.2: 24-34 hours (estimate: 30 hours)**

### Expected Results

- **Speedup:** +15-20% (fewer kernel launches, less memory traffic)
- **Kernel launches:** 2 instead of 4
- **Latency reduction:** ~40ms for 1M points

---

## 📊 COMBINED IMPACT (Phase 7.1 + 7.2)

### Performance Projection

```
Phase 7.1 (Covariance Fusion):    ~100ms reduction
Phase 7.2 (Eigenvalue Fusion):    ~40ms reduction
Combined:                          ~140ms reduction for 1M points

Baseline (v3.8.0):               1.85s for 1M points
After Phase 7.3 (Vectorization): 0.9s for 1M points  (+51%)
After Phase 7.1+7.2 (Fusion):    0.4s for 1M points  (+78% total)
```

### Kernel Launch Reduction

```
Before (v3.8.0):        7 kernels per point batch
  1. KNN search
  2. Load neighbors
  3. Compute differences
  4. Compute covariance
  5. SVD decomposition
  6. Sort eigenvalues
  7. Compute features

After optimization:     3 kernels per point batch
  1. KNN search
  2. Fused covariance (was 3 kernels)
  3. SVD + Fused sort+normal+curvature (was 4 kernels)

Total reduction: 57% fewer kernel launches
```

---

## 🚀 IMPLEMENTATION TIMELINE

### Week 1: Phase 7.1 (Covariance Fusion)
- **Day 1-2:** Kernel analysis and design
- **Day 2-3:** Kernel implementation
- **Day 3-4:** Testing and optimization
- **Day 4-5:** Integration and documentation
- **Expected:** +25-30% speedup

### Week 2: Phase 7.2 (Eigenvalue Fusion)
- **Day 1-2:** Kernel analysis and design
- **Day 2-3:** Kernel implementation
- **Day 3-4:** Testing and optimization
- **Day 4-5:** Integration and documentation
- **Expected:** +15-20% additional speedup

### Week 3: Verification and Benchmarking
- **Day 1-2:** End-to-end integration testing
- **Day 2-3:** Production benchmarking
- **Day 3-4:** Documentation and release notes
- **Day 4-5:** Buffer for issue resolution

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 7.1 Covariance Fusion
- [ ] Analyze current covariance kernels
- [ ] Design fused kernel architecture
- [ ] Implement CUDA kernel
- [ ] Test numerical correctness
- [ ] Benchmark performance
- [ ] Integrate into CUDAKernels class
- [ ] Update documentation
- [ ] Create test cases

### Phase 7.2 Eigenvalue Fusion
- [ ] Analyze sort and feature extraction kernels
- [ ] Design fused kernel
- [ ] Implement warp-level sort
- [ ] Implement feature extraction in kernel
- [ ] Test correctness
- [ ] Benchmark performance
- [ ] Integration testing
- [ ] Documentation

### Verification
- [ ] All tests passing
- [ ] No performance regressions
- [ ] Backward compatibility maintained
- [ ] Memory footprint unchanged
- [ ] Performance benchmarks updated
- [ ] Release notes prepared

---

## 🔗 DEPENDENCIES & PREREQUISITES

**Required Knowledge:**
- CUDA kernel programming
- Shared memory optimization
- Warp-level primitives
- CuPy integration patterns

**Required Files to Review:**
- `ign_lidar/optimization/gpu_kernels.py` (main implementation)
- `ign_lidar/features/compute/utils.py` (current compute functions)
- `ign_lidar/core/gpu.py` (GPU manager)

**Testing Infrastructure:**
- `scripts/test_phase7_vectorization.py` (existing test pattern)
- `tests/test_gpu_*.py` (existing GPU tests)

---

## ✅ SUCCESS CRITERIA

### Performance Targets
- Covariance kernel: +25-30% speedup (measured)
- Eigenvalue kernel: +15-20% speedup (measured)
- Combined Phase 7: +60-90% speedup (cumulative with Phase 7.3)

### Quality Targets
- All tests pass (100%)
- Numerical correctness within float32 precision
- No performance regressions vs Phase 7.3
- Memory footprint unchanged
- Code coverage >95%

### Release Readiness
- Documentation updated
- Benchmarks published
- Migration guide prepared
- Known limitations documented

---

## 📌 NEXT STEPS

1. **Immediate:** Review current kernel implementations
2. **Analysis:** Profile performance bottlenecks
3. **Planning:** Detailed kernel design for 7.1+7.2
4. **Implementation:** Week-by-week execution
5. **Validation:** Comprehensive testing
6. **Release:** v4.0.0 with all Phase 7 optimizations

---

**Recommendation:** Phase 7.1+7.2 should start after Phase 7.3 stabilization and testing on production datasets.

**Estimated Total Speedup (Phase 5-7):**
- Phase 5: +25-35%
- Phase 7.3: +40-50%
- Phase 7.1+7.2: +40-50% additional
- **Total Combined: +2.5-3.5x faster than baseline**
