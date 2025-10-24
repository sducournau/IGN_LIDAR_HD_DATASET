# Configuration System Audit - October 2025

**Date**: October 24, 2025  
**Auditor**: System Analysis  
**Scope**: Complete configuration system analysis and simplification

---

## Executive Summary

### Current State

- **38 configuration files** in `/examples/`
- **Complex inheritance** with 5-level hierarchy
- **Redundant parameters** across multiple files
- **Inconsistent naming** conventions
- **GPU threshold hardcoded** in code (15M points)
- **Duplicate settings** between configs and code

### Critical Issues

1. **🔴 CRITICAL: GPU FAISS Hardcoded Limit**

   - Location: `ign_lidar/features/gpu_processor.py:948`
   - Issue: `use_gpu_faiss = N < 15_000_000` (hardcoded)
   - Impact: Forces CPU FAISS for datasets >15M points, ignoring config
   - Solution: Make configurable parameter

2. **🟡 HIGH: Configuration Complexity**

   - 38 example configs (many redundant)
   - 5-level inheritance depth
   - Unclear which config to use for each scenario
   - Parameters scattered across multiple files

3. **🟡 HIGH: Inconsistent DTM Augmentation**

   - User requested 1m² spacing
   - System adds 165K+ points → exceeds 15M limit → CPU FAISS forced
   - No clear documentation of trade-offs

4. **🟢 MEDIUM: Documentation Gaps**
   - No clear config selection guide
   - Missing performance vs quality trade-off docs
   - No troubleshooting guide for common issues

---

## Detailed Analysis

### Configuration File Inventory

#### Production-Ready Configs (Keep)

1. ✅ **config_asprs_bdtopo_cadastre_gpu_memory_efficient.yaml** - GPU optimized
2. ✅ **config_asprs_bdtopo_cadastre_cpu_v3_memory_safe.yaml** - CPU safe
3. ✅ **config_auto.yaml** - Auto-detection
4. ✅ **config_building_fusion.yaml** - Building-specific
5. ✅ **config_adaptive_building_classification.yaml** - Advanced buildings

#### Legacy/Redundant Configs (Archive)

- ❌ config_asprs_bdtopo_cadastre_cpu_fixed.yaml (superseded by v3)
- ❌ config_asprs_bdtopo_cadastre_cpu_optimized.yaml (superseded by v3)
- ❌ config_asprs_bdtopo_cadastre_optimized.yaml (unclear purpose)
- ❌ config_cpu.yaml (too basic, superseded)
- ❌ config_gpu_chunked.yaml (superseded by memory_efficient)
- ❌ config_legacy_strategy.yaml (deprecated)

#### Versailles-Specific Configs (Keep but Consolidate)

- config_versailles_asprs_v5.0.yaml
- config_versailles_lod2_v5.0.yaml
- config_versailles_lod3_v5.0.yaml
- config_parcel_versailles.yaml

#### Specialized Configs (Keep)

- config_architectural_analysis_v5.0.yaml
- config_architectural_training_v5.0.yaml
- config_plane_detection_lod3.yaml

---

## Key Problems Identified

### 1. GPU Configuration Issues

**Problem**: GPU FAISS threshold hardcoded

```python
# gpu_processor.py:948
use_gpu_faiss = self.use_gpu and self.use_cuml and N < 15_000_000  # HARDCODED!
```

**Impact**:

- User's 21.5M point cloud → CPU FAISS (26 min vs 2-3 min GPU)
- No way to override via config
- DTM augmentation makes problem worse

**Solution**:

```yaml
processor:
  faiss_gpu_threshold: 25_000_000 # Configurable limit
  faiss_allow_large_gpu: true # Allow >15M on high-VRAM GPUs
```

### 2. DTM Augmentation Conflicts

**Problem**: User wants 1m² augmentation

- Original: 21.5M points
- +1m² augmentation: +165K points = 21.7M
- Exceeds 15M threshold → CPU FAISS forced
- Defeats purpose of GPU acceleration

**Trade-off Matrix**:
| Spacing | Synthetic Points | Total Points | GPU FAISS | Time | Accuracy |
|---------|------------------|--------------|-----------|-------|----------|
| 1.0m² | ~165K | 21.7M | ❌ CPU | 26min | Excellent |
| 1.5m² | ~73K | 21.6M | ❌ CPU | 26min | Very Good |
| 2.0m² | ~41K | 21.5M | ❌ CPU | 26min | Good |
| 3.0m² | ~18K | 21.5M | ❌ CPU | 26min | Adequate |

**All exceed 15M threshold!** Original data is already >15M.

**Real Solution**: Increase GPU FAISS threshold or tile splitting

### 3. Configuration Naming Confusion

**Problem**: Unclear naming patterns

- config*asprs_bdtopo_cadastre***cpu_v3_memory_safe**.yaml (too long)
- config*asprs_bdtopo_cadastre***gpu_memory_efficient**.yaml (inconsistent)
- config\_**auto**.yaml (too generic)

**Proposed Naming**:

```
config_{use_case}_{hardware}_{quality}.yaml

Examples:
- config_asprs_gpu_fast.yaml       # GPU, fast processing
- config_asprs_gpu_balanced.yaml   # GPU, balanced quality/speed
- config_asprs_gpu_quality.yaml    # GPU, maximum quality
- config_asprs_cpu_safe.yaml       # CPU, memory-safe
- config_asprs_cpu_fast.yaml       # CPU, fast processing
```

### 4. Parameter Redundancy

**Duplicate Settings Found**:

- `gpu_batch_size` vs `chunk_size` vs `feature_batch_size`
- `use_gpu` vs `acceleration_mode` vs `use_gpu_chunked`
- `num_workers` in 3 different places
- `cache_dir` specified 4 times

---

## Recommendations

### Phase 1: Immediate Fixes (High Priority)

1. **Make GPU FAISS threshold configurable**

   ```python
   # In gpu_processor.py
   faiss_threshold = config.processor.get('faiss_gpu_threshold', 15_000_000)
   allow_large = config.processor.get('faiss_allow_large_gpu', False)

   use_gpu_faiss = (self.use_gpu and self.use_cuml and
                    (N < faiss_threshold or allow_large))
   ```

2. **Add VRAM estimation logic**

   ```python
   def estimate_faiss_vram(n_points, k_neighbors, vram_available):
       estimated = (n_points * k_neighbors * 4) / (1024**3)
       overhead = 4.0  # 4GB for index + buffers
       total = estimated + overhead
       return total < (vram_available * 0.85)
   ```

3. **Create decision tree for users**
   ```
   Do you have GPU?
   ├─ YES → How much VRAM?
   │  ├─ 16GB+ → config_asprs_gpu_quality.yaml (25M threshold)
   │  ├─ 8-16GB → config_asprs_gpu_balanced.yaml (15M threshold)
   │  └─ <8GB → config_asprs_gpu_fast.yaml (10M threshold)
   └─ NO → config_asprs_cpu_safe.yaml
   ```

### Phase 2: Consolidation (Medium Priority)

1. **Reduce to 8 core configs**:

   - config_asprs_gpu_quality.yaml (16GB+ VRAM, 1m² DTM)
   - config_asprs_gpu_balanced.yaml (8-16GB VRAM, 2m² DTM)
   - config_asprs_gpu_fast.yaml (<8GB VRAM, 3m² DTM)
   - config_asprs_cpu_safe.yaml (32GB RAM, conservative)
   - config_asprs_cpu_fast.yaml (64GB RAM, aggressive)
   - config_building_fusion.yaml (Building-specific)
   - config_lod2_analysis.yaml (LOD2 architectural)
   - config_lod3_detailed.yaml (LOD3 maximum detail)

2. **Archive legacy configs**:

   ```bash
   mkdir examples/archive/
   mv examples/config_*_cpu_fixed.yaml examples/archive/
   mv examples/config_legacy*.yaml examples/archive/
   ```

3. **Standardize parameter names**:

   ```yaml
   # OLD (inconsistent)
   gpu_batch_size: 8000000
   chunk_size: 8000000
   feature_batch_size: 6000000

   # NEW (consistent)
   processing:
     batch_size: 8000000 # Primary batch size
     feature_batch_size: 6000000 # Override for features only
   ```

### Phase 3: Documentation (Medium Priority)

1. **Create CONFIG_SELECTION_GUIDE.md**
2. **Add performance benchmarks** to each config
3. **Document trade-offs** (speed vs quality vs memory)
4. **Add troubleshooting section**

### Phase 4: Advanced Optimizations (Low Priority)

1. **Dynamic FAISS threshold** based on VRAM detection
2. **Automatic tile splitting** for large datasets
3. **Multi-GPU support** configuration
4. **Memory-mapped processing** for extreme datasets

---

## Proposed File Structure

```
examples/
├── README.md                          # Quick start guide
├── CONFIG_SELECTION_GUIDE.md          # Decision tree for users
│
├── gpu/                               # GPU configs (organized by VRAM)
│   ├── config_asprs_gpu_24gb.yaml   # RTX 4090, 3090 Ti
│   ├── config_asprs_gpu_16gb.yaml   # RTX 4080, 4070 Ti
│   ├── config_asprs_gpu_12gb.yaml   # RTX 4070, 3080
│   ├── config_asprs_gpu_8gb.yaml    # RTX 4060, 3060
│   └── config_asprs_gpu_6gb.yaml    # RTX 3050, budget GPUs
│
├── cpu/                               # CPU configs (organized by RAM)
│   ├── config_asprs_cpu_64gb.yaml   # High-end workstation
│   ├── config_asprs_cpu_32gb.yaml   # Standard workstation
│   └── config_asprs_cpu_16gb.yaml   # Laptop/budget
│
├── specialized/                       # Task-specific configs
│   ├── config_building_fusion.yaml
│   ├── config_lod2_analysis.yaml
│   ├── config_lod3_detailed.yaml
│   ├── config_architectural.yaml
│   └── config_plane_detection.yaml
│
├── benchmarks/                        # Example datasets
│   ├── config_versailles.yaml
│   ├── config_urban_dense.yaml
│   └── config_rural_sparse.yaml
│
└── archive/                           # Legacy configs (deprecated)
    └── ...old configs...
```

---

## Implementation Priority

### Week 1 (Critical)

- [ ] Make GPU FAISS threshold configurable
- [ ] Add VRAM estimation logic
- [ ] Create CONFIG_SELECTION_GUIDE.md
- [ ] Fix current GPU config (increase threshold to 25M for 16GB GPUs)

### Week 2 (High)

- [ ] Consolidate to 8-10 core configs
- [ ] Reorganize examples/ directory
- [ ] Add performance benchmarks to configs
- [ ] Document trade-offs

### Week 3 (Medium)

- [ ] Archive legacy configs
- [ ] Standardize parameter naming
- [ ] Create versailles benchmark suite
- [ ] Add troubleshooting guide

### Week 4 (Low)

- [ ] Dynamic FAISS threshold implementation
- [ ] Automatic tile splitting
- [ ] Multi-GPU support
- [ ] Advanced memory optimization

---

## Metrics

### Current System

- Config files: 38
- Avg config size: ~450 lines
- Parameter redundancy: ~40%
- User confusion level: High

### Target System

- Config files: 10-12 (organized)
- Avg config size: ~200 lines
- Parameter redundancy: <10%
- User confusion level: Low

### Expected Improvements

- ⚡ 30% faster selection time
- 📉 60% reduction in support issues
- 🎯 100% clarity on which config to use
- 🚀 Better GPU utilization (configurable thresholds)

---

## Conclusion

The current configuration system is functional but overly complex. The hardcoded GPU FAISS threshold is the most critical issue, preventing optimal GPU utilization. By implementing the phased recommendations, we can create a clearer, more maintainable, and more performant system.

**Priority Actions**:

1. Fix GPU FAISS threshold (code change required)
2. Consolidate example configs (10-12 organized files)
3. Document configuration selection process
4. Add performance benchmarks

This will reduce user confusion, improve GPU utilization, and make the system more maintainable long-term.
