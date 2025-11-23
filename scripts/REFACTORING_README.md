# Code Refactoring Scripts

This directory contains automated refactoring scripts to improve code quality, reduce duplication, and optimize GPU performance.

## 📊 Audit Results

- **Duplicate lines:** ~22,900 (11.7% of codebase)
- **Duplicate functions:** 173
- **GPU bottlenecks:** 90+ transfers per tile
- **GPU utilization:** 60-70% (target: 85-95%)

See full report: [`../docs/audit_reports/CODE_QUALITY_AUDIT_NOV22_2025.md`](../docs/audit_reports/CODE_QUALITY_AUDIT_NOV22_2025.md)

## 🚀 Quick Start

```bash
# View audit summary
./scripts/audit_quickstart.sh

# Or manually review
cat docs/audit_reports/REFACTORING_SUMMARY_NOV22_2025.md
```

## 🛠️ Refactoring Scripts

### Phase 1: Remove Critical Duplications

**Script:** `refactor_phase1_remove_duplicates.py`

**What it does:**

- Removes 7 duplicate `compute_normals()` implementations
- Consolidates to single canonical version in `features/compute/normals.py`
- Adds deprecation warnings for backward compatibility
- Removes duplicate `validate_normals()`
- Deprecates `GPUProcessor` in favor of `FeatureOrchestrator`

**Expected benefits:**

- ~400 lines removed
- Easier maintenance
- Consistent behavior across codebase

**Usage:**

```bash
# Run refactoring (creates backups automatically)
python scripts/refactor_phase1_remove_duplicates.py

# Test changes
pytest tests/test_features*.py -v

# Review changes
git diff

# If problems, restore from backups
# (*.py.backup files created automatically)
```

**Migration guide created:** `docs/migration_guides/compute_normals_consolidation.md`

---

### Phase 2: Optimize GPU Transfers

**Script:** `refactor_phase2_optimize_gpu.py`

**What it does:**

- Creates `GPUTransferProfiler` for monitoring CPU↔GPU transfers
- Adds `return_gpu` parameter to `KNNEngine.search()` for lazy transfers
- Integrates `CUDAStreamManager` into `FeatureOrchestrator`
- Creates benchmark script for measuring improvements

**Expected benefits:**

- Reduce transfers from 90+ to <5 per tile
- +20-30% GPU throughput
- 85-95% GPU utilization

**Usage:**

```bash
# Step 1: Baseline benchmark
conda run -n ign_gpu python scripts/benchmark_gpu_transfers.py \
    --mode baseline --output baseline.json

# Step 2: Apply optimizations
python scripts/refactor_phase2_optimize_gpu.py

# Step 3: Benchmark optimized version
conda run -n ign_gpu python scripts/benchmark_gpu_transfers.py \
    --mode optimized --output optimized.json

# Step 4: Compare results
python scripts/benchmark_gpu_transfers.py \
    --compare baseline.json optimized.json
```

**Target metrics:**

- ✅ GPU transfers < 5 per tile
- ✅ Throughput improvement > 20%
- ✅ GPU utilization > 80%

---

## 🔬 Analysis Tools

### Duplication Analyzer

**Script:** `analyze_duplication.py`

Analyzes codebase for duplicate functions and classes.

```bash
# Run analysis
python scripts/analyze_duplication.py

# Show only functions with 3+ duplicates
python scripts/analyze_duplication.py --min-duplicates 3
```

**Output includes:**

- List of duplicate function names
- File locations
- Estimated duplicate lines
- Prioritized recommendations

---

### GPU Transfer Profiler

**Created by:** Phase 2 refactoring script

**Usage:**

```python
from ign_lidar.optimization.gpu_transfer_profiler import GPUTransferProfiler

profiler = GPUTransferProfiler()
with profiler:
    # Your GPU code
    points_gpu = cp.asarray(points)
    features = compute_features_gpu(points_gpu)
    result = cp.asnumpy(features)

profiler.print_report()
```

**Tracks:**

- Number of CPU→GPU and GPU→CPU transfers
- Transfer sizes
- Bandwidth utilization
- Hotspots (with `track_stacks=True`)

---

## 📈 Benchmarking

### GPU Transfer Benchmark

**Script:** `benchmark_gpu_transfers.py` (created by Phase 2)

```bash
# Baseline
python scripts/benchmark_gpu_transfers.py \
    --mode baseline \
    --points 100000 \
    --output baseline.json

# Optimized
python scripts/benchmark_gpu_transfers.py \
    --mode optimized \
    --points 100000 \
    --output optimized.json

# Compare
python scripts/benchmark_gpu_transfers.py \
    --compare baseline.json optimized.json
```

---

## ⚠️ Safety Features

All refactoring scripts include:

✅ **Automatic backups** (`.backup` extension)  
✅ **Dry-run mode** (preview changes)  
✅ **Confirmation prompts**  
✅ **Rollback capability**  
✅ **Validation tests**

### Rollback Process

If something goes wrong:

```bash
# Restore from backup
find . -name "*.py.backup" -exec bash -c 'mv "$1" "${1%.backup}"' _ {} \;

# Or manually
mv ign_lidar/features/feature_computer.py.backup \
   ign_lidar/features/feature_computer.py
```

---

## 📝 Version Planning

- **v3.6.0** (current target): Phase 1 + Phase 2 with deprecation warnings
- **v3.7.0-3.9.0**: Continued deprecation warnings
- **v4.0.0**: Remove deprecated code, complete Phase 3

---

## 🎯 Success Metrics

| Metric              | Before   | After Phase 1 | After Phase 2 | Target  |
| ------------------- | -------- | ------------- | ------------- | ------- |
| Duplicate lines     | 22,900   | ~19,000       | ~19,000       | <10,000 |
| `compute_normals()` | 7        | 1             | 1             | 1 ✅    |
| GPU transfers/tile  | 90+      | 90+           | <5            | <5 ✅   |
| GPU utilization     | 60-70%   | 60-70%        | 85-95%        | >80% ✅ |
| Throughput          | baseline | baseline      | +20-30%       | +20% ✅ |

---

## 📚 Documentation

- **Audit report:** `docs/audit_reports/CODE_QUALITY_AUDIT_NOV22_2025.md`
- **Summary:** `docs/audit_reports/REFACTORING_SUMMARY_NOV22_2025.md`
- **Migration guide:** `docs/migration_guides/compute_normals_consolidation.md` (created by Phase 1)

---

## 🤝 Contributing

Before running refactoring scripts:

1. **Commit current work:** `git add . && git commit -m "checkpoint"`
2. **Create branch:** `git checkout -b refactor/phase1-duplications`
3. **Run script:** `python scripts/refactor_phase1_remove_duplicates.py`
4. **Test:** `pytest tests/ -v`
5. **Review:** `git diff`
6. **Commit:** `git commit -m "refactor: consolidate compute_normals()"`

---

## 📞 Support

Questions or issues?

- **GitHub Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Documentation:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/

---

**Last Updated:** November 22, 2025  
**Audit Version:** 1.0
