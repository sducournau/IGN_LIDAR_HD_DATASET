#!/usr/bin/env python3
"""
Generate GitHub Issues from Audit Report

Creates GitHub issues for each refactoring phase with proper labels,
milestones, and task lists.

Usage:
    # Preview issues (dry-run)
    python scripts/create_github_issues.py --dry-run
    
    # Create issues (requires gh CLI)
    python scripts/create_github_issues.py --create
    
    # Create specific phase only
    python scripts/create_github_issues.py --phase 1 --create

Requirements:
    - GitHub CLI (gh) installed and authenticated
    - Or use --dry-run to just see what would be created

Author: GitHub Copilot
Date: November 22, 2025
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


# Issue templates for each phase
ISSUES = [
    {
        "phase": 1,
        "title": "Refactor: Consolidate compute_normals() implementations (Phase 1)",
        "labels": ["refactoring", "phase-1", "high-priority", "code-quality"],
        "milestone": "v3.6.0",
        "body": """## 🎯 Objective

Consolidate 7 duplicate implementations of `compute_normals()` into a single canonical version to improve code maintainability and reduce duplication.

## 📊 Current State

**Problem:** Multiple duplicate implementations across the codebase
- 7 implementations of `compute_normals()`
- 2 implementations of `validate_normals()`
- ~400 lines of duplicated code
- Maintenance burden (bug fixes need to be applied 7 times)

**Locations:**
1. `ign_lidar/features/feature_computer.py:160`
2. `ign_lidar/features/gpu_processor.py:376`
3. `ign_lidar/features/compute/normals.py:37` ✅ (canonical)
4. `ign_lidar/features/gpu_processor.py:726` (_compute_normals_cpu)
5. `ign_lidar/features/compute/normals.py:107` (_compute_normals_cpu)
6. `ign_lidar/features/utils.py:206` (validate_normals)
7. `ign_lidar/features/compute/utils.py:63` (validate_normals) ✅ (canonical)

## ✅ Success Criteria

- [ ] Single canonical implementation in `ign_lidar/features/compute/normals.py`
- [ ] All tests pass (100%)
- [ ] Backward compatibility maintained (deprecation warnings)
- [ ] ~400 lines removed
- [ ] Migration guide created

## 🛠️ Implementation

### Automated Script

```bash
# Run refactoring script
python scripts/refactor_phase1_remove_duplicates.py

# Validate changes
python scripts/validate_refactoring.py --phase 1
```

### Manual Steps

1. **Create branch**
   ```bash
   git checkout -b refactor/phase1-compute-normals
   ```

2. **Execute refactoring**
   ```bash
   python scripts/refactor_phase1_remove_duplicates.py
   ```

3. **Review changes**
   ```bash
   git diff
   ```

4. **Run tests**
   ```bash
   pytest tests/test_features*.py -v
   pytest tests/test_compute*.py -v
   pytest tests/ -v
   ```

5. **Validate**
   ```bash
   python scripts/validate_refactoring.py --phase 1
   ```

6. **Commit**
   ```bash
   git add .
   git commit -m "refactor: consolidate compute_normals() implementations"
   ```

## 📋 Task Checklist

### Preparation
- [ ] Read audit report: `docs/audit_reports/CODE_QUALITY_AUDIT_NOV22_2025.md`
- [ ] Read quick start: `docs/QUICKSTART_REFACTORING.md`
- [ ] Verify tests pass currently
- [ ] Create branch

### Execution
- [ ] Run refactoring script
- [ ] Verify backups created (*.backup files)
- [ ] Review git diff

### Validation
- [ ] All tests pass
- [ ] Imports work correctly
- [ ] Deprecation warnings emit
- [ ] No performance regression

### Documentation
- [ ] Migration guide reviewed
- [ ] CHANGELOG.md updated
- [ ] Docstrings updated if needed

### Review & Merge
- [ ] Create PR
- [ ] Code review approved
- [ ] CI/CD passes
- [ ] Merge to main

## 📈 Expected Impact

- **Lines removed:** ~400
- **Codebase reduction:** -3%
- **Maintenance:** Easier (single source of truth)
- **Risk:** Low (backward compatible)

## 📚 References

- Audit Report: `docs/audit_reports/CODE_QUALITY_AUDIT_NOV22_2025.md`
- Quick Start: `docs/QUICKSTART_REFACTORING.md`
- Script: `scripts/refactor_phase1_remove_duplicates.py`
- Validation: `scripts/validate_refactoring.py`

## ⚠️ Notes

- 100% backward compatible
- Deprecation warnings added (removal in v4.0)
- All backups created automatically
- Rollback available if issues

**Estimated time:** 1-2 days
"""
    },
    {
        "phase": 2,
        "title": "Performance: Optimize GPU transfers and add CUDA streams (Phase 2)",
        "labels": ["performance", "phase-2", "gpu", "optimization"],
        "milestone": "v3.6.0",
        "body": """## 🎯 Objective

Optimize GPU memory transfers and add CUDA stream support to improve GPU utilization from 60-70% to 85-95% and reduce transfer bottlenecks.

## 📊 Current State

**Problem:** Excessive CPU↔GPU transfers causing performance bottleneck
- **90+ transfers** per tile (target: <5)
- **60-70% GPU utilization** (target: 85-95%)
- No CUDA streams (async execution not used)
- Excessive synchronization blocking GPU pipeline
- **~30-40% performance loss** due to PCIe bottleneck

**Hotspots identified:**
- `optimization/knn_engine.py`: Forces `.get()` sync
- `preprocessing/rgb_augmentation.py`: Redundant transfers
- `optimization/gpu_accelerated_ops.py`: Multiple transfers

## ✅ Success Criteria

- [ ] GPU transfers < 5 per tile
- [ ] GPU utilization 85-95%
- [ ] Throughput improvement > 20%
- [ ] CUDA streams integrated
- [ ] All tests pass

## 🛠️ Implementation

### Automated Script

```bash
# Step 1: Baseline benchmark
conda run -n ign_gpu python scripts/benchmark_gpu_transfers.py \\
    --mode baseline --output baseline.json

# Step 2: Apply optimizations
python scripts/refactor_phase2_optimize_gpu.py

# Step 3: Optimized benchmark
conda run -n ign_gpu python scripts/benchmark_gpu_transfers.py \\
    --mode optimized --output optimized.json

# Step 4: Compare
python scripts/benchmark_gpu_transfers.py \\
    --compare baseline.json optimized.json
```

### Key Changes

1. **Add `GPUTransferProfiler`**
   - Monitor and track all GPU transfers
   - Identify hotspots
   - Measure bandwidth utilization

2. **Add `return_gpu` parameter to `KNNEngine.search()`**
   - Enable lazy GPU array transfers
   - Keep results on GPU when possible
   - Reduce unnecessary synchronization

3. **Integrate `CUDAStreamManager` into `FeatureOrchestrator`**
   - Enable async GPU operations
   - Overlap computation and transfers
   - Improve GPU utilization

## 📋 Task Checklist

### Preparation
- [ ] Verify GPU available: `conda run -n ign_gpu python -c "import cupy"`
- [ ] Activate GPU environment: `conda activate ign_gpu`
- [ ] Create branch: `git checkout -b refactor/phase2-gpu-optimization`

### Baseline Benchmark
- [ ] Run baseline benchmark
- [ ] Document transfer count (~90+)
- [ ] Document GPU utilization (~60-70%)

### Execution
- [ ] Run refactoring script
- [ ] Verify `GPUTransferProfiler` created
- [ ] Verify `KNNEngine` modified
- [ ] Verify `FeatureOrchestrator` modified

### Validation
- [ ] GPU tests pass: `pytest tests/test_gpu*.py -v`
- [ ] Validation script passes
- [ ] Imports work correctly

### Optimized Benchmark
- [ ] Run optimized benchmark
- [ ] Compare with baseline
- [ ] Verify transfers < 5
- [ ] Verify throughput +20%+

### Documentation
- [ ] Document benchmarks in CHANGELOG
- [ ] Update GPU usage guidelines
- [ ] Create profiler usage examples

### Review & Merge
- [ ] Create PR with benchmark results
- [ ] Code review approved
- [ ] CI/CD passes (GPU tests)
- [ ] Merge to main

## 📈 Expected Impact

- **Transfer reduction:** 90+ → <5 (-95%)
- **GPU utilization:** 60-70% → 85-95% (+30%)
- **Throughput:** +20-30%
- **Latency:** Reduced significantly

## 📚 References

- Audit Report: Section 6 "Goulots d'étranglement GPU"
- Script: `scripts/refactor_phase2_optimize_gpu.py`
- Profiler: `ign_lidar/optimization/gpu_transfer_profiler.py` (created)
- Benchmark: `scripts/benchmark_gpu_transfers.py` (created)

## ⚠️ Prerequisites

- **Phase 1 must be completed first**
- GPU environment (`ign_gpu`) must be available
- CuPy must be installed

## 🔬 Validation Commands

```bash
# Check GPU available
conda run -n ign_gpu python -c "import cupy as cp; print(cp.cuda.Device())"

# Run validation
conda run -n ign_gpu python scripts/validate_refactoring.py --phase 2 --gpu

# Profile transfers
conda run -n ign_gpu python -c "
from ign_lidar.optimization.gpu_transfer_profiler import GPUTransferProfiler
profiler = GPUTransferProfiler()
print('Profiler available')
"
```

**Estimated time:** 2-3 days
"""
    },
    {
        "phase": 3,
        "title": "Architecture: Clean up Processor/Engine classes (Phase 3)",
        "labels": ["architecture", "phase-3", "code-quality", "long-term"],
        "milestone": "v4.0.0",
        "body": """## 🎯 Objective

Clean up architecture by reducing overlapping Processor/Computer/Engine classes from 34 to <25 and migrating all KNN operations to unified `KNNEngine`.

## 📊 Current State

**Problem:** Architecture confusion with overlapping responsibilities
- **34 classes** with similar names (Processor/Computer/Engine/Manager)
- `GPUProcessor` vs `FeatureOrchestrator` duplication
- `ProcessorCore` unclear utility
- Multiple KNN implementations despite unified `KNNEngine`

**Classes to review:**
- `ProcessorCore` - possibly redundant
- `FeatureEngine` vs `FeatureOrchestrator` - overlapping
- `GPUProcessor` - duplicates `FeatureOrchestrator`
- 6 KNN implementations despite `KNNEngine` exists

## ✅ Success Criteria

- [ ] Classes reduced from 34 to <25
- [ ] All KNN calls use unified `KNNEngine`
- [ ] No duplicate Processor classes
- [ ] Architecture documented (ADRs)
- [ ] Tests pass

## 🛠️ Implementation

### Analysis Phase

1. **Audit class usage**
   ```bash
   # Find all Processor/Engine classes
   grep -r "class.*Processor\\|class.*Engine\\|class.*Computer" ign_lidar/
   ```

2. **Check dependencies**
   ```python
   # For each candidate for removal
   git grep "ProcessorCore"
   git grep "FeatureEngine"
   ```

### KNN Migration

**Files to migrate:**
- `io/formatters/hybrid_formatter.py`
- `io/formatters/multi_arch_formatter.py`
- `optimization/gpu_accelerated_ops.py` (2 knn functions)

**Migration pattern:**
```python
# Old
from some.module import build_kdtree, knn_search

# New
from ign_lidar.optimization import KNNEngine
engine = KNNEngine(backend='auto', use_gpu=True)
distances, indices = engine.search(points, query, k=30)
```

### Class Consolidation

1. **Deprecate `GPUProcessor`**
   - Already duplicates `FeatureOrchestrator`
   - Add deprecation warning in v3.6
   - Remove in v4.0

2. **Remove or clarify `ProcessorCore`**
   - Audit all usages
   - Either remove or document clear purpose

3. **Merge `FeatureEngine` into `FeatureOrchestrator`**
   - If roles overlap, consolidate
   - Update all references

## 📋 Task Checklist

### Analysis
- [ ] List all Processor/Computer/Engine classes
- [ ] Audit usage of each class
- [ ] Identify truly redundant classes
- [ ] Check dependencies

### KNN Migration
- [ ] Migrate `hybrid_formatter.py`
- [ ] Migrate `multi_arch_formatter.py`
- [ ] Migrate `gpu_accelerated_ops.py`
- [ ] Add deprecation warnings to old KNN functions
- [ ] Update tests

### Class Consolidation
- [ ] Deprecate `GPUProcessor` (v3.6)
- [ ] Audit `ProcessorCore` usage
- [ ] Remove or document `ProcessorCore`
- [ ] Clarify `FeatureEngine` vs `FeatureOrchestrator`
- [ ] Update imports across codebase

### Documentation
- [ ] Create Architecture Decision Records (ADRs)
- [ ] Update architecture diagrams
- [ ] Document Strategy pattern for features
- [ ] Update developer guide

### Testing
- [ ] All tests pass
- [ ] No broken imports
- [ ] Performance unchanged

### Review & Merge
- [ ] Create PR
- [ ] Architecture review
- [ ] Code review approved
- [ ] Merge to main

## 📈 Expected Impact

- **Classes:** 34 → <25 (-26%)
- **KNN implementations:** 6 → 1 unified
- **Architecture:** Clear and documented
- **Maintenance:** Easier to understand

## 📚 References

- Audit Report: Section 2 "Prolifération de Classes"
- Audit Report: Section 3 "Duplication KNN"
- KNNEngine: `ign_lidar/optimization/knn_engine.py`

## ⚠️ Prerequisites

- **Phases 1 and 2 must be completed**
- More architectural changes (higher risk)
- Requires careful dependency analysis

## 🗺️ Migration Guide

### For Users of Deprecated Classes

**GPUProcessor → FeatureOrchestrator:**
```python
# Old (deprecated in v3.6, removed in v4.0)
from ign_lidar.features.gpu_processor import GPUProcessor
processor = GPUProcessor(config)
features = processor.compute_features(points)

# New
from ign_lidar.features import FeatureOrchestrator
orchestrator = FeatureOrchestrator(config)
features = orchestrator.compute_features(points)
```

**Legacy KNN → KNNEngine:**
```python
# Old (deprecated)
from ign_lidar.optimization.gpu_accelerated_ops import knn
distances, indices = knn(points, query, k=30)

# New
from ign_lidar.optimization import KNNEngine
engine = KNNEngine(backend='auto')
distances, indices = engine.search(points, query, k=30)
```

**Estimated time:** 3-5 days
"""
    }
]


def check_gh_cli():
    """Check if GitHub CLI is installed."""
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_issue_dry_run(issue: Dict):
    """Print issue in dry-run mode."""
    print(f"\n{'=' * 80}")
    print(f"PHASE {issue['phase']}: {issue['title']}")
    print(f"{'=' * 80}")
    print(f"Labels: {', '.join(issue['labels'])}")
    print(f"Milestone: {issue['milestone']}")
    print(f"\nBody preview (first 500 chars):")
    print(issue['body'][:500])
    print("...")
    print(f"\n[Would create issue with {len(issue['body'])} characters]")


def create_issue_github(issue: Dict) -> bool:
    """Create issue using GitHub CLI."""
    print(f"\nCreating Phase {issue['phase']} issue...")
    
    # Build gh command
    cmd = [
        "gh", "issue", "create",
        "--title", issue['title'],
        "--body", issue['body'],
        "--label", ",".join(issue['labels']),
    ]
    
    # Add milestone if exists
    # Note: Milestone must exist in repo first
    # cmd.extend(["--milestone", issue['milestone']])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            print(f"✅ Created: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate GitHub issues from audit report'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview issues without creating them'
    )
    parser.add_argument(
        '--create',
        action='store_true',
        help='Create issues on GitHub (requires gh CLI)'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3],
        help='Create only specific phase issue'
    )
    
    args = parser.parse_args()
    
    # Default to dry-run if neither specified
    if not args.dry_run and not args.create:
        args.dry_run = True
    
    print("\n" + "=" * 80)
    print("📝 GITHUB ISSUE GENERATOR - Code Refactoring")
    print("=" * 80)
    
    # Check gh CLI if creating
    if args.create:
        if not check_gh_cli():
            print("\n❌ GitHub CLI (gh) not found!")
            print("Install: https://cli.github.com/")
            print("\nOr use --dry-run to preview issues")
            sys.exit(1)
        
        print("\n✅ GitHub CLI found")
    
    # Filter issues by phase if specified
    issues_to_create = ISSUES
    if args.phase:
        issues_to_create = [i for i in ISSUES if i['phase'] == args.phase]
    
    print(f"\nWill create {len(issues_to_create)} issue(s)")
    
    # Create or preview issues
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Previewing issues...\n")
        for issue in issues_to_create:
            create_issue_dry_run(issue)
        
        print("\n" + "=" * 80)
        print("To create these issues:")
        print("  python scripts/create_github_issues.py --create")
        print("=" * 80)
    
    elif args.create:
        print("\n🚀 Creating issues on GitHub...\n")
        
        response = input("Proceed with issue creation? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)
        
        success = 0
        for issue in issues_to_create:
            if create_issue_github(issue):
                success += 1
        
        print(f"\n{'=' * 80}")
        print(f"✅ Created {success}/{len(issues_to_create)} issues")
        print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
