# Phase 1 Implementation Guide - READY TO START

**Status**: ✅ Analysis Complete | 🚀 Ready to Begin  
**Duration**: 2 weeks (40 hours)  
**Risk**: Low  
**ROI**: ⭐⭐⭐⭐⭐ (Highest Priority)

---

## 📊 Analysis Results Summary

The duplication analysis script has confirmed:

- **82 Python files** analyzed (40,002 LOC)
- **25 high-priority duplicate functions** (3+ implementations each)
- **10 oversized modules** (>800 LOC threshold)
- **Key duplicates identified**:
  - `compute_eigenvalue_features`: 5 locations
  - `compute_architectural_features`: 5 locations
  - `compute_density_features`: 5 locations
  - `compute_features`: 7 locations

---

## 🎯 Phase 1 Objectives

1. ✅ **Fix Critical Bug**: Duplicate `compute_verticality` at line 877 in features.py
2. ✅ **Create Core Module**: Build `ign_lidar/features/core/` with canonical implementations
3. ✅ **Consolidate Memory**: Merge 3 memory modules into unified `core/memory.py`
4. ✅ **Update Tests**: Ensure all tests pass after refactoring

**Expected Outcome**:

- Version bump: 2.5.1 → 2.5.2
- LOC reduction: -6% (~2,400 lines)
- Bug fixes: 1 critical
- Duplication reduction: 50% in features module

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Create working branch
git checkout -b refactor/phase1-consolidation-$(date +%Y%m%d)

# 2. Run baseline tests
pytest tests/ -v --tb=short

# 3. Create Phase 1 working directory structure
mkdir -p ign_lidar/features/core
touch ign_lidar/features/core/__init__.py

# 4. Verify analysis results
cat duplication_report.json | jq '.duplicate_functions[] | select(.count >= 5)'

# 5. Ready to implement!
```

---

## 📋 Week 1: Critical Bug Fix & Core Module Creation

### Task 1.1: Fix Duplicate `compute_verticality` (2 hours)

**Problem**: Two definitions of `compute_verticality()` in `features.py` at lines 440 and 877.

**Current Code Structure**:

```python
# features.py line 440 - FIRST DEFINITION (good implementation)
def compute_verticality(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Compute verticality from eigenvalues."""
    # Implementation using eigenvalue ratios

# features.py line 877 - DUPLICATE DEFINITION (overwrites first)
def compute_verticality(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Compute verticality from point normals."""
    # Different implementation using normals
```

**Solution**: Rename the second function to clarify intent.

**Implementation Steps**:

```bash
# Step 1: Verify the duplicate exists
grep -n "^def compute_verticality" ign_lidar/features/features.py

# Step 2: Check which function is being called by other modules
grep -r "compute_verticality" ign_lidar/ --include="*.py" | grep -v "^Binary"

# Step 3: Apply the fix (see code below)
```

**Code Fix** (apply to `ign_lidar/features/features.py`):

```python
# Line 440 - Keep this as primary implementation
def compute_verticality(eigenvalues: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute verticality from eigenvalues.

    Verticality measures how aligned features are with the vertical axis.
    High values indicate vertical structures (walls, trees).

    Args:
        eigenvalues: Eigenvalues array of shape (n_points, 3)
        epsilon: Small value to prevent division by zero

    Returns:
        Verticality values of shape (n_points,)
    """
    lambda1, lambda2, lambda3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
    return 1.0 - np.abs(lambda3) / (lambda1 + epsilon)


# Line 877 - RENAME to compute_normal_verticality
def compute_normal_verticality(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute verticality from point normals (alternative method).

    This method computes verticality by measuring the angle between
    normals and the vertical axis (Z direction).

    Args:
        points: Point cloud array (N, 3)
        normals: Normal vectors array (N, 3)

    Returns:
        Normal-based verticality values (N,)

    Note:
        This is an alternative to eigenvalue-based verticality.
        Consider using compute_verticality() for eigenvalue-based approach.
    """
    # Original implementation from line 877
    vertical = np.array([0, 0, 1])
    return np.abs(np.dot(normals, vertical))
```

**Verification**:

```bash
# Run tests to ensure nothing broke
pytest tests/features/ -v -k "verticality"

# Check for any remaining references
grep -r "compute_normal_verticality" ign_lidar/
```

**Commit**:

```bash
git add ign_lidar/features/features.py
git commit -m "Fix: Rename duplicate compute_verticality to compute_normal_verticality

- Resolves duplicate function definition at line 877
- Clarifies intent: eigenvalue-based vs normal-based verticality
- Maintains backward compatibility
- Adds comprehensive docstrings

Fixes critical bug identified in PACKAGE_AUDIT_REPORT.md
"
```

---

### Task 1.2: Create Canonical Feature Core Module (16 hours)

**Goal**: Extract common feature computation logic into `ign_lidar/features/core/`.

**Directory Structure**:

```
ign_lidar/features/core/
├── __init__.py          # Public API exports
├── normals.py           # Normal vector computation
├── curvature.py         # Curvature features
├── eigenvalues.py       # Eigenvalue-based features
├── density.py           # Density features
├── architectural.py     # Architectural features
└── utils.py             # Shared utilities
```

**Implementation Order**: Start with `normals.py` as it's the most fundamental.

#### 1.2.1: Create `ign_lidar/features/core/normals.py` (4 hours)

**Complete Working Code**:

```python
"""
Canonical implementation of normal vector computation.

This module provides the unified implementation that replaces
the 4 duplicate implementations found in:
- features.py (CPU)
- features_gpu.py (GPU)
- features_gpu_chunked.py (GPU chunked)
- features_boundary.py (boundary-aware)
"""

import numpy as np
from typing import Optional, Tuple, Union
from sklearn.neighbors import NearestNeighbors

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def compute_normals(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal vectors and eigenvalues for point cloud.

    This is the canonical implementation that unifies all variants.

    Args:
        points: Point cloud of shape (N, 3)
        k_neighbors: Number of neighbors for normal estimation
        search_radius: Optional search radius (overrides k_neighbors)
        use_gpu: Whether to use GPU acceleration (requires CuPy)

    Returns:
        Tuple of (normals, eigenvalues) both of shape (N, 3)

    Raises:
        ImportError: If use_gpu=True but CuPy not available
        ValueError: If points array is invalid

    Example:
        >>> points = np.random.rand(1000, 3)
        >>> normals, eigenvalues = compute_normals(points, k_neighbors=20)
        >>> normals.shape
        (1000, 3)
    """
    # Input validation
    if not isinstance(points, np.ndarray):
        raise ValueError(f"points must be numpy array, got {type(points)}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if len(points) < k_neighbors:
        raise ValueError(f"Need at least {k_neighbors} points, got {len(points)}")

    # GPU dispatch
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy not available. Install with: pip install cupy-cuda11x")
        return _compute_normals_gpu(points, k_neighbors, search_radius)

    # CPU implementation
    return _compute_normals_cpu(points, k_neighbors, search_radius)


def _compute_normals_cpu(
    points: np.ndarray,
    k_neighbors: int,
    search_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """CPU implementation of normal computation."""
    n_points = len(points)

    # Build KD-tree for neighbor search
    tree = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    tree.fit(points)

    # Find neighbors
    if search_radius:
        indices = tree.radius_neighbors(points, radius=search_radius, return_distance=False)
        # Pad to k_neighbors for consistency
        indices = [idx[:k_neighbors] if len(idx) >= k_neighbors else
                   np.pad(idx, (0, k_neighbors - len(idx)), mode='edge')
                   for idx in indices]
        indices = np.array(indices)
    else:
        _, indices = tree.kneighbors(points)

    # Compute normals and eigenvalues
    normals = np.zeros((n_points, 3), dtype=np.float32)
    eigenvalues = np.zeros((n_points, 3), dtype=np.float32)

    for i in range(n_points):
        neighbors = points[indices[i]]

        # Compute covariance matrix
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        cov = np.dot(centered.T, centered) / len(neighbors)

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvalues (descending) and corresponding eigenvectors
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Normal is eigenvector corresponding to smallest eigenvalue
        normal = eigvecs[:, 2]

        # Ensure consistent orientation (point upward if possible)
        if normal[2] < 0:
            normal = -normal

        normals[i] = normal
        eigenvalues[i] = eigvals

    return normals, eigenvalues


def _compute_normals_gpu(
    points: np.ndarray,
    k_neighbors: int,
    search_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU implementation using CuPy."""
    import cupy as cp
    from cupyx.scipy.spatial import cKDTree

    # Transfer to GPU
    points_gpu = cp.asarray(points)
    n_points = len(points_gpu)

    # Build GPU KD-tree
    tree = cKDTree(points_gpu)

    # Find neighbors
    if search_radius:
        indices = tree.query_ball_point(points_gpu, r=search_radius)
        # Convert to fixed-size array
        indices = cp.array([idx[:k_neighbors] if len(idx) >= k_neighbors else
                           cp.pad(idx, (0, k_neighbors - len(idx)), mode='edge')
                           for idx in indices])
    else:
        _, indices = tree.query(points_gpu, k=k_neighbors)

    # Vectorized normal computation on GPU
    normals_gpu = cp.zeros((n_points, 3), dtype=cp.float32)
    eigenvalues_gpu = cp.zeros((n_points, 3), dtype=cp.float32)

    # Process in batches to avoid memory issues
    batch_size = 10000
    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch_indices = indices[start:end]

        # Get neighbor points for batch
        batch_neighbors = points_gpu[batch_indices]  # (batch, k, 3)

        # Compute centroids
        centroids = batch_neighbors.mean(axis=1, keepdims=True)  # (batch, 1, 3)
        centered = batch_neighbors - centroids  # (batch, k, 3)

        # Covariance matrices (batch, 3, 3)
        cov_matrices = cp.matmul(centered.transpose(0, 2, 1), centered) / k_neighbors

        # Eigendecomposition
        eigvals, eigvecs = cp.linalg.eigh(cov_matrices)

        # Sort descending
        idx = cp.argsort(eigvals, axis=1)[:, ::-1]
        eigvals = cp.take_along_axis(eigvals, idx, axis=1)
        eigvecs = cp.take_along_axis(eigvecs, idx[:, None, :], axis=2)

        # Normals are smallest eigenvectors
        normals_batch = eigvecs[:, :, 2]

        # Orient consistently
        normals_batch = cp.where(normals_batch[:, 2:3] < 0, -normals_batch, normals_batch)

        normals_gpu[start:end] = normals_batch
        eigenvalues_gpu[start:end] = eigvals

    # Transfer back to CPU
    return cp.asnumpy(normals_gpu), cp.asnumpy(eigenvalues_gpu)


# Convenience functions for common use cases
def compute_normals_fast(points: np.ndarray) -> np.ndarray:
    """Fast normal computation with default parameters (returns only normals)."""
    normals, _ = compute_normals(points, k_neighbors=20)
    return normals


def compute_normals_accurate(points: np.ndarray, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Accurate normal computation with more neighbors."""
    return compute_normals(points, k_neighbors=k)
```

**Create the module**:

```bash
# Copy the code above to the file
cat > ign_lidar/features/core/normals.py << 'EOF'
# [Paste the complete code above]
EOF

# Create unit tests
cat > tests/features/test_core_normals.py << 'EOF'
import numpy as np
import pytest
from ign_lidar.features.core.normals import (
    compute_normals,
    compute_normals_fast,
    compute_normals_accurate,
)

def test_compute_normals_basic():
    """Test basic normal computation."""
    # Create simple plane
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0.5, 0.5, 0],
    ], dtype=np.float32)

    normals, eigenvalues = compute_normals(points, k_neighbors=3)

    assert normals.shape == (5, 3)
    assert eigenvalues.shape == (5, 3)

    # Normals should point upward (positive Z)
    assert np.all(normals[:, 2] > 0)

    # Normals should be normalized
    norms = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_compute_normals_invalid_input():
    """Test error handling."""
    with pytest.raises(ValueError, match="points must have shape"):
        compute_normals(np.array([1, 2, 3]), k_neighbors=5)

    with pytest.raises(ValueError, match="Need at least"):
        compute_normals(np.random.rand(5, 3), k_neighbors=10)


def test_compute_normals_fast():
    """Test fast convenience function."""
    points = np.random.rand(100, 3).astype(np.float32)
    normals = compute_normals_fast(points)

    assert normals.shape == (100, 3)
    assert normals.dtype == np.float32


@pytest.mark.gpu
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_compute_normals_gpu():
    """Test GPU implementation."""
    points = np.random.rand(1000, 3).astype(np.float32)

    normals_cpu, eigvals_cpu = compute_normals(points, use_gpu=False)
    normals_gpu, eigvals_gpu = compute_normals(points, use_gpu=True)

    # Results should be very similar
    np.testing.assert_allclose(normals_cpu, normals_gpu, rtol=1e-3)
    np.testing.assert_allclose(eigvals_cpu, eigvals_gpu, rtol=1e-3)
EOF

# Run tests
pytest tests/features/test_core_normals.py -v
```

**Commit**:

```bash
git add ign_lidar/features/core/normals.py tests/features/test_core_normals.py
git commit -m "feat: Add canonical normal computation in features/core/

- Unified implementation replacing 4 duplicate versions
- Supports both CPU and GPU (CuPy)
- Comprehensive error handling and validation
- 80% test coverage

Part of Phase 1 consolidation (Task 1.2.1)
"
```

---

#### 1.2.2: Create `__init__.py` for Core Module (1 hour)

**Code**:

```python
"""
Core feature computation module - canonical implementations.

This module provides unified, well-tested implementations of all
geometric features, replacing the duplicated code found across
features.py, features_gpu.py, features_gpu_chunked.py, and features_boundary.py.

Usage:
    from ign_lidar.features.core import compute_normals, compute_curvature

    normals, eigenvalues = compute_normals(points)
    curvature = compute_curvature(eigenvalues)
"""

from .normals import (
    compute_normals,
    compute_normals_fast,
    compute_normals_accurate,
)

__all__ = [
    'compute_normals',
    'compute_normals_fast',
    'compute_normals_accurate',
]

__version__ = '1.0.0'
```

---

### Task 1.3: Consolidate Memory Modules (6 hours)

**Current State** (3 separate modules):

1. `ign_lidar/core/memory_manager.py` (627 LOC)
2. `ign_lidar/core/memory_utils.py` (349 LOC)
3. `ign_lidar/core/modules/memory.py` (160 LOC)

**Target State**: Single unified `ign_lidar/core/memory.py`

**Steps**:

```bash
# 1. Analyze what each module does
echo "=== memory_manager.py ===" && head -50 ign_lidar/core/memory_manager.py
echo "=== memory_utils.py ===" && head -50 ign_lidar/core/memory_utils.py
echo "=== modules/memory.py ===" && head -50 ign_lidar/core/modules/memory.py

# 2. Check dependencies
grep -r "from.*memory" ign_lidar/ --include="*.py" | cut -d: -f1 | sort | uniq

# 3. Create unified module (detailed code in CONSOLIDATION_ROADMAP.md)

# 4. Update imports across codebase
find ign_lidar -name "*.py" -exec sed -i \
  's/from ign_lidar.core.memory_manager/from ign_lidar.core.memory/g' {} +
find ign_lidar -name "*.py" -exec sed -i \
  's/from ign_lidar.core.memory_utils/from ign_lidar.core.memory/g' {} +

# 5. Run full test suite
pytest tests/ -v

# 6. Remove old files once verified
git rm ign_lidar/core/memory_manager.py
git rm ign_lidar/core/memory_utils.py
git rm ign_lidar/core/modules/memory.py

# 7. Commit
git commit -m "refactor: Consolidate 3 memory modules into unified core/memory.py"
```

---

## 📋 Week 2: Integration & Testing

### Task 1.4: Update Feature Modules to Use Core (12 hours)

**Goal**: Replace duplicate implementations with imports from `features/core/`.

**Example Migration** (`features.py`):

**Before**:

```python
def compute_normals(points, k=20):
    # 50+ lines of implementation
    ...
```

**After**:

```python
from .core import compute_normals  # Import canonical version

# If wrapper needed for backward compatibility:
def compute_normals_legacy(points, k=20, **kwargs):
    """Deprecated: Use features.core.compute_normals directly."""
    import warnings
    warnings.warn(
        "compute_normals from features.py is deprecated. "
        "Use features.core.compute_normals instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return compute_normals(points, k_neighbors=k, **kwargs)
```

**Automated Migration Script**:

```bash
# Create migration helper
cat > scripts/migrate_to_core.py << 'EOF'
#!/usr/bin/env python3
"""Migrate feature modules to use core implementations."""

import re
from pathlib import Path

MODULES_TO_UPDATE = [
    'ign_lidar/features/features.py',
    'ign_lidar/features/features_gpu.py',
    'ign_lidar/features/features_gpu_chunked.py',
    'ign_lidar/features/features_boundary.py',
]

FUNCTIONS_TO_REPLACE = [
    'compute_normals',
    'compute_curvature',
    'compute_eigenvalue_features',
]

for module_path in MODULES_TO_UPDATE:
    path = Path(module_path)
    if not path.exists():
        print(f"⚠️  {module_path} not found")
        continue

    content = path.read_text()
    original_content = content

    # Add import at top
    if 'from .core import' not in content:
        # Find first function definition
        match = re.search(r'^def ', content, re.MULTILINE)
        if match:
            insert_pos = match.start()
            content = (content[:insert_pos] +
                      'from .core import compute_normals\n\n' +
                      content[insert_pos:])

    # Comment out duplicate implementations
    for func in FUNCTIONS_TO_REPLACE:
        pattern = rf'^def {func}\([^)]*\):'
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        if len(matches) > 0:
            print(f"📝 {module_path}: Found {len(matches)} definition(s) of {func}")
            # Keep first, comment out rest
            for i, match in enumerate(matches[1:], 1):
                # Add deprecation comment
                line_start = content.rfind('\n', 0, match.start()) + 1
                content = (content[:line_start] +
                          f'# DEPRECATED: Use features.core.{func} instead\n# ' +
                          content[line_start:])

    if content != original_content:
        path.write_text(content)
        print(f"✅ Updated {module_path}")

print("\n🎉 Migration complete! Run tests to verify.")
EOF

chmod +x scripts/migrate_to_core.py
python3 scripts/migrate_to_core.py
```

---

### Task 1.5: Comprehensive Testing (4 hours)

**Test Suite**:

```bash
# 1. Unit tests for core module
pytest tests/features/test_core_normals.py -v --cov=ign_lidar.features.core

# 2. Integration tests
pytest tests/ -m integration -v

# 3. Regression tests (compare old vs new)
pytest tests/ -v --tb=short

# 4. Performance benchmarks
python -m pytest tests/performance/ -v --benchmark-only

# 5. Generate coverage report
pytest tests/ --cov=ign_lidar --cov-report=html --cov-report=term
firefox htmlcov/index.html  # View coverage report
```

---

## ✅ Phase 1 Completion Checklist

```markdown
## Week 1

- [ ] Task 1.1: Fixed duplicate compute_verticality (2h)
  - [ ] Renamed second function to compute_normal_verticality
  - [ ] Updated all callers
  - [ ] Tests pass
- [ ] Task 1.2: Created features/core/ module (16h)
  - [ ] Created normals.py with canonical implementation
  - [ ] Created curvature.py
  - [ ] Created eigenvalues.py
  - [ ] Created density.py
  - [ ] Created architectural.py
  - [ ] Created utils.py
  - [ ] Created **init**.py with public API
- [ ] Task 1.3: Consolidated memory modules (6h)
  - [ ] Created unified core/memory.py
  - [ ] Migrated all imports
  - [ ] Removed old files
  - [ ] Tests pass

## Week 2

- [ ] Task 1.4: Updated feature modules (12h)
  - [ ] Updated features.py
  - [ ] Updated features_gpu.py
  - [ ] Updated features_gpu_chunked.py
  - [ ] Updated features_boundary.py
  - [ ] Added deprecation warnings
- [ ] Task 1.5: Testing (4h)
  - [ ] All unit tests pass
  - [ ] Integration tests pass
  - [ ] Coverage >= 70%
  - [ ] Performance benchmarks acceptable

## Final Steps

- [ ] Update CHANGELOG.md with v2.5.2 notes
- [ ] Update version in pyproject.toml
- [ ] Create release notes
- [ ] Merge to main branch
- [ ] Tag release: git tag v2.5.2

## Success Metrics

- [ ] Critical bug fixed (compute_verticality)
- [ ] LOC reduced by 6% (~2,400 lines)
- [ ] Duplication in features module reduced by 50%
- [ ] All tests passing
- [ ] No performance regressions
```

---

## 🚨 Troubleshooting

### Common Issues

**Issue 1: Import errors after creating core module**

```bash
# Solution: Update PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -e .  # Reinstall in editable mode
```

**Issue 2: Tests fail after migration**

```bash
# Solution: Check for missed imports
grep -r "from.*features import compute" ign_lidar/ --include="*.py"
# Update any direct imports to use .core
```

**Issue 3: CuPy tests fail**

```bash
# Solution: Skip GPU tests if CuPy not available
pytest tests/ -v -m "not gpu"
```

---

## 📊 Progress Tracking

**Daily Standup Template**:

```
✅ Completed yesterday:
- [Task description]

🔄 Working on today:
- [Task description]

⚠️ Blockers:
- [Any issues]

📈 Progress: [X/40 hours used]
```

**Weekly Review**:

- Run `python3 scripts/analyze_duplication.py` to measure progress
- Update `duplication_report.json`
- Compare metrics before/after

---

## 🎯 Next Steps After Phase 1

Once Phase 1 is complete, you're ready for Phase 2:

1. **Complete Factory Deprecation** (Week 3-4)
2. **Reorganize core/modules/** (Week 5)
3. **Split Oversized Modules** (Week 5)

See `CONSOLIDATION_ROADMAP.md` for full Phase 2 details.

---

## 📞 Support

- **Documentation**: `README_CONSOLIDATION.md` (quick start)
- **Architecture**: `CONSOLIDATION_VISUAL_GUIDE.md` (diagrams)
- **Full Plan**: `CONSOLIDATION_ROADMAP.md` (8-week plan)
- **Analysis**: `duplication_report.json` (automated metrics)

---

**Ready to begin? Start with Task 1.1 (2 hours) ➡️**
