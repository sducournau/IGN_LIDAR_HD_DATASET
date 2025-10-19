# GPU Refactoring Audit: Core Features Integration

**Date:** October 19, 2025  
**Auditor:** AI Code Analysis  
**Scope:** GPU and GPU-chunked feature computation refactoring to use core features module

---

## Executive Summary

This audit assesses the current state of GPU-accelerated feature computation implementations (`features_gpu.py` and `features_gpu_chunked.py`) and their integration with the canonical `ign_lidar.features.core` module. The goal is to identify code duplication, evaluate refactoring progress, and provide recommendations for completing the consolidation effort.

### Key Findings

✅ **Strengths:**

- Core module infrastructure is well-designed and comprehensive
- `features_gpu.py` shows good integration with core utilities
- Import structure is correctly set up for both files
- Performance optimizations (GPU batching, CUDA streams) are well-documented

⚠️ **Critical Issues:**

- **Major code duplication** in `features_gpu_chunked.py`
- Custom GPU implementations bypass core feature functions
- Inconsistent feature computation logic across files
- Missing opportunities for code reuse

---

## 1. Current Architecture

### 1.1 Module Structure

```
ign_lidar/features/
├── core/                           # ✅ Canonical implementations
│   ├── __init__.py                 # Unified API
│   ├── eigenvalues.py              # Eigenvalue-based features
│   ├── density.py                  # Density features
│   ├── architectural.py            # Architectural features
│   ├── normals.py                  # Normal computation
│   ├── curvature.py                # Curvature features
│   ├── height.py                   # Height features
│   ├── utils.py                    # Shared utilities
│   └── geometric.py                # Geometric features
│
├── features_gpu.py                 # ⚠️ Partial integration
├── features_gpu_chunked.py         # ❌ Minimal integration
├── strategy_gpu.py                 # Wrapper for features_gpu
└── strategy_gpu_chunked.py         # Wrapper for features_gpu_chunked
```

### 1.2 Import Analysis

#### `features_gpu.py` - ✅ Good Integration

```python
# Core feature implementations (IMPORTED)
from ..features.core import (
    compute_normals as core_compute_normals,
    compute_curvature as core_compute_curvature,
    compute_eigenvalue_features as core_compute_eigenvalue_features,
    compute_density_features as core_compute_density_features,
    compute_verticality as core_compute_verticality,
    extract_geometric_features as core_extract_geometric_features,
)

# Core utilities (USED)
from .core.utils import (
    batched_inverse_3x3,
    inverse_power_iteration,
    compute_eigenvalue_features_from_covariances,
    compute_covariances_from_neighbors,
)
from .core.height import compute_height_above_ground
from .core.curvature import compute_curvature_from_normals
```

#### `features_gpu_chunked.py` - ❌ Limited Integration

```python
# Core feature implementations (IMPORTED BUT NOT USED!)
from ..features.core import (
    compute_eigenvalue_features as core_compute_eigenvalue_features,  # ❌ NOT USED
    compute_density_features as core_compute_density_features,        # ❌ NOT USED
)

# Core utilities (PARTIALLY USED)
from .core.utils import (
    batched_inverse_3x3,
    inverse_power_iteration,
    compute_eigenvalue_features_from_covariances,  # ❌ NOT USED
    compute_covariances_from_neighbors,            # ❌ NOT USED
)
from .core.height import compute_height_above_ground  # ✅ USED
from .core.curvature import compute_curvature_from_normals  # ⚠️ RARELY USED
```

---

## 2. Detailed Audit by Feature Category

### 2.1 Eigenvalue Features

#### Current State: ❌ DUPLICATED LOGIC

**Issue:** `features_gpu_chunked.py` contains a complete custom implementation of eigenvalue feature computation that duplicates logic from `core/eigenvalues.py`.

**Evidence:**

##### `features_gpu_chunked.py` (Lines 1779-1905)

```python
def compute_eigenvalue_features(
    self,
    points: np.ndarray,
    normals: np.ndarray,
    neighbors_indices: np.ndarray,
    start_idx: int = None,
    end_idx: int = None
) -> Dict[str, np.ndarray]:
    """Compute eigenvalue-based features (FULL GPU-accelerated with chunking support)."""
    # ... [150+ lines of custom implementation]

    # Custom covariance computation
    cov_matrices = xp.einsum('nki,nkj->nij', centered, centered) / (k - 1)

    # Custom eigenvalue computation with batching
    eigenvalues = xp.linalg.eigvalsh(cov_matrices)
    eigenvalues = xp.sort(eigenvalues, axis=1)[:, ::-1]

    # Custom feature computation (duplicates core/eigenvalues.py)
    λ0 = eigenvalues[:, 0]
    λ1 = eigenvalues[:, 1]
    λ2 = eigenvalues[:, 2]
    sum_eigenvalues = λ0 + λ1 + λ2

    # Eigenentropy calculation (DUPLICATED from core)
    p0 = λ0 / (sum_eigenvalues + 1e-10)
    p1 = λ1 / (sum_eigenvalues + 1e-10)
    p2 = λ2 / (sum_eigenvalues + 1e-10)
    eigenentropy = -(p0 * xp.log(p0 + 1e-10) + ...)

    # Omnivariance (DUPLICATED from core)
    omnivariance = xp.cbrt(λ0 * λ1 * λ2)

    # Change of curvature (DUPLICATED from core)
    eigenvalue_variance = xp.var(eigenvalues, axis=1)
    change_curvature = xp.sqrt(eigenvalue_variance)

    return {
        'eigenvalue_1': λ0.astype(np.float32),
        'eigenvalue_2': λ1.astype(np.float32),
        'eigenvalue_3': λ2.astype(np.float32),
        'sum_eigenvalues': sum_eigenvalues.astype(np.float32),
        'eigenentropy': eigenentropy.astype(np.float32),
        'omnivariance': omnivariance.astype(np.float32),
        'change_curvature': change_curvature.astype(np.float32),
    }
```

##### `core/eigenvalues.py` (Lines 14-85) - CANONICAL VERSION

```python
def compute_eigenvalue_features(
    eigenvalues: np.ndarray,
    epsilon: float = 1e-10,
    include_all: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive eigenvalue-based geometric features.

    CANONICAL IMPLEMENTATION - should be used by all modules.
    """
    # Extract eigenvalues
    lambda1 = eigenvalues[:, 0]
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 2]

    # Linearity: (λ1 - λ2) / λ1
    features['linearity'] = (lambda1 - lambda2) / (lambda1 + epsilon)

    # Planarity: (λ2 - λ3) / λ1
    features['planarity'] = (lambda2 - lambda3) / (lambda1 + epsilon)

    # [... more features ...]
```

**Problem:** The GPU implementation computes eigenvalues AND features together, while the core module expects pre-computed eigenvalues. This architectural mismatch prevents reuse.

#### Recommendation: 🔧 REFACTOR REQUIRED

**Proposed Solution:**

1. Split GPU eigenvalue computation into two stages:

   - Stage 1: Compute covariances and eigenvalues (GPU-accelerated)
   - Stage 2: Compute features from eigenvalues (use `core.compute_eigenvalue_features`)

2. Create GPU-aware wrapper:

```python
def compute_eigenvalue_features_gpu(
    self,
    points: np.ndarray,
    neighbors_indices: np.ndarray,
    start_idx: int = None,
    end_idx: int = None
) -> Dict[str, np.ndarray]:
    """
    GPU-accelerated eigenvalue feature computation using core logic.
    """
    # Stage 1: Compute eigenvalues on GPU
    eigenvalues_gpu = self._compute_eigenvalues_from_neighbors_gpu(
        points, neighbors_indices, start_idx, end_idx
    )

    # Transfer to CPU for feature computation
    eigenvalues_cpu = self._to_cpu(eigenvalues_gpu)

    # Stage 2: Use CORE canonical implementation
    from ..features.core import compute_eigenvalue_features
    features = compute_eigenvalue_features(
        eigenvalues_cpu,
        epsilon=1e-10,
        include_all=True
    )

    return features
```

---

### 2.2 Density Features

#### Current State: ❌ DUPLICATED LOGIC

**Issue:** Custom density feature implementation in `features_gpu_chunked.py` bypasses `core/density.py`.

**Evidence:**

##### `features_gpu_chunked.py` (Lines 2208-2309)

```python
def compute_density_features(
    self,
    points: np.ndarray,
    neighbors_indices: np.ndarray,
    radius_2m: float = 2.0,
    start_idx: int = None,
    end_idx: int = None,
    points_gpu=None
) -> Dict[str, np.ndarray]:
    """Compute density features (CUSTOM GPU IMPLEMENTATION)."""

    # Custom distance computation
    distances = xp.linalg.norm(
        neighbors - chunk_points_gpu[:, cp.newaxis, :],
        axis=2
    )

    # Custom density calculation (DUPLICATES core/density.py)
    mean_distances = xp.mean(distances[:, 1:], axis=1)
    density = xp.clip(1.0 / (mean_distances + 1e-8), 0.0, 1000.0)

    # Custom neighborhood extent (NOT in core)
    neighborhood_extent = xp.max(distances, axis=1)

    # Custom height extent ratio
    z_coords = neighbors[:, :, 2]
    z_std = xp.std(z_coords, axis=1)
    height_extent_ratio = z_std / (neighborhood_extent + 1e-8)

    # Count points within radius (CUSTOM LOGIC)
    within_radius = xp.sum(distances <= radius_2m, axis=1)
    num_points_2m = within_radius.astype(xp.float32)

    return {
        'density': density.astype(np.float32),
        'num_points_2m': num_points_2m.astype(np.float32),
        'neighborhood_extent': neighborhood_extent.astype(np.float32),
        'height_extent_ratio': np.clip(height_extent_ratio, 0.0, 1.0).astype(np.float32),
        'vertical_std': vertical_std.astype(np.float32),
    }
```

##### `core/density.py` (Lines 14-90) - CANONICAL VERSION

```python
def compute_density_features(
    points: np.ndarray,
    k_neighbors: int = 20,
    search_radius: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive density-based features.

    CANONICAL IMPLEMENTATION.
    """
    # Build KD-tree for neighbor search
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Mean distance to k-nearest neighbors
    mean_distance = np.mean(distances[:, 1:], axis=1).astype(np.float32)

    # Point density: number of neighbors / volume
    max_distance = distances[:, -1]
    volume = (4.0 / 3.0) * np.pi * (max_distance ** 3)
    point_density = (k_neighbors / volume).astype(np.float32)

    # [... more features ...]
```

**Problem:** Different feature sets returned:

- Core: `point_density`, `mean_distance`, `std_distance`, `local_density_ratio`, `density`
- GPU Chunked: `density`, `num_points_2m`, `neighborhood_extent`, `height_extent_ratio`, `vertical_std`

#### Recommendation: 🔧 REFACTOR REQUIRED

**Proposed Solution:**

1. Standardize feature names and calculations
2. Create GPU-optimized version that calls core for final feature computation
3. Add missing features to core if they're valuable

---

### 2.3 Architectural Features

#### Current State: ❌ DUPLICATED LOGIC

**Issue:** Complete custom implementation in `features_gpu_chunked.py` with no reuse of `core/architectural.py`.

**Evidence:**

##### `features_gpu_chunked.py` (Lines 1930-2044)

```python
def compute_architectural_features(
    self,
    points: np.ndarray,
    normals: np.ndarray,
    neighbors_indices: np.ndarray,
    start_idx: int = None,
    end_idx: int = None
) -> Dict[str, np.ndarray]:
    """Compute architectural features (FULL GPU-accelerated with chunking)."""

    # Custom eigenvalue computation AGAIN
    cov_matrices = xp.einsum('nki,nkj->nij', centered, centered) / (k - 1)
    eigenvalues = xp.linalg.eigvalsh(cov_matrices)

    # Custom edge strength (NOT in core)
    edge_strength = xp.clip((λ0 - λ2) / (λ0 + 1e-8), 0.0, 1.0)

    # Custom corner likelihood (NOT in core)
    corner_likelihood = xp.clip(λ2 / (λ0 + 1e-8), 0.0, 1.0)

    # Custom normal variation
    normal_diffs = neighbor_normals - chunk_normals_gpu[:, cp.newaxis, :]
    normal_variation = xp.linalg.norm(normal_diffs, axis=2).mean(axis=1)

    # Custom overhang indicator (NOT in core)
    vertical_diffs = neighbor_normals[:, :, 2] - chunk_normals_gpu[:, 2:3]
    overhang_indicator = xp.abs(vertical_diffs).mean(axis=1)

    # Custom surface roughness (NOT in core)
    distances_to_centroid = xp.linalg.norm(centered, axis=2)
    surface_roughness = xp.std(distances_to_centroid, axis=1)

    return {
        'edge_strength': edge_strength.astype(np.float32),
        'corner_likelihood': corner_likelihood.astype(np.float32),
        'overhang_indicator': np.clip(overhang_indicator, 0.0, 1.0).astype(np.float32),
        'surface_roughness': surface_roughness.astype(np.float32),
    }
```

##### `core/architectural.py` (Lines 14-76) - CANONICAL VERSION

```python
def compute_architectural_features(
    points: np.ndarray,
    normals: np.ndarray,
    eigenvalues: np.ndarray,
    epsilon: float = 1e-10
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive architectural features.

    CANONICAL IMPLEMENTATION.
    """
    features = {}

    # Verticality (REUSED FUNCTION)
    features['verticality'] = compute_verticality(normals)

    # Horizontality (REUSED FUNCTION)
    features['horizontality'] = compute_horizontality(normals)

    # Wall likelihood (REUSED FUNCTION)
    planarity = compute_planarity(eigenvalues, epsilon)
    features['wall_likelihood'] = compute_wall_likelihood(normals, planarity)

    # Roof likelihood (REUSED FUNCTION)
    features['roof_likelihood'] = compute_roof_likelihood(normals, planarity)

    # [... more features ...]
```

**Problem:** GPU chunked implementation has different feature set and doesn't reuse any core functions. Features like `edge_strength`, `corner_likelihood`, `overhang_indicator` are unique to GPU implementation.

#### Recommendation: 🔧 CONSOLIDATE REQUIRED

**Proposed Solution:**

1. Move unique GPU features to core module if they're valuable
2. Align feature names and computation logic
3. Use core functions with GPU-computed eigenvalues as input

---

### 2.4 Geometric Features

#### Current State: ⚠️ MIXED - Some Integration

**Evidence:**

##### `features_gpu_chunked.py` (Lines 2045-2202)

```python
def _compute_geometric_features_from_neighbors(
    self,
    points: np.ndarray,
    neighbors_indices: np.ndarray,
    chunk_points: np.ndarray,
    points_gpu=None
) -> Dict[str, np.ndarray]:
    """
    Compute geometric features directly from pre-computed neighbor indices.

    OPTIMIZATION: Avoids rebuilding KDTree.
    """
    # ⚡ MEGA-OPTIMIZATION: Compute ENTIRE geometric features on GPU!

    if self.use_gpu and cp is not None:
        # FULLY GPU-ACCELERATED PATH
        neighbors_gpu = points_gpu[neighbors_indices_gpu]
        chunk_points_gpu = cp.asarray(chunk_points)

        # Covariance computation
        centroids_gpu = cp.mean(neighbors_gpu, axis=1, keepdims=True)
        centered_gpu = neighbors_gpu - centroids_gpu
        cov_matrices_gpu = cp.einsum('nki,nkj->nij', centered_gpu, centered_gpu) / (k - 1)

        # Eigenvalue computation with batching
        # [... 100+ lines of custom implementation ...]

        # Compute all geometric features from eigenvalues
        λ0_gpu = eigenvalues_gpu[:, 0]
        λ1_gpu = eigenvalues_gpu[:, 1]
        λ2_gpu = eigenvalues_gpu[:, 2]

        λ_sum_gpu = λ0_gpu + λ1_gpu + λ2_gpu

        # Linearity, planarity, etc. (DUPLICATES core logic)
        linearity = (λ0_gpu - λ1_gpu) / (λ0_gpu + 1e-10)
        planarity = (λ1_gpu - λ2_gpu) / (λ0_gpu + 1e-10)
        sphericity = λ2_gpu / (λ0_gpu + 1e-10)
        # [... more features ...]
```

#### Recommendation: 🔧 REFACTOR REQUIRED

Use core's `extract_geometric_features()` after computing eigenvalues on GPU.

---

### 2.5 Height Features

#### Current State: ✅ GOOD INTEGRATION

**Evidence:**

```python
# features_gpu_chunked.py uses core implementation
from .core.height import compute_height_above_ground

# Usage (Line 3183):
chunk_height = gpu_computer.compute_height_above_ground(
    chunk_points_cpu, chunk_classification
)
```

**Status:** ✅ Properly integrated with core module.

---

### 2.6 Curvature Features

#### Current State: ⚠️ MIXED

**Evidence:**

```python
# features_gpu_chunked.py imports but rarely uses core
from .core.curvature import compute_curvature_from_normals

# But has custom implementations (Lines 1358-1790):
def _compute_curvature_with_streams(...)
def _compute_curvature_batched(...)
def _compute_curvature_from_neighbors_gpu(...)
def compute_curvature_chunked(...)
def _compute_curvature_per_chunk(...)
```

**Problem:** Multiple custom curvature computation methods that could potentially use core logic.

#### Recommendation: ⚠️ EVALUATE NECESSITY

Assess whether custom GPU curvature methods provide significant performance benefit over using core with GPU-computed normals.

---

## 3. Code Duplication Analysis

### 3.1 Quantitative Metrics

| Feature Type  | Core Module (lines) | GPU Chunked Custom (lines) | Duplication % |
| ------------- | ------------------- | -------------------------- | ------------- |
| Eigenvalues   | ~120                | ~150                       | ~80%          |
| Density       | ~90                 | ~100                       | ~70%          |
| Architectural | ~150                | ~115                       | ~60%          |
| Geometric     | ~200                | ~200                       | ~75%          |
| **TOTAL**     | **~560**            | **~565**                   | **~71%**      |

**Estimated duplicate logic:** ~400 lines that could be consolidated.

### 3.2 Maintenance Risk

**High Risk Areas:**

1. **Eigenvalue computation logic** - duplicated in 3+ locations
2. **Feature formulas** - minor differences across implementations
3. **Parameter handling** - inconsistent epsilon values, normalization
4. **Feature naming** - same features with different names

**Impact:**

- Bug fixes need to be applied in multiple places
- Feature improvements don't propagate automatically
- Testing overhead (same logic tested multiple times)
- Documentation inconsistencies

---

## 4. Performance Considerations

### 4.1 Why Custom GPU Implementations Exist

**Valid Reasons:**

1. **Memory efficiency** - Chunked processing to avoid VRAM exhaustion
2. **Computation overlap** - GPU can compute covariances while doing eigendecomposition
3. **Reduced data transfers** - Computing multiple features on GPU before CPU transfer
4. **CUDA-specific optimizations** - Batch size limits for cuSOLVER

**Example from code (Lines 1834-1855):**

```python
# ⚡ FIX: cuSOLVER has batch size limits - sub-chunk eigenvalue computation
max_batch_size = 500000
if use_gpu and N > max_batch_size:
    # Sub-chunk eigenvalue computation for GPU
    eigenvalues = xp.zeros((N, 3), dtype=xp.float32)
    num_subbatches = (N + max_batch_size - 1) // max_batch_size

    for sb_idx in range(num_subbatches):
        sb_start = sb_idx * max_batch_size
        sb_end = min((sb_idx + 1) * max_batch_size, N)
        sb_eigenvalues = xp.linalg.eigvalsh(cov_matrices[sb_start:sb_end])
```

### 4.2 Performance vs. Maintainability Trade-off

**Current State:**

- ✅ Excellent performance (16× speedup documented)
- ❌ High maintenance cost
- ❌ Code duplication

**Proposed State:**

- ✅ Maintain performance benefits
- ✅ Reduce duplication via layered approach
- ✅ GPU-specific optimizations in wrapper layer

---

## 5. Recommendations

### 5.1 Immediate Actions (High Priority)

#### 1. Create GPU-Core Bridge Module ⭐⭐⭐

**Create:** `ign_lidar/features/core/gpu_bridge.py`

```python
"""
GPU-aware bridge to core feature implementations.

This module provides GPU-optimized wrappers around core feature functions
while maintaining performance and avoiding code duplication.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class GPUCoreBridge:
    """
    Bridge between GPU computations and core feature implementations.

    Handles:
    - GPU-accelerated covariance/eigenvalue computation
    - Efficient CPU/GPU data transfer
    - Delegation to core feature functions
    - Batching for GPU memory limits
    """

    def __init__(self, use_gpu: bool = True, batch_size: int = 500_000):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.batch_size = batch_size

    def compute_eigenvalues_gpu(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated eigenvalue computation from neighbor indices.

        Returns:
            eigenvalues: [N, 3] numpy array on CPU, ready for core functions
        """
        if not self.use_gpu:
            return self._compute_eigenvalues_cpu(points, neighbors_indices)

        # GPU path with batching
        N = len(neighbors_indices)
        k = neighbors_indices.shape[1]

        # Transfer to GPU
        points_gpu = cp.asarray(points)
        neighbors_indices_gpu = cp.asarray(neighbors_indices)

        # Fetch neighbors
        neighbors_gpu = points_gpu[neighbors_indices_gpu]

        # Compute covariances
        centroids_gpu = cp.mean(neighbors_gpu, axis=1, keepdims=True)
        centered_gpu = neighbors_gpu - centroids_gpu
        cov_matrices_gpu = cp.einsum('nki,nkj->nij', centered_gpu, centered_gpu) / (k - 1)

        # Compute eigenvalues with batching (cuSOLVER limit)
        if N > self.batch_size:
            eigenvalues_gpu = self._compute_eigenvalues_batched(cov_matrices_gpu, N)
        else:
            eigenvalues_gpu = cp.linalg.eigvalsh(cov_matrices_gpu)
            eigenvalues_gpu = cp.sort(eigenvalues_gpu, axis=1)[:, ::-1]

        # Transfer to CPU
        eigenvalues = cp.asnumpy(eigenvalues_gpu)

        return eigenvalues

    def compute_eigenvalue_features_gpu(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray,
        epsilon: float = 1e-10
    ) -> Dict[str, np.ndarray]:
        """
        GPU-accelerated eigenvalue feature computation using core logic.

        This is the RECOMMENDED way to compute eigenvalue features with GPU.
        """
        # Step 1: Compute eigenvalues on GPU (fast)
        eigenvalues = self.compute_eigenvalues_gpu(points, neighbors_indices)

        # Step 2: Use core canonical implementation (maintainable)
        from . import compute_eigenvalue_features
        features = compute_eigenvalue_features(eigenvalues, epsilon=epsilon)

        return features

    def _compute_eigenvalues_batched(
        self,
        cov_matrices_gpu,
        N: int
    ):
        """Handle cuSOLVER batch size limits."""
        eigenvalues_gpu = cp.zeros((N, 3), dtype=cp.float32)
        num_batches = (N + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, N)

            batch_eigenvalues = cp.linalg.eigvalsh(cov_matrices_gpu[start:end])
            batch_eigenvalues = cp.sort(batch_eigenvalues, axis=1)[:, ::-1]
            eigenvalues_gpu[start:end] = batch_eigenvalues

        return eigenvalues_gpu

    def _compute_eigenvalues_cpu(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray
    ) -> np.ndarray:
        """CPU fallback."""
        neighbors = points[neighbors_indices]
        centroids = np.mean(neighbors, axis=1, keepdims=True)
        centered = neighbors - centroids
        k = neighbors_indices.shape[1]
        cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)
        eigenvalues = np.linalg.eigvalsh(cov_matrices)
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
        return eigenvalues
```

**Benefits:**

- ✅ Maintains GPU performance benefits
- ✅ Eliminates code duplication
- ✅ Uses core canonical implementations
- ✅ Single source of truth for feature formulas
- ✅ Easy to test and maintain

#### 2. Refactor `features_gpu_chunked.py` to Use Bridge ⭐⭐⭐

**Replace custom implementations with:**

```python
class GPUChunkedFeatureComputer:
    def __init__(self, ...):
        # ... existing init code ...

        # NEW: Initialize GPU-Core bridge
        from .core.gpu_bridge import GPUCoreBridge
        self.gpu_bridge = GPUCoreBridge(
            use_gpu=self.use_gpu,
            batch_size=500_000  # cuSOLVER limit
        )

    def compute_eigenvalue_features(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        neighbors_indices: np.ndarray,
        start_idx: int = None,
        end_idx: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalue-based features using core implementation.

        ✅ REFACTORED: Now uses ign_lidar.features.core through GPU bridge.
        """
        # Use GPU bridge instead of custom implementation
        features = self.gpu_bridge.compute_eigenvalue_features_gpu(
            points=points,
            neighbors_indices=neighbors_indices,
            epsilon=1e-10
        )

        # Handle chunking if needed
        if start_idx is not None and end_idx is not None:
            for key in features:
                features[key] = features[key][start_idx:end_idx]

        return features
```

**Impact:**

- 🔻 Removes ~150 lines of duplicate code
- ✅ Uses canonical feature implementations
- ✅ Maintains GPU performance
- ✅ Easier to maintain and test

#### 3. Standardize Feature Names and Outputs ⭐⭐

**Create:** `FEATURE_SPECIFICATION.md`

Document the canonical feature set with:

- Feature names (standardized)
- Data types (np.float32)
- Value ranges ([0, 1] for normalized features)
- Computation formulas
- Expected use cases

**Example:**

```markdown
### Eigenvalue Features

| Feature Name      | Formula              | Range       | Description                    |
| ----------------- | -------------------- | ----------- | ------------------------------ |
| `linearity`       | (λ1 - λ2) / λ1       | [0, 1]      | Linear structure indicator     |
| `planarity`       | (λ2 - λ3) / λ1       | [0, 1]      | Planar structure indicator     |
| `sphericity`      | λ3 / λ1              | [0, 1]      | Volumetric structure indicator |
| `sum_eigenvalues` | λ1 + λ2 + λ3         | [0, ∞)      | Total local variance           |
| `eigenentropy`    | -Σ(pi \* log(pi))    | [0, log(3)] | Structural complexity          |
| `omnivariance`    | (λ1 _ λ2 _ λ3)^(1/3) | [0, ∞)      | Local volume measure           |
```

### 5.2 Medium-Term Actions (Medium Priority)

#### 4. Add Missing Features to Core ⭐⭐

Features currently only in GPU implementations that should be in core:

From `compute_architectural_features`:

- `edge_strength`
- `corner_likelihood`
- `overhang_indicator`
- `surface_roughness`

From `compute_density_features`:

- `neighborhood_extent`
- `height_extent_ratio`
- `vertical_std`
- `num_points_2m`

#### 5. Create Comprehensive Integration Tests ⭐⭐

**Create:** `tests/test_gpu_core_integration.py`

```python
"""
Integration tests to ensure GPU implementations match core implementations.
"""

import pytest
import numpy as np
from ign_lidar.features.core import compute_eigenvalue_features
from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer

def test_eigenvalue_features_consistency():
    """Test that GPU and core eigenvalue features match."""
    # Generate test data
    points = np.random.rand(1000, 3).astype(np.float32)

    # Compute eigenvalues (both methods should accept same input)
    # ... test code ...

    # Compare results (should be nearly identical)
    for key in expected_features:
        assert key in gpu_features
        assert key in core_features
        np.testing.assert_allclose(
            gpu_features[key],
            core_features[key],
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Feature {key} differs between GPU and core"
        )
```

### 5.3 Long-Term Actions (Low Priority)

#### 6. Deprecate Standalone GPU Functions ⭐

Add deprecation warnings to standalone functions:

```python
def compute_eigenvalue_features(...):
    """
    DEPRECATED: Use ign_lidar.features.core.compute_eigenvalue_features()
    with GPU bridge instead.

    This function will be removed in version 6.0.
    """
    warnings.warn(
        "compute_eigenvalue_features in features_gpu_chunked is deprecated. "
        "Use ign_lidar.features.core.compute_eigenvalue_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing code ...
```

#### 7. Documentation Update ⭐

Update documentation to recommend:

1. Core module as primary API
2. GPU bridge for performance
3. Migration guide from old to new API

---

## 6. Migration Plan

### Phase 1: Foundation (Week 1)

- [ ] Create `gpu_bridge.py` module
- [ ] Implement `compute_eigenvalues_gpu()` method
- [ ] Add unit tests for bridge module

### Phase 2: Eigenvalue Integration (Week 2)

- [ ] Refactor `compute_eigenvalue_features()` in `features_gpu_chunked.py`
- [ ] Add integration tests
- [ ] Verify performance is maintained

### Phase 3: Density & Architectural (Week 3)

- [ ] Standardize density features
- [ ] Refactor `compute_density_features()`
- [ ] Refactor `compute_architectural_features()`
- [ ] Add missing features to core

### Phase 4: Testing & Documentation (Week 4)

- [ ] Comprehensive integration tests
- [ ] Update API documentation
- [ ] Create migration guide
- [ ] Add deprecation warnings

### Phase 5: Cleanup (Week 5)

- [ ] Remove duplicate code
- [ ] Consolidate feature computation logic
- [ ] Final performance validation

---

## 7. Risk Assessment

### Technical Risks

| Risk                   | Likelihood | Impact | Mitigation                                         |
| ---------------------- | ---------- | ------ | -------------------------------------------------- |
| Performance regression | Low        | High   | Benchmark before/after, maintain GPU optimizations |
| Breaking changes       | Medium     | Medium | Thorough testing, deprecation period               |
| VRAM exhaustion        | Low        | Medium | Maintain chunking and batching logic               |
| Feature drift          | Low        | Low    | Integration tests, specification document          |

### Project Risks

| Risk                  | Likelihood | Impact | Mitigation                        |
| --------------------- | ---------- | ------ | --------------------------------- |
| Scope creep           | Medium     | Medium | Phased approach, clear milestones |
| Resource availability | Low        | Low    | Well-documented code, clear tasks |
| Testing complexity    | Medium     | Medium | Automated test suite, CI/CD       |

---

## 8. Success Criteria

### Quantitative Metrics

- [ ] Code duplication reduced by >70% (~400 lines removed)
- [ ] Test coverage increased to >90%
- [ ] Performance maintained within 5% of current
- [ ] All feature outputs identical (within floating-point tolerance)

### Qualitative Metrics

- [ ] Single source of truth for feature formulas
- [ ] Clear separation of GPU optimizations vs. feature logic
- [ ] Easy to add new features (core module)
- [ ] GPU implementations use core module
- [ ] Documentation clear and complete

---

## 9. Conclusion

### Current State Summary

The codebase has significant code duplication between GPU implementations and the core module, with ~71% duplicate logic across feature computation methods. While GPU implementations achieve excellent performance (16× speedup), they maintain separate implementations that don't leverage the canonical core features.

### Path Forward

The recommended approach is to create a **GPU-Core bridge layer** that:

1. Handles GPU-specific optimizations (batching, chunking, memory management)
2. Delegates feature computation to core canonical implementations
3. Maintains performance while eliminating duplication

### Expected Benefits

- ✅ **Maintainability:** Single source of truth for feature formulas
- ✅ **Performance:** GPU optimizations preserved
- ✅ **Reliability:** Fewer bugs, easier testing
- ✅ **Extensibility:** Easy to add new features
- ✅ **Documentation:** Clearer API, better docs

### Next Steps

1. Review and approve this audit report
2. Create GPU bridge module (Phase 1)
3. Begin integration with eigenvalue features (Phase 2)
4. Continue with remaining feature types

---

## Appendix A: References

### Key Files Analyzed

- `ign_lidar/features/features_gpu.py` (1,210 lines)
- `ign_lidar/features/features_gpu_chunked.py` (3,470 lines)
- `ign_lidar/features/strategy_gpu_chunked.py` (370 lines)
- `ign_lidar/features/core/__init__.py`
- `ign_lidar/features/core/eigenvalues.py` (231 lines)
- `ign_lidar/features/core/density.py` (360 lines)
- `ign_lidar/features/core/architectural.py` (337 lines)

### Related Documentation

- `docs/gpu-optimization-guide.md`
- `IMPLEMENTATION_GUIDE.md`
- `IMPLEMENTATION_STATUS.md`

---

**End of Audit Report**
