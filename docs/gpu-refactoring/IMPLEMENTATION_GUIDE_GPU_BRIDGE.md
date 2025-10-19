# GPU Bridge Implementation Guide - Phase 1

**Date:** October 19, 2025  
**Phase:** 1 - Foundation  
**Duration:** Week 1  
**Status:** Ready to Start

---

## 🎯 Objective

Create the GPU-Core Bridge module that enables GPU-accelerated feature computation while using canonical core implementations for feature logic.

---

## 📋 Prerequisites

Before starting, ensure:

- ✅ Audit documents reviewed and approved
- ✅ Development environment set up with:
  - CuPy installed (`pip install cupy-cuda11x` or `cupy-cuda12x`)
  - PyTorch or CUDA toolkit available
  - pytest for testing
- ✅ Baseline performance benchmarks recorded
- ✅ Git branch created: `feature/gpu-core-bridge`

---

## 📁 File Structure

### New Files to Create

```
ign_lidar/features/core/
├── gpu_bridge.py              # Main bridge implementation (NEW)
└── __init__.py                # Update exports

tests/
├── test_gpu_bridge.py         # Unit tests (NEW)
└── fixtures/
    └── test_data.py           # Test data generators (NEW)

scripts/
└── benchmark_gpu_bridge.py    # Performance validation (NEW)
```

---

## 🔨 Implementation Steps

### Step 1: Create GPU Bridge Module Structure

**File:** `ign_lidar/features/core/gpu_bridge.py`

```python
"""
GPU-Core Bridge for Feature Computation

This module provides GPU-optimized wrappers around core feature implementations
while maintaining performance and avoiding code duplication.

Architecture:
    1. GPU Layer: Accelerated eigenvalue/covariance computation
    2. Core Layer: Canonical feature computation (single source of truth)
    3. Bridge Layer: Efficient data transfer and delegation

Example:
    >>> from ign_lidar.features.core.gpu_bridge import GPUCoreBridge
    >>> bridge = GPUCoreBridge(use_gpu=True)
    >>>
    >>> # Compute eigenvalue features using GPU + core
    >>> features = bridge.compute_eigenvalue_features_gpu(
    ...     points=points,
    ...     neighbors_indices=neighbor_indices
    ... )

Author: IGN LiDAR HD Development Team
Date: October 2025
Version: 1.0.0
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)

# GPU imports with fallback
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("✓ CuPy available - GPU bridge enabled")
except ImportError:
    logger.warning("⚠ CuPy not available - GPU bridge will use CPU fallback")
    cp = None


class GPUCoreBridge:
    """
    Bridge between GPU computations and core feature implementations.

    This class handles:
    - GPU-accelerated covariance and eigenvalue computation
    - Efficient CPU/GPU data transfer
    - Delegation to core canonical feature functions
    - Batching for GPU memory limits (cuSOLVER)

    Performance:
    - Eigenvalue computation: GPU-accelerated (10-50× faster)
    - Feature computation: CPU using core module (maintainable)
    - Overall speedup: ~8-15× for large datasets

    Memory Management:
    - Automatic batching for large datasets (cuSOLVER 500K limit)
    - Configurable batch size
    - Efficient memory cleanup

    Attributes:
        use_gpu (bool): Whether GPU is available and enabled
        batch_size (int): Maximum batch size for eigenvalue computation
        _gpu_available (bool): Whether CuPy is available
    """

    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: int = 500_000,
        verbose: bool = False
    ):
        """
        Initialize GPU-Core Bridge.

        Args:
            use_gpu: Enable GPU if available (default: True)
            batch_size: Maximum points per batch for eigenvalue computation.
                       Limited by cuSOLVER (default: 500K)
            verbose: Enable detailed logging (default: False)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.batch_size = batch_size
        self.verbose = verbose
        self._gpu_available = GPU_AVAILABLE

        if self.use_gpu:
            logger.info(
                f"GPU Bridge initialized: GPU mode, batch_size={batch_size:,}"
            )
        else:
            logger.info("GPU Bridge initialized: CPU fallback mode")
            if use_gpu and not GPU_AVAILABLE:
                warnings.warn(
                    "GPU requested but CuPy not available. "
                    "Using CPU fallback. Install CuPy for GPU acceleration.",
                    RuntimeWarning
                )

    def compute_eigenvalues_gpu(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray,
        return_covariances: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute eigenvalues from neighbor indices using GPU acceleration.

        This is the core GPU-accelerated computation that enables fast
        eigenvalue calculation for large point clouds.

        Args:
            points: [N, 3] point cloud coordinates
            neighbors_indices: [N, k] indices of k-nearest neighbors
            return_covariances: If True, also return covariance matrices
                               (useful for advanced features)

        Returns:
            eigenvalues: [N, 3] sorted eigenvalues (λ1 >= λ2 >= λ3) on CPU
            covariances (optional): [N, 3, 3] covariance matrices on CPU

        Notes:
            - Eigenvalues are automatically sorted in descending order
            - Results are transferred to CPU for compatibility with core module
            - Batching is automatic for large datasets (cuSOLVER limits)

        Performance:
            - Small datasets (<500K): ~10-20× faster than CPU
            - Large datasets (>500K): ~8-15× faster (batching overhead)

        Example:
            >>> eigenvalues = bridge.compute_eigenvalues_gpu(points, neighbors)
            >>> print(eigenvalues.shape)  # (N, 3)
            >>> print(eigenvalues[0])     # [λ1, λ2, λ3] for first point
        """
        if not self.use_gpu:
            return self._compute_eigenvalues_cpu(
                points, neighbors_indices, return_covariances
            )

        N = len(neighbors_indices)
        k = neighbors_indices.shape[1]

        if self.verbose:
            logger.info(
                f"Computing eigenvalues for {N:,} points with k={k} neighbors "
                f"(GPU mode, batch_size={self.batch_size:,})"
            )

        # Transfer to GPU
        points_gpu = cp.asarray(points, dtype=cp.float32)
        neighbors_indices_gpu = cp.asarray(neighbors_indices, dtype=cp.int32)

        # Fetch neighbor points: [N, k, 3]
        neighbors_gpu = points_gpu[neighbors_indices_gpu]

        # Compute covariance matrices
        centroids_gpu = cp.mean(neighbors_gpu, axis=1, keepdims=True)  # [N, 1, 3]
        centered_gpu = neighbors_gpu - centroids_gpu  # [N, k, 3]

        # Covariance: [N, 3, 3]
        cov_matrices_gpu = cp.einsum(
            'nki,nkj->nij', centered_gpu, centered_gpu
        ) / (k - 1)

        # Compute eigenvalues with batching if needed
        if N > self.batch_size:
            eigenvalues_gpu = self._compute_eigenvalues_batched_gpu(
                cov_matrices_gpu, N
            )
        else:
            eigenvalues_gpu = cp.linalg.eigvalsh(cov_matrices_gpu)
            # Sort descending (core module expects this)
            eigenvalues_gpu = cp.sort(eigenvalues_gpu, axis=1)[:, ::-1]

        # Clamp to non-negative (numerical stability)
        eigenvalues_gpu = cp.maximum(eigenvalues_gpu, 0.0)

        # Transfer to CPU
        eigenvalues = cp.asnumpy(eigenvalues_gpu).astype(np.float32)

        if return_covariances:
            covariances = cp.asnumpy(cov_matrices_gpu).astype(np.float32)
            return eigenvalues, covariances

        return eigenvalues

    def _compute_eigenvalues_batched_gpu(
        self,
        cov_matrices_gpu,
        N: int
    ):
        """
        Compute eigenvalues in batches to handle cuSOLVER limits.

        cuSOLVER has a maximum batch size of ~500K matrices for eigvalsh.
        This method automatically batches large datasets.

        Args:
            cov_matrices_gpu: [N, 3, 3] covariance matrices on GPU
            N: Number of points

        Returns:
            eigenvalues_gpu: [N, 3] eigenvalues on GPU
        """
        eigenvalues_gpu = cp.zeros((N, 3), dtype=cp.float32)
        num_batches = (N + self.batch_size - 1) // self.batch_size

        if self.verbose:
            logger.info(
                f"Batching eigenvalue computation: {N:,} points → "
                f"{num_batches} batches of {self.batch_size:,}"
            )

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, N)

            # Compute eigenvalues for batch
            batch_eigenvalues = cp.linalg.eigvalsh(
                cov_matrices_gpu[start:end]
            )

            # Sort descending
            batch_eigenvalues = cp.sort(batch_eigenvalues, axis=1)[:, ::-1]

            eigenvalues_gpu[start:end] = batch_eigenvalues

            if self.verbose and batch_idx % 10 == 0:
                logger.debug(
                    f"  Batch {batch_idx + 1}/{num_batches}: "
                    f"{end - start:,} points processed"
                )

        return eigenvalues_gpu

    def _compute_eigenvalues_cpu(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray,
        return_covariances: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        CPU fallback for eigenvalue computation.

        Used when GPU is not available or explicitly disabled.

        Args:
            points: [N, 3] point cloud coordinates
            neighbors_indices: [N, k] indices of k-nearest neighbors
            return_covariances: If True, also return covariance matrices

        Returns:
            eigenvalues: [N, 3] sorted eigenvalues on CPU
            covariances (optional): [N, 3, 3] covariance matrices
        """
        if self.verbose:
            logger.info(
                f"Computing eigenvalues for {len(neighbors_indices):,} points "
                f"(CPU fallback mode)"
            )

        N = len(neighbors_indices)
        k = neighbors_indices.shape[1]

        # Fetch neighbors
        neighbors = points[neighbors_indices]  # [N, k, 3]

        # Compute covariances
        centroids = np.mean(neighbors, axis=1, keepdims=True)  # [N, 1, 3]
        centered = neighbors - centroids  # [N, k, 3]
        cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (k - 1)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrices)

        # Sort descending
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]

        # Clamp to non-negative
        eigenvalues = np.maximum(eigenvalues, 0.0).astype(np.float32)

        if return_covariances:
            return eigenvalues, cov_matrices.astype(np.float32)

        return eigenvalues

    def compute_eigenvalue_features_gpu(
        self,
        points: np.ndarray,
        neighbors_indices: np.ndarray,
        epsilon: float = 1e-10,
        include_all: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Compute eigenvalue-based features using GPU + core module.

        This is the RECOMMENDED way to compute eigenvalue features.
        It combines GPU acceleration for eigenvalue computation with
        the canonical core module for feature computation.

        Architecture:
            1. GPU: Fast eigenvalue computation from neighbors
            2. CPU Transfer: Efficient single transfer
            3. Core: Canonical feature computation (maintainable)

        Args:
            points: [N, 3] point cloud coordinates
            neighbors_indices: [N, k] indices of k-nearest neighbors
            epsilon: Small value to prevent division by zero
            include_all: If True, compute all features (default: True)

        Returns:
            features: Dictionary with eigenvalue-based features:
                - 'linearity': Linear structure indicator [0, 1]
                - 'planarity': Planar structure indicator [0, 1]
                - 'sphericity': Volumetric structure indicator [0, 1]
                - 'anisotropy': Degree of anisotropy [0, 1]
                - 'eigenentropy': Structural complexity [0, log(3)]
                - 'omnivariance': Local volume measure [0, ∞)
                - 'sum_eigenvalues': Total variance [0, ∞)
                - (if include_all) 'change_of_curvature', 'surface_variation'

        Example:
            >>> bridge = GPUCoreBridge()
            >>> features = bridge.compute_eigenvalue_features_gpu(
            ...     points, neighbors_indices
            ... )
            >>> print(features.keys())
            dict_keys(['linearity', 'planarity', 'sphericity', ...])

        Performance:
            - Eigenvalue computation: GPU-accelerated (~10-20× faster)
            - Feature computation: CPU via core module (maintainable)
            - Overall: ~8-15× faster than pure CPU
        """
        # Step 1: Compute eigenvalues on GPU (FAST)
        eigenvalues = self.compute_eigenvalues_gpu(points, neighbors_indices)

        # Step 2: Use core canonical implementation (MAINTAINABLE)
        from . import compute_eigenvalue_features
        features = compute_eigenvalue_features(
            eigenvalues,
            epsilon=epsilon,
            include_all=include_all
        )

        if self.verbose:
            logger.info(
                f"Computed {len(features)} eigenvalue features using "
                f"GPU bridge + core module"
            )

        return features

    def __repr__(self) -> str:
        """String representation."""
        mode = "GPU" if self.use_gpu else "CPU"
        return (
            f"GPUCoreBridge(mode={mode}, batch_size={self.batch_size:,}, "
            f"verbose={self.verbose})"
        )


# Convenience function for direct usage
def compute_eigenvalue_features_gpu(
    points: np.ndarray,
    neighbors_indices: np.ndarray,
    epsilon: float = 1e-10,
    include_all: bool = True,
    use_gpu: bool = True,
    batch_size: int = 500_000
) -> Dict[str, np.ndarray]:
    """
    Convenience function to compute eigenvalue features using GPU + core.

    This is a shortcut for creating a GPUCoreBridge and calling
    compute_eigenvalue_features_gpu().

    Args:
        points: [N, 3] point cloud coordinates
        neighbors_indices: [N, k] indices of k-nearest neighbors
        epsilon: Small value to prevent division by zero
        include_all: If True, compute all features
        use_gpu: Enable GPU acceleration if available
        batch_size: Maximum batch size for eigenvalue computation

    Returns:
        features: Dictionary with eigenvalue-based features

    Example:
        >>> from ign_lidar.features.core.gpu_bridge import compute_eigenvalue_features_gpu
        >>> features = compute_eigenvalue_features_gpu(points, neighbors)
    """
    bridge = GPUCoreBridge(use_gpu=use_gpu, batch_size=batch_size)
    return bridge.compute_eigenvalue_features_gpu(
        points, neighbors_indices, epsilon, include_all
    )


__all__ = [
    'GPUCoreBridge',
    'compute_eigenvalue_features_gpu',
]
```

---

### Step 2: Update Core Module Exports

**File:** `ign_lidar/features/core/__init__.py`

Add to the imports section:

```python
# GPU Bridge (Phase 1 refactoring)
try:
    from .gpu_bridge import (
        GPUCoreBridge,
        compute_eigenvalue_features_gpu,
    )
    GPU_BRIDGE_AVAILABLE = True
except ImportError:
    GPU_BRIDGE_AVAILABLE = False
    GPUCoreBridge = None
    compute_eigenvalue_features_gpu = None
```

Add to `__all__`:

```python
__all__ = [
    # ... existing exports ...

    # GPU Bridge
    'GPUCoreBridge',
    'compute_eigenvalue_features_gpu',
    'GPU_BRIDGE_AVAILABLE',
]
```

---

### Step 3: Create Unit Tests

**File:** `tests/test_gpu_bridge.py`

```python
"""
Unit tests for GPU-Core Bridge module.

Tests cover:
- GPU vs CPU consistency
- Eigenvalue computation correctness
- Batching for large datasets
- Error handling
- Memory management
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import logging

# Try to import GPU bridge
try:
    from ign_lidar.features.core.gpu_bridge import (
        GPUCoreBridge,
        compute_eigenvalue_features_gpu,
    )
    from ign_lidar.features.core import compute_eigenvalue_features
    GPU_BRIDGE_AVAILABLE = True
except ImportError:
    GPU_BRIDGE_AVAILABLE = False

# Try to import CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


# Test fixtures
@pytest.fixture
def small_point_cloud():
    """Small point cloud for quick tests (1000 points)."""
    np.random.seed(42)
    return np.random.rand(1000, 3).astype(np.float32) * 10


@pytest.fixture
def medium_point_cloud():
    """Medium point cloud for standard tests (100K points)."""
    np.random.seed(42)
    return np.random.rand(100_000, 3).astype(np.float32) * 10


@pytest.fixture
def large_point_cloud():
    """Large point cloud for batching tests (1M points)."""
    np.random.seed(42)
    return np.random.rand(1_000_000, 3).astype(np.float32) * 10


@pytest.fixture
def neighbor_indices_small():
    """Neighbor indices for small point cloud (k=20)."""
    np.random.seed(42)
    n_points = 1000
    k = 20
    # Generate random neighbor indices (for testing, not real KNN)
    indices = np.random.randint(0, n_points, size=(n_points, k))
    return indices


@pytest.fixture
def neighbor_indices_medium():
    """Neighbor indices for medium point cloud (k=20)."""
    np.random.seed(42)
    n_points = 100_000
    k = 20
    indices = np.random.randint(0, n_points, size=(n_points, k))
    return indices


# Tests
@pytest.mark.skipif(not GPU_BRIDGE_AVAILABLE, reason="GPU bridge not available")
class TestGPUCoreBridge:
    """Test suite for GPUCoreBridge class."""

    def test_initialization_gpu_available(self):
        """Test bridge initialization when GPU is available."""
        bridge = GPUCoreBridge(use_gpu=True)

        if GPU_AVAILABLE:
            assert bridge.use_gpu is True
            assert bridge._gpu_available is True
        else:
            assert bridge.use_gpu is False
            assert bridge._gpu_available is False

    def test_initialization_cpu_mode(self):
        """Test bridge initialization in CPU mode."""
        bridge = GPUCoreBridge(use_gpu=False)

        assert bridge.use_gpu is False
        assert bridge.batch_size == 500_000

    def test_eigenvalues_cpu_fallback(self, small_point_cloud, neighbor_indices_small):
        """Test eigenvalue computation with CPU fallback."""
        bridge = GPUCoreBridge(use_gpu=False)

        eigenvalues = bridge.compute_eigenvalues_gpu(
            small_point_cloud,
            neighbor_indices_small
        )

        # Check shape and type
        assert eigenvalues.shape == (len(small_point_cloud), 3)
        assert eigenvalues.dtype == np.float32

        # Check values are non-negative
        assert np.all(eigenvalues >= 0)

        # Check descending order
        assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] >= eigenvalues[:, 2])

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_eigenvalues_gpu_mode(self, small_point_cloud, neighbor_indices_small):
        """Test eigenvalue computation with GPU."""
        bridge = GPUCoreBridge(use_gpu=True)

        eigenvalues = bridge.compute_eigenvalues_gpu(
            small_point_cloud,
            neighbor_indices_small
        )

        # Check shape and type
        assert eigenvalues.shape == (len(small_point_cloud), 3)
        assert eigenvalues.dtype == np.float32

        # Check values are non-negative
        assert np.all(eigenvalues >= 0)

        # Check descending order
        assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])
        assert np.all(eigenvalues[:, 1] >= eigenvalues[:, 2])

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_cpu_consistency(self, medium_point_cloud, neighbor_indices_medium):
        """Test that GPU and CPU produce consistent results."""
        # GPU computation
        bridge_gpu = GPUCoreBridge(use_gpu=True)
        eigenvalues_gpu = bridge_gpu.compute_eigenvalues_gpu(
            medium_point_cloud,
            neighbor_indices_medium
        )

        # CPU computation
        bridge_cpu = GPUCoreBridge(use_gpu=False)
        eigenvalues_cpu = bridge_cpu.compute_eigenvalues_gpu(
            medium_point_cloud,
            neighbor_indices_medium
        )

        # Should match within floating-point tolerance
        assert_allclose(
            eigenvalues_gpu,
            eigenvalues_cpu,
            rtol=1e-5,
            atol=1e-7,
            err_msg="GPU and CPU eigenvalues differ"
        )

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_batching_large_dataset(self, large_point_cloud):
        """Test automatic batching for large datasets."""
        # Generate neighbor indices for large dataset
        np.random.seed(42)
        n_points = len(large_point_cloud)
        k = 20
        neighbor_indices = np.random.randint(0, n_points, size=(n_points, k))

        # Use small batch size to force batching
        bridge = GPUCoreBridge(use_gpu=True, batch_size=100_000, verbose=True)

        eigenvalues = bridge.compute_eigenvalues_gpu(
            large_point_cloud,
            neighbor_indices
        )

        # Check results
        assert eigenvalues.shape == (n_points, 3)
        assert np.all(eigenvalues >= 0)
        assert np.all(eigenvalues[:, 0] >= eigenvalues[:, 1])

    def test_eigenvalue_features_integration(
        self, small_point_cloud, neighbor_indices_small
    ):
        """Test integration with core eigenvalue features."""
        bridge = GPUCoreBridge(use_gpu=GPU_AVAILABLE)

        features = bridge.compute_eigenvalue_features_gpu(
            small_point_cloud,
            neighbor_indices_small,
            epsilon=1e-10,
            include_all=True
        )

        # Check expected features exist
        expected_features = [
            'linearity', 'planarity', 'sphericity', 'anisotropy',
            'eigenentropy', 'omnivariance', 'sum_eigenvalues'
        ]

        for feat in expected_features:
            assert feat in features, f"Feature '{feat}' missing"
            assert features[feat].shape == (len(small_point_cloud),)
            assert features[feat].dtype == np.float32

    def test_convenience_function(self, small_point_cloud, neighbor_indices_small):
        """Test convenience function."""
        features = compute_eigenvalue_features_gpu(
            small_point_cloud,
            neighbor_indices_small,
            use_gpu=GPU_AVAILABLE
        )

        # Should return same structure as bridge method
        assert isinstance(features, dict)
        assert 'linearity' in features
        assert 'planarity' in features


@pytest.mark.skipif(not GPU_BRIDGE_AVAILABLE, reason="GPU bridge not available")
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_input_shape(self):
        """Test handling of invalid input shapes."""
        bridge = GPUCoreBridge(use_gpu=False)

        # Wrong number of dimensions
        points = np.random.rand(100, 2)  # Should be (N, 3)
        neighbors = np.random.randint(0, 100, size=(100, 20))

        with pytest.raises((ValueError, IndexError)):
            bridge.compute_eigenvalues_gpu(points, neighbors)

    def test_empty_input(self):
        """Test handling of empty inputs."""
        bridge = GPUCoreBridge(use_gpu=False)

        points = np.empty((0, 3), dtype=np.float32)
        neighbors = np.empty((0, 20), dtype=np.int32)

        # Should handle gracefully or raise appropriate error
        try:
            eigenvalues = bridge.compute_eigenvalues_gpu(points, neighbors)
            assert eigenvalues.shape == (0, 3)
        except ValueError:
            # Also acceptable to raise error for empty input
            pass


# Performance benchmarks (optional, for validation)
@pytest.mark.benchmark
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestPerformance:
    """Performance benchmarks (run with pytest -m benchmark)."""

    def test_benchmark_eigenvalues_gpu_vs_cpu(
        self, medium_point_cloud, neighbor_indices_medium, benchmark
    ):
        """Benchmark GPU vs CPU eigenvalue computation."""
        bridge_gpu = GPUCoreBridge(use_gpu=True)
        bridge_cpu = GPUCoreBridge(use_gpu=False)

        # Warmup
        _ = bridge_gpu.compute_eigenvalues_gpu(
            medium_point_cloud[:1000], neighbor_indices_medium[:1000]
        )

        # Benchmark GPU
        result_gpu = benchmark(
            bridge_gpu.compute_eigenvalues_gpu,
            medium_point_cloud,
            neighbor_indices_medium
        )

        # Compare with CPU
        import time
        start = time.time()
        result_cpu = bridge_cpu.compute_eigenvalues_gpu(
            medium_point_cloud,
            neighbor_indices_medium
        )
        cpu_time = time.time() - start

        logger.info(f"CPU time: {cpu_time:.3f}s")
        logger.info(f"GPU should be faster than CPU")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

### Step 4: Create Test Data Fixtures

**File:** `tests/fixtures/test_data.py`

```python
"""
Test data generators for GPU bridge testing.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def generate_test_point_cloud(
    n_points: int = 10000,
    seed: int = 42,
    pattern: str = 'random'
) -> np.ndarray:
    """
    Generate test point cloud with various patterns.

    Args:
        n_points: Number of points
        seed: Random seed for reproducibility
        pattern: Pattern type ('random', 'planar', 'linear', 'clustered')

    Returns:
        points: [n_points, 3] point cloud
    """
    np.random.seed(seed)

    if pattern == 'random':
        # Uniform random points
        points = np.random.rand(n_points, 3).astype(np.float32) * 10

    elif pattern == 'planar':
        # Points on a plane with noise
        x = np.random.rand(n_points) * 10
        y = np.random.rand(n_points) * 10
        z = np.ones(n_points) * 5 + np.random.randn(n_points) * 0.1
        points = np.column_stack([x, y, z]).astype(np.float32)

    elif pattern == 'linear':
        # Points along a line with noise
        t = np.random.rand(n_points) * 10
        x = t + np.random.randn(n_points) * 0.1
        y = t + np.random.randn(n_points) * 0.1
        z = t + np.random.randn(n_points) * 0.1
        points = np.column_stack([x, y, z]).astype(np.float32)

    elif pattern == 'clustered':
        # Multiple clusters
        n_clusters = 5
        cluster_size = n_points // n_clusters
        points_list = []

        for i in range(n_clusters):
            center = np.random.rand(3) * 10
            cluster = np.random.randn(cluster_size, 3) * 0.5 + center
            points_list.append(cluster)

        points = np.vstack(points_list).astype(np.float32)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return points


def generate_neighbor_indices(
    points: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """
    Generate real k-nearest neighbor indices using sklearn.

    Args:
        points: [N, 3] point cloud
        k: Number of neighbors

    Returns:
        indices: [N, k] neighbor indices
    """
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    knn.fit(points)
    _, indices = knn.kneighbors(points)
    return indices.astype(np.int32)


def generate_test_dataset(
    n_points: int = 10000,
    k: int = 20,
    pattern: str = 'random',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete test dataset (points + neighbor indices).

    Args:
        n_points: Number of points
        k: Number of neighbors
        pattern: Point cloud pattern
        seed: Random seed

    Returns:
        points: [n_points, 3] point cloud
        neighbor_indices: [n_points, k] neighbor indices
    """
    points = generate_test_point_cloud(n_points, seed, pattern)
    neighbor_indices = generate_neighbor_indices(points, k)
    return points, neighbor_indices
```

---

### Step 5: Create Performance Benchmark Script

**File:** `scripts/benchmark_gpu_bridge.py`

```python
"""
Performance benchmark for GPU Bridge.

Compares GPU bridge performance against:
1. Pure CPU implementation
2. Current GPU chunked implementation (baseline)

Usage:
    python scripts/benchmark_gpu_bridge.py
    python scripts/benchmark_gpu_bridge.py --sizes 10000 100000 1000000
    python scripts/benchmark_gpu_bridge.py --detailed
"""

import argparse
import time
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ign_lidar.features.core.gpu_bridge import GPUCoreBridge
    from tests.fixtures.test_data import generate_test_dataset
    GPU_BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"Error importing GPU bridge: {e}")
    GPU_BRIDGE_AVAILABLE = False
    sys.exit(1)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: CuPy not available, GPU benchmarks will be skipped")


def benchmark_eigenvalues(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    n_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark eigenvalue computation (GPU vs CPU).

    Returns:
        Dict with timing results
    """
    results = {}

    # CPU benchmark
    bridge_cpu = GPUCoreBridge(use_gpu=False)

    # Warmup
    _ = bridge_cpu.compute_eigenvalues_gpu(points[:100], neighbor_indices[:100])

    # Timed runs
    times_cpu = []
    for _ in range(n_runs):
        start = time.time()
        _ = bridge_cpu.compute_eigenvalues_gpu(points, neighbor_indices)
        times_cpu.append(time.time() - start)

    results['cpu_mean'] = np.mean(times_cpu)
    results['cpu_std'] = np.std(times_cpu)

    # GPU benchmark
    if GPU_AVAILABLE:
        bridge_gpu = GPUCoreBridge(use_gpu=True)

        # Warmup
        _ = bridge_gpu.compute_eigenvalues_gpu(points[:100], neighbor_indices[:100])

        # Timed runs
        times_gpu = []
        for _ in range(n_runs):
            start = time.time()
            _ = bridge_gpu.compute_eigenvalues_gpu(points, neighbor_indices)
            times_gpu.append(time.time() - start)

        results['gpu_mean'] = np.mean(times_gpu)
        results['gpu_std'] = np.std(times_gpu)
        results['speedup'] = results['cpu_mean'] / results['gpu_mean']

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPU Bridge')
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[10_000, 100_000, 500_000],
        help='Point cloud sizes to test'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=20,
        help='Number of neighbors'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of runs per benchmark'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed results'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GPU Bridge Performance Benchmark")
    print("=" * 70)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"K-Neighbors: {args.k}")
    print(f"Runs per test: {args.runs}")
    print()

    # Run benchmarks
    for n_points in args.sizes:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {n_points:,} points, k={args.k}")
        print(f"{'=' * 70}")

        # Generate test data
        print("Generating test data...", end=' ')
        points, neighbor_indices = generate_test_dataset(
            n_points=n_points,
            k=args.k,
            pattern='random'
        )
        print("done")

        # Benchmark
        print("Running benchmarks...")
        results = benchmark_eigenvalues(points, neighbor_indices, args.runs)

        # Display results
        print(f"\nResults:")
        print(f"  CPU Time: {results['cpu_mean']:.3f}s ± {results['cpu_std']:.3f}s")

        if 'gpu_mean' in results:
            print(f"  GPU Time: {results['gpu_mean']:.3f}s ± {results['gpu_std']:.3f}s")
            print(f"  Speedup: {results['speedup']:.2f}×")

            # Check if speedup meets target (8-15×)
            if results['speedup'] >= 8.0:
                print(f"  ✅ Performance target met (>= 8×)")
            else:
                print(f"  ⚠️  Performance below target (< 8×)")
        else:
            print(f"  GPU: Skipped (not available)")

    print(f"\n{'=' * 70}")
    print("Benchmark complete")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
```

---

## ✅ Validation Checklist

Before proceeding to Phase 2, ensure:

- [ ] **GPU Bridge Module**

  - [ ] File created: `ign_lidar/features/core/gpu_bridge.py`
  - [ ] GPUCoreBridge class implemented
  - [ ] All methods documented with docstrings
  - [ ] GPU and CPU paths implemented

- [ ] **Tests**

  - [ ] File created: `tests/test_gpu_bridge.py`
  - [ ] All unit tests pass
  - [ ] GPU vs CPU consistency verified
  - [ ] Batching tested with large datasets

- [ ] **Performance**

  - [ ] Benchmark script runs successfully
  - [ ] GPU speedup >= 8× for large datasets
  - [ ] Memory usage acceptable

- [ ] **Integration**

  - [ ] Core module exports updated
  - [ ] No import errors
  - [ ] Works with existing code

- [ ] **Documentation**
  - [ ] All functions have docstrings
  - [ ] Examples included
  - [ ] Performance notes documented

---

## 🚀 Next Steps

Once Phase 1 is complete:

1. **Code Review** - Review and approve GPU bridge implementation
2. **Merge** - Merge feature branch to main
3. **Phase 2** - Begin integration with `features_gpu_chunked.py`

---

**Implementation Time Estimate:** 3-5 days  
**Testing Time Estimate:** 1-2 days  
**Total Phase 1 Duration:** ~1 week
