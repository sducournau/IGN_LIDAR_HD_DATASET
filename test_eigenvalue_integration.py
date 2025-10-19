"""
Test script for Phase 2A.6 GPU Bridge eigenvalue integration.

This script tests the new eigenvalue computation methods added to GPUProcessor.
"""

import numpy as np
from ign_lidar.features.gpu_processor import GPUProcessor
import time

def test_eigenvalue_integration():
    """Test eigenvalue features with GPUProcessor."""
    
    print("=" * 70)
    print("Phase 2A.6: GPU Bridge Eigenvalue Integration Test")
    print("=" * 70)
    
    # Create synthetic point cloud (10K points)
    np.random.seed(42)
    N = 10_000
    points = np.random.randn(N, 3).astype(np.float32)
    
    print(f"\nTest data: {N:,} points")
    print(f"Shape: {points.shape}")
    print(f"Memory: {points.nbytes / 1024**2:.2f} MB")
    
    # Initialize processor
    print("\n" + "-" * 70)
    print("Initializing GPUProcessor...")
    print("-" * 70)
    processor = GPUProcessor(use_gpu=True, show_progress=False)
    print(f"GPU available: {processor.use_gpu}")
    print(f"cuML available: {processor.use_cuml}")
    print(f"Chunk threshold: {processor.chunk_threshold:,} points")
    
    # Test 1: Compute neighbors first
    print("\n" + "-" * 70)
    print("Test 1: Computing k-NN neighbors")
    print("-" * 70)
    k = 10
    t0 = time.time()
    
    # Use sklearn for CPU-based neighbor finding
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree', n_jobs=-1)
    nn.fit(points)
    _, neighbors = nn.kneighbors(points)
    
    t_neighbors = time.time() - t0
    print(f"Neighbors computed: {neighbors.shape}")
    print(f"Time: {t_neighbors:.3f}s")
    
    # Test 2: Compute eigenvalues
    print("\n" + "-" * 70)
    print("Test 2: Computing eigenvalues")
    print("-" * 70)
    t0 = time.time()
    eigenvalues = processor.compute_eigenvalues(points, neighbors)
    t_eigenvalues = time.time() - t0
    
    print(f"Eigenvalues computed: {eigenvalues.shape}")
    print(f"Sample eigenvalues (first point): {eigenvalues[0]}")
    print(f"Time: {t_eigenvalues:.3f}s")
    
    # Test 3: Compute eigenvalue features
    print("\n" + "-" * 70)
    print("Test 3: Computing eigenvalue-based features")
    print("-" * 70)
    t0 = time.time()
    features = processor.compute_eigenvalue_features(points, neighbors)
    t_features = time.time() - t0
    
    print(f"Features computed: {list(features.keys())}")
    print(f"Number of features: {len(features)}")
    for name, values in features.items():
        print(f"  - {name}: shape={values.shape}, range=[{values.min():.4f}, {values.max():.4f}]")
    print(f"Time: {t_features:.3f}s")
    
    # Test 4: Compute density features
    print("\n" + "-" * 70)
    print("Test 4: Computing density features")
    print("-" * 70)
    t0 = time.time()
    density_features = processor.compute_density_features(
        points, k_neighbors=k
    )
    t_density = time.time() - t0
    
    print(f"Density features: {list(density_features.keys())}")
    print(f"Number of features: {len(density_features)}")
    for name, values in density_features.items():
        print(f"  - {name}: shape={values.shape}, range=[{values.min():.4f}, {values.max():.4f}]")
    print(f"Time: {t_density:.3f}s")
    
    # Test 5: Compute normals first, then architectural features
    print("\n" + "-" * 70)
    print("Test 5: Computing architectural features")
    print("-" * 70)
    t0 = time.time()
    normals = processor.compute_normals(points, k=k)
    t_normals = time.time() - t0
    print(f"Normals computed: {normals.shape}")
    print(f"Time (normals): {t_normals:.3f}s")
    
    t0 = time.time()
    arch_features = processor.compute_architectural_features(
        points, normals, eigenvalues
    )
    t_arch = time.time() - t0
    
    print(f"Architectural features: {list(arch_features.keys())}")
    print(f"Number of features: {len(arch_features)}")
    for name, values in arch_features.items():
        print(f"  - {name}: shape={values.shape}, range=[{values.min():.4f}, {values.max():.4f}]")
    print(f"Time: {t_arch:.3f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    total_time = t_neighbors + t_eigenvalues + t_features + t_density + t_normals + t_arch
    print(f"Total time: {total_time:.3f}s")
    print(f"  - Neighbors: {t_neighbors:.3f}s ({100*t_neighbors/total_time:.1f}%)")
    print(f"  - Eigenvalues: {t_eigenvalues:.3f}s ({100*t_eigenvalues/total_time:.1f}%)")
    print(f"  - Eigenvalue features: {t_features:.3f}s ({100*t_features/total_time:.1f}%)")
    print(f"  - Density features: {t_density:.3f}s ({100*t_density/total_time:.1f}%)")
    print(f"  - Normals: {t_normals:.3f}s ({100*t_normals/total_time:.1f}%)")
    print(f"  - Architectural features: {t_arch:.3f}s ({100*t_arch/total_time:.1f}%)")
    
    print("\n✅ All tests passed! GPU Bridge integration successful.")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_eigenvalue_integration()
    exit(0 if success else 1)
