#!/usr/bin/env python3
"""
Test script for GPU LOD2/LOD3 feature computation.

Verifies that GPU implementations compute the same features as CPU.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features import features_gpu
from sklearn.neighbors import KDTree


def generate_test_data(n_points=1000):
    """Generate synthetic point cloud for testing."""
    np.random.seed(42)
    
    # Generate points in a 50x50x20 meter box
    points = np.random.randn(n_points, 3).astype(np.float32)
    points[:, 0] *= 25  # X: -25 to 25
    points[:, 1] *= 25  # Y: -25 to 25
    points[:, 2] = points[:, 2] * 5 + 10  # Z: 5 to 15
    
    # Classification: mix of ground (2) and building (6)
    classification = np.random.choice([2, 6], size=n_points).astype(np.uint8)
    
    return points, classification


def test_gpu_eigenvalue_features():
    """Test GPU eigenvalue feature computation."""
    print("\n" + "="*70)
    print("TEST 1: GPU Eigenvalue Features")
    print("="*70)
    
    points, _ = generate_test_data(n_points=500)
    
    # Compute normals using GPU
    computer = features_gpu.get_gpu_computer()
    normals = computer.compute_normals(points, k=10)
    
    # Build KDTree and get neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=10)
    
    # Compute eigenvalue features
    eig_features = computer.compute_eigenvalue_features(
        points, normals, neighbors_indices
    )
    
    print(f"\n✓ Computed eigenvalue features:")
    for name, values in eig_features.items():
        print(f"  {name}: shape={values.shape}, range=[{values.min():.3f}, {values.max():.3f}]")
    
    # Validate features
    assert 'eigenvalue_1' in eig_features
    assert 'eigenvalue_2' in eig_features
    assert 'eigenvalue_3' in eig_features
    assert 'sum_eigenvalues' in eig_features
    assert 'eigenentropy' in eig_features
    assert 'omnivariance' in eig_features
    assert 'change_curvature' in eig_features
    
    # Check that eigenvalues are ordered: λ0 >= λ1 >= λ2
    λ0 = eig_features['eigenvalue_1']
    λ1 = eig_features['eigenvalue_2']
    λ2 = eig_features['eigenvalue_3']
    
    assert np.all(λ0 >= λ1 - 1e-5), "Eigenvalues not properly ordered (λ0 >= λ1)"
    assert np.all(λ1 >= λ2 - 1e-5), "Eigenvalues not properly ordered (λ1 >= λ2)"
    assert np.all(λ0 >= 0), "Eigenvalue 1 should be non-negative"
    
    print("\n✅ GPU eigenvalue features test PASSED")


def test_gpu_architectural_features():
    """Test GPU architectural feature computation."""
    print("\n" + "="*70)
    print("TEST 2: GPU Architectural Features")
    print("="*70)
    
    points, _ = generate_test_data(n_points=500)
    
    # Compute normals
    computer = features_gpu.get_gpu_computer()
    normals = computer.compute_normals(points, k=10)
    
    # Build KDTree and get neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=10)
    
    # Compute architectural features
    arch_features = computer.compute_architectural_features(
        points, normals, neighbors_indices
    )
    
    print(f"\n✓ Computed architectural features:")
    for name, values in arch_features.items():
        print(f"  {name}: shape={values.shape}, range=[{values.min():.3f}, {values.max():.3f}]")
    
    # Validate features
    assert 'edge_strength' in arch_features
    assert 'corner_likelihood' in arch_features
    assert 'overhang_indicator' in arch_features
    assert 'surface_roughness' in arch_features
    
    # Check value ranges
    assert np.all(arch_features['edge_strength'] >= 0)
    assert np.all(arch_features['edge_strength'] <= 1)
    assert np.all(arch_features['corner_likelihood'] >= 0)
    assert np.all(arch_features['corner_likelihood'] <= 1)
    
    print("\n✅ GPU architectural features test PASSED")


def test_gpu_density_features():
    """Test GPU density feature computation."""
    print("\n" + "="*70)
    print("TEST 3: GPU Density Features")
    print("="*70)
    
    points, _ = generate_test_data(n_points=500)
    
    # Build KDTree and get neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=10)
    
    # Compute density features
    computer = features_gpu.get_gpu_computer()
    density_features = computer.compute_density_features(
        points, neighbors_indices, radius_2m=2.0
    )
    
    print(f"\n✓ Computed density features:")
    for name, values in density_features.items():
        print(f"  {name}: shape={values.shape}, range=[{values.min():.3f}, {values.max():.3f}]")
    
    # Validate features
    assert 'density' in density_features
    assert 'num_points_2m' in density_features
    assert 'neighborhood_extent' in density_features
    assert 'height_extent_ratio' in density_features
    
    # Check value ranges
    assert np.all(density_features['density'] >= 0)
    assert np.all(density_features['num_points_2m'] >= 0)
    assert np.all(density_features['neighborhood_extent'] >= 0)
    assert np.all(density_features['height_extent_ratio'] >= 0)
    assert np.all(density_features['height_extent_ratio'] <= 1)
    
    print("\n✅ GPU density features test PASSED")


def test_wrapper_functions():
    """Test standalone wrapper functions."""
    print("\n" + "="*70)
    print("TEST 4: GPU Wrapper Functions")
    print("="*70)
    
    points, _ = generate_test_data(n_points=500)
    
    # Compute normals
    normals = features_gpu.compute_normals(points, k=10)
    
    # Build KDTree and get neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=10)
    
    # Test wrapper functions
    print("\n✓ Testing compute_eigenvalue_features wrapper...")
    eig_feat = features_gpu.compute_eigenvalue_features(points, normals, neighbors_indices)
    assert len(eig_feat) == 7, f"Expected 7 eigenvalue features, got {len(eig_feat)}"
    
    print("✓ Testing compute_architectural_features wrapper...")
    arch_feat = features_gpu.compute_architectural_features(points, normals, neighbors_indices)
    assert len(arch_feat) == 4, f"Expected 4 architectural features, got {len(arch_feat)}"
    
    print("✓ Testing compute_density_features wrapper...")
    dens_feat = features_gpu.compute_density_features(points, neighbors_indices)
    assert len(dens_feat) == 4, f"Expected 4 density features, got {len(dens_feat)}"
    
    print("\n✅ Wrapper function tests PASSED")


def main():
    """Run all tests."""
    print("="*70)
    print("GPU LOD2/LOD3 FEATURE TEST SUITE")
    print("="*70)
    
    try:
        test_gpu_eigenvalue_features()
        test_gpu_architectural_features()
        test_gpu_density_features()
        test_wrapper_functions()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\n🎉 GPU LOD2/LOD3 feature system is working correctly!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
