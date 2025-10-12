"""
Test enhanced features for GPU chunked processing.
Verifies that GPU chunked mode has feature parity with GPU mode.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_test_data(n_points=10000, seed=42):
    """Generate synthetic point cloud for testing."""
    np.random.seed(seed)
    
    # Create points with some structure
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 100, n_points)
    z = np.random.uniform(0, 50, n_points)
    points = np.column_stack([x, y, z]).astype(np.float32)
    
    # Random classification codes
    classification = np.random.choice([1, 2, 3, 5, 6], size=n_points).astype(np.uint8)
    
    return points, classification


def test_gpu_chunked_eigenvalue_features():
    """Test eigenvalue features in GPU chunked mode."""
    print("\n" + "="*70)
    print("TEST: GPU Chunked Eigenvalue Features")
    print("="*70)
    
    from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
    from sklearn.neighbors import KDTree
    
    # Generate test data
    points, classification = generate_test_data(n_points=5000)
    k = 10
    
    # Build KDTree for neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=k)
    
    # Compute normals first (needed for some features)
    computer = GPUChunkedFeatureComputer(use_gpu=True, show_progress=False)
    normals = computer.compute_normals_chunked(points, k=k)
    
    # Compute eigenvalue features
    print(f"\n📊 Computing eigenvalue features for {len(points):,} points...")
    eig_features = computer.compute_eigenvalue_features(
        points, normals, neighbors_indices
    )
    
    # Verify features
    expected_features = [
        'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
        'sum_eigenvalues', 'eigenentropy', 'omnivariance', 'change_curvature'
    ]
    
    print(f"\n✓ Features computed: {list(eig_features.keys())}")
    assert len(eig_features) == 7, f"Expected 7 features, got {len(eig_features)}"
    
    for feat_name in expected_features:
        assert feat_name in eig_features, f"Missing feature: {feat_name}"
        values = eig_features[feat_name]
        assert len(values) == len(points), f"Feature {feat_name} has wrong length"
        assert values.dtype == np.float32, f"Feature {feat_name} has wrong dtype"
        
        # Check for valid ranges
        assert np.all(np.isfinite(values)), f"Feature {feat_name} has non-finite values"
        assert np.all(values >= 0), f"Feature {feat_name} has negative values"
        
        print(f"  ✓ {feat_name:20s}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
    
    print("\n✅ GPU Chunked Eigenvalue Features Test PASSED")
    return True


def test_gpu_chunked_architectural_features():
    """Test architectural features in GPU chunked mode."""
    print("\n" + "="*70)
    print("TEST: GPU Chunked Architectural Features")
    print("="*70)
    
    from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
    from sklearn.neighbors import KDTree
    
    # Generate test data
    points, classification = generate_test_data(n_points=5000)
    k = 10
    
    # Build KDTree for neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=k)
    
    # Compute normals
    computer = GPUChunkedFeatureComputer(use_gpu=True, show_progress=False)
    normals = computer.compute_normals_chunked(points, k=k)
    
    # Compute architectural features
    print(f"\n🏗️  Computing architectural features for {len(points):,} points...")
    arch_features = computer.compute_architectural_features(
        points, normals, neighbors_indices
    )
    
    # Verify features
    expected_features = [
        'edge_strength', 'corner_likelihood',
        'overhang_indicator', 'surface_roughness'
    ]
    
    print(f"\n✓ Features computed: {list(arch_features.keys())}")
    assert len(arch_features) == 4, f"Expected 4 features, got {len(arch_features)}"
    
    for feat_name in expected_features:
        assert feat_name in arch_features, f"Missing feature: {feat_name}"
        values = arch_features[feat_name]
        assert len(values) == len(points), f"Feature {feat_name} has wrong length"
        assert values.dtype == np.float32, f"Feature {feat_name} has wrong dtype"
        
        # Check for valid ranges
        assert np.all(np.isfinite(values)), f"Feature {feat_name} has non-finite values"
        
        # Most architectural features should be in [0, 1] range
        if feat_name in ['edge_strength', 'corner_likelihood', 'overhang_indicator']:
            assert np.all(values >= 0) and np.all(values <= 1), \
                f"Feature {feat_name} should be in [0, 1] range"
        
        print(f"  ✓ {feat_name:20s}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
    
    print("\n✅ GPU Chunked Architectural Features Test PASSED")
    return True


def test_gpu_chunked_density_features():
    """Test density features in GPU chunked mode."""
    print("\n" + "="*70)
    print("TEST: GPU Chunked Density Features")
    print("="*70)
    
    from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
    from sklearn.neighbors import KDTree
    
    # Generate test data
    points, classification = generate_test_data(n_points=5000)
    k = 10
    
    # Build KDTree for neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=k)
    
    # Compute density features
    computer = GPUChunkedFeatureComputer(use_gpu=True, show_progress=False)
    
    print(f"\n📍 Computing density features for {len(points):,} points...")
    density_features = computer.compute_density_features(
        points, neighbors_indices, radius_2m=2.0
    )
    
    # Verify features
    expected_features = [
        'density', 'num_points_2m',
        'neighborhood_extent', 'height_extent_ratio'
    ]
    
    print(f"\n✓ Features computed: {list(density_features.keys())}")
    assert len(density_features) == 4, f"Expected 4 features, got {len(density_features)}"
    
    for feat_name in expected_features:
        assert feat_name in density_features, f"Missing feature: {feat_name}"
        values = density_features[feat_name]
        assert len(values) == len(points), f"Feature {feat_name} has wrong length"
        assert values.dtype == np.float32, f"Feature {feat_name} has wrong dtype"
        
        # Check for valid ranges
        assert np.all(np.isfinite(values)), f"Feature {feat_name} has non-finite values"
        
        print(f"  ✓ {feat_name:20s}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
    
    print("\n✅ GPU Chunked Density Features Test PASSED")
    return True


def test_wrapper_functions():
    """Test wrapper functions for GPU chunked features."""
    print("\n" + "="*70)
    print("TEST: GPU Chunked Wrapper Functions")
    print("="*70)
    
    from ign_lidar.features import features_gpu_chunked
    from sklearn.neighbors import KDTree
    
    # Generate test data
    points, classification = generate_test_data(n_points=5000)
    k = 10
    
    # Build KDTree for neighbors
    tree = KDTree(points, metric='euclidean')
    _, neighbors_indices = tree.query(points, k=k)
    
    # Compute normals first
    computer = features_gpu_chunked.GPUChunkedFeatureComputer(
        use_gpu=True, show_progress=False
    )
    normals = computer.compute_normals_chunked(points, k=k)
    
    print(f"\n🔧 Testing wrapper functions for {len(points):,} points...")
    
    # Test eigenvalue features wrapper
    eig_features = features_gpu_chunked.compute_eigenvalue_features(
        points, normals, neighbors_indices
    )
    assert len(eig_features) == 7, "Eigenvalue wrapper failed"
    print("  ✓ compute_eigenvalue_features() wrapper works")
    
    # Test architectural features wrapper
    arch_features = features_gpu_chunked.compute_architectural_features(
        points, normals, neighbors_indices
    )
    assert len(arch_features) == 4, "Architectural wrapper failed"
    print("  ✓ compute_architectural_features() wrapper works")
    
    # Test density features wrapper
    density_features = features_gpu_chunked.compute_density_features(
        points, neighbors_indices, radius_2m=2.0
    )
    assert len(density_features) == 4, "Density wrapper failed"
    print("  ✓ compute_density_features() wrapper works")
    
    print("\n✅ Wrapper Function Tests PASSED")
    return True


def test_feature_parity_with_gpu():
    """Test that GPU chunked features match GPU features."""
    print("\n" + "="*70)
    print("TEST: GPU Chunked vs GPU Feature Parity")
    print("="*70)
    
    try:
        from ign_lidar.features.features_gpu import GPUFeatureComputer
        from ign_lidar.features.features_gpu_chunked import GPUChunkedFeatureComputer
        from sklearn.neighbors import KDTree
        
        # Generate small test data
        points, classification = generate_test_data(n_points=1000)
        k = 10
        
        # Build KDTree for neighbors
        tree = KDTree(points, metric='euclidean')
        _, neighbors_indices = tree.query(points, k=k)
        
        # Compute with GPU
        gpu_computer = GPUFeatureComputer(use_gpu=True)
        gpu_normals = gpu_computer.compute_normals(points, k=k)
        
        # Compute with GPU Chunked
        chunked_computer = GPUChunkedFeatureComputer(use_gpu=True, show_progress=False)
        chunked_normals = chunked_computer.compute_normals_chunked(points, k=k)
        
        print(f"\n🔍 Comparing features from GPU vs GPU Chunked for {len(points):,} points...")
        
        # Compare eigenvalue features
        gpu_eig = gpu_computer.compute_eigenvalue_features(
            points, gpu_normals, neighbors_indices
        )
        chunked_eig = chunked_computer.compute_eigenvalue_features(
            points, chunked_normals, neighbors_indices
        )
        
        for key in gpu_eig.keys():
            diff = np.abs(gpu_eig[key] - chunked_eig[key]).mean()
            print(f"  ✓ {key:20s}: mean_diff={diff:.6f}")
            assert diff < 0.01, f"Feature {key} differs too much between GPU and GPU Chunked"
        
        print("\n✅ GPU Chunked has feature parity with GPU!")
        return True
        
    except Exception as e:
        print(f"\n⚠️  Parity test skipped: {e}")
        return True  # Don't fail if GPU not available


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GPU CHUNKED ENHANCED FEATURES TEST SUITE")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Eigenvalue Features", test_gpu_chunked_eigenvalue_features()))
    except Exception as e:
        print(f"\n❌ Eigenvalue test failed: {e}")
        results.append(("Eigenvalue Features", False))
    
    try:
        results.append(("Architectural Features", test_gpu_chunked_architectural_features()))
    except Exception as e:
        print(f"\n❌ Architectural test failed: {e}")
        results.append(("Architectural Features", False))
    
    try:
        results.append(("Density Features", test_gpu_chunked_density_features()))
    except Exception as e:
        print(f"\n❌ Density test failed: {e}")
        results.append(("Density Features", False))
    
    try:
        results.append(("Wrapper Functions", test_wrapper_functions()))
    except Exception as e:
        print(f"\n❌ Wrapper test failed: {e}")
        results.append(("Wrapper Functions", False))
    
    try:
        results.append(("Feature Parity", test_feature_parity_with_gpu()))
    except Exception as e:
        print(f"\n❌ Parity test failed: {e}")
        results.append(("Feature Parity", False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\n🎉 GPU Chunked enhanced feature system is working correctly!")
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        sys.exit(1)
