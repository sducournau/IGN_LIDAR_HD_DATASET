"""
Tests for GPU integration functionality.

Tests the GPU acceleration feature including:
- GPU module availability
- CPU fallback when GPU not available
- Feature computation with GPU
- Integration with CLI
"""

import pytest
import numpy as np


def test_gpu_wrapper_function_exists():
    """Test that GPU wrapper function can be imported."""
    from ign_lidar.features import compute_all_features_with_gpu
    assert compute_all_features_with_gpu is not None


def test_gpu_wrapper_cpu_fallback():
    """Test GPU wrapper works with CPU fallback."""
    from ign_lidar.features import compute_all_features_with_gpu
    
    # Create sample data
    np.random.seed(42)
    points = np.random.rand(1000, 3).astype(np.float32)
    points[:, 2] *= 10  # Scale Z for more realistic heights
    classification = np.random.randint(1, 6, size=1000, dtype=np.uint8)
    
    # Test with GPU disabled (should always work)
    normals, curvature, height, geo_features = compute_all_features_with_gpu(
        points, classification, k=10, auto_k=False, use_gpu=False
    )
    
    # Verify shapes
    assert normals.shape == (1000, 3), \
        f"Expected (1000, 3), got {normals.shape}"
    assert curvature.shape == (1000,), \
        f"Expected (1000,), got {curvature.shape}"
    assert height.shape == (1000,), \
        f"Expected (1000,), got {height.shape}"
    
    # Verify normals are normalized
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Normals should be normalized"
    
    # Verify geometric features
    expected_features = [
        'planarity', 'linearity', 'sphericity',
        'anisotropy', 'roughness', 'density'
    ]
    for feat in expected_features:
        assert feat in geo_features, f"Missing feature: {feat}"
        assert geo_features[feat].shape == (1000,), \
            f"Feature {feat} has wrong shape: {geo_features[feat].shape}"


def test_gpu_module_availability():
    """Test GPU module import and availability detection."""
    try:
        from ign_lidar.features_gpu import (
            GPUFeatureComputer,
            GPU_AVAILABLE,
            CUML_AVAILABLE
        )
        
        # Module imports successfully
        assert GPUFeatureComputer is not None
        assert isinstance(GPU_AVAILABLE, bool)
        assert isinstance(CUML_AVAILABLE, bool)
        
        if GPU_AVAILABLE:
            print("✓ GPU (CuPy) is available")
        else:
            print("⚠ GPU (CuPy) not available - CPU fallback will be used")
            
        if CUML_AVAILABLE:
            print("✓ RAPIDS cuML is available")
        else:
            print("⚠ RAPIDS cuML not available")
            
    except ImportError as e:
        pytest.skip(f"GPU module not available: {e}")


def test_gpu_feature_computation():
    """Test actual GPU feature computation (only if GPU available)."""
    from ign_lidar.features_gpu import GPUFeatureComputer, GPU_AVAILABLE
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    # Create larger dataset for GPU
    np.random.seed(42)
    points = np.random.rand(10000, 3).astype(np.float32)
    points[:, 2] *= 20  # Scale Z
    classification = np.random.randint(1, 6, size=10000, dtype=np.uint8)
    
    # Initialize GPU computer
    computer = GPUFeatureComputer(use_gpu=True)
    
    # Compute features
    normals = computer.compute_normals(points, k=10)
    curvature = computer.compute_curvature(points, normals, k=10)
    height = computer.compute_height_above_ground(points, classification)
    geo_features = computer.extract_geometric_features(points, normals, k=10)
    
    # Verify results
    assert normals.shape == (10000, 3)
    assert curvature.shape == (10000,)
    assert height.shape == (10000,)
    
    # Check normals are normalized
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    
    # Check features exist
    assert 'planarity' in geo_features
    assert 'linearity' in geo_features
    assert 'density' in geo_features


def test_gpu_wrapper_with_gpu_enabled():
    """Test GPU wrapper with GPU enabled (falls back if not available)."""
    from ign_lidar.features import compute_all_features_with_gpu
    
    # Create sample data
    np.random.seed(42)
    points = np.random.rand(1000, 3).astype(np.float32)
    points[:, 2] *= 10
    classification = np.random.randint(1, 6, size=1000, dtype=np.uint8)
    
    # Test with GPU enabled (will fallback to CPU if GPU not available)
    normals, curvature, height, geo_features = compute_all_features_with_gpu(
        points, classification, k=10, auto_k=False, use_gpu=True
    )
    
    # Verify shapes (should work regardless of GPU availability)
    assert normals.shape == (1000, 3)
    assert curvature.shape == (1000,)
    assert height.shape == (1000,)
    assert len(geo_features) >= 6  # At least 6 geometric features


def test_gpu_consistency_cpu_vs_gpu():
    """Test that GPU and CPU produce consistent results."""
    from ign_lidar.features import compute_all_features_with_gpu
    from ign_lidar.features_gpu import GPU_AVAILABLE
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available for consistency test")
    
    # Create deterministic test data
    np.random.seed(42)
    points = np.random.rand(1000, 3).astype(np.float32)
    points[:, 2] *= 10
    classification = np.random.randint(1, 6, size=1000, dtype=np.uint8)
    
    # Compute with CPU
    normals_cpu, curv_cpu, height_cpu, geo_cpu = \
        compute_all_features_with_gpu(
            points, classification, k=10, auto_k=False, use_gpu=False
        )
    
    # Compute with GPU
    normals_gpu, curv_gpu, height_gpu, geo_gpu = \
        compute_all_features_with_gpu(
            points, classification, k=10, auto_k=False, use_gpu=True
        )
    
    # Compare results (allow some numerical tolerance)
    assert np.allclose(normals_cpu, normals_gpu, atol=1e-3), \
        "Normals differ between CPU and GPU"
    assert np.allclose(curv_cpu, curv_gpu, atol=1e-3), \
        "Curvature differs between CPU and GPU"
    assert np.allclose(height_cpu, height_gpu, atol=1e-3), \
        "Height differs between CPU and GPU"
    
    # Compare geometric features
    for key in geo_cpu.keys():
        if key in geo_gpu:
            assert np.allclose(geo_cpu[key], geo_gpu[key], atol=1e-3), \
                f"Feature {key} differs between CPU and GPU"


def _check_gpu_available():
    """Helper to check if GPU is available."""
    try:
        from ign_lidar.features_gpu import GPU_AVAILABLE
        return GPU_AVAILABLE
    except ImportError:
        return False


if __name__ == "__main__":
    # Run basic tests
    print("Testing GPU integration...")
    
    test_gpu_wrapper_function_exists()
    print("✓ GPU wrapper function exists")
    
    test_gpu_wrapper_cpu_fallback()
    print("✓ CPU fallback works")
    
    test_gpu_module_availability()
    print("✓ GPU module availability checked")
    
    test_gpu_wrapper_with_gpu_enabled()
    print("✓ GPU wrapper with GPU enabled works")
    
    print("\nAll basic tests passed!")
