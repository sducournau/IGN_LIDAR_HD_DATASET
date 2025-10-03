"""
Test GPU Building-Specific Features

Tests the GPU implementations of:
- compute_verticality()
- compute_wall_score()
- compute_roof_score()
"""

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None


def test_gpu_building_features_import():
    """Test that GPU building features can be imported."""
    try:
        from ign_lidar.features_gpu import (
            compute_verticality,
            compute_wall_score,
            compute_roof_score
        )
        assert compute_verticality is not None
        assert compute_wall_score is not None
        assert compute_roof_score is not None
        print("✓ GPU building features imported successfully")
    except ImportError as e:
        if pytest:
            pytest.skip(f"GPU module not available: {e}")
        else:
            print(f"⚠ GPU module not available: {e}")


def test_compute_verticality():
    """Test verticality computation (CPU fallback)."""
    from ign_lidar.features_gpu import compute_verticality
    
    # Create test normals
    # Horizontal surface (roof): normal pointing up (0, 0, 1)
    # Vertical surface (wall): normal pointing horizontally (1, 0, 0)
    normals = np.array([
        [0, 0, 1],      # Horizontal (roof) - should be ~0
        [0, 0, -1],     # Horizontal (ground) - should be ~0
        [1, 0, 0],      # Vertical (wall) - should be ~1
        [0, 1, 0],      # Vertical (wall) - should be ~1
        [0.707, 0, 0.707],  # 45° slope - should be ~0.3
    ], dtype=np.float32)
    
    verticality = compute_verticality(normals)
    
    assert verticality.shape == (5,)
    assert verticality.dtype == np.float32
    
    # Check expected values
    assert verticality[0] < 0.1, "Horizontal surface should have low verticality"
    assert verticality[1] < 0.1, "Horizontal surface should have low verticality"
    assert verticality[2] > 0.9, "Vertical surface should have high verticality"
    assert verticality[3] > 0.9, "Vertical surface should have high verticality"
    assert 0.2 < verticality[4] < 0.4, "45° slope should have medium verticality"
    
    print("✓ Verticality computation correct")


def test_compute_wall_score():
    """Test wall score computation (CPU fallback)."""
    from ign_lidar.features_gpu import compute_wall_score
    
    # Test data:
    # - Vertical surfaces at different heights
    normals = np.array([
        [1, 0, 0],      # Vertical
        [1, 0, 0],      # Vertical
        [1, 0, 0],      # Vertical
        [0, 0, 1],      # Horizontal (not a wall)
    ], dtype=np.float32)
    
    height = np.array([
        0.5,   # Too low to be a wall
        2.0,   # Good wall height
        10.0,  # High wall
        5.0,   # High but horizontal (roof, not wall)
    ], dtype=np.float32)
    
    wall_score = compute_wall_score(normals, height, min_height=1.5)
    
    assert wall_score.shape == (4,)
    assert wall_score.dtype == np.float32
    
    # Check expected behavior
    msg1 = "Low vertical surface should have low wall score"
    assert wall_score[0] < 0.2, msg1
    msg2 = "Vertical surface at 2m should have wall score >= 0.09"
    assert wall_score[1] >= 0.09, msg2
    msg3 = "Higher wall should have higher score"
    assert wall_score[2] > wall_score[1], msg3
    msg4 = "Horizontal surface should have low wall score"
    assert wall_score[3] < 0.1, msg4
    
    print("✓ Wall score computation correct")


def test_compute_roof_score():
    """Test roof score computation (CPU fallback)."""
    from ign_lidar.features_gpu import compute_roof_score
    
    # Test data:
    # - Horizontal surfaces at different heights with different curvatures
    normals = np.array([
        [0, 0, 1],      # Horizontal
        [0, 0, 1],      # Horizontal
        [0, 0, 1],      # Horizontal
        [1, 0, 0],      # Vertical (wall, not roof)
    ], dtype=np.float32)
    
    height = np.array([
        0.5,   # Too low to be a roof
        5.0,   # Good roof height
        15.0,  # High roof
        10.0,  # High but vertical (wall)
    ], dtype=np.float32)
    
    curvature = np.array([
        0.1,   # Low curvature (planar)
        0.1,   # Low curvature (planar)
        0.8,   # High curvature (curved)
        0.1,   # Low curvature
    ], dtype=np.float32)
    
    roof_score = compute_roof_score(normals, height, curvature, min_height=2.0)
    
    assert roof_score.shape == (4,)
    assert roof_score.dtype == np.float32
    
    # Check expected behavior
    msg1 = "Low horizontal surface should have low roof score"
    assert roof_score[0] < 0.1, msg1
    msg2 = "Horizontal planar surface at 5m should be roof"
    assert roof_score[1] >= 0.25, msg2
    msg3 = "Curved surface should have lower roof score"
    assert roof_score[2] < roof_score[1], msg3
    msg4 = "Vertical surface should have low roof score"
    assert roof_score[3] < 0.1, msg4
    
    print("✓ Roof score computation correct")


def test_gpu_building_features_with_gpu():
    """Test building features with actual GPU (if available)."""
    from ign_lidar.features_gpu import (
        GPU_AVAILABLE,
        GPUFeatureComputer
    )
    
    if not GPU_AVAILABLE:
        pytest.skip("GPU not available")
    
    # Create larger dataset for GPU
    np.random.seed(42)
    N = 10000
    
    # Mix of horizontal and vertical normals
    normals = np.random.randn(N, 3).astype(np.float32)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    height = np.random.uniform(0, 20, N).astype(np.float32)
    curvature = np.random.uniform(0, 1, N).astype(np.float32)
    
    # Initialize GPU computer
    computer = GPUFeatureComputer(use_gpu=True)
    
    # Test verticality
    verticality = computer.compute_verticality(normals)
    assert verticality.shape == (N,)
    assert 0 <= verticality.min() <= 1
    assert 0 <= verticality.max() <= 1
    
    # Test wall score
    wall_score = computer.compute_wall_score(normals, height)
    assert wall_score.shape == (N,)
    assert 0 <= wall_score.min() <= 1
    assert 0 <= wall_score.max() <= 1
    
    # Test roof score
    roof_score = computer.compute_roof_score(normals, height, curvature)
    assert roof_score.shape == (N,)
    assert 0 <= roof_score.min() <= 1
    assert 0 <= roof_score.max() <= 1
    
    print("✓ GPU building features work correctly")


def test_compute_all_features_with_building():
    """Test compute_all_features with building features enabled."""
    from ign_lidar.features_gpu import GPUFeatureComputer
    
    # Create test data
    np.random.seed(42)
    N = 1000
    points = np.random.rand(N, 3).astype(np.float32) * 100
    classification = np.full(N, 6, dtype=np.uint8)  # Building class
    
    # Initialize computer
    computer = GPUFeatureComputer(use_gpu=False)  # Use CPU for testing
    
    # Compute with building features
    normals, curvature, height, geo_features = computer.compute_all_features(
        points,
        classification,
        k=10,
        include_building_features=True
    )
    
    # Check all features are present
    assert normals.shape == (N, 3)
    assert curvature.shape == (N,)
    assert height.shape == (N,)
    
    # Check building features are in geo_features
    assert 'verticality' in geo_features
    assert 'wall_score' in geo_features
    assert 'roof_score' in geo_features
    
    assert geo_features['verticality'].shape == (N,)
    assert geo_features['wall_score'].shape == (N,)
    assert geo_features['roof_score'].shape == (N,)
    
    print("✓ compute_all_features with building features works")


if __name__ == "__main__":
    """Run tests standalone."""
    print("\n" + "="*70)
    print("GPU Building Features Tests")
    print("="*70 + "\n")
    
    try:
        test_gpu_building_features_import()
        test_compute_verticality()
        test_compute_wall_score()
        test_compute_roof_score()
        test_compute_all_features_with_building()
        
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
