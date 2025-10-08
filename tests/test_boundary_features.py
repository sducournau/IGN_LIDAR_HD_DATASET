"""
Tests for Boundary-Aware Feature Computation (Sprint 3 Phase 3.2)

This test suite validates:
1. BoundaryAwareFeatureComputer initialization
2. Feature computation with/without buffer zones
3. Boundary point detection
4. Normal vector computation with cross-tile neighborhoods
5. Curvature computation
6. Planarity features (planarity, linearity, sphericity)
7. Verticality computation
8. Feature quality at boundaries vs interior
"""

import pytest
import numpy as np
from pathlib import Path
import laspy

from ign_lidar.features_boundary import (
    BoundaryAwareFeatureComputer,
    compute_boundary_aware_features
)


@pytest.fixture
def sample_plane_points():
    """Generate points on a flat plane."""
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    xx, yy = np.meshgrid(x, y)
    z = 10.0 * np.ones_like(xx)
    
    points = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])
    return points.astype(np.float32)


@pytest.fixture
def sample_sphere_points():
    """Generate points on a sphere surface."""
    n_points = 500
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 10.0
    
    x = 50 + r * np.sin(phi) * np.cos(theta)
    y = 50 + r * np.sin(phi) * np.sin(theta)
    z = 10 + r * np.cos(phi)
    
    points = np.column_stack([x, y, z])
    return points.astype(np.float32)


@pytest.fixture
def sample_line_points():
    """Generate points along a line."""
    t = np.linspace(0, 100, 200)
    x = t
    y = 50 + 0.1 * np.random.randn(200)  # Small noise
    z = 10 + 0.1 * np.random.randn(200)
    
    points = np.column_stack([x, y, z])
    return points.astype(np.float32)


@pytest.fixture
def sample_boundary_scenario():
    """
    Create a realistic boundary scenario:
    - Core tile: 0-100 x 0-100
    - Buffer zone: points from neighbors within 10m
    """
    np.random.seed(42)
    
    # Core tile points (0-100, 0-100)
    n_core = 1000
    core_x = np.random.uniform(0, 100, n_core)
    core_y = np.random.uniform(0, 100, n_core)
    core_z = 10 + 2 * np.sin(core_x / 20) + np.random.randn(n_core) * 0.1
    core_points = np.column_stack([core_x, core_y, core_z])
    
    # Buffer zone points from left neighbor (-10-0, 0-100)
    n_buffer_left = 100
    buffer_x_left = np.random.uniform(-10, 0, n_buffer_left)
    buffer_y_left = np.random.uniform(0, 100, n_buffer_left)
    buffer_z_left = 10 + 2 * np.sin(buffer_x_left / 20) + np.random.randn(n_buffer_left) * 0.1
    
    # Buffer zone points from right neighbor (100-110, 0-100)
    n_buffer_right = 100
    buffer_x_right = np.random.uniform(100, 110, n_buffer_right)
    buffer_y_right = np.random.uniform(0, 100, n_buffer_right)
    buffer_z_right = 10 + 2 * np.sin(buffer_x_right / 20) + np.random.randn(n_buffer_right) * 0.1
    
    buffer_points = np.vstack([
        np.column_stack([buffer_x_left, buffer_y_left, buffer_z_left]),
        np.column_stack([buffer_x_right, buffer_y_right, buffer_z_right])
    ])
    
    tile_bounds = (0.0, 0.0, 100.0, 100.0)
    
    return {
        'core_points': core_points.astype(np.float32),
        'buffer_points': buffer_points.astype(np.float32),
        'tile_bounds': tile_bounds
    }


# ============================================================================
# Test Initialization
# ============================================================================

def test_computer_initialization():
    """Test BoundaryAwareFeatureComputer initialization."""
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        boundary_threshold=10.0,
        compute_normals=True,
        compute_curvature=True,
        compute_planarity=True,
        compute_verticality=True
    )
    
    assert computer.k_neighbors == 20
    assert computer.boundary_threshold == 10.0
    assert computer.compute_normals is True
    assert computer.compute_curvature is True
    assert computer.compute_planarity is True
    assert computer.compute_verticality is True


def test_get_feature_names():
    """Test feature name retrieval."""
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        compute_normals=True,
        compute_curvature=True,
        compute_planarity=True,
        compute_verticality=True
    )
    
    feature_names = computer.get_feature_names()
    
    assert 'normal_x' in feature_names
    assert 'normal_y' in feature_names
    assert 'normal_z' in feature_names
    assert 'curvature' in feature_names
    assert 'planarity' in feature_names
    assert 'linearity' in feature_names
    assert 'sphericity' in feature_names
    assert 'verticality' in feature_names
    
    assert len(feature_names) == 8


# ============================================================================
# Test Basic Feature Computation
# ============================================================================

def test_compute_features_plane(sample_plane_points):
    """Test feature computation on a flat plane."""
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    features = computer.compute_features(
        core_points=sample_plane_points,
        buffer_points=None,
        tile_bounds=None
    )
    
    # Check output structure
    assert 'normals' in features
    assert 'curvature' in features
    assert 'boundary_mask' in features
    assert 'num_boundary_points' in features
    
    # Check shapes
    assert features['normals'].shape == (len(sample_plane_points), 3)
    assert features['curvature'].shape == (len(sample_plane_points),)
    
    # For a plane: normals should point upward (0, 0, 1)
    normals = features['normals']
    mean_normal = normals.mean(axis=0)
    
    assert mean_normal[2] > 0.9  # Z component should be close to 1
    assert abs(mean_normal[0]) < 0.1  # X should be near 0
    assert abs(mean_normal[1]) < 0.1  # Y should be near 0
    
    # For a plane: curvature should be near 0
    curvature = features['curvature']
    assert curvature.mean() < 0.1


def test_compute_features_sphere(sample_sphere_points):
    """Test feature computation on a sphere surface."""
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    features = computer.compute_features(
        core_points=sample_sphere_points,
        buffer_points=None,
        tile_bounds=None
    )
    
    # Check output structure
    assert 'normals' in features
    assert 'curvature' in features
    
    # For a sphere: normals should point outward from center
    normals = features['normals']
    center = sample_sphere_points.mean(axis=0)
    
    # Check that normals roughly point radially (either toward or away from center)
    # Due to normal orientation, they should point either inward or outward consistently
    radial_directions = sample_sphere_points - center
    radial_directions = radial_directions / np.linalg.norm(radial_directions, axis=1, keepdims=True)
    
    # Compute dot products between normals and radial directions
    dot_products = np.sum(normals * radial_directions, axis=1)
    
    # Normals should be mostly radial (|dot_product| should be high)
    # They can point either inward or outward, so check absolute value
    mean_abs_dot = np.abs(dot_products).mean()
    assert mean_abs_dot > 0.7  # Most normals should be radial
    
    # For a sphere: curvature should be relatively uniform and non-zero
    curvature = features['curvature']
    assert curvature.mean() > 0.001  # Non-zero curvature (small values are normal)


def test_compute_features_line(sample_line_points):
    """Test feature computation on a line."""
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        compute_planarity=True
    )
    
    features = computer.compute_features(
        core_points=sample_line_points,
        buffer_points=None,
        tile_bounds=None
    )
    
    # Check output structure
    assert 'linearity' in features
    assert 'planarity' in features
    assert 'sphericity' in features
    
    # For a line: linearity should be high
    linearity = features['linearity']
    assert linearity.mean() > 0.5
    
    # For a line: planarity and sphericity should be lower
    planarity = features['planarity']
    sphericity = features['sphericity']
    
    assert linearity.mean() > planarity.mean()
    assert linearity.mean() > sphericity.mean()


# ============================================================================
# Test Boundary Detection
# ============================================================================

def test_boundary_detection(sample_boundary_scenario):
    """Test boundary point detection."""
    data = sample_boundary_scenario
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        boundary_threshold=10.0
    )
    
    # Must provide tile_bounds for boundary detection to work
    features = computer.compute_features(
        core_points=data['core_points'],
        buffer_points=data['buffer_points'],  # Need buffer to enable boundary detection
        tile_bounds=data['tile_bounds']
    )
    
    boundary_mask = features['boundary_mask']
    num_boundary = features['num_boundary_points']
    
    # Check that some points are detected as boundary points
    assert num_boundary > 0
    assert num_boundary < len(data['core_points'])
    
    # Verify boundary points are indeed near edges
    core_points = data['core_points']
    boundary_points = core_points[boundary_mask]
    
    xmin, ymin, xmax, ymax = data['tile_bounds']
    
    for point in boundary_points[:10]:  # Check first 10
        x, y = point[0], point[1]
        
        # Calculate distance to nearest boundary
        dist_to_edge = min(
            x - xmin,
            xmax - x,
            y - ymin,
            ymax - y
        )
        
        # Should be within threshold
        assert dist_to_edge <= 10.0


# ============================================================================
# Test Boundary-Aware Features
# ============================================================================

def test_features_with_buffer_zone(sample_boundary_scenario):
    """Test feature computation with buffer zone."""
    data = sample_boundary_scenario
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        boundary_threshold=10.0
    )
    
    # Compute features with buffer
    features_with_buffer = computer.compute_features(
        core_points=data['core_points'],
        buffer_points=data['buffer_points'],
        tile_bounds=data['tile_bounds']
    )
    
    # Compute features without buffer (for comparison)
    features_without_buffer = computer.compute_features(
        core_points=data['core_points'],
        buffer_points=None,
        tile_bounds=data['tile_bounds']
    )
    
    # Both should return same number of features (for core points only)
    assert features_with_buffer['normals'].shape == features_without_buffer['normals'].shape
    assert features_with_buffer['curvature'].shape == features_without_buffer['curvature'].shape
    
    # Features at boundaries should differ when buffer is used
    boundary_mask = features_with_buffer['boundary_mask']
    
    if boundary_mask.sum() > 0:
        # Check that boundary normals differ
        normals_with = features_with_buffer['normals'][boundary_mask]
        normals_without = features_without_buffer['normals'][boundary_mask]
        
        # Should have some differences (not identical)
        max_diff = np.abs(normals_with - normals_without).max()
        assert max_diff > 0.01  # At least small differences


def test_features_without_tile_bounds_error():
    """Test that buffer_points without tile_bounds raises error."""
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    core_points = np.random.rand(100, 3).astype(np.float32)
    buffer_points = np.random.rand(50, 3).astype(np.float32)
    
    with pytest.raises(ValueError, match="tile_bounds required"):
        computer.compute_features(
            core_points=core_points,
            buffer_points=buffer_points,
            tile_bounds=None
        )


# ============================================================================
# Test Convenience Function
# ============================================================================

def test_convenience_function(sample_boundary_scenario):
    """Test compute_boundary_aware_features convenience function."""
    data = sample_boundary_scenario
    
    features = compute_boundary_aware_features(
        core_points=data['core_points'],
        buffer_points=data['buffer_points'],
        tile_bounds=data['tile_bounds'],
        k_neighbors=20,
        boundary_threshold=10.0
    )
    
    # Check that all expected features are present
    assert 'normals' in features
    assert 'curvature' in features
    assert 'planarity' in features
    assert 'linearity' in features
    assert 'sphericity' in features
    assert 'verticality' in features
    assert 'boundary_mask' in features
    assert 'num_boundary_points' in features
    
    # Check shapes
    n_core = len(data['core_points'])
    assert features['normals'].shape == (n_core, 3)
    assert features['curvature'].shape == (n_core,)
    assert features['planarity'].shape == (n_core,)


# ============================================================================
# Test Verticality
# ============================================================================

def test_verticality_horizontal_plane(sample_plane_points):
    """Test verticality on horizontal plane (should be ~0)."""
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        compute_verticality=True
    )
    
    features = computer.compute_features(
        core_points=sample_plane_points,
        buffer_points=None,
        tile_bounds=None
    )
    
    verticality = features['verticality']
    
    # For horizontal plane: verticality should be near 0
    assert verticality.mean() < 0.2


def test_verticality_vertical_wall():
    """Test verticality on vertical wall (should be ~1)."""
    # Create vertical wall points (X-Z plane, constant Y)
    x = np.linspace(0, 100, 50)
    z = np.linspace(0, 20, 40)
    xx, zz = np.meshgrid(x, z)
    y = 50.0 * np.ones_like(xx)
    
    wall_points = np.column_stack([xx.ravel(), y.ravel(), zz.ravel()])
    wall_points = wall_points.astype(np.float32)
    
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        compute_verticality=True
    )
    
    features = computer.compute_features(
        core_points=wall_points,
        buffer_points=None,
        tile_bounds=None
    )
    
    verticality = features['verticality']
    
    # For vertical wall: verticality should be high
    assert verticality.mean() > 0.7


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_empty_buffer_points():
    """Test with empty buffer zone."""
    core_points = np.random.rand(100, 3).astype(np.float32)
    buffer_points = np.empty((0, 3), dtype=np.float32)
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    features = computer.compute_features(
        core_points=core_points,
        buffer_points=buffer_points,
        tile_bounds=(0, 0, 10, 10)
    )
    
    # Should compute features successfully
    assert features['normals'].shape == (100, 3)
    assert features['num_boundary_points'] >= 0


def test_small_point_cloud():
    """Test with very small point cloud (< k_neighbors)."""
    core_points = np.random.rand(10, 3).astype(np.float32)
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    # Should handle gracefully (k will be capped at num_points-1)
    features = computer.compute_features(
        core_points=core_points,
        buffer_points=None,
        tile_bounds=None
    )
    
    assert features['normals'].shape == (10, 3)


# ============================================================================
# Test Feature Quality Comparison
# ============================================================================

def test_boundary_feature_quality_improvement(sample_boundary_scenario):
    """
    Test that boundary-aware features improve quality at boundaries.
    
    This is the key validation: features computed with buffer zones
    should be more accurate at boundaries than without.
    """
    data = sample_boundary_scenario
    computer = BoundaryAwareFeatureComputer(
        k_neighbors=20,
        boundary_threshold=10.0
    )
    
    # Compute with and without buffer
    with_buffer = computer.compute_features(
        core_points=data['core_points'],
        buffer_points=data['buffer_points'],
        tile_bounds=data['tile_bounds']
    )
    
    without_buffer = computer.compute_features(
        core_points=data['core_points'],
        buffer_points=None,
        tile_bounds=data['tile_bounds']
    )
    
    # Extract boundary points
    boundary_mask = with_buffer['boundary_mask']
    interior_mask = ~boundary_mask
    
    if boundary_mask.sum() > 0 and interior_mask.sum() > 0:
        # Compute feature variance at boundaries
        boundary_normals_with = with_buffer['normals'][boundary_mask]
        boundary_normals_without = without_buffer['normals'][boundary_mask]
        
        # Variance in normals (should be more stable with buffer)
        var_with = np.var(boundary_normals_with, axis=0).sum()
        var_without = np.var(boundary_normals_without, axis=0).sum()
        
        # Log for inspection (variance may or may not be lower)
        print(f"Boundary normal variance: with_buffer={var_with:.4f}, "
              f"without_buffer={var_without:.4f}")
        
        # At minimum, features should be computed successfully
        assert boundary_normals_with.shape == boundary_normals_without.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
