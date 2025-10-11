"""
Unit tests for Feature Computer Factory.

Tests the factory pattern for creating appropriate feature computers
based on configuration.
"""

import pytest
import numpy as np
from ign_lidar.features.factory import (
    FeatureComputerFactory,
    BaseFeatureComputer,
    CPUFeatureComputer,
    GPUFeatureComputer,
    GPUChunkedFeatureComputer,
    BoundaryAwareFeatureComputer,
)


class TestFeatureComputerFactory:
    """Test suite for FeatureComputerFactory."""

    def test_create_cpu_computer(self):
        """Test creating CPU feature computer."""
        computer = FeatureComputerFactory.create(use_gpu=False)
        assert isinstance(computer, CPUFeatureComputer)
        assert isinstance(computer, BaseFeatureComputer)
        assert computer.k_neighbors == 20  # default

    def test_create_cpu_with_custom_neighbors(self):
        """Test creating CPU computer with custom k_neighbors."""
        computer = FeatureComputerFactory.create(use_gpu=False, k_neighbors=15)
        assert isinstance(computer, CPUFeatureComputer)
        assert computer.k_neighbors == 15

    def test_create_gpu_computer(self):
        """Test creating GPU feature computer."""
        computer = FeatureComputerFactory.create(use_gpu=True)
        assert isinstance(computer, GPUFeatureComputer) or isinstance(computer, CPUFeatureComputer)
        # Falls back to CPU if GPU not available

    def test_create_gpu_chunked_computer(self):
        """Test creating GPU chunked feature computer."""
        computer = FeatureComputerFactory.create(
            use_gpu=True,
            use_chunked=True,
            gpu_batch_size=500_000
        )
        # Should be GPUChunkedFeatureComputer or fall back to CPU
        assert isinstance(computer, (GPUChunkedFeatureComputer, CPUFeatureComputer))

    def test_create_boundary_aware_computer(self):
        """Test creating boundary-aware feature computer."""
        computer = FeatureComputerFactory.create(
            use_boundary_aware=True,
            buffer_size=15.0,
            tile_bounds=(0, 0, 100, 100)
        )
        assert isinstance(computer, BoundaryAwareFeatureComputer)
        assert computer.buffer_size == 15.0
        assert computer.tile_bounds == (0, 0, 100, 100)

    def test_boundary_aware_takes_precedence(self):
        """Test that boundary-aware takes precedence over GPU."""
        computer = FeatureComputerFactory.create(
            use_gpu=True,
            use_chunked=True,
            use_boundary_aware=True,
            buffer_size=10.0
        )
        assert isinstance(computer, BoundaryAwareFeatureComputer)

    def test_list_available(self):
        """Test listing available feature computers."""
        availability = FeatureComputerFactory.list_available()
        
        assert isinstance(availability, dict)
        assert 'cpu' in availability
        assert 'gpu' in availability
        assert 'gpu_chunked' in availability
        assert 'boundary_aware' in availability
        
        assert availability['cpu'] is True
        assert availability['boundary_aware'] is True
        # GPU availability depends on environment

    def test_get_recommended_cpu(self):
        """Test getting recommended computer for CPU scenario."""
        recommendation = FeatureComputerFactory.get_recommended(
            num_points=1_000_000,
            has_gpu=False,
            stitching_enabled=False
        )
        assert recommendation == 'cpu'

    def test_get_recommended_gpu(self):
        """Test getting recommended computer for GPU scenario."""
        recommendation = FeatureComputerFactory.get_recommended(
            num_points=1_000_000,
            has_gpu=True,
            stitching_enabled=False
        )
        assert recommendation == 'gpu'

    def test_get_recommended_gpu_chunked(self):
        """Test getting recommended computer for large GPU scenario."""
        recommendation = FeatureComputerFactory.get_recommended(
            num_points=5_000_000,
            has_gpu=True,
            stitching_enabled=False
        )
        assert recommendation == 'gpu_chunked'

    def test_get_recommended_boundary_aware(self):
        """Test getting recommended computer for stitching scenario."""
        recommendation = FeatureComputerFactory.get_recommended(
            num_points=1_000_000,
            has_gpu=True,
            stitching_enabled=True
        )
        assert recommendation == 'boundary_aware'


class TestCPUFeatureComputer:
    """Test suite for CPUFeatureComputer."""

    def test_is_available(self):
        """Test CPU computer is always available."""
        computer = CPUFeatureComputer(k_neighbors=20)
        assert computer.is_available() is True

    @pytest.mark.skip(reason="Requires actual point cloud data")
    def test_compute_features(self):
        """Test computing features with CPU computer."""
        # This test requires actual point cloud data
        # Would need sample data fixture
        pass


class TestGPUFeatureComputer:
    """Test suite for GPUFeatureComputer."""

    def test_initialization(self):
        """Test GPU computer initialization."""
        computer = GPUFeatureComputer(k_neighbors=20)
        # Should initialize without error
        assert computer.k_neighbors == 20

    def test_availability_check(self):
        """Test GPU availability check."""
        computer = GPUFeatureComputer(k_neighbors=20)
        # May be True or False depending on environment
        assert isinstance(computer.is_available(), bool)


class TestBoundaryAwareFeatureComputer:
    """Test suite for BoundaryAwareFeatureComputer."""

    def test_initialization(self):
        """Test boundary-aware computer initialization."""
        computer = BoundaryAwareFeatureComputer(
            k_neighbors=20,
            buffer_size=10.0,
            tile_bounds=(0, 0, 100, 100)
        )
        assert computer.k_neighbors == 20
        assert computer.buffer_size == 10.0
        assert computer.tile_bounds == (0, 0, 100, 100)

    def test_is_available(self):
        """Test boundary-aware computer is always available."""
        computer = BoundaryAwareFeatureComputer(k_neighbors=20)
        assert computer.is_available() is True


class TestWallRoofScoreFeatures:
    """Test suite for wall and roof score features."""

    def test_cpu_computer_includes_wall_roof_scores(self):
        """Test that CPU computer computes wall and roof scores."""
        # Create sample points (simple vertical plane for wall)
        n_points = 100
        points = np.column_stack([
            np.random.uniform(0, 10, n_points),  # x
            np.random.uniform(0, 1, n_points),   # y (thin)
            np.random.uniform(0, 10, n_points),  # z
        ]).astype(np.float32)
        classification = np.full(n_points, 6, dtype=np.uint8)  # building
        
        computer = FeatureComputerFactory.create(use_gpu=False, k_neighbors=10)
        features = computer.compute_features(points, classification)
        
        # Check that wall_score and roof_score are present
        assert 'wall_score' in features
        assert 'roof_score' in features
        assert features['wall_score'].shape == (n_points,)
        assert features['roof_score'].shape == (n_points,)
        
        # Check value ranges [0, 1]
        assert np.all(features['wall_score'] >= 0.0)
        assert np.all(features['wall_score'] <= 1.0)
        assert np.all(features['roof_score'] >= 0.0)
        assert np.all(features['roof_score'] <= 1.0)

    def test_wall_score_for_vertical_plane(self):
        """Test that vertical plane has high wall score."""
        # Create a perfect vertical plane (X-Z plane, constant Y)
        x = np.linspace(0, 10, 20)
        z = np.linspace(0, 10, 20)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, 5.0)  # Constant Y = vertical plane
        
        points = np.column_stack([
            X.ravel(),
            Y.ravel(),
            Z.ravel()
        ]).astype(np.float32)
        classification = np.full(len(points), 6, dtype=np.uint8)
        
        computer = FeatureComputerFactory.create(use_gpu=False, k_neighbors=10)
        features = computer.compute_features(points, classification)
        
        # Vertical plane should have high planarity and high verticality
        # Therefore, high wall_score
        mean_wall_score = np.mean(features['wall_score'])
        mean_roof_score = np.mean(features['roof_score'])
        
        # Wall score should be significantly higher than roof score
        assert mean_wall_score > 0.5, f"Expected high wall score, got {mean_wall_score}"
        assert mean_wall_score > mean_roof_score, "Wall score should exceed roof score for vertical plane"

    def test_roof_score_for_horizontal_plane(self):
        """Test that horizontal plane has high roof score."""
        # Create a perfect horizontal plane (X-Y plane, constant Z)
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, 5.0)  # Constant Z = horizontal plane
        
        points = np.column_stack([
            X.ravel(),
            Y.ravel(),
            Z.ravel()
        ]).astype(np.float32)
        classification = np.full(len(points), 6, dtype=np.uint8)
        
        computer = FeatureComputerFactory.create(use_gpu=False, k_neighbors=10)
        features = computer.compute_features(points, classification)
        
        # Horizontal plane should have high planarity and high horizontality
        # Therefore, high roof_score
        mean_wall_score = np.mean(features['wall_score'])
        mean_roof_score = np.mean(features['roof_score'])
        
        # Roof score should be significantly higher than wall score
        assert mean_roof_score > 0.5, f"Expected high roof score, got {mean_roof_score}"
        assert mean_roof_score > mean_wall_score, "Roof score should exceed wall score for horizontal plane"

    def test_features_sum_relationship(self):
        """Test that wall_score + roof_score correlates with planarity."""
        # Create random points
        n_points = 100
        points = np.random.uniform(0, 10, (n_points, 3)).astype(np.float32)
        classification = np.full(n_points, 6, dtype=np.uint8)
        
        computer = FeatureComputerFactory.create(use_gpu=False, k_neighbors=10)
        features = computer.compute_features(points, classification)
        
        # For planar surfaces: wall_score + roof_score ≈ planarity
        # (since verticality + horizontality = 1)
        wall_plus_roof = features['wall_score'] + features['roof_score']
        planarity = features['planarity']
        
        # They should be approximately equal for most points
        correlation = np.corrcoef(wall_plus_roof, planarity)[0, 1]
        assert correlation > 0.9, f"Expected strong correlation, got {correlation}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
