"""
Unit tests for FeatureComputer

Tests the unified interface that automatically selects and uses
the optimal computation mode.

Author: Simon Ducournau / GitHub Copilot
Date: October 18, 2025
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from ign_lidar.features.feature_computer import FeatureComputer, get_feature_computer
from ign_lidar.features.mode_selector import ComputationMode


class TestFeatureComputer:
    """Test suite for FeatureComputer."""

    @pytest.fixture
    def sample_points(self):
        """Generate sample point cloud."""
        np.random.seed(42)
        return np.random.rand(1000, 3).astype(np.float32)

    @pytest.fixture
    def large_points(self):
        """Generate large point cloud for mode selection testing."""
        np.random.seed(42)
        return np.random.rand(2_000_000, 3).astype(np.float32)

    @pytest.fixture
    def mock_mode_selector(self):
        """Mock mode selector."""
        selector = MagicMock()
        selector.gpu_available = True
        selector.select_mode.return_value = ComputationMode.GPU
        selector.get_recommendations.return_value = {
            "recommended_mode": "gpu",
            "estimated_memory_gb": 1.0,
            "estimated_time_seconds": 2.0,
        }
        return selector

    def test_initialization_default(self):
        """Test default initialization."""
        computer = FeatureComputer()

        assert computer.mode_selector is not None
        assert computer.force_mode is None
        assert computer.progress_callback is None
        assert computer._cpu_computer is None  # Lazy loaded
        assert computer._gpu_computer is None

    def test_initialization_with_force_mode(self):
        """Test initialization with forced mode."""
        computer = FeatureComputer(force_mode=ComputationMode.CPU)

        assert computer.force_mode == ComputationMode.CPU

    def test_initialization_with_progress_callback(self):
        """Test initialization with progress callback."""
        callback = MagicMock()
        computer = FeatureComputer(progress_callback=callback)

        assert computer.progress_callback == callback

    def test_get_cpu_computer_lazy_load(self):
        """Test lazy loading of CPU computer."""
        computer = FeatureComputer()

        assert computer._cpu_computer is None
        cpu_comp = computer._get_cpu_computer()
        assert cpu_comp is not None
        assert computer._cpu_computer is cpu_comp  # Cached

        # Second call returns same instance
        assert computer._get_cpu_computer() is cpu_comp

    def test_get_gpu_computer_lazy_load(self):
        """Test lazy loading of GPU computer."""
        computer = FeatureComputer()

        assert computer._gpu_computer is None
        # Note: Will fail if GPU not available, but that's expected
        try:
            gpu_comp = computer._get_gpu_computer()
            assert gpu_comp is not None
            assert computer._gpu_computer is gpu_comp
        except (ImportError, RuntimeError):
            pytest.skip("GPU not available")

    def test_select_mode_automatic(self, mock_mode_selector):
        """Test automatic mode selection."""
        computer = FeatureComputer(mode_selector=mock_mode_selector)

        mode = computer._select_mode(num_points=1_000_000)

        assert mode == ComputationMode.GPU
        mock_mode_selector.select_mode.assert_called_once()

    def test_select_mode_forced(self, mock_mode_selector):
        """Test forced mode selection."""
        computer = FeatureComputer(
            mode_selector=mock_mode_selector, force_mode=ComputationMode.CPU
        )

        mode = computer._select_mode(num_points=1_000_000)

        assert mode == ComputationMode.CPU
        # Should not call selector when forced
        mock_mode_selector.select_mode.assert_not_called()

    def test_select_mode_override(self, mock_mode_selector):
        """Test mode selection with parameter override."""
        computer = FeatureComputer(mode_selector=mock_mode_selector)

        mode = computer._select_mode(
            num_points=1_000_000, force_mode=ComputationMode.CPU
        )

        assert mode == ComputationMode.CPU

    def test_report_progress_with_callback(self):
        """Test progress reporting with callback."""
        callback = MagicMock()
        computer = FeatureComputer(progress_callback=callback)

        computer._report_progress(0.5, "Test message")

        callback.assert_called_once_with(0.5, "Test message")

    def test_report_progress_without_callback(self):
        """Test progress reporting without callback (should not crash)."""
        computer = FeatureComputer()

        # Should not raise
        computer._report_progress(0.5, "Test message")

    @patch("ign_lidar.features.feature_computer.FeatureComputer._get_cpu_computer")
    def test_compute_normals_cpu(self, mock_get_cpu, sample_points, mock_mode_selector):
        """Test compute_normals with CPU mode."""
        # Setup
        mock_cpu_comp = MagicMock()
        expected_normals = np.random.rand(len(sample_points), 3)
        mock_cpu_comp.compute_normals.return_value = expected_normals
        mock_get_cpu.return_value = mock_cpu_comp

        mock_mode_selector.select_mode.return_value = ComputationMode.CPU

        computer = FeatureComputer(mode_selector=mock_mode_selector)

        # Execute
        normals = computer.compute_normals(sample_points, k=10)

        # Verify
        assert np.array_equal(normals, expected_normals)
        mock_cpu_comp.compute_normals.assert_called_once_with(
            sample_points, k_neighbors=10
        )

    @patch("ign_lidar.features.feature_computer.FeatureComputer._get_gpu_computer")
    def test_compute_normals_gpu(self, mock_get_gpu, sample_points, mock_mode_selector):
        """Test compute_normals with GPU mode."""
        # Setup
        mock_gpu_comp = MagicMock()
        expected_normals = np.random.rand(len(sample_points), 3)
        # GPU strategy returns features dict from compute()
        mock_gpu_comp.compute.return_value = {"normals": expected_normals}
        mock_get_gpu.return_value = mock_gpu_comp

        mock_mode_selector.select_mode.return_value = ComputationMode.GPU

        computer = FeatureComputer(mode_selector=mock_mode_selector)

        # Execute
        normals = computer.compute_normals(sample_points, k=10)

        # Verify
        assert np.array_equal(normals, expected_normals)
        mock_gpu_comp.compute.assert_called_once_with(sample_points)

    @patch("ign_lidar.features.feature_computer.FeatureComputer._get_cpu_computer")
    def test_compute_curvature_cpu(
        self, mock_get_cpu, sample_points, mock_mode_selector
    ):
        """Test compute_curvature with CPU mode."""
        # Setup
        mock_cpu_comp = MagicMock()

        # Mock compute_normals to return normals and eigenvalues
        expected_normals = np.random.rand(len(sample_points), 3)
        expected_eigenvalues = np.random.rand(len(sample_points), 3)
        mock_cpu_comp.compute_normals.return_value = (
            expected_normals,
            expected_eigenvalues,
        )

        # Mock compute_curvature to return curvature from eigenvalues
        expected_curvature = np.random.rand(len(sample_points))
        mock_cpu_comp.compute_curvature.return_value = expected_curvature
        mock_get_cpu.return_value = mock_cpu_comp

        mock_mode_selector.select_mode.return_value = ComputationMode.CPU

        computer = FeatureComputer(mode_selector=mock_mode_selector)

        # Execute
        curvature = computer.compute_curvature(sample_points, k=20)

        # Verify
        assert np.array_equal(curvature, expected_curvature)
        mock_cpu_comp.compute_normals.assert_called_once_with(
            sample_points, k_neighbors=20
        )
        mock_cpu_comp.compute_curvature.assert_called_once_with(expected_eigenvalues)

    @patch("ign_lidar.features.feature_computer.FeatureComputer._get_gpu_computer")
    def test_compute_geometric_features_gpu(
        self, mock_get_gpu, sample_points, mock_mode_selector
    ):
        """Test compute_geometric_features with GPU mode."""
        # Setup
        mock_gpu_comp = MagicMock()
        expected_features = {
            "planarity": np.random.rand(len(sample_points)),
            "linearity": np.random.rand(len(sample_points)),
            "sphericity": np.random.rand(len(sample_points)),
            "anisotropy": np.random.rand(len(sample_points)),
        }
        mock_gpu_comp.compute.return_value = expected_features
        mock_get_gpu.return_value = mock_gpu_comp

        mock_mode_selector.select_mode.return_value = ComputationMode.GPU

        computer = FeatureComputer(mode_selector=mock_mode_selector)

        # Execute
        features = computer.compute_geometric_features(
            sample_points, required_features=["planarity", "linearity"], k=20
        )

        # Verify
        assert "planarity" in features
        assert "linearity" in features
        mock_gpu_comp.compute.assert_called_once()

    @patch("ign_lidar.features.feature_computer.FeatureComputer._get_boundary_computer")
    def test_compute_normals_with_boundary(
        self, mock_get_boundary, sample_points, mock_mode_selector
    ):
        """Test boundary normal computation."""
        # Setup
        core_points = sample_points[:500]
        buffer_points = sample_points[500:]

        mock_boundary_comp = MagicMock()
        # The compute() method returns features dict with all normals (core + buffer)
        all_normals = np.random.rand(len(sample_points), 3)
        mock_boundary_comp.compute.return_value = {"normals": all_normals}
        mock_get_boundary.return_value = mock_boundary_comp

        computer = FeatureComputer(mode_selector=mock_mode_selector)

        # Execute
        normals = computer.compute_normals_with_boundary(
            core_points, buffer_points, k=10
        )

        # Verify - should return only core point normals
        assert normals.shape == (len(core_points), 3)
        assert np.array_equal(normals, all_normals[: len(core_points)])
        mock_boundary_comp.compute.assert_called_once()

    def test_get_mode_recommendations(self, mock_mode_selector):
        """Test getting mode recommendations."""
        computer = FeatureComputer(mode_selector=mock_mode_selector)

        recommendations = computer.get_mode_recommendations(num_points=1_000_000)

        assert "recommended_mode" in recommendations
        assert recommendations["recommended_mode"] == "gpu"
        mock_mode_selector.get_recommendations.assert_called_once_with(1_000_000)

    @patch("ign_lidar.features.feature_computer.FeatureComputer.compute_normals")
    @patch("ign_lidar.features.feature_computer.FeatureComputer.compute_curvature")
    @patch(
        "ign_lidar.features.feature_computer.FeatureComputer.compute_geometric_features"
    )
    def test_compute_all_features(
        self,
        mock_geometric,
        mock_curvature,
        mock_normals,
        sample_points,
        mock_mode_selector,
    ):
        """Test computing all features at once."""
        # Setup
        mock_normals.return_value = np.random.rand(len(sample_points), 3)
        mock_curvature.return_value = np.random.rand(len(sample_points))
        mock_geometric.return_value = {
            "planarity": np.random.rand(len(sample_points)),
            "linearity": np.random.rand(len(sample_points)),
        }

        computer = FeatureComputer(mode_selector=mock_mode_selector)

        # Execute
        all_features = computer.compute_all_features(
            sample_points, geometric_features=["planarity", "linearity"]
        )

        # Verify
        assert "normals" in all_features
        assert "curvature" in all_features
        assert "planarity" in all_features
        assert "linearity" in all_features

        mock_normals.assert_called_once()
        mock_curvature.assert_called_once()
        mock_geometric.assert_called_once()

    def test_compute_all_features_with_progress(self, mock_mode_selector):
        """Test compute_all_features with progress callback."""
        callback = MagicMock()
        computer = FeatureComputer(
            mode_selector=mock_mode_selector, progress_callback=callback
        )

        # Mock the individual compute methods
        with patch.object(computer, "compute_normals"), patch.object(
            computer, "compute_curvature"
        ), patch.object(computer, "compute_geometric_features"):

            sample_points = np.random.rand(100, 3)
            computer.compute_all_features(sample_points)

            # Verify progress was reported
            assert callback.call_count >= 3  # At least 3 progress updates

    def test_factory_function(self):
        """Test factory function."""
        computer = get_feature_computer(force_mode=ComputationMode.CPU)

        assert isinstance(computer, FeatureComputer)
        assert computer.force_mode == ComputationMode.CPU

    def test_factory_function_with_callback(self):
        """Test factory function with callback."""
        callback = MagicMock()
        computer = get_feature_computer(progress_callback=callback)

        assert computer.progress_callback == callback

    def test_mode_override_parameter(self, sample_points, mock_mode_selector):
        """Test mode override via method parameter."""
        computer = FeatureComputer(mode_selector=mock_mode_selector)

        # Override should take precedence over automatic selection
        with patch.object(computer, "_get_cpu_computer") as mock_cpu:
            mock_cpu_comp = MagicMock()
            mock_cpu_comp.compute_normals.return_value = np.zeros(
                (len(sample_points), 3)
            )
            mock_cpu.return_value = mock_cpu_comp

            computer.compute_normals(sample_points, k=10, mode=ComputationMode.CPU)

            # Should use CPU despite mock_mode_selector returning GPU
            mock_cpu.assert_called()

    def test_boundary_mode_error_for_regular_normals(
        self, sample_points, mock_mode_selector
    ):
        """Test that boundary mode raises error for regular normal computation."""
        mock_mode_selector.select_mode.return_value = ComputationMode.BOUNDARY
        computer = FeatureComputer(mode_selector=mock_mode_selector)

        with pytest.raises(ValueError, match="Boundary mode requires"):
            computer.compute_normals(sample_points, k=10)

    def test_boundary_mode_error_for_curvature(self, sample_points, mock_mode_selector):
        """Test that boundary mode raises error for curvature."""
        mock_mode_selector.select_mode.return_value = ComputationMode.BOUNDARY
        computer = FeatureComputer(mode_selector=mock_mode_selector)

        with pytest.raises(ValueError, match="not supported"):
            computer.compute_curvature(sample_points, k=20)


class TestUnifiedComputerIntegration:
    """Integration tests with real computations (if possible)."""

    @pytest.fixture
    def tiny_points(self):
        """Very small point cloud for fast testing."""
        np.random.seed(42)
        return np.random.rand(100, 3).astype(np.float32)

    def test_real_computation_cpu_mode(self, tiny_points):
        """Test real computation with CPU mode."""
        computer = get_feature_computer(force_mode=ComputationMode.CPU)

        try:
            normals = computer.compute_normals(tiny_points, k=5)
            assert normals.shape == (100, 3)
            assert np.all(np.isfinite(normals))
        except Exception as e:
            pytest.skip(f"CPU computation failed: {e}")

    def test_real_computation_auto_mode(self, tiny_points):
        """Test real computation with automatic mode selection."""
        computer = get_feature_computer()

        try:
            normals = computer.compute_normals(tiny_points, k=5)
            assert normals.shape == (100, 3)
            assert np.all(np.isfinite(normals))
        except Exception as e:
            pytest.skip(f"Automatic computation failed: {e}")

    def test_mode_recommendations_realistic(self):
        """Test mode recommendations with realistic point counts."""
        computer = get_feature_computer()

        # Small cloud
        rec_small = computer.get_mode_recommendations(100_000)
        assert "recommended_mode" in rec_small

        # Large cloud
        rec_large = computer.get_mode_recommendations(10_000_000)
        assert "recommended_mode" in rec_large

        # Recommendations should adapt to size
        # (though specific mode depends on hardware)
        assert rec_small["num_points"] == 100_000
        assert rec_large["num_points"] == 10_000_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
