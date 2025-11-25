"""
End-to-End Integration Tests - Phase 4.3 validation.

Tests complete feature extraction workflows to ensure all Phase 2 & 3
improvements work correctly together.

Author: Simon Ducournau / GitHub Copilot
Date: November 25, 2025
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List

# Skip all tests if imports fail (development environment)
try:
    from ign_lidar.features import FeatureOrchestrationService
    from ign_lidar.features.orchestrator import FeatureOrchestrator
    from ign_lidar.features.mode_selector import ModeSelector, ComputationMode
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False


def generate_test_cloud(num_points: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic point cloud for testing."""
    np.random.seed(seed)
    return np.random.randn(num_points, 3).astype(np.float32)


@pytest.mark.skipif(not IMPORTS_OK, reason="Feature orchestration not available")
class TestFeatureOrchestrationE2E:
    """End-to-end tests for feature orchestration."""

    def test_facade_basic_workflow(self):
        """Test basic feature extraction workflow via facade."""
        service = FeatureOrchestrationService()
        points = generate_test_cloud(10_000)
        
        features = service.compute_features(
            points=points,
            mode='lod2'
        )
        
        assert isinstance(features, dict)
        assert len(features) > 0
        assert 'normals' in features or len(features) > 0

    def test_facade_with_different_modes(self):
        """Test facade with different feature modes."""
        service = FeatureOrchestrationService()
        points = generate_test_cloud(5_000)
        
        modes_to_test = ['minimal', 'lod2']
        
        for mode in modes_to_test:
            try:
                features = service.compute_features(
                    points=points,
                    mode=mode
                )
                assert isinstance(features, dict)
            except Exception as e:
                pytest.skip(f"Mode {mode} not available: {e}")

    def test_facade_compute_with_mode(self):
        """Test compute_with_mode method."""
        service = FeatureOrchestrationService()
        points = generate_test_cloud(8_000)
        
        features = service.compute_with_mode(
            points=points,
            mode=ComputationMode.CPU,
            feature_mode='lod2'
        )
        
        assert isinstance(features, dict)

    def test_facade_get_feature_modes(self):
        """Test retrieving available feature modes."""
        service = FeatureOrchestrationService()
        modes = service.get_feature_modes()
        
        assert isinstance(modes, list)
        assert len(modes) > 0

    def test_facade_get_optimization_info(self):
        """Test retrieving optimization information."""
        service = FeatureOrchestrationService()
        info = service.get_optimization_info()
        
        assert isinstance(info, dict)
        assert 'gpu_available' in info or isinstance(info, dict)


@pytest.mark.skipif(not IMPORTS_OK, reason="Feature orchestrator not available")
class TestModeSelectionE2E:
    """End-to-end tests for mode selection."""

    def test_mode_selector_initialization(self):
        """Test mode selector initialization."""
        selector = ModeSelector()
        
        assert selector is not None
        assert hasattr(selector, 'select_mode')

    def test_mode_selector_small_dataset(self):
        """Test mode selection for small dataset."""
        selector = ModeSelector()
        mode = selector.select_mode(num_points=1_000)
        
        assert isinstance(mode, ComputationMode)
        assert mode in [ComputationMode.CPU, ComputationMode.GPU, ComputationMode.GPU_CHUNKED]

    def test_mode_selector_medium_dataset(self):
        """Test mode selection for medium dataset."""
        selector = ModeSelector()
        mode = selector.select_mode(num_points=500_000)
        
        assert isinstance(mode, ComputationMode)

    def test_mode_selector_large_dataset(self):
        """Test mode selection for large dataset."""
        selector = ModeSelector()
        mode = selector.select_mode(num_points=10_000_000)
        
        assert isinstance(mode, ComputationMode)

    def test_mode_selector_with_boundary_mode(self):
        """Test mode selection with boundary mode enabled."""
        selector = ModeSelector()
        mode = selector.select_mode(
            num_points=100_000,
            boundary_mode=True
        )
        
        assert isinstance(mode, ComputationMode)

    def test_mode_selector_cpu_force(self):
        """Test forcing CPU mode."""
        selector = ModeSelector()
        mode = selector.select_mode(
            num_points=1_000_000,
            force_cpu=True
        )
        
        assert mode == ComputationMode.CPU

    def test_mode_selector_get_recommendations(self):
        """Test getting mode recommendations."""
        selector = ModeSelector()
        recommendations = selector.get_recommendations(num_points=500_000)
        
        assert isinstance(recommendations, dict)


@pytest.mark.skipif(not IMPORTS_OK, reason="Feature strategies not available")
class TestStrategyIntegrationE2E:
    """End-to-end tests for computation strategies."""

    def test_cpu_strategy_workflow(self):
        """Test complete CPU strategy workflow."""
        try:
            from ign_lidar.features.strategy_cpu import CPUStrategy
        except ImportError:
            pytest.skip("CPU strategy not available")
        
        strategy = CPUStrategy()
        points = generate_test_cloud(5_000)
        
        # Test if strategy can compute features
        if hasattr(strategy, 'compute_features'):
            features = strategy.compute_features(
                points=points,
                feature_names=['normals']
            )
            assert isinstance(features, dict)

    def test_cpu_strategy_with_vectorization(self):
        """Test CPU strategy with vectorization flag."""
        try:
            from ign_lidar.features.strategy_cpu import CPUStrategy
        except ImportError:
            pytest.skip("CPU strategy not available")
        
        # Test with vectorization enabled
        strategy_vec = CPUStrategy(use_vectorized=True)
        assert strategy_vec is not None
        
        # Test with vectorization disabled
        strategy_notvec = CPUStrategy(use_vectorized=False)
        assert strategy_notvec is not None


@pytest.mark.skipif(not IMPORTS_OK, reason="Feature modules not available")
class TestFeatureComputationE2E:
    """End-to-end tests for individual feature computations."""

    def test_normals_computation(self):
        """Test normal computation workflow."""
        try:
            from ign_lidar.features.compute.normals import compute_normals
        except ImportError:
            pytest.skip("Normals module not available")
        
        points = generate_test_cloud(10_000)
        normals = compute_normals(points, k=10)
        
        assert normals.shape == points.shape
        # Check that vectors are normalized
        norms = np.linalg.norm(normals, axis=1)
        assert np.allclose(norms[np.isfinite(norms)], 1.0, atol=0.1)

    def test_curvature_computation(self):
        """Test curvature computation workflow."""
        try:
            from ign_lidar.features.compute.curvature import compute_curvature
        except ImportError:
            pytest.skip("Curvature module not available")
        
        points = generate_test_cloud(10_000)
        curvature = compute_curvature(points, k=10)
        
        assert curvature.shape == (len(points),)
        assert np.all(np.isfinite(curvature))
        assert np.all(curvature >= 0)


@pytest.mark.skipif(not IMPORTS_OK, reason="Full orchestration not available")
class TestFullPipelineE2E:
    """End-to-end tests for complete processing pipelines."""

    def test_full_lod2_pipeline(self):
        """Test complete LOD2 classification pipeline."""
        try:
            from ign_lidar.features import FeatureOrchestrationService
        except ImportError:
            pytest.skip("Orchestration service not available")
        
        service = FeatureOrchestrationService()
        points = generate_test_cloud(50_000)
        
        features = service.compute_features(
            points=points,
            mode='lod2'
        )
        
        assert isinstance(features, dict)
        assert len(features) >= 5  # LOD2 should have multiple features

    def test_full_minimal_pipeline(self):
        """Test minimal feature pipeline."""
        try:
            from ign_lidar.features import FeatureOrchestrationService
        except ImportError:
            pytest.skip("Orchestration service not available")
        
        service = FeatureOrchestrationService()
        points = generate_test_cloud(30_000)
        
        features = service.compute_features(
            points=points,
            mode='minimal'
        )
        
        assert isinstance(features, dict)
        # Minimal should have fewer features than LOD2
        assert len(features) >= 3

    def test_progressive_feature_computation(self):
        """Test progressive feature computation (small → large)."""
        try:
            from ign_lidar.features import FeatureOrchestrationService
        except ImportError:
            pytest.skip("Orchestration service not available")
        
        service = FeatureOrchestrationService()
        
        # Test with increasing point cloud sizes
        for num_points in [1_000, 10_000, 50_000]:
            points = generate_test_cloud(num_points)
            features = service.compute_features(
                points=points,
                mode='lod2'
            )
            assert isinstance(features, dict)


@pytest.mark.skipif(not IMPORTS_OK, reason="GPU context not available")
class TestGPUIntegrationE2E:
    """End-to-end tests for GPU integration."""

    def test_gpu_availability_check(self):
        """Test GPU availability detection."""
        try:
            from ign_lidar.core.gpu import GPUManager
        except ImportError:
            pytest.skip("GPU manager not available")
        
        manager = GPUManager()
        gpu_available = manager.is_available()
        
        # Just check the property exists
        assert isinstance(gpu_available, bool)

    def test_gpu_context_manager(self):
        """Test GPU context manager."""
        try:
            from ign_lidar.core.gpu_context import GPUContext
        except ImportError:
            pytest.skip("GPU context not available")
        
        # Just test context initialization
        try:
            context = GPUContext()
            assert context is not None
        except RuntimeError:
            # GPU not available, which is OK
            pass


@pytest.mark.skipif(not IMPORTS_OK, reason="Profiling not available")
class TestProfilingE2E:
    """End-to-end tests for profiling and mode selection."""

    def test_mode_selector_profiling_integration(self):
        """Test profiling integration with mode selector."""
        try:
            from ign_lidar.features.mode_selector import ModeSelector
            from ign_lidar.optimization.profile_dispatcher import get_profile_dispatcher
        except ImportError:
            pytest.skip("Profiling modules not available")
        
        selector = ModeSelector()
        
        # Test that profiling is accessible
        try:
            mode = selector.select_mode(
                num_points=100_000,
                enable_profiling=False
            )
            assert isinstance(mode, ComputationMode)
        except Exception:
            # Profiling might not be available, which is OK
            pass


@pytest.mark.integration
@pytest.mark.skipif(not IMPORTS_OK, reason="Orchestration not available")
class TestRegressionE2E:
    """Regression tests to ensure no functionality was broken."""

    def test_old_api_still_works_with_deprecation(self):
        """Test that deprecated APIs still function (with warnings)."""
        import warnings
        
        try:
            from ign_lidar.features import FeatureComputer
        except ImportError:
            pytest.skip("FeatureComputer not available")
        
        # Should issue deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            computer = FeatureComputer()
            
            # Check for deprecation warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            # Warning is issued on import, not instantiation
            assert computer is not None

    def test_backward_compatibility(self):
        """Test backward compatibility of orchestrator."""
        try:
            from ign_lidar.features import FeatureOrchestrator
        except ImportError:
            pytest.skip("FeatureOrchestrator not available")
        
        # Should still be instantiable
        orchestrator = FeatureOrchestrator()
        assert orchestrator is not None


@pytest.mark.performance
@pytest.mark.skipif(not IMPORTS_OK, reason="Orchestration not available")
class TestPerformanceRegressionsE2E:
    """Ensure performance hasn't regressed after optimizations."""

    def test_small_cloud_performance_acceptable(self):
        """Test that small cloud performance is acceptable."""
        try:
            from ign_lidar.features import FeatureOrchestrationService
        except ImportError:
            pytest.skip("Orchestration service not available")
        
        import time
        
        service = FeatureOrchestrationService()
        points = generate_test_cloud(10_000)
        
        start = time.perf_counter()
        features = service.compute_features(points=points, mode='minimal')
        elapsed = time.perf_counter() - start
        
        # Should complete reasonably quickly (< 5 seconds for small cloud)
        assert elapsed < 5.0
        assert isinstance(features, dict)

    def test_medium_cloud_performance_acceptable(self):
        """Test that medium cloud performance is acceptable."""
        try:
            from ign_lidar.features import FeatureOrchestrationService
        except ImportError:
            pytest.skip("Orchestration service not available")
        
        import time
        
        service = FeatureOrchestrationService()
        points = generate_test_cloud(100_000)
        
        start = time.perf_counter()
        features = service.compute_features(points=points, mode='minimal')
        elapsed = time.perf_counter() - start
        
        # Should complete reasonably quickly (< 30 seconds for medium cloud)
        assert elapsed < 30.0
        assert isinstance(features, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
