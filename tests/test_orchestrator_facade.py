"""
Comprehensive tests for FeatureOrchestrationService facade.

Tests cover:
1. Initialization and lazy loading
2. High-level API methods
3. Feature computation modes
4. Error handling
5. Utility methods
6. Integration scenarios
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf, DictConfig

# Mock the orchestrator before importing facade
import sys
sys.modules['ign_lidar.features.orchestrator'] = MagicMock()

from ign_lidar.features.orchestrator_facade import FeatureOrchestrationService


class TestFeatureOrchestrationServiceInitialization:
    """Test initialization and lazy loading."""

    @pytest.mark.unit
    def test_init_with_config(self):
        """Test service initialization with config."""
        config = OmegaConf.create({"processor": {}, "features": {}})
        service = FeatureOrchestrationService(config)

        assert service.config is config
        assert service.progress_callback is None
        assert service.verbose is False
        assert service._initialized is False
        assert service._orchestrator is None

    @pytest.mark.unit
    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        config = OmegaConf.create({})
        callback = Mock()

        service = FeatureOrchestrationService(
            config, progress_callback=callback, verbose=True
        )

        assert service.progress_callback is callback
        assert service.verbose is True

    @pytest.mark.unit
    def test_lazy_loading_orchestrator(self):
        """Test lazy loading of orchestrator."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        # Before access
        assert service._orchestrator is None
        assert service._initialized is False

        # Mock the orchestrator import
        with patch('ign_lidar.features.orchestrator_facade.FeatureOrchestrator') as MockOrch:
            mock_orch = MagicMock()
            MockOrch.return_value = mock_orch

            # Access orchestrator
            orch = service.orchestrator

            # After access
            assert service._orchestrator is mock_orch
            assert service._initialized is True
            MockOrch.assert_called_once()

    @pytest.mark.unit
    def test_orchestrator_import_error_handling(self):
        """Test handling of import errors."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        with patch('ign_lidar.features.orchestrator_facade.FeatureOrchestrator',
                   side_effect=ImportError("Missing dependency")):
            with pytest.raises(ImportError):
                _ = service.orchestrator


class TestHighLevelAPI:
    """Test high-level API methods."""

    @pytest.fixture
    def service_with_mock_orchestrator(self):
        """Create service with mocked orchestrator."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        # Mock orchestrator
        mock_orch = MagicMock()
        service._orchestrator = mock_orch
        service._initialized = True

        return service, mock_orch

    @pytest.mark.unit
    def test_compute_features_basic(self, service_with_mock_orchestrator):
        """Test basic compute_features call."""
        service, mock_orch = service_with_mock_orchestrator

        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)
        expected_features = {
            "normals": np.random.rand(100, 3),
            "curvature": np.random.rand(100),
        }

        mock_orch.compute_features.return_value = expected_features

        result = service.compute_features(points, classification)

        assert result == expected_features
        mock_orch.compute_features.assert_called_once()
        call_args = mock_orch.compute_features.call_args
        np.testing.assert_array_equal(call_args[1]["points"], points)
        np.testing.assert_array_equal(call_args[1]["classification"], classification)

    @pytest.mark.unit
    def test_compute_features_with_rgb(self, service_with_mock_orchestrator):
        """Test compute_features with RGB data."""
        service, mock_orch = service_with_mock_orchestrator

        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)
        rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        expected_features = {"spectral": np.random.rand(100, 3)}

        mock_orch.compute_features.return_value = expected_features

        result = service.compute_features(points, classification, rgb=rgb)

        assert result == expected_features
        call_args = mock_orch.compute_features.call_args
        np.testing.assert_array_equal(call_args[1]["rgb"], rgb)

    @pytest.mark.unit
    def test_compute_features_with_nir(self, service_with_mock_orchestrator):
        """Test compute_features with NIR data."""
        service, mock_orch = service_with_mock_orchestrator

        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)
        nir = np.random.rand(512, 512)
        expected_features = {"vegetation": np.random.rand(100)}

        mock_orch.compute_features.return_value = expected_features

        result = service.compute_features(points, classification, nir=nir)

        assert result == expected_features
        call_args = mock_orch.compute_features.call_args
        np.testing.assert_array_equal(call_args[1]["nir"], nir)

    @pytest.mark.unit
    def test_compute_with_mode_lod2(self, service_with_mock_orchestrator):
        """Test compute_with_mode with LOD2."""
        service, mock_orch = service_with_mock_orchestrator

        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)
        expected_features = {"feature1": np.array([1, 2, 3])}

        mock_orch.compute_features.return_value = expected_features

        result = service.compute_with_mode(
            points, classification, mode="lod2"
        )

        assert result == expected_features
        call_args = mock_orch.compute_features.call_args
        assert call_args[1]["mode"] == "LOD2"

    @pytest.mark.unit
    def test_compute_with_mode_lod3(self, service_with_mock_orchestrator):
        """Test compute_with_mode with LOD3."""
        service, mock_orch = service_with_mock_orchestrator

        points = np.random.rand(1000, 3)
        classification = np.random.randint(0, 6, 1000)
        expected_features = {"feature1": np.array([1, 2, 3])}

        mock_orch.compute_features.return_value = expected_features

        result = service.compute_with_mode(
            points, classification, mode="lod3", use_gpu=True
        )

        assert result == expected_features
        call_args = mock_orch.compute_features.call_args
        assert call_args[1]["mode"] == "LOD3"
        assert call_args[1]["use_gpu"] is True

    @pytest.mark.unit
    def test_compute_with_mode_all_params(self, service_with_mock_orchestrator):
        """Test compute_with_mode with all parameters."""
        service, mock_orch = service_with_mock_orchestrator

        points = np.random.rand(1000, 3)
        classification = np.random.randint(0, 6, 1000)
        rgb = np.random.rand(512, 512, 3)
        expected_features = {}

        mock_orch.compute_features.return_value = expected_features

        result = service.compute_with_mode(
            points=points,
            classification=classification,
            mode="full",
            use_gpu=True,
            use_rgb=True,
            use_infrared=True,
            k_neighbors=50,
            search_radius=5.0,
            extra_param="custom",
        )

        assert result == expected_features
        call_args = mock_orch.compute_features.call_args
        assert call_args[1]["mode"] == "FULL"
        assert call_args[1]["use_gpu"] is True
        assert call_args[1]["use_rgb"] is True
        assert call_args[1]["use_infrared"] is True
        assert call_args[1]["k_neighbors"] == 50
        assert call_args[1]["search_radius"] == 5.0
        assert call_args[1]["extra_param"] == "custom"


class TestLowLevelAPI:
    """Test low-level API methods."""

    @pytest.mark.unit
    def test_get_orchestrator_direct_access(self):
        """Test direct access to orchestrator."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        mock_orch = MagicMock()
        service._orchestrator = mock_orch
        service._initialized = True

        retrieved_orch = service.get_orchestrator()

        assert retrieved_orch is mock_orch

    @pytest.mark.unit
    def test_get_orchestrator_triggers_lazy_loading(self):
        """Test that get_orchestrator triggers lazy loading if needed."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        mock_orch = MagicMock()
        with patch('ign_lidar.features.orchestrator_facade.FeatureOrchestrator',
                   return_value=mock_orch):
            orch = service.get_orchestrator()

            assert orch is mock_orch
            assert service._initialized is True


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.fixture
    def service_with_mock(self):
        """Service with mock orchestrator."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)
        mock_orch = MagicMock()
        service._orchestrator = mock_orch
        service._initialized = True
        return service, mock_orch

    @pytest.mark.unit
    def test_get_feature_modes(self, service_with_mock):
        """Test getting available feature modes."""
        service, _ = service_with_mock

        modes = service.get_feature_modes()

        assert isinstance(modes, dict)
        assert "minimal" in modes
        assert "lod2" in modes
        assert "lod3" in modes
        assert "asprs" in modes
        assert "full" in modes
        assert len(modes) == 5

    @pytest.mark.unit
    def test_get_optimization_info_full(self, service_with_mock):
        """Test getting optimization info when orchestrator has attributes."""
        service, mock_orch = service_with_mock

        mock_orch.strategy_name = "GPU"
        mock_orch.gpu_available = True
        mock_orch.feature_mode = "LOD2"

        info = service.get_optimization_info()

        assert info["strategy"] == "GPU"
        assert info["gpu_available"] is True
        assert info["feature_mode"] == "LOD2"
        assert info["initialized"] is True

    @pytest.mark.unit
    def test_get_optimization_info_missing_attrs(self, service_with_mock):
        """Test getting optimization info with missing attributes."""
        service, mock_orch = service_with_mock

        # Don't set attributes, use defaults
        info = service.get_optimization_info()

        assert "strategy" in info
        assert "gpu_available" in info
        assert info["initialized"] is True

    @pytest.mark.unit
    def test_get_optimization_info_exception_handling(self, service_with_mock):
        """Test exception handling in get_optimization_info."""
        service, mock_orch = service_with_mock

        # Make orchestrator raise an exception
        mock_orch.strategy_name = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Test error"))
        )

        info = service.get_optimization_info()

        assert isinstance(info, dict)

    @pytest.mark.unit
    def test_clear_cache(self, service_with_mock):
        """Test clearing cache."""
        service, mock_orch = service_with_mock

        service.clear_cache()

        mock_orch.clear_cache.assert_called_once()

    @pytest.mark.unit
    def test_clear_cache_no_orchestrator(self):
        """Test clearing cache when orchestrator not initialized."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        # Should not raise
        service.clear_cache()

    @pytest.mark.unit
    def test_clear_cache_method_not_available(self, service_with_mock):
        """Test clearing cache when method not available."""
        service, mock_orch = service_with_mock

        mock_orch.clear_cache = MagicMock(side_effect=AttributeError)

        # Should not raise
        service.clear_cache()

    @pytest.mark.unit
    def test_get_performance_summary(self, service_with_mock):
        """Test getting performance summary."""
        service, mock_orch = service_with_mock

        expected_summary = {
            "total_time": 1.5,
            "feature_computation_time": 1.2,
        }
        mock_orch.get_performance_summary.return_value = expected_summary

        summary = service.get_performance_summary()

        assert summary == expected_summary

    @pytest.mark.unit
    def test_get_performance_summary_no_orchestrator(self):
        """Test performance summary when orchestrator not initialized."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        summary = service.get_performance_summary()

        assert summary is None

    @pytest.mark.unit
    def test_get_performance_summary_method_not_available(self, service_with_mock):
        """Test performance summary when method not available."""
        service, mock_orch = service_with_mock

        mock_orch.get_performance_summary = MagicMock(side_effect=AttributeError)

        summary = service.get_performance_summary()

        assert summary is None


class TestErrorHandling:
    """Test error handling."""

    @pytest.fixture
    def service_with_mock_orchestrator(self):
        """Service with mock."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)
        mock_orch = MagicMock()
        service._orchestrator = mock_orch
        service._initialized = True
        return service, mock_orch

    @pytest.mark.unit
    def test_compute_features_exception(self, service_with_mock_orchestrator):
        """Test exception handling in compute_features."""
        service, mock_orch = service_with_mock_orchestrator

        mock_orch.compute_features.side_effect = RuntimeError("GPU OOM")

        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)

        with pytest.raises(RuntimeError, match="GPU OOM"):
            service.compute_features(points, classification)

    @pytest.mark.unit
    def test_compute_with_mode_exception(self, service_with_mock_orchestrator):
        """Test exception handling in compute_with_mode."""
        service, mock_orch = service_with_mock_orchestrator

        mock_orch.compute_features.side_effect = ValueError("Invalid mode")

        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)

        with pytest.raises(ValueError):
            service.compute_with_mode(
                points, classification, mode="invalid"
            )


class TestStringRepresentation:
    """Test string representation."""

    @pytest.mark.unit
    def test_repr_before_initialization(self):
        """Test __repr__ before initialization."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)

        repr_str = repr(service)

        assert "FeatureOrchestrationService" in repr_str
        assert "lazy" in repr_str

    @pytest.mark.unit
    def test_repr_after_initialization(self):
        """Test __repr__ after initialization."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)
        service._orchestrator = MagicMock()
        service._initialized = True

        repr_str = repr(service)

        assert "FeatureOrchestrationService" in repr_str
        assert "initialized" in repr_str


class TestIntegration:
    """Integration tests."""

    @pytest.mark.unit
    def test_workflow_compute_features_then_clear(self):
        """Test typical workflow: compute then clear."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)
        mock_orch = MagicMock()
        service._orchestrator = mock_orch
        service._initialized = True

        # Set up mock
        mock_orch.compute_features.return_value = {"feat": np.array([1, 2, 3])}

        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)

        # Compute
        result = service.compute_features(points, classification)
        assert "feat" in result

        # Clear
        service.clear_cache()
        mock_orch.clear_cache.assert_called_once()

    @pytest.mark.unit
    def test_workflow_get_info_then_compute(self):
        """Test workflow: check info then compute."""
        config = OmegaConf.create({})
        service = FeatureOrchestrationService(config)
        mock_orch = MagicMock()
        service._orchestrator = mock_orch
        service._initialized = True

        # Get info
        mock_orch.strategy_name = "GPU"
        info = service.get_optimization_info()
        assert info["strategy"] == "GPU"

        # Compute with appropriate mode
        mock_orch.compute_features.return_value = {}
        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 6, 100)

        service.compute_with_mode(
            points, classification, mode="lod3", use_gpu=True
        )

        mock_orch.compute_features.assert_called_once()


class TestProgressCallback:
    """Test progress callback functionality."""

    @pytest.mark.unit
    def test_progress_callback_stored(self):
        """Test that progress callback is stored."""
        config = OmegaConf.create({})
        callback = Mock()

        service = FeatureOrchestrationService(config, progress_callback=callback)

        assert service.progress_callback is callback

    @pytest.mark.unit
    def test_progress_callback_passed_to_orchestrator(self):
        """Test that callback is passed to orchestrator."""
        config = OmegaConf.create({})
        callback = Mock()
        service = FeatureOrchestrationService(config, progress_callback=callback)

        with patch('ign_lidar.features.orchestrator_facade.FeatureOrchestrator') as MockOrch:
            mock_orch = MagicMock()
            MockOrch.return_value = mock_orch

            _ = service.orchestrator

            MockOrch.assert_called_once_with(config, callback)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
