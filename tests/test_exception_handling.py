"""
Test suite for improved exception handling in IGN LiDAR HD.

Tests custom exception types and their usage in key modules.

Author: Task 10 - Exception Handling Improvements
Date: 2025-11-21
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ign_lidar.core.error_handler import (
    ProcessingError,
    GPUMemoryError,
    GPUNotAvailableError,
    MemoryPressureError,
    FileProcessingError,
    ConfigurationError,
    FeatureComputationError,
    CacheError,
    DataFetchError,
    InitializationError,
)


class TestProcessingError:
    """Test base ProcessingError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ProcessingError(message="Test error")
        assert "Test error" in str(error)
        assert "ERROR: Test error" in str(error)

    def test_error_with_context(self):
        """Test error with context information."""
        error = ProcessingError(
            message="Test error",
            context={"file": "test.laz", "size": "1.5GB"}
        )
        error_str = str(error)
        assert "file: test.laz" in error_str
        assert "size: 1.5GB" in error_str

    def test_error_with_suggestions(self):
        """Test error with suggestions."""
        error = ProcessingError(
            message="Test error",
            suggestions=["Try option 1", "Try option 2"]
        )
        error_str = str(error)
        assert "Try option 1" in error_str
        assert "Try option 2" in error_str


class TestGPUMemoryError:
    """Test GPU memory error handling."""

    def test_from_cuda_error(self):
        """Test creating error from CUDA exception."""
        cuda_error = RuntimeError("CUDA out of memory")
        
        error = GPUMemoryError.from_cuda_error(
            error=cuda_error,
            current_vram_gb=7.5,
            total_vram_gb=8.0,
            chunk_size=1000000,
            num_points=5000000
        )
        
        error_str = str(error)
        assert "out of memory" in error_str.lower()
        assert "7.5GB" in error_str
        assert "1,000,000" in error_str
        assert "Reduce chunk size" in error_str

    def test_without_context(self):
        """Test GPU error without detailed context."""
        cuda_error = RuntimeError("CUDA error")
        
        error = GPUMemoryError.from_cuda_error(error=cuda_error)
        
        assert "GPU out of memory" in str(error)


class TestGPUNotAvailableError:
    """Test GPU not available error."""

    def test_create_gpu_error(self):
        """Test GPU not available error creation."""
        error = GPUNotAvailableError.create(reason="CuPy not installed")
        
        error_str = str(error)
        assert "GPU requested but not available" in error_str
        assert "CuPy not installed" in error_str
        assert "Install CuPy" in error_str


class TestFeatureComputationError:
    """Test feature computation error."""

    def test_create_feature_error(self):
        """Test feature computation error creation."""
        original_error = ValueError("Invalid k_neighbors value")
        
        error = FeatureComputationError.create(
            feature_name="normals",
            error=original_error,
            num_points=10000,
            stage="neighborhood search"
        )
        
        error_str = str(error)
        assert "normals" in error_str
        assert "10,000" in error_str
        assert "neighborhood search" in error_str
        assert "Invalid k_neighbors" in error_str

    def test_memory_error_suggestion(self):
        """Test feature error includes memory-specific suggestions."""
        original_error = MemoryError("Out of memory")
        
        error = FeatureComputationError.create(
            feature_name="geometric_features",
            error=original_error,
            num_points=1000000
        )
        
        error_str = str(error)
        assert "Reduce chunk size" in error_str or "GPU chunked mode" in error_str


class TestDataFetchError:
    """Test data fetching error."""

    def test_create_data_fetch_error(self):
        """Test data fetch error creation."""
        original_error = ConnectionError("Network unreachable")
        
        error = DataFetchError.create(
            data_type="RGB orthophoto",
            error=original_error,
            url="https://example.com/tile.tif",
            retry_count=3
        )
        
        error_str = str(error)
        assert "RGB orthophoto" in error_str
        assert "Network unreachable" in error_str
        assert "example.com" in error_str
        assert "3" in error_str  # retry count

    def test_timeout_specific_suggestions(self):
        """Test timeout error includes timeout-specific suggestions."""
        original_error = TimeoutError("Connection timed out")
        
        error = DataFetchError.create(
            data_type="WFS ground truth",
            error=original_error
        )
        
        error_str = str(error)
        assert "timeout" in error_str.lower()
        assert "Increase timeout" in error_str

    def test_404_specific_suggestions(self):
        """Test 404 error includes data availability suggestions."""
        original_error = Exception("HTTP 404 Not Found")
        
        error = DataFetchError.create(
            data_type="NIR infrared",
            error=original_error
        )
        
        error_str = str(error)
        assert "404" in error_str or "not found" in error_str.lower()
        assert "data is available" in error_str.lower()


class TestInitializationError:
    """Test component initialization error."""

    def test_create_init_error(self):
        """Test initialization error creation."""
        original_error = ImportError("No module named 'requests'")
        
        error = InitializationError.create(
            component="RGB fetcher",
            error=original_error,
            dependencies=["requests", "Pillow"]
        )
        
        error_str = str(error)
        assert "RGB fetcher" in error_str
        assert "requests" in error_str
        assert "Pillow" in error_str
        assert "pip install" in error_str

    def test_import_error_suggestions(self):
        """Test import error includes installation suggestions."""
        original_error = ImportError("No module named 'cupy'")
        
        error = InitializationError.create(
            component="GPU processor",
            error=original_error,
            dependencies=["cupy"]
        )
        
        error_str = str(error)
        assert "pip install cupy" in error_str


class TestCacheError:
    """Test cache error handling."""

    def test_create_cache_error(self):
        """Test cache error creation."""
        original_error = OSError("Permission denied")
        
        error = CacheError.create(
            cache_type="feature cache",
            operation="write",
            error=original_error,
            cache_path="/tmp/cache"
        )
        
        error_str = str(error)
        assert "feature cache" in error_str
        assert "write" in error_str
        assert "Permission denied" in error_str
        assert "/tmp/cache" in error_str

    def test_permission_specific_suggestions(self):
        """Test permission error includes chmod suggestions."""
        original_error = PermissionError("Permission denied")
        
        error = CacheError.create(
            cache_type="RGB cache",
            operation="create directory",
            error=original_error
        )
        
        error_str = str(error)
        assert "chmod" in error_str.lower()


class TestConfigurationError:
    """Test configuration error."""

    def test_create_config_error(self):
        """Test configuration error creation."""
        error = ConfigurationError.create(
            parameter="k_neighbors",
            value=-5,
            reason="Must be positive integer",
            valid_range="1-100"
        )
        
        error_str = str(error)
        assert "k_neighbors" in error_str
        assert "-5" in error_str
        assert "positive integer" in error_str
        assert "1-100" in error_str


class TestFileProcessingError:
    """Test file processing error."""

    def test_create_file_error(self):
        """Test file processing error creation."""
        original_error = ValueError("Invalid LAZ format")
        
        error = FileProcessingError.create(
            file_path="/data/tile_001.laz",
            error=original_error,
            stage="reading header"
        )
        
        error_str = str(error)
        assert "tile_001.laz" in error_str
        assert "reading header" in error_str
        assert "Invalid LAZ format" in error_str

    def test_corrupt_file_suggestions(self):
        """Test corrupt file error includes redownload suggestion."""
        original_error = Exception("Corrupt file header")
        
        error = FileProcessingError.create(
            file_path="/data/tile_001.laz",
            error=original_error,
            stage="validation"
        )
        
        error_str = str(error)
        assert "corrupt" in error_str.lower()
        assert "redownload" in error_str.lower()


class TestMemoryPressureError:
    """Test memory pressure error."""

    def test_create_memory_pressure_error(self):
        """Test memory pressure error creation."""
        error = MemoryPressureError.create(
            available_ram_gb=2.5,
            swap_used_percent=85.0,
            required_ram_gb=8.0
        )
        
        error_str = str(error)
        assert "2.5GB" in error_str
        assert "85" in error_str
        assert "8.0GB" in error_str
        assert "5.5GB" in error_str  # deficit


@pytest.mark.integration
class TestExceptionHandlingIntegration:
    """Integration tests for exception handling in orchestrator."""

    def test_orchestrator_handles_missing_cupy(self):
        """Test orchestrator gracefully handles missing CuPy."""
        from omegaconf import OmegaConf
        from ign_lidar.features.orchestrator import FeatureOrchestrator
        
        # Create config with GPU enabled
        config = OmegaConf.create({
            "processor": {"use_gpu": True},
            "features": {
                "k_neighbors": 30,
                "search_radius": 3.0,
                "feature_mode": "lod2"
            }
        })
        
        # Mock GPU_AVAILABLE to False
        with patch("ign_lidar.features.gpu_processor.GPU_AVAILABLE", False):
            # Should not crash, just warn and use CPU
            orchestrator = FeatureOrchestrator(config)
            assert orchestrator is not None

    def test_orchestrator_handles_invalid_config(self):
        """Test orchestrator handles invalid configuration."""
        from omegaconf import OmegaConf
        from ign_lidar.features.orchestrator import FeatureOrchestrator
        
        # Create config with invalid values
        config = OmegaConf.create({
            "processor": {},
            "features": {
                "k_neighbors": -10,  # Invalid
                "feature_mode": "lod2"
            }
        })
        
        # Should handle gracefully (may warn or use defaults)
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator is not None


@pytest.mark.unit
class TestErrorFormattingConsistency:
    """Test that all error types format consistently."""

    def test_all_errors_have_separator_lines(self):
        """Test all custom errors include separator lines."""
        errors = [
            ProcessingError("Test"),
            GPUMemoryError.from_cuda_error(RuntimeError("Test")),
            GPUNotAvailableError.create("Test"),
            FeatureComputationError.create("test_feature", ValueError("Test")),
            DataFetchError.create("test_data", ConnectionError("Test")),
            InitializationError.create("test_component", ImportError("Test")),
            CacheError.create("test_cache", "read", OSError("Test")),
            ConfigurationError.create("test_param", "test_value", "Test"),
            FileProcessingError.create("test.laz", ValueError("Test"), "test"),
            MemoryPressureError.create(1.0, 50.0),
        ]
        
        for error in errors:
            error_str = str(error)
            # All errors should have separator lines
            assert "=" * 70 in error_str
            # All errors should have ERROR: prefix
            assert "ERROR:" in error_str

    def test_all_errors_have_suggestions(self):
        """Test all custom errors provide actionable suggestions."""
        errors = [
            GPUMemoryError.from_cuda_error(RuntimeError("Test")),
            GPUNotAvailableError.create("Test"),
            FeatureComputationError.create("test_feature", ValueError("Test")),
            DataFetchError.create("test_data", ConnectionError("Test")),
            InitializationError.create("test_component", ImportError("Test")),
            CacheError.create("test_cache", "read", OSError("Test")),
            ConfigurationError.create("test_param", "test_value", "Test"),
            FileProcessingError.create("test.laz", ValueError("Test"), "test"),
            MemoryPressureError.create(1.0, 50.0),
        ]
        
        for error in errors:
            error_str = str(error)
            # All should have Suggestions section
            assert "Suggestions:" in error_str
            # At least one numbered suggestion
            assert "1." in error_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
