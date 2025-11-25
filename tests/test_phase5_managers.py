"""
Comprehensive tests for Phase 5 unified managers.

Tests cover:
- GPU Stream Manager
- Performance Manager
- Configuration Validator
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from ign_lidar.core.gpu_stream_manager import GPUStreamManager, GPUStream, StreamConfig
from ign_lidar.core.performance_manager import PerformanceManager, PhaseMetrics
from ign_lidar.core.config_validator import ConfigValidator, ValidationReport, ValidationLevel


# ============================================================================
# GPU Stream Manager Tests
# ============================================================================

class TestGPUStreamManagerInitialization:
    """Test GPU Stream Manager initialization."""

    @pytest.mark.unit
    def test_singleton_pattern(self):
        """Test singleton pattern."""
        manager1 = GPUStreamManager()
        manager2 = GPUStreamManager()
        assert manager1 is manager2

    @pytest.mark.unit
    def test_initialization(self):
        """Test manager initialization."""
        manager = GPUStreamManager()
        assert manager.streams is not None
        assert len(manager.streams) > 0
        assert manager.config is not None

    @pytest.mark.unit
    def test_stream_count(self):
        """Test stream count."""
        manager = GPUStreamManager()
        count = manager.get_stream_count()
        assert count == manager.config.pool_size


class TestGPUStreamManagerHighLevel:
    """Test high-level GPU Stream Manager API."""

    @pytest.mark.unit
    def test_async_transfer_basic(self):
        """Test basic async transfer."""
        import numpy as np

        manager = GPUStreamManager()
        src = np.random.rand(100, 3)
        dst = np.zeros((100, 3))

        result = manager.async_transfer(src, dst)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_wait_all(self):
        """Test wait_all synchronization."""
        manager = GPUStreamManager()
        result = manager.wait_all()
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_batch_transfers(self):
        """Test batch transfer."""
        import numpy as np

        manager = GPUStreamManager()
        transfers = [
            (np.random.rand(100, 3), np.zeros((100, 3)))
            for _ in range(3)
        ]

        result = manager.batch_transfers(transfers)
        assert isinstance(result, bool)


class TestGPUStreamManagerLowLevel:
    """Test low-level GPU Stream Manager API."""

    @pytest.mark.unit
    def test_get_stream(self):
        """Test getting stream."""
        manager = GPUStreamManager()
        stream = manager.get_stream()
        assert isinstance(stream, GPUStream)

    @pytest.mark.unit
    def test_get_stream_by_id(self):
        """Test getting stream by ID."""
        manager = GPUStreamManager()
        stream = manager.get_stream(0)
        assert isinstance(stream, GPUStream)
        assert stream.stream_id == 0

    @pytest.mark.unit
    def test_get_performance_stats(self):
        """Test getting performance stats."""
        manager = GPUStreamManager()
        stats = manager.get_performance_stats()
        assert "total_streams" in stats
        assert "transfer_stats" in stats


# ============================================================================
# Performance Manager Tests
# ============================================================================

class TestPerformanceManagerInitialization:
    """Test Performance Manager initialization."""

    @pytest.mark.unit
    def test_singleton_pattern(self):
        """Test singleton pattern."""
        manager1 = PerformanceManager()
        manager2 = PerformanceManager()
        assert manager1 is manager2

    @pytest.mark.unit
    def test_initialization(self):
        """Test manager initialization."""
        manager = PerformanceManager()
        assert manager.phases is not None
        assert manager.config is not None


class TestPerformanceManagerHighLevel:
    """Test high-level Performance Manager API."""

    @pytest.mark.unit
    def test_start_end_phase(self):
        """Test starting and ending a phase."""
        manager = PerformanceManager()
        manager.reset()

        manager.start_phase("test_phase")
        assert manager.current_phase == "test_phase"

        time.sleep(0.01)
        metrics = manager.end_phase()

        assert "duration" in metrics
        assert metrics["duration"] > 0

    @pytest.mark.unit
    def test_get_summary(self):
        """Test getting performance summary."""
        manager = PerformanceManager()
        manager.reset()

        manager.start_phase("phase1")
        time.sleep(0.01)
        manager.end_phase()

        summary = manager.get_summary()
        assert "total_time" in summary
        assert "phases" in summary

    @pytest.mark.unit
    def test_record_metric(self):
        """Test recording custom metric."""
        manager = PerformanceManager()
        manager.reset()

        manager.start_phase("test")
        manager.record_metric("accuracy", 0.95)
        manager.end_phase()

        stats = manager.get_phase_stats("test")
        assert "custom_metrics" in stats


class TestPerformanceManagerLowLevel:
    """Test low-level Performance Manager API."""

    @pytest.mark.unit
    def test_get_metric_stats(self):
        """Test getting metric statistics."""
        manager = PerformanceManager()
        manager.reset()

        manager.start_phase("test")
        for i in range(5):
            manager.record_metric("value", float(i))
        manager.end_phase()

        stats = manager.get_metric_stats("value")
        assert "mean" in stats
        assert "count" in stats

    @pytest.mark.unit
    def test_configure(self):
        """Test configuration."""
        manager = PerformanceManager()
        manager.configure(track_memory=False, verbose=True)

        assert manager.config.track_memory is False
        assert manager.config.verbose is True


# ============================================================================
# Configuration Validator Tests
# ============================================================================

class TestConfigValidatorInitialization:
    """Test Configuration Validator initialization."""

    @pytest.mark.unit
    def test_singleton_pattern(self):
        """Test singleton pattern."""
        validator1 = ConfigValidator()
        validator2 = ConfigValidator()
        assert validator1 is validator2

    @pytest.mark.unit
    def test_initialization(self):
        """Test validator initialization."""
        validator = ConfigValidator()
        assert validator.rules is not None
        assert validator.config is not None


class TestConfigValidatorHighLevel:
    """Test high-level Configuration Validator API."""

    @pytest.mark.unit
    def test_validate_simple(self):
        """Test simple validation."""
        validator = ConfigValidator()
        validator.clear_rules()

        config = {"field1": "value1"}
        is_valid, errors = validator.validate(config)

        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)

    @pytest.mark.unit
    def test_validate_detailed(self):
        """Test detailed validation."""
        validator = ConfigValidator()
        validator.clear_rules()

        config = {"field1": "value1"}
        report = validator.validate_detailed(config)

        assert isinstance(report, ValidationReport)
        assert hasattr(report, "errors")
        assert hasattr(report, "warnings")

    @pytest.mark.unit
    def test_validate_required_field(self):
        """Test required field validation."""
        validator = ConfigValidator()
        validator.clear_rules()
        validator.add_required_field("required_field")

        config = {"other_field": "value"}
        is_valid, errors = validator.validate(config)

        assert not is_valid
        assert len(errors) > 0


class TestConfigValidatorLowLevel:
    """Test low-level Configuration Validator API."""

    @pytest.mark.unit
    def test_add_rule(self):
        """Test adding custom rule."""
        validator = ConfigValidator()
        validator.clear_rules()

        rule = lambda x: x > 0
        validator.add_rule("positive_field", rule)

        assert "positive_field" in validator.rules

    @pytest.mark.unit
    def test_add_field_type(self):
        """Test adding field type."""
        validator = ConfigValidator()
        validator.clear_rules()

        validator.add_field_type("numeric_field", int)

        assert "numeric_field" in validator.field_types

    @pytest.mark.unit
    def test_add_lod_validator(self):
        """Test LOD validator."""
        validator = ConfigValidator()
        validator.clear_rules()
        validator.add_lod_validator()

        config = {"lod_level": "LOD2"}
        is_valid, errors = validator.validate(config)

        assert is_valid

    @pytest.mark.unit
    def test_add_numeric_range_validator(self):
        """Test numeric range validator."""
        validator = ConfigValidator()
        validator.clear_rules()
        validator.add_numeric_range_validator("batch_size", 1, 1000)

        config = {"batch_size": 500}
        is_valid, errors = validator.validate(config)

        assert is_valid


class TestConfigValidatorConfiguration:
    """Test Configuration Validator configuration."""

    @pytest.mark.unit
    def test_strict_mode(self):
        """Test strict mode."""
        validator = ConfigValidator()
        validator.clear_rules()
        validator.configure(strict_mode=True)

        assert validator.config.strict_mode is True

    @pytest.mark.unit
    def test_allow_unknown_fields(self):
        """Test allow unknown fields."""
        validator = ConfigValidator()
        validator.clear_rules()
        validator.configure(allow_unknown_fields=True)

        assert validator.config.allow_unknown_fields is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
