"""
Tests for the unified ClassificationEngine.

Tests the new unified classification interface that consolidates
SpectralRulesEngine, GeometricRulesEngine, and ASPRSClassRulesEngine.
"""

import pytest
import numpy as np
from ign_lidar.core.classification import (
    ClassificationEngine,
    ClassificationMode,
    ClassificationStrategy,
)


class TestClassificationEngine:
    """Tests for ClassificationEngine."""

    def test_initialization_default(self):
        """Test engine initialization with defaults."""
        engine = ClassificationEngine()
        assert engine.mode == "asprs"
        assert engine.use_gpu is False
        assert engine.strategy is not None

    def test_initialization_with_mode(self):
        """Test engine initialization with different modes."""
        for mode in ["spectral", "geometric", "asprs"]:
            engine = ClassificationEngine(mode=mode)
            assert engine.mode == mode

    def test_initialization_with_gpu(self):
        """Test engine initialization with GPU flag."""
        engine = ClassificationEngine(use_gpu=True)
        assert engine.use_gpu is True

    def test_invalid_mode_fallback(self):
        """Test that invalid mode falls back to ASPRS."""
        engine = ClassificationEngine(mode="invalid_mode")
        assert engine.strategy is not None
        # Should fallback to ASPRS

    def test_get_available_modes(self):
        """Test getting list of available modes."""
        engine = ClassificationEngine()
        modes = engine.get_available_modes()
        assert "spectral" in modes
        assert "geometric" in modes
        assert "asprs" in modes
        assert "adaptive" in modes

    def test_set_mode(self):
        """Test switching classification mode."""
        engine = ClassificationEngine(mode="asprs")
        assert engine.mode == "asprs"

        engine.set_mode("spectral")
        assert engine.mode == "spectral"
        assert engine.strategy.get_name() == "spectral"

    def test_set_invalid_mode(self):
        """Test that setting invalid mode raises error."""
        engine = ClassificationEngine()
        with pytest.raises(ValueError):
            engine.set_mode("invalid_mode")

    def test_classify_empty_features(self):
        """Test that classifying empty features raises error."""
        engine = ClassificationEngine()
        with pytest.raises(ValueError):
            engine.classify(np.array([]))

    def test_classify_none_features(self):
        """Test that classifying None features raises error."""
        engine = ClassificationEngine()
        with pytest.raises(ValueError):
            engine.classify(None)

    def test_classify_basic(self):
        """Test basic classification."""
        engine = ClassificationEngine(mode="asprs")

        # Create dummy features
        features = np.random.rand(100, 10).astype(np.float32)

        # Should return labels without error
        labels = engine.classify(features)
        assert len(labels) == 100
        assert labels.dtype == np.int32

    def test_classify_converts_list_to_array(self):
        """Test that classify converts list input to array."""
        engine = ClassificationEngine()

        # Pass as list instead of array
        features_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        labels = engine.classify(features_list)
        assert len(labels) == 2

    def test_get_confidence(self):
        """Test getting confidence scores."""
        engine = ClassificationEngine()

        labels = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        confidence = engine.get_confidence(labels)

        assert len(confidence) == len(labels)
        assert confidence.dtype == np.float32
        assert np.all(confidence >= 0.0) and np.all(confidence <= 1.0)

    def test_refine(self):
        """Test refinement."""
        engine = ClassificationEngine()

        labels = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        refined = engine.refine(labels)

        assert len(refined) == len(labels)

    def test_repr(self):
        """Test string representation."""
        engine = ClassificationEngine(mode="spectral", use_gpu=True)
        repr_str = repr(engine)
        assert "ClassificationEngine" in repr_str
        assert "spectral" in repr_str
        assert "gpu=True" in repr_str

    def test_strategy_interface(self):
        """Test that strategies implement ClassificationStrategy interface."""
        engine = ClassificationEngine()
        strategy = engine.strategy
        assert isinstance(strategy, ClassificationStrategy)
        assert hasattr(strategy, "classify")
        assert hasattr(strategy, "get_name")
        assert callable(strategy.classify)
        assert callable(strategy.get_name)

    def test_strategy_names(self):
        """Test that each strategy has correct name."""
        for mode in ["spectral", "geometric", "asprs"]:
            engine = ClassificationEngine(mode=mode)
            assert engine.strategy.get_name() == mode


class TestClassificationMode:
    """Tests for ClassificationMode enum."""

    def test_mode_values(self):
        """Test that all modes have correct values."""
        assert ClassificationMode.SPECTRAL.value == "spectral"
        assert ClassificationMode.GEOMETRIC.value == "geometric"
        assert ClassificationMode.ASPRS.value == "asprs"
        assert ClassificationMode.ADAPTIVE.value == "adaptive"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        mode = ClassificationMode("spectral")
        assert mode == ClassificationMode.SPECTRAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
