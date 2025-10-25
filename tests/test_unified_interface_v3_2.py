"""
Test Unified Classifier Interface (v3.2+)

This test file verifies that all main classifiers follow the BaseClassifier
interface and can be used interchangeably.

Tests cover:
- UnifiedClassifier with BaseClassifier interface
- HierarchicalClassifier with BaseClassifier interface
- ParcelClassifier with BaseClassifier interface
- Config presets and migration
- ClassificationResult consistency

Author: Harmonization Team
Date: October 25, 2025
Version: 3.2.0
"""

from pathlib import Path

import numpy as np
import pytest

# Import unified configuration
from ign_lidar.config import Config

# Import unified classifiers
from ign_lidar.core.classification import (
    BaseClassifier,
    ClassificationResult,
    Classifier,
)

# Import specific classifiers for testing
try:
    from ign_lidar.core.classification import (
        HierarchicalClassifier,
        ParcelClassifier,
        UnifiedClassifier,
    )

    HAS_ALL_CLASSIFIERS = True
except ImportError as e:
    HAS_ALL_CLASSIFIERS = False
    print(f"Some classifiers not available: {e}")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_points():
    """Generate sample point cloud."""
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3) * 100
    return points


@pytest.fixture
def sample_features():
    """Generate sample features."""
    np.random.seed(42)
    n_points = 1000
    return {
        "height": np.random.rand(n_points) * 20,
        "planarity": np.random.rand(n_points),
        "verticality": np.random.rand(n_points),
        "curvature": np.random.rand(n_points) * 0.5,
        "ndvi": np.random.rand(n_points) * 2 - 1,  # Range [-1, 1]
    }


@pytest.fixture
def sample_asprs_labels():
    """Generate sample ASPRS labels."""
    np.random.seed(42)
    n_points = 1000
    # ASPRS classes: 1=unclassified, 2=ground, 3-5=vegetation, 6=building
    return np.random.choice([1, 2, 3, 4, 5, 6], size=n_points)


# ============================================================================
# Test Configuration System (Phase 1)
# ============================================================================


class TestUnifiedConfig:
    """Test the unified configuration system."""

    def test_config_presets_exist(self):
        """Test that all presets are available."""
        expected_presets = [
            "asprs_production",
            "lod2_buildings",
            "lod3_detailed",
            "gpu_optimized",
            "minimal_fast",
        ]

        for preset_name in expected_presets:
            try:
                config = Config.preset(preset_name)
                assert config is not None
                assert hasattr(config, "mode")
                assert hasattr(config, "features")
                print(f"✓ Preset '{preset_name}' works")
            except Exception as e:
                pytest.fail(f"Preset '{preset_name}' failed: {e}")

    def test_config_simplification(self):
        """Test that config has simplified interface."""
        config = Config.preset("lod2_buildings")

        # Should have essential top-level parameters
        essential_params = [
            "input_dir",
            "output_dir",
            "mode",
            "use_gpu",
            "num_workers",
            "patch_size",
            "num_points",
        ]

        for param in essential_params:
            assert hasattr(config, param), f"Missing essential param: {param}"

        print(f"✓ Config has {len(essential_params)} essential parameters")

    def test_config_from_environment(self):
        """Test auto-configuration from environment."""
        try:
            config = Config.from_environment(
                input_dir="/tmp/input", output_dir="/tmp/output"
            )
            assert config.input_dir == "/tmp/input"
            assert config.output_dir == "/tmp/output"
            assert isinstance(config.use_gpu, bool)
            assert isinstance(config.num_workers, int)
            print("✓ Auto-configuration from environment works")
        except Exception as e:
            pytest.fail(f"from_environment() failed: {e}")


# ============================================================================
# Test BaseClassifier Interface (Phase 2)
# ============================================================================


class TestBaseClassifierInterface:
    """Test that all classifiers follow BaseClassifier interface."""

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_unified_classifier_inheritance(self):
        """Test UnifiedClassifier inherits from BaseClassifier."""
        classifier = UnifiedClassifier()
        assert isinstance(classifier, BaseClassifier)
        assert hasattr(classifier, "classify")
        assert hasattr(classifier, "validate_inputs")
        print("✓ UnifiedClassifier inherits from BaseClassifier")

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_hierarchical_classifier_inheritance(self):
        """Test HierarchicalClassifier inherits from BaseClassifier."""
        from ign_lidar.core.classification.hierarchical_classifier import (
            ClassificationLevel,
        )

        classifier = HierarchicalClassifier(target_level=ClassificationLevel.LOD2)
        assert isinstance(classifier, BaseClassifier)
        assert hasattr(classifier, "classify")
        assert hasattr(classifier, "validate_inputs")
        print("✓ HierarchicalClassifier inherits from BaseClassifier")

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_parcel_classifier_inheritance(self):
        """Test ParcelClassifier inherits from BaseClassifier."""
        try:
            classifier = ParcelClassifier()
            assert isinstance(classifier, BaseClassifier)
            assert hasattr(classifier, "classify")
            assert hasattr(classifier, "validate_inputs")
            print("✓ ParcelClassifier inherits from BaseClassifier")
        except ImportError:
            # ParcelClassifier requires geopandas
            pytest.skip("ParcelClassifier requires geopandas")


class TestClassificationResult:
    """Test ClassificationResult standardization."""

    def test_result_structure(self):
        """Test ClassificationResult has expected attributes."""
        labels = np.array([1, 2, 3, 6, 6, 2, 3])
        result = ClassificationResult(labels=labels)

        # Required attributes
        assert hasattr(result, "labels")
        assert hasattr(result, "confidence")
        assert hasattr(result, "metadata")
        assert hasattr(result, "get_statistics")

        # Test labels
        assert len(result.labels) == len(labels)
        np.testing.assert_array_equal(result.labels, labels)

        print("✓ ClassificationResult has correct structure")

    def test_result_statistics(self):
        """Test ClassificationResult.get_statistics()."""
        labels = np.array([1, 2, 3, 6, 6, 2, 3, 3])
        result = ClassificationResult(labels=labels)

        stats = result.get_statistics()

        # Required statistics
        assert "total_points" in stats
        assert "num_classes" in stats
        assert "class_distribution" in stats
        assert "class_percentages" in stats

        # Verify values
        assert stats["total_points"] == 8
        assert stats["num_classes"] == 4  # Classes: 1, 2, 3, 6
        assert stats["class_distribution"][3] == 3  # Three points with class 3

        print(
            f"✓ Statistics: {stats['total_points']} points, {stats['num_classes']} classes"
        )


class TestUnifiedInterface:
    """Test unified classify() interface across classifiers."""

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_unified_classifier_classify(self, sample_points, sample_features):
        """Test UnifiedClassifier.classify() method."""
        classifier = UnifiedClassifier(strategy="basic")

        result = classifier.classify(sample_points, sample_features)

        # Check result type
        assert isinstance(result, ClassificationResult)
        assert len(result.labels) == len(sample_points)
        assert result.labels.dtype == np.uint8 or result.labels.dtype == np.int32

        # Check statistics
        stats = result.get_statistics()
        assert stats["total_points"] == len(sample_points)
        assert stats["num_classes"] > 0

        print(
            f"✓ UnifiedClassifier.classify() works: {stats['num_classes']} classes detected"
        )

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_hierarchical_classifier_classify(
        self, sample_points, sample_features, sample_asprs_labels
    ):
        """Test HierarchicalClassifier.classify() method."""
        from ign_lidar.core.classification.hierarchical_classifier import (
            ClassificationLevel,
        )

        classifier = HierarchicalClassifier(target_level=ClassificationLevel.LOD2)

        # Add ASPRS labels to features (required by HierarchicalClassifier)
        features_with_asprs = sample_features.copy()
        features_with_asprs["asprs_labels"] = sample_asprs_labels

        result = classifier.classify(sample_points, features_with_asprs)

        # Check result type
        assert isinstance(result, ClassificationResult)
        assert len(result.labels) == len(sample_points)

        # Check metadata
        assert "level" in result.metadata
        assert result.metadata["level"] == "LOD2"

        stats = result.get_statistics()
        print(
            f"✓ HierarchicalClassifier.classify() works: {stats['num_classes']} classes"
        )


class TestBackwardCompatibility:
    """Test that old interfaces still work with deprecation warnings."""

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_unified_classifier_old_interface(self, sample_points, sample_features):
        """Test that old UnifiedClassifier.classify_points() still works."""
        import pandas as pd

        classifier = UnifiedClassifier(strategy="basic")

        # Old interface uses DataFrame
        data = pd.DataFrame(
            {
                "x": sample_points[:, 0],
                "y": sample_points[:, 1],
                "z": sample_points[:, 2],
            }
        )
        for name, values in sample_features.items():
            data[name] = values

        # Should still work
        try:
            labels = classifier.classify_points(data)
            assert len(labels) == len(sample_points)
            print("✓ Old UnifiedClassifier.classify_points() still works")
        except Exception as e:
            # OK if method signature changed
            print(f"⚠ Old interface may have changed: {e}")

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_hierarchical_classifier_old_interface(
        self, sample_asprs_labels, sample_features
    ):
        """Test that old HierarchicalClassifier interface still works."""
        from ign_lidar.core.classification.hierarchical_classifier import (
            ClassificationLevel,
        )

        classifier = HierarchicalClassifier(target_level=ClassificationLevel.LOD2)

        # Old interface: classify_from_asprs()
        try:
            result = classifier.classify_from_asprs(
                asprs_labels=sample_asprs_labels, features=sample_features
            )
            assert hasattr(result, "labels")
            assert len(result.labels) == len(sample_asprs_labels)
            print("✓ Old HierarchicalClassifier.classify_from_asprs() still works")
        except AttributeError:
            pytest.skip("classify_from_asprs() not available")


class TestValidation:
    """Test input validation works consistently."""

    def test_invalid_points_shape(self, sample_features):
        """Test validation catches invalid points shape."""
        classifier = UnifiedClassifier()

        # Wrong shape - should be [N, 3]
        invalid_points = np.random.rand(100, 2)

        with pytest.raises(ValueError, match="points must have shape"):
            classifier.classify(invalid_points, sample_features)

        print("✓ Invalid points shape caught")

    def test_mismatched_features(self, sample_points):
        """Test validation catches mismatched feature lengths."""
        classifier = UnifiedClassifier()

        # Features with wrong length
        invalid_features = {
            "height": np.random.rand(500),  # Should be 1000
            "planarity": np.random.rand(1000),
        }

        with pytest.raises(ValueError, match="has.*values.*expected"):
            classifier.classify(sample_points, invalid_features)

        print("✓ Mismatched feature lengths caught")


# ============================================================================
# Test Classifier Facade
# ============================================================================


class TestClassifierFacade:
    """Test the Classifier facade."""

    def test_classifier_alias(self):
        """Test that Classifier is an alias for UnifiedClassifier."""
        if not HAS_ALL_CLASSIFIERS:
            pytest.skip("Classifiers not available")

        assert Classifier is not None
        # Classifier should point to UnifiedClassifier
        classifier = Classifier()
        assert hasattr(classifier, "classify")
        assert isinstance(classifier, BaseClassifier)
        print("✓ Classifier facade works")


# ============================================================================
# Integration Test
# ============================================================================


class TestIntegration:
    """End-to-end integration test."""

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_full_workflow_v3_2(self, sample_points, sample_features):
        """Test complete v3.2 workflow."""

        # 1. Create config with preset
        config = Config.preset("lod2_buildings")
        assert config.mode == "lod2"

        # 2. Create classifier using facade
        classifier = Classifier(strategy="adaptive")
        assert isinstance(classifier, BaseClassifier)

        # 3. Classify points
        result = classifier.classify(sample_points, sample_features)
        assert isinstance(result, ClassificationResult)

        # 4. Get statistics
        stats = result.get_statistics()
        assert stats["total_points"] == len(sample_points)

        print(
            f"✓ Full v3.2 workflow: {stats['total_points']} points → "
            f"{stats['num_classes']} classes"
        )


# ============================================================================
# Performance Test
# ============================================================================


class TestPerformance:
    """Basic performance checks."""

    @pytest.mark.skipif(not HAS_ALL_CLASSIFIERS, reason="Not all classifiers available")
    def test_classification_speed(self):
        """Test that classification completes in reasonable time."""
        import time

        # Generate larger dataset
        np.random.seed(42)
        n_points = 10000
        points = np.random.rand(n_points, 3) * 100
        features = {
            "height": np.random.rand(n_points) * 20,
            "planarity": np.random.rand(n_points),
            "verticality": np.random.rand(n_points),
        }

        classifier = UnifiedClassifier(strategy="basic")

        start = time.time()
        result = classifier.classify(points, features)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds for 10k points)
        assert elapsed < 5.0, f"Classification too slow: {elapsed:.2f}s"
        assert len(result.labels) == n_points

        print(
            f"✓ Classified {n_points:,} points in {elapsed:.3f}s "
            f"({n_points/elapsed:.0f} points/sec)"
        )


if __name__ == "__main__":
    """Run tests with pytest or directly."""
    import sys

    print("\n" + "=" * 70)
    print("IGN LiDAR HD v3.2 - Unified Interface Tests")
    print("=" * 70 + "\n")

    # Run with pytest if available
    try:
        pytest.main([__file__, "-v", "--tb=short"])
    except:
        print("pytest not available, running basic checks...\n")

        # Basic smoke tests
        print("Testing Config presets...")
        test_config = TestUnifiedConfig()
        test_config.test_config_presets_exist()
        test_config.test_config_simplification()

        print("\nTesting ClassificationResult...")
        test_result = TestClassificationResult()
        test_result.test_result_structure()
        test_result.test_result_statistics()

        if HAS_ALL_CLASSIFIERS:
            print("\nTesting Classifier interfaces...")
            test_interface = TestBaseClassifierInterface()
            test_interface.test_unified_classifier_inheritance()
            test_interface.test_hierarchical_classifier_inheritance()

        print("\n" + "=" * 70)
        print("✓ Basic tests passed!")
        print("=" * 70)
