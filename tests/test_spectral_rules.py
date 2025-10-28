"""
Test suite for Spectral Rules Engine

Tests the advanced NIR-based classification using RGB + NIR data.
"""

import numpy as np
import pytest

from ign_lidar.core.classification.spectral_rules import SpectralRulesEngine


def test_spectral_rules_initialization():
    """Test that SpectralRulesEngine initializes correctly."""
    engine = SpectralRulesEngine()

    assert engine.nir_vegetation_threshold == 0.4
    assert engine.nir_building_threshold == 0.3
    assert engine.brightness_concrete_min == 0.4


def test_vegetation_classification():
    """Test vegetation classification with high NDVI and high NIR."""
    engine = SpectralRulesEngine()

    # Create sample vegetation data (high NDVI, high NIR)
    n_points = 1000
    rgb = np.zeros((n_points, 3))
    rgb[:, 0] = 0.2  # Low red
    rgb[:, 1] = 0.5  # High green
    rgb[:, 2] = 0.2  # Low blue

    nir = np.ones(n_points) * 0.7  # High NIR

    labels = np.ones(n_points, dtype=np.int32)  # All unclassified

    # Classify
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    # Should classify most as vegetation
    n_vegetation = np.sum(new_labels == 4)  # ASPRS_MEDIUM_VEGETATION
    assert n_vegetation > 500, f"Expected >500 vegetation points, got {n_vegetation}"
    assert stats["vegetation_spectral"] > 0


def test_water_classification():
    """Test water classification with negative NDVI and low NIR."""
    engine = SpectralRulesEngine()

    # Create sample water data (negative NDVI, very low NIR)
    n_points = 500
    rgb = np.zeros((n_points, 3))
    rgb[:, 0] = 0.15  # Low red
    rgb[:, 1] = 0.12  # Low green
    rgb[:, 2] = 0.18  # Slightly higher blue

    nir = np.ones(n_points) * 0.05  # Very low NIR (water absorbs NIR)

    labels = np.ones(n_points, dtype=np.int32)  # All unclassified

    # Classify
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    # Should classify most as water
    n_water = np.sum(new_labels == 9)  # ASPRS_WATER
    assert n_water > 200, f"Expected >200 water points, got {n_water}"
    assert stats["water_spectral"] > 0


def test_concrete_building_classification():
    """Test concrete building classification with moderate NIR and brightness."""
    engine = SpectralRulesEngine()

    # Create sample concrete data (moderate NIR, moderate brightness, low NDVI)
    n_points = 800
    rgb = np.ones((n_points, 3)) * 0.55  # Moderate brightness
    nir = np.ones(n_points) * 0.35  # Moderate NIR

    labels = np.ones(n_points, dtype=np.int32)  # All unclassified

    # Classify
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    # Should classify as building
    n_buildings = np.sum(new_labels == 6)  # ASPRS_BUILDING
    assert n_buildings > 400, f"Expected >400 building points, got {n_buildings}"
    assert stats["building_concrete_spectral"] > 0


def test_asphalt_classification():
    """Test asphalt classification with low NIR and low brightness."""
    engine = SpectralRulesEngine()

    # Create sample asphalt data (very low NIR, low brightness, low NDVI)
    n_points = 600
    rgb = np.ones((n_points, 3)) * 0.15  # Low brightness (dark)
    nir = np.ones(n_points) * 0.08  # Very low NIR

    labels = np.ones(n_points, dtype=np.int32)  # All unclassified

    # Classify
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    # Should classify as road (asphalt)
    n_roads = np.sum(new_labels == 11)  # ASPRS_ROAD
    assert n_roads > 300, f"Expected >300 road points, got {n_roads}"
    assert stats["road_asphalt_spectral"] > 0


def test_get_spectral_features():
    """Test spectral feature extraction."""
    engine = SpectralRulesEngine()

    n_points = 100
    rgb = np.random.rand(n_points, 3)
    nir = np.random.rand(n_points)

    features = engine.get_spectral_features(rgb=rgb, nir=nir)

    # Check all expected features are present
    assert "ndvi" in features
    assert "brightness" in features
    assert "nir_red_ratio" in features
    assert "saturation" in features

    # Check feature shapes
    assert features["ndvi"].shape == (n_points,)
    assert features["brightness"].shape == (n_points,)

    # Check NDVI is in valid range
    assert np.all(features["ndvi"] >= -1)
    assert np.all(features["ndvi"] <= 1)


def test_classify_with_confidence():
    """Test classification with confidence scores."""
    engine = SpectralRulesEngine()

    # Create high-confidence vegetation
    n_points = 500
    rgb = np.zeros((n_points, 3))
    rgb[:, 1] = 0.6  # High green
    nir = np.ones(n_points) * 0.8  # Very high NIR

    labels = np.ones(n_points, dtype=np.int32)  # All unclassified

    # Classify with confidence
    new_labels, confidence, stats = engine.classify_with_confidence(
        rgb=rgb, nir=nir, current_labels=labels, confidence_threshold=0.7
    )

    # Should have some high-confidence classifications
    assert np.any(confidence > 0.7)
    assert "high_confidence_vegetation" in stats


def test_apply_to_unclassified_only():
    """Test that apply_to_unclassified_only parameter works correctly."""
    engine = SpectralRulesEngine()

    n_points = 200
    rgb = np.random.rand(n_points, 3)
    nir = np.random.rand(n_points)

    # Some already classified as building
    labels = np.ones(n_points, dtype=np.int32)  # Unclassified
    labels[:50] = 6  # First 50 are buildings

    # Classify with apply_to_unclassified_only=True
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    # Buildings should remain unchanged
    assert np.all(new_labels[:50] == 6)


def test_statistics_tracking():
    """Test that classification statistics are properly tracked."""
    engine = SpectralRulesEngine()

    # Mix of different materials
    n_total = 2000
    rgb = np.random.rand(n_total, 3)
    nir = np.random.rand(n_total)
    labels = np.ones(n_total, dtype=np.int32)

    # Classify
    new_labels, stats = engine.classify_by_spectral_signature(
        rgb=rgb, nir=nir, current_labels=labels, apply_to_unclassified_only=True
    )

    # Check that statistics are consistent
    total_classified = (
        stats.get("vegetation_spectral", 0)
        + stats.get("water_spectral", 0)
        + stats.get("building_concrete_spectral", 0)
        + stats.get("building_metal_spectral", 0)
        + stats.get("road_asphalt_spectral", 0)
    )

    assert stats["total_reclassified"] == total_classified


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
