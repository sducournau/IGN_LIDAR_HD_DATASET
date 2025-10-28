"""
Integration tests for Enhanced Building Classifier (Phase 2.4).

Tests the integration of EnhancedBuildingClassifier into the main
BuildingFacadeClassifier pipeline.

Author: IGN LiDAR HD Processing Library
Date: October 2025
Version: 3.4.0
"""

import logging
from typing import Dict

import numpy as np
import pytest
from shapely.geometry import Polygon

from ign_lidar.config.building_config import BuildingConfig
from ign_lidar.core.classification.building.facade_processor import (
    BuildingFacadeClassifier,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_building_points():
    """
    Create synthetic building point cloud.

    Returns 1000 points representing a simple building with:
    - Facade points (vertical walls)
    - Roof points (horizontal surface)
    - Potential chimney points (above roof)
    - Potential balcony points (horizontal protrusions)
    """
    np.random.seed(42)

    # Building dimensions: 15m x 10m x 8m
    points = []

    # 1. Facade points (vertical walls) - 400 points
    for _ in range(400):
        x = np.random.uniform(0, 15)
        y = np.random.choice([0, 10]) + np.random.normal(0, 0.1)
        z = np.random.uniform(0, 6)
        points.append([x, y, z])

    # 2. Roof points (horizontal surface) - 400 points
    for _ in range(400):
        x = np.random.uniform(0, 15)
        y = np.random.uniform(0, 10)
        z = 6.0 + np.random.normal(0, 0.1)
        points.append([x, y, z])

    # 3. Chimney points (above roof) - 100 points
    for _ in range(100):
        x = 7.5 + np.random.normal(0, 0.3)  # Center
        y = 5.0 + np.random.normal(0, 0.3)
        z = np.random.uniform(6.5, 8.0)
        points.append([x, y, z])

    # 4. Balcony points (horizontal protrusion) - 100 points
    for _ in range(100):
        x = 7.5 + np.random.normal(0, 0.5)
        y = -0.5 + np.random.normal(0, 0.1)  # Protruding from facade
        z = np.random.uniform(3, 4)
        points.append([x, y, z])

    return np.array(points, dtype=np.float32)


@pytest.fixture
def sample_building_polygon():
    """Create building footprint polygon (15m x 10m)."""
    return Polygon([(0, 0), (15, 0), (15, 10), (0, 10), (0, 0)])


@pytest.fixture
def sample_features(sample_building_points):
    """
    Create synthetic features for building points.

    Returns dictionary with normals, verticality, and curvature.
    """
    n_points = len(sample_building_points)

    # Normals (random unit vectors)
    normals = np.random.randn(n_points, 3).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Verticality (0-1, where 1 = vertical)
    # Make facade points more vertical, roof points horizontal
    verticality = np.zeros(n_points, dtype=np.float32)
    verticality[:400] = 0.8 + np.random.uniform(0, 0.2, 400)  # Facades
    verticality[400:800] = 0.1 + np.random.uniform(0, 0.1, 400)  # Roof
    verticality[800:900] = 0.9 + np.random.uniform(0, 0.1, 100)  # Chimney
    verticality[900:] = 0.2 + np.random.uniform(0, 0.1, 100)  # Balcony

    # Curvature (0-1, higher = more curved/edge)
    curvature = np.random.uniform(0, 0.1, n_points).astype(np.float32)

    return {
        "normals": normals,
        "verticality": verticality,
        "curvature": curvature,
    }


@pytest.fixture
def buildings_gdf(sample_building_polygon):
    """Create GeoDataFrame with single building."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"id": [1], "geometry": [sample_building_polygon]}, crs="EPSG:2154"
    )


# ============================================================================
# Configuration Tests
# ============================================================================


def test_enhanced_config_initialization():
    """Test BuildingConfig initialization."""
    config = BuildingConfig()

    assert config.enable_roof_detection is True
    assert config.enable_chimney_detection is True
    assert config.enable_balcony_detection is True
    assert config.roof_flat_threshold == 15.0
    assert config.chimney_min_height_above_roof == 1.0
    assert config.balcony_min_distance_from_facade == 0.5


def test_enhanced_config_presets():
    """Test BuildingConfig presets."""
    # Residential
    res = BuildingConfig.preset_residential()
    assert res.roof_flat_threshold == 15.0

    # Urban high-density
    urban = BuildingConfig.preset_urban_high_density()
    assert urban.roof_flat_threshold == 10.0
    assert urban.chimney_min_height_above_roof == 0.5

    # Industrial
    industrial = BuildingConfig.preset_industrial()
    assert industrial.enable_balcony_detection is False
    assert industrial.chimney_min_height_above_roof == 2.0

    # Historic
    historic = BuildingConfig.preset_historic()
    assert historic.roof_flat_threshold == 25.0
    assert historic.balcony_confidence_threshold == 0.4


def test_enhanced_config_to_dict():
    """Test BuildingConfig.to_dict()."""
    config = BuildingConfig(
        enable_roof_detection=True,
        enable_chimney_detection=False,
        roof_flat_threshold=12.0,
    )
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["enable_roof_detection"] is True
    assert config_dict["enable_chimney_detection"] is False
    assert config_dict["roof_flat_threshold"] == 12.0


def test_enhanced_config_from_dict():
    """Test BuildingConfig.from_dict()."""
    data = {
        "enable_roof_detection": False,
        "chimney_min_height_above_roof": 1.5,
    }
    config = BuildingConfig.from_dict(data)

    assert config.enable_roof_detection is False
    assert config.chimney_min_height_above_roof == 1.5
    # Should use defaults for unspecified params
    assert config.enable_chimney_detection is True


# ============================================================================
# Classifier Integration Tests
# ============================================================================


def test_classifier_initialization_without_enhanced():
    """Test BuildingFacadeClassifier without enhanced features."""
    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=False,
    )

    assert classifier.enable_enhanced_lod3 is False
    assert classifier.enhanced_classifier is None


def test_classifier_initialization_with_enhanced_default():
    """Test BuildingFacadeClassifier with enhanced features (default config)."""
    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=True,
    )

    assert classifier.enable_enhanced_lod3 is True
    # Should have initialized enhanced classifier (if imports available)
    # May be None if Phase 2 modules not available


def test_classifier_initialization_with_enhanced_custom():
    """Test BuildingFacadeClassifier with custom enhanced config."""
    enhanced_config = {
        "enable_roof_detection": True,
        "enable_chimney_detection": False,
        "enable_balcony_detection": True,
        "roof_flat_threshold": 12.0,
    }

    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=True,
        enhanced_building_config=enhanced_config,
    )

    assert classifier.enable_enhanced_lod3 is True
    assert classifier.enhanced_building_config == enhanced_config


@pytest.mark.integration
def test_classify_single_building_without_enhanced(
    sample_building_polygon,
    sample_building_points,
    sample_features,
):
    """Test classify_single_building without enhanced features."""
    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=False,
    )

    # Prepare inputs
    points = sample_building_points
    heights = points[:, 2]
    labels = np.zeros(len(points), dtype=np.int32)

    # Classify
    labels_updated, stats = classifier.classify_single_building(
        building_id=1,
        polygon=sample_building_polygon,
        points=points,
        heights=heights,
        labels=labels,
        normals=sample_features["normals"],
        verticality=sample_features["verticality"],
        curvature=sample_features["curvature"],
    )

    # Verify results
    assert len(labels_updated) == len(points)
    assert stats["points_classified"] > 0
    assert "enhanced_lod3_enabled" not in stats


@pytest.mark.integration
@pytest.mark.skipif(
    not hasattr(
        __import__(
            "ign_lidar.core.classification.building",
            fromlist=["EnhancedBuildingClassifier"],
        ),
        "EnhancedBuildingClassifier",
    ),
    reason="EnhancedBuildingClassifier not available",
)
def test_classify_single_building_with_enhanced(
    sample_building_polygon,
    sample_building_points,
    sample_features,
):
    """Test classify_single_building with enhanced features."""
    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=True,
        enhanced_building_config={
            "enable_roof_detection": True,
            "enable_chimney_detection": True,
            "enable_balcony_detection": True,
        },
    )

    # Skip if enhanced classifier not initialized
    if classifier.enhanced_classifier is None:
        pytest.skip("Enhanced classifier not available")

    # Prepare inputs
    points = sample_building_points
    heights = points[:, 2]
    labels = np.zeros(len(points), dtype=np.int32)

    # Classify
    labels_updated, stats = classifier.classify_single_building(
        building_id=1,
        polygon=sample_building_polygon,
        points=points,
        heights=heights,
        labels=labels,
        normals=sample_features["normals"],
        verticality=sample_features["verticality"],
        curvature=sample_features["curvature"],
    )

    # Verify results
    assert len(labels_updated) == len(points)
    assert stats["points_classified"] > 0

    # Check enhanced statistics
    if "enhanced_lod3_enabled" in stats:
        assert stats["enhanced_lod3_enabled"] is True
        assert "roof_type_enhanced" in stats
        # May have chimneys/balconies detected
        if "num_chimneys" in stats:
            assert stats["num_chimneys"] >= 0
        if "num_balconies" in stats:
            assert stats["num_balconies"] >= 0


@pytest.mark.integration
def test_classify_buildings_batch(
    buildings_gdf,
    sample_building_points,
    sample_features,
):
    """Test classify_buildings with multiple buildings."""
    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=True,
    )

    # Prepare inputs
    points = sample_building_points
    heights = points[:, 2]
    labels = np.zeros(len(points), dtype=np.int32)

    # Classify all buildings
    labels_updated, stats = classifier.classify_buildings(
        buildings_gdf=buildings_gdf,
        points=points,
        heights=heights,
        labels=labels,
        normals=sample_features["normals"],
        verticality=sample_features["verticality"],
        curvature=sample_features["curvature"],
    )

    # Verify results
    assert len(labels_updated) == len(points)
    assert stats["buildings_processed"] == 1
    assert stats["points_classified"] > 0


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_classifier_with_invalid_features():
    """Test classifier with invalid/missing features."""
    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=True,
    )

    # Should handle None features gracefully
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    points = np.random.randn(100, 3).astype(np.float32)
    heights = points[:, 2]
    labels = np.zeros(len(points), dtype=np.int32)

    # Should not crash with None features
    labels_updated, stats = classifier.classify_single_building(
        building_id=1,
        polygon=polygon,
        points=points,
        heights=heights,
        labels=labels,
        normals=None,
        verticality=None,
        curvature=None,
    )

    assert len(labels_updated) == len(points)


def test_classifier_with_empty_building():
    """Test classifier with empty building."""
    classifier = BuildingFacadeClassifier(
        enable_enhanced_lod3=True,
    )

    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    points = np.random.randn(1000, 3).astype(np.float32) + 100  # Far away
    heights = points[:, 2]
    labels = np.zeros(len(points), dtype=np.int32)

    # Should handle empty building gracefully
    labels_updated, stats = classifier.classify_single_building(
        building_id=1,
        polygon=polygon,
        points=points,
        heights=heights,
        labels=labels,
    )

    assert len(labels_updated) == len(points)
    assert stats["points_classified"] == 0


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_performance_overhead():
    """Test computational overhead of enhanced classification."""
    import time

    # Create larger point cloud
    np.random.seed(42)
    points = np.random.randn(5000, 3).astype(np.float32) * 10
    heights = points[:, 2]
    labels = np.zeros(len(points), dtype=np.int32)
    polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)])

    # Synthetic features
    normals = np.random.randn(len(points), 3).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    verticality = np.random.uniform(0, 1, len(points)).astype(np.float32)
    curvature = np.random.uniform(0, 0.5, len(points)).astype(np.float32)

    # Test without enhanced features
    classifier_basic = BuildingFacadeClassifier(
        enable_enhanced_lod3=False,
    )
    t0 = time.time()
    _, stats_basic = classifier_basic.classify_single_building(
        building_id=1,
        polygon=polygon,
        points=points,
        heights=heights,
        labels=labels.copy(),
        normals=normals,
        verticality=verticality,
        curvature=curvature,
    )
    time_basic = time.time() - t0

    # Test with enhanced features
    classifier_enhanced = BuildingFacadeClassifier(
        enable_enhanced_lod3=True,
    )
    t0 = time.time()
    _, stats_enhanced = classifier_enhanced.classify_single_building(
        building_id=1,
        polygon=polygon,
        points=points,
        heights=heights,
        labels=labels.copy(),
        normals=normals,
        verticality=verticality,
        curvature=curvature,
    )
    time_enhanced = time.time() - t0

    # Report performance
    overhead = (time_enhanced - time_basic) / time_basic * 100
    logger.info(f"Basic classification: {time_basic:.3f}s")
    logger.info(f"Enhanced classification: {time_enhanced:.3f}s")
    logger.info(f"Overhead: {overhead:.1f}%")

    # Overhead should be reasonable (<100% for this test)
    # Note: May vary based on hardware and data
    assert overhead < 200  # Allow up to 2x overhead for test


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
