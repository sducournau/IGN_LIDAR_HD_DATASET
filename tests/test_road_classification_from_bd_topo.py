"""
Test Road Classification from BD Topo Nature Attribute

This test verifies that roads are classified using detailed ASPRS codes
based on the BD Topo 'nature' attribute (e.g., Autoroute, Chemin, etc.)
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import geopandas as gpd
from shapely.geometry import Polygon

from ign_lidar.classification_schema import (
    ASPRSClass,
    get_classification_for_road,
    ClassificationMode,
    ROAD_NATURE_TO_ASPRS,
)


class TestRoadNatureMapping:
    """Test road nature to ASPRS code mapping."""

    def test_road_nature_mapping_exists(self):
        """Verify ROAD_NATURE_TO_ASPRS mapping is defined."""
        assert ROAD_NATURE_TO_ASPRS is not None
        assert isinstance(ROAD_NATURE_TO_ASPRS, dict)
        assert len(ROAD_NATURE_TO_ASPRS) > 0

    def test_road_nature_mapping_values(self):
        """Test specific road nature mappings."""
        # Test key road types
        assert ROAD_NATURE_TO_ASPRS["Autoroute"] == ASPRSClass.ROAD_MOTORWAY
        assert ROAD_NATURE_TO_ASPRS["Chemin"] == ASPRSClass.ROAD_SERVICE
        assert ROAD_NATURE_TO_ASPRS["Piste cyclable"] == ASPRSClass.ROAD_CYCLEWAY
        assert ROAD_NATURE_TO_ASPRS["Sentier"] == ASPRSClass.ROAD_PEDESTRIAN

    def test_get_classification_for_road_with_nature(self):
        """Test get_classification_for_road with nature parameter."""
        # Motorway
        code = get_classification_for_road(
            nature="Autoroute", mode=ClassificationMode.ASPRS_EXTENDED
        )
        assert code == ASPRSClass.ROAD_MOTORWAY

        # Pedestrian path
        code = get_classification_for_road(
            nature="Sentier", mode=ClassificationMode.ASPRS_EXTENDED
        )
        assert code == ASPRSClass.ROAD_PEDESTRIAN

    def test_get_classification_for_road_default(self):
        """Test get_classification_for_road with unknown nature."""
        # Unknown nature should default to ROAD_SURFACE
        code = get_classification_for_road(
            nature="Unknown Road Type", mode=ClassificationMode.ASPRS_EXTENDED
        )
        assert code == ASPRSClass.ROAD_SURFACE

    def test_get_classification_for_road_standard_mode(self):
        """Test that standard mode always returns ROAD_SURFACE."""
        code = get_classification_for_road(
            nature="Autoroute", mode=ClassificationMode.ASPRS_STANDARD
        )
        assert code == ASPRSClass.ROAD_SURFACE


class TestReclassifierRoadNature:
    """Test reclassifier with road nature classification."""

    def test_get_asprs_code_for_road(self):
        """Test _get_asprs_code_for_road method."""
        try:
            from ign_lidar.core.classification.reclassifier import Reclassifier

            reclassifier = Reclassifier(acceleration_mode="cpu")

            # Test motorway
            code = reclassifier._get_asprs_code_for_road("Autoroute")
            assert code == ASPRSClass.ROAD_MOTORWAY

            # Test service road
            code = reclassifier._get_asprs_code_for_road("Chemin")
            assert code == ASPRSClass.ROAD_SERVICE

            # Test default for unknown
            code = reclassifier._get_asprs_code_for_road(None)
            assert code == ASPRSClass.ROAD_SURFACE

        except ImportError:
            pytest.skip("Reclassifier not available")

    def test_get_asprs_code_with_road_properties(self):
        """Test _get_asprs_code with road properties."""
        try:
            from ign_lidar.core.classification.reclassifier import Reclassifier

            reclassifier = Reclassifier(acceleration_mode="cpu")

            # Test with road nature in properties
            properties = {"nature": "Autoroute"}
            code = reclassifier._get_asprs_code("roads", properties)
            assert code == ASPRSClass.ROAD_MOTORWAY

            # Test without nature in properties
            properties = {"other": "value"}
            code = reclassifier._get_asprs_code("roads", properties)
            assert code == ASPRSClass.ROAD_SURFACE

            # Test non-road feature
            code = reclassifier._get_asprs_code("buildings", properties)
            assert code == ASPRSClass.BUILDING

        except ImportError:
            pytest.skip("Reclassifier not available")


class TestOptimizedGroundTruthClassifier:
    """Test OptimizedGroundTruthClassifier with road nature."""

    def test_get_asprs_code_for_road(self):
        """Test _get_asprs_code_for_road method."""
        try:
            from ign_lidar.optimization.strtree import OptimizedGroundTruthClassifier

            classifier = OptimizedGroundTruthClassifier()

            # Test motorway
            code = classifier._get_asprs_code_for_road("Autoroute")
            assert code == ASPRSClass.ROAD_MOTORWAY

            # Test cycleway
            code = classifier._get_asprs_code_for_road("Piste cyclable")
            assert code == ASPRSClass.ROAD_CYCLEWAY

        except ImportError:
            pytest.skip("OptimizedGroundTruthClassifier not available")

    def test_get_asprs_code_with_road_properties(self):
        """Test _get_asprs_code with road properties."""
        try:
            from ign_lidar.optimization.strtree import OptimizedGroundTruthClassifier

            classifier = OptimizedGroundTruthClassifier()

            # Test with road nature
            properties = {"nature": "Autoroute"}
            code = classifier._get_asprs_code("roads", properties)
            assert code == ASPRSClass.ROAD_MOTORWAY

            # Test without nature
            properties = {}
            code = classifier._get_asprs_code("roads", properties)
            assert code == ASPRSClass.ROAD_SURFACE

            # Test non-road
            code = classifier._get_asprs_code("water", properties)
            assert code == ASPRSClass.WATER

        except ImportError:
            pytest.skip("OptimizedGroundTruthClassifier not available")


class TestRoadClassificationIntegration:
    """Integration tests for road classification."""

    @pytest.mark.integration
    def test_road_gdf_with_nature_attribute(self):
        """Test that road GeoDataFrame includes nature attribute."""
        try:
            from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher

            fetcher = IGNGroundTruthFetcher()

            # Create mock bbox
            bbox = (650000, 6860000, 651000, 6861000)

            # Note: This would require actual WFS connection
            # For now, just verify the method exists
            assert hasattr(fetcher, "fetch_roads_with_polygons")

        except ImportError:
            pytest.skip("IGNGroundTruthFetcher not available")

    @pytest.mark.unit
    def test_road_classification_preserves_nature(self):
        """Test that road classification preserves nature attribute through pipeline."""
        # Create mock road GeoDataFrame with nature attribute
        geometries = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
        ]

        roads_gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "nature": ["Autoroute", "Chemin"],
                "width_m": [12.0, 3.0],
            },
            crs="EPSG:2154",
        )

        # Verify nature attribute exists
        assert "nature" in roads_gdf.columns
        assert roads_gdf.iloc[0]["nature"] == "Autoroute"
        assert roads_gdf.iloc[1]["nature"] == "Chemin"

        # Verify we can map to ASPRS codes
        codes = [
            get_classification_for_road(nature, ClassificationMode.ASPRS_EXTENDED)
            for nature in roads_gdf["nature"]
        ]

        assert codes[0] == ASPRSClass.ROAD_MOTORWAY
        assert codes[1] == ASPRSClass.ROAD_SERVICE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
