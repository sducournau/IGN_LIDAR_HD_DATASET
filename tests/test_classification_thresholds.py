"""
Tests for Unified Classification Thresholds

Verifies:
- Threshold consistency across modules
- No conflicting values
- Valid threshold ranges
- Mode-specific threshold retrieval

Author: IGN LiDAR HD Development Team
Date: October 16, 2025
"""

import pytest
import numpy as np

from ign_lidar.core.modules.classification_thresholds import UnifiedThresholds
from ign_lidar.core.modules.transport_detection import TransportDetectionConfig, TransportDetectionMode
from ign_lidar.core.modules.classification_refinement import RefinementConfig


class TestUnifiedThresholds:
    """Test the UnifiedThresholds class."""
    
    def test_road_thresholds_consistency(self):
        """Test that road thresholds are consistent and reasonable."""
        # Height thresholds
        assert UnifiedThresholds.ROAD_HEIGHT_MIN < 0, "Road min height should allow depressions"
        assert UnifiedThresholds.ROAD_HEIGHT_MAX > 0, "Road max height should be positive"
        assert UnifiedThresholds.ROAD_HEIGHT_MAX > abs(UnifiedThresholds.ROAD_HEIGHT_MIN), \
            "Max height should be greater than abs(min)"
        
        # Strict mode should be more restrictive
        assert UnifiedThresholds.ROAD_HEIGHT_MAX_STRICT < UnifiedThresholds.ROAD_HEIGHT_MAX, \
            "Strict mode should have lower max height"
        
        # Planarity thresholds
        assert 0 < UnifiedThresholds.ROAD_PLANARITY_MIN <= 1, "Planarity should be in (0, 1]"
        assert UnifiedThresholds.ROAD_PLANARITY_MIN_STRICT > UnifiedThresholds.ROAD_PLANARITY_MIN, \
            "Strict planarity should be higher"
        
        # Intensity thresholds
        assert 0 <= UnifiedThresholds.ROAD_INTENSITY_MIN < UnifiedThresholds.ROAD_INTENSITY_MAX <= 1, \
            "Intensity range should be valid"
    
    def test_railway_thresholds_consistency(self):
        """Test that railway thresholds are consistent and reasonable."""
        # Height thresholds
        assert UnifiedThresholds.RAIL_HEIGHT_MIN < 0, "Rail min height should allow depressions"
        assert UnifiedThresholds.RAIL_HEIGHT_MAX > 0, "Rail max height should be positive"
        
        # Railways should have similar or wider ranges than roads
        assert UnifiedThresholds.RAIL_PLANARITY_MIN <= UnifiedThresholds.ROAD_PLANARITY_MIN, \
            "Rails should allow lower planarity (ballast)"
        
        # Buffer multiplier
        assert UnifiedThresholds.RAIL_BUFFER_MULTIPLIER > 1.0, \
            "Railway buffer should be wider than road"
    
    def test_building_thresholds_consistency(self):
        """Test that building thresholds are consistent across modes."""
        # Height thresholds
        assert UnifiedThresholds.BUILDING_HEIGHT_MIN > 0, "Building min height should be positive"
        assert UnifiedThresholds.BUILDING_HEIGHT_MAX > UnifiedThresholds.BUILDING_HEIGHT_MIN, \
            "Max should be greater than min"
        
        # Mode-specific thresholds should get stricter: ASPRS < LOD2 < LOD3
        assert UnifiedThresholds.BUILDING_WALL_VERTICALITY_MIN_ASPRS < \
               UnifiedThresholds.BUILDING_WALL_VERTICALITY_MIN_LOD2 < \
               UnifiedThresholds.BUILDING_WALL_VERTICALITY_MIN_LOD3, \
            "Wall verticality should increase with LOD"
        
        assert UnifiedThresholds.BUILDING_ROOF_PLANARITY_MIN_ASPRS < \
               UnifiedThresholds.BUILDING_ROOF_PLANARITY_MIN_LOD2 <= \
               UnifiedThresholds.BUILDING_ROOF_PLANARITY_MIN_LOD3, \
            "Roof planarity should increase with LOD"
    
    def test_vegetation_thresholds_consistency(self):
        """Test that vegetation thresholds are consistent."""
        # Height ranges can overlap (transition zone)
        assert UnifiedThresholds.HIGH_VEG_HEIGHT_MIN > 0, "High veg min should be positive"
        assert UnifiedThresholds.LOW_VEG_HEIGHT_MAX > UnifiedThresholds.MEDIUM_VEG_HEIGHT_MIN, \
            "Low veg should overlap with medium"
        
        # NDVI thresholds
        assert -1 <= UnifiedThresholds.NDVI_VEG_THRESHOLD <= 1, "NDVI should be in [-1, 1]"
        assert UnifiedThresholds.NDVI_HIGH_VEG_THRESHOLD > UnifiedThresholds.NDVI_VEG_THRESHOLD, \
            "High veg should have higher NDVI"
        assert UnifiedThresholds.NDVI_BUILDING_THRESHOLD < UnifiedThresholds.NDVI_VEG_THRESHOLD, \
            "Buildings should have lower NDVI than vegetation"
    
    def test_get_building_thresholds(self):
        """Test building threshold retrieval by mode."""
        # Test ASPRS mode
        asprs_thresholds = UnifiedThresholds.get_building_thresholds('asprs')
        assert asprs_thresholds['height_min'] == UnifiedThresholds.BUILDING_HEIGHT_MIN
        assert asprs_thresholds['wall_verticality_min'] == \
               UnifiedThresholds.BUILDING_WALL_VERTICALITY_MIN_ASPRS
        
        # Test LOD2 mode
        lod2_thresholds = UnifiedThresholds.get_building_thresholds('lod2')
        assert lod2_thresholds['wall_verticality_min'] == \
               UnifiedThresholds.BUILDING_WALL_VERTICALITY_MIN_LOD2
        
        # Test LOD3 mode
        lod3_thresholds = UnifiedThresholds.get_building_thresholds('lod3')
        assert lod3_thresholds['wall_verticality_min'] == \
               UnifiedThresholds.BUILDING_WALL_VERTICALITY_MIN_LOD3
        
        # Test invalid mode
        with pytest.raises(ValueError):
            UnifiedThresholds.get_building_thresholds('invalid')
    
    def test_get_transport_thresholds(self):
        """Test transport threshold retrieval."""
        # Normal mode
        normal_thresholds = UnifiedThresholds.get_transport_thresholds(strict_mode=False)
        assert normal_thresholds['road_height_max'] == UnifiedThresholds.ROAD_HEIGHT_MAX
        assert normal_thresholds['road_planarity_min'] == UnifiedThresholds.ROAD_PLANARITY_MIN
        
        # Strict mode
        strict_thresholds = UnifiedThresholds.get_transport_thresholds(strict_mode=True)
        assert strict_thresholds['road_height_max'] == UnifiedThresholds.ROAD_HEIGHT_MAX_STRICT
        assert strict_thresholds['road_planarity_min'] == UnifiedThresholds.ROAD_PLANARITY_MIN_STRICT
        
        # Strict should be more restrictive
        assert strict_thresholds['road_height_max'] < normal_thresholds['road_height_max']
        assert strict_thresholds['road_planarity_min'] > normal_thresholds['road_planarity_min']
    
    def test_get_all_thresholds(self):
        """Test retrieval of all thresholds."""
        all_thresholds = UnifiedThresholds.get_all_thresholds()
        
        # Check all categories exist
        assert 'transport' in all_thresholds
        assert 'transport_strict' in all_thresholds
        assert 'building_asprs' in all_thresholds
        assert 'building_lod2' in all_thresholds
        assert 'building_lod3' in all_thresholds
        assert 'vegetation' in all_thresholds
        assert 'ground' in all_thresholds
        assert 'vehicle' in all_thresholds
        assert 'water' in all_thresholds
        assert 'bridge' in all_thresholds
    
    def test_validate_thresholds(self):
        """Test threshold validation."""
        warnings = UnifiedThresholds.validate_thresholds()
        
        # Should return a dictionary (may be empty or contain warnings)
        assert isinstance(warnings, dict)
        
        # Check for expected overlap warning (low veg / high veg)
        # This is intentional, so we just check it's documented
        if warnings:
            for key, msg in warnings.items():
                assert isinstance(msg, str)
                assert len(msg) > 0


class TestTransportDetectionConfigIntegration:
    """Test that TransportDetectionConfig uses UnifiedThresholds correctly."""
    
    def test_config_uses_unified_thresholds(self):
        """Verify TransportDetectionConfig uses values from UnifiedThresholds."""
        config = TransportDetectionConfig(TransportDetectionMode.ASPRS_STANDARD)
        
        # Road thresholds should match
        assert config.road_height_max == UnifiedThresholds.ROAD_HEIGHT_MAX
        assert config.road_height_min == UnifiedThresholds.ROAD_HEIGHT_MIN
        assert config.road_planarity_min == UnifiedThresholds.ROAD_PLANARITY_MIN
        assert config.road_intensity_min == UnifiedThresholds.ROAD_INTENSITY_MIN
        assert config.road_intensity_max == UnifiedThresholds.ROAD_INTENSITY_MAX
        
        # Rail thresholds should match
        assert config.rail_height_max == UnifiedThresholds.RAIL_HEIGHT_MAX
        assert config.rail_height_min == UnifiedThresholds.RAIL_HEIGHT_MIN
        assert config.rail_planarity_min == UnifiedThresholds.RAIL_PLANARITY_MIN
    
    def test_strict_mode_uses_strict_thresholds(self):
        """Verify strict mode uses stricter thresholds."""
        normal_config = TransportDetectionConfig(
            TransportDetectionMode.ASPRS_STANDARD, 
            strict_mode=False
        )
        strict_config = TransportDetectionConfig(
            TransportDetectionMode.ASPRS_STANDARD, 
            strict_mode=True
        )
        
        # Strict should have lower height limits
        assert strict_config.road_height_max < normal_config.road_height_max
        assert strict_config.rail_height_max < normal_config.rail_height_max
        
        # Strict should have higher planarity requirements
        assert strict_config.road_planarity_min > normal_config.road_planarity_min
        assert strict_config.rail_planarity_min > normal_config.rail_planarity_min


class TestRefinementConfigIntegration:
    """Test that RefinementConfig uses UnifiedThresholds correctly."""
    
    def test_config_uses_unified_thresholds(self):
        """Verify RefinementConfig uses values from UnifiedThresholds."""
        # Building thresholds
        assert RefinementConfig.BUILDING_HEIGHT_MIN == UnifiedThresholds.BUILDING_HEIGHT_MIN
        
        # Vegetation thresholds
        assert RefinementConfig.LOW_VEG_HEIGHT_MAX == UnifiedThresholds.LOW_VEG_HEIGHT_MAX
        assert RefinementConfig.HIGH_VEG_HEIGHT_MIN == UnifiedThresholds.HIGH_VEG_HEIGHT_MIN
        
        # Road thresholds
        assert RefinementConfig.ROAD_HEIGHT_MAX == UnifiedThresholds.ROAD_HEIGHT_MAX
        assert RefinementConfig.ROAD_HEIGHT_MIN == UnifiedThresholds.ROAD_HEIGHT_MIN
        assert RefinementConfig.ROAD_PLANARITY_MIN == UnifiedThresholds.ROAD_PLANARITY_MIN
        
        # Rail thresholds
        assert RefinementConfig.RAIL_HEIGHT_MAX == UnifiedThresholds.RAIL_HEIGHT_MAX
        assert RefinementConfig.RAIL_HEIGHT_MIN == UnifiedThresholds.RAIL_HEIGHT_MIN
        assert RefinementConfig.RAIL_PLANARITY_MIN == UnifiedThresholds.RAIL_PLANARITY_MIN
        
        # NDVI thresholds
        assert RefinementConfig.NDVI_VEGETATION_MIN == UnifiedThresholds.NDVI_VEG_THRESHOLD
        assert RefinementConfig.NDVI_HIGH_VEG_MIN == UnifiedThresholds.NDVI_HIGH_VEG_THRESHOLD


class TestThresholdRanges:
    """Test that thresholds are within valid physical ranges."""
    
    def test_height_thresholds_reasonable(self):
        """Test that height thresholds are physically reasonable."""
        # Roads
        assert -5 < UnifiedThresholds.ROAD_HEIGHT_MIN < 0, "Road min height should be reasonable"
        assert 0 < UnifiedThresholds.ROAD_HEIGHT_MAX < 10, "Road max height should be reasonable"
        
        # Railways
        assert -5 < UnifiedThresholds.RAIL_HEIGHT_MIN < 0, "Rail min height should be reasonable"
        assert 0 < UnifiedThresholds.RAIL_HEIGHT_MAX < 10, "Rail max height should be reasonable"
        
        # Buildings
        assert 1 < UnifiedThresholds.BUILDING_HEIGHT_MIN < 5, "Building min should be reasonable"
        assert 50 < UnifiedThresholds.BUILDING_HEIGHT_MAX < 500, "Building max should be reasonable"
        
        # Vegetation
        assert 0 < UnifiedThresholds.LOW_VEG_HEIGHT_MAX < 5, "Low veg max should be reasonable"
        assert 0 < UnifiedThresholds.HIGH_VEG_HEIGHT_MIN < 3, "High veg min should be reasonable"
    
    def test_geometric_thresholds_normalized(self):
        """Test that geometric thresholds are in valid [0, 1] range."""
        # Planarity
        assert 0 < UnifiedThresholds.ROAD_PLANARITY_MIN <= 1
        assert 0 < UnifiedThresholds.ROAD_PLANARITY_MIN_STRICT <= 1
        assert 0 < UnifiedThresholds.RAIL_PLANARITY_MIN <= 1
        
        # Building geometric features
        assert 0 < UnifiedThresholds.BUILDING_WALL_VERTICALITY_MIN_ASPRS <= 1
        assert 0 < UnifiedThresholds.BUILDING_WALL_PLANARITY_MIN_ASPRS <= 1
        assert 0 < UnifiedThresholds.BUILDING_ROOF_HORIZONTALITY_MIN_ASPRS <= 1
        assert 0 < UnifiedThresholds.BUILDING_ROOF_PLANARITY_MIN_ASPRS <= 1
        
        # Intensity
        assert 0 <= UnifiedThresholds.ROAD_INTENSITY_MIN <= 1
        assert 0 <= UnifiedThresholds.ROAD_INTENSITY_MAX <= 1


class TestConsistencyAcrossModules:
    """Test for Issue #8: No conflicting thresholds across modules."""
    
    def test_no_road_height_conflicts(self):
        """Test that road height thresholds are consistent everywhere."""
        # All modules should use the same values
        transport_config = TransportDetectionConfig(TransportDetectionMode.ASPRS_STANDARD)
        refinement_config = RefinementConfig
        
        # Normal mode should all match UnifiedThresholds
        assert transport_config.road_height_max == UnifiedThresholds.ROAD_HEIGHT_MAX
        assert refinement_config.ROAD_HEIGHT_MAX == UnifiedThresholds.ROAD_HEIGHT_MAX
        
        assert transport_config.road_height_min == UnifiedThresholds.ROAD_HEIGHT_MIN
        assert refinement_config.ROAD_HEIGHT_MIN == UnifiedThresholds.ROAD_HEIGHT_MIN
    
    def test_no_rail_height_conflicts(self):
        """Test that railway height thresholds are consistent everywhere."""
        transport_config = TransportDetectionConfig(TransportDetectionMode.ASPRS_STANDARD)
        refinement_config = RefinementConfig
        
        # All should match UnifiedThresholds
        assert transport_config.rail_height_max == UnifiedThresholds.RAIL_HEIGHT_MAX
        assert refinement_config.RAIL_HEIGHT_MAX == UnifiedThresholds.RAIL_HEIGHT_MAX
        
        assert transport_config.rail_height_min == UnifiedThresholds.RAIL_HEIGHT_MIN
        assert refinement_config.RAIL_HEIGHT_MIN == UnifiedThresholds.RAIL_HEIGHT_MIN
    
    def test_no_building_height_conflicts(self):
        """Test that building height thresholds are consistent."""
        refinement_config = RefinementConfig
        
        # Should all use the same building height minimum
        assert refinement_config.BUILDING_HEIGHT_MIN == UnifiedThresholds.BUILDING_HEIGHT_MIN


class TestUpdatedThresholds:
    """Test that Issues #1 and #4 fixes are in place."""
    
    def test_issue_1_road_height_increased(self):
        """Test Issue #1: Road height max increased from 1.5m to 2.0m."""
        assert UnifiedThresholds.ROAD_HEIGHT_MAX == 2.0, \
            "Road height max should be 2.0m (Issue #1)"
        assert UnifiedThresholds.ROAD_HEIGHT_MIN == -0.5, \
            "Road height min should be -0.5m (Issue #1)"
    
    def test_issue_4_rail_height_increased(self):
        """Test Issue #4: Rail height max increased from 1.2m to 2.0m."""
        assert UnifiedThresholds.RAIL_HEIGHT_MAX == 2.0, \
            "Rail height max should be 2.0m (Issue #4)"
        assert UnifiedThresholds.RAIL_HEIGHT_MIN == -0.5, \
            "Rail height min should be -0.5m (Issue #4)"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
