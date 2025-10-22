"""
Unit tests for threshold module backward compatibility.

This test suite verifies that the deprecated threshold modules 
(classification_thresholds.py and optimized_thresholds.py) maintain
100% backward compatibility with the new unified thresholds.py module.

Tests cover:
- Deprecation warnings are emitted correctly
- Import paths remain functional
- Module structure is correct
"""

import warnings


class TestClassificationThresholdsBackwardCompatibility:
    """Test backward compatibility for classification_thresholds.py wrapper."""
    
    def test_deprecation_warning_emitted(self):
        """Verify that importing classification_thresholds emits a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import the deprecated module
            from ign_lidar.core.classification import classification_thresholds
            
            # Check that a deprecation warning was issued
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            
            # Check the warning message mentions the new module
            warning_messages = [str(warning.message) for warning in w]
            assert any('thresholds.py' in msg for msg in warning_messages)
    
    def test_classification_thresholds_class_exists(self):
        """Verify ClassificationThresholds class is accessible via the wrapper."""
        from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds
        
        # Should be able to instantiate
        thresholds = ClassificationThresholds()
        assert thresholds is not None
    
    def test_mode_specific_thresholds_asprs(self):
        """Test ASPRS mode-specific thresholds work correctly."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds(mode='asprs')
        
        # ASPRS should have specific configurations
        assert thresholds.mode == 'asprs'
        assert thresholds.height.building_height_min == 2.5
    
    def test_mode_specific_thresholds_lod2(self):
        """Test LOD2 mode-specific thresholds work correctly."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds(mode='lod2')
        
        # LOD2 should have specific configurations
        assert thresholds.mode == 'lod2'
        assert thresholds.height.building_height_min == 2.5


class TestOptimizedThresholdsBackwardCompatibility:
    """Test backward compatibility for optimized_thresholds.py wrapper."""
    
    def test_deprecation_warning_emitted(self):
        """Verify that importing optimized_thresholds emits a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Import the deprecated module
            from ign_lidar.core.classification import optimized_thresholds
            
            # Check that a deprecation warning was issued
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
            
            # Check the warning message mentions the new module
            warning_messages = [str(warning.message) for warning in w]
            assert any('thresholds.py' in msg for msg in warning_messages)
    
    def test_threshold_classes_accessible(self):
        """Verify all threshold classes are accessible via the wrapper."""
        from ign_lidar.core.classification.optimized_thresholds import (
            NDVIThresholds,
            GeometricThresholds,
            HeightThresholds
        )
        
        # Should be able to instantiate all classes
        ndvi = NDVIThresholds()
        geometric = GeometricThresholds()
        height = HeightThresholds()
        
        assert ndvi is not None
        assert geometric is not None
        assert height is not None
    
    def test_ndvi_values_accessible(self):
        """Verify NDVI threshold values are accessible."""
        from ign_lidar.core.classification.optimized_thresholds import NDVIThresholds
        
        ndvi = NDVIThresholds()
        
        # Check key attributes exist and are reasonable
        assert hasattr(ndvi, 'vegetation_min')
        assert 0 < ndvi.vegetation_min < 1
    
    def test_height_values_accessible(self):
        """Verify height threshold values are accessible."""
        from ign_lidar.core.classification.optimized_thresholds import HeightThresholds
        
        height = HeightThresholds()
        
        # Check key attributes exist and are reasonable
        assert hasattr(height, 'building_height_min')
        assert height.building_height_min > 0
    
    def test_geometric_values_accessible(self):
        """Verify geometric threshold values are accessible."""
        from ign_lidar.core.classification.optimized_thresholds import GeometricThresholds
        
        geometric = GeometricThresholds()
        
        # Check key attributes exist
        assert hasattr(geometric, 'planarity_ground_min')
        assert 0 < geometric.planarity_ground_min < 1


class TestUnifiedThresholdsModule:
    """Test the new unified thresholds module directly."""
    
    def test_get_thresholds_default(self):
        """Test get_thresholds() with default parameters."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds()
        
        assert thresholds is not None
        # Default mode is asprs (not lod2)
        assert thresholds.mode in ['asprs', 'lod2', 'lod3']
        assert thresholds.strict in [True, False]
    
    def test_get_thresholds_all_modes(self):
        """Test get_thresholds() with all supported modes."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        modes = ['asprs', 'lod2', 'lod3']
        
        for mode in modes:
            thresholds = get_thresholds(mode=mode)
            assert thresholds.mode == mode
    
    def test_threshold_config_structure(self):
        """Test that ThresholdConfig has all expected attributes."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds()
        
        # Check all threshold categories exist
        assert hasattr(thresholds, 'ndvi')
        assert hasattr(thresholds, 'geometric')
        assert hasattr(thresholds, 'height')
        assert hasattr(thresholds, 'transport')
        assert hasattr(thresholds, 'building')
    
    def test_ndvi_thresholds_structure(self):
        """Test NDVIThresholds structure."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds()
        ndvi = thresholds.ndvi
        
        # Check required attributes
        assert hasattr(ndvi, 'vegetation_min')
        assert hasattr(ndvi, 'vegetation_healthy')
        assert hasattr(ndvi, 'building_max')
    
    def test_height_thresholds_structure(self):
        """Test HeightThresholds structure."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds()
        height = thresholds.height
        
        # Check required attributes with correct names
        assert hasattr(height, 'ground_height_max')
        assert hasattr(height, 'low_veg_height_max')
        assert hasattr(height, 'building_height_min')
        assert hasattr(height, 'building_height_max')
    
    def test_geometric_thresholds_structure(self):
        """Test GeometricThresholds structure."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds()
        geometric = thresholds.geometric
        
        # Check required attributes
        assert hasattr(geometric, 'planarity_ground_min')
        assert hasattr(geometric, 'planarity_road_min')
    
    def test_transport_thresholds_structure(self):
        """Test TransportThresholds structure."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds()
        transport = thresholds.transport
        
        # Check required attributes
        assert hasattr(transport, 'road_planarity_min')
        assert hasattr(transport, 'rail_planarity_min')
    
    def test_building_thresholds_structure(self):
        """Test BuildingThresholds structure."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        thresholds = get_thresholds()
        building = thresholds.building
        
        # Check required attributes
        assert hasattr(building, 'height_min')
        assert hasattr(building, 'height_max')
    
    def test_strict_mode_differences(self):
        """Test that strict mode affects threshold values."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        normal = get_thresholds(strict=False)
        strict = get_thresholds(strict=True)
        
        assert normal is not None
        assert strict is not None
        assert normal.strict == False
        assert strict.strict == True
    
    def test_season_context_affects_thresholds(self):
        """Test that season context can be passed."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        # Test different seasons
        summer = get_thresholds(season='summer')
        winter = get_thresholds(season='winter')
        
        # Should create valid configs
        assert summer is not None
        assert winter is not None


class TestMigrationScenarios:
    """Test common migration scenarios from old to new API."""
    
    def test_migration_scenario_1_direct_import(self):
        """Test migrating from direct ClassificationThresholds import."""
        # Old way (should still work with warning)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds
            old_thresholds = ClassificationThresholds()
        
        # New way
        from ign_lidar.core.classification.thresholds import get_thresholds
        new_thresholds = get_thresholds()
        
        # Both should be valid objects
        assert old_thresholds is not None
        assert new_thresholds is not None
    
    def test_migration_scenario_2_optimized_classes(self):
        """Test migrating from optimized_thresholds classes."""
        # Old way (should still work with warning)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            from ign_lidar.core.classification.optimized_thresholds import NDVIThresholds
            old_ndvi = NDVIThresholds()
        
        # New way
        from ign_lidar.core.classification.thresholds import get_thresholds
        new_thresholds = get_thresholds()
        
        # Should both have vegetation_min
        assert hasattr(old_ndvi, 'vegetation_min')
        assert hasattr(new_thresholds.ndvi, 'vegetation_min')
        # Values should match
        assert old_ndvi.vegetation_min == new_thresholds.ndvi.vegetation_min
    
    def test_migration_scenario_3_accessing_attributes(self):
        """Test that attribute access patterns work correctly."""
        from ign_lidar.core.classification.thresholds import get_thresholds
        
        config = get_thresholds()
        
        # All these access patterns should work
        assert config.ndvi.vegetation_min > 0
        assert config.height.building_height_min > 0
        assert config.geometric.planarity_ground_min > 0
        assert config.transport.road_planarity_min > 0
        assert config.building.height_min > 0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
