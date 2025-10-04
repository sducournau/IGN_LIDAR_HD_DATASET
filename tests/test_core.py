#!/usr/bin/env python3
"""
Test Core Functionality - IGN LiDAR HD Library

Tests for the main LiDARProcessor class and core functionality.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the modules to test
from ign_lidar import LiDARProcessor, LOD2_CLASSES, LOD3_CLASSES
from ign_lidar.processor import LiDARProcessor as ProcessorClass


class TestLiDARProcessor:
    """Test cases for the main LiDARProcessor class."""
    
    def test_processor_initialization(self):
        """Test processor can be initialized with different parameters."""
        # Default initialization
        processor = LiDARProcessor()
        assert processor.lod_level == "LOD2"
        assert processor.augment is False
        
        # Custom initialization
        processor = LiDARProcessor(
            lod_level="LOD3",
            augment=True,
            patch_size=200.0
        )
        assert processor.lod_level == "LOD3"
        assert processor.augment is True
    
    def test_lod_level_validation(self):
        """Test that invalid LOD levels are handled properly."""
        # Valid LOD levels should work
        LiDARProcessor(lod_level="LOD2")
        LiDARProcessor(lod_level="LOD3")
        
        # Invalid LOD level should raise an error or default
        try:
            processor = LiDARProcessor(lod_level="INVALID")
            # If no error, check it defaults to something sensible
            assert processor.lod_level in ["LOD2", "LOD3"]
        except ValueError:
            # This is also acceptable behavior
            pass
    
    def test_bbox_filtering(self, sample_point_cloud):
        """Test bounding box filtering functionality."""
        processor = LiDARProcessor(bbox=(25, 25, 75, 75))
        
        # Create test points
        points = sample_point_cloud['xyz']
        
        # Mock the bbox filtering (since we don't have the actual implementation)
        # This test verifies the interface exists
        assert hasattr(processor, 'bbox')
        assert processor.bbox == (25, 25, 75, 75)


class TestClassificationSchemas:
    """Test the classification schemas."""
    
    def test_lod2_classes_structure(self):
        """Test LOD2 classification schema."""
        assert isinstance(LOD2_CLASSES, dict)
        assert len(LOD2_CLASSES) > 0
        
        # Should have integer keys and string values
        for key, value in LOD2_CLASSES.items():
            assert isinstance(key, int)
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_lod3_classes_structure(self):
        """Test LOD3 classification schema."""
        assert isinstance(LOD3_CLASSES, dict)
        assert len(LOD3_CLASSES) > 0
        assert len(LOD3_CLASSES) > len(LOD2_CLASSES)  # LOD3 should have more classes
        
        # Should have integer keys and string values
        for key, value in LOD3_CLASSES.items():
            assert isinstance(key, int)
            assert isinstance(value, str)
            assert len(value) > 0
    
    def test_class_ids_are_sequential(self):
        """Test that class IDs are reasonably sequential."""
        lod2_ids = sorted(LOD2_CLASSES.keys())
        lod3_ids = sorted(LOD3_CLASSES.keys())
        
        # IDs should start from 0 or 1
        assert min(lod2_ids) in [0, 1]
        assert min(lod3_ids) in [0, 1]
        
        # Should not have huge gaps
        assert max(lod2_ids) < 50  # Reasonable upper bound
        assert max(lod3_ids) < 100  # Reasonable upper bound


class TestFeatureExtraction:
    """Test feature extraction functions."""
    
    def test_feature_functions_importable(self):
        """Test that feature extraction functions can be imported."""
        from ign_lidar import (
            compute_normals,
            compute_curvature, 
            extract_geometric_features
        )
        
        # Functions should be callable
        assert callable(compute_normals)
        assert callable(compute_curvature)
        assert callable(extract_geometric_features)
    
    @patch('ign_lidar.features.compute_normals')
    def test_normal_computation_interface(self, mock_normals, sample_point_cloud):
        """Test normal computation interface."""
        points = sample_point_cloud['xyz']
        
        # Mock return value
        expected_normals = np.random.rand(len(points), 3)
        mock_normals.return_value = expected_normals
        
        from ign_lidar import compute_normals
        result = compute_normals(points)
        
        # Verify function was called and returns expected shape
        mock_normals.assert_called_once()
        np.testing.assert_array_equal(result, expected_normals)


class TestTileManagement:
    """Test tile management functionality."""
    
    def test_working_tiles_import(self):
        """Test that working tiles can be imported."""
        from ign_lidar import WORKING_TILES
        
        assert isinstance(WORKING_TILES, list)
        assert len(WORKING_TILES) > 0
        
        # Each tile should have required attributes
        for tile in WORKING_TILES[:3]:  # Test first few tiles
            assert hasattr(tile, 'filename')
            assert hasattr(tile, 'environment')
            assert hasattr(tile, 'location')
            assert hasattr(tile, 'recommended_lod')
    
    def test_environment_filtering(self):
        """Test filtering tiles by environment."""
        from ign_lidar import get_tiles_by_environment
        
        # Should be able to get tiles by environment
        urban_tiles = get_tiles_by_environment("urban")
        assert isinstance(urban_tiles, list)
        
        # All returned tiles should match the environment
        for tile in urban_tiles:
            assert tile.environment == "urban"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])