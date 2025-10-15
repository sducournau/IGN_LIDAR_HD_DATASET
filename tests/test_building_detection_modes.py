"""
Tests for Multi-Mode Building Detection

Tests the building detection module across ASPRS, LOD2, and LOD3 modes.
"""

import pytest
import numpy as np
from ign_lidar.core.modules.building_detection import (
    BuildingDetectionMode,
    BuildingDetectionConfig,
    BuildingDetector,
    detect_buildings_multi_mode
)


@pytest.fixture
def sample_building_points():
    """Create synthetic building point cloud with various features."""
    n_points = 1000
    
    # Create points with varying characteristics
    height = np.random.uniform(0, 20, n_points)
    planarity = np.random.uniform(0, 1, n_points)
    verticality = np.random.uniform(0, 1, n_points)
    
    # Create normals (random for now)
    normals = np.random.randn(n_points, 3)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Create other features
    linearity = np.random.uniform(0, 1, n_points)
    anisotropy = np.random.uniform(0, 1, n_points)
    curvature = np.random.uniform(0, 0.5, n_points)
    intensity = np.random.uniform(0, 1, n_points)
    
    # Create building-like points
    building_mask = (
        (height > 3) & 
        (height < 15) & 
        (planarity > 0.6)
    )
    
    # Initial labels (all unclassified)
    labels = np.zeros(n_points, dtype=np.uint8)
    
    features = {
        'height': height,
        'planarity': planarity,
        'verticality': verticality,
        'normals': normals,
        'linearity': linearity,
        'anisotropy': anisotropy,
        'curvature': curvature,
        'intensity': intensity,
        'points': np.random.randn(n_points, 3)
    }
    
    return labels, features, building_mask


class TestBuildingDetectionMode:
    """Test BuildingDetectionMode enum."""
    
    def test_mode_values(self):
        """Test that all mode values are valid."""
        assert BuildingDetectionMode.ASPRS == "asprs"
        assert BuildingDetectionMode.LOD2 == "lod2"
        assert BuildingDetectionMode.LOD3 == "lod3"
    
    def test_mode_conversion(self):
        """Test string to mode conversion."""
        mode = BuildingDetectionMode("asprs")
        assert mode == BuildingDetectionMode.ASPRS


class TestBuildingDetectionConfig:
    """Test BuildingDetectionConfig class."""
    
    def test_asprs_config(self):
        """Test ASPRS mode configuration."""
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.ASPRS)
        assert config.mode == BuildingDetectionMode.ASPRS
        assert config.min_height == 2.5
        assert config.use_ground_truth is True
        assert config.use_wall_detection is True
    
    def test_lod2_config(self):
        """Test LOD2 mode configuration."""
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
        assert config.mode == BuildingDetectionMode.LOD2
        assert config.separate_walls_roofs is True
        assert config.detect_flat_roofs is True
        assert config.wall_verticality_min > 0.65
    
    def test_lod3_config(self):
        """Test LOD3 mode configuration."""
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD3)
        assert config.mode == BuildingDetectionMode.LOD3
        assert config.detect_windows is True
        assert config.detect_doors is True
        assert config.detect_balconies is True
        assert config.wall_verticality_min > 0.7


class TestBuildingDetector:
    """Test BuildingDetector class."""
    
    def test_detector_initialization_asprs(self):
        """Test detector initialization in ASPRS mode."""
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.ASPRS)
        detector = BuildingDetector(config=config)
        assert detector.config.mode == BuildingDetectionMode.ASPRS
    
    def test_detector_initialization_lod2(self):
        """Test detector initialization in LOD2 mode."""
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
        detector = BuildingDetector(config=config)
        assert detector.config.mode == BuildingDetectionMode.LOD2
    
    def test_detector_initialization_lod3(self):
        """Test detector initialization in LOD3 mode."""
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD3)
        detector = BuildingDetector(config=config)
        assert detector.config.mode == BuildingDetectionMode.LOD3
    
    def test_asprs_detection(self, sample_building_points):
        """Test ASPRS mode building detection."""
        labels, features, building_mask = sample_building_points
        
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.ASPRS)
        detector = BuildingDetector(config=config)
        
        refined, stats = detector.detect_buildings(
            labels=labels,
            height=features['height'],
            planarity=features['planarity'],
            verticality=features['verticality'],
            normals=features['normals']
        )
        
        # Check that some buildings were detected
        assert stats['total'] > 0
        assert (refined == 6).any()  # ASPRS building class is 6
    
    def test_lod2_detection(self, sample_building_points):
        """Test LOD2 mode building detection."""
        labels, features, building_mask = sample_building_points
        
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
        detector = BuildingDetector(config=config)
        
        refined, stats = detector.detect_buildings(
            labels=labels,
            height=features['height'],
            planarity=features['planarity'],
            verticality=features['verticality'],
            normals=features['normals'],
            linearity=features['linearity'],
            anisotropy=features['anisotropy']
        )
        
        # Check that building elements were detected
        assert stats['total_building'] > 0
        # LOD2 classes: 0=wall, 1=roof_flat, 2=roof_gable
        assert np.isin(refined, [0, 1, 2, 3]).any()
    
    def test_lod3_detection(self, sample_building_points):
        """Test LOD3 mode building detection."""
        labels, features, building_mask = sample_building_points
        
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD3)
        detector = BuildingDetector(config=config)
        
        refined, stats = detector.detect_buildings(
            labels=labels,
            height=features['height'],
            planarity=features['planarity'],
            verticality=features['verticality'],
            normals=features['normals'],
            linearity=features['linearity'],
            anisotropy=features['anisotropy'],
            curvature=features['curvature'],
            intensity=features['intensity'],
            points=features['points']
        )
        
        # Check that building elements were detected
        assert stats['total_building'] > 0
        # LOD3 includes all LOD2 classes plus details
        assert np.isin(refined, [0, 1, 2, 3, 13, 14, 15, 18, 20]).any()
    
    def test_ground_truth_override(self, sample_building_points):
        """Test that ground truth overrides geometric detection."""
        labels, features, building_mask = sample_building_points
        
        # Create ground truth mask
        ground_truth = np.zeros(len(labels), dtype=bool)
        ground_truth[:100] = True  # First 100 points are buildings
        
        config = BuildingDetectionConfig(mode=BuildingDetectionMode.ASPRS)
        detector = BuildingDetector(config=config)
        
        refined, stats = detector.detect_buildings(
            labels=labels,
            height=features['height'],
            planarity=features['planarity'],
            verticality=features['verticality'],
            ground_truth_mask=ground_truth
        )
        
        # Check that ground truth was applied
        assert (refined[:100] == 6).all()  # ASPRS building class
        assert stats['ground_truth'] == 100


class TestConvenienceFunction:
    """Test detect_buildings_multi_mode convenience function."""
    
    def test_asprs_mode(self, sample_building_points):
        """Test convenience function with ASPRS mode."""
        labels, features, _ = sample_building_points
        
        refined, stats = detect_buildings_multi_mode(
            labels=labels,
            features=features,
            mode='asprs'
        )
        
        assert stats['total'] > 0
        assert (refined == 6).any()
    
    def test_lod2_mode(self, sample_building_points):
        """Test convenience function with LOD2 mode."""
        labels, features, _ = sample_building_points
        
        refined, stats = detect_buildings_multi_mode(
            labels=labels,
            features=features,
            mode='lod2'
        )
        
        assert stats['total_building'] > 0
        assert np.isin(refined, [0, 1, 2, 3]).any()
    
    def test_lod3_mode(self, sample_building_points):
        """Test convenience function with LOD3 mode."""
        labels, features, _ = sample_building_points
        
        refined, stats = detect_buildings_multi_mode(
            labels=labels,
            features=features,
            mode='lod3'
        )
        
        assert stats['total_building'] > 0
    
    def test_invalid_mode(self, sample_building_points):
        """Test that invalid mode raises error."""
        labels, features, _ = sample_building_points
        
        with pytest.raises(ValueError):
            detect_buildings_multi_mode(
                labels=labels,
                features=features,
                mode='invalid_mode'
            )
    
    def test_missing_required_features(self):
        """Test that missing required features raises error."""
        labels = np.zeros(100, dtype=np.uint8)
        features = {}  # Empty features
        
        with pytest.raises(ValueError, match="Height and planarity"):
            detect_buildings_multi_mode(
                labels=labels,
                features=features,
                mode='asprs'
            )


class TestDetectionStrategies:
    """Test individual detection strategies."""
    
    def test_wall_detection(self, sample_building_points):
        """Test that walls are detected correctly."""
        labels, features, _ = sample_building_points
        
        # Create wall-like points (vertical, planar)
        features['verticality'][:100] = 0.9
        features['planarity'][:100] = 0.8
        features['height'][:100] = 5.0
        
        refined, stats = detect_buildings_multi_mode(
            labels=labels,
            features=features,
            mode='lod2'
        )
        
        # Check that walls were detected
        assert stats['walls'] > 0
        assert (refined[:100] == 0).sum() > 0  # LOD2 wall class
    
    def test_roof_detection(self, sample_building_points):
        """Test that roofs are detected correctly."""
        labels, features, _ = sample_building_points
        
        # Create roof-like points (horizontal, planar)
        features['normals'][:100, 2] = 0.95  # Nearly vertical normal (horizontal surface)
        features['planarity'][:100] = 0.85
        features['height'][:100] = 8.0
        
        refined, stats = detect_buildings_multi_mode(
            labels=labels,
            features=features,
            mode='lod2'
        )
        
        # Check that roofs were detected
        assert stats['flat_roofs'] + stats['sloped_roofs'] > 0
        # LOD2 roof classes: 1, 2, 3
        assert np.isin(refined[:100], [1, 2, 3]).any()
    
    def test_edge_detection(self, sample_building_points):
        """Test that edges are detected correctly."""
        labels, features, _ = sample_building_points
        
        # Create edge-like points (high linearity)
        features['linearity'][:50] = 0.8
        features['height'][:50] = 5.0
        
        refined, stats = detect_buildings_multi_mode(
            labels=labels,
            features=features,
            mode='asprs'
        )
        
        # Check that edges were detected
        assert stats['edges'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
