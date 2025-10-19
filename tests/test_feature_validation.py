"""
Unit tests for feature validation module.

Tests the FeatureValidator class and FeatureSignature dataclass.
"""

import pytest
import numpy as np
from typing import Dict

from ign_lidar.core.modules.feature_validator import (
    FeatureValidator,
    FeatureSignature
)


class TestFeatureSignature:
    """Test FeatureSignature class."""
    
    def test_vegetation_signature_match(self):
        """Test vegetation feature signature detection."""
        signature = FeatureSignature(
            curvature_min=0.15,
            planarity_max=0.70,
            ndvi_min=0.20
        )
        
        # Vegetation-like features
        veg_features = {
            'curvature': 0.35,
            'planarity': 0.45,
            'ndvi': 0.55
        }
        matches, conf = signature.matches(veg_features)
        assert matches is True
        assert conf > 0.9
    
    def test_vegetation_signature_no_match(self):
        """Test vegetation signature with non-vegetation features."""
        signature = FeatureSignature(
            curvature_min=0.15,
            planarity_max=0.70,
            ndvi_min=0.20
        )
        
        # Building-like features (flat, low NDVI)
        building_features = {
            'curvature': 0.05,
            'planarity': 0.90,
            'ndvi': 0.10
        }
        matches, conf = signature.matches(building_features)
        assert matches is False
        assert conf < 0.5
    
    def test_building_signature_match(self):
        """Test building feature signature detection."""
        signature = FeatureSignature(
            curvature_max=0.10,
            planarity_min=0.70,
            verticality_min=0.60
        )
        
        building_features = {
            'curvature': 0.05,
            'planarity': 0.85,
            'verticality': 0.75
        }
        matches, conf = signature.matches(building_features)
        assert matches is True
        assert conf > 0.9
    
    def test_road_signature_match(self):
        """Test road feature signature detection."""
        signature = FeatureSignature(
            curvature_max=0.05,
            planarity_min=0.85,
            normal_z_min=0.90,
            height_max=2.0
        )
        
        road_features = {
            'curvature': 0.02,
            'planarity': 0.92,
            'normal_z': 0.95,
            'height': 0.5
        }
        matches, conf = signature.matches(road_features)
        assert matches is True
        assert conf == 1.0
    
    def test_partial_feature_match(self):
        """Test signature matching with partial features."""
        signature = FeatureSignature(
            curvature_min=0.10,
            planarity_max=0.60,
            ndvi_min=0.20
        )
        
        # Only provide curvature and ndvi (missing planarity)
        partial_features = {
            'curvature': 0.25,
            'ndvi': 0.45
        }
        matches, conf = signature.matches(partial_features)
        assert matches is True  # 2/2 checks passed
        assert conf == 1.0


class TestFeatureValidator:
    """Test FeatureValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create validator with default config."""
        return FeatureValidator()
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature arrays."""
        n_points = 100
        return {
            'curvature': np.random.rand(n_points),
            'planarity': np.random.rand(n_points),
            'verticality': np.random.rand(n_points),
            'normals': np.random.randn(n_points, 3),
            'height': np.random.rand(n_points) * 10,
            'ndvi': np.random.rand(n_points),
            'nir': np.random.rand(n_points),
            'intensity': np.random.rand(n_points) * 255,
            'brightness': np.random.rand(n_points)
        }
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.vegetation_signature is not None
        assert validator.building_signature is not None
        assert validator.road_signature is not None
        assert validator.min_confidence > 0
    
    def test_vegetation_signature_check(self, validator):
        """Test vegetation signature checker."""
        features = {
            'curvature': np.array([0.35, 0.05]),
            'planarity': np.array([0.45, 0.90]),
            'ndvi': np.array([0.55, 0.10])
        }
        
        is_veg, conf = validator.check_vegetation_signature(features)
        assert is_veg[0] == True   # First point is vegetation
        assert is_veg[1] == False  # Second point is not vegetation
        assert conf[0] > 0.7
        assert conf[1] < 0.5
    
    def test_building_signature_check(self, validator):
        """Test building signature checker."""
        features = {
            'curvature': np.array([0.05, 0.40]),
            'planarity': np.array([0.85, 0.30]),
            'verticality': np.array([0.75, 0.20]),
            'height': np.array([5.0, 1.0])
        }
        
        is_building, conf = validator.check_building_signature(features)
        assert is_building[0] == True   # First point is building
        assert is_building[1] == False  # Second point is not building
    
    def test_road_signature_check(self, validator):
        """Test road signature checker."""
        features = {
            'curvature': np.array([0.02, 0.30]),
            'planarity': np.array([0.92, 0.40]),
            'normal_z': np.array([0.95, 0.50]),
            'height': np.array([0.5, 5.0])
        }
        
        is_road, conf = validator.check_road_signature(features)
        assert is_road[0] == True   # First point is road
        assert is_road[1] == False  # Second point is not road
    
    def test_validate_building_roof_vegetation(self, validator):
        """Test roof vegetation detection on buildings."""
        # Ground truth says building, but features indicate vegetation
        labels = np.array([6])  # ASPRS_BUILDING
        ground_truth_types = np.array(['building'], dtype=object)
        
        features = {
            'curvature': np.array([0.30]),   # High curvature (vegetation)
            'planarity': np.array([0.50]),   # Low planarity (vegetation)
            'ndvi': np.array([0.65]),        # High NDVI (vegetation)
            'nir': np.array([0.80]),         # High NIR (vegetation)
            'height': np.array([5.0]),       # On roof
            'normals': np.array([[0, 0, 0.6]])
        }
        
        validated, confidence, valid = validator.validate_ground_truth(
            labels, ground_truth_types, features
        )
        
        # Should be reclassified as vegetation (4 or 5)
        assert validated[0] in [4, 5]  # MEDIUM or HIGH vegetation
        assert confidence[0] > 0.6
        assert valid[0] == True
    
    def test_validate_tree_canopy_over_road(self, validator):
        """Test tree canopy detection over roads."""
        labels = np.array([11])  # ASPRS_ROAD
        ground_truth_types = np.array(['road'], dtype=object)
        
        features = {
            'curvature': np.array([0.35]),   # High curvature (vegetation)
            'planarity': np.array([0.45]),   # Low planarity (vegetation)
            'ndvi': np.array([0.60]),        # High NDVI (vegetation)
            'nir': np.array([0.75]),
            'height': np.array([8.0]),       # Tree height
            'normals': np.array([[0, 0, 0.5]])
        }
        
        validated, confidence, valid = validator.validate_ground_truth(
            labels, ground_truth_types, features
        )
        
        # Should be reclassified as high vegetation
        assert validated[0] == 5  # HIGH vegetation
        assert confidence[0] > 0.6
    
    def test_validate_correct_building(self, validator):
        """Test validation accepts correct building label."""
        labels = np.array([6])  # ASPRS_BUILDING
        ground_truth_types = np.array(['building'], dtype=object)
        
        features = {
            'curvature': np.array([0.05]),      # Low curvature (building)
            'planarity': np.array([0.85]),      # High planarity (building)
            'verticality': np.array([0.75]),    # High verticality (building)
            'ndvi': np.array([0.10]),           # Low NDVI (not vegetation)
            'height': np.array([5.0]),
            'normals': np.array([[0, 0, 0.2]])
        }
        
        validated, confidence, valid = validator.validate_ground_truth(
            labels, ground_truth_types, features
        )
        
        # Should keep building label
        assert validated[0] == 6  # ASPRS_BUILDING
        assert confidence[0] > 0.7
        assert valid[0] == True
    
    def test_validate_correct_road(self, validator):
        """Test validation accepts correct road label."""
        labels = np.array([11])  # ASPRS_ROAD
        ground_truth_types = np.array(['road'], dtype=object)
        
        features = {
            'curvature': np.array([0.02]),
            'planarity': np.array([0.92]),
            'normal_z': np.array([0.95]),
            'ndvi': np.array([0.08]),
            'height': np.array([0.3]),
            'normals': np.array([[0, 0, 0.95]])
        }
        
        validated, confidence, valid = validator.validate_ground_truth(
            labels, ground_truth_types, features
        )
        
        assert validated[0] == 11  # ASPRS_ROAD
        assert confidence[0] > 0.8
        assert valid[0] == True
    
    def test_filter_false_positives(self, validator):
        """Test false positive filtering."""
        # Mix of correct and incorrect labels
        labels = np.array([6, 6, 11, 11])  # Buildings and roads
        ground_truth_mask = np.array([True, True, True, True])
        ground_truth_types = np.array(['building', 'building', 'road', 'road'], dtype=object)
        
        features = {
            'curvature': np.array([0.05, 0.35, 0.02, 0.30]),
            'planarity': np.array([0.85, 0.45, 0.92, 0.40]),
            'verticality': np.array([0.75, 0.30, 0.20, 0.20]),
            'ndvi': np.array([0.10, 0.65, 0.08, 0.60]),
            'nir': np.array([0.30, 0.80, 0.25, 0.75]),
            'height': np.array([5.0, 5.0, 0.3, 8.0]),
            'normals': np.array([[0, 0, 0.2], [0, 0, 0.5], [0, 0, 0.95], [0, 0, 0.6]])
        }
        
        filtered_labels, filtered_mask = validator.filter_ground_truth_false_positives(
            labels, ground_truth_mask, features, ground_truth_types
        )
        
        # First building and road should be validated
        # Second building (roof veg) and road (tree canopy) should be reclassified
        assert filtered_labels[0] == 6   # Correct building
        assert filtered_labels[1] in [4, 5]  # Roof vegetation
        assert filtered_labels[2] == 11  # Correct road
        assert filtered_labels[3] == 5   # Tree canopy
    
    def test_classify_vegetation_by_height(self, validator):
        """Test vegetation height classification."""
        assert validator._classify_vegetation_by_height(0.3) == 3   # Low
        assert validator._classify_vegetation_by_height(1.0) == 4   # Medium
        assert validator._classify_vegetation_by_height(5.0) == 5   # High
    
    def test_validate_with_missing_features(self, validator, sample_features):
        """Test validation handles missing features gracefully."""
        labels = np.full(100, 6)  # All buildings
        ground_truth_types = np.full(100, 'building', dtype=object)
        
        # Remove some features
        limited_features = {
            'curvature': sample_features['curvature'],
            'planarity': sample_features['planarity'],
            'height': sample_features['height']
        }
        
        # Should not raise error
        validated, confidence, valid = validator.validate_ground_truth(
            labels, ground_truth_types, limited_features
        )
        
        assert len(validated) == 100
        assert len(confidence) == 100
        assert len(valid) == 100
    
    def test_custom_config(self):
        """Test validator with custom configuration."""
        config = {
            'min_validation_confidence': 0.8,
            'strict_validation': True,
            'vegetation_signature': {
                'curvature_min': 0.20,
                'ndvi_min': 0.30
            }
        }
        
        validator = FeatureValidator(config)
        assert validator.min_confidence == 0.8
        assert validator.strict_validation is True
        assert validator.vegetation_signature.curvature_min == 0.20


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_mixed_urban_scene(self):
        """Test validation on mixed urban scene."""
        validator = FeatureValidator()
        
        # Simulate mixed scene: buildings, roads, trees, roof vegetation
        n_points = 500
        labels = np.concatenate([
            np.full(100, 6),   # Buildings
            np.full(100, 11),  # Roads
            np.full(100, 5),   # Trees
            np.full(100, 6),   # Roof vegetation (labeled as building)
            np.full(100, 11)   # Tree canopy over road (labeled as road)
        ])
        
        ground_truth_types = np.concatenate([
            np.full(100, 'building', dtype=object),
            np.full(100, 'road', dtype=object),
            np.full(100, 'vegetation', dtype=object),
            np.full(100, 'building', dtype=object),  # False positive
            np.full(100, 'road', dtype=object)       # False positive
        ])
        
        # Generate realistic features
        features = {
            'curvature': np.concatenate([
                np.random.uniform(0.0, 0.1, 100),    # Buildings: low curvature
                np.random.uniform(0.0, 0.05, 100),   # Roads: very low curvature
                np.random.uniform(0.2, 0.5, 100),    # Trees: high curvature
                np.random.uniform(0.25, 0.45, 100),  # Roof veg: high curvature
                np.random.uniform(0.3, 0.5, 100)     # Tree canopy: high curvature
            ]),
            'planarity': np.concatenate([
                np.random.uniform(0.7, 0.95, 100),   # Buildings: high planarity
                np.random.uniform(0.85, 0.98, 100),  # Roads: very high planarity
                np.random.uniform(0.2, 0.6, 100),    # Trees: low planarity
                np.random.uniform(0.3, 0.6, 100),    # Roof veg: low planarity
                np.random.uniform(0.25, 0.55, 100)   # Tree canopy: low planarity
            ]),
            'ndvi': np.concatenate([
                np.random.uniform(0.0, 0.15, 100),   # Buildings: low NDVI
                np.random.uniform(0.0, 0.12, 100),   # Roads: low NDVI
                np.random.uniform(0.5, 0.75, 100),   # Trees: high NDVI
                np.random.uniform(0.55, 0.8, 100),   # Roof veg: high NDVI
                np.random.uniform(0.5, 0.7, 100)     # Tree canopy: high NDVI
            ]),
            'height': np.concatenate([
                np.random.uniform(3.0, 15.0, 100),   # Buildings
                np.random.uniform(0.0, 1.0, 100),    # Roads
                np.random.uniform(3.0, 20.0, 100),   # Trees
                np.random.uniform(4.0, 12.0, 100),   # Roof veg (on roofs)
                np.random.uniform(5.0, 15.0, 100)    # Tree canopy
            ]),
            'verticality': np.concatenate([
                np.random.uniform(0.6, 0.9, 100),    # Buildings: high
                np.random.uniform(0.1, 0.3, 100),    # Roads: low
                np.random.uniform(0.3, 0.7, 100),    # Trees: medium
                np.random.uniform(0.4, 0.7, 100),    # Roof veg: medium
                np.random.uniform(0.3, 0.6, 100)     # Tree canopy: medium
            ]),
            'normals': np.vstack([
                np.column_stack([np.zeros(100), np.zeros(100), np.random.uniform(0.1, 0.3, 100)]),
                np.column_stack([np.zeros(100), np.zeros(100), np.random.uniform(0.9, 0.99, 100)]),
                np.column_stack([np.zeros(100), np.zeros(100), np.random.uniform(0.4, 0.7, 100)]),
                np.column_stack([np.zeros(100), np.zeros(100), np.random.uniform(0.3, 0.6, 100)]),
                np.column_stack([np.zeros(100), np.zeros(100), np.random.uniform(0.4, 0.7, 100)])
            ]),
            'nir': np.concatenate([
                np.random.uniform(0.2, 0.4, 100),
                np.random.uniform(0.15, 0.35, 100),
                np.random.uniform(0.6, 0.9, 100),
                np.random.uniform(0.65, 0.9, 100),
                np.random.uniform(0.6, 0.85, 100)
            ])
        }
        
        # Validate
        validated, confidence, valid = validator.validate_ground_truth(
            labels, ground_truth_types, features
        )
        
        # Check correct classifications maintained
        assert np.all(validated[:100] == 6)    # Buildings stay as buildings
        assert np.all(validated[100:200] == 11)  # Roads stay as roads
        
        # Check vegetation correctly classified
        assert np.all(validated[200:300] >= 3) and np.all(validated[200:300] <= 5)
        
        # Check false positives corrected
        # Roof vegetation (points 300-400) should be reclassified to vegetation
        roof_veg_labels = validated[300:400]
        n_corrected_roof = np.sum((roof_veg_labels >= 3) & (roof_veg_labels <= 5))
        assert n_corrected_roof > 50  # At least half should be corrected
        
        # Tree canopy (points 400-500) should be reclassified to vegetation
        canopy_labels = validated[400:500]
        n_corrected_canopy = np.sum((canopy_labels >= 3) & (canopy_labels <= 5))
        assert n_corrected_canopy > 50
        
        # Check confidence scores
        assert np.mean(confidence[:300]) > 0.6  # High confidence for correct labels
        
        print(f"Integration test results:")
        print(f"  Roof vegetation corrections: {n_corrected_roof}/100")
        print(f"  Tree canopy corrections: {n_corrected_canopy}/100")
        print(f"  Mean confidence: {np.mean(confidence):.2f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
