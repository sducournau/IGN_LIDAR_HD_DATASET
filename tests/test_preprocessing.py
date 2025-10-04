"""
Tests for preprocessing module (artifact mitigation)
"""

import pytest
import numpy as np
from ign_lidar.preprocessing import (
    statistical_outlier_removal,
    radius_outlier_removal,
    voxel_downsample,
    preprocess_point_cloud,
    preprocess_for_features
)


class TestStatisticalOutlierRemoval:
    """Test Statistical Outlier Removal (SOR)"""
    
    def test_sor_removes_outliers(self):
        """Test that SOR removes clear outliers"""
        # Create clean cluster + outliers
        np.random.seed(42)
        
        # Main cluster
        cluster = np.random.randn(100, 3) * 0.1
        
        # Add outliers far from cluster
        outliers = np.array([
            [10, 10, 10],
            [-10, -10, -10],
            [0, 0, 20]
        ])
        
        points = np.vstack([cluster, outliers])
        
        # Apply SOR
        filtered, mask = statistical_outlier_removal(points, k=10, std_multiplier=2.0)
        
        # Should remove the 3 outliers
        assert len(filtered) < len(points)
        assert np.sum(~mask) >= 3  # At least 3 removed
        
        # Original cluster should be mostly intact
        assert np.sum(mask[:100]) > 90  # >90% of cluster kept
        
    def test_sor_preserves_clean_cloud(self):
        """Test that SOR doesn't remove points from clean cloud"""
        np.random.seed(42)
        points = np.random.randn(200, 3) * 0.5  # Uniform cloud
        
        filtered, mask = statistical_outlier_removal(points, k=12, std_multiplier=2.5)
        
        # Should keep most points (>95%)
        removal_rate = 1 - len(filtered) / len(points)
        assert removal_rate < 0.05
        
    def test_sor_with_few_points(self):
        """Test SOR with very few points"""
        points = np.random.randn(5, 3)
        
        # Should return all points with warning
        filtered, mask = statistical_outlier_removal(points, k=10)
        assert len(filtered) == len(points)
        assert np.all(mask)


class TestRadiusOutlierRemoval:
    """Test Radius Outlier Removal (ROR)"""
    
    def test_ror_removes_isolated_points(self):
        """Test that ROR removes isolated points"""
        # Dense cluster
        cluster = np.random.randn(100, 3) * 0.2
        
        # Isolated points (far from cluster)
        isolated = np.array([
            [5, 5, 5],
            [-5, -5, -5],
            [0, 10, 0]
        ])
        
        points = np.vstack([cluster, isolated])
        
        # Apply ROR with radius=1.0m, min_neighbors=4
        filtered, mask = radius_outlier_removal(points, radius=1.0, min_neighbors=4)
        
        # Should remove isolated points
        assert len(filtered) < len(points)
        assert np.sum(~mask) >= 3
        
    def test_ror_preserves_dense_areas(self):
        """Test that ROR keeps dense areas intact"""
        np.random.seed(42)
        # Create dense uniform cloud
        points = np.random.randn(500, 3) * 1.0
        
        filtered, mask = radius_outlier_removal(points, radius=1.5, min_neighbors=3)
        
        # Should keep most points
        removal_rate = 1 - len(filtered) / len(points)
        assert removal_rate < 0.15  # <15% removed


class TestVoxelDownsample:
    """Test Voxel Downsampling"""
    
    def test_voxel_reduces_points(self):
        """Test that voxelization reduces point count"""
        np.random.seed(42)
        # Dense cloud
        points = np.random.randn(10000, 3) * 5.0
        
        # Downsample with voxel_size=1.0m (larger to ensure reduction)
        downsampled, voxel_idx = voxel_downsample(points, voxel_size=1.0)
        
        # Should reduce significantly (45%+ reduction)
        assert len(downsampled) < len(points)
        reduction_ratio = 1 - len(downsampled) / len(points)
        assert reduction_ratio > 0.45  # At least 45% reduction
        
    def test_voxel_centroid_method(self):
        """Test centroid method produces averaged positions"""
        # Create 8 points in same voxel
        points = np.array([
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.3],
            [0.3, 0.2, 0.1],
            [0.1, 0.3, 0.2],
            [0.3, 0.1, 0.2]
        ])
        
        downsampled, _ = voxel_downsample(points, voxel_size=1.0, method='centroid')
        
        # All points in same voxel -> should produce 1 centroid
        assert len(downsampled) == 1
        
        # Centroid should be near mean
        expected_centroid = points.mean(axis=0)
        np.testing.assert_allclose(downsampled[0], expected_centroid, atol=0.01)
        
    def test_voxel_random_method(self):
        """Test random method returns one point per voxel"""
        np.random.seed(42)
        points = np.random.randn(1000, 3) * 5.0
        
        downsampled, _ = voxel_downsample(points, voxel_size=0.5, method='random')
        
        # Should produce fewer points
        assert len(downsampled) < len(points)
        
        # Each downsampled point should be from original set
        for ds_point in downsampled[:10]:  # Check first 10
            # At least one original point should be very close (same voxel)
            distances = np.linalg.norm(points - ds_point, axis=1)
            assert np.min(distances) < 0.01  # Should be exact match
            
    def test_voxel_preserves_sparse_clouds(self):
        """Test voxelization doesn't over-reduce sparse clouds"""
        # Sparse cloud (points far apart)
        points = np.array([
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [2, 2, 0],
            [2, 0, 2],
            [0, 2, 2],
            [2, 2, 2]
        ], dtype=np.float32)
        
        # Voxel size smaller than spacing
        downsampled, _ = voxel_downsample(points, voxel_size=0.5)
        
        # Should keep all points (each in different voxel)
        assert len(downsampled) == len(points)


class TestPreprocessPipeline:
    """Test full preprocessing pipeline"""
    
    def test_default_pipeline(self):
        """Test default preprocessing configuration"""
        np.random.seed(42)
        
        # Create test cloud with outliers
        clean = np.random.randn(500, 3) * 0.5
        outliers = np.array([[10, 10, 10], [-10, -10, -10]])
        points = np.vstack([clean, outliers])
        
        # Apply default preprocessing
        processed, stats = preprocess_point_cloud(points)
        
        # Check stats
        assert stats['original_points'] == len(points)
        assert stats['final_points'] < stats['original_points']
        assert 0 < stats['reduction_ratio'] < 1
        assert stats['processing_time_ms'] > 0
        
        # Should have removed some points
        assert 'sor_removed' in stats or 'ror_removed' in stats
        
    def test_custom_config(self):
        """Test custom preprocessing configuration"""
        np.random.seed(42)
        points = np.random.randn(1000, 3) * 2.0
        
        config = {
            'sor': {'enable': True, 'k': 10, 'std_multiplier': 2.5},
            'ror': {'enable': True, 'radius': 1.5, 'min_neighbors': 3},
            'voxel': {'enable': True, 'voxel_size': 0.3, 'method': 'centroid'}
        }
        
        processed, stats = preprocess_point_cloud(points, config)
        
        # Should have applied all three steps
        assert 'sor_removed' in stats
        assert 'ror_removed' in stats
        assert 'voxel_reduced' in stats
        
        # Should have reduced point count
        assert len(processed) < len(points)
        
    def test_disable_all_filters(self):
        """Test with all filters disabled"""
        np.random.seed(42)
        points = np.random.randn(200, 3)
        
        config = {
            'sor': {'enable': False},
            'ror': {'enable': False},
            'voxel': {'enable': False}
        }
        
        processed, stats = preprocess_point_cloud(points, config)
        
        # Should return unchanged points
        assert len(processed) == len(points)
        assert stats['reduction_ratio'] == 0.0
        np.testing.assert_array_equal(processed, points)
        
    def test_sor_only(self):
        """Test with only SOR enabled"""
        np.random.seed(42)
        points = np.random.randn(300, 3) * 1.0
        
        config = {
            'sor': {'enable': True, 'k': 12, 'std_multiplier': 2.0},
            'ror': {'enable': False},
            'voxel': {'enable': False}
        }
        
        processed, stats = preprocess_point_cloud(points, config)
        
        # Should have SOR stats but not others
        assert 'sor_removed' in stats
        assert 'ror_removed' not in stats
        assert 'voxel_reduced' not in stats


class TestConvenienceFunctions:
    """Test convenience preprocessing functions"""
    
    def test_preprocess_for_features_standard(self):
        """Test standard mode"""
        np.random.seed(42)
        points = np.random.randn(500, 3) * 1.0
        
        processed = preprocess_for_features(points, mode='standard')
        
        assert len(processed) <= len(points)
        assert len(processed) > 0
        
    def test_preprocess_for_features_light(self):
        """Test light mode (minimal filtering)"""
        np.random.seed(42)
        points = np.random.randn(500, 3) * 1.0
        
        processed = preprocess_for_features(points, mode='light')
        
        # Light mode should keep more points
        reduction = 1 - len(processed) / len(points)
        assert reduction < 0.1  # <10% removed
        
    def test_preprocess_for_features_aggressive(self):
        """Test aggressive mode (strong filtering)"""
        np.random.seed(42)
        points = np.random.randn(1000, 3) * 2.0
        
        processed = preprocess_for_features(points, mode='aggressive')
        
        # Aggressive mode should remove more points
        reduction = 1 - len(processed) / len(points)
        assert reduction > 0.05  # >5% removed (including voxel)
        
    def test_invalid_mode(self):
        """Test invalid mode raises error"""
        points = np.random.randn(100, 3)
        
        with pytest.raises(ValueError):
            preprocess_for_features(points, mode='invalid_mode')


class TestRealWorldScenario:
    """Test preprocessing on realistic scenarios"""
    
    def test_building_extraction_scenario(self):
        """Test preprocessing for building extraction"""
        np.random.seed(42)
        
        # Simulate building roof (planar surface)
        roof_points = np.random.randn(800, 3) * 0.1
        roof_points[:, 2] += 15.0  # Elevate to 15m
        
        # Add some outliers (birds, noise)
        outliers = np.array([
            [0, 0, 25],  # Bird flying over
            [0, 0, 5],   # Ground reflection
            [10, 10, 15],  # Isolated noise
        ])
        
        # Add some edge points (building corners)
        edges = np.random.randn(50, 3) * 0.05
        edges[:, 2] += 15.0
        edges[:, 0] += 5.0  # Offset to edge
        
        points = np.vstack([roof_points, outliers, edges])
        
        # Preprocess
        processed, stats = preprocess_point_cloud(points)
        
        # Should remove outliers but keep roof + edges
        assert len(processed) < len(points)
        assert len(processed) > 800  # Most of roof + edges kept
        
        # Reduction should be moderate (mostly outliers)
        # Note: With gentle default settings, reduction may be minimal
        assert 0 < stats['reduction_ratio'] < 0.15
        
    def test_vegetation_scenario(self):
        """Test preprocessing for vegetation (organic structures)"""
        np.random.seed(42)
        
        # Simulate tree (scattered points)
        tree_points = np.random.randn(1000, 3) * 1.5
        tree_points[:, 2] = np.abs(tree_points[:, 2])  # Positive Z
        
        # Add some ground noise
        ground_noise = np.random.randn(20, 3) * 0.1
        ground_noise[:, 2] = -0.5  # Below tree
        
        points = np.vstack([tree_points, ground_noise])
        
        # Preprocess with gentle settings
        from ign_lidar.preprocessing import preprocess_for_natural
        processed = preprocess_for_natural(points)
        
        # Should keep most tree structure
        assert len(processed) > 900  # >90% kept
        assert len(processed) < len(points)  # Some noise removed


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_point_cloud(self):
        """Test with empty point cloud"""
        points = np.array([]).reshape(0, 3)
        
        filtered, mask = radius_outlier_removal(points, radius=1.0)
        assert len(filtered) == 0
        assert len(mask) == 0
        
    def test_single_point(self):
        """Test with single point"""
        points = np.array([[0, 0, 0]])
        
        # SOR should return the point
        filtered, mask = statistical_outlier_removal(points, k=5)
        assert len(filtered) == 1
        assert mask[0] == True
        
    def test_very_large_cloud(self):
        """Test preprocessing scales to large clouds"""
        np.random.seed(42)
        # Simulate 1M point cloud (typical LiDAR tile)
        points = np.random.randn(100000, 3).astype(np.float32) * 10.0
        
        # Should complete without error
        processed, stats = preprocess_point_cloud(points)
        
        assert stats['processing_time_ms'] < 60000  # <60 seconds
        assert len(processed) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
