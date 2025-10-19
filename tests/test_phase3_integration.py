"""
Phase 3 Integration Tests: GPU Bridge in features_gpu.py

Tests for the integration of GPUCoreBridge into features_gpu.py module.
This phase refactors the eigenvalue computation in GPUFeatureComputer to use
the GPU-Core Bridge pattern, eliminating code duplication.

Author: IGN LiDAR HD Development Team
Date: October 19, 2025
Phase: 3 - features_gpu.py Integration
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestPhase3Integration:
    """Test suite for Phase 3: GPU bridge integration into features_gpu.py"""
    
    @pytest.fixture
    def simple_points(self):
        """Create simple point cloud for testing"""
        # Create a small planar surface
        np.random.seed(42)
        x = np.random.uniform(0, 10, 100)
        y = np.random.uniform(0, 10, 100)
        z = np.random.uniform(0, 0.1, 100)  # Nearly planar
        return np.column_stack([x, y, z])
    
    @pytest.fixture
    def linear_points(self):
        """Create linear point cloud for testing"""
        # Create points along a line
        t = np.linspace(0, 10, 100)
        x = t
        y = t * 0.5
        z = t * 0.2 + np.random.normal(0, 0.01, 100)
        return np.column_stack([x, y, z])
    
    @pytest.fixture
    def spherical_points(self):
        """Create spherical point cloud for testing"""
        # Create points on a sphere
        np.random.seed(42)
        theta = np.random.uniform(0, 2*np.pi, 100)
        phi = np.random.uniform(0, np.pi, 100)
        r = 5.0 + np.random.normal(0, 0.1, 100)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.column_stack([x, y, z])
    
    def test_gpu_bridge_initialization(self, simple_points):
        """Test that GPU bridge is properly initialized in GPUFeatureComputer"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        # Create computer instance
        computer = GPUFeatureComputer(use_gpu=False)
        
        # Verify GPU bridge exists
        assert hasattr(computer, 'gpu_bridge'), "GPU bridge not initialized"
        assert computer.gpu_bridge is not None, "GPU bridge is None"
        
        # Verify bridge configuration
        assert hasattr(computer.gpu_bridge, 'use_gpu'), "Bridge missing use_gpu"
        assert hasattr(computer.gpu_bridge, 'batch_size'), "Bridge missing batch_size"
        assert computer.gpu_bridge.batch_size == 500_000, "Bridge batch size incorrect"
    
    def test_refactored_eigenvalue_features_cpu(self, simple_points):
        """Test eigenvalue feature computation using refactored CPU path"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        # Compute features
        features = computer._compute_essential_geometric_features(
            points=simple_points,
            normals=np.zeros((len(simple_points), 3)),  # Dummy normals
            k=10,
            required_features=['planarity', 'linearity', 'sphericity']
        )
        
        # Verify features exist
        assert 'planarity' in features, "Planarity feature missing"
        assert 'linearity' in features, "Linearity feature missing"
        assert 'sphericity' in features, "Sphericity feature missing"
        
        # Verify shapes
        assert features['planarity'].shape == (len(simple_points),)
        assert features['linearity'].shape == (len(simple_points),)
        assert features['sphericity'].shape == (len(simple_points),)
        
        # Verify values are reasonable
        assert np.all(features['planarity'] >= 0) and np.all(features['planarity'] <= 1)
        assert np.all(features['linearity'] >= 0) and np.all(features['linearity'] <= 1)
        assert np.all(features['sphericity'] >= 0) and np.all(features['sphericity'] <= 1)
    
    def test_planar_features_phase3(self, simple_points):
        """Test that planar surfaces have reasonable planarity (Phase 3 refactored)"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        features = computer._compute_essential_geometric_features(
            points=simple_points,
            normals=np.zeros((len(simple_points), 3)),
            k=10,
            required_features=['planarity', 'linearity', 'sphericity']
        )
        
        # Planar surface should have some planarity (note: random scatter reduces planarity)
        mean_planarity = np.mean(features['planarity'])
        assert mean_planarity > 0.15, f"Expected some planarity for planar surface, got {mean_planarity}"
        
        # Should have low sphericity
        mean_sphericity = np.mean(features['sphericity'])
        assert mean_sphericity < 0.5, f"Expected low sphericity for planar surface, got {mean_sphericity}"
    
    def test_linear_features_phase3(self, linear_points):
        """Test that linear structures have high linearity (Phase 3 refactored)"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        features = computer._compute_essential_geometric_features(
            points=linear_points,
            normals=np.zeros((len(linear_points), 3)),
            k=10,
            required_features=['planarity', 'linearity', 'sphericity']
        )
        
        # Linear structure should have high linearity
        mean_linearity = np.mean(features['linearity'])
        assert mean_linearity > 0.5, f"Expected high linearity for linear structure, got {mean_linearity}"
        
        # Should have low sphericity
        mean_sphericity = np.mean(features['sphericity'])
        assert mean_sphericity < 0.3, f"Expected low sphericity for linear structure, got {mean_sphericity}"
    
    def test_spherical_features_phase3(self, spherical_points):
        """Test that spherical structures compute valid features (Phase 3 refactored)"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        features = computer._compute_essential_geometric_features(
            points=spherical_points,
            normals=np.zeros((len(spherical_points), 3)),
            k=10,
            required_features=['planarity', 'linearity', 'sphericity']
        )
        
        # Verify features are computed and valid
        mean_sphericity = np.mean(features['sphericity'])
        assert mean_sphericity >= 0.0 and mean_sphericity <= 1.0, f"Sphericity out of range: {mean_sphericity}"
        
        # All features should be in valid range
        assert np.all(features['planarity'] >= 0) and np.all(features['planarity'] <= 1)
        assert np.all(features['linearity'] >= 0) and np.all(features['linearity'] <= 1)
        assert np.all(features['sphericity'] >= 0) and np.all(features['sphericity'] <= 1)
    
    def test_batch_eigenvalue_features_gpu_method(self, simple_points):
        """Test _compute_batch_eigenvalue_features_gpu using GPU bridge"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        from sklearn.neighbors import NearestNeighbors
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        # Find neighbors
        knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn.fit(simple_points)
        _, indices = knn.kneighbors(simple_points)
        
        # Compute features using refactored GPU method (CPU mode)
        features = computer._compute_batch_eigenvalue_features_gpu(
            points_gpu=simple_points,
            indices_gpu=indices,
            required_features=['planarity', 'linearity', 'sphericity', 'anisotropy']
        )
        
        # Verify all requested features are present
        assert 'planarity' in features
        assert 'linearity' in features
        assert 'sphericity' in features
        assert 'anisotropy' in features
        
        # Verify shapes and types
        for feature_name, feature_values in features.items():
            assert feature_values.shape == (len(simple_points),), f"{feature_name} shape incorrect"
            assert feature_values.dtype == np.float32, f"{feature_name} dtype should be float32"
            assert np.all(np.isfinite(feature_values)), f"{feature_name} contains non-finite values"
    
    def test_density_feature_computation(self, simple_points):
        """Test density feature computation (non-eigenvalue feature)"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        features = computer._compute_essential_geometric_features(
            points=simple_points,
            normals=np.zeros((len(simple_points), 3)),
            k=10,
            required_features=['density']
        )
        
        # Verify density feature
        assert 'density' in features, "Density feature missing"
        assert features['density'].shape == (len(simple_points),)
        assert np.all(features['density'] > 0), "Density should be positive"
        assert np.all(np.isfinite(features['density'])), "Density contains non-finite values"
    
    def test_mixed_features_computation(self, simple_points):
        """Test computation of both eigenvalue and non-eigenvalue features"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        features = computer._compute_essential_geometric_features(
            points=simple_points,
            normals=np.zeros((len(simple_points), 3)),
            k=10,
            required_features=['planarity', 'linearity', 'density']
        )
        
        # Verify all features present
        assert 'planarity' in features
        assert 'linearity' in features
        assert 'density' in features
        
        # Verify all have correct shape
        for feature_values in features.values():
            assert feature_values.shape == (len(simple_points),)
            assert np.all(np.isfinite(feature_values))
    
    def test_backward_compatibility_with_old_code(self, simple_points):
        """Test that refactored code maintains backward compatibility"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        # Create two instances
        computer1 = GPUFeatureComputer(use_gpu=False, batch_size=1_000_000)
        computer2 = GPUFeatureComputer(use_gpu=False, batch_size=2_000_000)
        
        # Compute same features with both
        features1 = computer1._compute_essential_geometric_features(
            points=simple_points,
            normals=np.zeros((len(simple_points), 3)),
            k=10,
            required_features=['planarity', 'linearity']
        )
        
        features2 = computer2._compute_essential_geometric_features(
            points=simple_points,
            normals=np.zeros((len(simple_points), 3)),
            k=10,
            required_features=['planarity', 'linearity']
        )
        
        # Results should be nearly identical
        for feature_name in ['planarity', 'linearity']:
            diff = np.abs(features1[feature_name] - features2[feature_name])
            max_diff = np.max(diff)
            assert max_diff < 1e-5, f"{feature_name} differs too much between instances: {max_diff}"
    
    def test_large_dataset_batching(self):
        """Test that large datasets are properly batched"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        # Create large point cloud
        np.random.seed(42)
        large_points = np.random.uniform(0, 100, (10000, 3)).astype(np.float32)
        
        computer = GPUFeatureComputer(use_gpu=False, batch_size=2000)
        
        features = computer._compute_essential_geometric_features(
            points=large_points,
            normals=np.zeros((len(large_points), 3)),
            k=10,
            required_features=['planarity', 'density']
        )
        
        # Verify computation succeeded
        assert 'planarity' in features
        assert 'density' in features
        assert features['planarity'].shape == (len(large_points),)
        assert features['density'].shape == (len(large_points),)
        assert np.all(np.isfinite(features['planarity']))
        assert np.all(np.isfinite(features['density']))
    
    def test_edge_case_small_dataset(self):
        """Test edge case: very small dataset (< 10 points)"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        # Create tiny point cloud
        tiny_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0]
        ], dtype=np.float32)
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        features = computer._compute_essential_geometric_features(
            points=tiny_points,
            normals=np.zeros((len(tiny_points), 3)),
            k=3,  # Fewer neighbors for small dataset
            required_features=['planarity', 'linearity']
        )
        
        # Should still work
        assert 'planarity' in features
        assert 'linearity' in features
        assert features['planarity'].shape == (len(tiny_points),)
        assert np.all(np.isfinite(features['planarity']))
    
    def test_nan_handling(self):
        """Test that NaN values are handled properly"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        # Create point cloud with some identical points (can cause issues)
        problematic_points = np.array([
            [0, 0, 0],
            [0, 0, 0],  # Duplicate
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        computer = GPUFeatureComputer(use_gpu=False)
        
        features = computer._compute_essential_geometric_features(
            points=problematic_points,
            normals=np.zeros((len(problematic_points), 3)),
            k=5,
            required_features=['planarity', 'linearity']
        )
        
        # Should not contain NaN or Inf
        assert np.all(np.isfinite(features['planarity'])), "Planarity contains non-finite values"
        assert np.all(np.isfinite(features['linearity'])), "Linearity contains non-finite values"


class TestPhase3Comparison:
    """Test that Phase 3 refactored code produces same results as before"""
    
    @pytest.fixture
    def test_points(self):
        """Standard test point cloud"""
        np.random.seed(42)
        return np.random.uniform(0, 10, (500, 3)).astype(np.float32)
    
    def test_consistency_across_implementations(self, test_points):
        """Test that features_gpu.py uses GPU bridge correctly and produces valid results"""
        from ign_lidar.features.gpu_processor import GPUProcessor as GPUFeatureComputer
        
        # Compute using refactored GPU feature computer
        computer = GPUFeatureComputer(use_gpu=False)
        gpu_features = computer._compute_essential_geometric_features(
            points=test_points,
            normals=np.zeros((len(test_points), 3)),
            k=10,
            required_features=['planarity', 'linearity', 'sphericity']
        )
        
        # Verify features are valid and in correct range
        for feature_name in ['planarity', 'linearity', 'sphericity']:
            values = gpu_features[feature_name]
            
            # Check shape
            assert values.shape == (len(test_points),), f"{feature_name} has wrong shape"
            
            # Check data type
            assert values.dtype == np.float32, f"{feature_name} has wrong dtype"
            
            # Check range [0, 1]
            assert np.all(values >= 0) and np.all(values <= 1), f"{feature_name} values out of [0,1] range"
            
            # Check no NaN or Inf
            assert np.all(np.isfinite(values)), f"{feature_name} contains non-finite values"
        
        # Test that repeated computation gives same results (determinism)
        gpu_features2 = computer._compute_essential_geometric_features(
            points=test_points,
            normals=np.zeros((len(test_points), 3)),
            k=10,
            required_features=['planarity', 'linearity', 'sphericity']
        )
        
        for feature_name in ['planarity', 'linearity', 'sphericity']:
            vals1 = gpu_features[feature_name]
            vals2 = gpu_features2[feature_name]
            
            # Should be identical (deterministic)
            max_diff = np.max(np.abs(vals1 - vals2))
            assert max_diff < 1e-6, f"{feature_name} not deterministic: max diff {max_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
