"""
Unit tests for FeatureOrchestrator

Tests the unified feature computation orchestrator including:
- Resource initialization
- Strategy selection
- Feature mode management
- Feature computation
- Spectral feature handling

Author: Phase 4 Consolidation Tests
Date: October 13, 2025
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from omegaconf import OmegaConf

from ign_lidar.features.orchestrator import FeatureOrchestrator
from ign_lidar.features.feature_modes import FeatureMode


class TestFeatureOrchestratorInit:
    """Test FeatureOrchestrator initialization."""
    
    def test_init_minimal_config(self):
        """Test initialization with minimal configuration."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        assert orchestrator.strategy_name == 'cpu'
        assert orchestrator.feature_mode == FeatureMode.FULL
        assert not orchestrator.gpu_available
        assert not orchestrator.has_rgb
        assert not orchestrator.has_infrared
    
    def test_init_with_rgb(self):
        """Test initialization with RGB enabled."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {
                'mode': 'full',
                'k_neighbors': 20,
                'use_rgb': True
            }
        })
        
        with patch('ign_lidar.preprocessing.rgb_augmentation.IGNOrthophotoFetcher'):
            orchestrator = FeatureOrchestrator(config)
            assert orchestrator.use_rgb
    
    def test_init_with_nir(self):
        """Test initialization with NIR enabled."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {
                'mode': 'full',
                'k_neighbors': 20,
                'use_infrared': True
            }
        })
        
        with patch('ign_lidar.preprocessing.infrared_augmentation.IGNInfraredFetcher'):
            orchestrator = FeatureOrchestrator(config)
            assert orchestrator.use_infrared
    
    def test_init_gpu_fallback(self):
        """Test GPU initialization with fallback to CPU."""
        config = OmegaConf.create({
            'processor': {'use_gpu': True},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        # Mock GPU not available
        with patch('ign_lidar.features.orchestrator.FeatureOrchestrator._validate_gpu', return_value=False):
            orchestrator = FeatureOrchestrator(config)
            assert orchestrator.strategy_name == 'cpu'
            assert not orchestrator.gpu_available


class TestStrategySelection:
    """Test feature computer strategy selection."""
    
    def test_cpu_strategy(self):
        """Test CPU strategy selection."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.strategy_name == 'cpu'
    
    @patch('ign_lidar.features.orchestrator.FeatureOrchestrator._validate_gpu', return_value=True)
    def test_gpu_strategy(self, mock_gpu):
        """Test GPU strategy selection."""
        config = OmegaConf.create({
            'processor': {'use_gpu': True, 'use_gpu_chunked': False},
            'features': {'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.strategy_name == 'gpu'
        assert orchestrator.gpu_available
    
    @patch('ign_lidar.features.orchestrator.FeatureOrchestrator._validate_gpu', return_value=True)
    def test_gpu_chunked_strategy(self, mock_gpu):
        """Test GPU chunked strategy selection."""
        config = OmegaConf.create({
            'processor': {
                'use_gpu': True,
                'use_gpu_chunked': True,
                'gpu_batch_size': 50000
            },
            'features': {'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.strategy_name == 'gpu_chunked'
    
    def test_boundary_aware_strategy(self):
        """Test boundary-aware strategy selection."""
        config = OmegaConf.create({
            'processor': {
                'use_gpu': False,
                'use_boundary_aware': True,
                'buffer_size': 15.0
            },
            'features': {'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.strategy_name == 'boundary_aware'


class TestFeatureModeManagement:
    """Test feature mode initialization and enforcement."""
    
    def test_mode_minimal(self):
        """Test MINIMAL mode initialization."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'minimal', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.feature_mode == FeatureMode.MINIMAL
    
    def test_mode_lod2(self):
        """Test LOD2 mode initialization."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'lod2', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.feature_mode == FeatureMode.LOD2_SIMPLIFIED
    
    def test_mode_lod3(self):
        """Test LOD3 mode initialization."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'lod3', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.feature_mode == FeatureMode.LOD3_FULL
    
    def test_mode_full(self):
        """Test FULL mode initialization."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.feature_mode == FeatureMode.FULL
    
    def test_mode_invalid_fallback(self):
        """Test invalid mode falls back to FULL."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'invalid_mode', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.feature_mode == FeatureMode.FULL
    
    def test_validate_mode(self):
        """Test mode validation."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert orchestrator.validate_mode(FeatureMode.FULL)
        assert orchestrator.validate_mode(FeatureMode.MINIMAL)
    
    def test_get_feature_list(self):
        """Test getting feature list for mode."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        features = orchestrator.get_feature_list(FeatureMode.MINIMAL)
        
        assert isinstance(features, list)
        assert len(features) > 0
    
    def test_filter_features(self):
        """Test feature filtering by mode."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'minimal', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        # Create mock features dict with many features
        all_features = {
            'normals': np.random.rand(100, 3),
            'curvature': np.random.rand(100),
            'height': np.random.rand(100),
            'planarity': np.random.rand(100),
            'linearity': np.random.rand(100),
            'eigenvalue_1': np.random.rand(100),
            'eigenvalue_2': np.random.rand(100),
            'intensity': np.random.rand(100),
            'return_number': np.random.randint(1, 5, 100)
        }
        
        # Filter to MINIMAL mode
        filtered = orchestrator.filter_features(all_features, FeatureMode.MINIMAL)
        
        # Core features should always be present
        assert 'normals' in filtered
        assert 'curvature' in filtered
        assert 'height' in filtered
        
        # Filtered dict should be smaller than original
        assert len(filtered) <= len(all_features)


class TestFeatureComputation:
    """Test feature computation methods."""
    
    def test_compute_features_basic(self):
        """Test basic feature computation."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False, 'include_extra_features': True},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        # Mock computer.compute_features
        mock_features = {
            'normals': np.random.rand(100, 3),
            'curvature': np.random.rand(100),
            'height': np.random.rand(100),
            'planarity': np.random.rand(100)
        }
        orchestrator.computer.compute_features = Mock(return_value=mock_features)
        
        # Create tile data
        tile_data = {
            'points': np.random.rand(100, 3),
            'classification': np.random.randint(0, 10, 100),
            'intensity': np.random.rand(100),
            'return_number': np.random.randint(1, 5, 100)
        }
        
        # Compute features
        features = orchestrator.compute_features(tile_data)
        
        # Check results
        assert 'normals' in features
        assert 'curvature' in features
        assert 'height' in features
        assert 'intensity' in features
        assert 'return_number' in features
    
    def test_compute_features_with_enriched(self):
        """Test using enriched features from input."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        # Create tile data with enriched features
        enriched = {
            'normals': np.random.rand(100, 3),
            'curvature': np.random.rand(100),
            'height': np.random.rand(100)
        }
        
        tile_data = {
            'points': np.random.rand(100, 3),
            'classification': np.random.randint(0, 10, 100),
            'intensity': np.random.rand(100),
            'return_number': np.random.randint(1, 5, 100),
            'enriched_features': enriched
        }
        
        # Compute features with use_enriched=True
        features = orchestrator.compute_features(tile_data, use_enriched=True)
        
        # Should use enriched features
        assert np.array_equal(features['normals'], enriched['normals'])
        assert np.array_equal(features['curvature'], enriched['curvature'])
    
    def test_compute_geometric_features(self):
        """Test geometric feature computation."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False, 'include_extra_features': True},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        # Mock computer.compute_features
        mock_features = {
            'normals': np.random.rand(100, 3),
            'curvature': np.random.rand(100),
            'height': np.random.rand(100),
            'planarity': np.random.rand(100),
            'linearity': np.random.rand(100)
        }
        orchestrator.computer.compute_features = Mock(return_value=mock_features)
        
        # Compute
        points = np.random.rand(100, 3)
        classification = np.random.randint(0, 10, 100)
        
        normals, curvature, height, geo_features = orchestrator._compute_geometric_features(
            points, classification
        )
        
        # Check results
        assert normals.shape == (100, 3)
        assert curvature.shape == (100,)
        assert height.shape == (100,)
        assert isinstance(geo_features, dict)
        assert 'planarity' in geo_features
        assert 'linearity' in geo_features


class TestSpectralFeatures:
    """Test RGB, NIR, and NDVI feature handling."""
    
    def test_add_rgb_from_input(self):
        """Test adding RGB from input LAZ."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {'mode': 'full', 'use_rgb': True, 'k_neighbors': 20}
        })
        
        with patch('ign_lidar.preprocessing.rgb_augmentation.IGNOrthophotoFetcher'):
            orchestrator = FeatureOrchestrator(config)
            
            # Create tile data with input RGB
            input_rgb = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
            tile_data = {
                'points': np.random.rand(100, 3),
                'input_rgb': input_rgb
            }
            
            all_features = {}
            rgb_added = orchestrator._add_rgb_features(tile_data, all_features)
            
            assert rgb_added
            assert 'rgb' in all_features
            assert np.array_equal(all_features['rgb'], input_rgb)
    
    def test_add_nir_from_input(self):
        """Test adding NIR from input LAZ."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {'mode': 'full', 'use_infrared': True, 'k_neighbors': 20}
        })
        
        with patch('ign_lidar.preprocessing.infrared_augmentation.IGNInfraredFetcher'):
            orchestrator = FeatureOrchestrator(config)
            
            # Create tile data with input NIR
            input_nir = np.random.randint(0, 255, 100, dtype=np.uint8)
            tile_data = {
                'points': np.random.rand(100, 3),
                'input_nir': input_nir
            }
            
            all_features = {}
            nir_added = orchestrator._add_nir_features(tile_data, all_features)
            
            assert nir_added
            assert 'nir' in all_features
            assert np.array_equal(all_features['nir'], input_nir)
    
    def test_compute_ndvi(self):
        """Test NDVI computation from RGB and NIR."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False, 'compute_ndvi': True},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        # Create RGB and NIR
        rgb = np.array([[100, 50, 30], [150, 80, 40]], dtype=np.uint8)
        nir = np.array([120, 180], dtype=np.uint8)
        
        all_features = {'rgb': rgb, 'nir': nir}
        tile_data = {}
        
        orchestrator._add_ndvi_features(tile_data, all_features, True, True)
        
        assert 'ndvi' in all_features
        assert all_features['ndvi'].shape == (2,)
        
        # Check NDVI formula: (NIR - Red) / (NIR + Red)
        expected_ndvi_0 = (120 - 100) / (120 + 100)
        expected_ndvi_1 = (180 - 150) / (180 + 150)
        
        np.testing.assert_almost_equal(all_features['ndvi'][0], expected_ndvi_0, decimal=5)
        np.testing.assert_almost_equal(all_features['ndvi'][1], expected_ndvi_1, decimal=5)
    
    def test_ndvi_from_input(self):
        """Test using NDVI from input LAZ."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False, 'compute_ndvi': True},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        
        # Create tile data with input NDVI
        input_ndvi = np.random.rand(100)
        tile_data = {'input_ndvi': input_ndvi}
        
        all_features = {}
        orchestrator._add_ndvi_features(tile_data, all_features, False, False)
        
        assert 'ndvi' in all_features
        assert np.array_equal(all_features['ndvi'], input_ndvi)


class TestProperties:
    """Test orchestrator properties and utilities."""
    
    def test_has_rgb_property(self):
        """Test has_rgb property."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'full', 'use_rgb': False, 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert not orchestrator.has_rgb
    
    def test_has_infrared_property(self):
        """Test has_infrared property."""
        config = OmegaConf.create({
            'processor': {},
            'features': {'mode': 'full', 'use_infrared': False, 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert not orchestrator.has_infrared
    
    def test_has_gpu_property(self):
        """Test has_gpu property."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        assert not orchestrator.has_gpu
    
    def test_repr(self):
        """Test string representation."""
        config = OmegaConf.create({
            'processor': {'use_gpu': False},
            'features': {'mode': 'full', 'k_neighbors': 20}
        })
        
        orchestrator = FeatureOrchestrator(config)
        repr_str = repr(orchestrator)
        
        assert 'FeatureOrchestrator' in repr_str
        assert 'strategy=cpu' in repr_str
        assert 'mode=full' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
