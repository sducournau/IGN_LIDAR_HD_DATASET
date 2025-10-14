"""
Test to verify LiDARProcessor integration with FeatureOrchestrator.
"""

import pytest
from omegaconf import OmegaConf
from ign_lidar.core.processor import LiDARProcessor
from ign_lidar.features.orchestrator import FeatureOrchestrator


class TestProcessorOrchestatorIntegration:
    """Test that processor correctly uses orchestrator."""
    
    def test_processor_creates_orchestrator(self):
        """Test that processor creates orchestrator during initialization."""
        config = OmegaConf.create({
            'processor': {
                'lod_level': 'LOD2',
                'processing_mode': 'patches_only',
                'patch_size': 150.0,
                'patch_overlap': 0.1,
                'num_points': 16384,
                'use_gpu': False,
                'use_gpu_chunked': True,
                'gpu_batch_size': 1_000_000,
                'preprocess': False,
                'use_stitching': False,
                'buffer_size': 10.0,
                'architecture': 'pointnet++',
                'output_format': 'npz',
                'augment': False,
                'num_augmentations': 3,
            },
            'features': {
                'include_extra_features': False,
                'feature_mode': 'lod2',
                'k_neighbors': None,
                'include_architectural_style': False,
                'style_encoding': 'constant',
                'use_rgb': False,
                'rgb_cache_dir': None,
                'use_infrared': False,
                'compute_ndvi': False,
            }
        })
        
        processor = LiDARProcessor(config)
        
        # Verify orchestrator was created
        assert hasattr(processor, 'feature_orchestrator')
        assert isinstance(processor.feature_orchestrator, FeatureOrchestrator)
        
        # Verify backward compatibility alias
        assert processor.feature_manager is processor.feature_orchestrator
    
    def test_processor_backward_compat_properties(self):
        """Test that backward-compatible properties work."""
        config = OmegaConf.create({
            'processor': {
                'lod_level': 'LOD3',
                'processing_mode': 'patches_only',
                'patch_size': 150.0,
                'patch_overlap': 0.1,
                'num_points': 16384,
                'use_gpu': False,
                'use_gpu_chunked': True,
                'gpu_batch_size': 1_000_000,
                'preprocess': False,
                'use_stitching': False,
                'buffer_size': 10.0,
                'architecture': 'pointnet++',
                'output_format': 'npz',
                'augment': False,
                'num_augmentations': 3,
            },
            'features': {
                'include_extra_features': True,
                'feature_mode': 'lod3',
                'k_neighbors': 20,
                'include_architectural_style': True,
                'style_encoding': 'multihot',
                'use_rgb': True,
                'rgb_cache_dir': '/tmp/rgb_cache',
                'use_infrared': True,
                'compute_ndvi': True,
            }
        })
        
        processor = LiDARProcessor(config)
        
        # Test backward-compatible properties
        assert processor.use_gpu == False
        # Note: RGB/NIR fetchers require additional packages (e.g., requests, Pillow)
        # They may be None if dependencies are missing
        assert processor.include_rgb == True
        assert processor.include_infrared == True
        assert processor.compute_ndvi == True
        assert processor.include_architectural_style == True
        assert processor.feature_mode == 'lod3'
    
    def test_processor_legacy_kwargs_init(self):
        """Test that processor can still initialize with legacy kwargs."""
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=100.0,
            num_points=8192,
            include_rgb=False,
            include_infrared=False,
            augment=True,
            num_augmentations=5,
            architecture='dgcnn',
            output_format='hdf5'
        )
        
        # Verify orchestrator was created
        assert hasattr(processor, 'feature_orchestrator')
        assert isinstance(processor.feature_orchestrator, FeatureOrchestrator)
        
        # Verify settings propagated correctly
        assert processor.lod_level == 'LOD2'
        assert processor.patch_size == 100.0
        assert processor.num_points == 8192
        assert processor.architecture == 'dgcnn'
        assert processor.output_format == 'hdf5'
        assert processor.augment == True
        assert processor.num_augmentations == 5
    
    def test_orchestrator_properties_accessible(self):
        """Test that orchestrator properties are accessible through processor."""
        config = OmegaConf.create({
            'processor': {
                'lod_level': 'LOD2',
                'processing_mode': 'patches_only',
                'patch_size': 150.0,
                'patch_overlap': 0.1,
                'num_points': 16384,
                'use_gpu': False,
                'use_gpu_chunked': True,
                'gpu_batch_size': 1_000_000,
                'preprocess': False,
                'use_stitching': False,
                'buffer_size': 10.0,
                'architecture': 'pointnet++',
                'output_format': 'npz',
                'augment': False,
                'num_augmentations': 3,
            },
            'features': {
                'include_extra_features': False,
                'feature_mode': 'lod2',
                'k_neighbors': None,
                'include_architectural_style': False,
                'style_encoding': 'constant',
                'use_rgb': False,
                'rgb_cache_dir': None,
                'use_infrared': False,
                'compute_ndvi': False,
            }
        })
        
        processor = LiDARProcessor(config)
        
        # Test orchestrator properties (they are @property, not methods)
        assert processor.feature_orchestrator.has_rgb == False
        assert processor.feature_orchestrator.has_infrared == False
        assert processor.feature_orchestrator.has_gpu == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
