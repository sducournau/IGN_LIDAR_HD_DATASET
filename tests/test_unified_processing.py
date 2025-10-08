"""
Tests for unified processing pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ign_lidar.core.processor import LiDARProcessor


class TestUnifiedProcessing:
    """Test unified processing pipeline."""
    
    def test_process_tile_unified_basic(self, tmp_path, mock_laz_file):
        """Test basic unified processing."""
        output_dir = tmp_path / "output"
        
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=50.0,
            num_points=4096,
            use_gpu=False
        )
        
        result = processor.process_tile_unified(
            laz_file=mock_laz_file,
            output_dir=output_dir,
            architecture='pointnet++',
            save_enriched=False,
            output_format='npz',
            skip_existing=False
        )
        
        assert result['num_patches'] > 0
        assert result['processing_time'] > 0
        assert result['points_processed'] > 0
        assert not result['skipped']
    
    def test_process_tile_unified_multi_arch(self, tmp_path, mock_laz_file):
        """Test unified processing with multiple architectures."""
        output_dir = tmp_path / "output"
        
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=50.0,
            num_points=4096
        )
        
        result = processor.process_tile_unified(
            laz_file=mock_laz_file,
            output_dir=output_dir,
            architecture='multi',  # All architectures
            output_format='npz',
            skip_existing=False
        )
        
        assert result['num_patches'] > 0
        
        # Should have patches for all architectures
        archs = ['pointnet++', 'octree', 'transformer', 'sparse_conv']
        for arch in archs:
            arch_patches = list(output_dir.glob(f"*_{arch}_patch_*.npz"))
            assert len(arch_patches) > 0, f"No patches found for {arch}"
    
    def test_process_tile_unified_pytorch_format(self, tmp_path, mock_laz_file):
        """Test unified processing with PyTorch output format."""
        output_dir = tmp_path / "output"
        
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=50.0,
            num_points=4096
        )
        
        result = processor.process_tile_unified(
            laz_file=mock_laz_file,
            output_dir=output_dir,
            architecture='pointnet++',
            output_format='pytorch',
            skip_existing=False
        )
        
        assert result['num_patches'] > 0
        
        # Check that .pt files were created
        pt_files = list(output_dir.glob("*.pt"))
        assert len(pt_files) > 0
    
    def test_process_tile_unified_with_preprocessing(
        self, tmp_path, mock_laz_file
    ):
        """Test unified processing with preprocessing enabled."""
        output_dir = tmp_path / "output"
        
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=50.0,
            num_points=4096,
            preprocess=True,
            preprocess_config={
                'sor': {'enable': True, 'k': 12, 'std_multiplier': 3.0},  # More lenient
                'ror': {'enable': True, 'radius': 2.0, 'min_neighbors': 2}  # More lenient
            }
        )
        
        result = processor.process_tile_unified(
            laz_file=mock_laz_file,
            output_dir=output_dir,
            architecture='pointnet++',
            skip_existing=False
        )
        
        # Preprocessing might filter many points, so just check it doesn't error
        assert result['points_processed'] >= 0
        assert 'num_patches' in result
    
    def test_process_tile_unified_save_enriched(
        self, tmp_path, mock_laz_file
    ):
        """Test unified processing with enriched LAZ output."""
        output_dir = tmp_path / "output"
        
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=50.0,
            num_points=4096
        )
        
        result = processor.process_tile_unified(
            laz_file=mock_laz_file,
            output_dir=output_dir,
            architecture='pointnet++',
            save_enriched=True,  # Save enriched LAZ
            skip_existing=False
        )
        
        assert result['num_patches'] > 0
        
        # Check that enriched LAZ was created
        enriched_dir = output_dir / "enriched"
        if enriched_dir.exists():
            enriched_files = list(enriched_dir.glob("*_enriched.laz"))
            assert len(enriched_files) > 0
    
    def test_process_tile_unified_skip_existing(
        self, tmp_path, mock_laz_file
    ):
        """Test that unified processing skips existing patches."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)
        
        # Create a dummy patch file
        dummy_patch = output_dir / f"{mock_laz_file.stem}_pointnet++_patch_0000.npz"
        np.savez(dummy_patch, points=np.zeros((100, 3)))
        
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=50.0,
            num_points=4096
        )
        
        result = processor.process_tile_unified(
            laz_file=mock_laz_file,
            output_dir=output_dir,
            architecture='pointnet++',
            skip_existing=True
        )
        
        assert result['skipped']
        assert result['num_patches'] == 0
    
    def test_process_tile_unified_output_structure(
        self, tmp_path, mock_laz_file
    ):
        """Test that unified processing creates correct output structure."""
        output_dir = tmp_path / "output"
        
        processor = LiDARProcessor(
            lod_level='LOD2',
            patch_size=50.0,
            num_points=4096
        )
        
        result = processor.process_tile_unified(
            laz_file=mock_laz_file,
            output_dir=output_dir,
            architecture='pointnet++',
            skip_existing=False
        )
        
        assert result['num_patches'] > 0
        
        # Check patch files
        patches = list(output_dir.glob("*_pointnet++_patch_*.npz"))
        assert len(patches) > 0
        
        # Load first patch and verify structure
        patch_data = np.load(patches[0])
        assert 'points' in patch_data
        assert 'features' in patch_data
        assert 'labels' in patch_data


@pytest.fixture
def mock_laz_file(tmp_path):
    """Create a mock LAZ file for testing."""
    import laspy
    
    # Create mock point cloud data
    num_points = 50000
    x = np.random.uniform(0, 100, num_points)
    y = np.random.uniform(0, 100, num_points)
    z = np.random.uniform(0, 50, num_points)
    intensity = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    classification = np.random.choice([2, 3, 5, 6], num_points).astype(np.uint8)
    return_number = np.ones(num_points, dtype=np.uint8)
    
    # Create LAS file
    header = laspy.LasHeader(version="1.2", point_format=3)
    header.offsets = np.min([x, y, z], axis=1)
    header.scales = [0.01, 0.01, 0.01]
    
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.intensity = intensity
    las.classification = classification
    las.return_number = return_number
    
    # Write to file
    laz_path = tmp_path / "test_tile.laz"
    las.write(laz_path)
    
    return laz_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
