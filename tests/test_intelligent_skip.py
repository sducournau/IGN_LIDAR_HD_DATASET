"""
Unit tests for intelligent skip detection with metadata tracking.

Tests the ProcessingMetadata class and integration with PatchSkipChecker.
"""

import pytest
import json
import time
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from ign_lidar.core.processing_metadata import ProcessingMetadata


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "output"


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    config = {
        'processor': {
            'lod_level': 'LOD2',
            'building_detection_mode': 'lod2',
            'transport_detection_mode': 'lod2',
        },
        'features': {
            'mode': 'lod2',
            'k_neighbors': 30,
            'search_radius': 2.0,
            'compute_normals': True,
            'compute_planarity': True,
            'compute_curvature': True,
            'compute_architectural_features': True,
            'use_rgb': True,
            'use_infrared': True,
            'compute_ndvi': True,
            'include_extra': True,
        },
        'preprocess': {
            'sor_enabled': True,
            'sor_k': 12,
            'sor_std': 1.8,
            'ror_enabled': True,
            'ror_radius': 1.0,
            'ror_neighbors': 6,
        },
        'data_sources': {
            'bd_topo': {
                'enabled': True,
                'features': {'buildings': True, 'roads': True},
                'parameters': {'road_buffer_tolerance': 0.5},
            },
            'bd_foret': {'enabled': True},
            'rpg': {'enabled': True, 'year': 2024},
            'cadastre': {'enabled': True},
        },
        'ground_truth': {
            'enabled': True,
            'preclassify': True,
            'post_processing': {'enabled': True},
        },
    }
    return OmegaConf.create(config)


class TestProcessingMetadata:
    """Tests for ProcessingMetadata class."""
    
    def test_metadata_directory_creation(self, temp_output_dir):
        """Test that metadata directory is created automatically."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        assert metadata_mgr.metadata_dir.exists()
        assert metadata_mgr.metadata_dir.name == ".processing_metadata"
    
    def test_config_hash_computation(self, sample_config):
        """Test that config hash is computed consistently."""
        metadata_mgr = ProcessingMetadata(Path("."))
        
        hash1 = metadata_mgr.compute_config_hash(sample_config)
        hash2 = metadata_mgr.compute_config_hash(sample_config)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length
    
    def test_config_hash_changes_with_config(self, sample_config):
        """Test that config hash changes when config changes."""
        metadata_mgr = ProcessingMetadata(Path("."))
        
        hash1 = metadata_mgr.compute_config_hash(sample_config)
        
        # Modify config
        sample_config.features.compute_planarity = False
        hash2 = metadata_mgr.compute_config_hash(sample_config)
        
        assert hash1 != hash2
    
    def test_save_and_load_metadata(self, temp_output_dir, sample_config):
        """Test saving and loading metadata."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        
        tile_name = "test_tile_001"
        output_files = {
            'enriched_laz': {'path': '/path/to/enriched.laz', 'size_bytes': 52428800},
            'patches': {'count': 48, 'format': 'npz'},
        }
        
        # Save metadata
        metadata_mgr.save_metadata(
            tile_name=tile_name,
            config=sample_config,
            processing_time=45.2,
            num_points=12500000,
            output_files=output_files,
        )
        
        # Load metadata
        loaded = metadata_mgr.load_metadata(tile_name)
        
        assert loaded is not None
        assert loaded['tile_name'] == tile_name
        assert loaded['processing_time_seconds'] == 45.2
        assert loaded['num_points'] == 12500000
        assert 'config_hash' in loaded
        assert loaded['output_files'] == output_files
    
    def test_should_reprocess_no_metadata(self, temp_output_dir, sample_config):
        """Test that tile should be processed when no metadata exists."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        
        should_reprocess, reason = metadata_mgr.should_reprocess(
            "non_existent_tile", sample_config
        )
        
        assert should_reprocess is True
        assert reason == "no_metadata_found"
    
    def test_should_skip_with_valid_metadata(self, temp_output_dir, sample_config):
        """Test that tile is skipped when metadata matches and outputs exist."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        
        # Create fake output file
        enriched_path = temp_output_dir / "test_tile_enriched.laz"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        enriched_path.write_text("fake laz content")
        
        # Save metadata
        tile_name = "test_tile"
        output_files = {
            'enriched_laz': {'path': str(enriched_path), 'size_bytes': 1000},
        }
        metadata_mgr.save_metadata(
            tile_name=tile_name,
            config=sample_config,
            processing_time=10.0,
            num_points=1000000,
            output_files=output_files,
        )
        
        # Check should_reprocess
        should_reprocess, reason = metadata_mgr.should_reprocess(
            tile_name, sample_config
        )
        
        assert should_reprocess is False
        assert reason is None
    
    def test_should_reprocess_on_config_change(self, temp_output_dir, sample_config):
        """Test that tile is reprocessed when config changes."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        
        # Save metadata with original config
        tile_name = "test_tile"
        metadata_mgr.save_metadata(
            tile_name=tile_name,
            config=sample_config,
            processing_time=10.0,
            num_points=1000000,
            output_files={},
        )
        
        # Modify config
        sample_config.features.compute_curvature = False
        
        # Check should_reprocess
        should_reprocess, reason = metadata_mgr.should_reprocess(
            tile_name, sample_config
        )
        
        assert should_reprocess is True
        assert reason == "config_changed"
    
    def test_should_reprocess_on_missing_output(self, temp_output_dir, sample_config):
        """Test that tile is reprocessed when output file is missing."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        
        # Save metadata pointing to non-existent file
        tile_name = "test_tile"
        output_files = {
            'enriched_laz': {'path': '/non/existent/file.laz', 'size_bytes': 1000},
        }
        metadata_mgr.save_metadata(
            tile_name=tile_name,
            config=sample_config,
            processing_time=10.0,
            num_points=1000000,
            output_files=output_files,
        )
        
        # Check should_reprocess
        should_reprocess, reason = metadata_mgr.should_reprocess(
            tile_name, sample_config
        )
        
        assert should_reprocess is True
        assert reason == "output_file_missing_enriched_laz"
    
    def test_processing_stats(self, temp_output_dir, sample_config):
        """Test aggregate processing statistics."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        
        # Save metadata for multiple tiles
        for i in range(5):
            tile_name = f"test_tile_{i:03d}"
            metadata_mgr.save_metadata(
                tile_name=tile_name,
                config=sample_config,
                processing_time=10.0 + i,
                num_points=1000000 + i * 100000,
                output_files={},
            )
            time.sleep(0.01)  # Ensure different timestamps
        
        # Get stats
        stats = metadata_mgr.get_processing_stats()
        
        assert stats['total_tiles'] == 5
        assert stats['total_processing_time'] == 60.0  # 10+11+12+13+14
        assert stats['total_points'] == 5000000 + (0+1+2+3+4)*100000
        assert stats['unique_configs'] == 1  # All same config
        assert stats['oldest_processing'] is not None
        assert stats['newest_processing'] is not None
    
    def test_cleanup_orphaned_metadata(self, temp_output_dir, sample_config):
        """Test cleanup of metadata files without corresponding outputs."""
        metadata_mgr = ProcessingMetadata(temp_output_dir)
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata for 3 tiles
        for i in range(3):
            tile_name = f"test_tile_{i:03d}"
            metadata_mgr.save_metadata(
                tile_name=tile_name,
                config=sample_config,
                processing_time=10.0,
                num_points=1000000,
                output_files={},
            )
        
        # Create actual output for only one tile
        output_file = temp_output_dir / "test_tile_001_enriched.laz"
        output_file.write_text("fake content")
        
        # Cleanup orphaned metadata
        removed = metadata_mgr.cleanup_orphaned_metadata(temp_output_dir)
        
        # Should remove 2 out of 3 (tile_000 and tile_002 have no outputs)
        assert removed == 2
        
        # Verify only tile_001 metadata remains
        remaining = list(metadata_mgr.metadata_dir.glob("*.json"))
        assert len(remaining) == 1
        assert remaining[0].stem == "test_tile_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
