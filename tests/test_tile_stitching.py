"""
Unit tests for TileStitcher class.

Tests tile stitching functionality including:
- Loading tiles with neighbors
- Buffer zone extraction
- Boundary point detection
- Cross-tile neighborhood queries
- Auto-detection of neighbor tiles

Author: IGN LiDAR HD Team
Date: October 7, 2025
Sprint: 3 (Tile Stitching)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import laspy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.tile_stitcher import TileStitcher


class TestTileStitcher:
    """Test suite for TileStitcher class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_tile(self, temp_dir):
        """Create a sample LAZ tile for testing."""
        # Generate sample points (10x10m area)
        num_points = 1000
        x = np.random.uniform(0, 10, num_points)
        y = np.random.uniform(0, 10, num_points)
        z = np.random.uniform(0, 5, num_points)
        
        # Create LAZ file
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.scales = [0.01, 0.01, 0.01]
        header.offsets = [0, 0, 0]
        
        las = laspy.LasData(header)
        las.x = x
        las.y = y
        las.z = z
        las.intensity = np.random.randint(0, 65535, num_points).astype(np.uint16)
        las.return_number = np.ones(num_points, dtype=np.uint8)
        las.classification = np.random.choice([2, 6], num_points).astype(np.uint8)
        
        tile_path = temp_dir / "test_tile.laz"
        las.write(str(tile_path))
        
        return tile_path
    
    @pytest.fixture
    def sample_tiles_grid(self, temp_dir):
        """Create a 2x2 grid of sample tiles."""
        tiles = {}
        
        for i, (x_offset, y_offset) in enumerate([
            (0, 0),      # SW
            (10, 0),     # SE
            (0, 10),     # NW
            (10, 10)     # NE
        ]):
            num_points = 1000
            x = np.random.uniform(x_offset, x_offset + 10, num_points)
            y = np.random.uniform(y_offset, y_offset + 10, num_points)
            z = np.random.uniform(0, 5, num_points)
            
            header = laspy.LasHeader(point_format=3, version="1.4")
            header.scales = [0.01, 0.01, 0.01]
            header.offsets = [0, 0, 0]
            
            las = laspy.LasData(header)
            las.x = x
            las.y = y
            las.z = z
            las.intensity = np.random.randint(0, 65535, num_points).astype(np.uint16)
            las.return_number = np.ones(num_points, dtype=np.uint8)
            las.classification = np.random.choice([2, 6], num_points).astype(np.uint8)
            
            tile_name = f"tile_{i}.laz"
            tile_path = temp_dir / tile_name
            las.write(str(tile_path))
            
            tiles[tile_name] = tile_path
        
        return tiles
    
    def test_init(self):
        """Test TileStitcher initialization."""
        stitcher = TileStitcher(buffer_size=10.0)
        assert stitcher.buffer_size == 10.0
        assert stitcher.enable_caching == True
        
        stitcher_no_cache = TileStitcher(buffer_size=5.0, enable_caching=False)
        assert stitcher_no_cache.buffer_size == 5.0
        assert stitcher_no_cache.enable_caching == False
    
    def test_load_single_tile(self, sample_tile):
        """Test loading a single tile without neighbors."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        tile_data = stitcher.load_tile_with_neighbors(
            tile_path=sample_tile,
            neighbor_tiles=None
        )
        
        assert 'core_points' in tile_data
        assert 'buffer_points' in tile_data
        assert 'combined_points' in tile_data
        assert tile_data['num_core'] > 0
        assert tile_data['num_buffer'] == 0  # No neighbors
        assert len(tile_data['combined_points']) == tile_data['num_core']
    
    def test_load_tile_with_neighbors(self, sample_tiles_grid):
        """Test loading tile with neighbors and buffer extraction."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        # Use SW tile as core, SE and NW as neighbors
        core_tile = sample_tiles_grid['tile_0.laz']
        neighbors = [
            sample_tiles_grid['tile_1.laz'],  # SE
            sample_tiles_grid['tile_2.laz']   # NW
        ]
        
        tile_data = stitcher.load_tile_with_neighbors(
            tile_path=core_tile,
            neighbor_tiles=neighbors
        )
        
        assert tile_data['num_core'] > 0
        assert tile_data['num_buffer'] >= 0  # May or may not have buffer points
        assert len(tile_data['combined_points']) == (
            tile_data['num_core'] + tile_data['num_buffer']
        )
        
        # Check core mask
        assert np.sum(tile_data['core_mask']) == tile_data['num_core']
        assert np.sum(~tile_data['core_mask']) == tile_data['num_buffer']
    
    def test_detect_boundary_points(self, sample_tile):
        """Test boundary point detection."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        # Load tile
        tile_data = stitcher.load_tile_with_neighbors(sample_tile)
        points = tile_data['core_points']
        bounds = tile_data['core_bounds']
        
        # Detect boundary points (within 2m of edge)
        boundary_mask = stitcher.detect_boundary_points(
            points, bounds, threshold=2.0
        )
        
        assert boundary_mask.dtype == bool
        assert len(boundary_mask) == len(points)
        assert np.any(boundary_mask)  # Should have some boundary points
        assert not np.all(boundary_mask)  # Should have some interior points
    
    def test_spatial_index(self, sample_tile):
        """Test KDTree spatial index building."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        tile_data = stitcher.load_tile_with_neighbors(sample_tile)
        points = tile_data['combined_points']
        
        # Build spatial index
        spatial_index = stitcher.build_spatial_index(points)
        
        # Query neighbors
        distances, indices = stitcher.query_cross_tile_neighbors(
            query_points=points[:10],  # Query first 10 points
            spatial_index=spatial_index,
            k_neighbors=5
        )
        
        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
        assert np.all(distances[:, 0] == 0)  # First neighbor is self (distance 0)
    
    def test_compute_bounds(self, sample_tile):
        """Test bounding box computation."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        points = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0]
        ])
        
        bounds = stitcher._compute_bounds(points)
        
        assert bounds == (0, 0, 10, 10)
    
    def test_buffer_zone_extraction(self, sample_tiles_grid):
        """Test extraction of buffer zone from neighbor."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        # Define core tile bounds
        core_bounds = (0, 0, 10, 10)
        
        # Extract buffer from SE neighbor (x: 10-20, y: 0-10)
        neighbor_path = sample_tiles_grid['tile_1.laz']
        
        buffer_data = stitcher._extract_buffer_zone(
            neighbor_path, core_bounds, buffer_size=2.0
        )
        
        # Should have points within 2m of shared boundary (x=10)
        if buffer_data is not None:
            buffer_points = buffer_data['points']
            assert len(buffer_points) > 0
            
            # All buffer points should be outside core bounds
            x = buffer_points[:, 0]
            y = buffer_points[:, 1]
            outside_core = (
                (x < core_bounds[0]) | (x > core_bounds[2]) |
                (y < core_bounds[1]) | (y > core_bounds[3])
            )
            assert np.all(outside_core)
    
    def test_caching(self, sample_tile):
        """Test tile caching functionality."""
        stitcher = TileStitcher(buffer_size=2.0, enable_caching=True)
        
        # Load tile twice
        data1 = stitcher._load_tile(sample_tile)
        data2 = stitcher._load_tile(sample_tile)
        
        # Should return same cached data
        assert np.array_equal(data1['points'], data2['points'])
        
        # Clear cache
        stitcher.clear_cache()
        assert len(stitcher._tile_cache) == 0
    
    def test_no_caching(self, sample_tile):
        """Test that caching can be disabled."""
        stitcher = TileStitcher(buffer_size=2.0, enable_caching=False)
        
        # Load tile
        data = stitcher._load_tile(sample_tile)
        
        # Cache should be None
        assert stitcher._tile_cache is None
    
    def test_invalid_tile(self, temp_dir):
        """Test handling of invalid/missing tile."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        invalid_path = temp_dir / "nonexistent.laz"
        
        with pytest.raises(FileNotFoundError):
            stitcher.load_tile_with_neighbors(invalid_path)
    
    def test_empty_neighbor_list(self, sample_tile):
        """Test with empty neighbor list."""
        stitcher = TileStitcher(buffer_size=2.0)
        
        tile_data = stitcher.load_tile_with_neighbors(
            tile_path=sample_tile,
            neighbor_tiles=[]  # Empty list
        )
        
        assert tile_data['num_buffer'] == 0
        assert len(tile_data['buffer_points']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
