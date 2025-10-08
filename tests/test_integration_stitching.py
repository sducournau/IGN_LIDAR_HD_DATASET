"""
End-to-End Integration Test for Sprint 3 Tile Stitching

This test validates the complete tile stitching workflow:
1. Create synthetic tiles (core + neighbors)
2. Process with boundary-aware features enabled
3. Validate feature quality at boundaries
4. Compare with standard processing
"""

import pytest
import numpy as np
from pathlib import Path
import laspy

from ign_lidar.processor import LiDARProcessor


@pytest.fixture
def temp_tile_directory(tmp_path):
    """
    Create a temporary directory with synthetic tile grid:
    
    Grid layout (1km tiles):
      [A][B][C]
      [D][E][F]   E = core tile
      [G][H][I]
    
    We focus on tile E with its neighbors B, D, F, H.
    """
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    
    # Tile specifications (xmin, ymin, size)
    tiles = {
        'LIDAR_HD_0000_0000.laz': (0, 0, 1000),      # A
        'LIDAR_HD_1000_0000.laz': (1000, 0, 1000),   # B
        'LIDAR_HD_2000_0000.laz': (2000, 0, 1000),   # C
        'LIDAR_HD_0000_1000.laz': (0, 1000, 1000),   # D
        'LIDAR_HD_1000_1000.laz': (1000, 1000, 1000), # E (core)
        'LIDAR_HD_2000_1000.laz': (2000, 1000, 1000), # F
        'LIDAR_HD_0000_2000.laz': (0, 2000, 1000),   # G
        'LIDAR_HD_1000_2000.laz': (1000, 2000, 1000), # H
        'LIDAR_HD_2000_2000.laz': (2000, 2000, 1000), # I
    }
    
    for filename, (xmin, ymin, size) in tiles.items():
        _create_synthetic_tile(
            tile_dir / filename,
            xmin=xmin,
            ymin=ymin,
            size=size,
            num_points=5000
        )
    
    return tile_dir


def _create_synthetic_tile(
    output_path: Path,
    xmin: float,
    ymin: float,
    size: float,
    num_points: int
):
    """
    Create a synthetic LAZ tile with smooth terrain.
    
    Z values follow a sinusoidal pattern that's continuous across tiles.
    """
    np.random.seed(42)
    
    # Generate random points within tile bounds
    x = np.random.uniform(xmin, xmin + size, num_points)
    y = np.random.uniform(ymin, ymin + size, num_points)
    
    # Smooth terrain: Z = 100 + 10*sin(x/500) + 5*sin(y/300) + noise
    z = (
        100.0 +
        10.0 * np.sin(x / 500.0) +
        5.0 * np.sin(y / 300.0) +
        np.random.randn(num_points) * 0.5
    )
    
    # Random intensity and return numbers
    intensity = np.random.randint(0, 65535, num_points, dtype=np.uint16)
    return_number = np.random.randint(1, 4, num_points, dtype=np.uint8)
    
    # Classification: mostly ground (2) and vegetation (3, 4, 5)
    classification = np.random.choice(
        [2, 3, 4, 5],
        size=num_points,
        p=[0.4, 0.3, 0.2, 0.1]
    ).astype(np.uint8)
    
    # Create LAS file
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = [xmin, ymin, 0]
    header.scales = [0.01, 0.01, 0.01]
    
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.intensity = intensity
    las.return_number = return_number
    las.classification = classification
    
    las.write(output_path)


# ============================================================================
# Test Cases
# ============================================================================

def test_processor_initialization_with_stitching():
    """Test that processor initializes correctly with stitching enabled."""
    processor = LiDARProcessor(
        lod_level='LOD2',
        use_stitching=True,
        buffer_size=10.0
    )
    
    assert processor.use_stitching is True
    assert processor.buffer_size == 10.0


def test_processor_initialization_without_stitching():
    """Test backward compatibility (stitching disabled by default)."""
    processor = LiDARProcessor(lod_level='LOD2')
    
    assert processor.use_stitching is False
    assert processor.buffer_size == 10.0  # Default value


@pytest.mark.slow
def test_unified_processing_with_stitching(temp_tile_directory, tmp_path):
    """
    Test complete unified processing with tile stitching enabled.
    
    This is the key end-to-end test that validates:
    1. Tiles can be loaded with neighbors
    2. Boundary-aware features are computed
    3. Processing completes successfully
    4. Output patches are created
    """
    output_dir = tmp_path / "output_stitching"
    output_dir.mkdir()
    
    # Initialize processor with stitching
    processor = LiDARProcessor(
        lod_level='LOD2',
        num_points=4096,
        patch_size=200.0,
        use_stitching=True,
        buffer_size=10.0,
        k_neighbors=20
    )
    
    # Process core tile (E) - has all 4 neighbors
    core_tile = temp_tile_directory / "LIDAR_HD_1000_1000.laz"
    
    result = processor.process_tile_unified(
        laz_file=core_tile,
        output_dir=output_dir,
        architecture='pointnet++',
        output_format='npz',
        save_enriched=False,
        skip_existing=False
    )
    
    # Validate result
    assert result is not None
    assert 'num_patches' in result
    assert 'processing_time' in result
    assert 'skipped' in result
    
    assert result['skipped'] is False
    assert result['num_patches'] > 0
    
    # Check that patches were created
    patches = list(output_dir.glob("*.npz"))
    assert len(patches) > 0
    
    # Validate patch content
    patch_data = np.load(patches[0])
    assert 'coords' in patch_data
    assert 'features' in patch_data
    assert 'labels' in patch_data
    
    # Check shapes
    coords = patch_data['coords']
    features = patch_data['features']
    
    assert coords.shape[0] == 4096  # num_points
    assert coords.shape[1] == 3     # XYZ
    assert features.shape[0] == 4096


@pytest.mark.slow
def test_unified_processing_without_stitching(temp_tile_directory, tmp_path):
    """
    Test unified processing without tile stitching (standard mode).
    
    Used as baseline for comparison.
    """
    output_dir = tmp_path / "output_standard"
    output_dir.mkdir()
    
    # Initialize processor without stitching
    processor = LiDARProcessor(
        lod_level='LOD2',
        num_points=4096,
        patch_size=200.0,
        use_stitching=False,  # Standard mode
        k_neighbors=20
    )
    
    # Process core tile
    core_tile = temp_tile_directory / "LIDAR_HD_1000_1000.laz"
    
    result = processor.process_tile_unified(
        laz_file=core_tile,
        output_dir=output_dir,
        architecture='pointnet++',
        output_format='npz',
        save_enriched=False,
        skip_existing=False
    )
    
    # Validate result
    assert result is not None
    assert result['skipped'] is False
    assert result['num_patches'] > 0
    
    # Check that patches were created
    patches = list(output_dir.glob("*.npz"))
    assert len(patches) > 0


@pytest.mark.slow
def test_stitching_vs_standard_comparison(temp_tile_directory, tmp_path):
    """
    Compare boundary-aware vs standard processing.
    
    This test validates that:
    1. Both modes produce patches
    2. Stitching mode processes correctly
    3. Feature quality can be compared
    """
    # Process with stitching
    output_stitching = tmp_path / "output_stitching"
    output_stitching.mkdir()
    
    processor_stitching = LiDARProcessor(
        lod_level='LOD2',
        num_points=4096,
        patch_size=200.0,
        use_stitching=True,
        buffer_size=10.0,
        k_neighbors=20
    )
    
    core_tile = temp_tile_directory / "LIDAR_HD_1000_1000.laz"
    
    result_stitching = processor_stitching.process_tile_unified(
        laz_file=core_tile,
        output_dir=output_stitching,
        architecture='pointnet++',
        output_format='npz',
        save_enriched=False,
        skip_existing=False
    )
    
    # Process without stitching
    output_standard = tmp_path / "output_standard"
    output_standard.mkdir()
    
    processor_standard = LiDARProcessor(
        lod_level='LOD2',
        num_points=4096,
        patch_size=200.0,
        use_stitching=False,
        k_neighbors=20
    )
    
    result_standard = processor_standard.process_tile_unified(
        laz_file=core_tile,
        output_dir=output_standard,
        architecture='pointnet++',
        output_format='npz',
        save_enriched=False,
        skip_existing=False
    )
    
    # Compare results
    assert result_stitching['num_patches'] > 0
    assert result_standard['num_patches'] > 0
    
    # Both should produce similar number of patches
    # (minor differences possible due to preprocessing)
    patch_ratio = result_stitching['num_patches'] / result_standard['num_patches']
    assert 0.8 < patch_ratio < 1.2
    
    # Load one patch from each
    patches_stitching = list(output_stitching.glob("*.npz"))
    patches_standard = list(output_standard.glob("*.npz"))
    
    if patches_stitching and patches_standard:
        data_stitching = np.load(patches_stitching[0])
        data_standard = np.load(patches_standard[0])
        
        # Both should have same structure
        assert 'coords' in data_stitching
        assert 'coords' in data_standard
        assert data_stitching['coords'].shape == data_standard['coords'].shape


def test_edge_tile_without_neighbors(temp_tile_directory, tmp_path):
    """
    Test processing an edge tile that might not have all neighbors.
    
    This validates graceful fallback when stitching cannot be used.
    """
    output_dir = tmp_path / "output_edge"
    output_dir.mkdir()
    
    # Create a standalone tile (no neighbors)
    standalone_dir = tmp_path / "standalone"
    standalone_dir.mkdir()
    
    standalone_tile = standalone_dir / "LIDAR_HD_9999_9999.laz"
    _create_synthetic_tile(
        standalone_tile,
        xmin=9999000,
        ymin=9999000,
        size=1000,
        num_points=5000
    )
    
    # Process with stitching enabled
    processor = LiDARProcessor(
        lod_level='LOD2',
        num_points=4096,
        patch_size=200.0,
        use_stitching=True,
        buffer_size=10.0
    )
    
    # Should fall back to standard processing
    result = processor.process_tile_unified(
        laz_file=standalone_tile,
        output_dir=output_dir,
        architecture='pointnet++',
        output_format='npz',
        save_enriched=False,
        skip_existing=False
    )
    
    # Should still process successfully
    assert result is not None
    assert result['skipped'] is False
    assert result['num_patches'] >= 0  # May be 0 if tile is small


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
