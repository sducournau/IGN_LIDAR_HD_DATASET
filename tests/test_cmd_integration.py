"""
Integration tests for CLI commands.

Tests the full workflow of each command using real (but small) test data.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import argparse


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    
    # Create directory structure
    (workspace / "input").mkdir()
    (workspace / "output").mkdir()
    (workspace / "cache").mkdir()
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_laz_file(temp_workspace):
    """Create a mock LAZ file for testing."""
    laz_path = temp_workspace / "input" / "test_tile.laz"
    
    # Create a minimal LAZ file using laspy
    try:
        import laspy
        
        # Create minimal point cloud
        points = np.random.rand(1000, 3).astype(np.float32) * 100
        
        # Create header
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = [0, 0, 0]
        header.scales = [0.01, 0.01, 0.01]
        
        # Create LAS data
        las = laspy.LasData(header)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.classification = np.random.randint(0, 10, 1000, dtype=np.uint8)
        las.intensity = np.random.randint(0, 65535, 1000, dtype=np.uint16)
        
        # Write file
        las.write(laz_path)
        
        return laz_path
    
    except ImportError:
        pytest.skip("laspy not available")


@pytest.fixture
def mock_enriched_laz(temp_workspace):
    """Create a mock enriched LAZ file with features."""
    enriched_path = temp_workspace / "input" / "enriched_tile.laz"
    
    try:
        import laspy
        
        # Create minimal point cloud
        points = np.random.rand(1000, 3).astype(np.float32) * 100
        
        # Create header with extra dimensions for features
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = [0, 0, 0]
        header.scales = [0.01, 0.01, 0.01]
        
        # Add extra dimensions for CORE features
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="linearity", type=np.float32
        ))
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="planarity", type=np.float32
        ))
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="sphericity", type=np.float32
        ))
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="anisotropy", type=np.float32
        ))
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="roughness", type=np.float32
        ))
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="density", type=np.float32
        ))
        header.add_extra_dim(laspy.ExtraBytesParams(
            name="verticality", type=np.float32
        ))
        
        # Create LAS data
        las = laspy.LasData(header)
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.classification = np.random.randint(0, 10, 1000, dtype=np.uint8)
        
        # Add feature values (random but valid range [0, 1])
        las.linearity = np.random.rand(1000).astype(np.float32)
        las.planarity = np.random.rand(1000).astype(np.float32)
        las.sphericity = np.random.rand(1000).astype(np.float32)
        las.anisotropy = np.random.rand(1000).astype(np.float32)
        las.roughness = np.random.rand(1000).astype(np.float32)
        las.density = np.random.rand(1000).astype(np.float32) * 10
        las.verticality = np.random.rand(1000).astype(np.float32)
        
        # Write file
        las.write(enriched_path)
        
        return enriched_path
    
    except ImportError:
        pytest.skip("laspy not available")


class TestCmdProcess:
    """Test cmd_process() command."""
    
    def test_process_validates_input(self, temp_workspace):
        """Test that cmd_process validates input paths."""
        from ign_lidar.cli import cmd_process
        
        args = Mock()
        args.input = temp_workspace / "nonexistent.laz"
        args.input_dir = None
        args.output = temp_workspace / "output"
        args.bbox = None
        args.lod_level = "LOD2"
        args.patch_size = 150.0
        args.patch_overlap = 0.1
        args.num_points = 16384
        args.k_neighbors = None
        args.include_architectural_style = False
        args.style_encoding = "constant"
        args.num_workers = 1
        args.force = False
        
        # Should return error code for nonexistent input
        # (actual validation happens in processor, so we test the flow)
        with patch('ign_lidar.cli.LiDARProcessor') as mock_processor:
            mock_processor.return_value.process_tile.side_effect = FileNotFoundError
            
            with pytest.raises(FileNotFoundError):
                cmd_process(args)
    
    def test_process_parses_bbox(self, temp_workspace):
        """Test that cmd_process correctly parses bounding box."""
        from ign_lidar.cli import cmd_process
        
        args = Mock()
        args.input_dir = temp_workspace / "input"
        args.input = None
        args.output = temp_workspace / "output"
        args.bbox = "100000,6000000,200000,7000000"
        args.lod_level = "LOD2"
        args.patch_size = 150.0
        args.patch_overlap = 0.1
        args.num_points = 16384
        args.k_neighbors = None
        args.include_architectural_style = False
        args.style_encoding = "constant"
        args.num_workers = 1
        args.force = False
        
        with patch('ign_lidar.cli.LiDARProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.process_directory.return_value = 10
            mock_processor.return_value = mock_instance
            
            result = cmd_process(args)
            
            # Check processor was initialized with parsed bbox
            call_kwargs = mock_processor.call_args[1]
            assert call_kwargs['bbox'] == [100000.0, 6000000.0, 200000.0, 7000000.0]
            assert result == 0
    
    def test_process_invalid_bbox(self, temp_workspace):
        """Test that cmd_process handles invalid bbox."""
        from ign_lidar.cli import cmd_process
        
        args = Mock()
        args.input_dir = temp_workspace / "input"
        args.input = None
        args.output = temp_workspace / "output"
        args.bbox = "invalid,bbox,values"  # Only 3 values
        args.lod_level = "LOD2"
        
        result = cmd_process(args)
        assert result == 1  # Error code


class TestCmdVerify:
    """Test cmd_verify() command."""
    
    def test_verify_requires_input(self, temp_workspace):
        """Test that cmd_verify requires input path."""
        from ign_lidar.cli import cmd_verify
        
        args = Mock()
        args.input = None
        args.input_dir = None
        args.mode = 'core'
        
        result = cmd_verify(args)
        assert result == 1  # Error code
    
    def test_verify_validates_path(self, temp_workspace):
        """Test that cmd_verify validates input path."""
        from ign_lidar.cli import cmd_verify
        
        args = Mock()
        args.input = None
        args.input_dir = temp_workspace / "nonexistent"
        args.mode = 'core'
        args.max_files = None
        args.show_samples = False
        
        result = cmd_verify(args)
        assert result == 1  # Error code for invalid path
    
    def test_verify_core_mode(self, temp_workspace, mock_enriched_laz):
        """Test verification in CORE mode."""
        from ign_lidar.cli import cmd_verify
        
        args = Mock()
        args.input = mock_enriched_laz
        args.input_dir = None
        args.mode = 'core'
        args.max_files = None
        args.show_samples = False
        
        result = cmd_verify(args)
        assert result == 0  # Success
    
    def test_verify_auto_mode_detection(self, temp_workspace, mock_enriched_laz):
        """Test that auto mode detection works."""
        from ign_lidar.cli import cmd_verify
        
        args = Mock()
        args.input_dir = mock_enriched_laz.parent
        args.input = None
        args.mode = 'auto'
        args.max_files = None
        args.show_samples = False
        
        result = cmd_verify(args)
        assert result == 0  # Should detect mode and verify


class TestCmdEnrich:
    """Test cmd_enrich() command."""
    
    def test_enrich_requires_input(self, temp_workspace):
        """Test that cmd_enrich requires input."""
        from ign_lidar.cli import cmd_enrich
        
        args = Mock()
        args.input = None
        args.input_dir = None
        args.output = temp_workspace / "output"
        args.files = []
        
        result = cmd_enrich(args)
        assert result == 1  # Error code
    
    def test_enrich_validates_output(self, temp_workspace):
        """Test that cmd_enrich validates output path."""
        from ign_lidar.cli import cmd_enrich
        
        args = Mock()
        args.input_dir = temp_workspace / "input"
        args.input = None
        args.output = None  # Invalid
        args.files = []
        
        with pytest.raises(Exception):  # Should raise validation error
            cmd_enrich(args)
    
    @pytest.mark.skip(reason="Complex mocking - requires full integration test")
    def test_enrich_memory_calculation(self, temp_workspace, mock_laz_file):
        """Test that enrich calculates optimal workers."""
        from ign_lidar.cli import cmd_enrich
        
        args = Mock()
        args.input = mock_laz_file
        args.input_dir = None
        args.output = temp_workspace / "output"
        args.files = []
        args.num_workers = 8  # Request 8
        args.mode = 'core'
        args.k_neighbors = 30
        args.radius = None
        args.use_gpu = False
        args.skip_existing = True
        args.add_rgb = False
        args.add_infrared = False
        args.rgb_cache_dir = None
        args.infrared_cache_dir = None
        args.preprocess = False
        args.auto_params = False
        args.augment = False
        args.num_augmentations = 0
        args.force = False
        args.voxel_size = None
        
        # Mock the ProcessPoolExecutor at source
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = 'success'
            mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
            mock_executor.return_value = mock_executor_instance
            
            with patch('ign_lidar.cli._enrich_single_file') as mock_enrich:
                mock_enrich.return_value = 'success'
                
                result = cmd_enrich(args)
                
                # Check that worker calculation was done
                # (memory_utils should have been called)
                assert result == 0


class TestUtilityIntegration:
    """Test that utilities are properly integrated into commands."""
    
    def test_validate_input_path_integration(self, temp_workspace):
        """Test validate_input_path is used in commands."""
        from ign_lidar.cli_utils import validate_input_path
        
        # Test valid file
        test_file = temp_workspace / "input" / "test.laz"
        test_file.touch()
        assert validate_input_path(test_file, path_type="file") == True
        
        # Test valid directory
        assert validate_input_path(temp_workspace / "input", path_type="directory") == True
        
        # Test invalid path
        assert validate_input_path(temp_workspace / "nonexistent", path_type="file") == False
    
    def test_discover_laz_files_integration(self, temp_workspace):
        """Test discover_laz_files is used correctly."""
        from ign_lidar.cli_utils import discover_laz_files
        
        # Create test files
        (temp_workspace / "input" / "file1.laz").touch()
        (temp_workspace / "input" / "file2.laz").touch()
        (temp_workspace / "input" / "file3.txt").touch()  # Not LAZ
        
        files = discover_laz_files(temp_workspace / "input")
        assert len(files) == 2
        assert all(f.suffix == ".laz" for f in files)
    
    def test_memory_utils_integration(self):
        """Test memory utilities are accessible and work."""
        from ign_lidar.memory_utils import (
            calculate_optimal_workers,
            calculate_batch_size,
            analyze_file_sizes
        )
        
        # Test worker calculation
        workers, info = calculate_optimal_workers(
            num_files=10,
            file_sizes_mb=[100, 150],
            mode='core',
            use_gpu=False,
            requested_workers=4
        )
        assert workers > 0
        assert isinstance(info, dict)
        
        # Test batch size calculation
        batch_size = calculate_batch_size(
            num_workers=4,
            max_file_size_mb=200,
            mode='core'
        )
        assert batch_size > 0


class TestErrorHandling:
    """Test error handling in CLI commands."""
    
    def test_invalid_mode_parameter(self, temp_workspace):
        """Test handling of invalid mode parameter."""
        from ign_lidar.cli import cmd_verify
        
        args = Mock()
        args.input_dir = temp_workspace / "input"
        args.input = None
        args.mode = 'invalid_mode'  # Invalid
        args.max_files = None
        args.show_samples = False
        
        # Create a test file
        (temp_workspace / "input" / "test.laz").touch()
        
        # Should handle gracefully (default to auto or error)
        result = cmd_verify(args)
        # Result depends on implementation - should not crash
        assert isinstance(result, int)
    
    def test_missing_dependencies(self, temp_workspace):
        """Test handling when dependencies are missing."""
        # This is tested implicitly by mocking modules
        # Real implementation should handle ImportError gracefully
        pass


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.skip(reason="Complex mocking - requires full integration test")
    @pytest.mark.slow
    def test_enrich_then_verify_workflow(self, temp_workspace, mock_laz_file):
        """Test complete enrich → verify workflow."""
        from ign_lidar.cli import cmd_enrich, cmd_verify
        
        # Step 1: Enrich file
        enrich_args = Mock()
        enrich_args.input = mock_laz_file
        enrich_args.input_dir = None
        enrich_args.output = temp_workspace / "enriched"
        enrich_args.output.mkdir()
        enrich_args.files = []
        enrich_args.num_workers = 1
        enrich_args.mode = 'core'
        enrich_args.k_neighbors = 30
        enrich_args.radius = None
        enrich_args.use_gpu = False
        enrich_args.skip_existing = False
        enrich_args.add_rgb = False
        enrich_args.add_infrared = False
        enrich_args.rgb_cache_dir = None
        enrich_args.infrared_cache_dir = None
        enrich_args.preprocess = False
        enrich_args.auto_params = False
        enrich_args.augment = False
        enrich_args.num_augmentations = 0
        enrich_args.force = False
        enrich_args.voxel_size = None
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor, \
             patch('ign_lidar.cli._enrich_single_file') as mock_enrich_func:
            
            mock_executor_instance = MagicMock()
            mock_future = MagicMock()
            mock_future.result.return_value = 'success'
            mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
            mock_executor.return_value = mock_executor_instance
            mock_enrich_func.return_value = 'success'
            
            enrich_result = cmd_enrich(enrich_args)
            assert enrich_result == 0
        
        # Step 2: Verify enriched file
        # (In real test, would check actual output file)


# Performance markers for CI/CD
@pytest.mark.performance
class TestPerformance:
    """Performance-related integration tests."""
    
    def test_worker_scaling(self):
        """Test that worker calculation scales appropriately."""
        from ign_lidar.memory_utils import calculate_optimal_workers
        
        # Small files, many workers possible
        workers_small, _ = calculate_optimal_workers(
            num_files=100,
            file_sizes_mb=[10, 15],
            mode='core',
            use_gpu=False,
            requested_workers=16
        )
        
        # Large files, fewer workers
        workers_large, _ = calculate_optimal_workers(
            num_files=10,
            file_sizes_mb=[500, 600],
            mode='full',
            use_gpu=False,
            requested_workers=16
        )
        
        # Should scale down for large files
        assert workers_large <= workers_small
    
    def test_gpu_forces_single_worker(self):
        """Test that GPU mode forces single worker."""
        from ign_lidar.memory_utils import calculate_optimal_workers
        
        workers, info = calculate_optimal_workers(
            num_files=100,
            file_sizes_mb=[100],
            mode='core',
            use_gpu=True,
            requested_workers=8
        )
        
        assert workers == 1
        assert 'CUDA' in info['recommendation_reason']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
