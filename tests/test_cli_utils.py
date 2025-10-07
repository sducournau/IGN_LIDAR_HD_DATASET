"""Tests for CLI utility functions."""

import pytest
from pathlib import Path
import tempfile
import shutil
from ign_lidar.cli_utils import (
    validate_input_path,
    ensure_output_dir,
    discover_laz_files,
    format_file_size,
    log_processing_summary
)
from ign_lidar.cli_config import CLI_DEFAULTS, get_preprocessing_config


class TestValidation:
    """Test validation functions."""
    
    def test_validate_existing_directory(self, tmp_path):
        """Test validation of existing directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        assert validate_input_path(test_dir, path_type="directory")
    
    def test_validate_missing_directory(self, tmp_path):
        """Test validation of missing directory."""
        test_dir = tmp_path / "missing_dir"
        
        assert not validate_input_path(test_dir, must_exist=True, path_type="directory")
    
    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        assert validate_input_path(test_file, path_type="file")
    
    def test_validate_wrong_type(self, tmp_path):
        """Test validation with wrong type."""
        test_file = tmp_path / "test.laz"
        test_file.touch()
        
        # File exists but we're checking for directory
        assert not validate_input_path(test_file, path_type="directory")


class TestOutputDirectory:
    """Test output directory management."""
    
    def test_ensure_output_dir_creates(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "output" / "nested" / "dir"
        
        assert ensure_output_dir(output_dir)
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_ensure_output_dir_existing(self, tmp_path):
        """Test with existing output directory."""
        output_dir = tmp_path / "existing"
        output_dir.mkdir()
        
        assert ensure_output_dir(output_dir)
        assert output_dir.exists()


class TestFileDiscovery:
    """Test file discovery functions."""
    
    def test_discover_single_file(self, tmp_path):
        """Test discovering a single LAZ file."""
        laz_file = tmp_path / "test.laz"
        laz_file.touch()
        
        files = discover_laz_files(laz_file)
        assert len(files) == 1
        assert files[0] == laz_file
    
    def test_discover_non_laz_file(self, tmp_path):
        """Test discovering non-LAZ file returns empty."""
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        
        files = discover_laz_files(txt_file)
        assert len(files) == 0
    
    def test_discover_directory_recursive(self, tmp_path):
        """Test recursive directory discovery."""
        # Create directory structure
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "dir2").mkdir()
        
        # Create LAZ files
        (tmp_path / "file1.laz").touch()
        (tmp_path / "dir1" / "file2.laz").touch()
        (tmp_path / "dir1" / "dir2" / "file3.laz").touch()
        (tmp_path / "dir1" / "other.txt").touch()
        
        files = discover_laz_files(tmp_path, recursive=True)
        assert len(files) == 3
    
    def test_discover_directory_non_recursive(self, tmp_path):
        """Test non-recursive directory discovery."""
        # Create directory structure
        (tmp_path / "subdir").mkdir()
        
        # Create LAZ files
        (tmp_path / "file1.laz").touch()
        (tmp_path / "file2.laz").touch()
        (tmp_path / "subdir" / "file3.laz").touch()
        
        files = discover_laz_files(tmp_path, recursive=False)
        assert len(files) == 2
    
    def test_discover_with_max_files(self, tmp_path):
        """Test max_files limit."""
        # Create multiple LAZ files
        for i in range(10):
            (tmp_path / f"file{i}.laz").touch()
        
        files = discover_laz_files(tmp_path, max_files=5)
        assert len(files) == 5


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(500) == "500.0 B"
        assert format_file_size(1500) == "1.5 KB"
        assert format_file_size(1500000) == "1.4 MB"
        assert format_file_size(1500000000) == "1.4 GB"
    
    def test_log_processing_summary_success(self, caplog):
        """Test logging successful processing."""
        import logging
        caplog.set_level(logging.INFO)
        
        log_processing_summary(
            total_files=10,
            success_count=10,
            operation="Test"
        )
        
        assert "10/10 files succeeded" in caplog.text
    
    def test_log_processing_summary_with_failures(self, caplog):
        """Test logging with failures."""
        import logging
        caplog.set_level(logging.INFO)
        
        failed = [Path("/test/file1.laz"), Path("/test/file2.laz")]
        
        log_processing_summary(
            total_files=10,
            success_count=8,
            failed_files=failed,
            operation="Test"
        )
        
        assert "8 succeeded, 2 failed" in caplog.text


class TestConfiguration:
    """Test configuration classes."""
    
    def test_cli_defaults(self):
        """Test CLI defaults are set."""
        assert CLI_DEFAULTS.DEFAULT_K_NEIGHBORS == 10
        assert CLI_DEFAULTS.DEFAULT_PATCH_SIZE == 150.0
        assert CLI_DEFAULTS.MAX_SAMPLE_POINTS == 1000
        assert len(CLI_DEFAULTS.EXPECTED_GEOMETRIC_FEATURES) == 8
    
    def test_get_preprocessing_config_defaults(self):
        """Test preprocessing config with defaults."""
        config = get_preprocessing_config()
        
        assert config['sor']['enable'] is True
        assert config['sor']['k'] == 12
        assert config['sor']['std_multiplier'] == 2.0
        
        assert config['ror']['enable'] is True
        assert config['ror']['radius'] == 1.0
        assert config['ror']['min_neighbors'] == 4
        
        assert config['voxel']['enable'] is False
    
    def test_get_preprocessing_config_custom(self):
        """Test preprocessing config with custom values."""
        config = get_preprocessing_config(
            enable_sor=False,
            sor_k=20,
            enable_voxel=True,
            voxel_size=0.3
        )
        
        assert config['sor']['enable'] is False
        assert config['sor']['k'] == 20
        assert config['voxel']['enable'] is True
        assert config['voxel']['voxel_size'] == 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
