"""
Example test demonstrating how to use test integration data fixtures.

This file shows how to write tests using the fixtures defined in conftest.py
and how to access test data from data/test_integration/.
"""
import pytest
from pathlib import Path


@pytest.mark.integration
def test_integration_data_directory_exists(test_data_dir):
    """Test that integration data directory exists."""
    assert test_data_dir.exists(), f"Test data dir should exist: {test_data_dir}"
    assert test_data_dir.is_dir(), "Test data dir should be a directory"


@pytest.mark.integration
def test_can_access_sample_laz_file(sample_laz_file):
    """Test that we can access a sample LAZ file from test integration data."""
    assert sample_laz_file.exists(), f"Sample LAZ file should exist: {sample_laz_file}"
    assert sample_laz_file.suffix == ".laz", "File should be a LAZ file"
    assert sample_laz_file.stat().st_size > 0, "LAZ file should not be empty"


@pytest.mark.integration
def test_output_directory_created(test_output_dir):
    """Test that output directory is created and writable."""
    assert test_output_dir.exists(), "Output directory should be created"
    assert test_output_dir.is_dir(), "Output should be a directory"
    
    # Test that we can write to it
    test_file = test_output_dir / "test_write.txt"
    test_file.write_text("test")
    assert test_file.exists(), "Should be able to write to output directory"
    test_file.unlink()  # Clean up


@pytest.mark.unit
def test_project_structure(project_root):
    """Test that project has expected structure."""
    assert (project_root / "ign_lidar").exists(), "ign_lidar package should exist"
    assert (project_root / "tests").exists(), "tests directory should exist"
    assert (project_root / "pyproject.toml").exists(), "pyproject.toml should exist"
    assert (project_root / "README.md").exists(), "README.md should exist"


@pytest.mark.unit
def test_cache_directory_available(cache_dir):
    """Test that cache directory is available."""
    assert cache_dir.exists(), "Cache directory should exist"
    assert cache_dir.is_dir(), "Cache should be a directory"


@pytest.mark.integration
def test_sample_config_exists(sample_config):
    """Test that sample configuration is available."""
    assert sample_config.exists(), f"Sample config should exist: {sample_config}"
    assert sample_config.suffix in [".yaml", ".yml"], "Config should be YAML"


@pytest.mark.unit
def test_example_unit_test():
    """Example of a simple unit test without dependencies."""
    # Simple logic test
    result = 2 + 2
    assert result == 4, "Basic math should work"


@pytest.mark.integration
@pytest.mark.slow
def test_example_slow_integration_test(test_data_dir):
    """Example of a test marked as both integration and slow."""
    # This test would contain slow operations like processing large files
    import time
    time.sleep(0.1)  # Simulate slow operation
    assert test_data_dir.exists()


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])
