"""
Pytest configuration and fixtures for IGN LiDAR HD tests.

This file provides shared fixtures and configuration for all tests,
including paths to test integration data.
"""
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test integration data directory."""
    return project_root / "data" / "test_integration"


@pytest.fixture(scope="session")
def test_output_dir(project_root):
    """Return the test output directory."""
    output_dir = project_root / "data" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def cache_dir(project_root):
    """Return the cache directory."""
    cache = project_root / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture
def sample_laz_file(test_data_dir):
    """
    Return a sample LAZ file from test integration data.
    
    This fixture will look for any .laz file in the test_integration directory.
    If no file is found, the test should be skipped.
    """
    if not test_data_dir.exists():
        pytest.skip(f"Test integration data directory not found: {test_data_dir}")
    
    laz_files = list(test_data_dir.glob("*.laz"))
    if not laz_files:
        pytest.skip(f"No LAZ files found in {test_data_dir}")
    
    return laz_files[0]


@pytest.fixture
def sample_config(project_root):
    """Return a sample configuration file path."""
    config_path = project_root / "examples" / "config_complete.yaml"
    if not config_path.exists():
        pytest.skip(f"Sample config not found: {config_path}")
    return config_path


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location and content.
    
    - Tests in test_integration_*.py are marked as integration tests
    - Tests with 'gpu' in the name are marked as gpu tests
    """
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark GPU tests
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if "slow" in item.nodeid.lower() or "e2e" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
