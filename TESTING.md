# Testing Configuration Guide

This document explains how to use the pytest integration in VS Code for the IGN LiDAR HD package.

## Setup

### 1. Install Development Dependencies

```bash
pip install -e .[dev]
# or
pip install pytest pytest-cov
```

### 2. Verify Configuration

The project is now configured with:

- **pytest.ini**: Main pytest configuration
- **pyproject.toml**: Tool-specific pytest settings
- **.vscode/settings.json**: VS Code Python testing integration
- **.vscode/tasks.json**: Quick run tasks for different test scenarios
- **.vscode/launch.json**: Debug configurations for tests
- **tests/conftest.py**: Shared fixtures and test data helpers

## Using Tests in VS Code

### Testing Panel

1. Open the Testing view (beaker icon in the sidebar or `Ctrl+Shift+T`)
2. Click "Configure Python Tests" if needed
3. Tests should auto-discover from the `tests/` directory
4. Click the play button next to any test to run it
5. Click the debug icon to debug a specific test

### Run via Tasks

Press `Ctrl+Shift+P` and type "Tasks: Run Task", then select:

- **Run All Tests**: Execute all tests
- **Run Integration Tests**: Only integration tests (marked with `@pytest.mark.integration`)
- **Run Unit Tests**: Only unit tests (marked with `@pytest.mark.unit`)
- **Run Tests with Coverage**: Generate coverage report
- **Run Current Test File**: Test the currently open file

### Debug Tests

1. Set breakpoints in your test file
2. Press `F5` or open Run and Debug panel
3. Select a debug configuration:
   - **Debug Current Test File**: Debug all tests in open file
   - **Debug All Tests**: Debug entire test suite
   - **Debug Integration Tests**: Debug only integration tests
   - **Debug Specific Test**: Debug selected test function

## Test Organization

### Test Markers

Tests can be marked for selective execution:

```python
import pytest

@pytest.mark.integration
def test_with_real_data():
    """Integration test using data/test_integration/"""
    pass

@pytest.mark.unit
def test_pure_logic():
    """Unit test without external dependencies"""
    pass

@pytest.mark.slow
def test_long_running():
    """Test that takes significant time"""
    pass

@pytest.mark.gpu
def test_gpu_acceleration():
    """Test requiring GPU"""
    pass
```

### Running Specific Test Groups

```bash
# Run only integration tests
pytest -m integration

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"

# Run specific file
pytest tests/test_integration_e2e.py

# Run specific test
pytest tests/test_integration_e2e.py::test_function_name
```

## Test Data

### Integration Test Data

Integration tests use data from `data/test_integration/`:

```python
def test_with_integration_data(test_data_dir, sample_laz_file):
    """
    Use fixtures from conftest.py to access test data.

    Args:
        test_data_dir: Path to data/test_integration/
        sample_laz_file: Path to a sample .laz file
    """
    assert test_data_dir.exists()
    assert sample_laz_file.exists()
```

### Fixtures Available

From `tests/conftest.py`:

- `project_root`: Project root directory path
- `test_data_dir`: Path to data/test_integration/
- `test_output_dir`: Path to data/test_output/
- `cache_dir`: Path to data/cache/
- `sample_laz_file`: First .laz file found in test_integration/
- `sample_config`: Path to examples/config_complete.yaml

## Coverage Reports

Generate coverage reports:

```bash
# Terminal coverage report
pytest --cov=ign_lidar --cov-report=term

# HTML coverage report (opens in browser)
pytest --cov=ign_lidar --cov-report=html
open htmlcov/index.html  # macOS/Linux
```

## Troubleshooting

### Tests Not Discovered

1. Ensure pytest is installed: `pip install pytest`
2. Check VS Code settings: `.vscode/settings.json` should have `python.testing.pytestEnabled: true`
3. Reload VS Code window: `Ctrl+Shift+P` → "Developer: Reload Window"
4. Check Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"

### Import Errors in Tests

- Make sure package is installed in editable mode: `pip install -e .`
- Check that the Python interpreter matches your environment
- Verify `PYTHONPATH` includes the project root

### Test Data Not Found

- Integration tests expect data in `data/test_integration/`
- Tests will auto-skip if data is not available (using pytest.skip)
- Add your test LAZ files to this directory as needed

## Best Practices

1. **Keep tests fast**: Mark slow tests with `@pytest.mark.slow`
2. **Use fixtures**: Share common setup via conftest.py
3. **Meaningful names**: Use descriptive test function names
4. **Test isolation**: Each test should be independent
5. **Mark appropriately**: Use markers for integration, unit, gpu, etc.
6. **Auto-discovery**: Tests will be automatically marked based on filename patterns

## Command Line Reference

```bash
# Basic test runs
pytest                              # Run all tests
pytest tests/                       # Run tests in directory
pytest tests/test_file.py          # Run specific file
pytest tests/test_file.py::test_func  # Run specific test

# With options
pytest -v                          # Verbose output
pytest -s                          # Show print statements
pytest -x                          # Stop on first failure
pytest -k "pattern"                # Run tests matching pattern
pytest --lf                        # Run last failed tests
pytest --tb=short                  # Short traceback

# With markers
pytest -m integration              # Run integration tests
pytest -m "not slow"               # Skip slow tests
pytest -m "unit or integration"    # Multiple markers

# With coverage
pytest --cov=ign_lidar            # Coverage report
pytest --cov-report=html          # HTML coverage
pytest --cov-report=term-missing  # Show missing lines
```

## VS Code Keyboard Shortcuts

- `Ctrl+Shift+T`: Open Testing view
- `F5`: Start debugging
- `Shift+F5`: Stop debugging
- `F9`: Toggle breakpoint
- `F10`: Step over
- `F11`: Step into
- `Shift+F11`: Step out

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [VS Code Python Testing](https://code.visualstudio.com/docs/python/testing)
- [pytest markers](https://docs.pytest.org/en/stable/example/markers.html)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
