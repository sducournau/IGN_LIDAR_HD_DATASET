# Tests Directory

Unit tests and integration tests for the IGN LiDAR HD library.

## Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_core.py             # Core functionality tests
├── test_features.py         # Feature extraction tests
├── test_building_features.py # Building-specific feature tests
├── test_cli.py              # CLI tests
├── test_config_gpu.py       # GPU configuration tests
├── test_configuration.py    # General configuration tests
└── test_new_features.py     # New feature validation tests
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_core.py
```

### Run with coverage
```bash
pytest --cov=ign_lidar tests/
```

### Run with verbose output
```bash
pytest -v tests/
```

## Test Categories

### Unit Tests
- `test_core.py`: Core classes and functions
- `test_features.py`: Feature extraction algorithms
- `test_building_features.py`: Building-specific features

### Integration Tests
- `test_cli.py`: Command-line interface
- `test_configuration.py`: Configuration management

### Validation Tests
- `test_config_gpu.py`: GPU setup validation
- `test_new_features.py`: New feature validation

## Guidelines

- Tests should be fast and independent
- Use fixtures in `conftest.py` for shared setup
- Mock external dependencies (files, network, etc.)
- Aim for high code coverage
- Follow naming convention: `test_<functionality>.py`

## Moved Scripts

The following non-test scripts have been moved:
- `validate_features.py` → `scripts/validation/`
- `benchmark_optimization.py` → `scripts/benchmarks/`
- `test_consolidation.py` → `scripts/legacy/` (archived)

These were validation/benchmark scripts, not unit tests.
