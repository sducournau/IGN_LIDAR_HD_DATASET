# Coding Patterns & Best Practices

## Common Patterns in IGN LiDAR HD

### 1. GPU/CPU Fallback Pattern

**Context:** Many operations support GPU acceleration but need CPU fallback.

**Pattern:**

```python
def compute_feature(points: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Compute feature with automatic GPU/CPU fallback."""
    if use_gpu and GPU_AVAILABLE:
        try:
            import cupy as cp
            # Convert to GPU
            points_gpu = cp.asarray(points)
            # GPU computation
            result = _compute_gpu(points_gpu)
            # Convert back to CPU
            return cp.asnumpy(result)
        except (cp.cuda.runtime.CUDARuntimeError, MemoryError) as e:
            logger.warning(f"GPU computation failed: {e}, falling back to CPU")
            # Fall through to CPU

    # CPU computation
    return _compute_cpu(points)
```

**Used in:**

- `features/compute/normals.py`
- `features/compute/curvature.py`
- `features/compute/eigenvalues.py`

### 2. Strategy Pattern for Processing

**Context:** Different processing strategies (CPU, GPU, GPU_CHUNKED) need same interface.

**Pattern:**

```python
from abc import ABC, abstractmethod

class ProcessingStrategy(ABC):
    """Base class for processing strategies."""

    @abstractmethod
    def compute_features(self, points: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Compute features for point cloud."""
        pass

    @abstractmethod
    def supports_batch_size(self) -> bool:
        """Whether strategy supports batch size parameter."""
        pass

class CPUStrategy(ProcessingStrategy):
    """CPU-based processing using scikit-learn."""

    def compute_features(self, points: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        # CPU implementation
        return features

class GPUStrategy(ProcessingStrategy):
    """GPU-based processing using CuPy/cuML."""

    def compute_features(self, points: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        # GPU implementation
        return features
```

**Used in:**

- `features/strategies.py`
- `features/strategy_cpu.py`
- `features/strategy_gpu.py`
- `features/strategy_gpu_chunked.py`

### 3. Configuration with Hydra Pattern

**Context:** Hierarchical configuration with defaults and overrides.

**Pattern:**

```python
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict, Any

def initialize_from_config(config: Union[DictConfig, Dict, None] = None, **kwargs) -> object:
    """Initialize object from Hydra config or kwargs."""

    # If no config, build from kwargs (backward compatibility)
    if config is None:
        config = OmegaConf.create(kwargs)
    elif isinstance(config, dict):
        config = OmegaConf.create(config)

    # Access with defaults
    param1 = OmegaConf.select(config, 'section.param1', default='default_value')
    param2 = config.section.get('param2', 'default_value')

    # Validate
    if param1 not in ['valid1', 'valid2']:
        raise ValueError(f"Invalid param1: {param1}")

    return MyClass(param1, param2)
```

**Used in:**

- `core/processor.py::LiDARProcessor.__init__`
- `features/orchestrator.py::FeatureOrchestrator`
- `config/schema.py`

### 4. Memory-Aware Chunking Pattern

**Context:** Process large datasets in chunks to avoid OOM.

**Pattern:**

```python
def process_in_chunks(
    data: np.ndarray,
    chunk_size: int,
    process_func: callable,
    show_progress: bool = True
) -> List[Any]:
    """Process large array in memory-safe chunks."""
    n_points = len(data)
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    results = []
    iterator = range(n_chunks)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing chunks")

    for i in iterator:
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_points)
        chunk = data[start_idx:end_idx]

        # Process chunk
        result = process_func(chunk)
        results.append(result)

        # Clean up
        del chunk
        if i % 10 == 0:  # Periodic cleanup
            gc.collect()

    return results
```

**Used in:**

- `features/strategy_gpu_chunked.py`
- `core/processor.py::_process_tile_core`
- `io/wfs_ground_truth.py`

### 5. Feature Dictionary Pattern

**Context:** Consistent structure for feature dictionaries.

**Pattern:**

```python
def compute_features(points: np.ndarray, **params) -> Dict[str, np.ndarray]:
    """
    Compute geometric features for point cloud.

    Returns:
        Dictionary with feature arrays:
        - 'normals': [N, 3] normal vectors
        - 'curvature': [N] curvature values
        - 'planarity': [N] planarity measure
        - etc.
    """
    features = {}

    # Compute features
    features['normals'] = compute_normals(points)
    features['curvature'] = compute_curvature(points, features['normals'])
    features['planarity'] = compute_planarity(points)

    # Validate all features have correct shape
    n_points = len(points)
    for name, values in features.items():
        if len(values) != n_points:
            raise ValueError(f"Feature {name} has wrong length: {len(values)} != {n_points}")

    return features
```

**Used throughout:**

- `features/orchestrator.py`
- `features/compute/*.py`
- `core/processor.py`

### 6. Error Handling with Custom Exceptions Pattern

**Context:** Provide clear, actionable error messages.

**Pattern:**

```python
from ign_lidar.core.error_handler import (
    ProcessingError,
    GPUMemoryError,
    FileProcessingError
)

def process_tile(tile_path: Path) -> Dict:
    """Process LiDAR tile with comprehensive error handling."""

    # Validate input
    if not tile_path.exists():
        raise FileProcessingError(
            f"Tile file not found: {tile_path}",
            file_path=tile_path
        )

    try:
        # Load data
        data = load_laz(tile_path)

        # Process with GPU
        try:
            result = process_gpu(data)
        except (cp.cuda.runtime.CUDARuntimeError, MemoryError) as e:
            # GPU OOM - raise specific error with recovery suggestion
            raise GPUMemoryError(
                f"GPU out of memory processing {tile_path}. "
                f"Try reducing gpu_batch_size or disabling GPU.",
                tile_path=tile_path,
                required_memory=estimate_memory(data)
            ) from e

        return result

    except Exception as e:
        # Wrap unexpected errors
        raise ProcessingError(
            f"Failed to process tile {tile_path}: {e}",
            tile_path=tile_path
        ) from e
```

**Used in:**

- `core/processor.py`
- `features/orchestrator.py`
- `io/laz.py`

### 7. Skip Checker Pattern

**Context:** Resume interrupted workflows by skipping completed files.

**Pattern:**

```python
class PatchSkipChecker:
    """Check if patch already exists and is valid."""

    def should_skip(
        self,
        output_dir: Path,
        patch_name: str,
        validate_content: bool = True
    ) -> bool:
        """Check if patch can be skipped."""

        # Check all output formats
        for fmt in self.output_formats:
            patch_path = output_dir / f"{patch_name}.{fmt}"

            # File doesn't exist
            if not patch_path.exists():
                return False

            # File too small
            if patch_path.stat().st_size < self.min_file_size:
                logger.warning(f"File too small, reprocessing: {patch_path}")
                return False

            # Validate content if requested
            if validate_content:
                if not self._validate_patch(patch_path):
                    logger.warning(f"Invalid content, reprocessing: {patch_path}")
                    return False

        return True  # All formats exist and valid
```

**Used in:**

- `core/skip_checker.py`
- `core/processor.py`

### 8. Performance Monitoring Pattern

**Context:** Track performance metrics during processing.

**Pattern:**

```python
from ign_lidar.core import PerformanceMonitor

def process_with_monitoring(data: np.ndarray) -> Dict:
    """Process data with performance monitoring."""

    monitor = PerformanceMonitor()

    # Phase 1: Loading
    monitor.start_phase("loading")
    loaded_data = load_data(data)
    monitor.end_phase("loading")

    # Phase 2: Feature computation
    monitor.start_phase("features")
    features = compute_features(loaded_data)
    monitor.end_phase("features")

    # Phase 3: Saving
    monitor.start_phase("saving")
    save_results(features)
    monitor.end_phase("saving")

    # Get summary
    summary = monitor.get_summary()
    logger.info(f"Loading: {summary['loading']['duration']:.2f}s")
    logger.info(f"Features: {summary['features']['duration']:.2f}s")
    logger.info(f"Saving: {summary['saving']['duration']:.2f}s")

    return features
```

**Used in:**

- `core/processor.py`
- `features/orchestrator.py`

### 9. Deprecation Warning Pattern

**Context:** Support legacy API while encouraging migration.

**Pattern:**

```python
import warnings
from typing import Optional

def legacy_function(param1: str) -> str:
    """Legacy function (deprecated)."""
    warnings.warn(
        "\n" + "="*70 + "\n"
        "DEPRECATION WARNING\n"
        "="*70 + "\n"
        "legacy_function() is deprecated and will be removed in v4.0.\n"
        "Use new_function() instead.\n\n"
        "Migration:\n"
        "  OLD: legacy_function(param1)\n"
        "  NEW: new_function(param1)\n"
        "="*70,
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(param1)

class LegacyClass:
    """Legacy class with deprecation warnings."""

    @property
    def old_property(self) -> str:
        """Old property (deprecated)."""
        warnings.warn(
            "old_property is deprecated, use new_property",
            DeprecationWarning,
            stacklevel=2
        )
        return self.new_property
```

**Used in:**

- `__init__.py` (legacy imports)
- `core/processor.py` (legacy properties)

### 10. Metadata Tracking Pattern

**Context:** Track processing metadata for reproducibility.

**Pattern:**

```python
from pathlib import Path
from typing import Dict, Any
import json

class ProcessingMetadata:
    """Track metadata for processed tiles."""

    def __init__(self):
        self.metadata = {
            'version': __version__,
            'timestamp': time.time(),
            'config': {},
            'features': [],
            'performance': {}
        }

    def add_config(self, config: Dict[str, Any]):
        """Add configuration to metadata."""
        self.metadata['config'] = OmegaConf.to_container(config)

    def add_features(self, feature_names: List[str]):
        """Add feature list to metadata."""
        self.metadata['features'] = feature_names

    def add_performance(self, phase: str, duration: float):
        """Add performance metrics."""
        self.metadata['performance'][phase] = duration

    def save(self, output_path: Path):
        """Save metadata to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
```

**Used in:**

- `core/processing_metadata.py`
- `io/metadata.py`

## Anti-Patterns to Avoid

### 1. GPU + Multiprocessing ❌

```python
# DON'T: Use GPU with multiple workers
processor = LiDARProcessor(use_gpu=True)
processor.process_directory(input_dir, output_dir, num_workers=4)  # FAILS!

# DO: Use GPU with single worker OR CPU with multiple workers
processor = LiDARProcessor(use_gpu=True)
processor.process_directory(input_dir, output_dir, num_workers=1)  # OK
```

### 2. Missing Memory Cleanup ❌

```python
# DON'T: Process large datasets without cleanup
for i in range(1000):
    data = load_large_array()
    result = process(data)
    results.append(result)

# DO: Clean up periodically
for i in range(1000):
    data = load_large_array()
    result = process(data)
    results.append(result)
    del data
    if i % 10 == 0:
        gc.collect()
```

### 3. Hardcoded Paths ❌

```python
# DON'T: Hardcode paths
data = load("/home/user/data/tile.laz")

# DO: Use Path objects and config
from pathlib import Path
data_path = Path(config.input_dir) / "tile.laz"
data = load(data_path)
```

### 4. Silent Failures ❌

```python
# DON'T: Catch and ignore exceptions
try:
    result = process_important_data()
except Exception:
    pass  # Silent failure!

# DO: Log and re-raise or handle explicitly
try:
    result = process_important_data()
except SpecificError as e:
    logger.error(f"Failed to process: {e}")
    # Handle or re-raise
    raise
```

### 5. Missing Type Hints ❌

```python
# DON'T: No type hints
def compute_features(points, k):
    return features

# DO: Clear type hints
def compute_features(
    points: np.ndarray,
    k: int
) -> Dict[str, np.ndarray]:
    return features
```

## Code Review Checklist

When reviewing code in this project, check:

- [ ] **Type hints:** All public functions have type annotations
- [ ] **Docstrings:** Google-style docstrings with Args/Returns/Raises
- [ ] **Error handling:** Appropriate custom exceptions with helpful messages
- [ ] **GPU fallback:** GPU code has CPU fallback
- [ ] **Memory cleanup:** Large arrays deleted, gc.collect() called
- [ ] **Tests:** Unit tests and integration tests added
- [ ] **Config support:** New parameters added to config schema
- [ ] **Deprecation:** Legacy code marked deprecated with warnings
- [ ] **Performance:** No obvious performance bottlenecks
- [ ] **Documentation:** Updated docs for new features

## Testing Patterns

### Test Fixtures Pattern

```python
import pytest
import numpy as np

@pytest.fixture
def sample_point_cloud():
    """Generate sample point cloud for testing."""
    np.random.seed(42)
    points = np.random.rand(1000, 3) * 100
    return points

@pytest.fixture
def sample_config():
    """Create sample configuration."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        'processor': {'lod_level': 'LOD2'},
        'features': {'mode': 'lod2'}
    })

def test_feature_computation(sample_point_cloud, sample_config):
    """Test feature computation with fixtures."""
    features = compute_features(sample_point_cloud, sample_config)
    assert 'normals' in features
    assert len(features['normals']) == len(sample_point_cloud)
```

### Parametrized Tests Pattern

```python
@pytest.mark.parametrize("lod_level,expected_features", [
    ("LOD2", 12),
    ("LOD3", 38),
])
def test_feature_modes(lod_level, expected_features):
    """Test different LOD levels produce expected feature counts."""
    config = {'processor': {'lod_level': lod_level}}
    processor = LiDARProcessor(config)
    assert len(processor.feature_list) == expected_features
```

### GPU Test Pattern

```python
@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_gpu_features():
    """Test GPU feature computation."""
    points = np.random.rand(10000, 3)
    features_gpu = compute_features(points, use_gpu=True)
    features_cpu = compute_features(points, use_gpu=False)

    # Results should be similar
    np.testing.assert_allclose(
        features_gpu['normals'],
        features_cpu['normals'],
        rtol=1e-5
    )
```
