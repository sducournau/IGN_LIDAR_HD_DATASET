# Design Patterns and Guidelines

## Core Design Patterns

### 1. Strategy Pattern (Feature Computation)
**Location:** `ign_lidar/features/strategies.py`, `strategy_cpu.py`, `strategy_gpu.py`, `strategy_gpu_chunked.py`

**Purpose:** Select CPU or GPU computation strategy at runtime

**Implementation:**
```python
class ComputeStrategy:
    def compute_neighbors(self, points, k): pass

class CPUStrategy(ComputeStrategy):
    def compute_neighbors(self, points, k):
        # scikit-learn implementation
        pass

class GPUStrategy(ComputeStrategy):
    def compute_neighbors(self, points, k):
        # CuPy/cuML implementation
        pass
```

**When to use:** When adding new feature computation methods that can benefit from GPU acceleration

### 2. Orchestrator Pattern (Feature Management)
**Location:** `ign_lidar/features/orchestrator.py`

**Purpose:** Unified API for feature computation and management

**Key Methods:**
- `compute_features()` - Main entry point
- `get_feature_array()` - Retrieve computed features
- `add_custom_feature()` - Extend with custom features

**When to use:** When adding new feature types or computation workflows

### 3. Factory Pattern (Optimization)
**Location:** `ign_lidar/optimization/`

**Purpose:** Create optimized processing components based on system capabilities

**When to use:** When creating adaptive processing logic that depends on hardware

### 4. Configuration Pattern (Hydra)
**Location:** `ign_lidar/config/schema.py`, example YAML configs

**Purpose:** Hierarchical, type-safe configuration with validation

**Structure:**
```yaml
processor:
  lod_level: LOD2
  use_gpu: true
  
features:
  mode: lod2
  k_neighbors: 30
  
data_sources:
  bd_topo:
    buildings: true
```

**When to use:** When adding new configurable parameters

## Architectural Guidelines

### GPU/CPU Compatibility
**Always provide CPU fallback:**
```python
def compute_feature(points: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Compute feature with GPU/CPU support."""
    if use_gpu and GPU_AVAILABLE:
        try:
            import cupy as cp
            points_gpu = cp.asarray(points)
            result = _compute_gpu(points_gpu)
            return cp.asnumpy(result)
        except Exception as e:
            logger.warning(f"GPU failed, falling back to CPU: {e}")
            # Fall through to CPU
    
    # CPU implementation
    return _compute_cpu(points)
```

### Memory Management
**Use AdaptiveMemoryManager for large datasets:**
```python
from ign_lidar.core import AdaptiveMemoryManager

memory_manager = AdaptiveMemoryManager(
    available_memory_gb=32,
    safety_margin=0.2
)

if memory_manager.check_available_memory(required_gb=5.0):
    result = process_large_dataset(data)
```

### Error Handling
**Use custom exceptions:**
```python
from ign_lidar.core.error_handler import (
    ProcessingError,
    GPUMemoryError,
    FileProcessingError
)

try:
    result = process_tile(tile_path)
except GPUMemoryError:
    logger.warning("GPU OOM, falling back to CPU")
    result = process_tile_cpu(tile_path)
except FileProcessingError as e:
    logger.error(f"Failed to process {tile_path}: {e}")
    raise
```

### Performance Monitoring
**Track performance metrics:**
```python
from ign_lidar.core import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_phase("feature_computation")
# ... processing ...
monitor.end_phase("feature_computation")

stats = monitor.get_summary()
print(f"Feature computation: {stats['feature_computation']['duration']:.2f}s")
```

## Common Pitfalls to Avoid

### 1. GPU + Multiprocessing
❌ **Don't:** Use GPU with `num_workers > 1` (CUDA context issues)
✅ **Do:** Use GPU OR multiprocessing, not both

### 2. Memory Leaks
❌ **Don't:** Forget to clean up large arrays
✅ **Do:** Call `gc.collect()` after large operations

### 3. Tile Boundaries
❌ **Don't:** Ignore boundary artifacts in features
✅ **Do:** Use tile stitching for boundary points

### 4. Classification Recalculation
❌ **Don't:** Assume classification-dependent features are valid after ground truth update
✅ **Do:** Recompute classification-dependent features after ground truth

### 5. Config Validation
❌ **Don't:** Process without validating configuration
✅ **Do:** Use `ConfigValidator` before processing

## Extension Points

### Adding New Features
1. Add computation in `features/compute/`
2. Update `FeatureOrchestrator` to include feature
3. Add to appropriate feature mode in `feature_modes.py`
4. Update tests in `tests/test_feature_*.py`
5. Document in `docs/docs/features/`

### Adding New Classification Rules
1. Create rule class extending `BaseRule` in `core/classification/rules/`
2. Implement `evaluate()` method
3. Register with `RuleEngine`
4. Add tests
5. Document in examples

### Adding New Processing Modes
1. Update `ProcessingMode` enum in `config/schema.py`
2. Implement mode logic in `core/processor.py`
3. Add example config in `examples/`
4. Update documentation

## Key Files for Reference

When extending functionality, always check:
- `ign_lidar/core/processor.py` - Main processing logic
- `ign_lidar/features/orchestrator.py` - Feature management
- `ign_lidar/classification_schema.py` - Classification definitions
- `ign_lidar/config/schema.py` - Configuration schema
- `examples/*.yaml` - Example configurations
- `tests/conftest.py` - Test fixtures and utilities
- `.github/copilot-instructions.md` - Detailed coding guidelines
