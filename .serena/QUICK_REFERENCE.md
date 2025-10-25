# Quick Reference Guide

## 🚀 Getting Started

### Essential Files

- **Main Entry Point:** `ign_lidar/core/processor.py::LiDARProcessor`
- **Feature System:** `ign_lidar/features/orchestrator.py::FeatureOrchestrator`
- **Classification:** `ign_lidar/classification_schema.py`
- **Config Schema:** `ign_lidar/config/schema.py`

### Quick Commands

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/ -v -m unit

# Run with coverage
pytest tests/ -v --cov=ign_lidar --cov-report=html

# Format code
black ign_lidar/

# Install in dev mode
pip install -e ".[dev]"
```

## 📋 Common Tasks

### Add New Feature

1. Create computation in `features/compute/new_feature.py`
2. Add to `FeatureOrchestrator` in `features/orchestrator.py`
3. Update feature mode in `features/feature_modes.py`
4. Add test in `tests/test_feature_new.py`
5. Document in `docs/docs/features/`

### Add New Test

```python
import pytest
import numpy as np

@pytest.mark.unit
def test_my_feature(sample_point_cloud):
    """Test feature computation."""
    result = compute_my_feature(sample_point_cloud)
    assert result.shape[0] == len(sample_point_cloud)
```

### GPU-Compatible Code

```python
def compute_feature(points: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """Compute with GPU/CPU fallback."""
    if use_gpu and GPU_AVAILABLE:
        try:
            import cupy as cp
            points_gpu = cp.asarray(points)
            result = _gpu_compute(points_gpu)
            return cp.asnumpy(result)
        except Exception as e:
            logger.warning(f"GPU failed: {e}, using CPU")
    return _cpu_compute(points)
```

### Add Configuration Parameter

1. Add to schema in `config/schema.py`
2. Update default in `config/base_complete.yaml`
3. Access in code: `OmegaConf.select(config, 'section.param', default=value)`
4. Document in `docs/docs/guides/CONFIG_GUIDE.md`

## 🔍 Finding Code

### Key Classes

- `LiDARProcessor` - Main processing pipeline
- `FeatureOrchestrator` - Feature computation management
- `AdaptiveMemoryManager` - Memory management
- `PerformanceMonitor` - Performance tracking
- `ConfigValidator` - Configuration validation

### Key Functions

- `compute_normals()` - Normal vector computation
- `compute_curvature()` - Curvature computation
- `extract_geometric_features()` - Full feature extraction
- `process_tile()` - Single tile processing
- `process_directory()` - Batch processing

### Search Patterns

```python
# Find all feature computation functions
grep -r "def compute_" ign_lidar/features/compute/

# Find GPU-related code
grep -r "cupy\|cuml" ign_lidar/

# Find configuration access
grep -r "OmegaConf.select" ign_lidar/
```

## ⚠️ Common Pitfalls

### DON'T

- ❌ Use GPU with `num_workers > 1`
- ❌ Forget to clean up large arrays (`del`, `gc.collect()`)
- ❌ Hardcode file paths
- ❌ Catch exceptions without logging
- ❌ Skip type hints and docstrings

### DO

- ✅ Provide GPU/CPU fallback
- ✅ Clean up memory periodically
- ✅ Use `Path` objects for paths
- ✅ Log errors with context
- ✅ Add comprehensive type hints and docstrings

## 🧪 Testing Checklist

- [ ] Unit tests added (`@pytest.mark.unit`)
- [ ] Integration test if needed (`@pytest.mark.integration`)
- [ ] GPU test if GPU code (`@pytest.mark.gpu`)
- [ ] Test passes locally: `pytest tests/test_my_feature.py -v`
- [ ] Coverage maintained: `pytest --cov=ign_lidar`

## 📝 Documentation Checklist

- [ ] Google-style docstring with Args/Returns/Raises
- [ ] Type hints on all parameters and return
- [ ] Example usage in docstring
- [ ] Update relevant guide in `docs/docs/guides/`
- [ ] Add to changelog if user-facing

## 🎯 Code Review Checklist

- [ ] Follows PEP 8 (run Black)
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] Error handling appropriate
- [ ] GPU fallback if applicable
- [ ] Memory cleanup if large arrays
- [ ] Tests added/updated
- [ ] Config updated if new parameters
- [ ] Documentation updated
- [ ] No obvious performance issues

## 📊 Performance Tips

### GPU Optimization

- Use `gpu_batch_size` to control memory
- Aim for >80% GPU utilization
- Don't use with `num_workers > 1`

### Memory Optimization

- Use chunked processing for >10M points
- Call `gc.collect()` periodically
- Monitor with `AdaptiveMemoryManager`

### Speed Optimization

- Enable skip existing: `skip_existing=True`
- Cache KD-trees and ground truth
- Use parallel workers for CPU (not GPU)

## 🔧 Troubleshooting

### GPU Issues

```python
# Check GPU availability
from ign_lidar.features.gpu_processor import GPU_AVAILABLE
print(f"GPU available: {GPU_AVAILABLE}")

# Force CPU
processor = LiDARProcessor(config)
processor.feature_orchestrator.use_gpu = False
```

### Memory Issues

```python
# Enable verbose memory monitoring
from ign_lidar.core import AdaptiveMemoryManager
memory_manager = AdaptiveMemoryManager(verbose=True)

# Reduce batch size
config.processor.gpu_batch_size = 1_000_000  # Lower value
```

### Config Issues

```bash
# Validate configuration
ign-lidar-hd validate-config examples/my_config.yaml

# Show resolved config
ign-lidar-hd show-config --config-name my_config
```

## 📞 Getting Help

- **Issues:** https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
- **Docs:** https://sducournau.github.io/IGN_LIDAR_HD_DATASET/
- **Email:** simon.ducournau@gmail.com

## 🎓 Learning Resources

### Essential Reading

1. `README.md` - Project overview
2. `.github/copilot-instructions.md` - Full coding guide
3. `.serena/project_overview.md` - Architecture details
4. `.serena/coding_patterns.md` - Common patterns
5. `docs/docs/architecture.md` - System design

### Example Code

- `examples/` - YAML configurations
- `tests/` - Test examples
- `ign_lidar/core/processor.py` - Main pipeline
- `ign_lidar/features/orchestrator.py` - Feature system

### Key Concepts

- **LOD2 vs LOD3** - Feature complexity levels
- **ASPRS codes** - Standard classification codes
- **Strategy pattern** - CPU/GPU/GPU_CHUNKED selection
- **Hydra config** - Hierarchical configuration
- **Ground truth** - BD TOPO® integration
