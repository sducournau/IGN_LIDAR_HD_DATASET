# Code Quality Improvements Implementation Plan

## 📊 Quality Audit Summary

**Overall Code Quality Score: 7.2/10**

- **Total Python Files**: 167 files (~3.1 MB)
- **Recent Activity**: 272 commits in last 3 months
- **Largest Files**: processor.py (3,082 lines), orchestrator.py (2,000 lines)

---

## ✅ Phase 1: Completed Improvements (Current Session)

### 1.1 Debug Logging Cleanup ✅

**Status**: COMPLETED

**Changes Made**:

- Removed excessive debug logging from `processor.py`
- Cleaned up emoji-heavy debug messages (🔍 DEBUG:)
- Converted verbose INFO logs to proper DEBUG level
- Reduced noise in production logs

**Files Modified**:

- `ign_lidar/core/processor.py` (lines 288-420)

**Impact**:

- Cleaner production logs
- Reduced performance overhead
- Better log level separation

### 1.2 TODO Resolution ✅

**Status**: COMPLETED

**Changes Made**:

- Fixed railway classification in `asprs_class_rules.py`
- Changed from `UNCLASSIFIED` to proper `RAIL` (10) classification
- Added clear documentation for ASPRS standard vs extended codes

**Files Modified**:

- `ign_lidar/core/classification/asprs_class_rules.py` (line 565-571)

**Impact**:

- Proper railway point classification
- Eliminated technical debt marker

---

## 🎯 Phase 2: High Priority Refactoring (Next Sprint)

### 2.1 God Class Decomposition

**Priority**: CRITICAL
**Estimated Effort**: 2-3 weeks

#### Target: LiDARProcessor (3,082 lines)

**Proposed Architecture**:

```python
# New structure
ign_lidar/core/
├── processor_core.py         # ProcessorCore (config, initialization)
├── tile_processor.py          # TileProcessor (tile processing logic)
├── patch_generator.py         # PatchGenerator (patch extraction)
├── ground_truth_manager.py    # GroundTruthManager (data fetcher)
├── augmentation_manager.py    # AugmentationManager (DTM augmentation)
└── processor.py               # LiDARProcessor (facade/coordinator)
```

**Refactoring Steps**:

1. **Extract ProcessorCore** (Week 1)

   ```python
   class ProcessorCore:
       """Core initialization and configuration management."""
       def __init__(self, config):
           self.config = config
           self._validate_config()
           self._init_components()

       def _validate_config(self): ...
       def _build_config_from_kwargs(self): ...
       def _apply_auto_optimization(self): ...
   ```

2. **Extract TileProcessor** (Week 1)

   ```python
   class TileProcessor:
       """Tile processing operations."""
       def process_tile(self, laz_file, output_dir): ...
       def _process_tile_core(self, points, classification): ...
       def _prefetch_ground_truth(self, laz_file): ...
   ```

3. **Extract PatchGenerator** (Week 2)

   ```python
   class PatchGenerator:
       """Patch extraction and augmentation."""
       def extract_patches(self, points, features): ...
       def augment_patches(self, patches): ...
       def _save_patch(self, patch, output_path): ...
   ```

4. **Extract GroundTruthManager** (Week 2)

   ```python
   class GroundTruthManager:
       """Ground truth data fetching and management."""
       def __init__(self, data_fetcher): ...
       def prefetch_for_tile(self, bbox): ...
       def classify_points(self, points, ground_truth): ...
   ```

5. **Refactor LiDARProcessor as Facade** (Week 3)
   ```python
   class LiDARProcessor:
       """Main facade coordinating all processing components."""
       def __init__(self, config):
           self.core = ProcessorCore(config)
           self.tile_processor = TileProcessor(self.core)
           self.patch_generator = PatchGenerator(self.core)
           self.ground_truth = GroundTruthManager(self.core.data_fetcher)

       def process_tile(self, laz_file):
           return self.tile_processor.process_tile(laz_file)

       def process_directory(self, input_dir):
           # Coordinate multi-tile processing
           ...
   ```

**Testing Strategy**:

- Create parallel implementation
- Run integration tests on both old and new implementations
- Compare outputs bit-by-bit
- Gradual migration with feature flags

### 2.2 FeatureOrchestrator Refactoring

**Priority**: HIGH
**Estimated Effort**: 1-2 weeks

#### Target: FeatureOrchestrator (2,000 lines)

**Proposed Architecture**:

```python
ign_lidar/features/
├── orchestrator.py           # FeatureOrchestrator (simplified)
├── cache_manager.py          # FeatureCacheManager
├── strategy_factory.py       # StrategyFactory
├── multi_scale_manager.py    # MultiScaleManager
└── performance_tracker.py    # PerformanceTracker
```

**Extraction Plan**:

1. **FeatureCacheManager**:

   ```python
   class FeatureCacheManager:
       """Feature caching with LRU and memory management."""
       def __init__(self, max_size_gb):
           self._cache = {}
           self._max_size = max_size_gb

       def get(self, cache_key): ...
       def put(self, cache_key, features): ...
       def clear(self): ...
   ```

2. **StrategyFactory**:

   ```python
   class ComputationStrategyFactory:
       """Factory for CPU/GPU/GPU_CHUNKED strategies."""
       @staticmethod
       def create_strategy(config, system_info):
           if config.use_gpu and GPU_AVAILABLE:
               if system_info.should_use_chunked():
                   return GPUChunkedStrategy(config)
               return GPUStrategy(config)
           return CPUStrategy(config)
   ```

3. **MultiScaleManager**:
   ```python
   class MultiScaleFeatureManager:
       """Multi-scale feature computation coordination."""
       def compute_multi_scale_features(self, points): ...
       def aggregate_scales(self, scale_features): ...
   ```

### 2.3 Configuration Migration Utility

**Priority**: HIGH
**Estimated Effort**: 1 week

**Create ConfigMigrator**:

```python
# ign_lidar/config/migrator.py
class ConfigMigrator:
    """Migrate V4 (flat) configs to V5 (nested) structure."""

    @staticmethod
    def migrate_v4_to_v5(config: Dict) -> DictConfig:
        """
        Convert flat V4 config to nested V5 structure.

        V4 Example:
            data_sources:
                bd_topo_buildings: true
                bd_topo_roads: true

        V5 Example:
            data_sources:
                bd_topo:
                    enabled: true
                    features:
                        buildings: true
                        roads: true
        """
        if ConfigMigrator.is_v5_format(config):
            return config

        migrated = ConfigMigrator._convert_data_sources(config)
        ConfigMigrator._add_deprecation_warnings(config)
        return OmegaConf.create(migrated)

    @staticmethod
    def is_v5_format(config: Dict) -> bool:
        """Check if config is already V5 format."""
        return ("data_sources" in config and
                "bd_topo" in config.data_sources and
                "enabled" in config.data_sources.bd_topo)
```

**Usage**:

```python
# In processor.__init__
if not ConfigMigrator.is_v5_format(config):
    logger.warning("⚠️  V4 config detected. Consider migrating to V5 format.")
    logger.warning("   Run: ign-lidar-hd migrate-config <old_config.yaml>")
    config = ConfigMigrator.migrate_v4_to_v5(config)
```

---

## 📚 Phase 3: Documentation & Testing (Sprint 2)

### 3.1 Docstring Enhancement

**Priority**: MEDIUM
**Estimated Effort**: 1 week

**Target Files**:

1. `processor.py` - All private methods
2. `orchestrator.py` - Complex methods
3. `classification/` - Rule engine methods

**Template to Apply**:

```python
def _process_tile_core(
    self,
    points: np.ndarray,
    classification: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Core tile processing with feature computation and classification.

    This method performs the main processing pipeline:
    1. Feature computation (geometric, spectral if available)
    2. Ground truth classification (if data sources enabled)
    3. Patch extraction and augmentation
    4. Output generation in requested formats

    Args:
        points: Point cloud array [N, 3] with XYZ coordinates in Lambert-93
        classification: Initial ASPRS classification codes [N]
        metadata: Optional tile metadata including:
            - 'tile_name': str, filename without extension
            - 'bbox': Tuple[float, float, float, float], bounding box
            - 'crs': str, coordinate reference system

    Returns:
        Tuple containing:
        - processed_points: Enhanced point cloud with computed features [N, M]
        - processing_stats: Dictionary with:
            - 'n_points': int, total points processed
            - 'n_patches': int, patches generated
            - 'feature_time_ms': float, feature computation time
            - 'classification_time_ms': float, classification time

    Raises:
        ProcessingError: If tile processing fails
        GPUMemoryError: If GPU runs out of memory (auto-fallback to CPU)
        FileProcessingError: If LAZ file is corrupted

    Example:
        >>> processor = LiDARProcessor(config)
        >>> points = np.random.rand(10000, 3)
        >>> classification = np.ones(10000, dtype=np.uint8)
        >>> processed, stats = processor._process_tile_core(points, classification)
        >>> print(f"Processed {stats['n_points']} points in {stats['feature_time_ms']:.2f}ms")

    Note:
        This is a long-running operation for large tiles (>10M points).
        Consider enabling progress_callback for user feedback.

        For tiles at boundaries, enable stitching to avoid edge artifacts:
        config.stitching.enabled = true

    See Also:
        process_tile: High-level tile processing with file I/O
        TileStitcher: Boundary-aware feature computation
    """
    # Implementation...
```

### 3.2 Test Organization

**Priority**: MEDIUM
**Estimated Effort**: 1 week

**New Test Structure**:

```
tests/
├── unit/                          # Fast, isolated tests (< 1s each)
│   ├── core/
│   │   ├── test_processor_core.py
│   │   ├── test_tile_processor.py
│   │   └── test_patch_generator.py
│   ├── features/
│   │   ├── test_orchestrator.py
│   │   ├── test_cache_manager.py
│   │   └── test_strategies.py
│   ├── classification/
│   │   ├── test_rules_engine.py
│   │   └── test_thresholds.py
│   └── io/
│       ├── test_laz_io.py
│       └── test_data_fetcher.py
├── integration/                   # Component integration (< 10s each)
│   ├── test_feature_pipeline.py
│   ├── test_classification_pipeline.py
│   └── test_e2e_small_tile.py
├── performance/                   # Benchmark tests
│   ├── test_gpu_performance.py
│   ├── test_memory_usage.py
│   └── test_large_tiles.py
└── fixtures/                      # Shared test data
    ├── sample_tiles/
    └── sample_configs/
```

**Coverage Goals**:

- Unit tests: >85%
- Integration tests: Critical paths
- Error recovery: All error paths tested

### 3.3 Property-Based Testing

**Priority**: MEDIUM
**Estimated Effort**: 3 days

**Add Hypothesis Tests**:

```python
# tests/unit/features/test_geometric_properties.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    points=st.lists(
        st.tuples(st.floats(-1000, 1000), st.floats(-1000, 1000), st.floats(-100, 100)),
        min_size=10,
        max_size=1000
    )
)
def test_normal_computation_invariants(points):
    """Test that normal computation satisfies mathematical invariants."""
    points_array = np.array(points)
    normals = compute_normals(points_array, k_neighbors=5)

    # Property 1: Normals should be unit vectors
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)

    # Property 2: Normals should be orthogonal to local plane
    # (tested via dot product with local variations)

    # Property 3: Output shape should match input
    assert normals.shape == points_array.shape
```

---

## 🔧 Phase 4: Code Smell Elimination (Sprint 3)

### 4.1 Extract Protocols/Interfaces

**Priority**: MEDIUM
**Estimated Effort**: 1 week

**Create Protocol Definitions**:

```python
# ign_lidar/protocols.py
from typing import Protocol, Dict, Tuple
import numpy as np
from omegaconf import DictConfig

class FeatureComputerProtocol(Protocol):
    """Protocol for feature computation strategies."""

    def compute_features(
        self,
        points: np.ndarray,
        k_neighbors: int,
        search_radius: float
    ) -> Dict[str, np.ndarray]:
        """Compute geometric features from point cloud."""
        ...

class DataFetcherProtocol(Protocol):
    """Protocol for ground truth data fetching."""

    def fetch_ground_truth(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> Dict[str, Any]:
        """Fetch ground truth data for bounding box."""
        ...

class ClassificationStrategyProtocol(Protocol):
    """Protocol for classification strategies."""

    def classify_points(
        self,
        points: np.ndarray,
        features: Dict[str, np.ndarray],
        ground_truth: Optional[Dict] = None
    ) -> np.ndarray:
        """Classify points based on features and ground truth."""
        ...
```

**Benefits**:

- Better testability (easy mocking)
- Clearer contracts
- Type checking with mypy
- Easier to add new implementations

### 4.2 Configuration Caching Optimization

**Priority**: LOW
**Estimated Effort**: 2 days

**Add Caching Decorator**:

```python
# ign_lidar/core/config_cache.py
from functools import lru_cache
from typing import Any
from omegaconf import DictConfig, OmegaConf

class ConfigCache:
    """Cache for frequently accessed config values."""

    def __init__(self, config: DictConfig):
        self._config = config
        self._cache = {}

    @lru_cache(maxsize=128)
    def get(self, path: str, default: Any = None) -> Any:
        """Get config value with caching."""
        return OmegaConf.select(self._config, path, default=default)

    def invalidate(self):
        """Clear cache (call after config updates)."""
        self.get.cache_clear()
        self._cache.clear()

# Usage in LiDARProcessor
class LiDARProcessor:
    def __init__(self, config):
        self.config_cache = ConfigCache(config)

        # Fast cached access
        self.use_gpu = self.config_cache.get("processor.use_gpu", default=False)
        self.patch_size = self.config_cache.get("processor.patch_size", default=150.0)
```

---

## 🚀 Phase 5: Performance Optimizations (Sprint 4)

### 5.1 Memory Profiling Integration

**Priority**: MEDIUM
**Estimated Effort**: 3 days

**Add Memory Monitoring**:

```python
# ign_lidar/core/memory_profiler.py
import psutil
import tracemalloc
from contextlib import contextmanager

class MemoryProfiler:
    """Memory profiling for large-scale processing."""

    @contextmanager
    def profile_memory(self, operation_name: str):
        """Context manager for memory profiling."""
        tracemalloc.start()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB

        try:
            yield
        finally:
            mem_after = process.memory_info().rss / 1024**2
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            logger.info(
                f"Memory for {operation_name}: "
                f"RSS={mem_after-mem_before:.1f}MB, "
                f"Peak={peak/1024**2:.1f}MB"
            )

# Usage
with profiler.profile_memory("feature_computation"):
    features = orchestrator.compute_features(points)
```

### 5.2 Streaming for Large Tiles

**Priority**: MEDIUM
**Estimated Effort**: 1 week

**Implement Streaming Processing**:

```python
# ign_lidar/core/streaming_processor.py
class StreamingTileProcessor:
    """Process tiles in chunks for memory efficiency."""

    def process_large_tile(
        self,
        laz_file: Path,
        chunk_size: int = 5_000_000
    ):
        """Process tile in streaming chunks."""
        with laspy.open(laz_file) as las:
            total_points = las.header.point_count

            for i in range(0, total_points, chunk_size):
                chunk = las.read_points(i, min(chunk_size, total_points - i))

                # Process chunk
                features = self.compute_features(chunk)
                patches = self.extract_patches(chunk, features)
                self.save_patches(patches)

                # Free memory
                del chunk, features, patches
                gc.collect()
```

---

## 📋 Implementation Checklist

### Sprint 1 (Current - 2 weeks)

- [x] Phase 1.1: Debug logging cleanup
- [x] Phase 1.2: TODO resolution
- [x] Phase 2.1: Extract ProcessorCore class
- [ ] Phase 2.1: Extract TileProcessor class (IN PROGRESS)
- [ ] Phase 2.2: Create FeatureCacheManager
- [ ] Phase 2.3: Create ConfigMigrator

### Sprint 2 (2 weeks)

- [ ] Phase 2.1: Extract PatchGenerator
- [ ] Phase 2.1: Extract GroundTruthManager
- [ ] Phase 2.1: Refactor LiDARProcessor as facade
- [ ] Phase 3.1: Add comprehensive docstrings
- [ ] Phase 3.2: Reorganize test suite

### Sprint 3 (2 weeks)

- [ ] Phase 2.2: Complete FeatureOrchestrator refactoring
- [ ] Phase 3.3: Add property-based tests
- [ ] Phase 4.1: Define protocols/interfaces
- [ ] Phase 4.2: Configuration caching

### Sprint 4 (1 week)

- [ ] Phase 5.1: Memory profiling integration
- [ ] Phase 5.2: Streaming for large tiles
- [ ] Final integration testing
- [ ] Documentation updates

---

## 🎯 Success Metrics

### Code Quality Metrics

- **Target Score**: 8.5/10 (up from 7.2/10)
- **Max File Size**: <1,000 lines per file
- **Max Method Size**: <100 lines per method
- **Test Coverage**: >85% for core modules
- **Documentation Coverage**: 100% for public APIs

### Performance Metrics

- **Memory Reduction**: 20% less peak memory usage
- **Processing Speed**: Maintain or improve current performance
- **GPU Utilization**: >80% when GPU enabled
- **Cache Hit Rate**: >60% for repeated operations

### Maintainability Metrics

- **Cyclomatic Complexity**: <15 per method
- **Coupling**: Reduced inter-module dependencies
- **Cohesion**: Improved within-class focus
- **Technical Debt**: Eliminate all TODO/FIXME comments

---

## 📖 Migration Guide for Users

### V4 → V5 Config Migration

**Automated Migration**:

```bash
# New CLI command
ign-lidar-hd migrate-config config_v4.yaml --output config_v5.yaml
```

**Manual Migration Example**:

```yaml
# V4 (DEPRECATED)
data_sources:
  bd_topo_buildings: true
  bd_topo_roads: true
  bd_topo_road_width_fallback: 4.0

# V5 (CURRENT)
data_sources:
  bd_topo:
    enabled: true
    features:
      buildings: true
      roads: true
    parameters:
      road_width_fallback: 4.0
```

---

## 🔗 Related Documentation

- Architecture diagrams: `docs/diagrams/refactored_architecture.png`
- API documentation: `docs/docs/api/refactored_modules.md`
- Migration guide: `docs/docs/guides/v4_to_v5_migration.md`
- Performance benchmarks: `docs/docs/benchmarks/refactoring_impact.md`

---

**Last Updated**: October 26, 2025  
**Status**: Phase 1 COMPLETED, Phase 2 IN PROGRESS  
**Next Review**: Sprint Planning (Week of October 28, 2025)
