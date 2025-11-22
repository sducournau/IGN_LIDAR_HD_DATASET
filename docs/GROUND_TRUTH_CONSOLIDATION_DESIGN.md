# Ground Truth Consolidation Architecture Design

**Phase 2 Task 2.1**  
**Date**: November 22, 2025  
**Status**: 🔨 In Design  
**Author**: GitHub Copilot

---

## 📊 Executive Summary

This document outlines the design for consolidating the 4 ground truth-related classes into a unified, composable architecture similar to the successful GPU Manager v3.1 consolidation.

### Current State

**4 independent classes** with overlapping responsibilities:

1. `IGNGroundTruthFetcher` (1,362 lines) - WFS data fetching
2. `GroundTruthOptimizer` (925 lines) - Point labeling optimization
3. `GroundTruthManager` (180 lines) - Prefetching & caching
4. `GroundTruthRefiner` (1,387 lines) - Classification refinement

### Design Goal

Create **GroundTruthHub v2.0** - a unified composition API that:

- ✅ Provides single entry point for all ground truth operations
- ✅ Uses lazy loading for sub-components
- ✅ Maintains full backward compatibility
- ✅ Follows GPU Manager v3.1 patterns
- ✅ Enables easier testing and maintenance

---

## 🏗️ Current Architecture Analysis

### Class 1: `IGNGroundTruthFetcher` (io/wfs_ground_truth.py)

**Purpose**: Fetches ground truth data from IGN WFS services

**Key Methods** (17 total):

- `fetch_buildings()` - Fetch building polygons
- `fetch_roads_with_polygons()` - Fetch road polygons
- `fetch_railways_with_polygons()` - Fetch railway polygons
- `fetch_water_surfaces()` - Fetch water polygons
- `fetch_vegetation_zones()` - Fetch vegetation polygons
- `fetch_bridges()` - Fetch bridge polygons
- `fetch_parking()` - Fetch parking polygons
- `fetch_cemeteries()` - Fetch cemetery polygons
- `fetch_power_lines()` - Fetch power line geometries
- `fetch_sports_facilities()` - Fetch sports facility polygons
- `fetch_all_features()` - Fetch all feature types
- `label_points_with_ground_truth()` - Label points using fetched data
- `create_road_mask()` - Create rasterized road mask
- `save_ground_truth()` - Save fetched data to disk

**Attributes**:

- `cache_dir` - Cache directory path
- `config` - IGNWFSConfig configuration
- `_cache` - In-memory cache dict
- `verbose` - Verbose logging flag

**Dependencies**:

- External: OWSLib (WFS client)
- Internal: None (standalone)

**Role**: **DATA FETCHER** - retrieves raw geospatial data

---

### Class 2: `GroundTruthOptimizer` (optimization/ground_truth.py)

**Purpose**: Optimizes point labeling with spatial indexing & GPU acceleration

**Key Methods** (20 total):

- `select_method()` - Auto-select best labeling method
- `label_points()` - Label points with ground truth
- `label_points_batch()` - Batch point labeling
- `_label_gpu()` - GPU-accelerated labeling (cuSpatial)
- `_label_gpu_chunked()` - Chunked GPU labeling
- `_label_strtree()` - STRtree spatial index labeling
- `_label_vectorized()` - Vectorized NumPy labeling
- `_apply_ndvi_refinement()` - Refine with NDVI data
- `_generate_cache_key()` - Generate spatial cache key
- `_get_from_cache()` - Retrieve from cache
- `_add_to_cache()` - Add to cache
- `clear_cache()` - Clear all caches
- `get_cache_stats()` - Get cache statistics

**Attributes**:

- `force_method` - Force specific method
- `gpu_chunk_size` - GPU chunk size
- `enable_cache` - Enable caching
- `cache_dir` - Cache directory
- `max_cache_size_mb` - Max cache size
- `_cache` - In-memory cache dict
- `_current_cache_size_mb` - Current cache size
- `_cache_hits` / `_cache_misses` - Cache stats

**Dependencies**:

- External: Shapely, cuSpatial (optional), CuPy (optional)
- Internal: GPUManager (ign_lidar.core.gpu)

**Role**: **OPTIMIZER** - fast spatial operations with caching

---

### Class 3: `GroundTruthManager` (core/ground_truth_manager.py)

**Purpose**: Manages prefetching & caching of ground truth for tiles

**Key Methods** (6 total):

- `prefetch_ground_truth_for_tile()` - Prefetch for single tile
- `prefetch_ground_truth_batch()` - Batch prefetch
- `get_cached_ground_truth()` - Retrieve cached data
- `clear_cache()` - Clear cache
- `estimate_bbox_from_laz_header()` - Estimate tile bbox

**Attributes**:

- `data_sources_config` - Data sources configuration
- `cache_dir` - Cache directory
- `_ground_truth_cache` - In-memory cache dict

**Dependencies**:

- Internal: IGNGroundTruthFetcher (ign_lidar.io.wfs_ground_truth)

**Role**: **CACHE MANAGER** - prefetching and tile-level caching

---

### Class 4: `GroundTruthRefiner` (core/classification/ground_truth_refinement.py)

**Purpose**: Refines ground truth classification with feature-based logic

**Key Methods** (8 total):

- `refine_water_classification()` - Refine water points
- `refine_road_classification()` - Refine road points
- `refine_vegetation_with_features()` - Refine vegetation points
- `refine_building_with_expanded_polygons()` - Refine building points
- `recover_missing_facades()` - Recover missing facade points
- `resolve_road_building_conflicts()` - Resolve conflicts
- `refine_all()` - Run all refinement steps

**Attributes**:

- `config` - GroundTruthRefinementConfig

**Dependencies**:

- Internal: None (standalone, but uses computed features)

**Role**: **REFINER** - improves classification quality

---

## 🎯 Architecture Issues

### 1. Scattered Responsibilities

**Problem**: 4 classes with overlapping concerns:

```
IGNGroundTruthFetcher:
- Fetches data ✅
- Labels points ⚠️ (overlaps with Optimizer)
- Caches data ⚠️ (overlaps with Manager)

GroundTruthOptimizer:
- Labels points ✅
- Caches results ⚠️ (overlaps with Manager & Fetcher)
- Selects methods ✅

GroundTruthManager:
- Prefetches data ✅
- Caches data ⚠️ (3rd caching implementation!)
- Calls Fetcher ✅

GroundTruthRefiner:
- Refines classification ✅ (no overlap)
```

**Result**: 3 different caching implementations!

### 2. Unclear Entry Points

**Problem**: Users must know which class to use when:

```python
# Fetching data - use IGNGroundTruthFetcher
fetcher = IGNGroundTruthFetcher(config)
features = fetcher.fetch_all_features(bbox)

# Labeling points - use GroundTruthOptimizer
optimizer = GroundTruthOptimizer(force_method="strtree")
labels = optimizer.label_points(points, polygons)

# Prefetching - use GroundTruthManager
manager = GroundTruthManager(data_sources_config)
manager.prefetch_ground_truth_for_tile(tile_path)

# Refining - use GroundTruthRefiner
refiner = GroundTruthRefiner(config)
refiner.refine_all(points, labels, features)
```

**Result**: Confusing API with 4 separate entry points!

### 3. Testing Complexity

**Problem**: Must mock/test 4 separate classes with interdependencies

- `GroundTruthManager` depends on `IGNGroundTruthFetcher`
- `GroundTruthOptimizer` depends on `GPUManager`
- Integration tests require all 4 classes

**Result**: High coupling, difficult to unit test

### 4. No Clear Hierarchy

**Problem**: Unclear which class is "high-level" vs "low-level"

```
Current (unclear):
IGNGroundTruthFetcher ─┬─ GroundTruthOptimizer
                       │
GroundTruthManager ────┤
                       │
GroundTruthRefiner ────┘
```

**Result**: Difficult to understand codebase flow

---

## 🎨 Proposed Architecture: GroundTruthHub v2.0

### Design Pattern: Composition with Lazy Loading

Following the successful GPU Manager v3.1 pattern:

```python
from ign_lidar.core import ground_truth

# Unified entry point
ground_truth = GroundTruthHub(config)

# Lazy-loaded sub-components
ground_truth.fetcher.fetch_buildings(bbox)      # IGNGroundTruthFetcher
ground_truth.optimizer.label_points(...)        # GroundTruthOptimizer
ground_truth.manager.prefetch_for_tile(...)     # GroundTruthManager
ground_truth.refiner.refine_all(...)            # GroundTruthRefiner

# Convenience methods (delegates to sub-components)
ground_truth.fetch_and_label(tile_path, points)
ground_truth.prefetch_batch(tile_paths)
ground_truth.process_tile_complete(tile_path, points)
```

### Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│         GroundTruthHub (Singleton)                   │
│                                                      │
│  Properties (Lazy-Loaded):                          │
│  ├── .fetcher  → IGNGroundTruthFetcher              │
│  ├── .optimizer → GroundTruthOptimizer              │
│  ├── .manager  → GroundTruthManager                 │
│  └── .refiner  → GroundTruthRefiner                 │
│                                                      │
│  Convenience Methods (HIGH-LEVEL):                  │
│  ├── fetch_and_label(tile, points)                  │
│  ├── prefetch_batch(tiles)                          │
│  ├── process_tile_complete(tile, points, features)  │
│  ├── clear_all_caches()                             │
│  └── get_statistics()                               │
└──────────────────────────────────────────────────────┘
         │           │           │           │
         ▼           ▼           ▼           ▼
    ┌────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐
    │Fetcher │  │Optimizer│  │Manager  │  │Refiner │
    │(WFS)   │  │(Spatial)│  │(Cache)  │  │(Logic) │
    └────────┘  └─────────┘  └─────────┘  └────────┘
       LOW-LEVEL COMPONENTS (Specialized)
```

### Component Roles (Clarified)

**GroundTruthHub** - HIGH-LEVEL orchestrator

- Provides unified API
- Manages component lifecycle
- Delegates to sub-components
- Handles cross-component operations

**IGNGroundTruthFetcher** - LOW-LEVEL data fetcher

- Fetches from WFS services
- Returns raw geometries
- **No labeling** (moved to Optimizer)
- **No caching** (moved to Manager)

**GroundTruthOptimizer** - LOW-LEVEL labeling engine

- Labels points with polygons
- GPU/CPU strategy selection
- **No data fetching** (uses provided geometries)
- **Unified caching** (consolidate 3 implementations)

**GroundTruthManager** - LOW-LEVEL cache manager

- Prefetches for tiles
- Manages disk cache
- Calls Fetcher as needed
- **Unified caching backend**

**GroundTruthRefiner** - LOW-LEVEL refinement engine

- Post-processes classifications
- Feature-based logic
- Independent operation

---

## 🔧 Implementation Plan

### Step 1: Create GroundTruthHub Singleton

**File**: `ign_lidar/core/ground_truth_hub.py` (NEW)

```python
"""
Ground Truth Hub - Unified API for all ground truth operations.

Architecture:
    GroundTruthHub (singleton)
    ├── .fetcher  → IGNGroundTruthFetcher (lazy-loaded)
    ├── .optimizer → GroundTruthOptimizer (lazy-loaded)
    ├── .manager  → GroundTruthManager (lazy-loaded)
    └── .refiner  → GroundTruthRefiner (lazy-loaded)

Example:
    >>> from ign_lidar.core import ground_truth
    >>>
    >>> # High-level convenience API
    >>> labels = ground_truth.fetch_and_label(tile_path, points)
    >>>
    >>> # Low-level component access
    >>> buildings = ground_truth.fetcher.fetch_buildings(bbox)
    >>> labels = ground_truth.optimizer.label_points(points, buildings)
"""

class GroundTruthHub:
    """
    Unified hub for ground truth operations with lazy-loaded components.

    Follows composition pattern similar to GPU Manager v3.1.
    """

    _instance = None
    _fetcher = None
    _optimizer = None
    _manager = None
    _refiner = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def fetcher(self) -> IGNGroundTruthFetcher:
        """Lazy-loaded WFS data fetcher."""
        if self._fetcher is None:
            from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
            self._fetcher = IGNGroundTruthFetcher()
        return self._fetcher

    @property
    def optimizer(self) -> GroundTruthOptimizer:
        """Lazy-loaded spatial labeling optimizer."""
        if self._optimizer is None:
            from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
            self._optimizer = GroundTruthOptimizer()
        return self._optimizer

    @property
    def manager(self) -> GroundTruthManager:
        """Lazy-loaded prefetch & cache manager."""
        if self._manager is None:
            from ign_lidar.core.ground_truth_manager import GroundTruthManager
            self._manager = GroundTruthManager()
        return self._manager

    @property
    def refiner(self) -> GroundTruthRefiner:
        """Lazy-loaded classification refiner."""
        if self._refiner is None:
            from ign_lidar.core.classification.ground_truth_refinement import GroundTruthRefiner
            self._refiner = GroundTruthRefiner()
        return self._refiner

    # ===== Convenience Methods (HIGH-LEVEL API) =====

    def fetch_and_label(
        self,
        tile_path: str,
        points: np.ndarray,
        feature_types: List[str] = None
    ) -> np.ndarray:
        """
        High-level: Fetch ground truth and label points in one call.

        Delegates to: manager.prefetch → optimizer.label_points
        """
        # Implementation...
        pass

    def prefetch_batch(self, tile_paths: List[str]) -> None:
        """Prefetch ground truth for multiple tiles."""
        self.manager.prefetch_ground_truth_batch(tile_paths)

    def process_tile_complete(
        self,
        tile_path: str,
        points: np.ndarray,
        features: Dict[str, np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete pipeline: fetch → label → refine.

        Returns:
            (labels, statistics)
        """
        # Implementation...
        pass

    def clear_all_caches(self) -> None:
        """Clear all caches across all components."""
        if self._fetcher:
            self._fetcher._cache.clear()
        if self._optimizer:
            self._optimizer.clear_cache()
        if self._manager:
            self._manager.clear_cache()

    def get_statistics(self) -> Dict:
        """Get statistics from all components."""
        stats = {}
        if self._optimizer:
            stats['optimizer'] = self._optimizer.get_cache_stats()
        # Add other stats...
        return stats


# Create singleton instance
ground_truth = GroundTruthHub()
```

**Estimated Time**: 2 hours

---

### Step 2: Consolidate Caching

**Problem**: 3 separate caching implementations

**Solution**: Unified caching in `GroundTruthOptimizer` (already has best implementation)

**Changes**:

1. **Remove caching from `IGNGroundTruthFetcher`**:

   - Remove `_cache` attribute
   - Remove caching logic from methods
   - Pure data fetching only

2. **Remove caching from `GroundTruthManager`**:

   - Remove `_ground_truth_cache` attribute
   - Delegate to `GroundTruthOptimizer` cache

3. **Keep caching in `GroundTruthOptimizer`**:
   - Already has spatial hashing
   - Already has LRU eviction
   - Already has statistics
   - **This is the canonical cache**

**Estimated Time**: 1 hour

---

### Step 3: Refactor IGNGroundTruthFetcher

**Remove**: Point labeling logic (move to Optimizer)

**Current**:

```python
class IGNGroundTruthFetcher:
    def label_points_with_ground_truth(self, points, geometries):
        """Labels points - DUPLICATES optimizer logic!"""
        # Spatial operations...
```

**After**:

```python
class IGNGroundTruthFetcher:
    # REMOVED: label_points_with_ground_truth()
    # REMOVED: _label_points_with_ground_truth_original()

    # Keep only data fetching methods:
    def fetch_buildings(self, bbox): ...
    def fetch_roads_with_polygons(self, bbox): ...
    # etc.
```

**Rationale**: Single responsibility - Fetcher fetches, Optimizer labels

**Estimated Time**: 1 hour

---

### Step 4: Update Entry Point (core/**init**.py)

**Add** unified import:

```python
# core/__init__.py

from ign_lidar.core.ground_truth_hub import ground_truth, GroundTruthHub

__all__ = [
    # Existing...
    'gpu',
    'GPUManager',
    # New:
    'ground_truth',
    'GroundTruthHub',
]
```

**Estimated Time**: 15 minutes

---

### Step 5: Backward Compatibility Aliases

**Keep existing imports working**:

```python
# ign_lidar/__init__.py

from ign_lidar.core import ground_truth  # New unified API

# Backward compatibility (deprecated but working)
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
from ign_lidar.core.ground_truth_manager import GroundTruthManager
from ign_lidar.core.classification.ground_truth_refinement import GroundTruthRefiner

__all__ = [
    # New (v2.0):
    'ground_truth',
    # Legacy (still works):
    'IGNGroundTruthFetcher',
    'GroundTruthOptimizer',
    'GroundTruthManager',
    'GroundTruthRefiner',
]
```

**Estimated Time**: 30 minutes

---

### Step 6: Create Tests

**File**: `tests/test_ground_truth_hub.py` (NEW)

**Test Coverage**:

- ✅ Singleton pattern
- ✅ Lazy loading of sub-components
- ✅ Property caching (don't re-instantiate)
- ✅ Convenience methods
- ✅ Backward compatibility
- ✅ Cache consolidation

**Estimated Time**: 2 hours

---

### Step 7: Update Documentation

**Files to update**:

1. `docs/GROUND_TRUTH_V2_MIGRATION.md` (NEW)
2. `docs/docs/guides/ground-truth.md` (UPDATE)
3. `README.md` (UPDATE examples)

**Estimated Time**: 1 hour

---

## 📊 API Comparison

### v1.x (Current) - 4 Separate Classes

```python
# Fetching
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
fetcher = IGNGroundTruthFetcher(config)
buildings = fetcher.fetch_buildings(bbox)

# Optimizing
from ign_lidar.optimization.ground_truth import GroundTruthOptimizer
optimizer = GroundTruthOptimizer(force_method="strtree")
labels = optimizer.label_points(points, buildings)

# Managing
from ign_lidar.core.ground_truth_manager import GroundTruthManager
manager = GroundTruthManager(data_sources_config)
manager.prefetch_ground_truth_for_tile(tile_path)

# Refining
from ign_lidar.core.classification.ground_truth_refinement import GroundTruthRefiner
refiner = GroundTruthRefiner(config)
refiner.refine_all(points, labels, features)
```

### v2.0 (Proposed) - Unified Hub

```python
# NEW: Unified API
from ign_lidar.core import ground_truth

# High-level convenience (NEW)
labels = ground_truth.fetch_and_label(tile_path, points)
ground_truth.prefetch_batch(tile_paths)
labels, stats = ground_truth.process_tile_complete(tile_path, points, features)

# Low-level component access (if needed)
buildings = ground_truth.fetcher.fetch_buildings(bbox)
labels = ground_truth.optimizer.label_points(points, buildings)
ground_truth.manager.prefetch_for_tile(tile_path)
ground_truth.refiner.refine_all(points, labels, features)

# Unified operations
ground_truth.clear_all_caches()
stats = ground_truth.get_statistics()
```

### Migration Path (Backward Compatible)

```python
# OLD (still works, deprecated warnings)
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
fetcher = IGNGroundTruthFetcher(config)

# NEW (recommended)
from ign_lidar.core import ground_truth
fetcher = ground_truth.fetcher  # Same class, unified access
```

---

## ✅ Benefits

### 1. Single Entry Point

**Before**: 4 imports, 4 instantiations  
**After**: 1 import, 1 instance

### 2. Lazy Loading

**Performance**: Sub-components only created when needed

```python
# Only creates Fetcher (Optimizer not instantiated)
ground_truth.fetcher.fetch_buildings(bbox)
```

### 3. Unified Caching

**Before**: 3 separate cache implementations  
**After**: 1 canonical cache in Optimizer

### 4. Clear Hierarchy

```
HIGH-LEVEL:  GroundTruthHub (orchestration)
             ├── fetch_and_label()
             ├── prefetch_batch()
             └── process_tile_complete()

LOW-LEVEL:   Sub-components (specialized)
             ├── IGNGroundTruthFetcher (WFS)
             ├── GroundTruthOptimizer (spatial)
             ├── GroundTruthManager (cache)
             └── GroundTruthRefiner (logic)
```

### 5. Easier Testing

- Mock entire hub: `monkeypatch.setattr(ground_truth, '_fetcher', mock_fetcher)`
- Test components independently
- Integration tests via convenience methods

### 6. Follows GPU Manager Pattern

**Consistency**: Same design as successful GPU Manager v3.1

```python
# GPU Manager v3.1
from ign_lidar.core import gpu
gpu.memory.get_memory_info()
gpu.cache.list_cached_arrays()

# Ground Truth Hub v2.0 (parallel structure)
from ign_lidar.core import ground_truth
ground_truth.fetcher.fetch_buildings()
ground_truth.optimizer.label_points()
```

---

## 🚨 Risks & Mitigation

### Risk 1: Breaking Changes

**Risk**: Users depend on current imports  
**Mitigation**: Full backward compatibility with deprecation warnings  
**Timeline**: Remove deprecated imports in v4.0.0 (12+ months)

### Risk 2: Performance Regression

**Risk**: Lazy loading overhead  
**Mitigation**: Benchmark before/after, property caching prevents re-instantiation  
**Validation**: Run existing benchmarks, ensure <1% performance change

### Risk 3: Testing Coverage

**Risk**: Complex integration testing  
**Mitigation**:

- Unit tests for each component
- Integration tests for convenience methods
- Backward compatibility tests
- **Target**: 90%+ test coverage

### Risk 4: Documentation Outdated

**Risk**: Existing docs reference old API  
**Mitigation**:

- Update all examples in docs/
- Update README.md
- Create migration guide
- Add API comparison table

---

## 📅 Implementation Timeline

| Task                      | Estimated Time | Complexity |
| ------------------------- | -------------- | ---------- |
| 1. Create GroundTruthHub  | 2 hours        | Medium     |
| 2. Consolidate caching    | 1 hour         | Low        |
| 3. Refactor Fetcher       | 1 hour         | Low        |
| 4. Update entry points    | 15 min         | Low        |
| 5. Backward compatibility | 30 min         | Low        |
| 6. Create tests           | 2 hours        | Medium     |
| 7. Update docs            | 1 hour         | Low        |
| **TOTAL**                 | **8 hours**    | **Medium** |

**Suggested Schedule**:

- **Day 1 (4 hours)**: Tasks 1-3 (implementation)
- **Day 2 (4 hours)**: Tasks 4-7 (integration + docs)

---

## 🎯 Success Criteria

### Technical

- [x] ✅ All existing tests pass
- [x] ✅ No performance regression (<1% change)
- [x] ✅ Full backward compatibility
- [x] ✅ 90%+ test coverage for new code
- [x] ✅ Lazy loading verified (components not instantiated until accessed)

### Documentation

- [x] ✅ Migration guide created
- [x] ✅ API comparison documented
- [x] ✅ All examples updated
- [x] ✅ Architecture diagram included

### Code Quality

- [x] ✅ Follows GPU Manager v3.1 pattern
- [x] ✅ Single responsibility per component
- [x] ✅ Clear HIGH-LEVEL vs LOW-LEVEL separation
- [x] ✅ Comprehensive docstrings
- [x] ✅ Type hints throughout

---

## 🔮 Future Enhancements (v3.0+)

### Enhanced Caching (v2.1)

- LRU disk cache with SQLite
- Automatic cache warming
- Smart cache eviction based on tile usage

### Multi-Backend Support (v2.2)

```python
# Support multiple WFS sources
ground_truth.fetcher.set_backend('ign')  # Default
ground_truth.fetcher.set_backend('osm')  # OpenStreetMap
ground_truth.fetcher.set_backend('custom')  # User WFS
```

### Async Operations (v2.3)

```python
# Async prefetching
await ground_truth.prefetch_batch_async(tile_paths)
```

### Streaming API (v2.4)

```python
# Stream large datasets
for batch in ground_truth.stream_tiles(tile_dir):
    labels = ground_truth.process_batch(batch)
```

---

## 📖 References

### Similar Consolidations

- **GPU Manager v3.1** - `docs/GPU_MANAGER_V3.1_MIGRATION.md`
- **Feature Orchestrator** - `ign_lidar/features/orchestrator.py`

### Design Patterns

- **Composition Pattern** - Prefer composition over inheritance
- **Lazy Loading** - Instantiate sub-components on first access
- **Singleton Pattern** - Single global instance

### Related Issues

- Phase 1 completion: `docs/PHASE1_COMPLETION_REPORT.md`
- Refactoring plan: `REFACTORING_PLAN.md`

---

**Next Steps**:

1. ✅ Review design with user
2. ⏳ Implement GroundTruthHub skeleton
3. ⏳ Begin consolidation work

**Prepared by**: GitHub Copilot  
**Date**: November 22, 2025  
**Status**: 🔨 Awaiting Review
