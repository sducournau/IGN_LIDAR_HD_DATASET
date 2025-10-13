# Phase 3.4: Refactor \_process_tile Method

**Date:** October 13, 2025  
**Session:** 6 (Planning)  
**Status:** READY TO START

## Overview

Apply the proven manager pattern from Phase 3.3.2 to refactor the `_process_tile` method, extracting tile loading and feature computation into dedicated modules.

## Current State

### \_process_tile Method Analysis

- **Current Size:** ~800 lines
- **Target Size:** ~200 lines (75% reduction)
- **Current Structure:** Monolithic with multiple responsibilities
- **Issues:** Hard to test, maintain, and extend

### Responsibilities to Extract

1. **Tile Loading & I/O** → `TileLoader` module
   - LAZ file reading
   - Point cloud filtering
   - Coordinate transformations
   - Data validation
2. **Feature Computation** → `FeatureComputer` module
   - Geometric feature calculation
   - RGB/NIR feature extraction
   - GPU batch processing
   - Feature array assembly

## Phase 3.4 Plan

### Step 1: Create TileLoader Module (2 hours)

**Location:** `ign_lidar/core/modules/tile_loader.py`

**Responsibilities:**

```python
class TileLoader:
    """Handles tile loading and preprocessing."""

    def __init__(self, config: DictConfig):
        """Initialize with tile loading configuration."""

    def load_tile(self, tile_path: str, bbox: Optional[BBox]) -> np.ndarray:
        """Load and filter tile points."""

    def validate_tile(self, points: np.ndarray) -> bool:
        """Validate tile has sufficient points."""

    def transform_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Apply coordinate transformations."""
```

**Methods to Extract:**

- LAZ file reading logic
- BBox filtering
- Point validation
- Coordinate transformation
- Class filtering (if applicable)

### Step 2: Create FeatureComputer Module (2 hours)

**Location:** `ign_lidar/core/modules/feature_computer.py`

**Responsibilities:**

```python
class FeatureComputer:
    """Handles feature computation for point clouds."""

    def __init__(self, config: DictConfig, feature_manager: FeatureManager):
        """Initialize with feature configuration."""

    def compute_features(
        self,
        points: np.ndarray,
        tile_info: Dict
    ) -> Dict[str, np.ndarray]:
        """Compute all requested features."""

    def compute_geometric_features(self, points: np.ndarray) -> Dict:
        """Compute geometric features (CPU or GPU)."""

    def compute_rgb_features(self, points: np.ndarray, tile_info: Dict) -> np.ndarray:
        """Extract RGB features if available."""

    def compute_infrared_features(self, points: np.ndarray, tile_info: Dict) -> np.ndarray:
        """Extract NIR features if available."""
```

**Methods to Extract:**

- Geometric feature computation
- RGB feature extraction
- NIR feature extraction
- GPU batch processing
- Feature array assembly

### Step 3: Refactor \_process_tile (1.5 hours)

**New Structure:**

```python
def _process_tile(self, tile_path: str, *args, **kwargs) -> Optional[Dict]:
    """Process a single LiDAR tile using manager modules."""

    # 1. Load tile using TileLoader
    tile_loader = TileLoader(self.config)
    points = tile_loader.load_tile(tile_path, self.bbox)

    if not tile_loader.validate_tile(points):
        return None

    # 2. Transform coordinates if needed
    points = tile_loader.transform_coordinates(points)

    # 3. Compute features using FeatureComputer
    feature_computer = FeatureComputer(self.config, self.feature_manager)
    features = feature_computer.compute_features(points, tile_info)

    # 4. Create patches if needed
    if self._should_create_patches():
        patches = self._create_patches(points, features)
        self._save_patches(patches, tile_info)

    # 5. Save enriched LAZ if requested
    if self._should_save_enriched():
        self._save_enriched_laz(points, features, tile_info)

    return self._build_result_dict(tile_info, features)
```

**Target:** ~200 lines (75% reduction from 800)

### Step 4: Add Backward Compatibility (0.5 hours)

**Properties to Add:**

```python
# Direct access to tile loader (if needed)
@property
def tile_loader(self) -> TileLoader:
    """Access tile loader (backward compatibility)."""
    return TileLoader(self.config)

# Direct access to feature computer (if needed)
@property
def feature_computer(self) -> FeatureComputer:
    """Access feature computer (backward compatibility)."""
    return FeatureComputer(self.config, self.feature_manager)
```

## Testing Strategy

### Unit Tests

```python
# test_tile_loader.py
def test_load_tile_success()
def test_load_tile_with_bbox()
def test_validate_tile_sufficient_points()
def test_validate_tile_insufficient_points()
def test_transform_coordinates()

# test_feature_computer.py
def test_compute_geometric_features_cpu()
def test_compute_geometric_features_gpu()
def test_compute_rgb_features()
def test_compute_infrared_features()
def test_feature_array_assembly()
```

### Integration Tests

```python
# test_refactored_process_tile.py
def test_process_tile_with_modules()
def test_process_tile_patches_only()
def test_process_tile_enriched_only()
def test_process_tile_both_modes()
def test_backward_compatibility()
```

### Validation Approach

1. Run existing tests first (baseline)
2. Create new module tests
3. Refactor \_process_tile incrementally
4. Run integration tests after each change
5. Final full suite validation

## Success Criteria

### Code Quality

- ✅ `_process_tile` reduced to ~200 lines (75% reduction)
- ✅ Clear separation of concerns
- ✅ Reusable TileLoader and FeatureComputer modules
- ✅ Maintainable and testable code

### Functionality

- ✅ All existing tests pass
- ✅ Zero breaking changes
- ✅ Backward compatibility maintained
- ✅ Same performance or better

### Architecture

- ✅ Manager pattern consistently applied
- ✅ Config-driven design
- ✅ Modular and extensible
- ✅ Clear interfaces

## Risk Assessment

### Low Risk

- **Pattern Proven:** Same approach as Phase 3.3.2
- **Incremental:** Can refactor step-by-step
- **Well-Tested:** Existing test suite catches regressions
- **Reversible:** Git history enables rollback

### Mitigation Strategies

1. **Incremental Development:** One module at a time
2. **Continuous Testing:** Run tests after each change
3. **Rollback Plan:** Git branches for safety
4. **Documentation:** Track decisions and changes

## Timeline

### Estimated: 6 hours

**Module Creation:**

- TileLoader: 2 hours
- FeatureComputer: 2 hours

**Refactoring:**

- \_process_tile method: 1.5 hours
- Testing & validation: 0.5 hours

**Documentation:**

- Inline comments: Concurrent
- Completion doc: 0.5 hours (already included)

## Dependencies

### Required Before Starting

- ✅ Phase 3.3.2 complete (DONE)
- ✅ FeatureManager available (DONE)
- ✅ ConfigValidator available (DONE)
- ✅ Test suite working (DONE)

### No Blockers

All prerequisites met. Ready to begin immediately.

## Next Steps

### Immediate Actions

1. **Analyze \_process_tile:** Map out all responsibilities (30 min)
2. **Create TileLoader:** Extract I/O logic (2 hours)
3. **Create FeatureComputer:** Extract feature logic (2 hours)
4. **Refactor \_process_tile:** Integrate modules (1.5 hours)
5. **Validate:** Run full test suite (included in timeline)

### First Command

```bash
# Analyze _process_tile method structure
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
grep -n "def _process_tile" ign_lidar/core/processor.py
```

## Expected Outcomes

### After Phase 3.4 Completion

1. **Cleaner Code:** 75% reduction in method size
2. **Better Testing:** Isolated unit tests for each module
3. **Easier Maintenance:** Clear responsibilities
4. **Foundation Set:** Pattern for remaining methods

### Progress Toward Overall Goal

- Phase 3.3: ✅ COMPLETE (40% of Phase 3)
- Phase 3.4: 🎯 STARTING (30% of Phase 3)
- Remaining: Phase 3.5-3.6 (30% of Phase 3)

**Overall Consolidation Progress: 62% → 72% (after Phase 3.4)**

---

**Status:** ✅ READY TO START  
**Confidence:** HIGH (proven methodology)  
**Risk:** LOW (incremental approach)

**Next Action:** Begin TileLoader module creation
