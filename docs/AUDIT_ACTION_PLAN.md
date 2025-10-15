# Classification System Audit - Action Plan

**Date:** October 16, 2025  
**Status:** Ready for Implementation  
**Priority:** Address critical issues first

---

## Quick Summary

The classification system is **well-designed** with good architecture and comprehensive features. However, there are **15 identified issues** that should be addressed to improve consistency, performance, and reliability.

**Overall Rating:** 7.5/10 (Good, with room for improvement)

---

## Critical Issues (Implement Immediately)

### Issue #8: Conflicting Height Thresholds ⚠️ CRITICAL

**Problem:** Transport detection uses different height thresholds in different modules.

**Files Affected:**

- `ign_lidar/core/modules/transport_detection.py`
- `ign_lidar/core/modules/classification_refinement.py`
- `ign_lidar/core/modules/advanced_classification.py`

**Current State:**

```python
# TransportDetectionConfig
road_height_max: 0.5m
rail_height_max: 0.8m

# RefinementConfig
ROAD_HEIGHT_MAX: 1.5m
RAIL_HEIGHT_MAX: 1.2m

# Ground truth filtering (advanced_classification.py)
Road: -0.3m to 1.5m
Rail: -0.2m to 1.2m
```

**Recommended Fix:**

Create a unified threshold module:

```python
# ign_lidar/core/modules/classification_thresholds.py

class UnifiedThresholds:
    """Unified thresholds for all classification modules."""

    # Transport thresholds
    ROAD_HEIGHT_MAX = 1.5  # Meters
    ROAD_HEIGHT_MIN = -0.3
    RAIL_HEIGHT_MAX = 1.2
    RAIL_HEIGHT_MIN = -0.2

    # For strict mode (optional)
    ROAD_HEIGHT_MAX_STRICT = 0.5
    RAIL_HEIGHT_MAX_STRICT = 0.8

    # Building thresholds
    BUILDING_HEIGHT_MIN = 2.5
    BUILDING_HEIGHT_MAX = 200.0

    # Vegetation thresholds
    LOW_VEG_HEIGHT_MAX = 2.0
    HIGH_VEG_HEIGHT_MIN = 1.5
```

**Implementation Steps:**

1. Create `classification_thresholds.py`
2. Import in all classification modules
3. Replace hardcoded values
4. Add tests to verify consistency
5. Update configuration files

**Estimated Time:** 2-3 hours

---

### Issue #14: Performance Bottleneck ⚠️ CRITICAL

**Problem:** O(n\*m) complexity in ground truth spatial queries.

**File Affected:**

- `ign_lidar/core/modules/advanced_classification.py`

**Current Code:**

```python
for i, point_geom in enumerate(point_geoms):
    if polygon.contains(point_geom):
        # Classify point
```

**Recommended Fix:**

Use spatial indexing (R-tree):

```python
from shapely.strtree import STRtree

def _classify_with_spatial_index(
    self,
    labels: np.ndarray,
    point_geoms: List[Point],
    gdf: 'gpd.GeoDataFrame',
    asprs_class: int
) -> np.ndarray:
    """Classify using spatial index for O(n log m) performance."""

    # Build R-tree index once
    tree = STRtree(point_geoms)

    for idx, row in gdf.iterrows():
        polygon = row['geometry']
        if not isinstance(polygon, (Polygon, MultiPolygon)):
            continue

        # Query only candidate points (much faster)
        candidate_indices = list(tree.query(polygon))

        # Test only candidates
        for i in candidate_indices:
            if polygon.contains(point_geoms[i]):
                labels[i] = asprs_class

    return labels
```

**Implementation Steps:**

1. Add spatial indexing to `_classify_roads_with_buffer()`
2. Add spatial indexing to `_classify_railways_with_buffer()`
3. Add spatial indexing to general ground truth classification
4. Benchmark performance improvement
5. Document in user guide

**Estimated Time:** 3-4 hours

**Expected Improvement:** 10-50x faster for large datasets

---

## High Priority Issues (Implement Soon)

### Issue #1: Road Height Filter Too Restrictive

**File:** `ign_lidar/core/modules/advanced_classification.py`

**Change:**

```python
# Current
if height[i] > 1.5 or height[i] < -0.3:

# Recommended
if height[i] > 2.0 or height[i] < -0.5:
```

**Estimated Time:** 15 minutes

---

### Issue #4: Railway Height Filter Too Restrictive

**File:** `ign_lidar/core/modules/advanced_classification.py`

**Change:**

```python
# Current
if height[i] > 1.2 or height[i] < -0.2:

# Recommended
if height[i] > 2.0 or height[i] < -0.5:
```

**Estimated Time:** 15 minutes

---

### Issue #5: Building Height Inconsistency

**Files:**

- `ign_lidar/core/modules/building_detection.py`
- `ign_lidar/core/modules/classification_refinement.py`
- `ign_lidar/core/modules/advanced_classification.py`

**Fix:** Use unified threshold (see Issue #8 fix above)

**Estimated Time:** Included in Issue #8 fix

---

### Issue #6: Ground Truth Early Return

**File:** `ign_lidar/core/modules/building_detection.py`

**Current Code:**

```python
if ground_truth_mask is not None and self.config.ground_truth_priority:
    labels[ground_truth_mask] = 6
    stats['ground_truth_building'] = np.sum(ground_truth_mask)
    return labels, stats  # ❌ Early return
```

**Recommended Fix:**

```python
if ground_truth_mask is not None and self.config.ground_truth_priority:
    labels[ground_truth_mask] = 6
    stats['ground_truth_building'] = np.sum(ground_truth_mask)
    # ✅ Continue with geometric detection for non-GT points

# Continue with geometric detection
non_gt_mask = ~ground_truth_mask if ground_truth_mask is not None else np.ones(len(labels), dtype=bool)
# Apply geometric detection only to non-GT points
```

**Estimated Time:** 1 hour

---

### Issue #13: Config vs Code Defaults

**Problem:** Thresholds defined in both YAML and Python classes.

**Recommended Approach:**

Add configuration loader:

```python
# ign_lidar/core/modules/config_loader.py

import yaml
from pathlib import Path
from typing import Optional

class ClassificationConfig:
    """Load and manage classification configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)

    def _load_config(self, path: Optional[Path]) -> dict:
        if path is None:
            # Use default config
            path = Path(__file__).parent.parent.parent / 'configs' / 'classification_config.yaml'

        with open(path) as f:
            return yaml.safe_load(f)

    def get_threshold(self, key: str, default: float) -> float:
        """Get threshold with fallback to default."""
        return self.config.get('classification', {}).get('thresholds', {}).get(key, default)
```

**Estimated Time:** 2-3 hours

---

## Medium Priority Issues (Fix When Possible)

### Issue #2: Planarity Threshold Review

**Action:** Collect real-world data and analyze false positive/negative rates.

**Estimated Time:** 4-6 hours (requires data analysis)

---

### Issue #3: Intensity Filter Assumptions

**Action:** Add multi-material support and make intensity filtering optional.

**Estimated Time:** 2-3 hours

---

### Issue #7: Window Detection Heuristics

**Action:** Add spatial clustering and size constraints for LOD3 windows.

**Estimated Time:** 3-4 hours

---

### Issue #9: Height Threshold Overlap

**Action:** Document the overlap as intentional or fix separation.

**Estimated Time:** 30 minutes (documentation only)

---

### Issue #10: NDVI Building Refinement

**Action:** Add sub-classification for buildings with vegetation.

**Estimated Time:** 1-2 hours

---

## Low Priority Issues (Nice to Have)

### Issue #11: Missing Test Coverage

**Action:** Add comprehensive test suite.

**Tests to Add:**

```python
# tests/test_transport_detection.py
def test_road_with_ground_truth()
def test_rail_with_buffer()
def test_height_boundary_cases()
def test_conflicting_ground_truth()
def test_missing_features()

# tests/test_integration.py
def test_full_pipeline()
def test_large_point_cloud_performance()
def test_edge_cases()
```

**Estimated Time:** 8-10 hours

---

### Issue #12: Documentation Gaps

**Action:** Add missing documentation.

**Documents to Create:**

- `docs/TRANSPORT_CLASSIFICATION_GUIDE.md`
- `docs/BUILDING_DETECTION_GUIDE.md`
- `docs/THRESHOLD_TUNING_GUIDE.md`
- `docs/PERFORMANCE_OPTIMIZATION.md`

**Estimated Time:** 6-8 hours

---

### Issue #15: Generic Exception Handling

**Action:** Replace generic `Exception` with specific exceptions.

**Estimated Time:** 1-2 hours

---

## Implementation Timeline

### Week 1: Critical Issues

- [ ] Day 1-2: Issue #8 - Unify thresholds (2-3h)
- [ ] Day 3-4: Issue #14 - Spatial indexing (3-4h)
- [ ] Day 5: Testing and validation (4h)

**Total Week 1:** ~12 hours

### Week 2: High Priority Issues

- [ ] Day 1: Issues #1, #4 - Height filters (1h)
- [ ] Day 2: Issue #6 - Ground truth logic (1h)
- [ ] Day 3-4: Issue #13 - Config loader (2-3h)
- [ ] Day 5: Testing and validation (3h)

**Total Week 2:** ~8 hours

### Week 3: Medium Priority Issues

- [ ] Issues #2, #3, #7, #9, #10
- [ ] Testing and validation

**Total Week 3:** ~12 hours

### Week 4: Low Priority Issues

- [ ] Issue #11 - Tests
- [ ] Issue #12 - Documentation
- [ ] Issue #15 - Error handling

**Total Week 4:** ~16 hours

**Total Estimated Time:** ~48 hours (6 working days)

---

## Testing Strategy

### Unit Tests

1. Test each threshold change independently
2. Test spatial indexing with various dataset sizes
3. Test configuration loading with valid/invalid configs

### Integration Tests

1. Test full classification pipeline
2. Test with real LiDAR data
3. Benchmark performance before/after changes

### Regression Tests

1. Compare results before/after changes
2. Ensure no degradation in classification quality
3. Document any intentional behavior changes

---

## Success Metrics

### Performance

- [ ] Ground truth classification >10x faster with spatial indexing
- [ ] No regression in processing time for other operations

### Quality

- [ ] No decrease in classification accuracy
- [ ] Reduced false negatives for roads/rails (height filter changes)
- [ ] Consistent behavior across all modules

### Code Quality

- [ ] All thresholds defined in single location
- [ ] > 90% test coverage for critical paths
- [ ] Comprehensive documentation

---

## Risk Assessment

### Low Risk Changes

- Issue #1, #4: Height filter adjustments (easily reversible)
- Issue #5: Building height unification (no logic change)
- Issue #15: Exception handling (code quality only)

### Medium Risk Changes

- Issue #6: Ground truth logic (requires careful testing)
- Issue #8: Threshold unification (affects multiple modules)
- Issue #13: Config system refactor (architectural change)

### High Risk Changes

- Issue #14: Spatial indexing (performance critical, requires validation)

**Mitigation:**

- Comprehensive testing for each change
- Feature flags for new behavior
- Benchmarking before/after
- Rollback plan for each change

---

## Rollback Plan

For each change:

1. Create feature branch
2. Implement change
3. Run full test suite
4. Benchmark performance
5. If tests fail or performance degrades:
   - Revert changes
   - Analyze issue
   - Refine approach

**Git Strategy:**

- One branch per issue
- Pull request with tests
- Code review before merge
- Tag stable versions

---

## Next Steps

1. **Review this plan** with team
2. **Prioritize issues** based on project needs
3. **Assign issues** to developers
4. **Create tracking tickets** (GitHub Issues)
5. **Start with Week 1** critical issues
6. **Monitor progress** and adjust timeline as needed

---

## Questions to Address

1. What is acceptable false positive/negative rate for roads?
2. Should we maintain backward compatibility with existing configs?
3. What is target performance improvement for large datasets?
4. Should we add machine learning for threshold optimization?
5. What is priority: accuracy or performance?

---

**Document Owner:** Development Team  
**Last Updated:** October 16, 2025  
**Next Review:** After Week 1 completion
