# Transport Module Migration Guide

**Version**: 3.1.0 → 3.1.1+  
**Status**: Deprecation Period (until v4.0.0, mid-2026)  
**Breaking Changes**: None (backward compatibility maintained)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Import Path Changes](#import-path-changes)
4. [API Changes](#api-changes)
5. [Configuration Changes](#configuration-changes)
6. [Return Type Changes](#return-type-changes)
7. [Code Examples](#code-examples)
8. [Deprecation Timeline](#deprecation-timeline)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What Changed?

The transport classification subsystem has been refactored from 2 monolithic modules into a modern package structure:

**Old Structure** (v3.0.x):

```
ign_lidar/core/classification/
├── transport_detection.py      # 567 lines
└── transport_enhancement.py    # 731 lines
```

**New Structure** (v3.1.1+):

```
ign_lidar/core/classification/
├── transport/
│   ├── __init__.py          # Public API
│   ├── base.py              # Abstract base classes, enums, configs
│   ├── utils.py             # Shared utilities
│   ├── detection.py         # Detection implementation
│   └── enhancement.py       # Enhancement implementation
├── transport_detection.py   # Deprecated wrapper (remove in v4.0)
└── transport_enhancement.py # Deprecated wrapper (remove in v4.0)
```

### Why Migrate?

1. **Cleaner Code**: 19.2% code reduction through deduplication
2. **Better Architecture**: Abstract base classes, shared utilities
3. **Type Safety**: Dataclass return types instead of tuples
4. **Future-Proof**: New features will only be added to new structure
5. **Deprecation Warning**: Old paths emit warnings and will be removed in v4.0

### Do I Need to Migrate Now?

**No, but recommended**. Old import paths still work in v3.1.x via backward compatibility wrappers, but:

- ⚠️ They emit `DeprecationWarning` messages
- ⚠️ They will be removed in v4.0.0 (mid-2026)
- ✅ Migrating now is quick and easy

---

## Quick Start

### Minimal Migration (5 minutes)

**Step 1**: Update your imports

```python
# OLD (deprecated)
from ign_lidar.core.classification.transport_detection import TransportDetector

# NEW (recommended)
from ign_lidar.core.classification.transport import TransportDetector
```

**Step 2**: Run your code

- If you see deprecation warnings, update those imports
- If no warnings, you're done! ✅

**Step 3**: Test

```bash
pytest tests -v  # Ensure all tests pass
```

---

## Import Path Changes

### Detection Module

| Old Import                                                     | New Import                                           | Status         |
| -------------------------------------------------------------- | ---------------------------------------------------- | -------------- |
| `from .transport_detection import TransportDetector`           | `from .transport import TransportDetector`           | ✅ Recommended |
| `from .transport_detection import TransportDetectionMode`      | `from .transport import TransportMode`               | ⚠️ **Renamed** |
| `from .transport_detection import TransportDetectionConfig`    | `from .transport import DetectionConfig`             | ⚠️ **Renamed** |
| `from .transport_detection import detect_transport_multi_mode` | `from .transport import detect_transport_multi_mode` | ✅ Recommended |

### Enhancement Module

| Old Import                                                      | New Import                                          | Status         |
| --------------------------------------------------------------- | --------------------------------------------------- | -------------- |
| `from .transport_enhancement import AdaptiveTransportBuffer`    | `from .transport import AdaptiveTransportBuffer`    | ✅ Recommended |
| `from .transport_enhancement import SpatialTransportClassifier` | `from .transport import SpatialTransportClassifier` | ✅ Recommended |
| `from .transport_enhancement import AdaptiveBufferConfig`       | `from .transport import BufferingConfig`            | ⚠️ **Renamed** |
| `from .transport_enhancement import SpatialIndexConfig`         | `from .transport import IndexingConfig`             | ⚠️ **Renamed** |
| `from .transport_enhancement import QualityMetricsConfig`       | `from .transport import QualityMetricsConfig`       | ✅ Recommended |

### Utility Functions

All utility functions moved to `transport.utils` but are re-exported from `transport.__init__`:

| Old Location                                    | New Location      | Import                                             |
| ----------------------------------------------- | ----------------- | -------------------------------------------------- |
| `transport_detection.validate_transport_height` | `transport.utils` | `from .transport import validate_transport_height` |
| `transport_enhancement.calculate_curvature`     | `transport.utils` | `from .transport import calculate_curvature`       |
| `transport_enhancement.compute_adaptive_width`  | `transport.utils` | `from .transport import compute_adaptive_width`    |

---

## API Changes

### Renamed Classes and Enums

#### 1. `TransportDetectionMode` → `TransportMode`

**Reason**: More concise, consistent with naming conventions

```python
# OLD
from ign_lidar.core.classification.transport_detection import TransportDetectionMode
mode = TransportDetectionMode.ASPRS_STANDARD

# NEW
from ign_lidar.core.classification.transport import TransportMode
mode = TransportMode.ASPRS_STANDARD
```

**Enum Values** (unchanged):

- `ASPRS_STANDARD` - Basic detection
- `ASPRS_EXTENDED` - Advanced detection
- `LOD2` - Ultra-precise detection

#### 2. `TransportDetectionConfig` → `DetectionConfig`

**Reason**: Simpler name, already namespaced under `transport` module

```python
# OLD
from ign_lidar.core.classification.transport_detection import TransportDetectionConfig
config = TransportDetectionConfig(mode=TransportDetectionMode.ASPRS_STANDARD)

# NEW
from ign_lidar.core.classification.transport import DetectionConfig, TransportMode
config = DetectionConfig(mode=TransportMode.ASPRS_STANDARD)
```

#### 3. `AdaptiveBufferConfig` → `BufferingConfig`

**Reason**: More descriptive, matches functionality

```python
# OLD
from ign_lidar.core.classification.transport_enhancement import AdaptiveBufferConfig
config = AdaptiveBufferConfig(min_width=2.0, max_width=5.0)

# NEW
from ign_lidar.core.classification.transport import BufferingConfig
config = BufferingConfig(min_width=2.0, max_width=5.0)
```

#### 4. `SpatialIndexConfig` → `IndexingConfig`

**Reason**: Simpler, already namespaced

```python
# OLD
from ign_lidar.core.classification.transport_enhancement import SpatialIndexConfig
config = SpatialIndexConfig(use_rtree=True)

# NEW
from ign_lidar.core.classification.transport import IndexingConfig
config = IndexingConfig(use_rtree=True)
```

### New Abstract Base Classes

The new structure introduces 3 abstract base classes (you don't need to use these unless extending the module):

```python
from ign_lidar.core.classification.transport.base import (
    TransportDetectorBase,      # For custom detectors
    TransportBufferBase,         # For custom buffering strategies
    TransportClassifierBase      # For custom classifiers
)
```

---

## Configuration Changes

### Auto-Configuration by Mode

The new `DetectionConfig` class auto-configures thresholds based on the selected mode:

```python
from ign_lidar.core.classification.transport import DetectionConfig, TransportMode

# Auto-configuration (recommended)
config = DetectionConfig(mode=TransportMode.ASPRS_STANDARD)
# Automatically sets: max_height=3.0, min_planarity=0.7, ...

config = DetectionConfig(mode=TransportMode.ASPRS_EXTENDED)
# Automatically sets: max_height=2.5, min_planarity=0.8, ...

config = DetectionConfig(mode=TransportMode.LOD2)
# Automatically sets: max_height=2.0, min_planarity=0.85, ...
```

### Manual Configuration (still supported)

```python
# Override auto-configured values
config = DetectionConfig(
    mode=TransportMode.ASPRS_STANDARD,
    max_height=4.0,           # Override default 3.0
    min_planarity=0.6         # Override default 0.7
)
```

### Unified Threshold Access

All threshold-related configurations inherit from `TransportConfigBase`:

```python
from ign_lidar.core.classification.transport import DetectionConfig

config = DetectionConfig(mode=TransportMode.ASPRS_STANDARD)

# Access thresholds
print(config.thresholds.transport_max_height)  # 3.0
print(config.thresholds.transport_min_planarity)  # 0.7
```

---

## Return Type Changes

### Detection Results

**OLD**: Tuple return type

```python
from ign_lidar.core.classification.transport_detection import TransportDetector

detector = TransportDetector(mode=TransportDetectionMode.ASPRS_STANDARD)
road_mask, railway_mask, stats_dict = detector.detect(points, geometries)
```

**NEW**: Type-safe dataclass

```python
from ign_lidar.core.classification.transport import TransportDetector, TransportMode

detector = TransportDetector(mode=TransportMode.ASPRS_STANDARD)
result = detector.detect(points, geometries)

# Access results
road_mask = result.road_mask
railway_mask = result.railway_mask
stats = result.stats  # TransportStats dataclass

# Stats fields
print(stats.total_points)
print(stats.road_points)
print(stats.railway_points)
print(stats.detection_mode)
```

---

## Code Examples

### Example 1: Basic Detection (Before & After)

**BEFORE (v3.0.x)**:

```python
from ign_lidar.core.classification.transport_detection import (
    TransportDetector,
    TransportDetectionMode
)

# Configure
mode = TransportDetectionMode.ASPRS_STANDARD
detector = TransportDetector(mode=mode)

# Detect
road_mask, railway_mask, stats = detector.detect(points, road_geometries, railway_geometries)

# Use results
print(f"Road points: {stats['road_points']}")
print(f"Railway points: {stats['railway_points']}")
```

**AFTER (v3.1.1+)**:

```python
from ign_lidar.core.classification.transport import (
    TransportDetector,
    TransportMode
)

# Configure
mode = TransportMode.ASPRS_STANDARD
detector = TransportDetector(mode=mode)

# Detect
result = detector.detect(points, road_geometries, railway_geometries)

# Use results (type-safe)
print(f"Road points: {result.stats.road_points}")
print(f"Railway points: {result.stats.railway_points}")
```

### Example 2: Adaptive Buffering (Before & After)

**BEFORE (v3.0.x)**:

```python
from ign_lidar.core.classification.transport_enhancement import (
    AdaptiveTransportBuffer,
    AdaptiveBufferConfig
)

# Configure
config = AdaptiveBufferConfig(
    min_width=2.0,
    max_width=5.0,
    curvature_threshold=0.1
)
buffer = AdaptiveTransportBuffer(config)

# Apply buffering
buffered_geoms = buffer.buffer_geometries(geometries)
```

**AFTER (v3.1.1+)**:

```python
from ign_lidar.core.classification.transport import (
    AdaptiveTransportBuffer,
    BufferingConfig
)

# Configure
config = BufferingConfig(
    min_width=2.0,
    max_width=5.0,
    curvature_threshold=0.1
)
buffer = AdaptiveTransportBuffer(config)

# Apply buffering (same API)
buffered_geoms = buffer.buffer_geometries(geometries)
```

### Example 3: Spatial Classification (Before & After)

**BEFORE (v3.0.x)**:

```python
from ign_lidar.core.classification.transport_enhancement import (
    SpatialTransportClassifier,
    SpatialIndexConfig
)

# Configure
config = SpatialIndexConfig(use_rtree=True, leaf_size=10)
classifier = SpatialTransportClassifier(config)

# Classify
road_mask = classifier.classify_points(points, road_geometries, buffer_distance=3.0)
```

**AFTER (v3.1.1+)**:

```python
from ign_lidar.core.classification.transport import (
    SpatialTransportClassifier,
    IndexingConfig
)

# Configure
config = IndexingConfig(use_rtree=True, leaf_size=10)
classifier = SpatialTransportClassifier(config)

# Classify (same API)
road_mask = classifier.classify_points(points, road_geometries, buffer_distance=3.0)
```

### Example 4: Multi-Mode Detection (Before & After)

**BEFORE (v3.0.x)**:

```python
from ign_lidar.core.classification.transport_detection import detect_transport_multi_mode

results = detect_transport_multi_mode(
    points=points,
    road_geometries=roads,
    railway_geometries=railways,
    modes=['ASPRS_STANDARD', 'ASPRS_EXTENDED']
)

for mode_name, (road_mask, railway_mask, stats) in results.items():
    print(f"{mode_name}: {stats['road_points']} road points")
```

**AFTER (v3.1.1+)**:

```python
from ign_lidar.core.classification.transport import detect_transport_multi_mode

results = detect_transport_multi_mode(
    points=points,
    road_geometries=roads,
    railway_geometries=railways,
    modes=['ASPRS_STANDARD', 'ASPRS_EXTENDED']
)

for mode_name, result in results.items():
    print(f"{mode_name}: {result.stats.road_points} road points")
```

---

## Deprecation Timeline

### v3.1.0 (January 2025) - **Current Release**

- ✅ New `transport/` package structure available
- ✅ Old import paths still work (backward compatibility)
- ⚠️ Old imports emit `DeprecationWarning`
- ✅ All tests passing with both old and new paths

### v3.1.x - v3.9.x (2025-2026) - **Grace Period**

- ✅ Both old and new paths supported
- ⚠️ Deprecation warnings continue
- 📚 Documentation recommends new paths
- 🔄 Gradual migration encouraged

### v4.0.0 (mid-2026) - **Breaking Change**

- ❌ Old import paths removed (`transport_detection.py`, `transport_enhancement.py`)
- ✅ Only new `transport/` package available
- 💥 Code using old paths will break
- 📖 Migration guide updated with v4.0 specifics

---

## Troubleshooting

### Issue 1: DeprecationWarning Messages

**Problem**:

```
DeprecationWarning: transport_detection module is deprecated and will be removed in v4.0.0.
Use 'from ign_lidar.core.classification.transport import TransportDetector' instead.
```

**Solution**:

Update your import:

```python
# OLD
from ign_lidar.core.classification.transport_detection import TransportDetector

# NEW
from ign_lidar.core.classification.transport import TransportDetector
```

### Issue 2: AttributeError - 'TransportDetectionMode' not found

**Problem**:

```python
from ign_lidar.core.classification.transport import TransportDetectionMode
# AttributeError: module has no attribute 'TransportDetectionMode'
```

**Solution**:

Class was renamed to `TransportMode`:

```python
from ign_lidar.core.classification.transport import TransportMode
```

### Issue 3: Tuple Unpacking with New Result Type

**Problem**:

```python
# Old code expecting tuple
road_mask, railway_mask, stats = detector.detect(points, roads, railways)
# ValueError: not enough values to unpack
```

**Solution**:

Use dataclass result:

```python
result = detector.detect(points, roads, railways)
road_mask = result.road_mask
railway_mask = result.railway_mask
stats = result.stats
```

### Issue 4: Missing 'AdaptiveBufferConfig'

**Problem**:

```python
from ign_lidar.core.classification.transport import AdaptiveBufferConfig
# AttributeError
```

**Solution**:

Class was renamed to `BufferingConfig`:

```python
from ign_lidar.core.classification.transport import BufferingConfig
```

### Issue 5: Tests Failing After Migration

**Problem**: Tests fail after updating imports

**Solution**:

1. Check for hardcoded module paths in tests
2. Update test imports to new paths
3. Update assertions to use dataclass fields instead of tuple indices
4. Run pytest with verbose output: `pytest tests -v`

---

## FAQ

### Q: Do I need to migrate immediately?

**A**: No, old paths work until v4.0.0 (mid-2026). But we recommend migrating soon to avoid warnings.

### Q: Will my old code break?

**A**: No, backward compatibility is maintained in v3.1.x. Old code will emit warnings but still work.

### Q: What if I find a bug?

**A**: Report issues on GitHub. Both old and new paths are supported during the transition.

### Q: Can I use both old and new paths in the same project?

**A**: Yes, but not recommended. Mixing paths causes confusion and duplicate warnings.

### Q: How do I silence deprecation warnings?

**A**: Not recommended, but you can filter them:

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

Better solution: Migrate your code!

### Q: Are there performance differences?

**A**: No, the underlying implementation is identical. The old wrappers just redirect to new paths.

---

## Additional Resources

- **Phase 3 Completion Summary**: `docs/PHASE_3_COMPLETION_SUMMARY.md`
- **Phase 3A Analysis**: `docs/PHASE_3A_TRANSPORT_ANALYSIS.md`
- **API Documentation**: `docs/api/transport_module.md`
- **GitHub Issues**: Report migration problems

---

## Migration Checklist

Use this checklist to track your migration progress:

### Import Updates

- [ ] Replace `transport_detection` imports with `transport` imports
- [ ] Replace `transport_enhancement` imports with `transport` imports
- [ ] Update `TransportDetectionMode` → `TransportMode`
- [ ] Update `TransportDetectionConfig` → `DetectionConfig`
- [ ] Update `AdaptiveBufferConfig` → `BufferingConfig`
- [ ] Update `SpatialIndexConfig` → `IndexingConfig`

### Code Updates

- [ ] Update tuple unpacking to use dataclass results
- [ ] Update dictionary-style stats access to dataclass fields
- [ ] Test all transport-related functionality
- [ ] Run full test suite: `pytest tests -v`
- [ ] Fix any failing tests

### Documentation

- [ ] Update code examples in documentation
- [ ] Update API references
- [ ] Update configuration examples
- [ ] Update inline comments mentioning old paths

### Verification

- [ ] No deprecation warnings in console output
- [ ] All tests passing
- [ ] Code review completed
- [ ] Documentation updated

---

**Need Help?** Check the [troubleshooting section](#troubleshooting) or open a GitHub issue.

**Migration Status**: ✅ Ready for production use in v3.1.1+
