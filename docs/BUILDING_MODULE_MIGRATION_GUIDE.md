# Building Module Migration Guide

**Version:** v3.1.0 → v4.0.0  
**Last Updated:** October 22, 2025  
**Deprecation Timeline:** Old imports will be removed in v4.0.0 (mid-2026)

---

## 📋 Overview

In v3.1.0, the building classification modules were restructured into an organized `building/` subdirectory. This guide helps you migrate from the old import paths to the new structure.

**Good news:** Your existing code will continue to work! The old import paths are maintained via backward compatibility wrappers that emit deprecation warnings.

---

## 🚀 Quick Migration

### Before (Deprecated)

```python
from ign_lidar.core.classification.adaptive_building_classifier import AdaptiveBuildingClassifier
from ign_lidar.core.classification.building_detection import BuildingDetector
from ign_lidar.core.classification.building_clustering import BuildingClusterer
from ign_lidar.core.classification.building_fusion import BuildingFusion
```

### After (Recommended)

```python
from ign_lidar.core.classification.building import (
    AdaptiveBuildingClassifier,
    BuildingDetector,
    BuildingClusterer,
    BuildingFusion
)
```

**That's it!** The API remains unchanged - only the import path changed.

---

## 📖 Detailed Migration

### Module Mapping

| Old Module                     | New Module                          | Status        |
| ------------------------------ | ----------------------------------- | ------------- |
| `adaptive_building_classifier` | `building.adaptive` or `building`   | ⚠️ Deprecated |
| `building_detection`           | `building.detection` or `building`  | ⚠️ Deprecated |
| `building_clustering`          | `building.clustering` or `building` | ⚠️ Deprecated |
| `building_fusion`              | `building.fusion` or `building`     | ⚠️ Deprecated |

### Class & Function Mapping

All classes and functions have the same names and APIs. Only the import path changed.

#### AdaptiveBuildingClassifier

```python
# Old (deprecated)
from ign_lidar.core.classification.adaptive_building_classifier import (
    AdaptiveBuildingClassifier,
    BuildingFeatureSignature,
    PointBuildingScore
)

# New (recommended) - Import from building module
from ign_lidar.core.classification.building import (
    AdaptiveBuildingClassifier,
    BuildingFeatureSignature,
    PointBuildingScore
)

# Alternative - Import from specific submodule
from ign_lidar.core.classification.building.adaptive import (
    AdaptiveBuildingClassifier,
    BuildingFeatureSignature,
    PointBuildingScore
)
```

#### BuildingDetector

```python
# Old (deprecated)
from ign_lidar.core.classification.building_detection import (
    BuildingDetector,
    BuildingDetectionConfig,
    BuildingDetectionMode
)

# New (recommended)
from ign_lidar.core.classification.building import (
    BuildingDetector,
    BuildingDetectionConfig,
    BuildingDetectionMode
)

# Alternative
from ign_lidar.core.classification.building.detection import (
    BuildingDetector,
    BuildingDetectionConfig,
    BuildingDetectionMode
)
```

#### BuildingClusterer

```python
# Old (deprecated)
from ign_lidar.core.classification.building_clustering import (
    BuildingClusterer,
    BuildingCluster,
    cluster_buildings_multi_source
)

# New (recommended)
from ign_lidar.core.classification.building import (
    BuildingClusterer,
    BuildingCluster,
    cluster_buildings_multi_source
)

# Alternative
from ign_lidar.core.classification.building.clustering import (
    BuildingClusterer,
    BuildingCluster,
    cluster_buildings_multi_source
)
```

#### BuildingFusion

```python
# Old (deprecated)
from ign_lidar.core.classification.building_fusion import (
    BuildingFusion,
    BuildingSource,
    PolygonQuality
)

# New (recommended)
from ign_lidar.core.classification.building import (
    BuildingFusion,
    BuildingSource,
    PolygonQuality
)

# Alternative
from ign_lidar.core.classification.building.fusion import (
    BuildingFusion,
    BuildingSource,
    PolygonQuality
)
```

---

## 🆕 New Imports Available

The restructuring also introduced new base classes and utilities:

### Base Classes

```python
from ign_lidar.core.classification.building import (
    BuildingClassifierBase,      # Abstract base class for classifiers
    BuildingDetectorBase,         # Abstract base class for detectors
    BuildingClustererBase,        # Abstract base class for clusterers
    BuildingFusionBase,           # Abstract base class for fusion
)
```

### Enumerations

```python
from ign_lidar.core.classification.building import (
    BuildingMode,                 # ASPRS, LOD2, LOD3
    BuildingSource,               # BD_TOPO, CADASTRE, OSM, FUSED
    ClassificationConfidence,     # CERTAIN, HIGH, MEDIUM, LOW, UNCERTAIN
)
```

### Configuration & Results

```python
from ign_lidar.core.classification.building import (
    BuildingConfigBase,           # Base configuration class
    BuildingClassificationResult, # Standardized result format
)
```

### Utility Functions

```python
from ign_lidar.core.classification.building import utils

# Or import specific utilities
from ign_lidar.core.classification.building.utils import (
    points_in_polygon,
    filter_by_height,
    compute_verticality,
    compute_planarity,
    # ... and 17 more utility functions
)
```

---

## ⚙️ Migration Steps

### Step 1: Find Old Imports

Search your codebase for old import patterns:

```bash
# Using grep
grep -r "from ign_lidar.core.classification.adaptive_building_classifier" .
grep -r "from ign_lidar.core.classification.building_detection" .
grep -r "from ign_lidar.core.classification.building_clustering" .
grep -r "from ign_lidar.core.classification.building_fusion" .

# Or using git grep (faster)
git grep "adaptive_building_classifier import"
git grep "building_detection import"
git grep "building_clustering import"
git grep "building_fusion import"
```

### Step 2: Update Imports

Replace old imports with new ones:

```python
# Before
from ign_lidar.core.classification.adaptive_building_classifier import AdaptiveBuildingClassifier

# After
from ign_lidar.core.classification.building import AdaptiveBuildingClassifier
```

### Step 3: Test Your Code

Run your tests to ensure everything still works:

```bash
pytest tests/ -v
```

### Step 4: Verify No Deprecation Warnings

Run your code and check for deprecation warnings:

```bash
python -W default your_script.py
```

You should no longer see:

```
DeprecationWarning: Module 'ign_lidar.core.classification.adaptive_building_classifier' is deprecated...
```

---

## 🔍 Finding Deprecated Imports in Your Code

### Using Python

```python
import warnings

# Show all deprecation warnings
warnings.filterwarnings('default', category=DeprecationWarning)

# Your imports here - warnings will be shown
from ign_lidar.core.classification.adaptive_building_classifier import AdaptiveBuildingClassifier
```

### Using pytest

```bash
# Show all warnings during test execution
pytest tests/ -v -W default

# Or show only deprecation warnings
pytest tests/ -v -W default::DeprecationWarning
```

---

## 📝 Migration Examples

### Example 1: Simple Script

**Before:**

```python
#!/usr/bin/env python
"""Building classification script"""
import numpy as np
from ign_lidar.core.classification.adaptive_building_classifier import AdaptiveBuildingClassifier
from ign_lidar.core.classification.building_detection import BuildingDetector

# Load data
points = np.load("points.npy")

# Create classifier
classifier = AdaptiveBuildingClassifier()
detector = BuildingDetector()

# Use them...
```

**After:**

```python
#!/usr/bin/env python
"""Building classification script"""
import numpy as np
from ign_lidar.core.classification.building import (
    AdaptiveBuildingClassifier,
    BuildingDetector
)

# Load data
points = np.load("points.npy")

# Create classifier
classifier = AdaptiveBuildingClassifier()
detector = BuildingDetector()

# Use them...
```

### Example 2: Complex Pipeline

**Before:**

```python
from ign_lidar.core.classification.building_detection import (
    BuildingDetector,
    BuildingDetectionConfig,
    BuildingDetectionMode
)
from ign_lidar.core.classification.building_clustering import (
    BuildingClusterer,
    cluster_buildings_multi_source
)
from ign_lidar.core.classification.building_fusion import (
    BuildingFusion,
    BuildingSource
)

# Configure
config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
detector = BuildingDetector(config=config)
clusterer = BuildingClusterer()
fusion = BuildingFusion()

# Pipeline
clusters = cluster_buildings_multi_source(points, labels, buildings_gdf)
```

**After:**

```python
from ign_lidar.core.classification.building import (
    BuildingDetector,
    BuildingDetectionConfig,
    BuildingDetectionMode,
    BuildingClusterer,
    cluster_buildings_multi_source,
    BuildingFusion,
    BuildingSource
)

# Configure
config = BuildingDetectionConfig(mode=BuildingDetectionMode.LOD2)
detector = BuildingDetector(config=config)
clusterer = BuildingClusterer()
fusion = BuildingFusion()

# Pipeline
clusters = cluster_buildings_multi_source(points, labels, buildings_gdf)
```

### Example 3: Using New Base Classes

If you're creating custom building classifiers, you can now extend the base classes:

```python
from ign_lidar.core.classification.building import (
    BuildingClassifierBase,
    BuildingMode,
    ClassificationConfidence,
    BuildingClassificationResult
)
import numpy as np

class CustomBuildingClassifier(BuildingClassifierBase):
    """Custom building classifier extending the base class"""

    def __init__(self, mode: BuildingMode = BuildingMode.ASPRS):
        super().__init__(mode=mode)

    def classify(self, points: np.ndarray, features: dict) -> BuildingClassificationResult:
        """Implement custom classification logic"""
        # Your implementation here
        labels = np.zeros(len(points), dtype=np.uint8)
        confidences = np.ones(len(points), dtype=np.float32)

        return BuildingClassificationResult(
            labels=labels,
            confidences=confidences,
            confidence_level=ClassificationConfidence.HIGH,
            metadata={'method': 'custom'}
        )
```

---

## ⏰ Deprecation Timeline

| Version           | Status            | Action Required                      |
| ----------------- | ----------------- | ------------------------------------ |
| v3.0.x            | Old imports only  | No changes needed                    |
| v3.1.0            | Both imports work | Deprecation warnings for old imports |
| v3.2.x - v3.9.x   | Both imports work | Recommended to migrate               |
| v4.0.0 (mid-2026) | Only new imports  | **Old imports removed**              |

**Recommendation:** Migrate before v4.0.0 to avoid breaking changes.

---

## 🐛 Troubleshooting

### Issue: "No module named 'ign_lidar.core.classification.building'"

**Cause:** Using v3.0.x or earlier.

**Solution:** Upgrade to v3.1.0+:

```bash
pip install --upgrade ign-lidar-hd
```

### Issue: Still seeing deprecation warnings after migration

**Cause:** Some imports not updated, or transitive dependencies using old imports.

**Solution:**

1. Search for any remaining old imports:
   ```bash
   git grep "adaptive_building_classifier import"
   ```
2. Check if any imported libraries use old import paths (they'll get warnings too)

### Issue: Import error after migration

**Cause:** Typo in new import path.

**Solution:** Ensure you're using the correct path:

```python
# Correct
from ign_lidar.core.classification.building import AdaptiveBuildingClassifier

# Incorrect (note: no 's' at the end)
from ign_lidar.core.classification.buildings import AdaptiveBuildingClassifier
```

---

## 📚 Additional Resources

- [Phase 2 Completion Summary](PHASE_2_COMPLETION_SUMMARY.md) - Detailed information about the restructuring
- [Classification Consolidation Plan](CLASSIFICATION_CONSOLIDATION_PLAN.md) - Overall consolidation strategy
- [API Documentation](../docs/) - Full API reference
- [CHANGELOG](../CHANGELOG.md) - Version history

---

## ❓ FAQ

**Q: Do I need to change my code immediately?**  
A: No, old imports still work. However, we recommend migrating before v4.0.0 (mid-2026).

**Q: Will the API change?**  
A: No, only import paths changed. All class names, methods, and parameters remain the same.

**Q: What about type hints and IDE autocomplete?**  
A: Type hints work with both old and new imports. IDEs will show deprecation warnings for old imports.

**Q: Can I mix old and new imports?**  
A: Yes, but it's not recommended. Stick to one style for consistency.

**Q: Will this affect my production code?**  
A: Old imports work fine in production. The only difference is deprecation warnings in logs.

**Q: How do I suppress deprecation warnings temporarily?**  
A: Use Python's warnings filter:

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning,
                       module='ign_lidar.core.classification')
```

**Q: Where can I get help?**  
A: Open an issue on GitHub or check the documentation.

---

## ✅ Migration Checklist

Use this checklist to track your migration:

- [ ] Search codebase for old imports
- [ ] Update all `adaptive_building_classifier` imports
- [ ] Update all `building_detection` imports
- [ ] Update all `building_clustering` imports
- [ ] Update all `building_fusion` imports
- [ ] Run test suite to verify everything works
- [ ] Check for deprecation warnings
- [ ] Update documentation/examples
- [ ] Commit changes
- [ ] Deploy to testing environment
- [ ] Verify in production

---

**Need help?** Open an issue on GitHub with the `migration` label.
