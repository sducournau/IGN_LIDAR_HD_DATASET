# Classification Module - Developer Style Guide

**Date:** October 23, 2025  
**Version:** 1.0.0  
**Module:** `ign_lidar/core/classification`  
**Status:** Official Style Guide

---

## 📚 Purpose

This style guide establishes coding conventions and best practices for the classification module. Following these guidelines ensures:

- ✅ **Consistency** across the codebase
- ✅ **Maintainability** for future developers
- ✅ **Quality** through established patterns
- ✅ **Compatibility** with existing code

---

## 🔧 Import Conventions

### Preferred: Absolute Imports from Package Root

Use absolute imports for clarity and IDE support:

```python
# ✅ RECOMMENDED - Clear and unambiguous
from ign_lidar.classification_schema import ASPRSClass, LOD2Class, LOD3Class
from ign_lidar.core.classification.thresholds import get_thresholds, ThresholdConfig
from ign_lidar.core.classification.building import AdaptiveBuildingClassifier
from ign_lidar.core.classification.transport import TransportDetector
```

### Alternative: Relative Imports Within Module

Acceptable for internal module imports:

```python
# ✅ ACCEPTABLE - Within classification module files
from ...classification_schema import ASPRSClass
from .thresholds import get_thresholds
from .building import AdaptiveBuildingClassifier
from .rules.validation import validate_features
```

### Avoid: Mixed Import Styles

Don't mix absolute and relative imports in the same file:

```python
# ❌ INCONSISTENT - Don't mix styles
from ign_lidar.classification_schema import ASPRSClass
from .thresholds import get_thresholds  # Mixing styles
from ign_lidar.core.classification.building import AdaptiveBuildingClassifier
from .transport import TransportDetector  # Inconsistent
```

### Import Organization

Organize imports in the following order:

```python
# 1. Standard library imports
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 2. Third-party imports
import numpy as np
import pandas as pd

# 3. Local package imports (absolute)
from ign_lidar.classification_schema import ASPRSClass, LOD2Class

# 4. Local module imports (relative)
from .thresholds import get_thresholds
from .building import BuildingDetector
```

---

## 🏗️ Configuration Patterns

### Use @dataclass for Configuration Classes

Configuration classes should use `@dataclass` for clarity:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MyModuleConfig:
    """
    Configuration for MyModule.

    Args:
        threshold: Detection threshold (0.0-1.0)
        mode: Operation mode ('standard' or 'strict')
        max_iterations: Maximum iterations for algorithm
        enable_feature: Enable optional feature
    """
    threshold: float = 0.5
    mode: str = 'standard'
    max_iterations: int = 100
    enable_feature: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

        if self.mode not in ['standard', 'strict']:
            raise ValueError(f"mode must be 'standard' or 'strict', got {self.mode}")

        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations}")
```

### Configuration Naming Convention

- Class name: `*Config` suffix (e.g., `ThresholdConfig`, `DetectionConfig`)
- Clear parameter names with type hints
- Provide sensible defaults
- Document valid ranges/values
- Validate in `__post_init__()`

---

## 📝 Naming Conventions

### Classes

Use `PascalCase` for class names:

```python
# ✅ CORRECT
class BuildingDetector:
    pass

class AdaptiveBuildingClassifier:
    pass

class TransportDetectionResult:
    pass
```

### Functions and Methods

Use `snake_case` for functions and methods:

```python
# ✅ CORRECT
def detect_buildings(points, features):
    pass

def compute_confidence_scores(labels, features):
    pass

def _internal_helper_function():  # Private function
    pass
```

### Constants

Use `UPPER_SNAKE_CASE` for module-level constants:

```python
# ✅ CORRECT
MAX_HEIGHT = 50.0
DEFAULT_THRESHOLD = 0.7
ASPRS_BUILDING_CODE = 6
MIN_POINTS_PER_CLUSTER = 10
```

### Private Members

Use leading underscore for private/internal members:

```python
# ✅ CORRECT
class Classifier:
    def __init__(self):
        self._internal_state = {}  # Private attribute

    def _compute_intermediate_result(self):  # Private method
        pass

    def classify(self):  # Public method
        result = self._compute_intermediate_result()
        return result
```

### Special Suffixes

Use consistent suffixes for specific types:

```python
# Configuration classes
ThresholdConfig
DetectionConfig
ClassificationConfig

# Result classes
ClassificationResult
DetectionResult
ValidationResult

# Enum classes
BuildingMode
TransportType
RulePriority
DetectionStrategy
```

---

## 🚨 Error Handling

### Use Specific Exceptions

Create and use specific exception classes:

```python
# ✅ CORRECT - Specific exceptions
class ClassificationError(Exception):
    """Base exception for classification errors."""
    pass

class FeatureValidationError(ClassificationError):
    """Raised when feature validation fails."""
    pass

class LiDARLoadError(ClassificationError):
    """Raised when LiDAR data loading fails."""
    pass

# Usage
def validate_features(features):
    if 'height' not in features:
        raise FeatureValidationError("Required feature 'height' is missing")

    if np.any(np.isnan(features['height'])):
        raise FeatureValidationError("Feature 'height' contains NaN values")
```

```python
# ❌ INCORRECT - Generic exceptions
def validate_features(features):
    if 'height' not in features:
        raise Exception("Missing height")  # Too generic

    if np.any(np.isnan(features['height'])):
        raise ValueError("NaN in height")  # Not specific to domain
```

### Graceful Degradation for Optional Dependencies

Handle optional dependencies gracefully:

```python
# ✅ CORRECT - Graceful degradation
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    DBSCAN = None

def cluster_points(points):
    if not HAS_SKLEARN:
        logger.warning("sklearn not available - using fallback clustering")
        return _fallback_clustering(points)

    clustering = DBSCAN(eps=2.0, min_samples=10)
    return clustering.fit(points)
```

### Error Context and Logging

Provide helpful error messages with context:

```python
# ✅ CORRECT - Informative error messages
def load_point_cloud(file_path):
    if not os.path.exists(file_path):
        raise LiDARLoadError(
            f"Point cloud file not found: {file_path}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Please check the file path and try again."
        )

    try:
        data = np.load(file_path)
    except Exception as e:
        raise LiDARLoadError(
            f"Failed to load point cloud from {file_path}: {e}\n"
            f"File size: {os.path.getsize(file_path)} bytes\n"
            f"Expected format: .npy or .npz"
        ) from e
```

---

## 📖 Documentation Standards

### Module Docstrings

Every module should have a comprehensive docstring:

```python
"""
Building Detection Module

This module provides building detection and classification functionality for
LiDAR point clouds using multiple strategies:

- Ground truth-based detection (BD TOPO, cadastre)
- Geometric feature-based detection (height, planarity, verticality)
- Hybrid detection combining ground truth and geometry

Key Components:
    - BuildingDetector: Main detection class
    - BuildingMode: Enumeration of detection modes
    - DetectionResult: Detection result dataclass

Usage:
    from ign_lidar.core.classification.building import BuildingDetector

    detector = BuildingDetector(mode='lod2')
    result = detector.detect(points, features)
    building_mask = result.mask

Author: IGN LiDAR HD Classification Team
Date: October 2025
Version: 3.2.0
"""
```

### Function/Method Docstrings

Use comprehensive docstrings with Google/NumPy style:

```python
def classify_points(
    points: np.ndarray,
    features: Dict[str, np.ndarray],
    mode: str = 'asprs_standard',
    ground_truth_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Classify point cloud using specified mode and features.

    This function performs multi-strategy classification combining geometric
    features, spectral information, and optional ground truth data.

    Args:
        points: Point coordinates [N, 3] where N is number of points.
               Columns are [x, y, z] in meters.
        features: Dictionary of feature arrays. Required keys:
                 - 'height': Height above ground [N]
                 - 'planarity': Planarity values [N] in [0, 1]
                 Optional keys:
                 - 'ndvi': NDVI values [N] in [-1, 1]
                 - 'intensity': LiDAR intensity [N]
        mode: Classification mode. Options:
             - 'asprs_standard': ASPRS LAS 1.4 standard (22 classes)
             - 'asprs_extended': Extended ASPRS (100+ classes)
             - 'lod2': LOD2 building-focused (15 classes)
             - 'lod3': LOD3 detailed architecture (30 classes)
        ground_truth_mask: Optional boolean mask [N] for ground truth buildings.
                          If provided, these points will be classified with
                          high confidence.

    Returns:
        Classification labels [N] as integer array. Label values depend on mode:
        - asprs_standard: 0-31 (ASPRS codes)
        - asprs_extended: 0-255 (extended ASPRS)
        - lod2: 0-14 (LOD2 classes)
        - lod3: 0-29 (LOD3 classes)

    Raises:
        FeatureValidationError: If required features are missing or invalid
        ValueError: If mode is not recognized

    Example:
        >>> points = np.random.rand(1000, 3) * 100
        >>> features = {
        ...     'height': np.random.rand(1000) * 20,
        ...     'planarity': np.random.rand(1000)
        ... }
        >>> labels = classify_points(points, features, mode='asprs_standard')
        >>> print(f"Classified {len(labels)} points")
        Classified 1000 points
        >>> print(f"Unique classes: {np.unique(labels)}")
        Unique classes: [2 3 5 6 9 11]

    Notes:
        - Point coordinates should be in a metric CRS (e.g., Lambert-93)
        - Height should be normalized to ground level
        - Planarity values are typically computed with 1-2m radius
        - Ground truth masks improve accuracy but are optional

    See Also:
        - get_thresholds: Get classification thresholds for mode
        - validate_features: Validate feature dictionary
        - ASPRSClass: ASPRS classification codes enumeration
    """
    # Implementation
    pass
```

### Class Docstrings

Document class purpose, attributes, and usage:

```python
class BuildingDetector:
    """
    Detect buildings in LiDAR point clouds using multiple strategies.

    This class implements building detection with configurable modes:
    - ASPRS standard: Simple building/non-building classification
    - LOD2: Roof/wall/ground classification for training
    - LOD3: Detailed architectural element detection

    The detector can use:
    1. Ground truth polygons (BD TOPO, cadastre)
    2. Geometric features (height, planarity, verticality)
    3. Hybrid approach combining both

    Attributes:
        mode: Detection mode (BuildingMode enum)
        config: Configuration settings (DetectionConfig)
        thresholds: Detection thresholds (ThresholdConfig)

    Example:
        >>> from ign_lidar.core.classification.building import BuildingDetector
        >>> detector = BuildingDetector(mode='lod2')
        >>> result = detector.detect(points, features)
        >>> print(f"Detected {result.mask.sum()} building points")
        Detected 15234 building points
        >>> print(f"Confidence: {result.confidence.mean():.2f}")
        Confidence: 0.87

    Notes:
        - Requires height, planarity features minimum
        - Verticality feature improves wall detection
        - Ground truth polygons give highest confidence

    See Also:
        - BuildingMode: Available detection modes
        - DetectionConfig: Configuration options
        - DetectionResult: Result format
    """

    def __init__(self, mode: str = 'asprs', config: Optional[DetectionConfig] = None):
        """
        Initialize building detector.

        Args:
            mode: Detection mode ('asprs', 'lod2', or 'lod3')
            config: Optional configuration. If None, uses defaults for mode.
        """
        pass
```

---

## 🔢 Type Hints

### Always Use Type Hints

Provide type hints for all function signatures:

```python
# ✅ CORRECT - Complete type hints
from typing import Dict, Optional, Tuple, List
import numpy as np

def process_features(
    features: Dict[str, np.ndarray],
    thresholds: Dict[str, float],
    enable_filtering: bool = True
) -> Tuple[np.ndarray, int]:
    """Process features with optional filtering."""
    pass

def get_building_mask(
    labels: np.ndarray,
    building_codes: List[int]
) -> np.ndarray:
    """Extract building mask from labels."""
    pass
```

```python
# ❌ INCORRECT - Missing type hints
def process_features(features, thresholds, enable_filtering=True):
    pass
```

### NumPy Array Annotations

Document array shapes in docstrings:

```python
def compute_normals(
    points: np.ndarray,  # [N, 3]
    search_radius: float = 1.0
) -> np.ndarray:  # [N, 3]
    """
    Compute surface normals for point cloud.

    Args:
        points: Point coordinates [N, 3] in meters
        search_radius: Neighborhood search radius in meters

    Returns:
        Surface normals [N, 3], unit vectors
    """
    pass
```

---

## 🧪 Code Organization

### File Structure

Organize code logically within files:

```python
# 1. Module docstring
"""
Module description here
"""

# 2. Imports (organized as shown above)
import logging
from typing import Dict
import numpy as np
from .base import BaseClass

# 3. Module-level constants
MAX_ITERATIONS = 100
DEFAULT_THRESHOLD = 0.5

# 4. Module-level logger
logger = logging.getLogger(__name__)

# 5. Exception classes
class ModuleError(Exception):
    pass

# 6. Enumerations
class DetectionMode(Enum):
    SIMPLE = 'simple'
    ADVANCED = 'advanced'

# 7. Dataclasses / Configuration
@dataclass
class ModuleConfig:
    pass

# 8. Main classes
class MainClassifier:
    pass

# 9. Utility functions
def helper_function():
    pass

# 10. Convenience/exported functions
def public_api_function():
    pass
```

### Method Organization Within Classes

```python
class Classifier:
    """Classifier class."""

    # 1. Class-level constants
    VERSION = '1.0.0'

    # 2. __init__
    def __init__(self, config):
        pass

    # 3. Public methods (alphabetically or by importance)
    def classify(self, data):
        pass

    def validate(self, data):
        pass

    # 4. Private methods (alphabetically or by usage)
    def _preprocess(self, data):
        pass

    def _postprocess(self, result):
        pass

    # 5. Properties
    @property
    def config(self):
        return self._config

    # 6. Special methods
    def __repr__(self):
        return f"Classifier(mode={self.mode})"
```

---

## ✅ Best Practices

### Feature Validation

Always validate input features:

```python
def classify(self, points: np.ndarray, features: Dict[str, np.ndarray]):
    """Classify points."""
    # Validate required features
    required = {'height', 'planarity'}
    if not required.issubset(features.keys()):
        missing = required - set(features.keys())
        raise FeatureValidationError(f"Missing required features: {missing}")

    # Validate array shapes
    n_points = len(points)
    for name, values in features.items():
        if len(values) != n_points:
            raise FeatureValidationError(
                f"Feature '{name}' has {len(values)} values, "
                f"expected {n_points} (matching points array)"
            )

    # Validate value ranges
    if np.any(np.isnan(features['height'])):
        raise FeatureValidationError("Feature 'height' contains NaN values")

    # Proceed with classification
    pass
```

### Defensive Programming

Check assumptions and handle edge cases:

```python
def cluster_by_footprint(self, points: np.ndarray, labels: np.ndarray):
    """Cluster points by building footprint."""
    # Check for empty inputs
    if len(points) == 0:
        logger.warning("Empty point cloud provided")
        return np.array([]), {}

    # Check for valid labels
    building_mask = labels == self.BUILDING_CODE
    if not building_mask.any():
        logger.info("No building points found")
        return np.array([]), {}

    # Check minimum points for clustering
    if building_mask.sum() < self.MIN_CLUSTER_POINTS:
        logger.warning(
            f"Only {building_mask.sum()} building points, "
            f"need at least {self.MIN_CLUSTER_POINTS} for clustering"
        )
        return points[building_mask], {}

    # Proceed with clustering
    pass
```

### Logging

Use appropriate logging levels:

```python
import logging
logger = logging.getLogger(__name__)

def process(self, data):
    logger.debug(f"Processing {len(data)} points")  # Verbose details
    logger.info("Starting classification")  # Important milestones
    logger.warning("Feature 'ndvi' not available, using fallback")  # Degraded mode
    logger.error(f"Classification failed: {error}")  # Recoverable errors
    logger.critical("Corrupted data detected, aborting")  # Fatal issues
```

---

## 🎨 Code Style

### Line Length

Keep lines under 100 characters when possible:

```python
# ✅ GOOD - Readable line breaks
result = detector.detect(
    points=point_cloud,
    features=feature_dict,
    ground_truth_mask=building_mask
)

# ❌ BAD - Too long
result = detector.detect(points=point_cloud, features=feature_dict, ground_truth_mask=building_mask, use_confidence=True)
```

### Whitespace

Use whitespace for readability:

```python
# ✅ GOOD - Clear separation
def classify_points(points, features):
    # Validation
    validate_features(features)

    # Preprocessing
    normalized = normalize_features(features)

    # Classification
    labels = apply_rules(points, normalized)

    # Postprocessing
    refined = refine_labels(labels, points)

    return refined
```

### Comments

Write clear, helpful comments:

```python
# ✅ GOOD - Explains WHY
# Use 95th percentile instead of max to avoid outliers
roof_height = np.percentile(building_heights, 95)

# ❌ BAD - States WHAT (obvious from code)
# Calculate 95th percentile
roof_height = np.percentile(building_heights, 95)
```

---

## 📦 Module Exports

### **init**.py Structure

Carefully control what's exported:

```python
"""
Building classification module.
"""

# Import from submodules
from .detection import BuildingDetector, DetectionConfig
from .base import BuildingMode, DetectionResult
from .adaptive import AdaptiveBuildingClassifier

# Define public API
__all__ = [
    'BuildingDetector',
    'DetectionConfig',
    'BuildingMode',
    'DetectionResult',
    'AdaptiveBuildingClassifier',
]

__version__ = '3.2.0'
```

---

## 🔄 Backward Compatibility

### Deprecation Warnings

When deprecating features, provide clear migration path:

```python
import warnings

def old_function(data):
    """
    Old classification function.

    .. deprecated:: 3.1.0
        Use :func:`new_function` instead. Will be removed in v4.0.0.
    """
    warnings.warn(
        "old_function is deprecated and will be removed in v4.0.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(data)

def new_function(data):
    """New improved classification function."""
    pass
```

---

## ✨ Summary

### Quick Checklist

Before committing code, verify:

- [ ] Imports are organized (stdlib → 3rd party → local)
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Configuration uses @dataclass
- [ ] Error handling is specific and informative
- [ ] Optional dependencies handled gracefully
- [ ] Input validation is defensive
- [ ] Logging uses appropriate levels
- [ ] Code follows naming conventions
- [ ] Line length < 100 characters
- [ ] Tests are included
- [ ] Documentation is updated

---

**Style Guide Version:** 1.0.0  
**Last Updated:** October 23, 2025  
**Maintained By:** IGN LiDAR HD Classification Team
