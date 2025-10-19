# Phase 3 Implementation Plan: Module Directory Reorganization

**Date**: October 19, 2025  
**Status**: READY TO EXECUTE  
**Risk**: MEDIUM (affects 50+ files)

---

## Overview

Rename two directories for better semantic clarity:

1. `core/modules/` → `core/classification/`
2. `features/core/` → `features/compute/`

---

## Impact Analysis

### Files Affected by `core/modules/` → `core/classification/`

**Import Pattern**: `from ign_lidar.core.modules.X import Y`

**Affected Files** (~30+ files):

- Tests: 10+ test files
- Scripts: 10+ script files
- Examples: 1 example file
- Documentation: 10+ markdown files

**Key Modules**:

- `advanced_classification.py`
- `classification_thresholds.py`
- `ground_truth_refinement.py`
- `ground_truth_artifact_checker.py`
- `adaptive_classifier.py`
- `parcel_classifier.py`
- `feature_validator.py`
- `geometric_rules.py`
- `spectral_rules.py`
- and 20 more...

### Files Affected by `features/core/` → `features/compute/`

**Import Pattern**: `from ign_lidar.features.core.X import Y`

**Affected Files** (~15+ files):

- Tests: 5+ test files
- Scripts: 5+ script files
- Documentation: 5+ markdown files
- Internal: 1 file (gpu_bridge.py imports itself)

**Key Modules**:

- `gpu_bridge.py`
- `normals.py`
- `curvature.py`
- `eigenvalues.py`
- `height.py`
- `features.py`
- `utils.py`
- and 5 more...

---

## Implementation Strategy

### Step 1: Rename Directories

```bash
# Rename core/modules to core/classification
git mv ign_lidar/core/modules ign_lidar/core/classification

# Rename features/core to features/compute
git mv ign_lidar/features/core ign_lidar/features/compute
```

### Step 2: Add Backward Compatibility

Create import redirection in `__init__.py` files to maintain v3.x compatibility.

#### In `ign_lidar/core/__init__.py`:

```python
# Backward compatibility: core.modules moved to core.classification in v3.1.0
import sys
import warnings

# Redirect old imports
class _ModulesRedirect:
    \"\"\"Redirect imports from core.modules to core.classification\"\"\"

    def __getattr__(self, name):
        warnings.warn(
            f"Importing from ign_lidar.core.modules is deprecated. "
            f"Use ign_lidar.core.classification instead. "
            f"This will be removed in v4.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
        import importlib
        module = importlib.import_module(f'ign_lidar.core.classification.{name}')
        return module

# Create alias
sys.modules['ign_lidar.core.modules'] = _ModulesRedirect()
```

#### In `ign_lidar/features/__init__.py`:

Similar approach for `features.core` → `features.compute`.

### Step 3: Update Internal Imports

Update imports within the moved directories themselves (internal references).

#### In `core/classification/` (formerly `core/modules/`):

Find and replace:

```
from ign_lidar.core.modules. → from ign_lidar.core.classification.
from ..modules. → from ..classification.
from .modules. → from .classification.
```

#### In `features/compute/` (formerly `features/core/`):

Find and replace:

```
from ign_lidar.features.core. → from ign_lidar.features.compute.
from ..core. → from ..compute.
from .core. → from .compute.
```

### Step 4: Update `__init__.py` Files

Update the `__init__.py` in the renamed directories to reflect new location.

---

## Detailed Steps

### Phase 3A: Rename `core/modules/` → `core/classification/`

1. **Git Rename**:

   ```bash
   git mv ign_lidar/core/modules ign_lidar/core/classification
   ```

2. **Update Internal Imports**:
   - Search: `from ign_lidar.core.modules.`
   - Replace: `from ign_lidar.core.classification.`
   - Files: All files in `core/classification/`
3. **Update `core/classification/__init__.py`**:

   - Update header comments
   - Update module docstring

4. **Add Backward Compatibility**:

   - Update `core/__init__.py` with redirect logic

5. **Test**:

   ```python
   # Test new import
   from ign_lidar.core.classification.classification_thresholds import ClassificationThresholds

   # Test old import (should work with warning)
   from ign_lidar.core.modules.classification_thresholds import ClassificationThresholds
   ```

### Phase 3B: Rename `features/core/` → `features/compute/`

1. **Git Rename**:

   ```bash
   git mv ign_lidar/features/core ign_lidar/features/compute
   ```

2. **Update Internal Imports**:
   - Search: `from ign_lidar.features.core.`
   - Replace: `from ign_lidar.features.compute.`
   - Files: All files in `features/compute/`
3. **Update `features/compute/__init__.py`**:

   - Update header comments
   - Update module docstring

4. **Add Backward Compatibility**:

   - Update `features/__init__.py` with redirect logic

5. **Test**:

   ```python
   # Test new import
   from ign_lidar.features.compute.eigenvalues import compute_eigenvalues

   # Test old import (should work with warning)
   from ign_lidar.features.core.eigenvalues import compute_eigenvalues
   ```

---

## Testing Strategy

### Unit Tests

After each rename, run:

```bash
# Test classification modules
pytest tests/test_*classification*.py -v

# Test feature modules
pytest tests/test_core_*.py -v
pytest tests/test_gpu_bridge.py -v
```

### Import Tests

Create test script to verify all import paths:

```python
# Test new paths
from ign_lidar.core.classification import *
from ign_lidar.features.compute import *

# Test old paths (should issue warnings)
import warnings
warnings.simplefilter('always', DeprecationWarning)

from ign_lidar.core.modules.classification_thresholds import ClassificationThresholds
from ign_lidar.features.core.eigenvalues import compute_eigenvalues
```

### Integration Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific integration tests
pytest tests/test_*integration*.py -v
```

---

## Rollback Plan

If issues arise:

```bash
# Rollback Phase 3A
git mv ign_lidar/core/classification ign_lidar/core/modules

# Rollback Phase 3B
git mv ign_lidar/features/compute ign_lidar/features/core

# Revert __init__.py changes
git checkout ign_lidar/core/__init__.py
git checkout ign_lidar/features/__init__.py
```

---

## Success Criteria

- [ ] Directories renamed successfully
- [ ] Internal imports updated
- [ ] Backward compatibility working
- [ ] All tests passing
- [ ] Deprecation warnings showing
- [ ] No import errors

---

## Estimated Time

- Phase 3A: 1-2 hours
- Phase 3B: 1 hour
- Testing: 1 hour
- **Total**: 3-4 hours

---

## Notes

### Why Not Update All Imports Now?

We're deferring external import updates to Phase 4 because:

1. Backward compatibility allows gradual migration
2. Can update in batches (tests, scripts, docs separately)
3. Lower risk - one step at a time
4. Users have time to migrate their code

### Documentation Update

Will be handled in Phase 6 after all structural changes are complete.

---

**Ready to Execute**: Phase 3A first, then Phase 3B after verification.
