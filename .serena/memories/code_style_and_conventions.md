# Code Style and Conventions

## Code Modification Rules (CRITICAL)

### Always Follow These Rules:
1. **Modify Existing Files First:** Never create new files without checking if functionality can be added to existing files
2. **Update Before Create:** Always update existing implementations before creating new ones
3. **Avoid Duplication:** Search for similar functionality and extend/refactor rather than duplicate
4. **No Redundant Prefixes:** Avoid "unified", "enhanced", "new", "improved" prefixes - name by purpose
5. **Refactor Over Rewrite:** Refactor existing code rather than creating parallel versions

### What NOT to Do:
- ❌ Creating `unified_feature_computer.py` when `feature_computer.py` exists
- ❌ Adding `enhanced_process_tile()` when `process_tile()` can be improved
- ❌ Creating `new_classifier.py` alongside `classifier.py`

### What to Do:
- ✅ Update `feature_computer.py` with new capabilities
- ✅ Refactor `process_tile()` to handle new cases
- ✅ Extend `classifier.py` with additional methods

## Python Style

### PEP 8 Compliance
- **Line length:** 88 characters (Black formatter)
- **Target version:** Python 3.8+
- **Formatter:** Black
- **Linter:** Flake8 with `--max-line-length=88 --ignore=E203,W503`
- **Import sorter:** isort with `--profile=black`
- **Type checker:** mypy with `--ignore-missing-imports`

### Type Hints
- Use comprehensive type annotations (Python 3.8+ syntax)
- Import from `typing`: `Dict`, `List`, `Optional`, `Union`, `Any`, `Tuple`
- Example: `def process(data: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:`

### Docstrings
- **Style:** Google-style docstrings
- **Required for:** All public functions, classes, and methods
- **Format:**
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    One-line summary.

    Longer description explaining purpose and details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Example:
        >>> result = function_name(data, config)
        >>> print(result.shape)
        (1000, 3)

    Note:
        Additional notes or warnings.
    """
```

### Import Organization
Order imports as: stdlib, third-party, local
```python
# Standard library
from typing import Dict, Optional
from pathlib import Path

# Third-party
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Local
from ign_lidar.core import LiDARProcessor
from ign_lidar.features import FeatureOrchestrator
```

### Naming Conventions
- **Classes:** `PascalCase` (e.g., `LiDARProcessor`, `FeatureOrchestrator`)
- **Functions/Methods:** `snake_case` (e.g., `compute_features`, `process_tile`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `ASPRS_CLASS_NAMES`, `LOD2_CLASSES`)
- **Private members:** Prefix with `_` (e.g., `_process_tile_core`, `_validate_config`)
- **Protected members:** Single underscore prefix `_`
- **Name mangling:** Double underscore prefix `__` (rarely used)

### Error Handling
- Use custom exceptions from `core.error_handler`
- Provide clear error messages
- Include context in exceptions
- Example:
```python
from ign_lidar.core.error_handler import ProcessingError, GPUMemoryError

try:
    result = process_tile(tile_path)
except GPUMemoryError:
    logger.warning("GPU OOM, falling back to CPU")
    result = process_tile_cpu(tile_path)
except ProcessingError as e:
    logger.error(f"Failed to process {tile_path}: {e}")
    raise
```

## File Organization
- One class per file for major components
- Group related functions in modules
- Use `__init__.py` for public API exports
- Keep files focused and under ~500 lines when possible
