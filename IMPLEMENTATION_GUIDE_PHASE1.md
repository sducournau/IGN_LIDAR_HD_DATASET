# GPU Module Consolidation - Technical Implementation Guide

**Target:** Phase 1 - Critical GPU Module Consolidation  
**Duration:** 1-2 days  
**Impact:** 30% reduction in GPU code duplication

---

## 1. Step-by-Step Implementation

### Step 1: Audit Current State

**File: `ign_lidar/optimization/gpu.py`**

```python
# Current state
logger = logging.getLogger(__name__)
__all__ = []
# COMPLETELY EMPTY - NO EXPORTS
```

**File: `ign_lidar/optimization/gpu_memory.py`**

```python
# Current state
logger = logging.getLogger(__name__)
__all__ = []
# COMPLETELY EMPTY - NO EXPORTS
```

**File: `ign_lidar/core/gpu.py`** (PRIMARY)

```python
# 914 LOC - KEEP THIS ONE
class GPUManager:
    """Primary GPU management"""

class MultiGPUManager:
    """Multi-GPU support"""

def get_gpu_manager():
    """Singleton GPU manager"""
```

**File: `ign_lidar/core/gpu_memory.py`** (PRIMARY)

```python
# 611 LOC - KEEP THIS ONE
class GPUMemoryManager:
    """GPU memory allocation"""

class GPUMemoryPool:
    """Memory pooling"""

def get_gpu_memory_manager():
    """Singleton memory manager"""

def check_gpu_memory(size_gb):
    """Check available GPU memory"""
```

**File: `ign_lidar/optimization/gpu_wrapper.py`**

```python
# 278 LOC - REVIEW FOR UNIQUE FUNCTIONALITY
class GPUWrapper:
    """GPU wrapper (check if unique from gpu.py)"""
```

---

### Step 2: Find All Imports

**Search command:**

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET

# Find all imports of empty modules
grep -r "from ign_lidar.optimization.gpu import\|from ign_lidar.optimization.gpu_memory import" \
  ign_lidar --include="*.py" | tee /tmp/imports_to_fix.txt
```

**Expected output:**

```
ign_lidar/features/strategy_gpu.py:from ign_lidar.optimization.gpu import ...
ign_lidar/features/strategy_gpu_chunked.py:from ign_lidar.optimization.gpu import ...
ign_lidar/features/gpu_processor.py:from ign_lidar.optimization.gpu import ...
... (more files)
```

---

### Step 3: Analyze gpu_wrapper.py for Unique Functionality

```bash
# Check what GPUWrapper provides that gpu.py doesn't
grep -n "def " ign_lidar/optimization/gpu_wrapper.py

# Expected: GPU context management, device selection, etc.
# If similar to gpu.py, merge into core/gpu.py
# If unique, create specialized module in core/
```

---

### Step 4: Implementation - Phase 1a (Remove Empty Stubs)

**Delete these files:**

```bash
# REMOVE - these are completely empty
rm ign_lidar/optimization/gpu.py
rm ign_lidar/optimization/gpu_memory.py
```

**Create compatibility shim** (if needed for backward compatibility):

```python
# ign_lidar/optimization/gpu.py (NEW - DEPRECATED REDIRECT)
"""
DEPRECATED: GPU management moved to ign_lidar.core.gpu

This module is kept for backward compatibility only.
New code should import from ign_lidar.core.gpu
"""

import warnings
from ign_lidar.core.gpu import *  # noqa: F401, F403

warnings.warn(
    "Importing from ign_lidar.optimization.gpu is deprecated. "
    "Use ign_lidar.core.gpu instead.",
    DeprecationWarning,
    stacklevel=2
)
```

---

### Step 5: Update All Imports

**Automated replacement using Python:**

```python
# SCRIPT: migrate_gpu_imports.py
import os
import re
from pathlib import Path

PROJECT_ROOT = Path("/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET")
ign_lidar_dir = PROJECT_ROOT / "ign_lidar"

# Replacements to make
replacements = [
    (
        r"from ign_lidar\.optimization\.gpu import ",
        "from ign_lidar.core.gpu import "
    ),
    (
        r"from ign_lidar\.optimization\.gpu_memory import ",
        "from ign_lidar.core.gpu_memory import "
    ),
    (
        r"import ign_lidar\.optimization\.gpu( as ",
        "import ign_lidar.core.gpu\\1"
    ),
    (
        r"import ign_lidar\.optimization\.gpu_memory( as ",
        "import ign_lidar.core.gpu_memory\\1"
    ),
]

# Find all Python files
for py_file in ign_lidar_dir.rglob("*.py"):
    with open(py_file, "r") as f:
        content = f.read()

    original_content = content

    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Write back if changed
    if content != original_content:
        with open(py_file, "w") as f:
            f.write(content)
        print(f"Updated: {py_file}")

print("✓ All imports migrated")
```

**Run it:**

```bash
cd /mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET
python3 migrate_gpu_imports.py
```

---

### Step 6: Handle gpu_wrapper.py

**Option A: If GPUWrapper is Unique**

```bash
# Move to core and keep specialized
mv ign_lidar/optimization/gpu_wrapper.py ign_lidar/core/gpu_wrapper.py
# Update imports
grep -r "from ign_lidar.optimization.gpu_wrapper" ign_lidar --include="*.py" | \
  sed 's/:.*//g' | sort -u | xargs -I {} sed -i \
  's/from ign_lidar\.optimization\.gpu_wrapper/from ign_lidar.core.gpu_wrapper/g' {}
```

**Option B: If GPUWrapper Duplicates gpu.py**

```bash
# Merge into core/gpu.py and remove
python3 -c "
# Read gpu_wrapper.py
with open('ign_lidar/optimization/gpu_wrapper.py') as f:
    wrapper_code = f.read()

# Extract unique methods/classes
# Add to ign_lidar/core/gpu.py if not already present

# Remove wrapper file
import os
os.remove('ign_lidar/optimization/gpu_wrapper.py')
"
```

---

### Step 7: Consolidate GPU Profiling

**Similar process for GPU profilers:**

```bash
# Compare the two profilers
diff -u ign_lidar/core/gpu_profiler.py ign_lidar/optimization/gpu_profiler.py

# If optimization/gpu_profiler.py has unique features:
# 1. Extract those methods
# 2. Add to core/gpu_profiler.py
# 3. Remove optimization/gpu_profiler.py

# If just duplicate:
rm ign_lidar/optimization/gpu_profiler.py

# Create deprecated redirect:
cat > ign_lidar/optimization/gpu_profiler.py << 'EOF'
"""DEPRECATED: Use ign_lidar.core.gpu_profiler instead"""
import warnings
from ign_lidar.core.gpu_profiler import *  # noqa: F401, F403

warnings.warn(
    "Import from ign_lidar.core.gpu_profiler instead",
    DeprecationWarning, stacklevel=2
)
EOF
```

---

## 2. Verification Checklist

### After Each Step, Verify:

```bash
# 1. Check no orphaned imports exist
grep -r "from ign_lidar.optimization.gpu" ign_lidar --include="*.py" | grep -v "DEPRECATED\|redirect" | wc -l
# Should output: 0

# 2. Run imports test
python3 -c "
from ign_lidar.core.gpu import GPUManager, get_gpu_manager
from ign_lidar.core.gpu_memory import GPUMemoryManager, get_gpu_memory_manager
print('✓ All core GPU imports work')
"

# 3. Run full test suite
pytest tests/ -xvs -k "gpu" --tb=short

# 4. Check for import errors
python3 -c "
import ign_lidar
import ign_lidar.features
import ign_lidar.features.strategy_gpu
import ign_lidar.features.gpu_processor
print('✓ All feature imports work')
"

# 5. Verify no circular imports
python3 -c "
import sys
sys.path.insert(0, '.')
from ign_lidar.core import gpu
from ign_lidar.core import gpu_memory
from ign_lidar.features import strategy_gpu
print('✓ No circular imports')
"
```

---

## 3. GPU Memory Consolidation Details

### Current Duplication Issue:

**File A: `ign_lidar/core/gpu_memory.py` (611 LOC)**

```python
class GPUMemoryManager:
    """Manages GPU memory allocation and tracking"""

    def allocate(self, size_bytes):
        pass

    def deallocate(self, ptr):
        pass

    def get_free_memory(self):
        pass

class GPUMemoryPool:
    """GPU memory pool for efficient allocation"""

    def __init__(self, size_gb=8.0):
        pass

    def get_array(self, shape, dtype):
        pass
```

**File B: `ign_lidar/optimization/gpu_memory.py` (43 LOC)**

```python
logger = logging.getLogger(__name__)
__all__ = []
# EMPTY - JUST IMPORTS, NO IMPLEMENTATION
```

### Solution: Single Unified Module

**Target: `ign_lidar/core/gpu_memory.py` (PRIMARY)**

```python
"""GPU memory management and pooling

This module provides the unified GPU memory management API.
It handles allocation, pooling, and lifecycle management of GPU arrays.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any
import warnings

logger = logging.getLogger(__name__)

__all__ = [
    "GPUMemoryManager",
    "GPUMemoryPool",
    "get_gpu_memory_manager",
    "get_gpu_memory_pool",
    "cleanup_gpu_memory",
    "check_gpu_memory",
]


class GPUMemoryManager:
    """Unified GPU memory allocation manager

    Handles memory allocation, tracking, and cleanup.
    Provides factory methods for different GPU frameworks.
    """

    def __init__(self):
        self._allocations: Dict[int, Tuple[int, str]] = {}
        self._total_allocated = 0
        self._logger = logger

    def allocate(self, size_bytes: int, source: str = "unknown") -> Optional[int]:
        """Allocate GPU memory

        Args:
            size_bytes: Number of bytes to allocate
            source: Source of allocation request (for tracking)

        Returns:
            Allocation ID or None if failed
        """
        try:
            import cupy as cp
            ptr = cp.cuda.malloc(size_bytes)
            alloc_id = id(ptr)
            self._allocations[alloc_id] = (size_bytes, source)
            self._total_allocated += size_bytes
            return alloc_id
        except Exception as e:
            self._logger.error(f"GPU allocation failed: {e}")
            return None

    def deallocate(self, alloc_id: int) -> bool:
        """Deallocate GPU memory

        Args:
            alloc_id: Allocation ID to deallocate

        Returns:
            True if successful, False otherwise
        """
        if alloc_id not in self._allocations:
            self._logger.warning(f"Unknown allocation ID: {alloc_id}")
            return False

        size_bytes, source = self._allocations.pop(alloc_id)
        self._total_allocated -= size_bytes
        return True

    def get_total_allocated(self) -> int:
        """Get total allocated GPU memory in bytes"""
        return self._total_allocated

    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        return {
            "total_allocated_mb": self._total_allocated / (1024 * 1024),
            "num_allocations": len(self._allocations),
            "allocations_by_source": self._get_source_stats(),
        }

    def _get_source_stats(self) -> Dict[str, int]:
        """Get allocation stats by source"""
        stats = {}
        for size, source in self._allocations.values():
            stats[source] = stats.get(source, 0) + size
        return stats


class GPUMemoryPool:
    """GPU memory pool for efficient array allocation

    Maintains a pool of pre-allocated GPU arrays to avoid
    repeated allocation/deallocation overhead.
    """

    def __init__(self, size_gb: float = 8.0):
        """Initialize GPU memory pool

        Args:
            size_gb: Size of pool in GB
        """
        try:
            import cupy as cp
            self.pool = cp.cuda.MemoryPool()
            self.pool.set_limit(size=int(size_gb * 1024 * 1024 * 1024))
            self._enabled = True
        except Exception as e:
            logger.warning(f"GPU memory pool initialization failed: {e}")
            self._enabled = False

    def allocate_array(
        self,
        shape: Tuple,
        dtype: np.dtype = np.float32,
        zero_fill: bool = True
    ) -> Optional[Any]:
        """Allocate array from pool

        Args:
            shape: Array shape
            dtype: Data type
            zero_fill: Whether to zero-fill the array

        Returns:
            GPU array or None if failed
        """
        if not self._enabled:
            return None

        try:
            import cupy as cp
            with self.pool:
                if zero_fill:
                    array = cp.zeros(shape, dtype=dtype)
                else:
                    array = cp.empty(shape, dtype=dtype)
            return array
        except Exception as e:
            logger.error(f"Array allocation from pool failed: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        try:
            import cupy as cp
            free, total = cp.cuda.mempool.get_allocator().free_blocks()
            return {
                "free_mb": free / (1024 * 1024),
                "total_mb": total / (1024 * 1024),
                "utilization_percent": (total - free) / total * 100 if total > 0 else 0,
            }
        except Exception:
            return {}


# Singleton instances
_gpu_memory_manager_instance: Optional[GPUMemoryManager] = None
_gpu_memory_pool: Optional[GPUMemoryPool] = None


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get singleton GPU memory manager"""
    global _gpu_memory_manager_instance
    if _gpu_memory_manager_instance is None:
        _gpu_memory_manager_instance = GPUMemoryManager()
    return _gpu_memory_manager_instance


def get_gpu_memory_pool(size_gb: float = 8.0) -> Optional[GPUMemoryPool]:
    """Get singleton GPU memory pool

    Args:
        size_gb: Size of pool in GB (only used on first call)

    Returns:
        GPU memory pool or None if GPU not available
    """
    global _gpu_memory_pool
    if _gpu_memory_pool is None:
        _gpu_memory_pool = GPUMemoryPool(size_gb=size_gb)
    return _gpu_memory_pool if _gpu_memory_pool._enabled else None


def cleanup_gpu_memory() -> bool:
    """Clean up GPU memory pools

    Returns:
        True if successful, False otherwise
    """
    try:
        import cupy as cp
        cp.cuda.MemoryPool().free_all_blocks()
        logger.info("GPU memory cleaned up")
        return True
    except Exception as e:
        logger.error(f"GPU memory cleanup failed: {e}")
        return False


def check_gpu_memory(size_gb: float) -> bool:
    """Check if sufficient GPU memory is available

    Args:
        size_gb: Required memory in GB

    Returns:
        True if sufficient memory is available, False otherwise
    """
    try:
        import cupy as cp
        free, _ = cp.cuda.mempool.get_allocator().free_blocks()
        return free >= size_gb * 1024 * 1024 * 1024
    except Exception:
        return False
```

---

## 4. Testing Strategy

### Unit Tests to Add:

```python
# tests/test_gpu_module_consolidation.py
import pytest
from pathlib import Path

def test_no_optimization_gpu_imports():
    """Verify no code imports from optimization.gpu (except compatibility shims)"""
    ign_lidar = Path("ign_lidar")

    for py_file in ign_lidar.rglob("*.py"):
        if "optimization" in str(py_file):
            continue  # Skip optimization module itself

        with open(py_file) as f:
            content = f.read()

        # Should not import from optimization.gpu except deprecation shims
        if "from ign_lidar.optimization.gpu" in content:
            # OK if in deprecation redirect module
            if "DEPRECATED" not in content:
                pytest.fail(f"{py_file} imports from optimization.gpu")


def test_gpu_memory_single_source():
    """Verify all GPU memory functionality comes from core.gpu_memory"""
    from ign_lidar.core.gpu_memory import (
        GPUMemoryManager,
        GPUMemoryPool,
        get_gpu_memory_manager,
        get_gpu_memory_pool,
        cleanup_gpu_memory,
        check_gpu_memory,
    )

    # All imports should work
    assert GPUMemoryManager is not None
    assert GPUMemoryPool is not None
    assert callable(get_gpu_memory_manager)
    assert callable(get_gpu_memory_pool)
    assert callable(cleanup_gpu_memory)
    assert callable(check_gpu_memory)


def test_gpu_manager_singleton():
    """Verify GPU manager is singleton"""
    from ign_lidar.core.gpu_memory import get_gpu_memory_manager

    manager1 = get_gpu_memory_manager()
    manager2 = get_gpu_memory_manager()

    assert manager1 is manager2


def test_backward_compatibility():
    """Test deprecated imports still work (with warnings)"""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from ign_lidar.optimization.gpu import GPUManager  # noqa: F401

        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
```

---

## 5. Rollback Plan

If issues arise during consolidation:

```bash
# 1. Restore deleted files from git
git checkout ign_lidar/optimization/gpu.py
git checkout ign_lidar/optimization/gpu_memory.py

# 2. Revert import changes
git checkout -- ign_lidar/

# 3. Revert to pre-consolidation state
git reset --hard HEAD~<number_of_commits>
```

---

## 6. Success Criteria

✓ **Must-haves:**

- [ ] All tests pass: `pytest tests/ -xvs`
- [ ] No import errors when importing `ign_lidar`
- [ ] No circular imports detected
- [ ] GPU operations work correctly with benchmark tests

✓ **Should-haves:**

- [ ] 0 deprecation warnings in normal operation
- [ ] Backward compatibility maintained for 1 release cycle
- [ ] Documentation updated

✓ **Performance:**

- [ ] No performance regression
- [ ] GPU memory management shows improvements

---

**Next Steps:**

1. Execute Steps 1-2 (Audit & Find Imports)
2. Run verification checks
3. Execute Steps 3-7 (Implementation)
4. Run full test suite
5. Commit with message: "refactor: consolidate GPU modules - phase 1"
