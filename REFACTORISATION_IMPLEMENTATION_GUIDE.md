# REFACTORISATION GUIDE - STEP BY STEP IMPLEMENTATION

## Overview

This guide provides detailed, step-by-step instructions for implementing each consolidation phase.
Each phase includes code examples, testing strategy, and risk mitigation.

---

## PHASE 1: GPU MANAGER CONSOLIDATION

### Objective

Consolidate 5 GPU management classes into 1 unified `GPUManager`

### Files Involved

**Primary**:

- `ign_lidar/core/gpu.py` (will be expanded)
- `ign_lidar/core/gpu_memory.py` (will be merged)
- `ign_lidar/core/gpu_stream_manager.py` (will be merged)

**To Delete**:

- `ign_lidar/core/gpu_unified.py`
- `ign_lidar/optimization/cuda_streams.py`

**Imports to Update**: 15+ files

### Step 1.1: Identify Current API

```bash
# Find all public methods in each manager
grep -n "def " ign_lidar/core/gpu.py
grep -n "def " ign_lidar/core/gpu_memory.py
grep -n "def " ign_lidar/core/gpu_stream_manager.py

# Find all usages
grep -r "from.*gpu import GPUManager" ign_lidar/
grep -r "from.*gpu_memory import" ign_lidar/
grep -r "from.*gpu_stream_manager import" ign_lidar/
```

### Step 1.2: Create Unified GPU Manager

**File**: `ign_lidar/core/gpu.py`

```python
class GPUManager(metaclass=SingletonMeta):
    """Unified GPU management (detection, memory, streams)."""

    def __init__(self):
        # === DETECTION (from old GPUManager) ===
        self.capabilities = self._detect_gpu()
        self.is_available = self.capabilities is not None

        # === MEMORY (from old GPUMemoryManager) ===
        if self.is_available:
            self.memory_info = self._get_memory_info()
            self.memory_pool = None  # Lazy initialization

        # === STREAMS (from old GPUStreamManager) ===
        self.stream_pool = StreamPool() if self.is_available else None

        # === COMPATIBILITY (from old UnifiedGPUManager) ===
        self.detector = self  # For backward compatibility
        self.memory_manager = self
        self.stream_manager = self

    # --- DETECTION API (from gpu.py) ---

    def get_capability(self, attr: str):
        """Get GPU capability (backward compat)."""
        if not self.is_available:
            raise GPUNotAvailableError()
        return self.capabilities[attr]

    def supports_fp64(self) -> bool:
        """Check if GPU supports float64."""
        return self.is_available and self.capabilities.get('compute_capability', (0, 0)) >= (1, 3)

    # --- MEMORY API (from gpu_memory.py) ---

    def allocate(self, size: int, dtype=cp.float32) -> cp.ndarray:
        """Allocate GPU memory."""
        if not self.is_available:
            raise GPUNotAvailableError()
        return cp.asarray(cp.zeros(size, dtype=dtype))

    def get_free_memory(self) -> int:
        """Get free GPU memory in bytes."""
        if not self.is_available:
            return 0
        mempool = cp.get_default_memory_pool()
        return mempool.get_limit() - mempool.used_bytes()

    def synchronize_memory(self):
        """Synchronize all GPU memory operations."""
        if self.is_available:
            cp.cuda.Stream.null.synchronize()

    # --- STREAM API (from gpu_stream_manager.py) ---

    def create_stream(self) -> cp.cuda.Stream:
        """Create a new CUDA stream."""
        if not self.is_available:
            raise GPUNotAvailableError()
        return cp.cuda.Stream()

    def synchronize_stream(self, stream: cp.cuda.Stream = None):
        """Synchronize a stream (or default)."""
        if stream is None:
            stream = cp.cuda.Stream.null
        stream.synchronize()

    def get_stream_pool(self) -> 'StreamPool':
        """Get the stream pool."""
        if not self.is_available:
            raise GPUNotAvailableError()
        return self.stream_pool

    # --- INTERNAL HELPERS ---

    def _detect_gpu(self) -> dict:
        """Detect GPU and get capabilities."""
        try:
            import cupy as cp
            device = cp.cuda.Device()
            return {
                'name': device.get_attribute(cp.cuda.device_attribute.COMPUTE_CAPABILITY),
                'memory': device.get_attribute(cp.cuda.device_attribute.TOTAL_MEMORY),
                # ... other attributes
            }
        except Exception:
            return None

    def _get_memory_info(self) -> dict:
        """Get current GPU memory info."""
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            return {
                'free': self.get_free_memory(),
                'used': mempool.used_bytes(),
                'limit': mempool.get_limit(),
            }
        except Exception:
            return {}


class StreamPool:
    """Manage a pool of CUDA streams."""

    def __init__(self, size: int = 4):
        self.streams = [cp.cuda.Stream() for _ in range(size)]
        self.current = 0

    def get_stream(self) -> cp.cuda.Stream:
        """Get next stream from pool (round-robin)."""
        stream = self.streams[self.current]
        self.current = (self.current + 1) % len(self.streams)
        return stream

    def synchronize_all(self):
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()
```

### Step 1.3: Update All Imports

**Template for each file**:

```python
# OLD CODE
from ign_lidar.core.gpu import GPUManager as BaseGPUManager
from ign_lidar.core.gpu_memory import GPUMemoryManager
from ign_lidar.core.gpu_stream_manager import GPUStreamManager

# NEW CODE
from ign_lidar.core.gpu import GPUManager

gpu_manager = GPUManager()

# Access methods (all same interface)
gpu_manager.get_free_memory()
gpu_manager.create_stream()
```

**Files to update** (identified via grep):

```
ign_lidar/features/gpu_processor.py
ign_lidar/features/strategy_gpu.py
ign_lidar/features/strategy_gpu_chunked.py
ign_lidar/optimization/gpu_kernels.py
ign_lidar/optimization/gpu_async.py
ign_lidar/optimization/gpu_wrapper.py
ign_lidar/optimization/gpu_safety.py
[... and others ...]
```

### Step 1.4: Delete Redundant Files

```bash
# After all imports updated and tests pass:
rm ign_lidar/core/gpu_unified.py
rm ign_lidar/core/gpu_memory.py
rm ign_lidar/optimization/cuda_streams.py

# Update __init__.py if needed
```

### Step 1.5: Test Strategy

```bash
# Unit tests (no GPU required)
pytest tests/test_core_gpu_manager.py -v -m "not gpu"

# GPU tests (if GPU available)
pytest tests/test_core_gpu_manager.py -v -m "gpu"

# Integration tests
pytest tests/ -v -k "gpu" --tb=short

# Performance regression check
python scripts/benchmark_gpu.py --before-after
```

---

## PHASE 2: RGB/NIR FEATURES DEDUPLICATION

### Objective

Extract RGB/NIR computation from 3 strategies into 1 reusable module

### Step 2.1: Create Generic RGB/NIR Module

**File**: `ign_lidar/features/compute/rgb_nir.py`

```python
"""Unified RGB/NIR feature computation."""

from typing import Dict, Union
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def compute_rgb_features(
    rgb: np.ndarray,
    backend: str = 'auto'
) -> Dict[str, np.ndarray]:
    """
    Compute RGB-based features.

    Args:
        rgb: RGB array [N, 3] with values in [0, 255]
        backend: 'cpu' (numpy), 'gpu' (cupy), 'numba', 'auto'

    Returns:
        Dictionary with features:
        - brightness: Mean intensity
        - saturation: Color saturation
        - hue: Color hue
        - red_intensity: R channel
        - green_intensity: G channel
        - blue_intensity: B channel

    Raises:
        ValueError: If backend not available
        RuntimeError: If gpu backend requested but cupy not available
    """

    # Auto-detect backend
    if backend == 'auto':
        backend = 'gpu' if cp is not None else 'cpu'

    # Dispatch to appropriate backend
    if backend == 'gpu':
        if cp is None:
            raise RuntimeError("CuPy not available, cannot use GPU backend")
        return _compute_rgb_features_gpu(rgb)
    elif backend == 'cpu':
        return _compute_rgb_features_cpu(rgb)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _compute_rgb_features_cpu(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """CPU implementation using NumPy."""

    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float32) / 255.0

    # Extract channels
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]

    # Brightness (mean intensity)
    brightness = np.mean(rgb_norm, axis=-1)

    # Saturation (HSV)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = np.where(
        max_c > 0,
        (max_c - min_c) / max_c,
        0
    )

    # Hue (HSV)
    delta = max_c - min_c
    hue = np.zeros_like(r)

    # Red max
    mask_r = max_c == r
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r] + 0)) % 360

    # Green max
    mask_g = max_c == g
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)) % 360

    # Blue max
    mask_b = max_c == b
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)) % 360

    return {
        'brightness': brightness,
        'saturation': saturation,
        'hue': hue / 360,  # Normalize to [0, 1]
        'red_intensity': r,
        'green_intensity': g,
        'blue_intensity': b,
    }


def _compute_rgb_features_gpu(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """GPU implementation using CuPy."""

    # Convert to GPU array
    rgb_gpu = cp.asarray(rgb)

    # Normalize to [0, 1]
    rgb_norm = rgb_gpu.astype(cp.float32) / 255.0

    # Extract channels
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]

    # Brightness
    brightness = cp.mean(rgb_norm, axis=-1)

    # Saturation
    max_c = cp.maximum(cp.maximum(r, g), b)
    min_c = cp.minimum(cp.minimum(r, g), b)
    saturation = cp.where(
        max_c > 0,
        (max_c - min_c) / max_c,
        0
    )

    # Hue
    delta = max_c - min_c
    hue = cp.zeros_like(r)

    mask_r = max_c == r
    hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r] + 0)) % 360

    mask_g = max_c == g
    hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)) % 360

    mask_b = max_c == b
    hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)) % 360

    # Return as CPU arrays for consistency
    return {
        'brightness': cp.asnumpy(brightness),
        'saturation': cp.asnumpy(saturation),
        'hue': cp.asnumpy(hue / 360),
        'red_intensity': cp.asnumpy(r),
        'green_intensity': cp.asnumpy(g),
        'blue_intensity': cp.asnumpy(b),
    }
```

### Step 2.2: Update Strategies to Use New Module

**Example: strategy_cpu.py**

```python
# OLD CODE
def _compute_rgb_features_cpu(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute RGB features (30 lines of duplicated code)."""
    # ... 30 lines of implementation ...

# NEW CODE
from ign_lidar.features.compute.rgb_nir import compute_rgb_features

def _compute_features_rgb(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute RGB features (1 line, calls shared implementation)."""
    return compute_rgb_features(rgb, backend='cpu')
```

**Apply to**:

- `strategy_cpu.py`
- `strategy_gpu.py`
- `strategy_gpu_chunked.py`

### Step 2.3: Testing

```python
# tests/test_rgb_nir_deduplication.py

import numpy as np
from ign_lidar.features.compute.rgb_nir import compute_rgb_features

def test_rgb_features_cpu():
    """Test RGB features on CPU."""
    rgb = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)
    features = compute_rgb_features(rgb, backend='cpu')

    assert 'brightness' in features
    assert features['brightness'].shape == (1000,)
    assert 0 <= features['brightness'].min() <= features['brightness'].max() <= 1

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_rgb_features_gpu():
    """Test RGB features on GPU."""
    rgb = np.random.randint(0, 256, (1000, 3), dtype=np.uint8)

    # CPU reference
    features_cpu = compute_rgb_features(rgb, backend='cpu')

    # GPU results
    features_gpu = compute_rgb_features(rgb, backend='gpu')

    # Compare (allowing small numerical differences)
    for key in features_cpu:
        np.testing.assert_allclose(
            features_cpu[key],
            features_gpu[key],
            rtol=1e-5
        )

def test_auto_backend():
    """Test automatic backend selection."""
    rgb = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
    features = compute_rgb_features(rgb, backend='auto')
    assert 'brightness' in features
```

---

## PHASE 3-8: Additional Phases

(Detailed implementation guides for Phases 3-8 would follow similar structure)

---

## ROLLBACK STRATEGY

If issues arise during implementation:

### Quick Rollback

```bash
# If changes not committed
git checkout -- <modified_files>

# If committed but not pushed
git reset --hard HEAD~1

# Partial rollback (revert specific commit)
git revert <commit_hash>
```

### Longer Rollback

```bash
# Create a new branch from stable commit
git checkout -b rollback_point <stable_hash>

# Cherry-pick good changes
git cherry-pick <good_commit>...

# Merge back when stable
git checkout main
git merge rollback_point
```

---

## VALIDATION CHECKLIST

After each phase:

- [ ] All unit tests pass
- [ ] No performance regression
- [ ] No new lint/style issues
- [ ] Documentation updated
- [ ] No duplicate code detected
- [ ] Code coverage > 95%
- [ ] Git history clean (no WIP commits)

---

## Next Steps

1. Review this guide with team
2. Assign phases to developers
3. Create feature branches per phase
4. Run full test suite before each phase
5. Perform benchmarking after critical phases
6. Document lessons learned
7. Update internal wiki with new architecture
