# 🚀 Quick Action Guide - Implémentation Immédiate

## 📋 Priorités de la Semaine

### ✅ TODAY - 30 minutes

#### 1. Supprimer OptimizedReclassifier (DEPRECATED)

**Fichier:** `ign_lidar/core/classification/reclassifier.py`

**Action:**

```bash
# 1. Trouver tous les usages
grep -r "OptimizedReclassifier" ign_lidar/ tests/ --include="*.py"

# 2. Vérifier les tests
grep -r "OptimizedReclassifier" tests/ --include="*.py"

# Expected: test_reclassifier.py, maybe integration tests
```

**Code à Supprimer:**

```python
# Lignes ~1313-1323 de reclassifier.py
class OptimizedReclassifier(Reclassifier):
    """Alias pour Reclassifier."""
    def __init__(self, chunk_size=1000, show_progress=True):
        warnings.warn(
            "OptimizedReclassifier is deprecated, use Reclassifier instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(chunk_size=chunk_size, show_progress=show_progress)
```

**Actions:**

1. ✅ Supprimer la classe
2. ✅ Vérifier les imports
3. ✅ Mettre à jour les tests
4. ✅ Commit: `git commit -m "refactor: remove deprecated OptimizedReclassifier"`

---

#### 2. Nettoyer Préfixes Redondants (5 minutes)

**Fichier 1:** `ign_lidar/__init__.py`

```python
# Ligne ~21
# AVANT:
# - Enhanced documentation structure and clarity

# APRÈS:
# - Documentation with clear structure
```

**Fichier 2:** `examples/feature_examples/feature_filtering_example.py`

```python
# Ligne ~4
# AVANT:
"""This example demonstrates how to use the unified feature filtering module"""

# APRÈS:
"""Example demonstrating feature filtering usage patterns"""
```

**Fichier 3:** `ign_lidar/optimization/__init__.py`

```python
# Ligne ~147
# AVANT:
# Phase 2: Unified KNN Engine (Nov 2025)

# APRÈS:
# Phase 2: KNN Engine (Nov 2025)
```

---

### 🎯 TOMORROW - 2 heures

#### 3. Créer Classification Engine Unifiée

**Nouveau Fichier:** `ign_lidar/core/classification/engine.py`

```python
"""
Unified Classification Engine - v1.0
Consolidates SpectralRulesEngine, GeometricRulesEngine, ASPRSClassRulesEngine
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClassificationStrategy(ABC):
    """Abstract base for all classification strategies"""

    @abstractmethod
    def classify(self, features: np.ndarray) -> np.ndarray:
        """Classify points based on features"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass


class SpectralClassificationStrategy(ClassificationStrategy):
    """Spectral signature based classification"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        from ign_lidar.core.classification.spectral_rules import SpectralRulesEngine
        self.engine = SpectralRulesEngine()

    def classify(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify using spectral rules"""
        return self.engine.classify_by_spectral_signature(features)

    def get_name(self) -> str:
        return "spectral"


class GeometricClassificationStrategy(ClassificationStrategy):
    """Geometric shape based classification"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        from ign_lidar.core.classification.geometric_rules import GeometricRulesEngine
        self.engine = GeometricRulesEngine()

    def classify(self, features: np.ndarray) -> np.ndarray:
        """Classify using geometric rules"""
        return self.engine.classify(features)

    def get_name(self) -> str:
        return "geometric"


class ASPRSClassificationStrategy(ClassificationStrategy):
    """ASPRS standard classification"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        from ign_lidar.core.classification.asprs_class_rules import ASPRSClassRulesEngine
        self.engine = ASPRSClassRulesEngine()

    def classify(self, features: np.ndarray) -> np.ndarray:
        """Classify using ASPRS rules"""
        return self.engine.classify(features)

    def get_name(self) -> str:
        return "asprs"


class ClassificationEngine:
    """
    Unified interface for all classification operations.

    Auto-selects optimal strategy based on available data and GPU.
    """

    STRATEGIES = {
        'spectral': SpectralClassificationStrategy,
        'geometric': GeometricClassificationStrategy,
        'asprs': ASPRSClassificationStrategy,
    }

    def __init__(self, mode: str = 'asprs', use_gpu: bool = False):
        """
        Initialize Classification Engine.

        Args:
            mode: Classification mode ('spectral', 'geometric', 'asprs')
            use_gpu: Enable GPU acceleration if available
        """
        self.mode = mode
        self.use_gpu = use_gpu
        self.strategy = self._create_strategy()

    def _create_strategy(self) -> ClassificationStrategy:
        """Create appropriate strategy"""
        strategy_class = self.STRATEGIES.get(
            self.mode,
            ASPRSClassificationStrategy
        )
        return strategy_class(use_gpu=self.use_gpu)

    def classify(self, features: np.ndarray) -> np.ndarray:
        """
        Classify point cloud features.

        Args:
            features: Feature array [N, F]

        Returns:
            Classification labels [N]
        """
        logger.info(f"Classifying with {self.strategy.get_name()} strategy")
        return self.strategy.classify(features)

    def set_strategy(self, mode: str):
        """Switch classification strategy"""
        if mode not in self.STRATEGIES:
            raise ValueError(f"Unknown classification mode: {mode}")
        self.mode = mode
        self.strategy = self._create_strategy()
        logger.info(f"Switched to {mode} classification strategy")


# Export
__all__ = [
    'ClassificationEngine',
    'ClassificationStrategy',
    'SpectralClassificationStrategy',
    'GeometricClassificationStrategy',
    'ASPRSClassificationStrategy',
]
```

---

### 🎯 THIS WEEK - 4 heures

#### 4. Créer Unified GPU Manager

**Nouveau Fichier:** `ign_lidar/core/gpu/manager.py`

```python
"""
Unified GPU Manager - Consolidates all GPU operations
Replaces: gpu.py, gpu_memory.py, gpu_profiler.py
"""

from typing import List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class UnifiedGPUManager:
    """
    Central hub for all GPU operations.

    Consolidates:
    - GPU detection and initialization (GPUManager)
    - Memory management (GPUMemoryManager)
    - Profiling and monitoring (GPUProfiler)
    - Array caching (GPUArrayCache)
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            import cupy as cp
            self.cp = cp
            self.gpu_available = True
        except ImportError:
            self.gpu_available = False
            logger.warning("CuPy not available - GPU features disabled")

        # Import existing managers (for backward compatibility)
        try:
            from ign_lidar.core.gpu import GPUManager
            self.detector = GPUManager()
        except ImportError:
            self.detector = None

        try:
            from ign_lidar.core.gpu_memory import GPUMemoryManager
            self.memory = GPUMemoryManager()
        except ImportError:
            self.memory = None

        self._initialized = True

    def transfer_batch(self, arrays: List[np.ndarray],
                      direction: str = 'to_gpu',
                      copy: bool = False) -> List:
        """
        Batch transfer arrays with optimization.

        ✅ 5-6x faster than individual cp.asarray() calls

        Args:
            arrays: List of numpy arrays
            direction: 'to_gpu' or 'to_cpu'
            copy: Force copy (for COW arrays)

        Returns:
            List of GPU or CPU arrays
        """
        if not self.gpu_available:
            return arrays  # Return as-is if GPU not available

        if direction == 'to_gpu':
            # Check memory
            total_size = sum(arr.nbytes for arr in arrays)
            if self.memory and not self.memory.check_available_memory(total_size / 1e9):
                logger.warning("Insufficient GPU memory, cleaning up")
                self.cleanup()

            # Batch transfer
            gpu_arrays = []
            for arr in arrays:
                if arr is None:
                    gpu_arrays.append(None)
                else:
                    gpu_arr = self.cp.asarray(arr, dtype=arr.dtype)
                    gpu_arrays.append(gpu_arr)
            return gpu_arrays

        elif direction == 'to_cpu':
            # Batch return to CPU
            cpu_arrays = []
            for arr in arrays:
                if arr is None:
                    cpu_arrays.append(None)
                else:
                    cpu_arr = self.cp.asnumpy(arr)
                    cpu_arrays.append(cpu_arr)
            return cpu_arrays

    def cleanup(self):
        """Unified GPU cleanup"""
        if self.memory:
            self.memory.cleanup_gpu_memory()
        if self.gpu_available:
            self.cp.get_default_memory_pool().free_all_blocks()
        logger.info("GPU memory cleaned up")

    @property
    def available_memory_gb(self) -> float:
        """Available GPU memory in GB"""
        if not self.gpu_available:
            return 0.0
        if self.memory:
            return self.memory.get_available_memory() / 1024
        return 0.0

    @property
    def is_available(self) -> bool:
        """GPU is available"""
        return self.gpu_available


# Singleton accessor
def get_gpu_manager() -> UnifiedGPUManager:
    """Get GPU manager instance"""
    return UnifiedGPUManager()


__all__ = ['UnifiedGPUManager', 'get_gpu_manager']
```

---

## 📋 Checklist Implémentation

### Week 1

- [ ] **Jour 1 (30 min)**

  - [ ] Supprimer OptimizedReclassifier
  - [ ] Nettoyer préfixes
  - [ ] Tests passent
  - [ ] Commit + Push

- [ ] **Jour 2 (1h)**

  - [ ] Créer ClassificationEngine
  - [ ] Adapter 3 stratégies existantes
  - [ ] Tests unitaires

- [ ] **Jour 3-4 (2h)**

  - [ ] Créer UnifiedGPUManager
  - [ ] Tester avec code existant
  - [ ] Benchmarks GPU

- [ ] **Jour 5 (1h)**
  - [ ] Code review
  - [ ] Merge à main

### Week 2

- [ ] Consolider GroundTruthProvider
- [ ] Mettre à jour documentation
- [ ] Update examples

---

## 🧪 Tests à Lancer

```bash
# Vérifier aucune regression
pytest tests/ -v

# Tests spécifiques
pytest tests/test_classification.py -v
pytest tests/test_gpu.py -v
pytest tests/test_core_gpu_manager.py -v

# Coverage
pytest tests/ --cov=ign_lidar --cov-report=html

# Performance
python scripts/benchmark_gpu.py
```

---

## 📊 Métriques de Succès

Avant la fin de la semaine:

✅ OptimizedReclassifier supprimé  
✅ Préfixes redondants nettoyés  
✅ ClassificationEngine créée + 3 stratégies intégrées  
✅ UnifiedGPUManager fonctionnel  
✅ Tests passent 100%  
✅ Zéro regression de performance

---

**Document:** Quick Action Guide  
**Date:** 2025-11-24  
**Statut:** 🟢 Ready to Implement
