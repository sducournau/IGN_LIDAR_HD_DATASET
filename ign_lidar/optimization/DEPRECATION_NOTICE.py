"""
DEPRECATION NOTICE: Enhanced Optimization Modules

The "enhanced" optimization modules (enhanced_optimizer.py, enhanced_gpu.py, 
enhanced_cpu.py, enhanced_integration.py) have been consolidated into the base
optimization modules to reduce code duplication and maintenance burden.

All optimizations previously available in the "enhanced" modules are now
integrated directly into the main optimization classes:

- enhanced_gpu.py functionality → gpu.py
- enhanced_cpu.py functionality → strtree.py and vectorized.py  
- enhanced_optimizer.py functionality → auto_select.py
- enhanced_integration.py functionality → performance_monitor.py

MIGRATION GUIDE:

OLD (deprecated):
```python
from ign_lidar.optimization.enhanced_integration import EnhancedOptimizationManager
manager = EnhancedOptimizationManager()
```

NEW (recommended):
```python
from ign_lidar.optimization.auto_select import AutoOptimizer
optimizer = AutoOptimizer()
```

OLD (deprecated):
```python
from ign_lidar.optimization.enhanced_gpu import EnhancedGPUOptimizer
optimizer = EnhancedGPUOptimizer()
```

NEW (recommended):
```python
from ign_lidar.optimization.gpu import GPUGroundTruthClassifier
optimizer = GPUGroundTruthClassifier()
```

These deprecated modules will be removed in version 3.0.0.
"""

import warnings

warnings.warn(
    "Enhanced optimization modules are deprecated and will be removed in version 3.0.0. "
    "Use the consolidated optimization modules in ign_lidar.optimization instead.",
    DeprecationWarning,
    stacklevel=2
)