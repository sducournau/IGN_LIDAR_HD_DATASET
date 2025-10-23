"""
Rule-based classification module.

This module provides a comprehensive framework for rule-based point cloud classification:
- Abstract base classes for creating custom rules
- Multiple rule types (geometric, spectral, grammar-based)
- Hierarchical rule execution with configurable strategies
- Confidence scoring and combination methods
- Feature validation utilities

Quick Start:
    from ign_lidar.core.classification.rules import (
        GeometricRule,
        SpectralRule,
        GrammarRule,
        HierarchicalRuleEngine
    )

Architecture:
    rules/
    ├── base.py          - Abstract base classes and data structures
    ├── validation.py    - Feature validation utilities
    ├── confidence.py    - Confidence scoring methods
    ├── hierarchy.py     - Hierarchical execution
    ├── geometric.py     - Geometric rules (height, planarity, etc.)
    ├── spectral.py      - Spectral rules (intensity, RGB, NDVI)
    └── grammar.py       - Grammar-based rules (shape patterns)

Version: 3.2.0
Status: Phase 4B Infrastructure Complete
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ================================
# Core Base Classes and Enums
# ================================

from .base import (
    # Enumerations
    RuleType,
    RulePriority,
    ExecutionStrategy,
    ConflictResolution,
    
    # Data classes
    RuleStats,
    RuleResult,
    RuleConfig,
    RuleEngineConfig,
    
    # Abstract base classes
    BaseRule,
    RuleEngine,
    
    # Utility functions
    create_empty_result,
    merge_rule_results,
)

# ================================
# Validation Utilities
# ================================

from .validation import (
    FeatureRequirements,
    validate_features,
    validate_feature_shape,
    check_feature_quality,
    check_all_feature_quality,
    validate_feature_ranges,
    validate_points_array,
    get_feature_statistics,
)

# ================================
# Confidence Utilities
# ================================

from .confidence import (
    ConfidenceMethod,
    CombinationMethod,
    calculate_confidence,
    combine_confidences,
    normalize_confidence,
    calibrate_confidence,
    apply_confidence_threshold,
)

# ================================
# Hierarchical Execution
# ================================

from .hierarchy import (
    RuleLevel,
    HierarchicalRuleEngine,
)

# ================================
# Legacy Engine Adapters
# ================================

from .adapters import (
    LegacyEngineAdapter,
    MultiClassAdapter,
)

from .spectral_adapter import (
    SpectralRulesAdapter,
    create_spectral_vegetation_rule,
    create_spectral_water_rule,
)

from .geometric_adapter import (
    GeometricRulesAdapter,
    create_geometric_building_rule,
    create_geometric_road_rule,
)


# ================================
# Public API
# ================================

__all__ = [
    # Enums
    'RuleType',
    'RulePriority',
    'ExecutionStrategy',
    'ConflictResolution',
    'ConfidenceMethod',
    'CombinationMethod',
    
    # Data classes
    'RuleStats',
    'RuleResult',
    'RuleConfig',
    'RuleEngineConfig',
    'FeatureRequirements',
    'RuleLevel',
    
    # Base classes
    'BaseRule',
    'RuleEngine',
    'HierarchicalRuleEngine',
    
    # Legacy Adapters
    'LegacyEngineAdapter',
    'MultiClassAdapter',
    'SpectralRulesAdapter',
    'GeometricRulesAdapter',
    
    # Adapter Factory Functions
    'create_spectral_vegetation_rule',
    'create_spectral_water_rule',
    'create_geometric_building_rule',
    'create_geometric_road_rule',
    
    # Validation
    'validate_features',
    'validate_feature_shape',
    'check_feature_quality',
    'check_all_feature_quality',
    'validate_feature_ranges',
    'validate_points_array',
    'get_feature_statistics',
    
    # Confidence
    'calculate_confidence',
    'combine_confidences',
    'normalize_confidence',
    'calibrate_confidence',
    'apply_confidence_threshold',
    
    # Utilities
    'create_empty_result',
    'merge_rule_results',
    'get_module_status',
]


# ================================
# Module Status
# ================================

def get_module_status() -> Dict[str, any]:
    """Get status and configuration of rules module
    
    Returns:
        Dictionary with module information
    """
    # Try to import optional dependencies
    optional_deps = {}
    
    try:
        import scipy
        optional_deps['scipy'] = scipy.__version__
    except ImportError:
        optional_deps['scipy'] = None
    
    try:
        import shapely
        optional_deps['shapely'] = shapely.__version__
    except ImportError:
        optional_deps['shapely'] = None
    
    try:
        from rtree import index
        optional_deps['rtree'] = 'available'
    except ImportError:
        optional_deps['rtree'] = None
    
    return {
        'version': '3.2.0',
        'phase': 'Phase 4B - Infrastructure Complete',
        'base_classes': [
            'BaseRule',
            'RuleEngine',
            'HierarchicalRuleEngine',
        ],
        'rule_types_available': [
            'geometric',  # To be migrated in Phase 4C
            'spectral',   # To be migrated in Phase 4C
            'grammar',    # To be migrated in Phase 4C
        ],
        'features': {
            'hierarchical_execution': True,
            'confidence_methods': len(ConfidenceMethod),
            'execution_strategies': len(ExecutionStrategy),
            'conflict_resolution': len(ConflictResolution),
        },
        'optional_dependencies': optional_deps,
        'infrastructure_complete': True,
        'migration_ready': True,
    }


# ================================
# Module Initialization
# ================================

logger.info("Rules module initialized (Phase 4B - Infrastructure)")
logger.debug(f"Module status: {get_module_status()}")
