"""
Deprecation Wrappers for Phase 5 Unified Managers

This module provides backward-compatible wrappers for code migrating from
legacy GPU stream, performance monitoring, and configuration validation modules.

All warnings will be removed in v4.0.0.

Version: 3.6.1
Status: Production-ready with deprecation notices
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

# Lazy imports to avoid circular dependencies
def _get_gpu_stream_manager():
    """Get GPUStreamManager lazily."""
    from . import gpu_stream_manager
    return gpu_stream_manager.GPUStreamManager

def _get_performance_manager():
    """Get PerformanceManager lazily."""
    from . import performance_manager
    return performance_manager.PerformanceManager

def _get_config_validator():
    """Get ConfigValidator lazily."""
    from . import config_validator
    return config_validator.ConfigValidator


def _deprecated(old_name: str, new_name: str, version: str = "4.0.0") -> None:
    """Issue deprecation warning."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v{version}. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# ============================================================================
# GPU Stream Management Wrappers (Backward Compatibility)
# ============================================================================


class StreamManager:
    """
    DEPRECATED: Use GPUStreamManager directly instead.

    This is a backward-compatibility wrapper for legacy code.
    """

    def __init__(self, pool_size: int = 4):
        """Initialize legacy stream manager wrapper."""
        _deprecated("StreamManager", "GPUStreamManager")
        GPUStreamManager = _get_gpu_stream_manager()
        self._manager = GPUStreamManager(pool_size=pool_size)

    def create_stream(self, priority: int = 0) -> Any:
        """Create a new GPU stream."""
        return self._manager.create_stream(priority=priority)

    def get_stream(self) -> Any:
        """Get an available stream."""
        return self._manager.get_stream()

    def return_stream(self, stream: Any) -> None:
        """Return a stream to the pool."""
        self._manager.return_stream(stream)

    def synchronize(self) -> None:
        """Synchronize all streams."""
        self._manager.synchronize()

    def configure(self, **kwargs) -> None:
        """Configure stream manager."""
        self._manager.configure(**kwargs)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self._manager.get_performance_stats()

    def reset(self) -> None:
        """Reset all streams."""
        self._manager.reset()


def create_stream_manager(pool_size: int = 4) -> StreamManager:
    """
    DEPRECATED: Use GPUStreamManager directly.

    Create a legacy stream manager wrapper.
    """
    _deprecated("create_stream_manager()", "GPUStreamManager()")
    return StreamManager(pool_size=pool_size)


# ============================================================================
# Performance Monitoring Wrappers (Backward Compatibility)
# ============================================================================


class PerformanceTracker:
    """
    DEPRECATED: Use PerformanceManager directly instead.

    This is a backward-compatibility wrapper for legacy code.
    """

    def __init__(self):
        """Initialize legacy performance tracker wrapper."""
        _deprecated("PerformanceTracker", "PerformanceManager")
        PerformanceManager = _get_performance_manager()
        self._manager = PerformanceManager()

    def start_phase(self, phase_name: str) -> None:
        """Start tracking a phase."""
        self._manager.start_phase(phase_name)

    def end_phase(self, phase_name: str) -> None:
        """End tracking a phase."""
        self._manager.end_phase(phase_name)

    def record_operation(
        self, operation_name: str, duration_ms: float, metadata: Optional[Dict] = None
    ) -> None:
        """Record an operation."""
        self._manager.record_operation(
            operation_name=operation_name, duration_ms=duration_ms, metadata=metadata
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self._manager.get_summary()

    def get_phase_duration(self, phase_name: str) -> float:
        """Get phase duration."""
        summary = self._manager.get_summary()
        if phase_name in summary:
            return summary[phase_name].get("duration", 0.0)
        return 0.0

    def reset(self) -> None:
        """Reset all tracking."""
        self._manager.reset()

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics."""
        return self._manager.get_summary()


def create_performance_tracker() -> PerformanceTracker:
    """
    DEPRECATED: Use PerformanceManager directly.

    Create a legacy performance tracker wrapper.
    """
    _deprecated("create_performance_tracker()", "PerformanceManager()")
    return PerformanceTracker()


# ============================================================================
# Configuration Validation Wrappers (Backward Compatibility)
# ============================================================================


class ConfigurationValidator:
    """
    DEPRECATED: Use ConfigValidator directly instead.

    This is a backward-compatibility wrapper for legacy code.
    """

    def __init__(self):
        """Initialize legacy configuration validator wrapper."""
        _deprecated("ConfigurationValidator", "ConfigValidator")
        ConfigValidator = _get_config_validator()
        self._validator = ConfigValidator()

    def add_rule(
        self,
        rule_name: str,
        condition: Callable,
        error_message: str,
        category: str = "general",
    ) -> None:
        """Add a validation rule."""
        self._validator.add_rule(
            rule_name=rule_name,
            condition=condition,
            error_message=error_message,
            category=category,
        )

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a configuration."""
        return self._validator.validate(config)

    def validate_strict(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration strictly."""
        is_valid, errors = self._validator.validate(config)
        if not is_valid:
            raise ValueError(f"Configuration validation failed: {errors}")
        return config

    def clear_rules(self) -> Tuple[bool, List[str]]:
        """Clear all rules."""
        return self._validator.clear_rules()

    def get_rule_count(self) -> int:
        """Get number of rules."""
        rules = self._validator._get_internal_rules()
        return len(rules) if rules else 0

    def get_categories(self) -> List[str]:
        """Get all rule categories."""
        return self._validator.get_categories()


def create_configuration_validator() -> ConfigurationValidator:
    """
    DEPRECATED: Use ConfigValidator directly.

    Create a legacy configuration validator wrapper.
    """
    _deprecated("create_configuration_validator()", "ConfigValidator()")
    return ConfigurationValidator()


# ============================================================================
# Legacy Module Imports (Backward Compatibility)
# ============================================================================


def get_legacy_stream_manager(pool_size: int = 4) -> StreamManager:
    """
    DEPRECATED: Use GPUStreamManager directly.

    Get a legacy stream manager for old code.

    Args:
        pool_size: Number of streams in pool

    Returns:
        StreamManager: Wrapped GPUStreamManager

    Example:
        >>> # Old code
        >>> manager = get_legacy_stream_manager(pool_size=8)
        >>>
        >>> # New code
        >>> from ign_lidar.core import GPUStreamManager
        >>> manager = GPUStreamManager(pool_size=8)
    """
    _deprecated("get_legacy_stream_manager()", "GPUStreamManager()")
    return StreamManager(pool_size=pool_size)


def get_legacy_performance_tracker() -> PerformanceTracker:
    """
    DEPRECATED: Use PerformanceManager directly.

    Get a legacy performance tracker for old code.

    Returns:
        PerformanceTracker: Wrapped PerformanceManager

    Example:
        >>> # Old code
        >>> tracker = get_legacy_performance_tracker()
        >>>
        >>> # New code
        >>> from ign_lidar.core import PerformanceManager
        >>> tracker = PerformanceManager()
    """
    _deprecated("get_legacy_performance_tracker()", "PerformanceManager()")
    return PerformanceTracker()


def get_legacy_config_validator() -> ConfigurationValidator:
    """
    DEPRECATED: Use ConfigValidator directly.

    Get a legacy configuration validator for old code.

    Returns:
        ConfigurationValidator: Wrapped ConfigValidator

    Example:
        >>> # Old code
        >>> validator = get_legacy_config_validator()
        >>>
        >>> # New code
        >>> from ign_lidar.core import ConfigValidator
        >>> validator = ConfigValidator()
    """
    _deprecated("get_legacy_config_validator()", "ConfigValidator()")
    return ConfigurationValidator()


__all__ = [
    # Legacy wrappers
    "StreamManager",
    "PerformanceTracker",
    "ConfigurationValidator",
    # Factory functions
    "create_stream_manager",
    "create_performance_tracker",
    "create_configuration_validator",
    "get_legacy_stream_manager",
    "get_legacy_performance_tracker",
    "get_legacy_config_validator",
]
