"""
Configuration Management Improvements - Demo Implementation

This file demonstrates the recommended improvements to the config system.
These are examples that can be integrated into the main codebase.

Author: Config Audit Team
Date: October 16, 2025
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Enhanced Dataclass with Validation
# ============================================================================

@dataclass
class EnhancedProcessorConfig:
    """
    Enhanced processor configuration with comprehensive validation.
    
    This demonstrates best practices for config dataclasses:
    - Type annotations for all fields
    - Sensible defaults
    - __post_init__ validation
    - Cross-field validation
    - Clear documentation
    """
    
    # Core settings
    lod_level: str = "LOD2"  # Use str instead of Literal for OmegaConf compatibility
    use_gpu: bool = False
    num_workers: int = 4
    
    # Patch settings
    patch_size: float = 150.0
    patch_overlap: float = 0.1
    num_points: int = 16384
    
    # Performance settings
    batch_size: int = 32
    prefetch_factor: int = 2
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_lod_level()
        self._validate_patch_config()
        self._validate_performance()
        
    def _validate_lod_level(self):
        """Validate LOD level."""
        valid_levels = ["LOD2", "LOD3"]
        if self.lod_level not in valid_levels:
            raise ValueError(
                f"Invalid LOD level: {self.lod_level}. "
                f"Must be one of: {valid_levels}"
            )
    
    def _validate_patch_config(self):
        """Validate patch configuration."""
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        
        if not 0 <= self.patch_overlap < 1:
            raise ValueError(
                f"patch_overlap must be in [0, 1), got {self.patch_overlap}"
            )
        
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {self.num_points}")
    
    def _validate_performance(self):
        """Validate performance settings."""
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


# ============================================================================
# 2. Config Merger with Deep Merge
# ============================================================================

class ConfigMerger:
    """
    Intelligent configuration merger with validation.
    
    Provides deep merge capabilities with validation and type safety.
    """
    
    @staticmethod
    def merge(base: DictConfig, *overrides: DictConfig, 
              validate: bool = True) -> DictConfig:
        """
        Deep merge configurations with validation.
        
        Args:
            base: Base configuration
            *overrides: Override configurations (applied in order)
            validate: Whether to validate after merge
            
        Returns:
            Merged configuration
            
        Example:
            >>> base = OmegaConf.create({"a": 1, "b": {"c": 2}})
            >>> override = OmegaConf.create({"b": {"d": 3}})
            >>> result = ConfigMerger.merge(base, override)
            >>> # result = {"a": 1, "b": {"c": 2, "d": 3}}
        """
        # Use OmegaConf.merge for deep merging
        result = OmegaConf.merge(base, *overrides)
        
        if validate:
            # Re-validate after merge
            if hasattr(result, 'validate'):
                try:
                    warnings = result.validate()
                    for w in warnings:
                        logger.warning(f"Config warning: {w}")
                except Exception as e:
                    logger.error(f"Config validation failed: {e}")
                    raise
        
        return result
    
    @staticmethod
    def selective_merge(base: DictConfig, override: DictConfig,
                       keys: List[str]) -> DictConfig:
        """
        Merge only specific keys from override into base.
        
        Args:
            base: Base configuration
            override: Override configuration
            keys: List of keys to merge (supports dot notation)
            
        Returns:
            Selectively merged configuration
            
        Example:
            >>> base = OmegaConf.create({"a": 1, "b": 2, "c": 3})
            >>> override = OmegaConf.create({"a": 10, "b": 20})
            >>> result = ConfigMerger.selective_merge(base, override, ["a"])
            >>> # result = {"a": 10, "b": 2, "c": 3}
        """
        # Create a copy of base
        result = OmegaConf.create(OmegaConf.to_container(base))
        
        # Merge only specified keys
        for key in keys:
            value = OmegaConf.select(override, key)
            if value is not None:
                OmegaConf.update(result, key, value, merge=True)
        
        return result
    
    @staticmethod
    def merge_with_priority(configs: Dict[str, DictConfig], 
                           priority_order: List[str]) -> DictConfig:
        """
        Merge multiple configs with explicit priority.
        
        Args:
            configs: Dictionary of named configurations
            priority_order: List of config names in priority order (last wins)
            
        Returns:
            Merged configuration
            
        Example:
            >>> configs = {
            ...     "defaults": OmegaConf.create({"a": 1, "b": 2}),
            ...     "preset": OmegaConf.create({"b": 3}),
            ...     "user": OmegaConf.create({"c": 4})
            ... }
            >>> result = ConfigMerger.merge_with_priority(
            ...     configs, ["defaults", "preset", "user"]
            ... )
            >>> # result = {"a": 1, "b": 3, "c": 4}
        """
        if not priority_order:
            raise ValueError("priority_order cannot be empty")
        
        # Start with first config
        result = OmegaConf.create(
            OmegaConf.to_container(configs[priority_order[0]])
        )
        
        # Merge remaining in priority order
        for name in priority_order[1:]:
            if name not in configs:
                raise KeyError(f"Config '{name}' not found in configs dict")
            result = OmegaConf.merge(result, configs[name])
        
        return result


# ============================================================================
# 3. Config Registry Pattern
# ============================================================================

class ConfigRegistry:
    """
    Central registry for configuration access.
    
    Provides singleton access to configuration with type safety.
    """
    
    _instance = None
    _config: Optional[DictConfig] = None
    
    @classmethod
    def initialize(cls, config: DictConfig, validate: bool = True):
        """
        Initialize registry with configuration.
        
        Args:
            config: Configuration to register
            validate: Whether to validate configuration
        """
        if validate and hasattr(config, 'validate'):
            config.validate()
        
        cls._config = config
        logger.info("ConfigRegistry initialized")
    
    @classmethod
    def get(cls, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path.
        
        Args:
            path: Dot-separated path (e.g., "processor.use_gpu")
            default: Default value if path not found
            
        Returns:
            Configuration value
            
        Example:
            >>> ConfigRegistry.initialize(config)
            >>> use_gpu = ConfigRegistry.get("processor.use_gpu", default=False)
        """
        if cls._config is None:
            raise RuntimeError(
                "ConfigRegistry not initialized. "
                "Call ConfigRegistry.initialize(config) first."
            )
        
        return OmegaConf.select(cls._config, path, default=default)
    
    @classmethod
    def get_section(cls, section: str) -> DictConfig:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "processor", "features")
            
        Returns:
            Configuration section
        """
        value = cls.get(section)
        if value is None:
            raise KeyError(f"Configuration section '{section}' not found")
        return value
    
    @classmethod
    def override(cls, path: str, value: Any, validate: bool = True):
        """
        Override configuration value at runtime.
        
        Args:
            path: Dot-separated path
            value: New value
            validate: Whether to re-validate after override
        """
        if cls._config is None:
            raise RuntimeError("ConfigRegistry not initialized")
        
        OmegaConf.update(cls._config, path, value)
        
        if validate and hasattr(cls._config, 'validate'):
            cls._config.validate()
        
        logger.info(f"Config override: {path} = {value}")
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to plain dictionary."""
        if cls._config is None:
            raise RuntimeError("ConfigRegistry not initialized")
        return OmegaConf.to_container(cls._config, resolve=True)


# ============================================================================
# 4. Enhanced Validator
# ============================================================================

class ConfigValidator:
    """
    Comprehensive configuration validator.
    
    Validates:
    - Field values
    - Cross-field dependencies
    - Resource constraints
    - Data source compatibility
    """
    
    @staticmethod
    def validate_full(config: DictConfig) -> List[str]:
        """
        Comprehensive validation.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of warning messages (raises on errors)
        """
        warnings = []
        
        # Field-level validation
        ConfigValidator._validate_fields(config)
        
        # Cross-field dependencies
        warnings.extend(ConfigValidator._validate_dependencies(config))
        
        # Resource constraints
        warnings.extend(ConfigValidator._validate_resources(config))
        
        return warnings
    
    @staticmethod
    def _validate_fields(config: DictConfig):
        """Validate individual fields."""
        # Call __post_init__ if structured config
        if hasattr(config, '__post_init__'):
            config.__post_init__()
    
    @staticmethod
    def _validate_dependencies(config: DictConfig) -> List[str]:
        """Validate cross-field dependencies."""
        warnings = []
        
        # NDVI requires infrared
        if OmegaConf.select(config, "features.compute_ndvi", default=False):
            if not OmegaConf.select(config, "features.use_infrared", default=False):
                raise ValueError(
                    "features.compute_ndvi=True requires "
                    "features.use_infrared=True"
                )
        
        # GPU requires CuPy
        if OmegaConf.select(config, "processor.use_gpu", default=False):
            try:
                import cupy
            except ImportError:
                warnings.append(
                    "processor.use_gpu=True but CuPy not installed. "
                    "Install with: pip install cupy-cuda12x"
                )
        
        # Stitching buffer validation
        if OmegaConf.select(config, "stitching.enabled", default=False):
            buffer_size = OmegaConf.select(config, "stitching.buffer_size", default=10.0)
            k_neighbors = OmegaConf.select(config, "features.k_neighbors", default=20)
            
            # Estimate minimum buffer (rough heuristic)
            min_buffer = k_neighbors * 0.5 * 2
            if buffer_size < min_buffer:
                warnings.append(
                    f"stitching.buffer_size ({buffer_size}m) < "
                    f"recommended minimum ({min_buffer:.1f}m) based on "
                    f"features.k_neighbors ({k_neighbors})"
                )
        
        return warnings
    
    @staticmethod
    def _validate_resources(config: DictConfig) -> List[str]:
        """Validate resource constraints."""
        warnings = []
        
        # Check worker count
        import os
        max_workers = os.cpu_count() or 4
        num_workers = OmegaConf.select(config, "processor.num_workers", default=4)
        
        if num_workers > max_workers:
            warnings.append(
                f"processor.num_workers ({num_workers}) > "
                f"available CPU cores ({max_workers})"
            )
        
        # Check GPU memory (if GPU enabled)
        if OmegaConf.select(config, "processor.use_gpu", default=False):
            try:
                import cupy as cp
                free_mem_bytes, total_mem_bytes = cp.cuda.Device().mem_info
                free_mem_gb = free_mem_bytes / 1e9
                
                # Estimate memory usage (rough)
                num_points = OmegaConf.select(config, "processor.num_points", default=16384)
                batch_size = OmegaConf.select(config, "processor.batch_size", default=32)
                
                # Assume ~100 bytes per point (features + coordinates)
                estimated_mem_gb = (num_points * batch_size * 100) / 1e9
                
                if estimated_mem_gb > free_mem_gb * 0.8:
                    warnings.append(
                        f"Estimated GPU memory usage ({estimated_mem_gb:.1f}GB) "
                        f"exceeds 80% of free memory ({free_mem_gb:.1f}GB). "
                        f"Consider reducing batch_size or num_points."
                    )
            except:
                pass
        
        return warnings


# ============================================================================
# 5. Usage Examples
# ============================================================================

def example_basic_usage():
    """Example: Basic configuration loading and validation."""
    from omegaconf import OmegaConf
    
    # Create base configuration
    base_config = OmegaConf.structured(EnhancedProcessorConfig)
    
    # Load YAML config
    yaml_config = OmegaConf.load("configs/processing_config.yaml")
    
    # Merge with validation
    config = ConfigMerger.merge(base_config, yaml_config, validate=True)
    
    # Initialize registry
    ConfigRegistry.initialize(config)
    
    # Access configuration
    use_gpu = ConfigRegistry.get("use_gpu", default=False)
    print(f"GPU enabled: {use_gpu}")


def example_selective_merge():
    """Example: Selective configuration merge."""
    from omegaconf import OmegaConf
    
    # Base config
    base = OmegaConf.create({
        "processor": {
            "use_gpu": False,
            "num_workers": 4,
            "batch_size": 32
        },
        "features": {
            "k_neighbors": 20
        }
    })
    
    # Override only GPU settings
    override = OmegaConf.create({
        "processor": {
            "use_gpu": True,
            "batch_size": 64  # Don't change num_workers
        }
    })
    
    # Selective merge - only update GPU settings
    result = ConfigMerger.selective_merge(
        base, override, 
        keys=["processor.use_gpu", "processor.batch_size"]
    )
    
    print(f"GPU: {result.processor.use_gpu}")  # True
    print(f"Workers: {result.processor.num_workers}")  # 4 (unchanged)
    print(f"Batch size: {result.processor.batch_size}")  # 64


def example_priority_merge():
    """Example: Merge with explicit priority."""
    from omegaconf import OmegaConf
    
    configs = {
        "defaults": OmegaConf.create({
            "processor": {"use_gpu": False, "batch_size": 32},
            "features": {"k_neighbors": 20}
        }),
        "gpu_preset": OmegaConf.create({
            "processor": {"use_gpu": True, "batch_size": 64}
        }),
        "user_override": OmegaConf.create({
            "features": {"k_neighbors": 30}
        })
    }
    
    # Merge in priority order: defaults < preset < user
    result = ConfigMerger.merge_with_priority(
        configs, 
        ["defaults", "gpu_preset", "user_override"]
    )
    
    print(f"GPU: {result.processor.use_gpu}")  # True (from preset)
    print(f"Batch: {result.processor.batch_size}")  # 64 (from preset)
    print(f"K-neighbors: {result.features.k_neighbors}")  # 30 (from user)


def example_validation():
    """Example: Configuration validation."""
    from omegaconf import OmegaConf
    
    # Valid config
    valid_config = OmegaConf.structured(EnhancedProcessorConfig(
        lod_level="LOD2",
        use_gpu=False,
        patch_size=150.0
    ))
    
    # This will succeed
    warnings = ConfigValidator.validate_full(valid_config)
    print(f"Validation warnings: {len(warnings)}")
    
    # Invalid config
    try:
        invalid_config = EnhancedProcessorConfig(
            lod_level="LOD5",  # Invalid!
            patch_size=-10.0   # Invalid!
        )
    except ValueError as e:
        print(f"Validation error: {e}")


if __name__ == "__main__":
    print("Configuration Improvements Demo")
    print("=" * 70)
    
    print("\n1. Basic Usage")
    print("-" * 70)
    # example_basic_usage()  # Uncomment to run
    
    print("\n2. Selective Merge")
    print("-" * 70)
    example_selective_merge()
    
    print("\n3. Priority Merge")
    print("-" * 70)
    example_priority_merge()
    
    print("\n4. Validation")
    print("-" * 70)
    example_validation()
