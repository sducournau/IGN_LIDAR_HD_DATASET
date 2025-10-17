"""
Modern Configuration Loader with Preset Support (Week 3)

This module provides a modern configuration loading system with:
- Base configuration inheritance
- Preset system (minimal, lod2, lod3, asprs, full)
- Custom config merging
- CLI override support
- Dot-notation parameter access

Author: Week 3 Team
Date: October 17, 2025
Version: 5.1.0
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import copy

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

logger = logging.getLogger(__name__)


class ConfigLoaderError(Exception):
    """Exception raised for configuration loading errors."""
    pass


class PresetConfigLoader:
    """
    Modern configuration loader with preset support.
    
    Loading order (inheritance chain):
    1. base.yaml (defaults)
    2. presets/{preset}.yaml (if specified)
    3. custom.yaml (if specified)
    4. CLI overrides (highest priority)
    
    Example:
        >>> loader = PresetConfigLoader()
        >>> 
        >>> # List available presets
        >>> presets = loader.list_presets()
        >>> print(presets)  # ['minimal', 'lod2', 'lod3', 'asprs', 'full']
        >>> 
        >>> # Load with preset only
        >>> config = loader.load(preset="lod2")
        >>> 
        >>> # Load with preset + custom config
        >>> config = loader.load(
        ...     preset="lod2",
        ...     config_file="my_config.yaml"
        ... )
        >>> 
        >>> # Load with preset + overrides
        >>> config = loader.load(
        ...     preset="lod2",
        ...     overrides={"processor.gpu_batch_size": 2000000}
        ... )
    """
    
    def __init__(self, verbose: bool = True, configs_dir: Optional[Path] = None):
        """
        Initialize the config loader.
        
        Args:
            verbose: Print loading information
            configs_dir: Custom configs directory (default: auto-detect)
        """
        self.verbose = verbose
        
        # Auto-detect configs directory
        if configs_dir is None:
            # Try to find configs directory relative to this file
            module_dir = Path(__file__).parent
            self.configs_dir = module_dir / "configs"
            
            # Fallback to package root
            if not self.configs_dir.exists():
                package_root = module_dir.parent
                self.configs_dir = package_root / "configs"
        else:
            self.configs_dir = Path(configs_dir)
        
        if not self.configs_dir.exists():
            raise ConfigLoaderError(
                f"Configs directory not found: {self.configs_dir}\n"
                f"Please specify configs_dir explicitly."
            )
        
        self.base_file = self.configs_dir / "base.yaml"
        self.presets_dir = self.configs_dir / "presets"
        
        # Check if base.yaml exists
        if not self.base_file.exists():
            raise ConfigLoaderError(
                f"Base configuration not found: {self.base_file}\n"
                f"Please create base.yaml with default settings."
            )
        
        if self.verbose:
            logger.info(f"✓ ConfigLoader initialized")
            logger.info(f"  Configs dir: {self.configs_dir}")
            logger.info(f"  Base config: {self.base_file}")
            logger.info(f"  Presets dir: {self.presets_dir}")
    
    def load(
        self,
        preset: Optional[str] = None,
        config_file: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load configuration with inheritance chain.
        
        Args:
            preset: Preset name (minimal, lod2, lod3, asprs, full)
            config_file: Custom config file path
            overrides: CLI overrides (dot-notation supported)
        
        Returns:
            Merged configuration dictionary
        
        Example:
            >>> config = loader.load(
            ...     preset="lod2",
            ...     config_file="versailles.yaml",
            ...     overrides={"processor.gpu_batch_size": 2000000}
            ... )
        """
        if not YAML_AVAILABLE:
            raise ConfigLoaderError(
                "PyYAML is required. Install with: pip install pyyaml"
            )
        
        if self.verbose:
            logger.info("=" * 80)
            logger.info("Loading Configuration")
            logger.info("=" * 80)
        
        # Step 1: Load base configuration
        config = self._load_yaml(self.base_file)
        if self.verbose:
            logger.info(f"✓ Loaded base config: {self.base_file}")
        
        # Step 2: Merge preset (if specified)
        if preset:
            preset_file = self.presets_dir / f"{preset}.yaml"
            if not preset_file.exists():
                raise ConfigLoaderError(
                    f"Preset '{preset}' not found: {preset_file}\n"
                    f"Available presets: {', '.join(self.list_presets())}"
                )
            
            preset_config = self._load_yaml(preset_file)
            config = self._deep_merge(config, preset_config)
            if self.verbose:
                logger.info(f"✓ Merged preset: {preset}")
        
        # Step 3: Merge custom config (if specified)
        if config_file:
            custom_path = Path(config_file)
            if not custom_path.exists():
                raise ConfigLoaderError(f"Config file not found: {custom_path}")
            
            custom_config = self._load_yaml(custom_path)
            config = self._deep_merge(config, custom_config)
            if self.verbose:
                logger.info(f"✓ Merged custom config: {config_file}")
        
        # Step 4: Apply CLI overrides (if specified)
        if overrides:
            config = self._apply_overrides(config, overrides)
            if self.verbose:
                logger.info(f"✓ Applied {len(overrides)} override(s)")
        
        # Step 5: Validate merged config
        self._validate(config)
        if self.verbose:
            logger.info("✓ Configuration validated")
            logger.info("=" * 80)
        
        return config
    
    def list_presets(self) -> List[str]:
        """
        List available presets.
        
        Returns:
            List of preset names (without .yaml extension)
        
        Example:
            >>> loader.list_presets()
            ['minimal', 'lod2', 'lod3', 'asprs', 'full']
        """
        if not self.presets_dir.exists():
            return []
        
        presets = []
        for file in self.presets_dir.glob("*.yaml"):
            presets.append(file.stem)
        
        return sorted(presets)
    
    def get_preset_info(self, preset: str) -> Dict[str, str]:
        """
        Get information about a specific preset.
        
        Args:
            preset: Preset name
        
        Returns:
            Dictionary with preset metadata (use_case, speed, features)
        
        Example:
            >>> info = loader.get_preset_info("lod2")
            >>> print(info['use_case'])
            'Building modeling, facade detection, roof planes'
        """
        preset_file = self.presets_dir / f"{preset}.yaml"
        if not preset_file.exists():
            raise ConfigLoaderError(f"Preset '{preset}' not found")
        
        # Parse comments from YAML file for metadata
        with open(preset_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        info = {
            'name': preset,
            'use_case': 'Unknown',
            'speed': 'Unknown',
            'features': 'Unknown'
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('# USE CASE:'):
                info['use_case'] = line.split(':', 1)[1].strip()
            elif line.startswith('# SPEED:'):
                info['speed'] = line.split(':', 1)[1].strip()
            elif line.startswith('# FEATURES:'):
                info['features'] = line.split(':', 1)[1].strip()
        
        return info
    
    def print_presets(self) -> None:
        """
        Print available presets with descriptions.
        
        Example:
            >>> loader.print_presets()
            Available Presets:
            ------------------
            minimal: Quick preview, testing, development
            lod2: Building modeling, facade detection, roof planes
            ...
        """
        presets = self.list_presets()
        
        print("\nAvailable Presets:")
        print("-" * 80)
        
        for preset in presets:
            try:
                info = self.get_preset_info(preset)
                print(f"\n{preset}:")
                print(f"  Use case: {info['use_case']}")
                print(f"  Speed: {info['speed']}")
            except Exception:
                print(f"\n{preset}: (info unavailable)")
        
        print("\n" + "-" * 80)
        print(f"\nUsage: loader.load(preset='<name>')")
        print(f"Example: loader.load(preset='lod2')\n")
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        Load YAML file and return dictionary.
        
        Args:
            file_path: Path to YAML file
        
        Returns:
            Dictionary from YAML
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Handle Hydra defaults directive
            if isinstance(data, dict) and 'defaults' in data:
                # Remove defaults from config (handled separately if needed)
                defaults = data.pop('defaults', None)
                if self.verbose and defaults:
                    logger.debug(f"  Found defaults directive: {defaults}")
            
            return data if isinstance(data, dict) else {}
            
        except yaml.YAMLError as e:
            raise ConfigLoaderError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise ConfigLoaderError(f"Error reading {file_path}: {e}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
        
        Returns:
            Merged dictionary (base + override)
        
        Note:
            - Nested dicts are merged recursively
            - Lists are replaced (not merged)
            - None values in override delete keys from base
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if value is None:
                # None means delete the key
                result.pop(key, None)
            elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Replace value (including lists)
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply CLI overrides using dot-notation.
        
        Args:
            config: Configuration dictionary
            overrides: Overrides (supports dot-notation keys)
        
        Returns:
            Configuration with overrides applied
        
        Example:
            >>> overrides = {
            ...     "processor.gpu_batch_size": 2000000,
            ...     "features.k_neighbors": 30
            ... }
            >>> config = loader._apply_overrides(config, overrides)
        """
        result = copy.deepcopy(config)
        
        for key, value in overrides.items():
            # Support dot-notation for nested keys
            if '.' in key:
                parts = key.split('.')
                current = result
                
                # Navigate to parent dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set value
                current[parts[-1]] = value
            else:
                # Top-level key
                result[key] = value
        
        return result
    
    def _validate(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary
        
        Raises:
            ConfigLoaderError: If validation fails
        """
        # Check for required top-level keys
        required_keys = ['processor', 'features', 'data_sources', 'output']
        missing = [key for key in required_keys if key not in config]
        
        if missing:
            raise ConfigLoaderError(
                f"Missing required configuration sections: {', '.join(missing)}"
            )
        
        # Basic type checking
        if not isinstance(config['processor'], dict):
            raise ConfigLoaderError("'processor' must be a dictionary")
        
        if not isinstance(config['features'], dict):
            raise ConfigLoaderError("'features' must be a dictionary")
        
        if not isinstance(config['data_sources'], dict):
            raise ConfigLoaderError("'data_sources' must be a dictionary")
        
        if not isinstance(config['output'], dict):
            raise ConfigLoaderError("'output' must be a dictionary")
        
        # Validate GPU settings
        if 'gpu_batch_size' in config['processor']:
            batch_size = config['processor']['gpu_batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ConfigLoaderError(
                    f"Invalid gpu_batch_size: {batch_size} (must be positive integer)"
                )
        
        # Validate feature mode
        if 'mode' in config['features']:
            mode = config['features']['mode']
            valid_modes = ['minimal', 'lod2', 'lod3', 'asprs_classes', 'full']
            if mode not in valid_modes:
                raise ConfigLoaderError(
                    f"Invalid feature mode: {mode}. "
                    f"Valid modes: {', '.join(valid_modes)}"
                )


def load_config_with_preset(
    preset: Optional[str] = None,
    config_file: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for loading configuration with preset.
    
    Args:
        preset: Preset name (minimal, lod2, lod3, asprs, full)
        config_file: Custom config file path
        overrides: CLI overrides
        verbose: Print loading information
    
    Returns:
        Merged configuration dictionary
    
    Example:
        >>> config = load_config_with_preset(preset="lod2")
        >>> 
        >>> config = load_config_with_preset(
        ...     preset="lod2",
        ...     config_file="versailles.yaml",
        ...     overrides={"processor.gpu_batch_size": 2000000}
        ... )
    """
    loader = PresetConfigLoader(verbose=verbose)
    return loader.load(preset=preset, config_file=config_file, overrides=overrides)


# Backward compatibility
ConfigLoader = PresetConfigLoader
