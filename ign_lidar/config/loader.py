"""
Configuration Loading Utilities

Utilities for loading and validating YAML configuration files for the
unified classification system.

Author: Configuration Team
Date: October 15, 2025
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        ConfigError: If file not found, invalid YAML, or missing required sections
    """
    if not YAML_AVAILABLE:
        raise ConfigError(
            "PyYAML is not installed. Install it with: pip install pyyaml"
        )
    
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML syntax in {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error reading {config_path}: {e}")
    
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a dictionary, got {type(config)}")
    
    logger.info(f"✓ Loaded configuration from {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of warning messages (empty if no warnings)
        
    Raises:
        ConfigError: If critical validation errors found
    """
    warnings = []
    
    # Required top-level sections
    required_sections = ['data_sources', 'classification', 'cache']
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required section: {section}")
    
    # Validate data_sources
    data_sources = config['data_sources']
    
    # BD TOPO
    if 'bd_topo' in data_sources:
        bd_topo = data_sources['bd_topo']
        if not isinstance(bd_topo.get('features', {}), dict):
            raise ConfigError("bd_topo.features must be a dictionary")
    
    # BD Forêt
    if 'bd_foret' in data_sources:
        bd_foret = data_sources['bd_foret']
        if bd_foret.get('enabled', True):
            if 'layer' not in bd_foret:
                warnings.append("bd_foret.layer not specified, using default")
    
    # RPG
    if 'rpg' in data_sources:
        rpg = data_sources['rpg']
        if rpg.get('enabled', True):
            year = rpg.get('year', 2023)
            if not (2020 <= year <= 2024):
                warnings.append(f"RPG year {year} may not be available (valid: 2020-2023)")
    
    # Cadastre
    if 'cadastre' in data_sources:
        cadastre = data_sources['cadastre']
        if cadastre.get('enabled', True):
            if not cadastre.get('group_by_parcel', False):
                warnings.append("Cadastre enabled but group_by_parcel=False (limited functionality)")
    
    # Validate classification
    classification = config['classification']
    
    if 'methods' in classification:
        methods = classification['methods']
        valid_methods = ['geometric', 'ndvi', 'ground_truth', 'forest', 'agriculture']
        for method in methods:
            if method not in valid_methods:
                warnings.append(f"Unknown classification method: {method}")
    
    # Validate priority order
    if 'priority_order' in classification:
        priority = classification['priority_order']
        if not isinstance(priority, list):
            raise ConfigError("classification.priority_order must be a list")
        
        # Check for duplicate priorities
        if len(priority) != len(set(priority)):
            warnings.append("Duplicate classes in priority_order")
    
    # Validate ASPRS codes (if present)
    if 'asprs_codes' in config:
        codes = config['asprs_codes']
        
        # Check for standard codes
        standard_codes = [1, 2, 3, 4, 5, 6, 9, 10, 11, 17]
        if 'standard' in codes:
            for code in standard_codes:
                if code not in codes['standard']:
                    warnings.append(f"Missing standard ASPRS code {code}")
        
        # Check extended codes
        if 'extended' in codes:
            extended = codes['extended']
            for code, name in extended.items():
                if not isinstance(code, int) or code < 40:
                    warnings.append(f"Extended code {code} should be >= 40")
    
    # Validate cache settings
    cache = config['cache']
    
    if 'ttl' in cache:
        ttl = cache['ttl']
        if not isinstance(ttl, dict):
            warnings.append("cache.ttl should be a dictionary with source-specific TTLs")
    
    # Validate output settings (if present)
    if 'output' in config:
        output = config['output']
        
        if 'formats' in output:
            valid_formats = ['laz', 'las', 'geojson', 'csv', 'hdf5']
            for fmt in output['formats']:
                if fmt not in valid_formats:
                    warnings.append(f"Unknown output format: {fmt}")
    
    # Validate batch processing (if present)
    if 'batch_processing' in config:
        batch = config['batch_processing']
        
        if 'parallel' in batch:
            if batch['parallel'].get('enabled', False):
                n_workers = batch['parallel'].get('n_workers', 1)
                if n_workers < 1:
                    raise ConfigError("n_workers must be >= 1")
                if n_workers > 32:
                    warnings.append(f"n_workers={n_workers} is very high, may cause issues")
    
    logger.info(f"✓ Configuration validated with {len(warnings)} warning(s)")
    return warnings


def get_data_source_config(config: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    Extract configuration for a specific data source.
    
    Args:
        config: Full configuration dictionary
        source: Data source name ('bd_topo', 'bd_foret', 'rpg', 'cadastre')
        
    Returns:
        Configuration dictionary for the specified source
        
    Raises:
        ConfigError: If source not found in configuration
    """
    if 'data_sources' not in config:
        raise ConfigError("Missing 'data_sources' section in configuration")
    
    data_sources = config['data_sources']
    
    if source not in data_sources:
        raise ConfigError(f"Data source '{source}' not found in configuration")
    
    return data_sources[source]


def get_classification_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract classification configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Classification configuration dictionary
        
    Raises:
        ConfigError: If classification section not found
    """
    if 'classification' not in config:
        raise ConfigError("Missing 'classification' section in configuration")
    
    return config['classification']


def get_cache_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract cache configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Cache configuration dictionary
        
    Raises:
        ConfigError: If cache section not found
    """
    if 'cache' not in config:
        raise ConfigError("Missing 'cache' section in configuration")
    
    return config['cache']


def apply_config_to_fetcher(
    fetcher_class,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None
):
    """
    Create a fetcher instance from configuration.
    
    Args:
        fetcher_class: Fetcher class to instantiate
        config: Configuration dictionary
        cache_dir: Override cache directory (optional)
        
    Returns:
        Instantiated fetcher with configuration applied
        
    Example:
        >>> from ign_lidar.io.cadastre import CadastreFetcher
        >>> config = load_config_from_yaml(Path("config.yaml"))
        >>> cadastre_config = get_data_source_config(config, 'cadastre')
        >>> fetcher = apply_config_to_fetcher(
        ...     CadastreFetcher,
        ...     cadastre_config,
        ...     cache_dir=Path("cache/cadastre")
        ... )
    """
    # Extract parameters for the fetcher
    kwargs = {}
    
    # Cache directory
    if cache_dir is not None:
        kwargs['cache_dir'] = cache_dir
    elif 'cache_dir' in config:
        kwargs['cache_dir'] = Path(config['cache_dir'])
    
    # WFS parameters
    if 'wfs_url' in config:
        kwargs['wfs_url'] = config['wfs_url']
    
    if 'layer' in config:
        kwargs['layer_name'] = config['layer']
    
    # Timeout
    if 'timeout' in config:
        kwargs['timeout'] = config['timeout']
    
    # Instantiate fetcher
    try:
        fetcher = fetcher_class(**kwargs)
        logger.info(f"✓ Created {fetcher_class.__name__} from configuration")
        return fetcher
    except Exception as e:
        raise ConfigError(f"Failed to create {fetcher_class.__name__}: {e}")


def create_fetcher_from_config(
    config: Dict[str, Any],
    cache_root: Optional[Path] = None
):
    """
    Create a DataFetcher from configuration.
    
    Args:
        config: Full configuration dictionary
        cache_root: Root cache directory (optional)
        
    Returns:
        Configured DataFetcher instance
        
    Example:
        >>> config = load_config_from_yaml(Path("config.yaml"))
        >>> fetcher = create_fetcher_from_config(
        ...     config,
        ...     cache_root=Path("cache")
        ... )
        >>> data = fetcher.fetch_all(bbox=my_bbox)
    """
    from ign_lidar.io.data_fetcher import DataFetcher, DataFetchConfig
    
    # Extract data source configuration
    data_sources = config.get('data_sources', {})
    
    # Build DataFetchConfig
    bd_topo = data_sources.get('bd_topo', {})
    bd_topo_features = bd_topo.get('features', {})
    
    bd_foret = data_sources.get('bd_foret', {})
    rpg = data_sources.get('rpg', {})
    cadastre = data_sources.get('cadastre', {})
    
    # Extract cache configuration
    cache_config = config.get('cache', {})
    if cache_root is None:
        cache_root = Path(cache_config.get('directory', 'cache'))
    
    fetch_config = DataFetchConfig(
        # BD TOPO features
        include_buildings=bd_topo_features.get('buildings', True),
        include_roads=bd_topo_features.get('roads', True),
        include_railways=bd_topo_features.get('railways', False),
        include_water=bd_topo_features.get('water', True),
        include_vegetation=bd_topo_features.get('vegetation', True),
        include_bridges=bd_topo_features.get('bridges', False),
        include_parking=bd_topo_features.get('parking', False),
        include_cemeteries=bd_topo_features.get('cemeteries', False),
        include_power_lines=bd_topo_features.get('power_lines', False),
        include_sports=bd_topo_features.get('sports', False),
        
        # Other sources
        include_forest=bd_foret.get('enabled', False),
        include_agriculture=rpg.get('enabled', False),
        rpg_year=rpg.get('year', 2023),
        
        include_cadastre=cadastre.get('enabled', False),
        group_by_parcel=cadastre.get('group_by_parcel', False)
    )
    
    fetcher = DataFetcher(
        cache_dir=cache_root,
        config=fetch_config
    )
    
    logger.info("✓ Created DataFetcher from configuration")
    return fetcher


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("Configuration Summary")
    print("=" * 80)
    
    # Data sources
    if 'data_sources' in config:
        print("\n📁 Data Sources:")
        data_sources = config['data_sources']
        
        if 'bd_topo' in data_sources:
            bd_topo = data_sources['bd_topo']
            features = bd_topo.get('features', {})
            enabled = [k for k, v in features.items() if v]
            print(f"  BD TOPO: {len(enabled)} features enabled")
            print(f"    {', '.join(enabled)}")
        
        if 'bd_foret' in data_sources:
            bd_foret = data_sources['bd_foret']
            if bd_foret.get('enabled', False):
                print(f"  BD Forêt: Enabled")
            else:
                print(f"  BD Forêt: Disabled")
        
        if 'rpg' in data_sources:
            rpg = data_sources['rpg']
            if rpg.get('enabled', False):
                year = rpg.get('year', 2023)
                print(f"  RPG: Enabled (year {year})")
            else:
                print(f"  RPG: Disabled")
        
        if 'cadastre' in data_sources:
            cadastre = data_sources['cadastre']
            if cadastre.get('enabled', False):
                group = cadastre.get('group_by_parcel', False)
                print(f"  Cadastre: Enabled (grouping: {group})")
            else:
                print(f"  Cadastre: Disabled")
    
    # Classification
    if 'classification' in config:
        print("\n🎯 Classification:")
        classification = config['classification']
        
        if 'methods' in classification:
            methods = classification['methods']
            print(f"  Methods: {', '.join(methods)}")
        
        if 'priority_order' in classification:
            priority = classification['priority_order']
            print(f"  Priority order: {len(priority)} levels")
    
    # Cache
    if 'cache' in config:
        print("\n💾 Cache:")
        cache = config['cache']
        
        cache_dir = cache.get('directory', 'cache')
        print(f"  Directory: {cache_dir}")
        
        if 'ttl' in cache:
            ttl = cache['ttl']
            print(f"  TTL configuration: {len(ttl)} sources")
    
    # Output
    if 'output' in config:
        print("\n📤 Output:")
        output = config['output']
        
        if 'formats' in output:
            formats = output['formats']
            print(f"  Formats: {', '.join(formats)}")
    
    print("=" * 80)


# Convenience function for quick setup
def quick_setup(config_path: Path, cache_dir: Optional[Path] = None):
    """
    Quick setup: Load, validate, and create fetcher from config file.
    
    Args:
        config_path: Path to YAML configuration file
        cache_dir: Optional cache directory override
        
    Returns:
        Tuple of (config dict, DataFetcher instance, warnings list)
        
    Example:
        >>> config, fetcher, warnings = quick_setup(Path("config.yaml"))
        >>> if warnings:
        ...     for w in warnings:
        ...         print(f"Warning: {w}")
        >>> data = fetcher.fetch_all(bbox=my_bbox)
    """
    # Load configuration
    config = load_config_from_yaml(config_path)
    
    # Validate
    warnings = validate_config(config)
    
    # Create fetcher
    fetcher = create_fetcher_from_config(config, cache_root=cache_dir)
    
    # Print summary
    print_config_summary(config)
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    
    return config, fetcher, warnings


# Deprecated aliases for backward compatibility
def create_unified_fetcher_from_config(*args, **kwargs):
    """Deprecated. Use create_fetcher_from_config instead."""
    import warnings
    warnings.warn(
        "create_unified_fetcher_from_config is deprecated, use create_fetcher_from_config",
        DeprecationWarning,
        stacklevel=2
    )
    return create_fetcher_from_config(*args, **kwargs)
