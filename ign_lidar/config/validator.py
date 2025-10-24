"""
Configuration Schema Validator

Validates configuration at load time to catch errors early, not at runtime.
Provides clear error messages and suggestions for fixing configuration issues.

Usage:
    from ign_lidar.config.validator import ConfigSchemaValidator
    
    # Validate config dict
    errors = ConfigSchemaValidator.validate(config_dict)
    
    # Validate config file
    from ign_lidar.config.validator import validate_config_file
    validate_config_file("path/to/config.yaml", strict=True)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ConfigSchemaValidator:
    """Validates configuration against complete schema."""

    # Required top-level sections
    REQUIRED_SECTIONS = [
        'processor',
        'features',
        'data_sources',
        'ground_truth',
        'output',
        'logging',
        'optimizations',
        'validation',
        'hardware'
    ]

    # Required processor keys
    REQUIRED_PROCESSOR = [
        'lod_level',
        'processing_mode',
        'use_gpu',
        'num_workers',
        'ground_truth_method',
        'reclassification',
        'preprocess',
        'stitching',
        'output_format'
    ]

    # Required features keys
    REQUIRED_FEATURES = [
        'mode',
        'k_neighbors',
        'search_radius',
        'compute_normals',
        'compute_height',
        'use_gpu'
    ]

    # Valid enum values
    VALID_LOD_LEVELS = ['ASPRS', 'LOD2', 'LOD3']
    VALID_PROCESSING_MODES = ['patches_only', 'both', 'enriched_only', 'reclassify_only']
    VALID_FEATURE_MODES = ['minimal', 'lod2', 'lod3', 'asprs_classes', 'full']
    VALID_OUTPUT_FORMATS = ['laz', 'las']
    VALID_HEIGHT_METHODS = ['dtm_only', 'local_only', 'hybrid']
    VALID_GROUND_TRUTH_METHODS = ['auto', 'gpu', 'strtree', 'rtree', 'gpu_chunked']

    @classmethod
    def validate(cls, config: Dict[str, Any], strict: bool = True, 
                 allow_partial: bool = False) -> List[str]:
        """
        Validate configuration schema.

        Args:
            config: Configuration dictionary
            strict: If True, raise exception on error. If False, return warnings.
            allow_partial: If True, allow partial configs (for Hydra composition)

        Returns:
            List of validation warnings/errors

        Raises:
            ValueError: If strict=True and validation fails
        """
        errors = []

        # Check required sections (unless partial config)
        if not allow_partial:
            for section in cls.REQUIRED_SECTIONS:
                if section not in config:
                    errors.append(f"Missing required section: '{section}'")

        # Check processor keys
        if 'processor' in config:
            processor_errors = cls._validate_processor(config['processor'])
            errors.extend(processor_errors)

        # Check features keys
        if 'features' in config:
            features_errors = cls._validate_features(config['features'])
            errors.extend(features_errors)

        # Validate data sources
        if 'data_sources' in config:
            ds_errors = cls._validate_data_sources(config['data_sources'])
            errors.extend(ds_errors)

        # Validate output
        if 'output' in config:
            output_errors = cls._validate_output(config['output'])
            errors.extend(output_errors)

        if errors:
            error_msg = "\n".join(f"  - {e}" for e in errors)
            if strict:
                raise ValueError(f"Configuration validation failed:\n{error_msg}")
            else:
                logger.warning(f"Configuration warnings:\n{error_msg}")

        return errors

    @classmethod
    def _validate_processor(cls, processor: Dict[str, Any]) -> List[str]:
        """Validate processor configuration."""
        errors = []

        # Check required keys
        for key in cls.REQUIRED_PROCESSOR:
            if key not in processor:
                errors.append(f"Missing required key: 'processor.{key}'")

        # Validate lod_level enum
        if 'lod_level' in processor:
            lod = processor['lod_level']
            if lod not in cls.VALID_LOD_LEVELS:
                errors.append(
                    f"Invalid processor.lod_level: '{lod}'. "
                    f"Must be one of {cls.VALID_LOD_LEVELS}"
                )

        # Validate processing_mode enum
        if 'processing_mode' in processor:
            mode = processor['processing_mode']
            if mode not in cls.VALID_PROCESSING_MODES:
                errors.append(
                    f"Invalid processor.processing_mode: '{mode}'. "
                    f"Must be one of {cls.VALID_PROCESSING_MODES}"
                )

        # Validate ground_truth_method enum
        if 'ground_truth_method' in processor:
            method = processor['ground_truth_method']
            if method not in cls.VALID_GROUND_TRUTH_METHODS:
                errors.append(
                    f"Invalid processor.ground_truth_method: '{method}'. "
                    f"Must be one of {cls.VALID_GROUND_TRUTH_METHODS}"
                )

        # Validate output_format enum
        if 'output_format' in processor:
            fmt = processor['output_format']
            if fmt not in cls.VALID_OUTPUT_FORMATS:
                errors.append(
                    f"Invalid processor.output_format: '{fmt}'. "
                    f"Must be one of {cls.VALID_OUTPUT_FORMATS}"
                )

        # Validate GPU settings
        if 'use_gpu' in processor and processor['use_gpu']:
            gpu_errors = cls._validate_gpu_settings(processor)
            errors.extend(gpu_errors)

        # Validate numeric ranges
        if 'gpu_memory_target' in processor:
            target = processor['gpu_memory_target']
            if not (0.0 < target <= 1.0):
                errors.append(
                    f"processor.gpu_memory_target must be in range (0.0, 1.0], "
                    f"got {target}"
                )

        if 'num_workers' in processor:
            workers = processor['num_workers']
            if workers < 1:
                errors.append(f"processor.num_workers must be >= 1, got {workers}")

        return errors

    @classmethod
    def _validate_gpu_settings(cls, processor: Dict[str, Any]) -> List[str]:
        """Validate GPU-specific settings."""
        errors = []

        if 'gpu_batch_size' not in processor:
            errors.append("processor.gpu_batch_size required when use_gpu=true")
        elif processor['gpu_batch_size'] < 1000:
            errors.append(
                f"processor.gpu_batch_size too small: {processor['gpu_batch_size']}. "
                f"Should be >= 1000"
            )

        if 'gpu_memory_target' not in processor:
            errors.append("processor.gpu_memory_target required when use_gpu=true")

        return errors

    @classmethod
    def _validate_features(cls, features: Dict[str, Any]) -> List[str]:
        """Validate features configuration."""
        errors = []

        # Check required keys
        for key in cls.REQUIRED_FEATURES:
            if key not in features:
                errors.append(f"Missing required key: 'features.{key}'")

        # Validate mode enum
        if 'mode' in features:
            mode = features['mode']
            if mode not in cls.VALID_FEATURE_MODES:
                errors.append(
                    f"Invalid features.mode: '{mode}'. "
                    f"Must be one of {cls.VALID_FEATURE_MODES}"
                )

        # Validate height_method enum
        if 'height_method' in features:
            method = features['height_method']
            if method not in cls.VALID_HEIGHT_METHODS:
                errors.append(
                    f"Invalid features.height_method: '{method}'. "
                    f"Must be one of {cls.VALID_HEIGHT_METHODS}"
                )

        # Validate k_neighbors
        if 'k_neighbors' in features:
            k = features['k_neighbors']
            if k < 1:
                errors.append(f"features.k_neighbors must be >= 1, got {k}")
            elif k > 200:
                errors.append(
                    f"features.k_neighbors very high: {k}. "
                    f"Consider using <= 100 for performance"
                )

        # Validate search_radius
        if 'search_radius' in features:
            radius = features['search_radius']
            if radius <= 0:
                errors.append(
                    f"features.search_radius must be > 0, got {radius}"
                )

        # GPU feature settings
        if 'use_gpu' in features and features['use_gpu']:
            if 'gpu_batch_size' not in features:
                errors.append("features.gpu_batch_size required when use_gpu=true")

        return errors

    @classmethod
    def _validate_data_sources(cls, data_sources: Dict[str, Any]) -> List[str]:
        """Validate data sources configuration."""
        errors = []

        # Check BD TOPO structure
        if 'bd_topo' in data_sources:
            bd_topo = data_sources['bd_topo']
            if isinstance(bd_topo, dict):
                if 'enabled' not in bd_topo:
                    errors.append("data_sources.bd_topo.enabled is required")
                
                # Check features if enabled
                if bd_topo.get('enabled') and 'features' in bd_topo:
                    features = bd_topo['features']
                    if not isinstance(features, dict):
                        errors.append(
                            "data_sources.bd_topo.features must be a dictionary"
                        )

        # Check RGE ALTI settings
        if 'rge_alti' in data_sources:
            rge_alti = data_sources['rge_alti']
            if isinstance(rge_alti, dict):
                if 'enabled' not in rge_alti:
                    errors.append("data_sources.rge_alti.enabled is required")
                
                # Validate augmentation settings
                if rge_alti.get('augment_ground_points'):
                    if 'augmentation_spacing' in rge_alti:
                        spacing = rge_alti['augmentation_spacing']
                        if spacing <= 0:
                            errors.append(
                                f"data_sources.rge_alti.augmentation_spacing must be > 0, "
                                f"got {spacing}"
                            )

        return errors

    @classmethod
    def _validate_output(cls, output: Dict[str, Any]) -> List[str]:
        """Validate output configuration."""
        errors = []

        if 'format' in output:
            fmt = output['format']
            if fmt not in cls.VALID_OUTPUT_FORMATS:
                errors.append(
                    f"Invalid output.format: '{fmt}'. "
                    f"Must be one of {cls.VALID_OUTPUT_FORMATS}"
                )

        return errors

    @classmethod
    def get_suggestions(cls, error_msg: str) -> Optional[str]:
        """Get suggestions for fixing common configuration errors."""
        suggestions = {
            "Missing required section: 'preprocess'": (
                "Add preprocess section:\n"
                "  preprocess:\n"
                "    enabled: false\n"
                "    remove_duplicates: true"
            ),
            "Missing required section: 'stitching'": (
                "Add stitching section:\n"
                "  stitching:\n"
                "    enabled: false\n"
                "    buffer_size: 10.0"
            ),
            "processor.gpu_batch_size required": (
                "Add GPU batch size (e.g., for RTX 4080):\n"
                "  processor:\n"
                "    gpu_batch_size: 8_000_000"
            ),
            "features.k_neighbors must be >= 1": (
                "Set k_neighbors to valid value:\n"
                "  features:\n"
                "    k_neighbors: 20  # Typical value"
            ),
        }

        for key, suggestion in suggestions.items():
            if key in error_msg:
                return suggestion

        return None


def validate_config_file(config_path: Union[str, Path], 
                        strict: bool = True,
                        allow_partial: bool = False) -> bool:
    """
    Validate configuration file.

    Args:
        config_path: Path to configuration file
        strict: If True, raise exception on error
        allow_partial: If True, allow partial configs

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If strict=True and validation fails
        FileNotFoundError: If config file doesn't exist
    """
    import yaml
    from omegaconf import OmegaConf

    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Handle None (empty file)
        if config is None:
            if strict:
                raise ValueError(f"Empty configuration file: {config_path}")
            logger.warning(f"Empty configuration file: {config_path}")
            return False

        # Validate
        errors = ConfigSchemaValidator.validate(
            config, 
            strict=strict,
            allow_partial=allow_partial
        )

        if errors:
            logger.error(f"Validation failed for {config_path}:")
            for error in errors:
                logger.error(f"  - {error}")
                # Try to provide suggestions
                suggestion = ConfigSchemaValidator.get_suggestions(error)
                if suggestion:
                    logger.info(f"    Suggestion: {suggestion}")
            return False

        logger.info(f"✓ Configuration validated: {config_path}")
        return True

    except yaml.YAMLError as e:
        error_msg = f"YAML syntax error in {config_path}: {e}"
        if strict:
            raise ValueError(error_msg) from e
        logger.error(error_msg)
        return False
    except Exception as e:
        error_msg = f"Error validating {config_path}: {e}"
        if strict:
            raise ValueError(error_msg) from e
        logger.error(error_msg)
        return False


def validate_config_dict(config: Dict[str, Any], 
                        config_name: str = "config",
                        strict: bool = True,
                        allow_partial: bool = False) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary
        config_name: Name for error messages
        strict: If True, raise exception on error
        allow_partial: If True, allow partial configs

    Returns:
        True if valid, False otherwise
    """
    try:
        errors = ConfigSchemaValidator.validate(
            config,
            strict=strict,
            allow_partial=allow_partial
        )

        if errors:
            logger.error(f"Validation failed for {config_name}:")
            for error in errors:
                logger.error(f"  - {error}")
            return False

        logger.info(f"✓ Configuration validated: {config_name}")
        return True

    except ValueError as e:
        if strict:
            raise
        logger.error(f"Validation error for {config_name}: {e}")
        return False


if __name__ == '__main__':
    """Test the validator with example configs."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validator.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        is_valid = validate_config_file(config_file, strict=False)
        sys.exit(0 if is_valid else 1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
