---
sidebar_position: 4
title: API de Configuration
description: Configuration system for LiDAR processing pipelines
keywords: [configuration, settings, parameters, pipeline, api]
---

# Configuration API

Comprehensive configuration system for IGN LiDAR HD processing pipelines.

## Configuration Classes

### Core Configuration

```python
from ign_lidar import Config

# Basic configuration
config = Config(
    input_format='las',
    output_format='laz',
    chunk_size=1000000,
    n_jobs=-1
)

# Load from file
config = Config.from_file('config.yaml')

# Load from dictionary
config_dict = {
    'processing': {
        'chunk_size': 500000,
        'use_gpu': True
    },
    'features': {
        'buildings': True,
        'vegetation': True
    }
}
config = Config.from_dict(config_dict)
```

### Traitementing Configuration

```python
class TraitementingConfig:
    """Configuration for data processing parameters."""

    def __init__(self,
                 chunk_size: int = 1000000,
                 overlap_size: int = 10,
                 n_jobs: int = -1,
                 use_gpu: bool = False,
                 memory_limit: str = '8GB'):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.memory_limit = memory_limit

    def validate(self):
        """Validate configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.n_jobs < -1 or self.n_jobs == 0:
            raise ValueError("n_jobs must be -1 or positive integer")
```

### Feature Configuration

```python
class FeatureConfig:
    """Configuration for feature extraction."""

    def __init__(self):
        self.buildings = BuildingConfig()
        self.vegetation = VegetationConfig()
        self.ground = GroundConfig()
        self.water = WaterConfig()

class BuildingConfig:
    """Building detection configuration."""

    def __init__(self,
                 min_points: int = 100,
                 min_height: float = 2.0,
                 max_height: float = 200.0,
                 planarity_threshold: float = 0.1,
                 density_threshold: float = 5.0):
        self.min_points = min_points
        self.min_height = min_height
        self.max_height = max_height
        self.planarity_threshold = planarity_threshold
        self.density_threshold = density_threshold
```

## Configuration File Formats

### YAML Configuration

```yaml
# config.yaml
processing:
  chunk_size: 1000000
  overlap_size: 10
  n_jobs: -1
  use_gpu: true
  memory_limit: "8GB"

features:
  buildings:
    enabled: true
    min_points: 100
    min_height: 2.0
    max_height: 200.0
    planarity_threshold: 0.1

  vegetation:
    enabled: true
    min_height: 0.5
    max_height: 50.0
    canopy_threshold: 0.8

  ground:
    enabled: true
    classification_method: "cloth_simulation"
    cloth_resolution: 0.5

output:
  format: "laz"
  compression: 7
  coordinate_system: "EPSG:2154"

quality:
  validation: true
  error_threshold: 0.1
  generate_reports: true
```

### JSON Configuration

```json
{
  "processing": {
    "chunk_size": 1000000,
    "overlap_size": 10,
    "n_jobs": -1,
    "use_gpu": true
  },
  "features": {
    "buildings": {
      "enabled": true,
      "min_points": 100,
      "min_height": 2.0
    },
    "vegetation": {
      "enabled": true,
      "canopy_threshold": 0.8
    }
  }
}
```

## Configuration Loading

### File-based Configuration

```python
from ign_lidar.config import ConfigLoader

# Load YAML configuration
loader = ConfigLoader()
config = loader.load_yaml('config.yaml')

# Load JSON configuration
config = loader.load_json('config.json')

# Load with environment variable substitution
config = loader.load_yaml('config.yaml',
                         substitute_env_vars=True)
```

### Environment Configuration

```python
import os
from ign_lidar import Config

# Configure from environment variables
config = Config(
    chunk_size=int(os.getenv('IGN_CHUNK_SIZE', 1000000)),
    n_jobs=int(os.getenv('IGN_N_JOBS', -1)),
    use_gpu=os.getenv('IGN_USE_GPU', 'false').lower() == 'true'
)
```

### Dynamic Configuration

```python
class DynamicConfig:
    """Configuration that adapts based on input data."""

    def __init__(self, base_config):
        self.base_config = base_config

    def adapt_to_data(self, data_info):
        """Adapt configuration based on data characteristics."""

        config = self.base_config.copy()

        # Adjust chunk size based on point density
        if data_info['point_density'] > 100:
            config.chunk_size = 500000  # Smaller chunks for dense data

        # Enable GPU for large datasets
        if data_info['total_points'] > 10000000:
            config.use_gpu = True

        # Adjust feature parameters based on area type
        if data_info['area_type'] == 'urban':
            config.features.buildings.min_points = 50
        elif data_info['area_type'] == 'rural':
            config.features.buildings.min_points = 200

        return config
```

## Configuration Validation

### Schema Validation

```python
from jsonschema import validate
import yaml

def validate_config(config_path):
    """Validate configuration against schema."""

    schema = {
        "type": "object",
        "properties": {
            "processing": {
                "type": "object",
                "properties": {
                    "chunk_size": {"type": "integer", "minimum": 1},
                    "n_jobs": {"type": "integer", "minimum": -1},
                    "use_gpu": {"type": "boolean"}
                },
                "required": ["chunk_size"]
            },
            "features": {
                "type": "object",
                "properties": {
                    "buildings": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "min_points": {"type": "integer", "minimum": 1}
                        }
                    }
                }
            }
        },
        "required": ["processing"]
    }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    validate(instance=config, schema=schema)
    return True
```

### Runtime Validation

```python
class ConfigValidator:
    """Validate configuration at runtime."""

    def validate_processing_config(self, config):
        """Validate processing configuration."""

        errors = []

        # Check chunk size
        if config.chunk_size <= 0:
            errors.append("chunk_size must be positive")

        # Check memory limit
        if not self._is_valid_memory_format(config.memory_limit):
            errors.append("Invalid memory_limit format")

        # Check GPU availability
        if config.use_gpu and not self._is_gpu_available():
            errors.append("GPU requested but not available")

        if errors:
            raise ValueError(f"Configuration errors: {errors}")

    def _is_gpu_available(self):
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
```

## Configuration Templates

### Urban Traitementing Template

```yaml
# urban_template.yaml
name: "Urban Traitementing Template"
description: "Optimized for dense urban environments"

processing:
  chunk_size: 500000
  overlap_size: 20
  use_gpu: true

features:
  buildings:
    enabled: true
    min_points: 50
    min_height: 2.0
    edge_detection: "enhanced"

  vegetation:
    enabled: true
    urban_trees: true
    park_detection: true

  infrastructure:
    roads: true
    bridges: true
    tunnels: true
```

### Rural Traitementing Template

```yaml
# rural_template.yaml
name: "Rural Traitementing Template"
description: "Optimized for rural and natural areas"

processing:
  chunk_size: 2000000
  overlap_size: 10

features:
  buildings:
    enabled: true
    min_points: 200
    agricultural_buildings: true

  vegetation:
    enabled: true
    forest_classification: true
    crop_detection: true

  terrain:
    elevation_analysis: true
    slope_calculation: true
```

### GPU Acceleration Template

```yaml
# gpu_template.yaml
name: "GPU Accelerated Template"
description: "Maximum performance with GPU acceleration"

processing:
  chunk_size: 2000000
  use_gpu: true
  gpu_memory_fraction: 0.8
  batch_processing: true

features:
  buildings:
    gpu_acceleration: true
    parallel_processing: true

  rgb_augmentation:
    gpu_interpolation: true
    batch_size: 10000
```

## Configuration Management

### Configuration Registry

```python
class ConfigRegistry:
    """Registry for managing configuration templates."""

    def __init__(self):
        self.templates = {}
        self.load_default_templates()

    def register_template(self, name, config):
        """Register a configuration template."""
        self.templates[name] = config

    def get_template(self, name):
        """Get configuration template by name."""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        return self.templates[name].copy()

    def list_templates(self):
        """List available templates."""
        return list(self.templates.keys())
```

### Configuration Inheritance

```python
class InheritableConfig(Config):
    """Configuration with inheritance support."""

    def __init__(self, base_config=None, **kwargs):
        if base_config:
            # Start with base configuration
            self.__dict__.update(base_config.__dict__)

        # Override with provided parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def merge_with(self, other_config):
        """Merge with another configuration."""
        merged = InheritableConfig(self)

        for key, value in other_config.__dict__.items():
            if isinstance(value, dict) and hasattr(merged, key):
                # Deep merge dictionaries
                merged_dict = getattr(merged, key).copy()
                merged_dict.update(value)
                setattr(merged, key, merged_dict)
            else:
                setattr(merged, key, value)

        return merged
```

## CLI Configuration

### Command Line Interface

```bash
# Generate configuration template
ign-lidar-hd config --template urban > urban_config.yaml

# Validate configuration
ign-lidar-hd config --validate config.yaml

# Show current configuration
ign-lidar-hd config --show

# Override configuration parameters
ign-lidar-hd process input.las --config config.yaml --chunk-size 500000 --use-gpu
```

### Configuration CLI Implementation

```python
import click
from ign_lidar.config import Config, ConfigRegistry

@click.group()
def config():
    """Configuration management commands."""
    pass

@config.command()
@click.option('--template', help='Template name')
@click.option('--output', help='Sortie file path')
def generate(template, output):
    """Generate configuration template."""

    registry = ConfigRegistry()

    if template:
        config = registry.get_template(template)
    else:
        config = Config.get_default()

    config_yaml = config.to_yaml()

    if output:
        with open(output, 'w') as f:
            f.write(config_yaml)
    else:
        click.echo(config_yaml)

@config.command()
@click.argument('config_file')
def validate(config_file):
    """Validate configuration file."""

    try:
        config = Config.from_file(config_file)
        config.validate()
        click.echo("Configuration is valid")
    except Exception as e:
        click.echo(f"Configuration error: {e}")
        raise click.Abort()
```

## Best Practices

### Configuration Organization

1. **Separate Concerns**: Use different config files for different aspects
2. **Environment Specific**: Maintain configs for dev/test/prod
3. **Documentation**: Comment configuration parameters
4. **Validation**: Always validate configurations before use
5. **Versioning**: Version control configuration files

### Performance Considerations

```python
# Optimize for your hardware
config = Config(
    chunk_size=calculate_optimal_chunk_size(),
    n_jobs=get_cpu_count(),
    memory_limit=get_available_memory() * 0.8
)

def calculate_optimal_chunk_size():
    """Calculate optimal chunk size based on available memory."""
    import psutil

    available_memory = psutil.virtual_memory().available
    point_size = 28  # bytes per point (XYZ + RGB + classification, etc.)

    # Use 50% of available memory for chunk
    optimal_points = (available_memory * 0.5) // point_size

    return min(optimal_points, 5000000)  # Cap at 5M points
```

### Security Considerations

```python
def load_secure_config(config_path):
    """Load configuration with security validation."""

    # Validate file permissions
    import stat
    file_stats = os.stat(config_path)
    if file_stats.st_mode & stat.S_IROTH:
        raise SecurityError("Config file is world-readable")

    # Sanitize paths
    config = Config.from_file(config_path)
    config.sanitize_paths()

    return config
```

## Related Documentation

- [Traitementing Guide](../guides/getting-started)
- [Performance Guide](../guides/performance)
- [API Reference](./features)
- [CLI Reference](./cli)
