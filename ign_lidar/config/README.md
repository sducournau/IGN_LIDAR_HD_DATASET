# Configuration Module (`config/`)

This directory contains the **Python configuration schema and validation** for the IGN LiDAR HD library.

## 📂 Directory Structure

```
ign_lidar/
├── config/              # Python schema & validation
│   ├── __init__.py     # Module exports
│   ├── schema.py       # Main configuration schema (Hydra-based)
│   ├── schema_simplified.py  # Simplified schema for basic usage
│   ├── preset_loader.py      # Preset configuration loader
│   └── validator.py          # Configuration validation
│
└── configs/            # YAML configuration files
    ├── base/           # Base configuration templates
    ├── presets/        # Pre-configured presets
    ├── profiles/       # Hardware-specific profiles
    ├── hardware/       # Hardware optimization configs
    └── advanced/       # Advanced feature configurations
```

## 🎯 Purpose

The `config/` directory provides:

1. **Python Schema Definition**: Dataclass-based configuration schema using Hydra/OmegaConf
2. **Type Validation**: Runtime validation of configuration values
3. **Default Values**: Sensible defaults for all configuration options
4. **Preset Management**: Loading and merging preset configurations

## 📝 Key Files

### `schema.py`

Main configuration schema with all available options:

```python
from ign_lidar.config import IGNLiDARConfig

config = IGNLiDARConfig(
    input_dir="/data/tiles",
    output_dir="/data/output",
    processor=ProcessorConfig(
        lod_level="LOD2",
        use_gpu=True
    )
)
```

### `schema_simplified.py`

Simplified schema for basic usage (legacy compatibility):

```python
from ign_lidar.config.schema_simplified import SimplifiedConfig

config = SimplifiedConfig(
    input_dir="/data/tiles",
    output_dir="/data/output",
    lod_level="LOD2"
)
```

### `validator.py`

Configuration validation utilities:

```python
from ign_lidar.config.validator import validate_config

# Validate configuration before processing
errors = validate_config(config)
if errors:
    raise ValueError(f"Invalid configuration: {errors}")
```

### `preset_loader.py`

Load and merge preset configurations:

```python
from ign_lidar.config.preset_loader import load_preset

# Load GPU-optimized preset
config = load_preset("gpu_optimized", overrides={
    "input_dir": "/my/data"
})
```

## 🔗 Relationship with `configs/`

| Directory  | Purpose         | Contents                                      |
| ---------- | --------------- | --------------------------------------------- |
| `config/`  | **Python code** | Schema definitions, validation, loaders       |
| `configs/` | **YAML files**  | Actual configuration files, presets, profiles |

**Think of it this way:**

- `config/` = The **library** (how to read/validate configs)
- `configs/` = The **books** (actual configuration data)

## 📖 Usage Examples

### Basic Usage

```python
from ign_lidar import LiDARProcessor

# Load from YAML file
processor = LiDARProcessor(config_path="configs/presets/production_gpu.yaml")

# Or create programmatically
from ign_lidar.config import IGNLiDARConfig, ProcessorConfig

config = IGNLiDARConfig(
    input_dir="/data/tiles",
    output_dir="/data/output",
    processor=ProcessorConfig(
        lod_level="LOD2",
        processing_mode="patches_only",
        use_gpu=True
    )
)

processor = LiDARProcessor(config=config)
```

### Using Presets

```python
# Load preset with overrides
config = load_preset(
    "gpu_optimized",
    overrides={
        "input_dir": "/my/data",
        "processor.patch_size": 200.0
    }
)
```

### Validation

```python
from ign_lidar.config.validator import validate_config

# Validate before processing
errors = validate_config(config)
if errors:
    for error in errors:
        print(f"❌ {error}")
else:
    print("✅ Configuration valid")
```

## 🚀 Migration from v2.x

Old (v2.x):

```python
from ign_lidar.processor import LiDARProcessor

processor = LiDARProcessor(
    lod_level="LOD2",
    use_gpu=True,
    patch_size=150.0
)
```

New (v3.x):

```python
from ign_lidar import LiDARProcessor

processor = LiDARProcessor(
    config_path="configs/presets/lod2_gpu.yaml"
)

# Or programmatically
config = IGNLiDARConfig(
    processor=ProcessorConfig(
        lod_level="LOD2",
        use_gpu=True,
        patch_size=150.0
    )
)
processor = LiDARProcessor(config=config)
```

## 📚 See Also

- [Configuration Guide](../../docs/docs/guides/configuration.md) - Complete configuration documentation
- [configs/ README](../configs/README.md) - YAML configuration files
- [Preset Reference](../configs/presets/README.md) - Available presets
- [Hardware Profiles](../configs/profiles/README.md) - Hardware-specific configurations

## 🔧 Development

### Adding New Configuration Options

1. Update schema in `schema.py`:

   ```python
   @dataclass
   class ProcessorConfig:
       new_option: str = "default_value"
   ```

2. Add validation in `validator.py`:

   ```python
   def validate_new_option(value: str) -> List[str]:
       if value not in VALID_OPTIONS:
           return [f"Invalid new_option: {value}"]
       return []
   ```

3. Update presets in `configs/presets/` as needed

4. Document in configuration guide

### Running Tests

```bash
# Test configuration loading
pytest tests/test_config_schema.py -v

# Test validation
pytest tests/test_config_validator.py -v

# Test preset loading
pytest tests/test_preset_loader.py -v
```

---

**Note:** This module is the **foundation** of the configuration system. All YAML files in `configs/` are validated against the schemas defined here.
