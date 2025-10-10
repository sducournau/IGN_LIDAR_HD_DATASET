---
sidebar_position: 3
title: CLI API
description: Command-line interface API reference and integration
keywords: [cli, api, command-line, integration, automation, hydra]
---

# CLI API Reference

**Version 2.0** - Dual CLI System

Comprehensive API documentation for both the modern Hydra CLI and legacy CLI in v2.0+.

:::tip Two CLI Systems Available
v2.0 provides **two CLI systems**: the modern **Hydra CLI** (recommended) and the **Legacy CLI** (backward compatible). Both are fully supported.
:::

---

## 🔄 CLI Evolution

### v1.x (Legacy CLI)

Traditional command-line interface with flag-based arguments:

```bash
ign-lidar-hd enrich --input-dir data/ --output output/ --use-gpu
ign-lidar-hd patch --input-dir enriched/ --output patches/
```

**Status:** ✅ Fully supported in v2.0 for backward compatibility

### v2.0 (Hydra CLI)

Modern configuration-based interface:

```bash
ign-lidar-hd process input_dir=data/ output_dir=output/ processor=gpu
```

**Status:** ⭐ Recommended for new projects

---

## 🎯 Hydra CLI (v2.0+)

### Main Command: `process`

Unified pipeline for RAW LAZ → Enriched LAZ / Patches.

#### Basic Usage

```bash
# Simplest form (required parameters only)
ign-lidar-hd process input_dir=data/ output_dir=output/

# With preset
ign-lidar-hd process input_dir=data/ output_dir=output/ preset=balanced

# With overrides
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  preset=balanced \
  processor=gpu \
  features.use_rgb=true \
  num_workers=8
```

#### Programmatic API

```python
from ign_lidar.core import LiDARProcessor
from omegaconf import OmegaConf

# Load Hydra configuration
cfg = OmegaConf.load("configs/config.yaml")

# Or create programmatically
cfg = OmegaConf.create({
    "input_dir": "data/",
    "output_dir": "output/",
    "preset": "balanced",
    "processor": "gpu",
    "features": {
        "use_rgb": True,
        "compute_ndvi": True
    },
    "num_workers": 8
})

# Initialize processor with config
processor = LiDARProcessor(cfg)

# Run pipeline
processor.run()
```

#### Parameters

**Core Parameters:**

| Parameter    | Type  | Required | Default    | Description           |
| ------------ | ----- | -------- | ---------- | --------------------- |
| `input_dir`  | `str` | ✅       | -          | Input directory path  |
| `output_dir` | `str` | ✅       | -          | Output directory path |
| `preset`     | `str` | ❌       | `balanced` | Workflow preset       |
| `processor`  | `str` | ❌       | `cpu`      | Processing backend    |
| `output`     | `str` | ❌       | `patches`  | Output mode           |

**Feature Parameters:**

| Parameter                 | Type   | Default    | Description           |
| ------------------------- | ------ | ---------- | --------------------- |
| `features`                | `str`  | `standard` | Feature set           |
| `features.use_rgb`        | `bool` | `true`     | Use RGB colors        |
| `features.compute_ndvi`   | `bool` | `false`    | Compute NDVI          |
| `features.boundary_aware` | `bool` | `false`    | Enable boundary-aware |
| `features.k_neighbors`    | `int`  | `30`       | Number of neighbors   |

**Performance Parameters:**

| Parameter       | Type   | Default | Description        |
| --------------- | ------ | ------- | ------------------ |
| `num_workers`   | `int`  | `4`     | Parallel workers   |
| `verbose`       | `bool` | `false` | Verbose output     |
| `show_progress` | `bool` | `true`  | Show progress bars |

Complete parameter reference: [Configuration System](/guides/configuration-system)

---

## 🔧 Legacy CLI (v1.x Compatible)

### Command: `download`

Download IGN LiDAR HD tiles.

```bash
# Basic download
ign-lidar-hd download \
  --bbox "xmin,ymin,xmax,ymax" \
  --output data/

# With specific tiles
ign-lidar-hd download \
  --tiles 0631_6275 0631_6276 \
  --output data/
```

#### Programmatic

```python
from ign_lidar.downloader import IGNDownloader

downloader = IGNDownloader(output_dir="data/")

# Download by bounding box
downloader.download_bbox(
    bbox=(631000, 6275000, 632000, 6276000),
    verbose=True
)

# Download specific tiles
downloader.download_tiles(
    tile_names=["0631_6275", "0631_6276"],
    overwrite=False
)
```

---

### Command: `enrich`

Add features to RAW LAZ files (Legacy workflow).

```bash
# Basic enrichment
ign-lidar-hd enrich \
  --input-dir data/raw/ \
  --output output/enriched/

# With features
ign-lidar-hd enrich \
  --input-dir data/raw/ \
  --output output/enriched/ \
  --use-rgb \
  --compute-ndvi \
  --use-gpu \
  --num-workers 4
```

#### Programmatic

```python
from ign_lidar.features import FeatureComputer
from ign_lidar.io import read_laz_file, write_laz_file

# Initialize feature computer
computer = FeatureComputer(
    use_rgb=True,
    compute_ndvi=True,
    use_gpu=True
)

# Process file
points, colors = read_laz_file("input.laz")
features = computer.compute(points, colors)

# Write enriched LAZ
write_laz_file(
    "output_enriched.laz",
    points,
    colors,
    features=features
)
```

---

### Command: `patch`

Generate patches from enriched LAZ (Legacy workflow).

```bash
# Basic patching
ign-lidar-hd patch \
  --input-dir enriched/ \
  --output patches/

# With parameters
ign-lidar-hd patch \
  --input-dir enriched/ \
  --output patches/ \
  --patch-size 50 \
  --points-per-patch 4096 \
  --architecture pointnet++
```

#### Programmatic

```python
from ign_lidar.datasets import PatchGenerator

generator = PatchGenerator(
    input_dir="enriched/",
    output_dir="patches/",
    patch_size=50.0,
    points_per_patch=4096,
    architecture="pointnet++"
)

# Generate patches
patches = generator.generate_all()

print(f"Generated {len(patches)} patches")
```

---

### Command: `verify`

Verify data integrity.

```bash
# Verify LAZ files
ign-lidar-hd verify --input-dir data/

# With detailed output
ign-lidar-hd verify --input-dir data/ --verbose
```

#### Programmatic

```python
from ign_lidar.io.verification import verify_laz_files

# Verify directory
results = verify_laz_files("data/")

# Check results
for file_path, status in results.items():
    if status.valid:
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path}: {status.error}")
```

---

## 🔀 CLI Comparison

### Processing Workflow

**Legacy CLI (Multi-step):**

```bash
# Step 1: Enrich
ign-lidar-hd enrich --input-dir data/ --output enriched/ --use-gpu

# Step 2: Patch
ign-lidar-hd patch --input-dir enriched/ --output patches/
```

**Hydra CLI (Single-step):**

```bash
# One command does everything!
ign-lidar-hd process \
  input_dir=data/ \
  output_dir=output/ \
  processor=gpu
```

### Parameter Syntax

**Legacy CLI:**

```bash
--input-dir value
--use-gpu
--num-workers 4
```

**Hydra CLI:**

```bash
input_dir=value
processor=gpu
num_workers=4
```

### Configuration

**Legacy CLI:**

- Command-line only
- No presets
- Manual parameter management

**Hydra CLI:**

- YAML configuration files
- 4 built-in presets
- Hierarchical composition
- Type validation

---

## 🎓 Advanced CLI Usage

### Batch Processing Script

```python
#!/usr/bin/env python3
"""
Batch process multiple directories with Hydra CLI
"""
from pathlib import Path
from ign_lidar.core import LiDARProcessor
from omegaconf import OmegaConf

# Base configuration
base_cfg = OmegaConf.create({
    "preset": "balanced",
    "processor": "gpu",
    "features": {
        "use_rgb": True,
        "compute_ndvi": True
    },
    "num_workers": 8
})

# Process multiple directories
input_dirs = [
    "/data/project1/tiles/",
    "/data/project2/tiles/",
    "/data/project3/tiles/"
]

for input_dir in input_dirs:
    # Create project-specific config
    cfg = OmegaConf.merge(
        base_cfg,
        {
            "input_dir": input_dir,
            "output_dir": input_dir.replace("/tiles/", "/output/")
        }
    )

    # Process
    processor = LiDARProcessor(cfg)
    processor.run()

    print(f"✅ Completed: {input_dir}")
```

### Parallel Multi-Project Processing

```python
from concurrent.futures import ProcessPoolExecutor
from ign_lidar.core import LiDARProcessor
from omegaconf import OmegaConf

def process_project(project_config):
    """Process a single project"""
    cfg = OmegaConf.create(project_config)
    processor = LiDARProcessor(cfg)
    processor.run()
    return cfg.output_dir

# Define projects
projects = [
    {
        "input_dir": "/data/urban/",
        "output_dir": "/output/urban/",
        "preset": "quality",
        "processor": "gpu"
    },
    {
        "input_dir": "/data/rural/",
        "output_dir": "/output/rural/",
        "preset": "balanced",
        "processor": "cpu"
    }
]

# Process in parallel
with ProcessPoolExecutor(max_workers=2) as executor:
    results = executor.map(process_project, projects)

for output_dir in results:
    print(f"✅ Output: {output_dir}")
```

### Custom CLI Wrapper

```python
import argparse
from ign_lidar.core import LiDARProcessor
from omegaconf import OmegaConf

def create_cli():
    parser = argparse.ArgumentParser(
        description="Custom IGN LiDAR HD Processor"
    )

    parser.add_argument("--project", required=True)
    parser.add_argument("--quality", default="balanced",
                       choices=["fast", "balanced", "quality", "ultra"])
    parser.add_argument("--gpu", action="store_true")

    return parser

def main():
    parser = create_cli()
    args = parser.parse_args()

    # Map to Hydra config
    cfg = OmegaConf.create({
        "input_dir": f"/data/{args.project}/raw/",
        "output_dir": f"/output/{args.project}/",
        "preset": args.quality,
        "processor": "gpu" if args.gpu else "cpu",
        "num_workers": 8
    })

    # Run processing
    processor = LiDARProcessor(cfg)
    processor.run()

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
python custom_cli.py --project urban --quality quality --gpu
```

## Configuration API

### CLIConfig

```python
from ign_lidar.cli.config import CLIConfig

# Load CLI configuration
config = CLIConfig.from_file("cli_config.yaml")

# Programmatic configuration
config = CLIConfig(
    default_output_format="laz",
    parallel_processing=True,
    max_workers=8,
    chunk_size=1000000,

    # Logging configuration
    log_level="INFO",
    log_format="%(asctime)s - %(levelname)s - %(message)s",

    # Performance settings
    gpu_acceleration=True,
    memory_limit="8GB"
)

# Apply configuration to CLI
cli = CommandLineInterface(config=config)
```

### Environment Configuration

```python
from ign_lidar.cli.config import setup_environment

# Configure environment variables
env_config = setup_environment(
    data_directory="/data/lidar",
    cache_directory="/tmp/ign_cache",
    temp_directory="/tmp/ign_temp",
    gpu_device="cuda:0"
)

# Environment variables set:
# IGN_DATA_DIR=/data/lidar
# IGN_CACHE_DIR=/tmp/ign_cache
# IGN_TEMP_DIR=/tmp/ign_temp
# CUDA_VISIBLE_DEVICES=0
```

## Progress Monitoring

### ProgressTracker

```python
from ign_lidar.cli.progress import ProgressTracker, ProgressCallback

class CustomProgressCallback(ProgressCallback):
    def on_start(self, total_items):
        print(f"Starting processing {total_items} items...")

    def on_progress(self, completed, total, current_item):
        percent = (completed / total) * 100
        print(f"Progress: {percent:.1f}% - {current_item}")

    def on_complete(self, total_items, elapsed_time):
        print(f"Completed {total_items} items in {elapsed_time:.2f}s")

# Use custom progress tracking
tracker = ProgressTracker(callback=CustomProgressCallback())

# Process with progress tracking
cli = CommandLineInterface(progress_tracker=tracker)
result = cli.enrich_batch(input_files)
```

### Real-time Monitoring

```python
from ign_lidar.cli.monitoring import ProcessMonitor

# Monitor CLI processes
monitor = ProcessMonitor()

# Start monitoring
monitor.start()

# Execute CLI command
result = cli.run_command("enrich", args={"input": "large_file.las"})

# Get monitoring data
stats = monitor.get_stats()
print(f"CPU Usage: {stats.cpu_percent:.1f}%")
print(f"Memory Usage: {stats.memory_mb:.1f} MB")
print(f"Disk I/O: {stats.disk_io_mbps:.1f} MB/s")

monitor.stop()
```

## Error Handling

### CLIException Classes

```python
from ign_lidar.cli.exceptions import (
    CLIError,
    CommandNotFoundError,
    InvalidArgumentError,
    ProcessingError,
    FileNotFoundError
)

try:
    result = cli.run_command("enrich", args={"invalid_arg": True})
except InvalidArgumentError as e:
    print(f"Invalid argument: {e.argument} - {e.message}")
except FileNotFoundError as e:
    print(f"File not found: {e.filepath}")
except ProcessingError as e:
    print(f"Processing failed: {e.details}")
```

### Retry and Recovery

```python
from ign_lidar.cli.retry import RetryManager

# Configure retry logic
retry_manager = RetryManager(
    max_attempts=3,
    backoff_factor=2.0,
    retry_on=[ProcessingError, IOError],
    exclude=[InvalidArgumentError]
)

# Execute with retry
def process_with_retry():
    return cli.run_command("enrich", args=enrich_args)

result = retry_manager.execute(process_with_retry)
```

## Validation and Testing

### CommandValidator

```python
from ign_lidar.cli.validation import CommandValidator

validator = CommandValidator()

# Validate command arguments
validation_result = validator.validate_command(
    command="enrich",
    args={
        "input": "input.las",
        "output": "output.laz",
        "features": ["buildings", "invalid_feature"]
    }
)

if not validation_result.is_valid:
    for error in validation_result.errors:
        print(f"Validation error: {error}")
```

### CLI Testing Framework

```python
from ign_lidar.cli.testing import CLITestRunner

# Test CLI commands
test_runner = CLITestRunner()

# Create test case
test_case = {
    "command": "enrich",
    "args": {
        "input": "test_data/sample.las",
        "output": "test_output/enriched.laz"
    },
    "expected_exit_code": 0,
    "expected_files": ["test_output/enriched.laz"],
    "timeout": 60
}

# Run test
test_result = test_runner.run_test(test_case)

if test_result.passed:
    print("✅ Test passed")
else:
    print(f"❌ Test failed: {test_result.error_message}")
```

## Integration Patterns

### Workflow Integration

```python
from ign_lidar.cli.workflows import WorkflowRunner

# Define processing workflow
workflow = {
    "name": "Complete Processing Pipeline",
    "steps": [
        {
            "command": "download",
            "args": {"tiles": ["0631_6275"], "output_dir": "data/raw"}
        },
        {
            "command": "patch",
            "args": {"input": "data/raw/0631_6275.las", "output": "data/patched/0631_6275.las"}
        },
        {
            "command": "enrich",
            "args": {
                "input": "data/patched/0631_6275.las",
                "output": "data/enriched/0631_6275.laz",
                "features": ["buildings", "vegetation"]
            }
        }
    ]
}

# Execute workflow
runner = WorkflowRunner(cli=cli)
workflow_result = runner.execute_workflow(workflow)

# Check results
for step_result in workflow_result.step_results:
    print(f"Step {step_result.step_name}: {'✅' if step_result.success else '❌'}")
```

### Batch Processing

```python
from ign_lidar.cli.batch import BatchProcessor

# Configure batch processing
batch_processor = BatchProcessor(
    cli=cli,
    max_parallel=4,
    retry_failed=True,
    progress_tracking=True
)

# Define batch job
batch_job = {
    "command": "enrich",
    "input_pattern": "data/raw/*.las",
    "output_pattern": "data/enriched/{basename}.laz",
    "common_args": {
        "features": ["buildings", "vegetation"],
        "rgb_source": "data/orthophotos"
    }
}

# Execute batch job
batch_result = batch_processor.execute_batch(batch_job)

# Summary
print(f"Total files: {batch_result.total_files}")
print(f"Successful: {batch_result.successful_files}")
print(f"Failed: {batch_result.failed_files}")
print(f"Total time: {batch_result.total_time:.2f}s")
```

### External System Integration

#### Database Integration

```python
from ign_lidar.cli.database import DatabaseIntegration

# Configure database connection
db_integration = DatabaseIntegration(
    connection_string="postgresql://user:pass@localhost/lidar_db",
    table_name="processing_jobs"
)

# Log CLI execution to database
def process_with_db_logging(command, args):
    job_id = db_integration.create_job(command, args)

    try:
        result = cli.run_command(command, args)
        db_integration.update_job_success(job_id, result)
        return result
    except Exception as e:
        db_integration.update_job_failure(job_id, str(e))
        raise

# Usage
result = process_with_db_logging("enrich", enrich_args)
```

#### Queue Integration

```python
from ign_lidar.cli.queue import TaskQueue
import redis

# Redis-based task queue
redis_client = redis.Redis(host='localhost', port=6379, db=0)
task_queue = TaskQueue(redis_client)

# Producer: Add tasks to queue
for las_file in las_files:
    task = {
        "command": "enrich",
        "args": {
            "input": las_file,
            "output": las_file.replace('.las', '_enriched.laz')
        }
    }
    task_queue.enqueue(task)

# Consumer: Process tasks from queue
def process_queue_tasks():
    while True:
        task = task_queue.dequeue(timeout=30)
        if task:
            try:
                result = cli.run_command(task["command"], task["args"])
                task_queue.mark_completed(task["id"], result)
            except Exception as e:
                task_queue.mark_failed(task["id"], str(e))
```

## Performance Optimization

### Caching API

```python
from ign_lidar.cli.cache import CLICache

# Configure caching
cache = CLICache(
    cache_dir="/tmp/ign_cli_cache",
    max_size_gb=10.0,
    ttl_hours=24
)

# Enable caching for CLI
cli = CommandLineInterface(cache=cache)

# Cached execution (subsequent calls with same args return cached result)
result1 = cli.run_command("download", {"tiles": ["0631_6275"]})  # Downloads
result2 = cli.run_command("download", {"tiles": ["0631_6275"]})  # From cache

print(f"Cache hit: {result2.from_cache}")
```

### Resource Management

```python
from ign_lidar.cli.resources import ResourceManager

# Configure resource limits
resource_manager = ResourceManager(
    max_cpu_percent=80.0,
    max_memory_gb=16.0,
    max_disk_io_mbps=500.0,
    gpu_memory_fraction=0.8
)

# Apply resource limits to CLI
cli = CommandLineInterface(resource_manager=resource_manager)

# CLI will automatically throttle to stay within limits
result = cli.run_command("enrich", large_file_args)
```

## Advanced Features

### Plugin System

```python
from ign_lidar.cli.plugins import CLIPlugin, register_plugin

class CustomPlugin(CLIPlugin):
    """Custom CLI plugin example."""

    def get_command_name(self):
        return "custom-process"

    def get_command_help(self):
        return "Custom processing command"

    def execute(self, args):
        # Custom processing logic
        return {"status": "success", "message": "Custom processing completed"}

# Register plugin
register_plugin(CustomPlugin())

# Use custom command
result = cli.run_command("custom-process", {"input": "data.las"})
```

### Script Generation

```python
from ign_lidar.cli.scripting import ScriptGenerator

# Generate bash script from CLI commands
generator = ScriptGenerator(target="bash")

script_content = generator.generate_script([
    ("download", {"tiles": ["0631_6275"], "output_dir": "data"}),
    ("enrich", {"input": "data/0631_6275.las", "output": "data/enriched.laz"})
])

# Save script
with open("process_lidar.sh", "w") as f:
    f.write(script_content)

# Generated script can be executed independently:
# #!/bin/bash
# ign-lidar-hd download --tiles 0631_6275 --output-dir data
# ign-lidar-hd enrich --input data/0631_6275.las --output data/enriched.laz
```

### Configuration Management

```python
from ign_lidar.cli.config_manager import ConfigManager

# Manage multiple configurations
config_manager = ConfigManager(config_dir="~/.ign-lidar/configs")

# Create named configurations
config_manager.save_config("production", {
    "output_format": "laz",
    "parallel_processing": True,
    "gpu_acceleration": True,
    "quality_level": "high"
})

config_manager.save_config("development", {
    "output_format": "las",
    "parallel_processing": False,
    "gpu_acceleration": False,
    "quality_level": "medium"
})

# Use specific configuration
cli = CommandLineInterface()
cli.load_config("production")

# List available configurations
configs = config_manager.list_configs()
print(f"Available configs: {configs}")
```

## Best Practices

### Error Handling Patterns

```python
from ign_lidar.cli.patterns import robust_execution

@robust_execution(
    retry_attempts=3,
    fallback_strategy="cpu",
    log_failures=True
)
def process_tile(input_file):
    return cli.run_command("enrich", {
        "input": input_file,
        "features": ["buildings", "vegetation"],
        "gpu": True
    })

# Automatically handles retries and fallbacks
result = process_tile("large_tile.las")
```

### Logging Best Practices

```python
import logging
from ign_lidar.cli.logging import setup_cli_logging

# Configure comprehensive logging
logger = setup_cli_logging(
    log_file="ign_processing.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    include_performance=True
)

# CLI operations are automatically logged
cli = CommandLineInterface(logger=logger)
result = cli.run_command("enrich", args)

# Logs include:
# - Command execution details
# - Performance metrics
# - Error traces
# - Resource usage
```

## Related Documentation

- [CLI Commands Guide](../guides/cli-commands)
- [Configuration Reference](./configuration)
- [Processor API](./processor)
- [Performance Guide](../guides/performance)
