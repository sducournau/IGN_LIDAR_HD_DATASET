---
sidebar_position: 3
title: CLI API
description: Command-line interface API reference and integration
keywords: [cli, api, command-line, integration, automation]
---

# CLI API Reference

Comprehensive API documentation for integrating with IGN LiDAR HD command-line interface.

## CLI Module

### CommandLineInterface

Main CLI interface class for programmatic command execution.

```python
from ign_lidar.cli import CommandLineInterface

cli = CommandLineInterface(
    verbose=True,
    log_file="ign_lidar.log",
    config_file="config.yaml"
)
```

#### Parameters

- **verbose** (`bool`): Enable verbose output
- **log_file** (`str`): Path to log file
- **config_file** (`str`): Default configuration file
- **working_directory** (`str`): Working directory for commands

#### Methods

##### `run_command(command, args=None, **kwargs)`

Execute CLI commands programmatically.

```python
# Download command
result = cli.run_command(
    command="download",
    args={
        "tiles": ["0631_6275", "0631_6276"],
        "output_dir": "/data/lidar",
        "format": "laz"
    }
)

# Check execution result
if result.success:
    print(f"Downloaded {len(result.files)} files")
    print(f"Total size: {result.total_size_mb:.1f} MB")
else:
    print(f"Error: {result.error_message}")
```

##### `enrich_batch(input_files, **kwargs)`

Batch enrichment processing.

```python
# Enrich multiple files
results = cli.enrich_batch(
    input_files=["tile1.las", "tile2.las", "tile3.las"],
    output_dir="/data/enriched",
    features=["buildings", "vegetation"],
    rgb_source="/data/orthophotos",
    parallel=True,
    max_workers=4
)

# Process results
for file_result in results:
    if file_result.success:
        print(f"✅ {file_result.input_file} -> {file_result.output_file}")
    else:
        print(f"❌ {file_result.input_file}: {file_result.error}")
```

### Command Classes

#### DownloadCommand

```python
from ign_lidar.cli.commands import DownloadCommand

download_cmd = DownloadCommand()

# Configure download parameters
download_result = download_cmd.execute(
    tiles=["0631_6275"],
    output_directory="/data/raw",
    file_format="laz",
    overwrite=False,
    verify_checksum=True
)

# Access download information
print(f"Files downloaded: {download_result.file_count}")
print(f"Download time: {download_result.duration:.2f}s")
print(f"Average speed: {download_result.speed_mbps:.1f} MB/s")
```

#### EnrichCommand

```python
from ign_lidar.cli.commands import EnrichCommand

enrich_cmd = EnrichCommand()

# Execute enrichment
enrich_result = enrich_cmd.execute(
    input_file="input.las",
    output_file="output.laz",
    config_file="enrich_config.yaml",
    features=["buildings", "vegetation", "ground"],
    rgb_orthophoto="orthophoto.tif",
    overwrite=False
)

# Check enrichment results
print(f"Points processed: {enrich_result.points_processed:,}")
print(f"Features added: {enrich_result.features_added}")
print(f"Processing time: {enrich_result.processing_time:.2f}s")
```

#### PatchCommand

```python
from ign_lidar.cli.commands import PatchCommand

patch_cmd = PatchCommand()

# Apply preprocessing patches
patch_result = patch_cmd.execute(
    input_file="raw.las",
    output_file="patched.las",
    patch_types=["noise_removal", "ground_classification"],
    patch_config="patch_config.yaml"
)
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
