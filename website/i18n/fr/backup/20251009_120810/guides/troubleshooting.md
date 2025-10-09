---
sidebar_position: 8
title: Guide de dépannage
description: Solutions aux problèmes courants de traitement LiDAR
keywords: [dépannage, erreurs, problèmes, solutions, aide]
---

# Dépannage Guide

Solutions to common issues encountered when processing LiDAR data with IGN LiDAR HD.

## Installation Issues

### Dependency Installation Failure

**Problem**: Python package installation fails

```bash
ERROR: Failed building wheel for some-package
```

**Solutions**:

1. Update pip

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. Install with conda

   ```bash
   conda install -c conda-forge ign-lidar-hd
   ```

3. Manual dependency installation
   ```bash
   pip install numpy scipy laspy pdal
   pip install ign-lidar-hd
   ```

### CUDA/GPU Issues

**Problem**: CUDA not detected

```bash
CUDA not available, falling back to CPU processing
```

**Diagnosis**:

```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions**:

1. Install NVIDIA drivers
2. Install CUDA Toolkit
3. Install PyTorch with CUDA support

## Traitementing Issues

### Memory Errors

**Problem**: Insufficient memory

```bash
MemoryError: Unable to allocate array
```

**Solutions**:

1. Reduce chunk size

   ```python
   processor = Traitementor(chunk_size=100000)
   ```

2. Traitement in small batches

   ```python
   for batch in split_large_file(input_file, max_points=500000):
       process_batch(batch)
   ```

3. Use pagination
   ```python
   processor = Traitementor(use_pagination=True, page_size=50000)
   ```

### GPU Out of Memory

**Problem**: Insufficient VRAM

```bash
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:

1. Limit GPU memory

   ```python
   processor = Traitementor(gpu_memory_limit=0.5)
   ```

2. Clear GPU cache

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. Hybrid CPU/GPU processing
   ```python
   processor = Traitementor(
       use_gpu=True,
       fallback_to_cpu=True
   )
   ```

### File Errors

**Problem**: Corrupted LAS file

```bash
LASError: Invalid LAS file header
```

**Diagnosis**:

```bash
# Verify with PDAL
pdal info input.las

# Validate with laspy
python -c "import laspy; f=laspy.read('input.las'); print('OK')"
```

**Solutions**:

1. Repair with PDAL

   ```bash
   pdal translate input.las output_fixed.las --writers.las.forward=all
   ```

2. Validate and clean
   ```python
   from ign_lidar.utils import validate_and_clean
   clean_file = validate_and_clean("input.las")
   ```

## Performance Issues

### Slow Traitementing

**Problem**: Very poor performance

**Diagnosis**:

```python
from ign_lidar import Profiler

with Profiler() as prof:
    result = processor.process_tile("input.las")
prof.print_bottlenecks()
```

**Solutions**:

1. Optimize parameters

   ```python
   processor = Traitementor(
       n_jobs=-1,  # All cores
       chunk_size=1000000,
       use_gpu=True
   )
   ```

2. Check disk speed

   ```bash
   # Test disk speed
   dd if=/dev/zero of=test_file bs=1M count=1000
   ```

3. Monitor resources
   ```bash
   htop  # CPU and RAM
   iotop  # Disk I/O
   nvidia-smi  # GPU
   ```

### I/O Bottlenecks

**Problem**: Slow read/write

**Solutions**:

1. Optimize buffers

   ```python
   processor = Traitementor(
       read_buffer_size='128MB',
       write_buffer_size='128MB',
       io_threads=4
   )
   ```

2. Use fast storage

   - Prefer NVMe SSDs
   - Avoid network drives pour traiter

3. Adaptive compression
   ```python
   # Balance compression/speed
   processor = Traitementor(
       output_format='laz',
       compression_level=1
   )
   ```

## Configuration Issues

### Invalid Configuration

**Problem**: Parameter errors

```bash
ConfigurationError: Invalid feature configuration
```

**Solutions**:

1. Validate configuration

   ```python
   from ign_lidar import Config

   config = Config.from_file("config.yaml")
   config.validate()
   ```

2. Use default configuration

   ```python
   config = Config.get_default()
   config.features = ['buildings', 'vegetation']
   ```

3. Generate configuration templates
   ```bash
   # Generate template
   ign-lidar-hd config --template > config.yaml
   ```

### Path Issues

**Problem**: Files not found

```bash
FileNotFoundError: No such file or directory
```

**Solutions**:

1. Verify paths

   ```python
   import os
   assert os.path.exists("input.las"), "File not found"
   ```

2. Use absolute paths

   ```python
   input_path = os.path.abspath("input.las")
   ```

3. Check permissions
   ```bash
   ls -la input.las
   chmod 644 input.las
   ```

## Specific Issues

### RGB Augmentation

**Problem**: Color augmentation failure

```bash
OrthophotoError: Cannot read orthophoto file
```

**Solutions**:

1. Verify format

   ```bash
   gdalinfo orthophoto.tif
   ```

2. Convert format

   ```bash
   gdal_translate -of GTiff input.jp2 output.tif
   ```

3. Check georeferencing
   ```python
   from ign_lidar.utils import check_crs_match
   match = check_crs_match("input.las", "orthophoto.tif")
   ```

### Building Detection

**Problem**: Poor building detection

**Solutions**:

1. Adjust parameters

   ```python
   config = Config(
       building_detection={
           'min_points': 100,
           'height_threshold': 2.0,
           'planarity_threshold': 0.1
       }
   )
   ```

2. Adaptive preprocessing
   ```python
   processor = Traitementor(
       ground_classification=True,
       noise_removal=True
   )
   ```

## Logging and Debugging

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
processor = Traitementor(verbose=True)
```

### Save Logs

```python
import logging

logging.basicConfig(
    filename='ign_lidar.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### Debug Mode

```bash
# Debug execution
IGN_LIDAR_DEBUG=1 python script.py

# Detailed profiling
IGN_LIDAR_PROFILE=1 python script.py
```

## Support and Help

### Documentation

- [Performance Guide](./performance)
- [GPU Guide](./gpu-acceleration)
- [API Reference](../api/features)

### Diagnostic Tools

```bash
# System information
ign-lidar-hd system-info

# Configuration test
ign-lidar-hd config-test

# Data validation
ign-lidar-hd validate input.las
```

### Bug Reporting

1. Collect information

   ```bash
   ign-lidar-hd system-info > system_info.txt
   ```

2. Minimal example

   ```python
   # Minimal code reproducing the issue
   from ign_lidar import Traitementor
   processor = Traitementor()
   # Error here...
   ```

3. Test files
   - Provide small test LAS file if possible
   - Include configuration used

### Useful Resources

- **GitHub Repository**: Issues and discussions
- **Documentation**: Detailed guides and API
- **Exemples**: Sample scripts
- **Community**: Discussion forums

## Quick Solutions

### General Checklist

1. ✅ Python 3.8+ installed
2. ✅ Dependencies correctly installed
3. ✅ Valid input files
4. ✅ Read/write permissions
5. ✅ Sufficient disk space
6. ✅ Available RAM for chunks
7. ✅ Updated GPU drivers (if used)

### Useful Commands

```bash
# Quick diagnostics
ign-lidar-hd doctor

# Clear cache
ign-lidar-hd cache --clear

# Reset configuration
ign-lidar-hd config --reset

# Performance test
ign-lidar-hd benchmark --quick
```
