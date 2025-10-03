---
sidebar_position: 4
---

# GPU Acceleration

Learn how to accelerate LiDAR processing using GPU computing with CUDA support.

## Overview

The IGN LiDAR HD library supports GPU acceleration for computationally intensive operations like feature extraction and point cloud processing.

## Requirements

- NVIDIA GPU with CUDA capability
- CUDA Toolkit 11.0 or higher
- GPU-specific Python packages

## Installation

Install the GPU version of the library:

```bash
pip install -r requirements_gpu.txt
```

## Configuration

Enable GPU processing in your configuration:

```python
from ign_lidar import Config

config = Config(
    use_gpu=True,
    gpu_memory_limit=0.8,  # Use 80% of GPU memory
    cuda_device=0  # Use first GPU
)
```

## Performance Benefits

GPU acceleration provides significant performance improvements:

- Feature extraction: Up to 10x faster
- Point cloud filtering: Up to 5x faster
- Geometric calculations: Up to 8x faster

## Monitoring GPU Usage

Monitor GPU utilization during processing:

```bash
nvidia-smi
```

## Troubleshooting

Common GPU-related issues and solutions:

### CUDA Out of Memory

Reduce batch size or memory limit:

```python
config.gpu_memory_limit = 0.5  # Use only 50% of GPU memory
```

### GPU Not Detected

Verify CUDA installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
