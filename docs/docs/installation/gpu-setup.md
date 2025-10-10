---
sidebar_position: 2
title: GPU Setup
description: Complete guide for setting up GPU acceleration for LiDAR processing
keywords: [gpu, cuda, installation, setup, nvidia, acceleration]
---

# GPU Setup Guide

Complete instructions for setting up GPU acceleration to dramatically improve LiDAR processing performance.

## System Requirements

### Hardware Requirements

**Minimum Requirements:**

- NVIDIA GPU with CUDA Compute Capability 3.5+
- 4GB VRAM minimum (8GB+ recommended)
- PCIe 3.0 x16 slot

**Recommended Requirements:**

- NVIDIA RTX 3060/4060 or better
- 12GB+ VRAM for large datasets
- PCIe 4.0 x16 for maximum bandwidth

**Supported GPUs:**

```bash
# Check GPU compatibility
nvidia-smi
```

### Software Requirements

- **CUDA Toolkit**: 11.8+ or 12.x
- **NVIDIA Driver**: 520.61.05+ (Linux) / 527.41+ (Windows)
- **Python**: 3.8-3.12
- **PyTorch**: 2.0+ with CUDA support

## Installation Steps

### Step 1: Install NVIDIA Drivers

#### Linux (Ubuntu/Debian)

```bash
# Add NVIDIA repository
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install latest driver
sudo apt install nvidia-driver-535
sudo reboot

# Verify installation
nvidia-smi
```

#### Windows

1. Download latest drivers from [NVIDIA Website](https://www.nvidia.com/drivers)
2. Run installer with default settings
3. Reboot system
4. Verify with `nvidia-smi` in Command Prompt

### Step 2: Install CUDA Toolkit

#### Option A: Conda Installation (Recommended)

```bash
# Create new environment with CUDA
conda create -n ign-gpu python=3.11
conda activate ign-gpu

# Install CUDA toolkit
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

#### Option B: Native CUDA Installation

```bash
# Download CUDA 12.1 from NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

### Step 3: Install IGN LiDAR HD with GPU Support

```bash
# Install with GPU dependencies
pip install ign-lidar-hd[gpu]

# Or install development version
pip install -e .[gpu]

# Verify GPU support
ign-lidar-hd system-info --gpu
```

## Configuration

### Environment Variables

```bash
# Set GPU memory allocation
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optional: Limit GPU memory usage
export IGN_GPU_MEMORY_FRACTION=0.8  # Use 80% of GPU memory
```

### GPU Configuration File

Create `~/.ign-lidar/gpu-config.yaml`:

```yaml
gpu:
  enabled: true
  device: "cuda:0" # GPU device to use
  memory_fraction: 0.8 # Fraction of GPU memory to use
  batch_size: 10000 # Points per GPU batch

  # Performance tuning
  pin_memory: true
  non_blocking: true
  compile_models: true # PyTorch 2.0+ compilation

  # Fallback options
  fallback_to_cpu: true
  auto_mixed_precision: true
```

### Multi-GPU Configuration

```yaml
gpu:
  enabled: true
  multi_gpu: true
  devices: ["cuda:0", "cuda:1"] # Multiple GPUs
  data_parallel: true

  # Load balancing
  gpu_weights: [1.0, 0.8] # Relative performance weights
```

## Verification and Testing

### Basic GPU Test

```python
from ign_lidar import Processor
import torch

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")

# Test GPU processing
processor = Processor(use_gpu=True)
print(f"GPU Enabled: {processor.gpu_enabled}")
print(f"GPU Device: {processor.device}")
```

### Performance Benchmark

```bash
# Run GPU benchmark
ign-lidar-hd benchmark --gpu --dataset-size large

# Compare CPU vs GPU performance
ign-lidar-hd benchmark --compare-devices
```

### Memory Test

```python
from ign_lidar.gpu import GPUMemoryManager

# Check GPU memory
memory_manager = GPUMemoryManager()
memory_info = memory_manager.get_memory_info()

print(f"Total GPU Memory: {memory_info['total']:.2f} GB")
print(f"Available Memory: {memory_info['available']:.2f} GB")
print(f"Recommended Batch Size: {memory_info['recommended_batch_size']}")
```

## Performance Optimization

### Memory Management

```python
from ign_lidar import Config

# Optimize for available GPU memory
config = Config(
    gpu_enabled=True,
    gpu_memory_fraction=0.8,

    # Batch processing configuration
    gpu_batch_size="auto",  # Auto-detect optimal size
    pin_memory=True,

    # Memory cleanup
    clear_cache_every=100,  # Clear GPU cache every N batches
)
```

### Processing Optimization

```python
# Enable mixed precision for better performance
config = Config(
    gpu_enabled=True,
    mixed_precision=True,  # Use FP16 where possible
    compile_models=True,   # PyTorch 2.0 compilation

    # Optimize data loading
    num_workers=4,
    prefetch_factor=2,
)
```

## Troubleshooting

### Common Issues

#### CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Out of Memory Errors

```python
# Reduce batch size
processor = Processor(
    use_gpu=True,
    gpu_batch_size=5000,  # Smaller batch
    gpu_memory_fraction=0.6  # Use less GPU memory
)

# Or use gradient checkpointing
processor = Processor(
    use_gpu=True,
    gradient_checkpointing=True
)
```

#### Driver Compatibility Issues

```bash
# Check driver version
nvidia-smi

# Verify CUDA compatibility
nvidia-smi --query-gpu=compute_cap --format=csv

# Update drivers if needed
sudo apt update && sudo apt upgrade nvidia-driver-535
```

### Performance Issues

#### Slow GPU Processing

```python
# Enable optimizations
import torch
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 on Ampere GPUs

# Use compiled models (PyTorch 2.0+)
processor = Processor(
    use_gpu=True,
    compile_models=True
)
```

#### CPU-GPU Transfer Bottlenecks

```python
# Optimize data transfer
config = Config(
    gpu_enabled=True,
    pin_memory=True,      # Faster CPU-GPU transfers
    non_blocking=True,    # Asynchronous transfers
    prefetch_to_gpu=True  # Prefetch data to GPU
)
```

## Monitoring and Profiling

### GPU Usage Monitoring

```bash
# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Log GPU usage to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 1 > gpu_usage.log
```

### Performance Profiling

```python
from ign_lidar.profiling import GPUProfiler

# Profile GPU performance
with GPUProfiler() as profiler:
    result = processor.process_tile("input.las")

# View profiling results
profiler.print_summary()
profiler.save_report("gpu_profile.html")
```

### Bottleneck Analysis

```python
# Identify bottlenecks
from ign_lidar.diagnostics import analyze_gpu_performance

analysis = analyze_gpu_performance(
    las_file="sample.las",
    config=config
)

print(f"GPU Utilization: {analysis['gpu_utilization']:.1f}%")
print(f"Memory Efficiency: {analysis['memory_efficiency']:.1f}%")
print(f"Bottleneck: {analysis['primary_bottleneck']}")
```

## Multi-GPU Setup

### Data Parallel Processing

```python
from ign_lidar import MultiGPUProcessor

# Configure multiple GPUs
processor = MultiGPUProcessor(
    devices=["cuda:0", "cuda:1"],
    strategy="data_parallel"
)

# Process with multiple GPUs
results = processor.process_batch(tile_list)
```

### Load Balancing

```yaml
# Multi-GPU configuration
multi_gpu:
  enabled: true
  devices: ["cuda:0", "cuda:1", "cuda:2"]

  # Load balancing based on GPU performance
  device_weights:
    "cuda:0": 1.0 # RTX 4090
    "cuda:1": 0.8 # RTX 3080
    "cuda:2": 0.6 # RTX 2080

  # Synchronization settings
  sync_batch_norm: true
  find_unused_parameters: false
```

## Cloud GPU Setup

### Google Colab

```python
# Check GPU availability in Colab
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Install IGN LiDAR HD
!pip install ign-lidar-hd[gpu]

# Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive')
```

### AWS EC2 GPU Instances

```bash
# Launch GPU instance (p3.2xlarge recommended)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-group-ids sg-your-security-group

# Install CUDA and IGN LiDAR HD
sudo apt update
sudo apt install nvidia-driver-535
pip install ign-lidar-hd[gpu]
```

## Best Practices

### Development Guidelines

1. **Always Check GPU Availability**

   ```python
   if not torch.cuda.is_available():
       print("GPU not available, falling back to CPU")
       use_gpu = False
   ```

2. **Monitor Memory Usage**

   ```python
   # Clear cache periodically
   if batch_count % 100 == 0:
       torch.cuda.empty_cache()
   ```

3. **Use Mixed Precision**

   ```python
   # Enable automatic mixed precision
   from torch.cuda.amp import autocast

   with autocast():
       result = model(input_data)
   ```

### Production Deployment

1. **Resource Allocation**

   - Reserve GPU memory for other processes
   - Set appropriate batch sizes
   - Monitor temperature and power consumption

2. **Error Handling**

   ```python
   try:
       result = gpu_processor.process(data)
   except torch.cuda.OutOfMemoryError:
       torch.cuda.empty_cache()
       result = cpu_processor.process(data)
   ```

3. **Monitoring**
   - Log GPU utilization
   - Track processing throughput
   - Monitor for thermal throttling

## Related Documentation

- [GPU Acceleration Guide](../guides/gpu-acceleration)
- [Performance Optimization](../guides/performance)
- [GPU API Reference](../api/gpu-api)
- [Troubleshooting Guide](../guides/troubleshooting)
