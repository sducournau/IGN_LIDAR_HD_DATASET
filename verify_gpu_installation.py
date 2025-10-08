#!/usr/bin/env python3
"""
Verify GPU setup for IGN LiDAR HD
"""
import sys
import warnings

# Suppress the CUDA deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='cuda')

print("\n🔍 Verifying GPU installation...\n")
print("   Python:", sys.version.split()[0])

try:
    import cupy as cp
    print(f"   ✓ CuPy: {cp.__version__}")
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props['name'].decode()
    total_mem = props['totalGlobalMem'] / (1024**3)
    compute_capability = f"{props['major']}.{props['minor']}"
    print(f"   ✓ GPU: {gpu_name} ({total_mem:.1f} GB)")
    print(f"     - Compute Capability: {compute_capability}")
except Exception as e:
    print(f"   ✗ CuPy error: {e}")
    sys.exit(1)

try:
    import cuml
    print(f"   ✓ RAPIDS cuML: {cuml.__version__}")
except Exception as e:
    print(f"   ✗ RAPIDS cuML error: {e}")
    sys.exit(1)

try:
    import torch
    print(f"   ✓ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"     - CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"     - CUDA version: {torch.version.cuda}")
        print(f"     - cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("   ⚠️  Warning: PyTorch installed but CUDA not available")
except Exception as e:
    print(f"   ✗ PyTorch error: {e}")
    sys.exit(1)

try:
    from ign_lidar.features.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE
    import ign_lidar
    print(f"   ✓ IGN LiDAR HD: {ign_lidar.__version__}")
    print(f"     - GPU_AVAILABLE: {GPU_AVAILABLE}")
    print(f"     - CUML_AVAILABLE: {CUML_AVAILABLE}")
    if not (GPU_AVAILABLE and CUML_AVAILABLE):
        print("   ⚠️  Warning: GPU or cuML not properly detected by IGN LiDAR")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ IGN LiDAR HD error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n   🎉 All components verified successfully!")
print("\n   Environment: ign_gpu")
print("   Ready for GPU-accelerated processing!")
