#!/usr/bin/env python3
"""
Quick GPU Setup Verification Script

Run this after installing RAPIDS cuML to verify everything is working correctly.

Usage:
    conda activate ign_gpu
    python scripts/verify_gpu_setup.py
"""

import sys
import platform

print("=" * 80)
print("GPU SETUP VERIFICATION")
print("=" * 80)
print()

# System info
print("🖥️  System Information:")
print(f"   Platform: {platform.system()} {platform.release()}")
print(f"   Python: {sys.version.split()[0]}")
print()

# Check CUDA/GPU availability
print("🎮 GPU Check:")
try:
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
         '--format=csv,noheader'],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        gpu_info = result.stdout.strip()
        print(f"   ✓ GPU detected: {gpu_info}")
    else:
        print("   ✗ nvidia-smi failed")
except Exception as e:
    print(f"   ✗ Could not detect GPU: {e}")
print()

# Check CuPy
print("🔧 CuPy Check:")
try:
    import cupy as cp
    print(f"   ✓ CuPy version: {cp.__version__}")
    
    # Get GPU device info
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props['name'].decode()
    total_mem = props['totalGlobalMem'] / (1024**3)
    compute_cap = device.compute_capability
    
    print(f"   ✓ GPU: {gpu_name}")
    print(f"   ✓ Memory: {total_mem:.1f} GB")
    print(f"   ✓ Compute Capability: {compute_cap}")
    
    # Test GPU operation
    test_array = cp.random.rand(1000, 1000)
    result = cp.sum(test_array)
    print(f"   ✓ GPU computation test: PASSED")
    
    cupy_ok = True
except ImportError:
    print("   ✗ CuPy not installed")
    print("     Install: conda install -c conda-forge cupy")
    cupy_ok = False
except Exception as e:
    print(f"   ✗ CuPy error: {e}")
    cupy_ok = False
print()

# Check RAPIDS cuML
print("🚀 RAPIDS cuML Check:")
try:
    import cuml
    print(f"   ✓ cuML version: {cuml.__version__}")
    
    # Test cuML NearestNeighbors
    from cuml.neighbors import NearestNeighbors
    import numpy as np
    
    test_data = np.random.rand(1000, 3).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(test_data)
    distances, indices = knn.kneighbors(test_data[:10])
    print(f"   ✓ cuML NearestNeighbors test: PASSED")
    
    # Test cuML PCA
    from cuml.decomposition import PCA
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(test_data)
    print(f"   ✓ cuML PCA test: PASSED")
    
    cuml_ok = True
except ImportError:
    print("   ✗ RAPIDS cuML not installed")
    print("     Install: conda install -c rapidsai -c conda-forge -c nvidia cuml")  # noqa: E501
    cuml_ok = False
except Exception as e:
    print(f"   ✗ cuML error: {e}")
    cuml_ok = False
print()

# Check IGN LiDAR HD integration
print("📦 IGN LiDAR HD Integration:")
try:
    from ign_lidar.features_gpu import GPU_AVAILABLE, CUML_AVAILABLE
    print(f"   ✓ GPU_AVAILABLE: {GPU_AVAILABLE}")
    print(f"   ✓ CUML_AVAILABLE: {CUML_AVAILABLE}")
    
    if GPU_AVAILABLE and CUML_AVAILABLE:
        print("   ✓ Full GPU mode ready! (12-20x speedup expected)")
    elif GPU_AVAILABLE:
        print("   ⚠ Hybrid GPU mode (6-8x speedup expected)")
        print("     Install RAPIDS cuML for full acceleration")
    else:
        print("   ⚠ CPU mode only")
    
    ign_ok = True
except ImportError as e:
    print(f"   ✗ Could not import ign_lidar: {e}")
    print("     Install: pip install -e .")
    ign_ok = False
except Exception as e:
    print(f"   ✗ Error: {e}")
    ign_ok = False
print()

# Summary
print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)

all_checks = []
if cupy_ok:
    all_checks.append(("CuPy (GPU Arrays)", "✓ Installed"))
else:
    all_checks.append(("CuPy (GPU Arrays)", "✗ Missing"))

if cuml_ok:
    all_checks.append(("RAPIDS cuML (GPU ML)", "✓ Installed"))
else:
    all_checks.append(("RAPIDS cuML (GPU ML)", "✗ Missing"))

if ign_ok:
    all_checks.append(("IGN LiDAR HD", "✓ Ready"))
else:
    all_checks.append(("IGN LiDAR HD", "✗ Not ready"))

for check_name, status in all_checks:
    print(f"{check_name:<30} {status}")

print()

# Recommendations
if cupy_ok and cuml_ok and ign_ok:
    print("🎉 EXCELLENT! Full GPU acceleration is ready!")
    print()
    print("Next steps:")
    print("  1. Run benchmark:")
    print("     python scripts/benchmarks/profile_gpu_bottlenecks.py --points 1000000")  # noqa: E501
    print()
    print("  2. Process your data:")
    print("     ign-lidar-hd enrich --input file.laz --output dir --use-gpu")
    print()
    print("Expected performance: 12-20x faster than CPU!")
    
elif cupy_ok and ign_ok:
    print("⚡ GOOD! Hybrid GPU mode is ready (6-8x speedup)")
    print()
    print("For maximum performance (12-20x), install RAPIDS cuML:")
    print("  conda install -c rapidsai -c conda-forge -c nvidia cuml=24.10")
    print()
    print("Current mode is still fast! You can use:")
    print("  ign-lidar-hd enrich --input file.laz --output dir --use-gpu")
    
else:
    print("⚠️  Setup incomplete. Please install missing components:")
    print()
    if not cupy_ok:
        print("  1. Install CuPy:")
        print("     conda install -c conda-forge cupy-cuda12x")
        print()
    if not cuml_ok:
        print("  2. Install RAPIDS cuML (optional, for max performance):")
        print("     conda install -c rapidsai -c conda-forge -c nvidia cuml=24.10")  # noqa: E501
        print()
    if not ign_ok:
        print("  3. Install IGN LiDAR HD:")
        print("     pip install -e .")
        print()

print("=" * 80)
