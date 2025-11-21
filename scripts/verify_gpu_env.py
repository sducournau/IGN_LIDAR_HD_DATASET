#!/usr/bin/env python3
"""
Verify GPU Environment Configuration

This script checks if you're running in the correct environment for GPU operations.

Usage:
    # Correct usage (will pass all checks):
    conda run -n ign_gpu python scripts/verify_gpu_env.py

    # Wrong usage (will show warnings):
    python scripts/verify_gpu_env.py
"""

import sys
import os
from pathlib import Path


def print_banner():
    """Print verification banner."""
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "        GPU Environment Verification".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()


def check_conda_env():
    """Check if running in ign_gpu conda environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    
    print("🔍 Checking Conda Environment...")
    print(f"   Current environment: {conda_env}")
    
    if conda_env == "ign_gpu":
        print("   ✅ CORRECT: Running in ign_gpu environment")
        return True
    else:
        print("   ❌ WRONG: Not running in ign_gpu environment")
        print()
        print("   📝 To fix, use:")
        print("      conda run -n ign_gpu python scripts/verify_gpu_env.py")
        print()
        return False


def check_gpu_libraries():
    """Check if GPU libraries are available."""
    print("\n🔍 Checking GPU Libraries...")
    
    # Core libraries (required)
    core_libraries = {
        "CuPy": "cupy",
        "RAPIDS cuML": "cuml",
        "FAISS-GPU": "faiss",
    }
    
    # Optional libraries
    optional_libraries = {
        "RAPIDS cuSpatial": "cuspatial",
    }
    
    core_available = True
    optional_available = True
    
    for name, module_name in core_libraries.items():
        try:
            __import__(module_name)
            print(f"   ✅ {name}: Available")
        except ImportError:
            print(f"   ❌ {name}: NOT available (REQUIRED)")
            core_available = False
    
    for name, module_name in optional_libraries.items():
        try:
            __import__(module_name)
            print(f"   ✅ {name}: Available")
        except ImportError:
            print(f"   ⚠️  {name}: NOT available (optional)")
            optional_available = False
    
    if not core_available:
        print()
        print("   📝 Missing REQUIRED libraries - make sure you're in ign_gpu environment:")
        print("      conda run -n ign_gpu python scripts/verify_gpu_env.py")
        print()
    elif not optional_available:
        print()
        print("   💡 Optional libraries missing (GPU operations will still work)")
        print()
    
    return core_available


def check_gpu_device():
    """Check if GPU device is accessible."""
    print("\n🔍 Checking GPU Device...")
    
    try:
        import cupy as cp
        
        # Try to access GPU
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"   ✅ GPU devices found: {device_count}")
        
        if device_count > 0:
            device = cp.cuda.Device(0)
            device_name = device.attributes.get('Name', 'Unknown')
            # Handle both bytes and str (CuPy versions differ)
            if isinstance(device_name, bytes):
                device_name = device_name.decode()
            print(f"   ✅ GPU 0: {device_name}")
            
            # Get memory info
            mem_info = cp.cuda.runtime.memGetInfo()
            free_gb = mem_info[0] / 1024**3
            total_gb = mem_info[1] / 1024**3
            print(f"   ✅ GPU Memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
            
            return True
        else:
            print("   ⚠️  No GPU devices detected")
            return False
            
    except Exception as e:
        print(f"   ❌ GPU check failed: {e}")
        print()
        print("   📝 Possible causes:")
        print("      - NVIDIA drivers not installed")
        print("      - CUDA not installed")
        print("      - Not running in ign_gpu environment")
        print()
        return False


def check_project_files():
    """Check if GPU documentation files exist."""
    print("\n🔍 Checking Documentation Files...")
    
    root = Path(__file__).parent.parent
    docs = {
        "GPU_QUICK_REFERENCE.md": "Quick command reference",
        "GPU_REMINDER.txt": "Visual reminder banner",
        "GPU_ENVIRONMENT_GUIDE.md": "Comprehensive guide",
        "GPU_ENVIRONMENT_CONFIGURATION.md": "Configuration summary",
    }
    
    all_exist = True
    
    for filename, description in docs.items():
        filepath = root / filename
        if filepath.exists():
            print(f"   ✅ {filename}: {description}")
        else:
            print(f"   ❌ {filename}: Missing")
            all_exist = False
    
    return all_exist


def print_summary(env_ok, libs_ok, gpu_ok, docs_ok):
    """Print verification summary."""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY".center(70))
    print("=" * 70)
    
    checks = [
        ("Conda Environment", env_ok),
        ("GPU Libraries", libs_ok),
        ("GPU Device", gpu_ok),
        ("Documentation", docs_ok),
    ]
    
    all_passed = all(status for _, status in checks)
    
    for check_name, status in checks:
        symbol = "✅" if status else "❌"
        status_text = "PASS" if status else "FAIL"
        print(f"   {symbol} {check_name}: {status_text}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All checks passed! GPU environment is correctly configured.")
        print("\n📚 Next steps:")
        print("   • Run benchmarks: conda run -n ign_gpu python scripts/benchmark_*.py --gpu")
        print("   • Run tests: conda run -n ign_gpu pytest tests/test_gpu_*.py")
        print("   • See GPU_QUICK_REFERENCE.md for more commands")
    else:
        print("\n⚠️  Some checks failed. See messages above for details.")
        print("\n📚 Resources:")
        print("   • GPU_QUICK_REFERENCE.md - Quick command reference")
        print("   • GPU_ENVIRONMENT_GUIDE.md - Comprehensive setup guide")
        print("   • GPU_REMINDER.txt - Visual quick reference")
    
    print()
    
    return 0 if all_passed else 1


def main():
    """Main verification function."""
    print_banner()
    
    # Run checks
    env_ok = check_conda_env()
    libs_ok = check_gpu_libraries()
    gpu_ok = check_gpu_device()
    docs_ok = check_project_files()
    
    # Print summary
    exit_code = print_summary(env_ok, libs_ok, gpu_ok, docs_ok)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
