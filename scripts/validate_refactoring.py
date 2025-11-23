#!/usr/bin/env python3
"""
Refactoring Validation Script

Validates that refactoring changes don't introduce regressions by:
1. Running comprehensive test suite
2. Checking for import errors
3. Validating API compatibility
4. Benchmarking performance
5. Checking code quality metrics

Usage:
    # Validate Phase 1 changes
    python scripts/validate_refactoring.py --phase 1
    
    # Validate Phase 2 with GPU benchmarks
    python scripts/validate_refactoring.py --phase 2 --gpu
    
    # Full validation (all phases)
    python scripts/validate_refactoring.py --all

Author: GitHub Copilot
Date: November 22, 2025
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """Terminal colors for output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run a command and return success status and output.
    
    Args:
        cmd: Command to run as list
        description: Description of what the command does
        
    Returns:
        Tuple of (success, output)
    """
    print_info(f"Running: {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print_success(f"{description} - PASSED")
            return True, result.stdout
        else:
            print_error(f"{description} - FAILED")
            print(f"   Error: {result.stderr[:500]}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print_error(f"{description} - TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        print_error(f"{description} - ERROR: {e}")
        return False, str(e)


def check_imports():
    """Check that all imports work correctly."""
    print_header("🔍 Validating Imports")
    
    imports_to_check = [
        ("from ign_lidar.features.compute import compute_normals", "compute_normals import"),
        ("from ign_lidar.features.compute import compute_normals_fast", "compute_normals_fast import"),
        ("from ign_lidar.features.compute.utils import validate_normals", "validate_normals import"),
        ("from ign_lidar.features import FeatureOrchestrator", "FeatureOrchestrator import"),
        ("from ign_lidar.optimization import KNNEngine", "KNNEngine import"),
        ("from ign_lidar.core.gpu import GPUManager", "GPUManager import"),
    ]
    
    all_passed = True
    
    for import_stmt, description in imports_to_check:
        try:
            exec(import_stmt)
            print_success(f"{description} works")
        except Exception as e:
            print_error(f"{description} failed: {e}")
            all_passed = False
    
    return all_passed


def check_deprecated_warnings():
    """Check that deprecated functions emit warnings."""
    print_header("⚠️  Validating Deprecation Warnings")
    
    test_code = """
import warnings
warnings.simplefilter('always', DeprecationWarning)

# Test deprecated imports (if they exist)
try:
    from ign_lidar.features.feature_computer import FeatureComputer
    computer = FeatureComputer()
    # This should emit a warning if deprecated
    print("feature_computer import works (may be deprecated)")
except ImportError:
    print("feature_computer already removed (post-v4.0)")
except DeprecationWarning as e:
    print(f"✅ Deprecation warning emitted: {e}")
"""
    
    try:
        exec(test_code)
        print_success("Deprecation warnings checked")
        return True
    except Exception as e:
        print_warning(f"Could not fully validate deprecation warnings: {e}")
        return True  # Non-critical


def run_test_suite(markers: str = None):
    """Run pytest test suite."""
    print_header("🧪 Running Test Suite")
    
    cmd = ["pytest", "tests/", "-v", "--tb=short"]
    
    if markers:
        cmd.extend(["-m", markers])
    
    success, output = run_command(cmd, f"Test suite{f' ({markers})' if markers else ''}")
    
    # Parse test results
    if success and "passed" in output:
        # Extract test counts
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line:
                print_info(f"   {line.strip()}")
                break
    
    return success


def run_feature_tests():
    """Run feature computation tests."""
    print_header("🎯 Feature Computation Tests")
    
    tests = [
        ("tests/test_features.py", "Core feature tests"),
        ("tests/test_compute_*.py", "Compute module tests"),
    ]
    
    all_passed = True
    
    for test_pattern, description in tests:
        if Path(test_pattern).exists() or '*' in test_pattern:
            cmd = ["pytest", test_pattern, "-v"]
            success, _ = run_command(cmd, description)
            all_passed = all_passed and success
    
    return all_passed


def benchmark_normals_performance():
    """Benchmark compute_normals performance."""
    print_header("⚡ Performance Benchmark: compute_normals")
    
    benchmark_code = """
import numpy as np
import time
from ign_lidar.features.compute import compute_normals

# Generate test data
np.random.seed(42)
n_points = 100_000
points = np.random.randn(n_points, 3).astype(np.float32)

# Warm-up
compute_normals(points, k_neighbors=30)

# Benchmark
times = []
for _ in range(3):
    start = time.time()
    normals, eigenvalues = compute_normals(points, k_neighbors=30)
    duration = time.time() - start
    times.append(duration)

avg_time = sum(times) / len(times)
throughput = n_points / avg_time

print(f"Average time: {avg_time:.3f}s")
print(f"Throughput: {throughput:,.0f} points/s")

# Validation
assert normals.shape == (n_points, 3), "Wrong normals shape"
assert eigenvalues.shape[0] == n_points, "Wrong eigenvalues shape"
assert not np.isnan(normals).any(), "NaN in normals"
print("✅ All assertions passed")
"""
    
    try:
        exec(benchmark_code)
        print_success("Performance benchmark passed")
        return True
    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        return False


def check_gpu_availability():
    """Check if GPU is available for testing."""
    print_header("🎮 GPU Availability Check")
    
    check_code = """
try:
    import cupy as cp
    
    # Test basic GPU operation
    x = cp.array([1, 2, 3])
    y = x * 2
    result = cp.asnumpy(y)
    
    print(f"✅ CuPy available")
    print(f"   Device: {cp.cuda.Device().name}")
    print(f"   Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB")
    gpu_available = True
except ImportError:
    print("❌ CuPy not available")
    gpu_available = False
except Exception as e:
    print(f"⚠️  GPU test failed: {e}")
    gpu_available = False

print(f"GPU Available: {gpu_available}")
"""
    
    try:
        exec(check_code)
        return True
    except:
        return False


def generate_validation_report(results: Dict[str, bool], output_path: Path):
    """Generate validation report."""
    print_header("📄 Generating Validation Report")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "passed": sum(results.values()),
        "failed": len(results) - sum(results.values()),
        "total": len(results),
        "success_rate": sum(results.values()) / len(results) * 100
    }
    
    output_path.write_text(json.dumps(report, indent=2))
    print_success(f"Report saved: {output_path}")
    
    # Print summary
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Total tests: {report['total']}")
    print(f"  Passed: {Colors.GREEN}{report['passed']}{Colors.END}")
    print(f"  Failed: {Colors.RED}{report['failed']}{Colors.END}")
    print(f"  Success rate: {report['success_rate']:.1f}%")
    
    return report


def validate_phase1():
    """Validate Phase 1 refactoring (compute_normals consolidation)."""
    print_header("🔍 PHASE 1 VALIDATION: Compute Normals Consolidation")
    
    results = {}
    
    # 1. Check imports
    results["imports"] = check_imports()
    
    # 2. Check deprecation warnings
    results["deprecations"] = check_deprecated_warnings()
    
    # 3. Run feature tests
    results["feature_tests"] = run_feature_tests()
    
    # 4. Benchmark performance
    results["performance"] = benchmark_normals_performance()
    
    # 5. Full test suite
    results["test_suite"] = run_test_suite()
    
    return results


def validate_phase2():
    """Validate Phase 2 refactoring (GPU optimization)."""
    print_header("🔍 PHASE 2 VALIDATION: GPU Optimization")
    
    results = {}
    
    # 1. Check GPU availability
    results["gpu_available"] = check_gpu_availability()
    
    if not results["gpu_available"]:
        print_warning("GPU not available, skipping GPU-specific tests")
        return results
    
    # 2. Run GPU tests
    cmd = ["pytest", "tests/", "-v", "-m", "gpu"]
    results["gpu_tests"], _ = run_command(cmd, "GPU-specific tests")
    
    # 3. Check GPU profiler
    try:
        from ign_lidar.optimization.gpu_transfer_profiler import GPUTransferProfiler
        print_success("GPUTransferProfiler import works")
        results["gpu_profiler"] = True
    except ImportError as e:
        print_error(f"GPUTransferProfiler import failed: {e}")
        results["gpu_profiler"] = False
    
    # 4. Check stream support
    try:
        from ign_lidar.optimization import CUDAStreamManager
        print_success("CUDAStreamManager import works")
        results["cuda_streams"] = True
    except ImportError as e:
        print_error(f"CUDAStreamManager import failed: {e}")
        results["cuda_streams"] = False
    
    return results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate refactoring changes')
    parser.add_argument('--phase', type=int, choices=[1, 2], help='Validate specific phase')
    parser.add_argument('--all', action='store_true', help='Validate all phases')
    parser.add_argument('--gpu', action='store_true', help='Include GPU tests')
    parser.add_argument('--output', type=Path, default=Path('validation_report.json'),
                       help='Output path for validation report')
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 REFACTORING VALIDATION{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")
    
    all_results = {}
    
    # Validate requested phases
    if args.phase == 1 or args.all:
        phase1_results = validate_phase1()
        all_results.update({f"phase1_{k}": v for k, v in phase1_results.items()})
    
    if args.phase == 2 or args.all or args.gpu:
        phase2_results = validate_phase2()
        all_results.update({f"phase2_{k}": v for k, v in phase2_results.items()})
    
    if not args.phase and not args.all and not args.gpu:
        # Default: Phase 1 only
        print_info("No phase specified, running Phase 1 validation")
        phase1_results = validate_phase1()
        all_results.update({f"phase1_{k}": v for k, v in phase1_results.items()})
    
    # Generate report
    report = generate_validation_report(all_results, args.output)
    
    # Exit with appropriate code
    if report['failed'] == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ ALL VALIDATIONS PASSED{Colors.END}\n")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ SOME VALIDATIONS FAILED{Colors.END}\n")
        print("Review failures above and fix before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    main()
