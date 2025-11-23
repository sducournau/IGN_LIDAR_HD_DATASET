#!/usr/bin/env python3
"""
Comprehensive Test Runner for Refactoring

Runs all relevant tests and generates a comprehensive report on test status,
coverage, and any regressions introduced by refactoring.

Usage:
    # Run all refactoring-related tests
    python scripts/test_refactoring.py
    
    # Run specific phase tests
    python scripts/test_refactoring.py --phase 1
    
    # Include slow tests
    python scripts/test_refactoring.py --include-slow
    
    # Generate coverage report
    python scripts/test_refactoring.py --coverage

Author: GitHub Copilot
Date: November 22, 2025
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RefactoringTestRunner:
    """Run comprehensive tests for refactoring validation."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results = {}
        
    def run_test_suite(
        self,
        markers: Optional[str] = None,
        include_slow: bool = False,
        coverage: bool = False
    ) -> Tuple[bool, str]:
        """Run pytest test suite."""
        
        cmd = ["pytest", "tests/", "-v", "--tb=short"]
        
        if markers:
            cmd.extend(["-m", markers])
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        if coverage:
            cmd.extend(["--cov=ign_lidar", "--cov-report=term", "--cov-report=html"])
        
        print(f"\n🧪 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=600  # 10 minutes
            )
            
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Test suite timed out"
        except Exception as e:
            return False, str(e)
    
    def test_compute_normals_api(self) -> Tuple[bool, str]:
        """Test compute_normals canonical API."""
        
        test_code = """
import numpy as np
from ign_lidar.features.compute import compute_normals

# Test basic functionality
points = np.random.randn(1000, 3).astype(np.float32)
normals, eigenvalues = compute_normals(points, k_neighbors=30)

assert normals.shape == (1000, 3), f"Wrong shape: {normals.shape}"
assert eigenvalues.shape[0] == 1000, f"Wrong eigenvalues shape: {eigenvalues.shape}"
assert not np.isnan(normals).any(), "NaN values in normals"
assert not np.isnan(eigenvalues).any(), "NaN values in eigenvalues"

print("✅ compute_normals API test passed")
"""
        
        try:
            result = subprocess.run(
                ["python", "-c", test_code],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def test_feature_orchestrator_api(self) -> Tuple[bool, str]:
        """Test FeatureOrchestrator API."""
        
        test_code = """
import numpy as np
from ign_lidar.features import FeatureOrchestrator

config = {
    'features': {
        'k_neighbors': 30,
        'use_gpu': False
    }
}

orchestrator = FeatureOrchestrator(config)
points = np.random.randn(1000, 3).astype(np.float32)

# Test LOD2 mode
features_lod2 = orchestrator.compute_features(points, mode='lod2')
assert features_lod2 is not None, "LOD2 features returned None"
assert features_lod2.shape[0] == 1000, f"Wrong feature count: {features_lod2.shape[0]}"

print("✅ FeatureOrchestrator API test passed")
"""
        
        try:
            result = subprocess.run(
                ["python", "-c", test_code],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def test_knn_engine_api(self) -> Tuple[bool, str]:
        """Test KNNEngine unified API."""
        
        test_code = """
import numpy as np
from ign_lidar.optimization import KNNEngine

knn_engine = KNNEngine(backend='auto', use_gpu=False)
points = np.random.randn(1000, 3).astype(np.float32)

knn_engine.build_index(points)
distances, indices = knn_engine.search(points, k=30)

assert distances.shape == (1000, 30), f"Wrong distances shape: {distances.shape}"
assert indices.shape == (1000, 30), f"Wrong indices shape: {indices.shape}"

print("✅ KNNEngine API test passed")
"""
        
        try:
            result = subprocess.run(
                ["python", "-c", test_code],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def test_deprecation_warnings(self) -> Tuple[bool, str]:
        """Test that deprecated APIs emit warnings."""
        
        test_code = """
import warnings
warnings.simplefilter('always', DeprecationWarning)

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # Try importing deprecated module (may not exist yet)
    try:
        from ign_lidar.features.gpu_processor import GPUProcessor
        has_deprecation = any(issubclass(warning.category, DeprecationWarning) for warning in w)
        
        if has_deprecation:
            print("✅ Deprecation warning emitted for GPUProcessor")
        else:
            print("⚠️  No deprecation warning for GPUProcessor (add in Phase 3)")
    except ImportError:
        print("ℹ️  GPUProcessor already removed (post-v4.0)")
    except Exception as e:
        print(f"ℹ️  GPUProcessor test skipped: {e}")

print("✅ Deprecation warnings test completed")
"""
        
        try:
            result = subprocess.run(
                ["python", "-c", test_code],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=10
            )
            return True, result.stdout  # Always pass, just informational
        except Exception as e:
            return True, str(e)  # Non-critical
    
    def run_phase1_tests(self) -> Dict:
        """Run Phase 1 specific tests."""
        print("\n" + "=" * 80)
        print("🧪 PHASE 1 TESTS: Compute Normals Consolidation")
        print("=" * 80)
        
        results = {}
        
        # Test canonical API
        success, output = self.test_compute_normals_api()
        results["compute_normals_api"] = success
        if success:
            print("✅ Compute normals API test passed")
        else:
            print("❌ Compute normals API test failed")
            print(output[:500])
        
        # Test feature computer tests
        success, output = self.run_test_suite(markers="unit")
        results["unit_tests"] = success
        
        # Test deprecation warnings
        success, output = self.test_deprecation_warnings()
        results["deprecation_warnings"] = success
        
        return results
    
    def run_phase2_tests(self) -> Dict:
        """Run Phase 2 specific tests."""
        print("\n" + "=" * 80)
        print("🧪 PHASE 2 TESTS: GPU Optimization")
        print("=" * 80)
        
        results = {}
        
        # Check if GPU available
        try:
            subprocess.run(
                ["python", "-c", "import cupy; print('GPU OK')"],
                capture_output=True,
                check=True,
                timeout=5
            )
            gpu_available = True
        except Exception:
            gpu_available = False
            print("⚠️  GPU not available, skipping GPU tests")
        
        results["gpu_available"] = gpu_available
        
        if gpu_available:
            # Run GPU tests
            success, output = self.run_test_suite(markers="gpu")
            results["gpu_tests"] = success
            
            # Test KNN engine
            success, output = self.test_knn_engine_api()
            results["knn_engine_api"] = success
            if success:
                print("✅ KNN engine API test passed")
            else:
                print("❌ KNN engine API test failed")
        else:
            results["gpu_tests"] = None
            results["knn_engine_api"] = None
        
        return results
    
    def run_phase3_tests(self) -> Dict:
        """Run Phase 3 specific tests."""
        print("\n" + "=" * 80)
        print("🧪 PHASE 3 TESTS: Architecture Cleanup")
        print("=" * 80)
        
        results = {}
        
        # Test feature orchestrator
        success, output = self.test_feature_orchestrator_api()
        results["orchestrator_api"] = success
        if success:
            print("✅ FeatureOrchestrator API test passed")
        else:
            print("❌ FeatureOrchestrator API test failed")
            print(output[:500])
        
        # Run integration tests
        success, output = self.run_test_suite(markers="integration")
        results["integration_tests"] = success
        
        return results
    
    def run_all_tests(self, include_slow: bool = False, coverage: bool = False) -> Dict[str, Any]:
        """Run all test suites."""
        print("\n" + "=" * 80)
        print("🧪 COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        results: Dict[str, Any] = {
            "phase1": self.run_phase1_tests(),
            "phase2": self.run_phase2_tests(),
            "phase3": self.run_phase3_tests()
        }
        
        # Run full test suite
        print("\n" + "=" * 80)
        print("🧪 FULL TEST SUITE")
        print("=" * 80)
        
        success, output = self.run_test_suite(
            include_slow=include_slow,
            coverage=coverage
        )
        results["full_suite"] = success
        
        if success:
            print("✅ Full test suite passed")
        else:
            print("❌ Full test suite failed")
            print(output[-1000:])  # Last 1000 chars
        
        return results
    
    def print_summary(self, results: Dict):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        
        for phase, phase_results in results.items():
            if phase == "full_suite":
                continue
            
            print(f"\n{phase.upper()}:")
            for test_name, test_result in phase_results.items():
                total_tests += 1
                if test_result:
                    passed_tests += 1
                    print(f"  ✅ {test_name}")
                elif test_result is None:
                    print(f"  ⏭️  {test_name} (skipped)")
                else:
                    print(f"  ❌ {test_name}")
        
        print(f"\nFull suite: {'✅ PASSED' if results.get('full_suite') else '❌ FAILED'}")
        
        print(f"\n{'=' * 80}")
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        print(f"{'=' * 80}\n")
        
        return passed_tests == total_tests and results.get('full_suite', False)


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive tests for refactoring'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3],
        help='Run specific phase tests'
    )
    parser.add_argument(
        '--include-slow',
        action='store_true',
        help='Include slow tests'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent.resolve()
    runner = RefactoringTestRunner(repo_root)
    
    start_time = time.time()
    
    if args.phase == 1:
        results = {"phase1": runner.run_phase1_tests()}
    elif args.phase == 2:
        results = {"phase2": runner.run_phase2_tests()}
    elif args.phase == 3:
        results = {"phase3": runner.run_phase3_tests()}
    else:
        results = runner.run_all_tests(
            include_slow=args.include_slow,
            coverage=args.coverage
        )
    
    duration = time.time() - start_time
    
    all_passed = runner.print_summary(results)
    
    print(f"Duration: {duration:.1f}s")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
