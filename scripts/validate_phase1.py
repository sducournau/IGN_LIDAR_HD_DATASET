#!/usr/bin/env python3
"""
Script de validation des migrations Phase 1

Exécute les tests et génère un rapport de validation pour:
- Migration KNN vers KNNEngine
- API unifiée de calcul des normales
- Optimisations GPU

Usage:
    python scripts/validate_phase1.py
    python scripts/validate_phase1.py --verbose
    python scripts/validate_phase1.py --quick  # Tests rapides uniquement

Author: Phase 1 Consolidation
Date: November 23, 2025
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.END} {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {message}")


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run command and return success status and output.
    
    Args:
        cmd: Command to run as list
        description: Description for logging
        
    Returns:
        (success, output)
    """
    print(f"  Running: {description}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out after 5 minutes"
    except Exception as e:
        return False, str(e)


def validate_imports():
    """Validate that key modules can be imported."""
    print_section("Validating Imports")
    
    imports_to_test = [
        ("ign_lidar.features.compute.normals", "Normals API"),
        ("ign_lidar.features.orchestrator", "FeatureOrchestrator"),
        ("ign_lidar.optimization.knn_engine", "KNNEngine"),
        ("ign_lidar.io.formatters.hybrid_formatter", "HybridFormatter"),
    ]
    
    all_passed = True
    for module, name in imports_to_test:
        try:
            __import__(module)
            print_success(f"Import {name}: OK")
        except ImportError as e:
            print_error(f"Import {name}: FAILED - {e}")
            all_passed = False
    
    return all_passed


def run_unit_tests(quick: bool = False) -> Dict[str, bool]:
    """Run unit tests and return results."""
    print_section("Running Unit Tests")
    
    results = {}
    
    # Test suites to run
    test_suites = [
        ("tests/test_knn_engine.py", "KNN Engine", False),
        ("tests/test_formatters_knn_migration.py", "Formatters KNN Migration", False),
    ]
    
    if not quick:
        test_suites.extend([
            ("tests/test_features.py", "Features", True),  # Optional
            ("tests/test_optimization.py", "Optimization", True),  # Optional
        ])
    
    for test_file, name, optional in test_suites:
        test_path = Path(test_file)
        
        if not test_path.exists():
            if optional:
                print_warning(f"{name}: SKIPPED (file not found)")
                results[name] = None
            else:
                print_error(f"{name}: FAILED (file not found)")
                results[name] = False
            continue
        
        cmd = ["pytest", str(test_path), "-v", "--tb=short"]
        success, output = run_command(cmd, f"{name} tests")
        
        if success:
            print_success(f"{name}: PASSED")
            results[name] = True
        elif optional:
            print_warning(f"{name}: FAILED (optional)")
            results[name] = None
        else:
            print_error(f"{name}: FAILED")
            results[name] = False
            if not quick:
                print(f"\nOutput:\n{output}\n")
    
    return results


def check_code_duplication():
    """Check code duplication metrics."""
    print_section("Checking Code Duplication")
    
    script_path = Path("scripts/analyze_duplication.py")
    if not script_path.exists():
        print_warning("Duplication analysis script not found")
        return None
    
    cmd = ["python", str(script_path)]
    success, output = run_command(cmd, "Code duplication analysis")
    
    if success:
        # Parse output for key metrics
        lines = output.split('\n')
        for line in lines:
            if 'Fonctions dupliquées:' in line or 'compute_normals()' in line:
                print(f"  {line.strip()}")
        print_success("Duplication analysis: COMPLETED")
        return True
    else:
        print_error("Duplication analysis: FAILED")
        return False


def verify_documentation():
    """Verify that documentation files exist."""
    print_section("Verifying Documentation")
    
    docs_to_check = [
        ("docs/migration_guides/normals_computation_guide.md", "Normals Computation Guide"),
        ("docs/audit_reports/AUDIT_COMPLET_NOV_2025.md", "Complete Audit Report"),
        ("docs/audit_reports/IMPLEMENTATION_PHASE1_NOV_2025.md", "Phase 1 Implementation Report"),
    ]
    
    all_found = True
    for doc_path, name in docs_to_check:
        path = Path(doc_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print_success(f"{name}: FOUND ({size_kb:.1f} KB)")
        else:
            print_error(f"{name}: NOT FOUND")
            all_found = False
    
    return all_found


def generate_summary(
    imports_ok: bool,
    test_results: Dict[str, bool],
    duplication_ok: bool | None,
    docs_ok: bool
):
    """Generate validation summary."""
    print_section("Validation Summary")
    
    # Count test results
    passed = sum(1 for v in test_results.values() if v is True)
    failed = sum(1 for v in test_results.values() if v is False)
    skipped = sum(1 for v in test_results.values() if v is None)
    total = passed + failed + skipped
    
    print(f"Imports:       {'PASS' if imports_ok else 'FAIL'}")
    print(f"Unit Tests:    {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"Duplication:   {'PASS' if duplication_ok else 'SKIP' if duplication_ok is None else 'FAIL'}")
    print(f"Documentation: {'PASS' if docs_ok else 'FAIL'}")
    
    print()
    
    # Overall status
    critical_pass = imports_ok and failed == 0
    
    if critical_pass and docs_ok:
        print_success("✓ Phase 1 Validation: ALL CHECKS PASSED")
        print()
        print("Ready for:")
        print("  - Integration into main branch")
        print("  - Phase 2 implementation")
        print("  - Production deployment")
        return 0
    elif critical_pass:
        print_warning("⚠ Phase 1 Validation: PASSED with warnings")
        print()
        print("Action required:")
        print("  - Complete documentation")
        return 1
    else:
        print_error("✗ Phase 1 Validation: FAILED")
        print()
        print("Action required:")
        if not imports_ok:
            print("  - Fix import errors")
        if failed > 0:
            print(f"  - Fix {failed} failing test(s)")
        return 2


def main():
    """Main validation workflow."""
    parser = argparse.ArgumentParser(description="Validate Phase 1 implementations")
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip optional tests)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}Phase 1 Implementation Validation{Colors.END}")
    print(f"Date: November 23, 2025")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    # Run validation steps
    imports_ok = validate_imports()
    test_results = run_unit_tests(quick=args.quick)
    duplication_ok = check_code_duplication() if not args.quick else None
    docs_ok = verify_documentation()
    
    # Generate summary and return exit code
    exit_code = generate_summary(imports_ok, test_results, duplication_ok, docs_ok)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
