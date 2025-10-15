#!/usr/bin/env python3
"""
Verify pytest test configuration and environment setup.

Run this script to check if your testing environment is properly configured.
"""
import sys
from pathlib import Path


def check_pytest_installed():
    """Check if pytest is installed."""
    try:
        import pytest
        print(f"✓ pytest is installed (version {pytest.__version__})")
        return True
    except ImportError:
        print("✗ pytest is NOT installed")
        print("  Install with: pip install pytest")
        return False


def check_project_structure():
    """Check if project has expected structure."""
    project_root = Path(__file__).parent
    required_paths = {
        "tests": "Tests directory",
        "ign_lidar": "Package directory",
        "pyproject.toml": "Project configuration",
        "pytest.ini": "Pytest configuration",
        "tests/conftest.py": "Pytest fixtures",
    }
    
    all_exist = True
    for path, desc in required_paths.items():
        full_path = project_root / path
        if full_path.exists():
            print(f"✓ {desc} exists: {path}")
        else:
            print(f"✗ {desc} MISSING: {path}")
            all_exist = False
    
    return all_exist


def check_test_data():
    """Check if test integration data directory exists."""
    project_root = Path(__file__).parent
    test_data = project_root / "data" / "test_integration"
    
    if test_data.exists():
        laz_files = list(test_data.glob("*.laz"))
        print(f"✓ Test integration data directory exists: {test_data}")
        print(f"  Found {len(laz_files)} LAZ file(s)")
        return True
    else:
        print(f"⚠ Test integration data directory does NOT exist: {test_data}")
        print("  Integration tests will be skipped without test data")
        return False


def check_vscode_config():
    """Check VS Code configuration."""
    project_root = Path(__file__).parent
    vscode_files = {
        ".vscode/settings.json": "VS Code settings",
        ".vscode/tasks.json": "VS Code tasks",
        ".vscode/launch.json": "VS Code launch config",
    }
    
    all_exist = True
    for path, desc in vscode_files.items():
        full_path = project_root / path
        if full_path.exists():
            print(f"✓ {desc} exists")
        else:
            print(f"✗ {desc} MISSING")
            all_exist = False
    
    return all_exist


def run_sample_tests():
    """Try to discover and list tests."""
    try:
        import pytest
        from pathlib import Path
        
        project_root = Path(__file__).parent
        test_dir = project_root / "tests"
        
        print("\n" + "="*60)
        print("Discovering tests...")
        print("="*60)
        
        # Run pytest in collect-only mode
        result = pytest.main([
            str(test_dir),
            "--collect-only",
            "-q"
        ])
        
        return result == 0
    except Exception as e:
        print(f"✗ Error running pytest: {e}")
        return False


def main():
    """Run all verification checks."""
    print("="*60)
    print("IGN LiDAR HD - Test Configuration Verification")
    print("="*60)
    print()
    
    checks = [
        ("pytest Installation", check_pytest_installed()),
        ("Project Structure", check_project_structure()),
        ("Test Data", check_test_data()),
        ("VS Code Configuration", check_vscode_config()),
    ]
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = all(result for _, result in checks)
    
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n✓ All checks passed! Your test environment is ready.")
        print("\nNext steps:")
        print("  1. Open VS Code Testing panel (Ctrl+Shift+T)")
        print("  2. Tests should auto-discover from 'tests/' directory")
        print("  3. Run tests by clicking the play button")
        print("  4. See TESTING.md for more information")
        
        # Try to discover tests
        run_sample_tests()
        return 0
    else:
        print("\n✗ Some checks failed. Please review the issues above.")
        print("\nTo fix:")
        print("  - Install missing dependencies: pip install -e .[dev]")
        print("  - Review the configuration files")
        print("  - See TESTING.md for detailed setup instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
