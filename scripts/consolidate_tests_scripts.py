#!/usr/bin/env python3
"""
Consolidate and Clean Scripts & Tests Directories

This script:
1. Organizes scripts by function (utils, validation, maintenance, etc.)
2. Moves non-test files from tests/ to appropriate locations
3. Consolidates duplicate functionality
4. Creates proper test structure
5. Updates documentation
"""

import shutil
from pathlib import Path
import sys

# Color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_action(action, path, dest=None):
    if dest:
        print(f"{Colors.BLUE}  [{action}]{Colors.ENDC} {path} → {dest}")
    else:
        print(f"{Colors.BLUE}  [{action}]{Colors.ENDC} {path}")


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.ENDC}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.ENDC}")


def consolidate_scripts_and_tests(dry_run=False):
    """Main consolidation function."""
    
    root = Path(__file__).parent.parent
    scripts_dir = root / "scripts"
    tests_dir = root / "tests"
    
    print_header("🧹 SCRIPTS & TESTS CONSOLIDATION")
    
    actions = []
    
    # ========================================================================
    # PHASE 1: Organize Scripts by Function
    # ========================================================================
    print_header("📦 Phase 1: Organize Scripts Directory")
    
    # Create subdirectories
    subdirs = {
        "validation": scripts_dir / "validation",
        "maintenance": scripts_dir / "maintenance",
        "analysis": scripts_dir / "analysis",
    }
    
    # Validation scripts
    validation_scripts = [
        "validate_improved_locations.py",
        "check_gpu_config.py",
    ]
    
    for script in validation_scripts:
        src = scripts_dir / script
        if src.exists():
            dest = subdirs["validation"] / script
            actions.append(("move", src, dest))
            print_action("MOVE", script, f"scripts/validation/{script}")
    
    # Maintenance scripts
    maintenance_scripts = [
        "consolidate_repo.py",
        "consolidate_root.py",
        "organize_repo.py",
        "clean_repo_interactive.py",
        "regenerate_metadata.py",
    ]
    
    for script in maintenance_scripts:
        src = scripts_dir / script
        if src.exists():
            dest = subdirs["maintenance"] / script
            actions.append(("move", src, dest))
            print_action("MOVE", script, f"scripts/maintenance/{script}")
    
    # Analysis scripts
    analysis_scripts = [
        "show_strategic_locations.py",
        "optimize_workers.py",
        "SYNTHESE.py",
    ]
    
    for script in analysis_scripts:
        src = scripts_dir / script
        if src.exists():
            dest = subdirs["analysis"] / script
            actions.append(("move", src, dest))
            print_action("MOVE", script, f"scripts/analysis/{script}")
    
    # replace_empty_locations.py is a one-time migration tool -> legacy
    src = scripts_dir / "replace_empty_locations.py"
    if src.exists():
        dest = scripts_dir / "legacy" / "replace_empty_locations.py"
        actions.append(("move", src, dest))
        print_action("ARCHIVE", "replace_empty_locations.py",
                     "scripts/legacy/replace_empty_locations.py")
    
    # ========================================================================
    # PHASE 2: Clean Tests Directory
    # ========================================================================
    print_header("🧪 Phase 2: Clean Tests Directory")
    
    # Move non-test files from tests/ to scripts/
    non_test_files = [
        ("validate_features.py", subdirs["validation"]),
        ("benchmark_optimization.py", scripts_dir / "benchmarks"),
    ]
    
    for filename, dest_dir in non_test_files:
        src = tests_dir / filename
        if src.exists():
            dest = dest_dir / filename
            actions.append(("move", src, dest))
            print_action("MOVE", f"tests/{filename}",
                        f"{dest_dir.relative_to(root)}/{filename}")
    
    # test_consolidation.py is a one-time test -> archive or remove
    src = tests_dir / "test_consolidation.py"
    if src.exists():
        dest = scripts_dir / "legacy" / "test_consolidation.py"
        actions.append(("move", src, dest))
        print_action("ARCHIVE", "tests/test_consolidation.py",
                     "scripts/legacy/test_consolidation.py")
    
    # ========================================================================
    # PHASE 3: Consolidate Duplicate Scripts
    # ========================================================================
    print_header("🔄 Phase 3: Check for Duplicates")
    
    # Check for similar scripts
    similar_groups = [
        ["consolidate_repo.py", "consolidate_root.py", "organize_repo.py"],
    ]
    
    print_info("  Found maintenance scripts:")
    print_info("  - consolidate_repo.py: General repo consolidation")
    print_info("  - consolidate_root.py: Root directory specific")
    print_info("  - organize_repo.py: File organization")
    print_info("  → Keeping all (different purposes)")
    
    # ========================================================================
    # PHASE 4: Create README files
    # ========================================================================
    print_header("📝 Phase 4: Create Documentation")
    
    readme_files = [
        (subdirs["validation"], "validation_readme"),
        (subdirs["maintenance"], "maintenance_readme"),
        (subdirs["analysis"], "analysis_readme"),
        (scripts_dir / "benchmarks", "benchmarks_readme"),
    ]
    
    for directory, readme_type in readme_files:
        readme_path = directory / "README.md"
        actions.append(("create_readme", readme_path, readme_type))
        print_action("CREATE", f"{directory.relative_to(root)}/README.md")
    
    # Update main tests README
    tests_readme = tests_dir / "README.md"
    actions.append(("create_readme", tests_readme, "tests_readme"))
    print_action("UPDATE", "tests/README.md")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("📊 SUMMARY")
    moves = sum(1 for a in actions if a[0] == "move")
    creates = sum(1 for a in actions if a[0] == "create_readme")
    
    print(f"Total actions: {len(actions)}")
    print(f"  - Moves: {moves}")
    print(f"  - README creations: {creates}")
    print(f"\nNew subdirectories:")
    for name, path in subdirs.items():
        print(f"  - scripts/{name}/")
    print(f"  - scripts/benchmarks/")
    
    if dry_run:
        print_warning("\n🔍 DRY RUN MODE - No changes made")
        return
    
    # ========================================================================
    # EXECUTE
    # ========================================================================
    print_header("⚡ EXECUTING ACTIONS")
    
    # Create directories
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
    
    errors = []
    
    for action_type, src, dest in actions:
        try:
            if action_type == "move":
                if not src.exists():
                    print_warning(f"Source not found: {src.name}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                print_success(f"Moved {src.name}")
            
            elif action_type == "create_readme":
                create_readme(src, dest)
                print_success(f"Created {src.relative_to(root)}")
        
        except Exception as e:
            errors.append((src, str(e)))
            print_error(f"Failed: {src} - {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_header("✅ CONSOLIDATION COMPLETE")
    
    if errors:
        print_error(f"{len(errors)} errors occurred:")
        for path, error in errors:
            print(f"  - {path}: {error}")
    else:
        print_success("All actions completed successfully!")
    
    print(f"\n{Colors.CYAN}📋 Next Steps:{Colors.ENDC}")
    print(f"  1. Review new structure: {Colors.CYAN}ls -la scripts/*/{Colors.ENDC}")
    print(f"  2. Update any imports in code")
    print(f"  3. Test scripts in new locations")
    print(f"  4. Commit changes")


def print_info(msg):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ {msg}{Colors.ENDC}")


def create_readme(path, readme_type):
    """Create README files for subdirectories."""
    
    readmes = {
        "validation_readme": """# Validation Scripts

Scripts for validating data, configurations, and features.

## Scripts

### validate_improved_locations.py
Validates the strategic locations data and checks for improvements.

**Usage:**
```bash
python scripts/validation/validate_improved_locations.py
```

### check_gpu_config.py
Checks GPU configuration and CUDA availability.

**Usage:**
```bash
python scripts/validation/check_gpu_config.py
```

### validate_features.py
Validates geometric feature computations (moved from tests/).

**Usage:**
```bash
python scripts/validation/validate_features.py
```

## Purpose

These scripts ensure data quality, configuration correctness, and
feature computation accuracy before processing or training.
""",
        
        "maintenance_readme": """# Maintenance Scripts

Scripts for repository maintenance, organization, and cleanup.

## Scripts

### consolidate_repo.py
General repository consolidation and organization.

**Usage:**
```bash
python scripts/maintenance/consolidate_repo.py --dry-run
python scripts/maintenance/consolidate_repo.py
```

### consolidate_root.py
Specifically cleans and organizes the root directory.

**Usage:**
```bash
python scripts/maintenance/consolidate_root.py --dry-run
python scripts/maintenance/consolidate_root.py
```

### organize_repo.py
Organizes files and directories according to best practices.

**Usage:**
```bash
python scripts/maintenance/organize_repo.py
```

### clean_repo_interactive.py
Interactive repository cleanup with user prompts.

**Usage:**
```bash
python scripts/maintenance/clean_repo_interactive.py
```

### regenerate_metadata.py
Regenerates metadata files for existing tiles.

**Usage:**
```bash
python scripts/maintenance/regenerate_metadata.py
```

## Purpose

These scripts help maintain a clean, organized repository structure.
""",
        
        "analysis_readme": """# Analysis Scripts

Scripts for analyzing data, performance, and generating reports.

## Scripts

### show_strategic_locations.py
Displays information about strategic tile locations.

**Usage:**
```bash
python scripts/analysis/show_strategic_locations.py
```

### optimize_workers.py
Analyzes and optimizes worker configurations.

**Usage:**
```bash
python scripts/analysis/optimize_workers.py
```

### SYNTHESE.py
Generates comprehensive synthesis reports.

**Usage:**
```bash
python scripts/analysis/SYNTHESE.py
```

## Purpose

These scripts provide insights into data, performance, and system
configuration to guide optimization decisions.
""",
        
        "benchmarks_readme": """# Benchmark Scripts

Performance benchmarking and optimization analysis.

## Scripts

### benchmark_optimization.py
Benchmarks feature computation performance (moved from tests/).

**Usage:**
```bash
python scripts/benchmarks/benchmark_optimization.py <laz_file>
```

## Purpose

These scripts measure performance of various components to identify
bottlenecks and validate optimizations.
""",
        
        "tests_readme": """# Tests Directory

Unit tests and integration tests for the IGN LiDAR HD library.

## Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_core.py             # Core functionality tests
├── test_features.py         # Feature extraction tests
├── test_building_features.py # Building-specific feature tests
├── test_cli.py              # CLI tests
├── test_config_gpu.py       # GPU configuration tests
├── test_configuration.py    # General configuration tests
└── test_new_features.py     # New feature validation tests
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_core.py
```

### Run with coverage
```bash
pytest --cov=ign_lidar tests/
```

### Run with verbose output
```bash
pytest -v tests/
```

## Test Categories

### Unit Tests
- `test_core.py`: Core classes and functions
- `test_features.py`: Feature extraction algorithms
- `test_building_features.py`: Building-specific features

### Integration Tests
- `test_cli.py`: Command-line interface
- `test_configuration.py`: Configuration management

### Validation Tests
- `test_config_gpu.py`: GPU setup validation
- `test_new_features.py`: New feature validation

## Guidelines

- Tests should be fast and independent
- Use fixtures in `conftest.py` for shared setup
- Mock external dependencies (files, network, etc.)
- Aim for high code coverage
- Follow naming convention: `test_<functionality>.py`

## Moved Scripts

The following non-test scripts have been moved:
- `validate_features.py` → `scripts/validation/`
- `benchmark_optimization.py` → `scripts/benchmarks/`
- `test_consolidation.py` → `scripts/legacy/` (archived)

These were validation/benchmark scripts, not unit tests.
"""
    }
    
    if readme_type in readmes:
        path.write_text(readmes[readme_type])


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Consolidate and clean scripts & tests directories",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them"
    )
    
    args = parser.parse_args()
    
    try:
        consolidate_scripts_and_tests(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print_warning("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
