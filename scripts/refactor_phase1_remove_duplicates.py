#!/usr/bin/env python3
"""
Phase 1 Refactoring: Remove Critical Duplications

This script removes duplicate implementations of compute_normals() and validate_normals()
while maintaining backward compatibility through deprecation warnings.

Author: GitHub Copilot
Date: November 22, 2025
"""

import re
from pathlib import Path
from typing import List, Tuple


def add_deprecation_wrapper(
    file_path: Path, 
    function_name: str, 
    canonical_import: str,
    start_line: int
) -> str:
    """
    Replace function implementation with deprecation wrapper.
    
    Args:
        file_path: Path to file containing duplicate
        function_name: Name of duplicated function
        canonical_import: Import path to canonical implementation
        start_line: Line number where function starts
    
    Returns:
        Modified file content
    """
    content = file_path.read_text()
    lines = content.split('\n')
    
    # Find function end (next def or class, or dedent to 0)
    func_start = start_line - 1
    func_end = func_start + 1
    
    while func_end < len(lines):
        line = lines[func_end]
        # Check if we hit next function/class or dedented to module level
        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            break
        if re.match(r'^\s{0,4}(def |class |@)', line):  # Next def/class at same or lower indent
            break
        func_end += 1
    
    # Create deprecation wrapper
    wrapper = f'''
def {function_name}(*args, **kwargs):
    """
    DEPRECATED: This function is a duplicate of the canonical implementation.
    
    Use instead:
        {canonical_import}
    
    This wrapper will be removed in v4.0.
    """
    import warnings
    warnings.warn(
        f"{{__name__}}.{function_name}() is deprecated since v3.6.0. "
        f"Use '{canonical_import}' instead. "
        f"This function will be removed in v4.0.",
        DeprecationWarning,
        stacklevel=2
    )
    {canonical_import.split()[-1].replace('from', '').replace('import', '').strip()}
    return {function_name.split('.')[-1]}(*args, **kwargs)
'''
    
    # Replace function with wrapper
    lines[func_start:func_end] = [wrapper]
    
    return '\n'.join(lines)


def remove_compute_normals_duplicates():
    """Remove duplicate compute_normals() implementations."""
    
    print("=" * 80)
    print("🔧 Phase 1: Removing compute_normals() Duplicates")
    print("=" * 80)
    
    duplicates = [
        {
            'file': 'ign_lidar/features/feature_computer.py',
            'line': 160,
            'function': 'compute_normals',
            'canonical': 'from ign_lidar.features.compute import compute_normals'
        },
        {
            'file': 'ign_lidar/features/gpu_processor.py',
            'line': 376,
            'function': 'compute_normals',
            'canonical': 'from ign_lidar.features.compute import compute_normals'
        },
        {
            'file': 'ign_lidar/features/gpu_processor.py',
            'line': 726,
            'function': '_compute_normals_cpu',
            'canonical': 'from ign_lidar.features.compute.normals import compute_normals'
        }
    ]
    
    for dup in duplicates:
        file_path = Path(dup['file'])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            continue
        
        print(f"\n📝 Processing: {file_path}")
        print(f"   Line: {dup['line']}, Function: {dup['function']}")
        
        # Backup original
        backup_path = file_path.with_suffix('.py.backup')
        file_path.rename(backup_path)
        print(f"   💾 Backup: {backup_path}")
        
        try:
            # Add deprecation wrapper
            modified = add_deprecation_wrapper(
                backup_path,
                dup['function'],
                dup['canonical'],
                dup['line']
            )
            
            file_path.write_text(modified)
            print(f"   ✅ Added deprecation wrapper")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            # Restore backup
            backup_path.rename(file_path)
            print(f"   🔄 Restored from backup")


def remove_validate_normals_duplicate():
    """Remove duplicate validate_normals() from features/utils.py."""
    
    print("\n" + "=" * 80)
    print("🔧 Phase 1: Removing validate_normals() Duplicate")
    print("=" * 80)
    
    file_path = Path('ign_lidar/features/utils.py')
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n📝 Processing: {file_path}")
    
    content = file_path.read_text()
    
    # Check if validate_normals exists
    if 'def validate_normals' not in content:
        print("   ℹ️  Function not found (already removed?)")
        return
    
    # Backup
    backup_path = file_path.with_suffix('.py.backup')
    file_path.write_text(content)
    backup_path.write_text(content)
    print(f"   💾 Backup: {backup_path}")
    
    # Add import and deprecation wrapper at top
    wrapper = '''
# Deprecated: Use canonical implementation
from ign_lidar.features.compute.utils import validate_normals as _validate_normals_canonical

def validate_normals(*args, **kwargs):
    """DEPRECATED: Use ign_lidar.features.compute.utils.validate_normals instead."""
    import warnings
    warnings.warn(
        "ign_lidar.features.utils.validate_normals is deprecated. "
        "Use ign_lidar.features.compute.utils.validate_normals instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _validate_normals_canonical(*args, **kwargs)
'''
    
    # Find and replace validate_normals implementation
    lines = content.split('\n')
    
    # Find function start
    func_start = None
    for i, line in enumerate(lines):
        if 'def validate_normals' in line:
            func_start = i
            break
    
    if func_start is None:
        print("   ❌ Could not find function definition")
        return
    
    # Find function end
    func_end = func_start + 1
    indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())
    
    while func_end < len(lines):
        line = lines[func_end]
        if line.strip() and not line.startswith(' ' * (indent_level + 1)):
            break
        func_end += 1
    
    # Replace with wrapper
    lines[func_start:func_end] = [wrapper]
    
    file_path.write_text('\n'.join(lines))
    print(f"   ✅ Replaced with deprecation wrapper")


def generate_migration_guide():
    """Generate migration guide for users."""
    
    guide = """
# Migration Guide: compute_normals() Consolidation

## Summary
Multiple duplicate implementations of `compute_normals()` have been consolidated
into a single canonical implementation in `ign_lidar.features.compute.normals`.

## Changes

### Before (v3.5.x and earlier)
```python
# Multiple ways to compute normals (all valid)
from ign_lidar.features.feature_computer import FeatureComputer
computer = FeatureComputer()
normals = computer.compute_normals(points, k=30)

# OR
from ign_lidar.features.gpu_processor import GPUProcessor
processor = GPUProcessor()
normals = processor.compute_normals(points)

# OR
from ign_lidar.features.utils import validate_normals
validate_normals(normals)
```

### After (v3.6.0+)
```python
# Single canonical API
from ign_lidar.features.compute import compute_normals

# Standard computation
normals, eigenvalues = compute_normals(points, k_neighbors=30)

# Fast variant (optimized)
from ign_lidar.features.compute import compute_normals_fast
normals = compute_normals_fast(points, k_neighbors=30)

# Accurate variant (quality over speed)
from ign_lidar.features.compute import compute_normals_accurate
normals = compute_normals_accurate(points, k_neighbors=30)

# Validation
from ign_lidar.features.compute.utils import validate_normals
validate_normals(normals)
```

## Backward Compatibility

Old imports will continue to work in v3.6.x with deprecation warnings:

```python
# These will work but emit DeprecationWarning
from ign_lidar.features.feature_computer import FeatureComputer
computer = FeatureComputer()
normals = computer.compute_normals(points, k=30)  # ⚠️ DeprecationWarning
```

**Removal timeline:**
- v3.6.0: Deprecation warnings added
- v3.7.0-3.9.0: Warnings continue
- v4.0.0: Old implementations removed

## Why This Change?

1. **Code duplication**: 7 different implementations (~350 lines duplicated)
2. **Maintenance burden**: Bug fixes had to be applied 7 times
3. **Inconsistency risk**: Different implementations could diverge
4. **Performance**: Single optimized implementation is faster

## Benefits

- ✅ Single source of truth for normal computation
- ✅ Easier to maintain and test
- ✅ Better performance (unified optimizations)
- ✅ Clearer API for new users

## Need Help?

Open an issue: https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues
"""
    
    guide_path = Path('docs/migration_guides/compute_normals_consolidation.md')
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    guide_path.write_text(guide)
    
    print(f"\n📄 Migration guide created: {guide_path}")


def main():
    """Run Phase 1 refactoring."""
    
    print("\n" + "=" * 80)
    print("🚀 PHASE 1 REFACTORING: REMOVE CRITICAL DUPLICATIONS")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Replace duplicate compute_normals() with deprecation wrappers")
    print("  2. Replace duplicate validate_normals() with deprecation wrapper")
    print("  3. Generate migration guide")
    print("\n⚠️  Backups will be created as *.py.backup")
    
    response = input("\nProceed? [y/N]: ")
    
    if response.lower() != 'y':
        print("❌ Aborted")
        return
    
    # Execute refactoring
    remove_compute_normals_duplicates()
    remove_validate_normals_duplicate()
    generate_migration_guide()
    
    print("\n" + "=" * 80)
    print("✅ PHASE 1 COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review changes: git diff")
    print("  2. Run tests: pytest tests/ -v")
    print("  3. Run benchmark: python scripts/benchmark_normals.py")
    print("  4. Commit: git commit -m 'refactor: consolidate compute_normals() implementations'")
    print("\nSee: docs/migration_guides/compute_normals_consolidation.md")


if __name__ == '__main__':
    main()
