#!/usr/bin/env python3
"""
Test script to verify PipelineConfig deprecation warning.

This script tests that:
1. PipelineConfig still works (backward compatibility)
2. DeprecationWarning is raised
3. Warning message contains migration information

Run with:
    python scripts/test_deprecation.py
"""

import sys
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_pipeline_config_deprecation():
    """Test that PipelineConfig raises deprecation warning."""
    
    print("=" * 60)
    print("Testing PipelineConfig Deprecation Warning")
    print("=" * 60)
    
    # Enable all warnings
    warnings.simplefilter('always', DeprecationWarning)
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        
        try:
            from ign_lidar.core.pipeline_config import PipelineConfig
            print("\n✅ PipelineConfig imported successfully")
            
            # Create a dummy config file to test instantiation
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("download:\n  enabled: true\n")
                temp_config = f.name
            
            try:
                # Try to instantiate - this should trigger the warning
                config = PipelineConfig(temp_config)
                print("✅ PipelineConfig instantiated (backward compatibility maintained)")
            except Exception as e:
                print(f"⚠️  Instantiation failed (expected if config invalid): {e}")
            finally:
                # Clean up
                import os
                if os.path.exists(temp_config):
                    os.unlink(temp_config)
            
        except ImportError as e:
            print(f"\n❌ Failed to import PipelineConfig: {e}")
            return False
    
    # Check if deprecation warning was raised
    deprecation_warnings = [
        w for w in caught_warnings 
        if issubclass(w.category, DeprecationWarning)
    ]
    
    if deprecation_warnings:
        print("\n✅ DeprecationWarning raised correctly")
        for w in deprecation_warnings:
            print(f"\n📋 Warning Message:")
            print(f"   {w.message}")
            
            # Check that warning contains migration info
            msg = str(w.message)
            if "v2.5.0" in msg and "Hydra" in msg and "MIGRATION_GUIDE" in msg:
                print("\n✅ Warning contains migration information:")
                print(f"   - Mentions v2.5.0 removal: {'v2.5.0' in msg}")
                print(f"   - Mentions Hydra: {'Hydra' in msg}")
                print(f"   - Mentions migration guide: {'MIGRATION_GUIDE' in msg}")
                return True
            else:
                print("\n⚠️  Warning missing some migration information")
                return False
    else:
        print("\n❌ No DeprecationWarning raised!")
        print("   PipelineConfig should issue a deprecation warning")
        return False


def test_migration_guide_exists():
    """Test that MIGRATION_GUIDE.md exists."""
    
    print("\n" + "=" * 60)
    print("Checking Migration Guide")
    print("=" * 60)
    
    guide_path = Path(__file__).parent.parent / "MIGRATION_GUIDE.md"
    
    if guide_path.exists():
        print(f"\n✅ Migration guide exists: {guide_path}")
        
        # Check file size
        size = guide_path.stat().st_size
        print(f"   File size: {size:,} bytes")
        
        # Check for key sections
        content = guide_path.read_text(encoding='utf-8')
        key_sections = [
            "Breaking Changes",
            "PipelineConfig",
            "Old Way",
            "New Way",
            "Migration Checklist"
        ]
        
        missing = [s for s in key_sections if s not in content]
        
        if not missing:
            print(f"   ✅ All key sections present")
            return True
        else:
            print(f"   ⚠️  Missing sections: {', '.join(missing)}")
            return False
    else:
        print(f"\n❌ Migration guide not found: {guide_path}")
        print("   Users need migration documentation!")
        return False


def test_new_imports_work():
    """Test that new module imports work."""
    
    print("\n" + "=" * 60)
    print("Testing New Module Imports")
    print("=" * 60)
    
    tests = []
    
    # Test 1: Import from modules
    try:
        from ign_lidar.core.modules.patch_extractor import extract_patches
        print("\n✅ Can import extract_patches from modules.patch_extractor")
        tests.append(True)
    except ImportError as e:
        print(f"\n❌ Cannot import from modules: {e}")
        tests.append(False)
    
    # Test 2: Import from preprocessing.utils (backward compatibility)
    try:
        from ign_lidar.preprocessing.utils import extract_patches
        print("✅ Can still import from preprocessing.utils (backward compat)")
        tests.append(True)
    except ImportError as e:
        print(f"❌ Cannot import from preprocessing.utils: {e}")
        tests.append(False)
    
    # Test 3: Check they're the same function
    try:
        from ign_lidar.core.modules.patch_extractor import extract_patches as ep_new
        from ign_lidar.preprocessing.utils import extract_patches as ep_old
        
        if ep_new is ep_old:
            print("✅ Both imports reference the same function (good re-export)")
            tests.append(True)
        else:
            print("⚠️  Imports reference different functions")
            tests.append(False)
    except ImportError:
        tests.append(False)
    
    return all(tests)


def main():
    """Run all tests."""
    
    print("\n" + "🔬" * 30)
    print("Deprecation Warning Test Suite")
    print("🔬" * 30)
    
    results = {
        "PipelineConfig deprecation": test_pipeline_config_deprecation(),
        "Migration guide exists": test_migration_guide_exists(),
        "New imports work": test_new_imports_work(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("⚠️  Some tests failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
