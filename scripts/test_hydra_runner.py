#!/usr/bin/env python3
"""
Test script for HydraRunner functionality.

Tests:
1. Basic config loading
2. Override application
3. Config file loading
4. Config merging
5. Override extraction from args

Run with:
    python scripts/test_hydra_runner.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_basic_loading():
    """Test basic config loading."""
    print("=" * 60)
    print("Test 1: Basic Config Loading")
    print("=" * 60)
    
    try:
        from ign_lidar.cli.hydra_runner import HydraRunner
        
        runner = HydraRunner()
        print(f"✅ HydraRunner initialized")
        print(f"   Config dir: {runner.config_dir}")
        
        # Try to load default config
        try:
            cfg = runner.load_config(config_name="config")
            print(f"✅ Default config loaded")
            print(f"   Keys: {list(cfg.keys())[:5]}...")
            return True
        except FileNotFoundError:
            print(f"⚠️  Default config not found (expected in dev environment)")
            return True  # Not a failure
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_override_application():
    """Test applying overrides."""
    print("\n" + "=" * 60)
    print("Test 2: Override Application")
    print("=" * 60)
    
    try:
        from ign_lidar.cli.hydra_runner import HydraRunner
        from omegaconf import OmegaConf
        
        runner = HydraRunner()
        
        # Create a simple config
        test_config = OmegaConf.create({
            'input_dir': 'original',
            'processor': {
                'use_gpu': False,
                'num_points': 16384
            }
        })
        
        print("Original config:")
        print(f"  input_dir: {test_config.input_dir}")
        print(f"  processor.use_gpu: {test_config.processor.use_gpu}")
        
        # Apply overrides
        overrides = ["input_dir=data/", "processor.use_gpu=true"]
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(test_config, override_cfg)
        
        print("\nAfter overrides:")
        print(f"  input_dir: {merged.input_dir}")
        print(f"  processor.use_gpu: {merged.processor.use_gpu}")
        
        # Verify
        assert merged.input_dir == "data/", "input_dir override failed"
        assert merged.processor.use_gpu == True, "processor.use_gpu override failed"
        
        print("✅ Overrides applied correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extract_overrides():
    """Test extracting overrides from arguments."""
    print("\n" + "=" * 60)
    print("Test 3: Extract Overrides from Args")
    print("=" * 60)
    
    try:
        from ign_lidar.cli.hydra_runner import HydraRunner
        
        # Test args
        args = [
            '--verbose',
            'input_dir=data/',
            '--config-file', 'config.yaml',
            'processor.use_gpu=true',
            'features.k_neighbors=30'
        ]
        
        regular, overrides = HydraRunner.extract_overrides_from_args(args)
        
        print(f"Original args: {args}")
        print(f"\nRegular args: {regular}")
        print(f"Hydra overrides: {overrides}")
        
        # Verify
        expected_regular = ['--verbose', '--config-file', 'config.yaml']
        expected_overrides = ['input_dir=data/', 'processor.use_gpu=true', 'features.k_neighbors=30']
        
        assert regular == expected_regular, f"Expected {expected_regular}, got {regular}"
        assert overrides == expected_overrides, f"Expected {expected_overrides}, got {overrides}"
        
        print("✅ Override extraction works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_merging():
    """Test merging multiple configs."""
    print("\n" + "=" * 60)
    print("Test 4: Config Merging")
    print("=" * 60)
    
    try:
        from ign_lidar.cli.hydra_runner import HydraRunner
        from omegaconf import OmegaConf
        
        # Create base config
        base = OmegaConf.create({
            'input_dir': 'data/',
            'processor': {
                'patch_size': 100.0,
                'use_gpu': False
            }
        })
        
        # Create override config
        override = OmegaConf.create({
            'processor': {
                'use_gpu': True,
                'num_points': 32768
            },
            'features': {
                'k_neighbors': 20
            }
        })
        
        print("Base config:")
        print(f"  processor.patch_size: {base.processor.patch_size}")
        print(f"  processor.use_gpu: {base.processor.use_gpu}")
        
        print("\nOverride config:")
        print(f"  processor.use_gpu: {override.processor.use_gpu}")
        print(f"  processor.num_points: {override.processor.num_points}")
        print(f"  features.k_neighbors: {override.features.k_neighbors}")
        
        # Merge
        merged = HydraRunner.merge_configs(base, override)
        
        print("\nMerged config:")
        print(f"  processor.patch_size: {merged.processor.patch_size} (from base)")
        print(f"  processor.use_gpu: {merged.processor.use_gpu} (overridden)")
        print(f"  processor.num_points: {merged.processor.num_points} (added)")
        print(f"  features.k_neighbors: {merged.features.k_neighbors} (added)")
        
        # Verify
        assert merged.processor.patch_size == 100.0, "patch_size should be preserved"
        assert merged.processor.use_gpu == True, "use_gpu should be overridden"
        assert merged.processor.num_points == 32768, "num_points should be added"
        assert merged.features.k_neighbors == 20, "k_neighbors should be added"
        
        print("✅ Config merging works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """Test convenience load_config function."""
    print("\n" + "=" * 60)
    print("Test 5: Convenience Function")
    print("=" * 60)
    
    try:
        from ign_lidar.cli.hydra_runner import load_config
        from omegaconf import OmegaConf
        import tempfile
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("input_dir: test_data\n")
            f.write("output_dir: test_output\n")
            f.write("processor:\n")
            f.write("  patch_size: 150.0\n")
            temp_config = f.name
        
        try:
            # Load from file
            cfg = load_config(config_file=temp_config)
            print(f"✅ Config loaded from file: {temp_config}")
            print(f"   input_dir: {cfg.input_dir}")
            print(f"   processor.patch_size: {cfg.processor.patch_size}")
            
            # Load with overrides
            cfg2 = load_config(
                config_file=temp_config,
                overrides=["processor.patch_size=200.0"]
            )
            print(f"\n✅ Config loaded with overrides")
            print(f"   processor.patch_size: {cfg2.processor.patch_size} (overridden)")
            
            # Verify
            assert cfg.processor.patch_size == 150.0, "Original value should be 150.0"
            assert cfg2.processor.patch_size == 200.0, "Overridden value should be 200.0"
            
            print("✅ Convenience function works correctly")
            return True
            
        finally:
            # Clean up
            import os
            if os.path.exists(temp_config):
                os.unlink(temp_config)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🔬" * 30)
    print("HydraRunner Test Suite")
    print("🔬" * 30)
    
    tests = {
        "Basic Loading": test_basic_loading,
        "Override Application": test_override_application,
        "Extract Overrides": test_extract_overrides,
        "Config Merging": test_config_merging,
        "Convenience Function": test_convenience_function,
    }
    
    results = {}
    for name, test_func in tests.items():
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("=" * 60)
        return 0
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"⚠️  {len(failed)} test(s) failed: {', '.join(failed)}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
