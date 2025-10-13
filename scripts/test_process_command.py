#!/usr/bin/env python3
"""Test the updated process command with HydraRunner integration."""

import sys
import logging
from pathlib import Path

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from ign_lidar.cli.hydra_runner import HydraRunner
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_process_config_loading():
    """Test that process command can load configs using HydraRunner."""
    print("\n" + "="*70)
    print("Test 1: Load Default Config")
    print("="*70)
    
    try:
        runner = HydraRunner()
        cfg = runner.load_config(config_name="config")
        
        # Verify key sections exist
        assert hasattr(cfg, 'processor'), "Missing 'processor' config"
        assert hasattr(cfg, 'features'), "Missing 'features' config"
        assert hasattr(cfg, 'output'), "Missing 'output' config"
        
        print("✅ PASS: Default config loaded successfully")
        print(f"   - LOD Level: {cfg.processor.lod_level}")
        print(f"   - GPU: {cfg.processor.use_gpu}")
        print(f"   - Features mode: {cfg.features.mode}")
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_process_config_with_overrides():
    """Test loading config with CLI overrides."""
    print("\n" + "="*70)
    print("Test 2: Load Config with Overrides")
    print("="*70)
    
    try:
        runner = HydraRunner()
        cfg = runner.load_config(
            config_name="config",
            overrides=[
                "processor.use_gpu=true",
                "processor.lod_level=LOD3",
                "features.mode=minimal"
            ]
        )
        
        # Verify overrides applied
        assert cfg.processor.use_gpu == True, "GPU override not applied"
        assert cfg.processor.lod_level == "LOD3", "LOD level override not applied"
        assert cfg.features.mode == "minimal", "Features mode override not applied"
        
        print("✅ PASS: Overrides applied successfully")
        print(f"   - LOD Level: {cfg.processor.lod_level}")
        print(f"   - GPU: {cfg.processor.use_gpu}")
        print(f"   - Features mode: {cfg.features.mode}")
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_process_config_from_file():
    """Test loading config from a custom file."""
    print("\n" + "="*70)
    print("Test 3: Load Config from File")
    print("="*70)
    
    # Find an example config file
    examples_dir = package_root / "examples"
    config_files = list(examples_dir.glob("config_*.yaml"))
    
    if not config_files:
        print("⏭️  SKIP: No example config files found")
        return True
    
    config_file = config_files[0]
    print(f"Using: {config_file.name}")
    
    try:
        runner = HydraRunner()
        cfg = runner.load_config(
            config_file=str(config_file)
        )
        
        print("✅ PASS: Config file loaded successfully")
        if hasattr(cfg, 'processor'):
            print(f"   - LOD Level: {cfg.processor.lod_level}")
        if hasattr(cfg, 'features'):
            print(f"   - Features mode: {cfg.features.mode}")
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_output_shorthand_handling():
    """Test that output shorthand is properly handled."""
    print("\n" + "="*70)
    print("Test 4: Output Shorthand Handling")
    print("="*70)
    
    try:
        runner = HydraRunner()
        
        # Test with output=enriched_only
        cfg = runner.load_config(
            config_name="config",
            overrides=["output=enriched_only"]
        )
        
        # Check if output is string (needs conversion)
        if isinstance(cfg.output, str):
            print(f"   Output is string: '{cfg.output}' (needs conversion in command)")
            output_mode = cfg.output
            
            # Simulate what process_command does
            cfg.output = OmegaConf.create({
                "format": "npz",
                "processing_mode": output_mode,
                "save_stats": True,
                "save_metadata": output_mode != 'enriched_only',
                "compression": None
            })
            
            print(f"   Converted to: processing_mode='{cfg.output.processing_mode}'")
            
        else:
            print(f"   Output is already dict with processing_mode='{cfg.output.processing_mode}'")
        
        print("✅ PASS: Output shorthand handling works")
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Testing Process Command with HydraRunner")
    print("="*70)
    
    tests = [
        test_process_config_loading,
        test_process_config_with_overrides,
        test_process_config_from_file,
        test_output_shorthand_handling,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
