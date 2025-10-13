#!/usr/bin/env python3
"""Test the new feature_manager and config_validator modules."""

import sys
import logging
from pathlib import Path

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from omegaconf import OmegaConf
from ign_lidar.core.modules.feature_manager import FeatureManager
from ign_lidar.core.modules.config_validator import ConfigValidator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_config_validator():
    """Test ConfigValidator functionality."""
    print("\n" + "="*70)
    print("Test 1: Config Validator")
    print("="*70)
    
    try:
        # Test output format validation
        formats = ConfigValidator.validate_output_format("npz")
        assert formats == ['npz'], "Single format validation failed"
        print("✅ Single format validation: npz")
        
        formats = ConfigValidator.validate_output_format("hdf5,laz")
        assert formats == ['hdf5', 'laz'], "Multi-format validation failed"
        print("✅ Multi-format validation: hdf5,laz")
        
        # Test invalid format
        try:
            ConfigValidator.validate_output_format("invalid")
            print("❌ Should have raised ValueError for invalid format")
            return False
        except ValueError as e:
            print(f"✅ Invalid format correctly rejected: {str(e)[:50]}...")
        
        # Test processing mode validation
        mode = ConfigValidator.validate_processing_mode("patches_only")
        assert mode == "patches_only", "Mode validation failed"
        print("✅ Processing mode validation: patches_only")
        
        # Test invalid mode
        try:
            ConfigValidator.validate_processing_mode("invalid_mode")
            print("❌ Should have raised ValueError for invalid mode")
            return False
        except ValueError as e:
            print(f"✅ Invalid mode correctly rejected: {str(e)[:50]}...")
        
        # Test preprocessing config setup
        preprocess_cfg = ConfigValidator.setup_preprocessing_config(
            preprocess=True,
            preprocess_config=None
        )
        assert preprocess_cfg is not None, "Preprocessing config should not be None"
        assert 'sor_k' in preprocess_cfg, "Missing default sor_k"
        print(f"✅ Preprocessing config with defaults: {len(preprocess_cfg)} keys")
        
        # Test stitching config setup
        stitch_cfg = ConfigValidator.setup_stitching_config(
            use_stitching=True,
            buffer_size=15.0
        )
        assert stitch_cfg['enabled'] == True, "Stitching should be enabled"
        assert stitch_cfg['buffer_size'] == 15.0, "Buffer size not set"
        print(f"✅ Stitching config: buffer_size={stitch_cfg['buffer_size']}m")
        
        print("\n✅ PASS: All ConfigValidator tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_manager():
    """Test FeatureManager functionality."""
    print("\n" + "="*70)
    print("Test 2: Feature Manager")
    print("="*70)
    
    try:
        # Create minimal config
        config = OmegaConf.create({
            'features': {
                'use_rgb': False,
                'use_infrared': False
            },
            'processor': {
                'use_gpu': False
            }
        })
        
        # Initialize manager
        manager = FeatureManager(config)
        
        # Test properties
        assert not manager.has_rgb, "RGB should not be available"
        assert not manager.has_infrared, "Infrared should not be available"
        assert not manager.has_gpu, "GPU should not be available"
        print("✅ Feature manager initialized with no features")
        
        # Test with RGB enabled (will fail gracefully if deps missing)
        config_rgb = OmegaConf.create({
            'features': {
                'use_rgb': True,
                'use_infrared': False,
                'rgb_cache_dir': None
            },
            'processor': {
                'use_gpu': False
            }
        })
        
        manager_rgb = FeatureManager(config_rgb)
        print(f"✅ RGB fetcher: {'available' if manager_rgb.has_rgb else 'not available (deps missing)'}")
        
        # Test with GPU enabled (will fail gracefully if CuPy missing)
        config_gpu = OmegaConf.create({
            'features': {
                'use_rgb': False,
                'use_infrared': False
            },
            'processor': {
                'use_gpu': True
            }
        })
        
        manager_gpu = FeatureManager(config_gpu)
        print(f"✅ GPU acceleration: {'available' if manager_gpu.has_gpu else 'not available (CuPy missing)'}")
        
        print("\n✅ PASS: All FeatureManager tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Testing New Processor Modules")
    print("="*70)
    
    tests = [
        test_config_validator,
        test_feature_manager,
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
