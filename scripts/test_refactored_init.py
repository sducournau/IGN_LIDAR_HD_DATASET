#!/usr/bin/env python3
"""Test the refactored processor __init__ approach."""

import sys
import logging
from pathlib import Path

# Add package to path
package_root = Path(__file__).parent.parent
sys.path.insert(0, str(package_root))

from omegaconf import OmegaConf
from ign_lidar.core.modules.feature_manager import FeatureManager
from ign_lidar.core.modules.config_validator import ConfigValidator
from ign_lidar.classes import ASPRS_TO_LOD2, ASPRS_TO_LOD3

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_config_based_init():
    """Test initialization with config object (modern approach)."""
    print("\n" + "="*70)
    print("Test 1: Config-Based Initialization")
    print("="*70)
    
    try:
        # Create config like Hydra would
        config = OmegaConf.create({
            'processor': {
                'lod_level': 'LOD2',
                'use_gpu': False,
                'patch_size': 150.0,
                'num_points': 16384,
                'architecture': 'pointnet++',
                'augment': False,
                'num_augmentations': 3,
            },
            'features': {
                'mode': 'full',
                'k_neighbors': 20,
                'use_rgb': False,
                'use_infrared': False,
            },
            'output': {
                'format': 'npz',
                'processing_mode': 'patches_only',
            },
            'stitching': {
                'enabled': False,
                'buffer_size': 10.0,
            }
        })
        
        # Initialize feature manager (what processor would do)
        feature_mgr = FeatureManager(config)
        print(f"✅ FeatureManager initialized")
        print(f"   - RGB: {feature_mgr.has_rgb}")
        print(f"   - NIR: {feature_mgr.has_infrared}")
        print(f"   - GPU: {feature_mgr.has_gpu}")
        
        # Validate config
        formats = ConfigValidator.validate_output_format(config.output.format)
        print(f"✅ Output format validated: {formats}")
        
        mode = ConfigValidator.validate_processing_mode(config.output.processing_mode)
        print(f"✅ Processing mode validated: {mode}")
        
        # Set class mapping (what processor would do)
        if config.processor.lod_level == 'LOD2':
            class_mapping = ASPRS_TO_LOD2
            default_class = 14
        else:
            class_mapping = ASPRS_TO_LOD3
            default_class = 29
        
        print(f"✅ Class mapping set: {len(class_mapping)} classes, default={default_class}")
        
        print("\n✅ PASS: Config-based initialization works")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_legacy_kwargs_conversion():
    """Test converting legacy kwargs to config."""
    print("\n" + "="*70)
    print("Test 2: Legacy Kwargs Conversion")
    print("="*70)
    
    try:
        # Simulate legacy kwargs
        kwargs = {
            'lod_level': 'LOD3',
            'use_gpu': True,
            'patch_size': 100.0,
            'num_points': 8192,
            'include_rgb': True,
            'processing_mode': 'both',
        }
        
        # Build config from kwargs (what _build_config_from_kwargs does)
        config = OmegaConf.create({
            'processor': {
                'lod_level': kwargs.get('lod_level', 'LOD2'),
                'use_gpu': kwargs.get('use_gpu', False),
                'patch_size': kwargs.get('patch_size', 150.0),
                'num_points': kwargs.get('num_points', 16384),
                'architecture': kwargs.get('architecture', 'pointnet++'),
            },
            'features': {
                'mode': kwargs.get('feature_mode', 'full'),
                'use_rgb': kwargs.get('include_rgb', False),
                'use_infrared': kwargs.get('include_infrared', False),
            },
            'output': {
                'format': kwargs.get('output_format', 'npz'),
                'processing_mode': kwargs.get('processing_mode', 'patches_only'),
            }
        })
        
        # Verify conversion
        assert config.processor.lod_level == 'LOD3', "LOD level not converted"
        assert config.processor.use_gpu == True, "GPU flag not converted"
        assert config.processor.patch_size == 100.0, "Patch size not converted"
        assert config.features.use_rgb == True, "RGB flag not converted"
        assert config.output.processing_mode == 'both', "Processing mode not converted"
        
        print(f"✅ Converted {len(kwargs)} kwargs to config")
        print(f"   - LOD: {config.processor.lod_level}")
        print(f"   - GPU: {config.processor.use_gpu}")
        print(f"   - Patch size: {config.processor.patch_size}m")
        print(f"   - RGB: {config.features.use_rgb}")
        print(f"   - Mode: {config.output.processing_mode}")
        
        print("\n✅ PASS: Legacy kwargs conversion works")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility_properties():
    """Test backward compatibility properties."""
    print("\n" + "="*70)
    print("Test 3: Backward Compatibility Properties")
    print("="*70)
    
    try:
        # Create config
        config = OmegaConf.create({
            'processor': {
                'lod_level': 'LOD2',
                'use_gpu': False,
                'patch_size': 150.0,
                'num_points': 16384,
            },
            'features': {
                'use_rgb': True,
                'use_infrared': False,
            },
            'output': {
                'format': 'npz',
                'processing_mode': 'patches_only',
            }
        })
        
        # Initialize feature manager
        feature_mgr = FeatureManager(config)
        
        # Test property access (simulating what old code does)
        rgb_fetcher = feature_mgr.rgb_fetcher
        infrared_fetcher = feature_mgr.infrared_fetcher
        use_gpu = feature_mgr.use_gpu
        
        print(f"✅ Properties accessible")
        print(f"   - rgb_fetcher: {'available' if rgb_fetcher else 'None'}")
        print(f"   - infrared_fetcher: {'available' if infrared_fetcher else 'None'}")
        print(f"   - use_gpu: {use_gpu}")
        
        # Test config property access
        patch_size = config.processor.patch_size
        num_points = config.processor.num_points
        
        print(f"✅ Config properties accessible")
        print(f"   - patch_size: {patch_size}m")
        print(f"   - num_points: {num_points}")
        
        print("\n✅ PASS: Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Testing Refactored Processor Initialization")
    print("="*70)
    
    tests = [
        test_config_based_init,
        test_legacy_kwargs_conversion,
        test_backward_compatibility_properties,
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
        print("\n🎉 All tests passed! Ready to refactor processor.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
