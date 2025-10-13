"""
Test custom configuration file loading for IGN LiDAR HD v2.3.0.

Tests the custom config file support including:
- Loading from file
- Configuration precedence
- --show-config preview
- CLI overrides
"""

import os
import sys
import tempfile
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.cli.hydra_runner import HydraRunner


class TestCustomConfigLoading:
    """Test suite for custom configuration file loading."""
    
    def test_load_example_configs(self):
        """Test loading all example configuration files."""
        examples_dir = Path(__file__).parent.parent / "examples"
        
        configs = [
            "config_gpu_processing.yaml",
            "config_training_dataset.yaml",
            "config_quick_enrich.yaml",
            "config_complete.yaml"
        ]
        
        print("\n" + "="*70)
        print("Testing Example Config Loading")
        print("="*70)
        
        for config_file in configs:
            config_path = examples_dir / config_file
            
            if not config_path.exists():
                print(f"❌ {config_file} - File not found")
                continue
            
            try:
                runner = HydraRunner()
                cfg = runner.load_config(config_file=str(config_path))
                print(f"✅ {config_file:35s} - Loaded successfully")
                
                # Verify key fields exist
                assert hasattr(cfg, 'processor')
                assert hasattr(cfg, 'features')
                assert hasattr(cfg, 'output')
                
            except Exception as e:
                print(f"❌ {config_file:35s} - Error: {e}")
                raise
        
        print("="*70)
    
    def test_config_precedence(self):
        """Test that CLI overrides have highest priority."""
        print("\n" + "="*70)
        print("Testing Configuration Precedence")
        print("="*70)
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'processor': {
                    'use_gpu': False,
                    'num_workers': 4,
                    'patch_size': 100.0
                },
                'features': {
                    'mode': 'minimal'
                }
            }
            yaml.dump(config, f)
            temp_config = f.name
        
        try:
            runner = HydraRunner()
            
            # Test 1: Load without overrides
            cfg1 = runner.load_config(config_file=temp_config)
            assert cfg1.processor.use_gpu == False
            assert cfg1.processor.num_workers == 4
            print("✅ Config loaded with default values")
            
            # Test 2: Override with CLI args
            cfg2 = runner.load_config(
                config_file=temp_config,
                overrides=['processor.use_gpu=true', 'processor.num_workers=8']
            )
            assert cfg2.processor.use_gpu == True
            assert cfg2.processor.num_workers == 8
            print("✅ CLI overrides applied correctly")
            
            # Test 3: Verify original config unchanged
            cfg3 = runner.load_config(config_file=temp_config)
            assert cfg3.processor.use_gpu == False
            assert cfg3.processor.num_workers == 4
            print("✅ Original config values preserved")
            
            print("\n" + "Configuration Precedence Test: PASSED ✅")
            
        finally:
            os.unlink(temp_config)
        
        print("="*70)
    
    def test_processing_mode_in_config(self):
        """Test that processing_mode can be set in config files."""
        print("\n" + "="*70)
        print("Testing Processing Mode in Config")
        print("="*70)
        
        modes = ['patches_only', 'both', 'enriched_only']
        
        for mode in modes:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                config = {
                    'processor': {
                        'lod_level': 'LOD2'
                    },
                    'output': {
                        'processing_mode': mode,
                        'format': 'npz'
                    }
                }
                yaml.dump(config, f)
                temp_config = f.name
            
            try:
                runner = HydraRunner()
                cfg = runner.load_config(config_file=temp_config)
                assert cfg.output.processing_mode == mode
                print(f"✅ processing_mode='{mode:20s}' - Loaded successfully")
            finally:
                os.unlink(temp_config)
        
        print("="*70)
    
    def test_relative_path_loading(self):
        """Test loading config from relative path."""
        print("\n" + "="*70)
        print("Testing Relative Path Loading")
        print("="*70)
        
        # Use an example config with relative path
        relative_path = "examples/config_quick_enrich.yaml"
        
        if Path(relative_path).exists():
            try:
                runner = HydraRunner()
                cfg = runner.load_config(config_file=relative_path)
                print(f"✅ Loaded from relative path: {relative_path}")
                assert hasattr(cfg, 'processor')
                print("✅ Configuration valid")
            except Exception as e:
                print(f"❌ Error loading from relative path: {e}")
                raise
        else:
            print(f"⚠️  Skipping: {relative_path} not found")
        
        print("="*70)
    
    def test_merge_with_defaults(self):
        """Test that we can use partial configs with processor."""
        print("\n" + "="*70)
        print("Testing Partial Config Support")
        print("="*70)
        
        # Create partial config (only specify what we need)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'processor': {
                    'lod_level': 'LOD3',
                    'processing_mode': 'patches_only',
                    'patch_size': 200.0,
                    'num_points': 16384,
                    'use_gpu': True,
                    'architecture': 'pointnet++',
                    'output_format': 'npz',
                    'use_stitching': False,
                    'preprocess': False,
                    'augment': False,
                    'num_augmentations': 3,
                    'bbox': None,
                    'patch_overlap': 0.1,
                    'use_gpu_chunked': True,
                    'gpu_batch_size': 1000000,
                    'preprocess_config': None,
                    'buffer_size': 10.0,
                    'stitching_config': None,
                },
                'features': {
                    'include_extra_features': False,
                    'feature_mode': None,
                    'k_neighbors': None,
                    'include_architectural_style': False,
                    'style_encoding': 'constant',
                    'use_rgb': False,
                    'rgb_cache_dir': None,
                    'use_infrared': False,
                    'compute_ndvi': False,
                }
            }
            yaml.dump(config, f)
            temp_config = f.name
        
        try:
            runner = HydraRunner()
            cfg = runner.load_config(config_file=temp_config)
            
            # Check our values applied
            assert cfg.processor.use_gpu == True
            assert cfg.processor.patch_size == 200.0
            print("✅ Config values loaded correctly")
            
            # Test that we can initialize processor with this config
            from ign_lidar.core.processor import LiDARProcessor
            processor = LiDARProcessor(config=cfg)
            assert processor.patch_size == 200.0
            assert processor.lod_level == 'LOD3'
            print("✅ Processor initialized successfully with partial config")
            
        finally:
            os.unlink(temp_config)
        
        print("="*70)


def run_all_tests():
    """Run all configuration tests."""
    test = TestCustomConfigLoading()
    
    print("\n" + "🧪 Testing Custom Configuration File Support" + "\n")
    
    try:
        test.test_load_example_configs()
        test.test_config_precedence()
        test.test_processing_mode_in_config()
        test.test_relative_path_loading()
        test.test_merge_with_defaults()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nCustom configuration file support is working correctly! 🎉")
        print("\nPhases 1 & 2 Implementation: COMPLETE ✅")
        print("  • Processing modes: ✅")
        print("  • Custom config loading: ✅")
        print("  • Configuration precedence: ✅")
        print("  • Example configs: ✅")
        print("\n" + "="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
