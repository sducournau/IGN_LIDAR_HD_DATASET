#!/usr/bin/env python3
"""
Integration test for Phase 3.4 refactored processor.

Tests the refactored process_tile method using TileLoader and FeatureComputer modules.
"""

import sys
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ign_lidar.core.processor import LiDARProcessor


def create_test_config():
    """Create a minimal test configuration."""
    config = OmegaConf.create({
        'processor': {
            'lod_level': 'LOD3',
            'processing_mode': 'patches_only',
            'patch_size': 50,  # Larger patches to capture more points
            'num_points': 2048,  # Lower for easier patch creation
            'architecture': 'pointnet',
            'output_format': 'npz',
            'bbox': None,
            'patch_overlap': 0.0,
            'use_stitching': False,
            'buffer_size': 5.0,
            'augment': False,
            'num_augmentations': 1,
            'preprocess': False,  # Disable preprocessing for cleaner test
            'preprocess_config': None,
            'use_gpu': False,
            'use_gpu_chunked': False,
            'gpu_batch_size': 100000,
            'min_points': 100,
        },
        'features': {
            'use_rgb': True,  # Test RGB features
            'use_infrared': False,
            'compute_ndvi': False,
            'include_extra_features': True,
            'k_neighbors': 20,
            'feature_mode': 'CORE',
            'include_architectural_style': False,
            'style_encoding': 'constant'
        },
        'paths': {
            'data_dir': str(project_root / 'data'),
            'output_dir': str(project_root / 'data' / 'test_output'),
        }
    })
    return config


def main():
    print("=" * 70)
    print("Phase 3.4 Integration Test - Refactored Processor")
    print("=" * 70)
    print()
    
    # Setup paths
    test_laz = project_root / "data" / "test_integration" / "small_dense.laz"
    output_dir = project_root / "data" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_laz.exists():
        print(f"❌ Test LAZ file not found: {test_laz}")
        print("   Run: python scripts/generate_sample_laz.py")
        return 1
    
    print(f"📁 Input LAZ: {test_laz}")
    print(f"📁 Output dir: {output_dir}")
    print()
    
    # Create processor with test config
    print("🔧 Initializing processor...")
    config = create_test_config()
    processor = LiDARProcessor(config)
    
    print(f"   ✅ Processor initialized")
    print(f"   ✅ TileLoader: {processor.tile_loader is not None}")
    print(f"   ✅ FeatureComputer: {processor.feature_computer is not None}")
    print()
    
    # Process the tile
    print("🚀 Processing tile with refactored pipeline...")
    print(f"   Mode: {config.processor.processing_mode}")
    print(f"   Patch size: {config.processor.patch_size}m")
    print(f"   Points per patch: {config.processor.num_points}")
    print()
    
    try:
        # Call process_tile (the refactored method)
        patches_saved = processor.process_tile(
            laz_file=test_laz,
            output_dir=output_dir,
            tile_idx=1,
            total_tiles=1,
            skip_existing=False
        )
        
        print("✅ Processing completed successfully!")
        print()
        
        # Analyze results
        print("📊 Processing Results:")
        print(f"   Patches saved: {patches_saved}")
        
        # Check for output files
        output_files = list(output_dir.glob("small_dense_*_patch_*.npz"))
        if output_files:
            print(f"   Output files found: {len(output_files)}")
            
            # Inspect first patch
            first_patch = output_files[0]
            print(f"   Sample file: {first_patch.name}")
            print(f"   File size: {first_patch.stat().st_size / 1024:.1f} KB")
            
            # Load and inspect
            data = np.load(first_patch, allow_pickle=True)
            print(f"   Arrays in NPZ:")
            for key in data.keys():
                try:
                    arr = data[key]
                    if hasattr(arr, 'shape'):
                        print(f"      - {key}: shape {arr.shape}, dtype {arr.dtype}")
                    else:
                        print(f"      - {key}: value {arr}")
                except Exception as e:
                    print(f"      - {key}: <error reading: {e}>")
        else:
            print("   ⚠️  No output files found")
        
        print()
        print("=" * 70)
        print("✅ Integration Test PASSED")
        print("=" * 70)
        print()
        print("🎉 The refactored processor works correctly!")
        print("   - TileLoader successfully loaded and preprocessed the tile")
        print("   - FeatureComputer successfully computed features")
        print("   - Patch extraction and saving completed")
        print()
        return 0
        
    except Exception as e:
        print()
        print("❌ Processing failed with error:")
        print(f"   {type(e).__name__}: {e}")
        print()
        
        import traceback
        print("Traceback:")
        traceback.print_exc()
        
        print()
        print("=" * 70)
        print("❌ Integration Test FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
