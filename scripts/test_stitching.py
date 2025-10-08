#!/usr/bin/env python3
"""
Test script for tile stitching functionality.

This script tests the new tiling/stitching support with various configurations
and validates the performance improvements.

Usage:
    python test_enhanced_stitching.py [test_data_dir]
"""

import sys
import time
import logging
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.core.tile_stitcher import TileStitcher
from ign_lidar.core.stitching_config import StitchingConfigManager, get_recommended_stitching_preset


def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_test_configs() -> List[dict]:
    """Create test configurations for different scenarios."""
    manager = StitchingConfigManager()
    
    configs = []
    
    # Basic configuration
    configs.append({
        'name': 'basic',
        'config': manager.load_config('basic'),
        'description': 'Basic stitching with minimal features'
    })
    
    # Enhanced configuration  
    configs.append({
        'name': 'enhanced', 
        'config': manager.load_config('enhanced'),
        'description': 'Enhanced stitching with adaptive buffers and boundary processing'
    })
    
    # Advanced configuration
    configs.append({
        'name': 'advanced',
        'config': manager.load_config('advanced'), 
        'description': 'Advanced stitching with multi-scale buffers and parallel processing'
    })
    
    # Custom hybrid configuration
    custom_config = manager.create_hybrid_config('enhanced', {
        'buffer_size': 25.0,
        'max_neighbors': 12,
        'parallel_loading': True,
        'boundary_smoothing': True
    })
    configs.append({
        'name': 'custom_hybrid',
        'config': custom_config,
        'description': 'Custom hybrid configuration with larger buffers'
    })
    
    return configs


def test_config_validation():
    """Test configuration validation and presets."""
    print("\n" + "="*60)
    print("Testing Configuration Management")
    print("="*60)
    
    manager = StitchingConfigManager()
    
    # Test preset loading
    for preset_name in ['disabled', 'basic', 'enhanced', 'advanced']:
        config = manager.load_config(preset_name)
        print(f"✓ Loaded preset '{preset_name}': {config['enabled']} - buffer_size: {config.get('buffer_size', 'N/A')}")
    
    # Test validation
    invalid_config = {
        'enabled': True,
        'buffer_size': -5.0,  # Invalid
        'max_neighbors': -1,  # Invalid
        'adaptive_buffer': True,
        'min_buffer': 15.0,   # Will be > max_buffer 
        'max_buffer': 10.0    # Invalid range
    }
    
    validated = manager.validate_config(invalid_config)
    print(f"✓ Config validation fixed {len(invalid_config) - len([k for k, v in validated.items() if k in invalid_config and invalid_config[k] == v])} invalid values")
    
    # Test recommendation system
    recommendations = []
    for tile_count, memory_gb in [(1, 8), (10, 4), (50, 8), (200, 16)]:
        preset = get_recommended_stitching_preset(tile_count, memory_gb)
        recommendations.append(f"{tile_count} tiles, {memory_gb}GB RAM → {preset}")
    
    print("✓ Preset recommendations:")
    for rec in recommendations:
        print(f"  {rec}")


def test_tile_stitcher(test_data_dir: Path):
    """Test the TileStitcher with sample data."""
    print("\n" + "="*60)  
    print("Testing Tile Stitcher")
    print("="*60)
    
    if not test_data_dir.exists():
        print(f"⚠️  Test data directory not found: {test_data_dir}")
        print("   Skipping stitcher tests")
        return
    
    # Find LAZ files
    laz_files = list(test_data_dir.glob("*.laz"))
    if not laz_files:
        print(f"⚠️  No LAZ files found in: {test_data_dir}")
        print("   Skipping stitcher tests")
        return
    
    print(f"📁 Found {len(laz_files)} LAZ files")
    
    # Test different configurations
    configs = create_test_configs()
    
    for config_info in configs:
        print(f"\n--- Testing {config_info['name']} configuration ---")
        print(f"Description: {config_info['description']}")
        
        try:
            # Initialize stitcher
            start_time = time.time()
            stitcher = TileStitcher(config=config_info['config'])
            init_time = time.time() - start_time
            
            # Test with first LAZ file
            test_file = laz_files[0]
            print(f"🔧 Processing: {test_file.name}")
            
            start_time = time.time()
            result = stitcher.load_tile_with_smart_neighbors(test_file)
            process_time = time.time() - start_time
            
            # Print results
            print(f"✓ Core points: {result['num_core']:,}")
            buffer_info = result.get('buffer_zones', {})
            for buffer_type, buffer_data in buffer_info.items():
                if isinstance(buffer_data, dict) and 'points' in buffer_data:
                    points_count = len(buffer_data['points']) if buffer_data['points'] is not None else 0
                    print(f"  Buffer ({buffer_type}): {points_count:,} points")
            
            metadata = result.get('stitching_metadata', {})
            neighbors_used = metadata.get('neighbors_used', 0)
            print(f"✓ Neighbors used: {neighbors_used}")
            print(f"✓ Processing time: {process_time:.2f}s (init: {init_time:.3f}s)")
            
            # Get statistics
            stats = stitcher.get_statistics()
            if stats:
                print(f"📊 Statistics:")
                print(f"  Cache hits: {stats.get('cache_hits', 0)}")
                print(f"  Cache misses: {stats.get('cache_misses', 0)}")
                print(f"  Memory usage: {stats.get('memory_usage_mb', 0):.1f} MB")
            
            # Cleanup
            stitcher.cleanup()
            
        except Exception as e:
            print(f"❌ Error testing {config_info['name']}: {e}")
            import traceback
            traceback.print_exc()


def test_performance_comparison(test_data_dir: Path):
    """Compare performance between basic and advanced stitching."""
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    
    if not test_data_dir.exists():
        print("⚠️  Skipping performance tests - no test data")
        return
    
    laz_files = list(test_data_dir.glob("*.laz"))
    if len(laz_files) < 3:
        print("⚠️  Need at least 3 LAZ files for performance testing")
        return
    
    manager = StitchingConfigManager()
    
    # Test basic vs enhanced configurations
    test_configs = [
        ('basic', manager.load_config('basic')),
        ('enhanced', manager.load_config('enhanced'))
    ]
    
    results = {}
    
    for config_name, config in test_configs:
        print(f"\n🚀 Testing {config_name} configuration...")
        
        times = []
        total_points = 0
        
        try:
            stitcher = TileStitcher(config=config)
            
            for i, laz_file in enumerate(laz_files[:3]):  # Test first 3 files
                start_time = time.time()
                result = stitcher.load_tile_with_smart_neighbors(laz_file)
                process_time = time.time() - start_time
                
                times.append(process_time)
                total_points += result['num_core']
                
                print(f"  File {i+1}: {process_time:.2f}s - {result['num_core']:,} points")
            
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            
            results[config_name] = {
                'avg_time': avg_time,
                'total_time': total_time,
                'total_points': total_points,
                'points_per_second': total_points / total_time if total_time > 0 else 0
            }
            
            stitcher.cleanup()
            
        except Exception as e:
            print(f"❌ Error in {config_name}: {e}")
            results[config_name] = None
    
    # Compare results
    print(f"\n📊 Performance Summary:")
    print(f"{'Configuration':<12} {'Avg Time':<10} {'Total Time':<12} {'Points/sec':<12}")
    print("-" * 50)
    
    for config_name, result in results.items():
        if result:
            print(f"{config_name:<12} {result['avg_time']:<10.2f} "
                  f"{result['total_time']:<12.2f} {result['points_per_second']:<12,.0f}")
        else:
            print(f"{config_name:<12} {'FAILED':<10} {'FAILED':<12} {'FAILED':<12}")
    
    # Performance improvement analysis
    if results.get('basic') and results.get('enhanced'):
        basic_pps = results['basic']['points_per_second']
        enhanced_pps = results['enhanced']['points_per_second']
        
        if basic_pps > 0:
            improvement = ((enhanced_pps - basic_pps) / basic_pps) * 100
            print(f"\n🎯 Enhanced vs Basic: {improvement:+.1f}% performance change")


def main():
    """Main test function."""
    setup_logging()
    
    print("🧪 IGN LiDAR HD Enhanced Stitching Test Suite")
    print("=" * 60)
    
    # Get test data directory
    if len(sys.argv) > 1:
        test_data_dir = Path(sys.argv[1])
    else:
        # Default test data locations
        possible_dirs = [
            Path("/mnt/c/Users/Simon/ign/raw_tiles/urban_dense"),
            Path("test_data"),
            Path("data/raw"),
            Path(".")
        ]
        
        test_data_dir = None
        for dir_path in possible_dirs:
            if dir_path.exists() and list(dir_path.glob("*.laz")):
                test_data_dir = dir_path
                break
        
        if test_data_dir is None:
            test_data_dir = Path(".")
    
    print(f"📂 Test data directory: {test_data_dir}")
    
    # Run tests
    try:
        test_config_validation()
        test_tile_stitcher(test_data_dir) 
        test_performance_comparison(test_data_dir)
        
        print("\n" + "="*60)
        print("✅ Enhanced stitching test suite completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()