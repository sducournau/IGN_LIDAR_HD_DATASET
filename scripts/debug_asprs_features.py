#!/usr/bin/env python3
"""
Debug ASPRS Feature Filtering and Saving

This script tests the complete pipeline to identify where features are dropped:
1. Feature computation
2. Feature mode filtering
3. LAZ file saving

Usage:
    python scripts/debug_asprs_features.py [input.laz]
    
If no input file provided, creates synthetic test data.
"""

import sys
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.features.feature_modes import ASPRS_FEATURES, FeatureMode
from ign_lidar.features.orchestrator import FeatureOrchestrator
from ign_lidar.core.classification.io import save_enriched_tile_laz

def create_synthetic_features(n_points=1000):
    """Create synthetic features matching ASPRS mode expectations."""
    logger.info(f"Creating synthetic features for {n_points:,} points")
    
    features = {
        # Normals (should be split into 3 dimensions)
        'normal_x': np.random.rand(n_points).astype(np.float32),
        'normal_y': np.random.rand(n_points).astype(np.float32),
        'normal_z': np.random.rand(n_points).astype(np.float32),
        
        # Shape descriptors
        'planarity': np.random.rand(n_points).astype(np.float32),
        'sphericity': np.random.rand(n_points).astype(np.float32),
        'curvature': np.random.rand(n_points).astype(np.float32) * 0.5,
        
        # Height features
        'height': np.random.rand(n_points).astype(np.float32) * 10,
        'height_above_ground': np.random.rand(n_points).astype(np.float32) * 5,
        
        # Building detection
        'verticality': np.random.rand(n_points).astype(np.float32),
        'horizontality': np.random.rand(n_points).astype(np.float32),
        
        # Density
        'density': np.random.rand(n_points).astype(np.float32) * 100,
        
        # Spectral (optional)
        'red': (np.random.rand(n_points) * 255).astype(np.uint8),
        'green': (np.random.rand(n_points) * 255).astype(np.uint8),
        'blue': (np.random.rand(n_points) * 255).astype(np.uint8),
        'nir': (np.random.rand(n_points) * 255).astype(np.uint8),
        'ndvi': (np.random.rand(n_points) * 2 - 1).astype(np.float32),
        
        # Additional features NOT in ASPRS mode (should be filtered)
        'linearity': np.random.rand(n_points).astype(np.float32),
        'anisotropy': np.random.rand(n_points).astype(np.float32),
        'eigenentropy': np.random.rand(n_points).astype(np.float32),
    }
    
    return features

def test_feature_filtering():
    """Test feature filtering with ASPRS mode."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Feature Mode Filtering")
    logger.info("="*80)
    
    # Create features
    features = create_synthetic_features()
    logger.info(f"\n📊 Created {len(features)} features:")
    for name in sorted(features.keys()):
        logger.info(f"   - {name}")
    
    # Check against ASPRS_FEATURES definition
    logger.info(f"\n📋 ASPRS_FEATURES set contains:")
    for name in sorted(ASPRS_FEATURES):
        logger.info(f"   - {name}")
    
    # Initialize orchestrator with ASPRS mode
    config = {
        'features': {
            'mode': 'asprs_classes',
            'k_neighbors': 20,
        }
    }
    
    orchestrator = FeatureOrchestrator(config, feature_mode=FeatureMode.ASPRS_CLASSES)
    
    # Test filtering
    logger.info(f"\n🔽 Applying ASPRS mode filter...")
    filtered = orchestrator.filter_features(features, FeatureMode.ASPRS_CLASSES)
    
    logger.info(f"\n📊 After filtering: {len(filtered)} features:")
    for name in sorted(filtered.keys()):
        logger.info(f"   ✓ {name}")
    
    # Identify dropped features
    dropped = set(features.keys()) - set(filtered.keys())
    if dropped:
        logger.warning(f"\n⚠️  DROPPED {len(dropped)} features:")
        for name in sorted(dropped):
            logger.warning(f"   ✗ {name}")
    else:
        logger.info(f"\n✅ No features dropped")
    
    # Check for features in ASPRS_FEATURES but not in filtered
    expected_in_mode = ASPRS_FEATURES - {'xyz'}  # xyz is coordinate, not a feature array
    missing_from_filtered = expected_in_mode - set(filtered.keys())
    if missing_from_filtered:
        logger.error(f"\n❌ Expected ASPRS features NOT in filtered result:")
        for name in sorted(missing_from_filtered):
            logger.error(f"   ✗ {name}")
    
    return filtered

def test_laz_saving(filtered_features):
    """Test saving filtered features to LAZ."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: LAZ File Saving")
    logger.info("="*80)
    
    n_points = len(next(iter(filtered_features.values())))
    
    # Create point cloud data
    points = np.random.rand(n_points, 3).astype(np.float32) * 100
    classification = np.random.randint(0, 10, n_points, dtype=np.uint8)
    intensity = np.random.rand(n_points).astype(np.float32)
    return_number = np.ones(n_points, dtype=np.float32)
    
    # Extract RGB/NIR if present
    input_rgb = None
    input_nir = None
    features_to_save = filtered_features.copy()
    
    if 'red' in features_to_save and 'green' in features_to_save and 'blue' in features_to_save:
        input_rgb = np.column_stack([
            features_to_save.pop('red'),
            features_to_save.pop('green'),
            features_to_save.pop('blue')
        ])
        logger.info(f"   Extracted RGB for standard LAZ field")
    
    if 'nir' in features_to_save:
        input_nir = features_to_save.pop('nir')
        logger.info(f"   Extracted NIR for standard LAZ field")
    
    logger.info(f"\n📊 Features to save as extra dimensions: {len(features_to_save)}")
    for name in sorted(features_to_save.keys()):
        data = features_to_save[name]
        logger.info(f"   - {name}: shape={data.shape}, dtype={data.dtype}")
    
    # Save to LAZ
    output_path = Path('/tmp/test_asprs_features_debug.laz')
    logger.info(f"\n💾 Saving to: {output_path}")
    
    try:
        save_enriched_tile_laz(
            save_path=output_path,
            points=points,
            classification=classification,
            intensity=intensity,
            return_number=return_number,
            features=features_to_save,
            original_las=None,
            header=None,
            input_rgb=input_rgb,
            input_nir=input_nir
        )
        logger.info(f"✅ Save successful!")
        
    except Exception as e:
        logger.error(f"❌ Save failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Verify saved file
    logger.info(f"\n🔍 Verifying saved file...")
    try:
        import laspy
        las = laspy.read(str(output_path))
        
        logger.info(f"   ✓ File loaded successfully")
        logger.info(f"   ✓ Points: {len(las.points):,}")
        logger.info(f"   ✓ Point format: {las.point_format.id}")
        
        # Check extra dimensions
        extra_dims = list(las.point_format.extra_dimension_names)
        logger.info(f"\n📊 Extra dimensions in LAZ: {len(extra_dims)}")
        for dim in sorted(extra_dims):
            logger.info(f"   ✓ {dim}")
        
        # Check for missing features
        missing = set(features_to_save.keys()) - set(extra_dims)
        if missing:
            logger.error(f"\n❌ Features NOT saved to LAZ:")
            for name in sorted(missing):
                logger.error(f"   ✗ {name}")
        else:
            logger.info(f"\n✅ All features saved successfully!")
        
        # Check standard fields
        logger.info(f"\n📊 Standard fields:")
        logger.info(f"   ✓ X, Y, Z")
        logger.info(f"   ✓ Classification")
        logger.info(f"   ✓ Intensity")
        logger.info(f"   ✓ Return Number")
        if hasattr(las, 'red'):
            logger.info(f"   ✓ RGB")
        if hasattr(las, 'nir'):
            logger.info(f"   ✓ NIR")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_real_file(laz_path):
    """Test with a real LAZ file."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Real LAZ File Processing")
    logger.info("="*80)
    
    logger.info(f"\n📂 Loading: {laz_path}")
    
    try:
        import laspy
        from ign_lidar.core.classification.io import TileLoader
        
        # Load tile
        loader = TileLoader()
        tile_data = loader.load_tile(Path(laz_path))
        
        if tile_data is None:
            logger.error(f"❌ Failed to load tile")
            return False
        
        logger.info(f"✓ Loaded {len(tile_data['points']):,} points")
        
        # Check for existing enriched features
        enriched = tile_data.get('enriched_features', {})
        if enriched:
            logger.info(f"\n✨ Found {len(enriched)} enriched features:")
            for name in sorted(enriched.keys()):
                logger.info(f"   - {name}")
        
        # Compute features with orchestrator
        config = {
            'features': {
                'mode': 'asprs_classes',
                'k_neighbors': 20,
                'search_radius': 1.0,
                'compute_normals': True,
                'compute_curvature': True,
                'compute_height': True,
                'compute_geometric': True,
            }
        }
        
        orchestrator = FeatureOrchestrator(config, feature_mode=FeatureMode.ASPRS_CLASSES)
        
        logger.info(f"\n⚙️  Computing features with ASPRS mode...")
        computed = orchestrator.compute_features(tile_data)
        
        logger.info(f"\n📊 Computed {len(computed)} features:")
        for name in sorted(computed.keys()):
            data = computed[name]
            if isinstance(data, np.ndarray):
                logger.info(f"   - {name}: shape={data.shape}, dtype={data.dtype}")
            else:
                logger.info(f"   - {name}: {type(data)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("ASPRS Feature Filtering and Saving Debug")
    logger.info("="*80)
    
    # Test 1: Feature filtering
    filtered = test_feature_filtering()
    
    # Test 2: LAZ saving
    success = test_laz_saving(filtered)
    
    # Test 3: Real file (if provided)
    if len(sys.argv) > 1:
        laz_path = sys.argv[1]
        if Path(laz_path).exists():
            test_with_real_file(laz_path)
        else:
            logger.warning(f"\n⚠️  File not found: {laz_path}")
    else:
        logger.info(f"\n💡 Tip: Provide a LAZ file path to test with real data:")
        logger.info(f"   python {sys.argv[0]} /path/to/file.laz")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if success:
        logger.info("✅ All tests passed")
        logger.info(f"   Test file saved to: /tmp/test_asprs_features_debug.laz")
        logger.info(f"   Verify with: python scripts/check_laz_features_v3.py /tmp/test_asprs_features_debug.laz")
    else:
        logger.error("❌ Tests failed - check logs above for details")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
