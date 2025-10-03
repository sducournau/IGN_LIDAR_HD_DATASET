"""
Demo: Data Augmentation at ENRICH Phase

This example demonstrates the improved data augmentation approach where
geometric transformations are applied BEFORE feature computation.

Benefits:
1. ✅ Features computed on augmented geometry (coherent)
2. ✅ Normals, curvature, planarity reflect true augmented surface
3. ✅ Better model training quality
4. ✅ No feature-geometry mismatch

Old approach (PATCH phase):
- Rotate/jitter coordinates
- Rotate normals
- Copy other features (WRONG! ❌)

New approach (ENRICH phase):
- Rotate/jitter coordinates
- Recompute ALL features on augmented geometry (CORRECT! ✅)
"""

from pathlib import Path
from ign_lidar import LiDARProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def demo_augmentation_enrich():
    """
    Demo augmentation during ENRICH phase.
    
    This processes a LAZ tile and creates multiple augmented versions,
    each with properly computed geometric features.
    """
    
    # Input/output paths
    input_dir = Path("data/raw")
    output_dir = Path("data/patches_augmented")
    
    print("=" * 70)
    print("DATA AUGMENTATION AT ENRICH PHASE")
    print("=" * 70)
    print()
    print("Old approach (PATCH): Augment after features → Inconsistent")
    print("New approach (ENRICH): Augment before features → Coherent ✅")
    print()
    
    # Create processor with augmentation enabled
    processor = LiDARProcessor(
        lod_level='LOD2',
        patch_size=150.0,
        num_points=16384,
        
        # Augmentation settings
        augment=True,               # Enable augmentation
        num_augmentations=3,        # Create 3 augmented versions
        
        # Feature computation
        include_extra_features=True,  # Full building features
        k_neighbors=20,               # Manual k (or None for auto)
        use_gpu=True,                 # Use GPU if available
        
        # RGB augmentation (optional)
        include_rgb=False
    )
    
    print(f"Configuration:")
    print(f"  - LOD Level: LOD2 (6 classes)")
    print(f"  - Patch size: 150m")
    print(f"  - Points per patch: 16,384")
    print(f"  - Augmentations: 3 versions per tile")
    print(f"  - Features: BUILDING mode (full)")
    print(f"  - GPU: {'Enabled' if processor.use_gpu else 'CPU only'}")
    print()
    
    # Process tiles
    print("Processing tiles with augmentation...")
    print("-" * 70)
    
    num_patches = processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        num_workers=4,
        skip_existing=False  # Force reprocessing to see effect
    )
    
    print()
    print("=" * 70)
    print(f"✅ Complete! Created {num_patches} total patches")
    print(f"   (Original + augmented versions with coherent features)")
    print("=" * 70)
    print()
    print("What was computed:")
    print("  For EACH version (original + 3 augmented):")
    print("    1. Apply rotation, jitter, scaling, dropout")
    print("    2. Compute normals on augmented geometry")
    print("    3. Compute curvature on augmented geometry")
    print("    4. Compute planarity/linearity on augmented geometry")
    print("    5. Extract patches with coherent features")
    print()
    print("Result: Training data with proper feature-geometry alignment!")


def compare_approaches():
    """
    Conceptual comparison of old vs new approach.
    """
    print()
    print("=" * 70)
    print("APPROACH COMPARISON")
    print("=" * 70)
    print()
    
    print("📊 OLD APPROACH (Augmentation at PATCH phase):")
    print("-" * 70)
    print("  1. Load LAZ")
    print("  2. Compute features (normals, curvature, etc.)")
    print("  3. Extract patches")
    print("  4. For each patch:")
    print("       - Rotate coordinates")
    print("       - Rotate normals")
    print("       - COPY other features (curvature, planarity, etc.)")
    print()
    print("  ❌ PROBLEM: Curvature/planarity don't match rotated geometry!")
    print("  ❌ Model learns incorrect feature-geometry associations")
    print()
    
    print("📊 NEW APPROACH (Augmentation at ENRICH phase):")
    print("-" * 70)
    print("  1. Load LAZ")
    print("  2. For each version (original + augmentations):")
    print("       a. Apply rotation, jitter, scaling, dropout")
    print("       b. Compute ALL features on augmented points")
    print("       c. Extract patches")
    print()
    print("  ✅ BENEFIT: All features computed on correct geometry!")
    print("  ✅ Model learns consistent patterns")
    print("  ✅ Better generalization")
    print()
    
    print("💰 COST:")
    print("  - ~40% more computation time (features computed per version)")
    print("  - Worth it for better training data quality!")
    print()


if __name__ == "__main__":
    compare_approaches()
    
    print("\nReady to run demo? (Make sure you have LAZ files in data/raw/)")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        demo_augmentation_enrich()
    else:
        print("Demo cancelled.")
