#!/usr/bin/env python3
"""
Advanced Processing Example - IGN LiDAR HD

This example shows advanced features like batch processing,
custom filtering, and parallel processing.
"""

from pathlib import Path
from ign_lidar import LiDARProcessor, get_tiles_by_environment


def batch_processing_example():
    """Demonstrate batch processing of multiple files."""
    print("📦 Batch Processing Example")
    print("=" * 40)
    
    processor = LiDARProcessor(
        lod_level="LOD3",
        augment=True,
        num_augmentations=3,
        patch_size=100.0,  # Smaller patches for dense areas
        patch_overlap=0.2   # More overlap for better coverage
    )
    
    # Process all LAZ files in a directory
    input_dir = Path("input_data/")
    output_dir = Path("output_patches/")
    
    if input_dir.exists():
        print(f"🔍 Processing directory: {input_dir}")
        
        # Process with parallel workers
        all_patches = processor.process_directory(
            str(input_dir),
            str(output_dir),
            num_workers=4
        )
        
        print(f"✅ Total patches generated: {len(all_patches)}")
    else:
        print(f"⚠️  Input directory {input_dir} not found")


def filtered_processing_example():
    """Demonstrate spatial filtering with bounding boxes."""
    print("\n🗺️ Filtered Processing Example")
    print("=" * 40)
    
    # Process only a specific region (example coordinates)
    bbox = (650000, 6860000, 651000, 6861000)  # Lambert93 coordinates
    
    processor = LiDARProcessor(
        lod_level="LOD2",
        bbox=bbox,  # Only process points within this bbox
        augment=False  # No augmentation for precise analysis
    )
    
    print(f"📍 Processing with spatial filter:")
    print(f"   Bounding box: {bbox}")
    print("   This allows focused analysis of specific areas")


def environment_based_processing():
    """Demonstrate processing tiles by environment type."""
    print("\n🏞️ Environment-Based Processing")
    print("=" * 40)
    
    # Get tiles by environment
    urban_tiles = get_tiles_by_environment("urban")
    coastal_tiles = get_tiles_by_environment("coastal")
    rural_tiles = get_tiles_by_environment("rural")
    
    print(f"Available tiles by environment:")
    print(f"  🏙️  Urban: {len(urban_tiles)} tiles")
    print(f"  🌊 Coastal: {len(coastal_tiles)} tiles")
    print(f"  🌾 Rural: {len(rural_tiles)} tiles")
    
    # Different processing strategies per environment
    processors = {
        "urban": LiDARProcessor(lod_level="LOD3", patch_size=100.0),
        "coastal": LiDARProcessor(lod_level="LOD2", patch_size=150.0),
        "rural": LiDARProcessor(lod_level="LOD2", patch_size=200.0)
    }
    
    print("\n🔧 Recommended processing strategies:")
    print("  Urban areas: LOD3, smaller patches (100m)")
    print("  Coastal areas: LOD2, medium patches (150m)")
    print("  Rural areas: LOD2, larger patches (200m)")


def custom_feature_processing():
    """Show how to use custom feature extraction."""
    print("\n⚙️ Custom Feature Processing")
    print("=" * 40)
    
    processor = LiDARProcessor(lod_level="LOD2")
    
    # The processor automatically extracts these features:
    features_info = {
        "Geometric": ["X, Y, Z coordinates", "Height above ground"],
        "Surface": ["Normal vectors (nx, ny, nz)", "Curvature"],
        "Density": ["Local point density", "Neighborhood statistics"],
        "Statistical": ["Planarity", "Verticality", "Anisotropy"]
    }
    
    print("Automatically extracted features:")
    for category, features in features_info.items():
        print(f"  {category}:")
        for feature in features:
            print(f"    • {feature}")


def main():
    """Run advanced examples."""
    print("🚀 IGN LiDAR HD Library - Advanced Examples")
    print("=" * 55)
    
    batch_processing_example()
    filtered_processing_example()
    environment_based_processing()
    custom_feature_processing()
    
    print("\n" + "=" * 55)
    print("✅ Advanced examples completed!")
    print("\nFor more details, check the library documentation.")


if __name__ == "__main__":
    main()