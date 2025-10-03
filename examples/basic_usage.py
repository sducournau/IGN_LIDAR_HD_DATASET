#!/usr/bin/env python3
"""
Basic Usage Example - IGN LiDAR HD Processing

This example demonstrates the basic functionality of the ign-lidar-hd library
for processing LiDAR data into machine learning datasets.
"""

from pathlib import Path
from ign_lidar import LiDARProcessor, IGNLiDARDownloader


def basic_processing_example():
    """Demonstrate basic LiDAR processing workflow."""
    print("🔄 Basic LiDAR Processing Example")
    print("=" * 40)
    
    # Initialize processor for LOD2 classification
    processor = LiDARProcessor(
        lod_level="LOD2",
        augment=True,
        num_augmentations=2,
        patch_size=150.0
    )
    
    # Example: Process a single LAS/LAZ file
    input_file = "sample_data.laz"  # Replace with your file
    output_dir = "output/"
    
    if Path(input_file).exists():
        print(f"📁 Processing: {input_file}")
        patches = processor.process_tile(input_file, output_dir)
        print(f"✅ Generated {len(patches)} patches")
    else:
        print(f"⚠️  Sample file {input_file} not found")
        print("   Create a sample LAS/LAZ file or download one using the downloader")


def download_and_process_example():
    """Demonstrate downloading and processing workflow."""
    print("\n🌐 Download and Process Example")
    print("=" * 40)
    
    # Initialize downloader
    downloader = IGNLiDARDownloader(
        output_dir="downloads/",
        num_workers=2
    )
    
    # Download a few sample tiles (uncomment to try)
    # tiles_to_download = ["0186_6834", "0192_6838"]  # Sample tile IDs
    # downloaded_files = downloader.download_tiles(tiles_to_download)
    
    print("📋 Available sample tiles in library:")
    from ign_lidar import WORKING_TILES
    
    # Show first 3 tiles as examples
    for i, tile in enumerate(WORKING_TILES[:3]):
        print(f"  {i+1}. {tile.location} ({tile.environment})")
        print(f"     File: {tile.filename}")
        print(f"     LOD: {tile.recommended_lod}")
        print()


def feature_extraction_example():
    """Demonstrate feature extraction capabilities."""
    print("🔧 Feature Extraction Example")
    print("=" * 40)
    
    # Import feature functions
    from ign_lidar import (
        compute_normals, 
        compute_curvature, 
        extract_geometric_features
    )
    
    print("Available feature extraction functions:")
    print("  • compute_normals() - Surface normal vectors")
    print("  • compute_curvature() - Surface curvature")
    print("  • extract_geometric_features() - Comprehensive feature set")
    
    # Note: These functions work on numpy arrays of XYZ coordinates
    # Example usage would require actual point cloud data


def classification_schema_example():
    """Show available classification schemas."""
    print("\n📊 Classification Schemas")
    print("=" * 40)
    
    from ign_lidar import LOD2_CLASSES, LOD3_CLASSES
    
    print(f"LOD2 Schema: {len(LOD2_CLASSES)} classes")
    for class_id, class_name in list(LOD2_CLASSES.items())[:5]:
        print(f"  {class_id}: {class_name}")
    print("  ...")
    
    print(f"\nLOD3 Schema: {len(LOD3_CLASSES)} classes")
    for class_id, class_name in list(LOD3_CLASSES.items())[:5]:
        print(f"  {class_id}: {class_name}")
    print("  ...")


def main():
    """Run all examples."""
    print("🚀 IGN LiDAR HD Library - Examples")
    print("=" * 50)
    
    basic_processing_example()
    download_and_process_example()
    feature_extraction_example()
    classification_schema_example()
    
    print("\n" + "=" * 50)
    print("✅ Examples completed!")
    print("\nNext steps:")
    print("1. Install the library: pip install ign-lidar-hd")
    print("2. Download sample data or use your own LAS/LAZ files")
    print("3. Try the CLI: ign-lidar-process --help")


if __name__ == "__main__":
    main()