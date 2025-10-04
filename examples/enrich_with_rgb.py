#!/usr/bin/env python3
"""
Example: Enrich LAZ files with geometric features AND RGB colors from IGN orthophotos

This example demonstrates how to use the enrich command with RGB augmentation
to add both geometric features and RGB colors from IGN's orthophoto service.
"""

import subprocess
from pathlib import Path


def enrich_with_rgb_example():
    """
    Example of enriching LAZ files with geometric features + RGB colors.
    
    The --add-rgb flag enables RGB augmentation from IGN orthophotos.
    Optionally, use --rgb-cache-dir to cache downloaded orthophotos.
    """
    
    # Input and output directories
    input_dir = Path("data/raw")
    output_dir = Path("data/enriched_rgb")
    rgb_cache_dir = Path("cache/orthophotos")
    
    # Build the command
    cmd = [
        "python", "-m", "ign_lidar.cli",
        "enrich",
        "--input-dir", str(input_dir),
        "--output", str(output_dir),
        "--num-workers", "4",
        "--k-neighbors", "10",
        "--mode", "core",  # or 'full' for full features
        "--add-rgb",  # Enable RGB augmentation
        "--rgb-cache-dir", str(rgb_cache_dir),  # Optional: cache orthophotos
    ]
    
    print("Running enrichment with RGB augmentation...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the command
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print("\n✅ Enrichment with RGB completed successfully!")
        print(f"Output directory: {output_dir}")
    else:
        print(f"\n❌ Enrichment failed with return code {result.returncode}")


def enrich_without_rgb_cache():
    """
    Example without RGB cache (orthophotos fetched each time).
    """
    input_dir = Path("data/raw")
    output_dir = Path("data/enriched_rgb_nocache")
    
    cmd = [
        "python", "-m", "ign_lidar.cli",
        "enrich",
        "--input-dir", str(input_dir),
        "--output", str(output_dir),
        "--add-rgb",  # Enable RGB, but no cache
    ]
    
    subprocess.run(cmd)


def enrich_building_mode_with_rgb():
    """
    Example with building mode (full features) + RGB augmentation.
    """
    input_file = Path("data/raw/tile_0001.laz")
    output_dir = Path("data/enriched_building_rgb")
    rgb_cache_dir = Path("cache/orthophotos")
    
    cmd = [
        "python", "-m", "ign_lidar.cli",
        "enrich",
        "--input", str(input_file),
        "--output", str(output_dir),
        "--mode", "full",  # Full building features
        "--add-rgb",  # Plus RGB colors
        "--rgb-cache-dir", str(rgb_cache_dir),
        "--use-gpu",  # Optional: use GPU acceleration
    ]
    
    subprocess.run(cmd)


if __name__ == "__main__":
    # Run the main example
    enrich_with_rgb_example()
    
    # Uncomment to try other examples:
    # enrich_without_rgb_cache()
    # enrich_building_mode_with_rgb()
