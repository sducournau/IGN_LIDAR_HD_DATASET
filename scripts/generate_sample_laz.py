#!/usr/bin/env python3
"""
Generate sample LAZ files for testing unified processing pipeline.

This script creates synthetic LAZ files with various characteristics:
- Different point densities
- Multiple classification labels
- RGB colors
- Various spatial distributions
"""

import numpy as np
import laspy
from pathlib import Path
import argparse


def generate_sample_laz(
    output_path: Path,
    num_points: int = 100000,
    extent: float = 100.0,
    classification_labels: list = None,
    add_rgb: bool = True,
    add_infrared: bool = False,
    seed: int = 42
):
    """
    Generate a sample LAZ file with synthetic LiDAR data.
    
    Args:
        output_path: Path to output LAZ file
        num_points: Number of points to generate
        extent: Spatial extent in meters (square area)
        classification_labels: List of classification labels to use
        add_rgb: Whether to add RGB colors
        add_infrared: Whether to add infrared channel
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    if classification_labels is None:
        # Default: building, ground, vegetation, water, unclassified
        classification_labels = [6, 2, 3, 9, 1]
    
    print(f"Generating {num_points:,} points in {extent}x{extent}m area...")
    
    # Create header
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0, 0, 0]
    
    # Generate random 3D points
    x = np.random.uniform(0, extent, num_points)
    y = np.random.uniform(0, extent, num_points)
    
    # Generate elevation with some structure
    # Buildings: high and flat
    # Ground: low and relatively flat
    # Vegetation: medium height with variation
    z = np.zeros(num_points)
    
    # Assign classifications
    # Create uniform probabilities based on number of labels
    num_labels = len(classification_labels)
    probabilities = [1.0 / num_labels] * num_labels
    
    classification = np.random.choice(
        classification_labels,
        size=num_points,
        p=probabilities
    )
    
    for i in range(num_points):
        if classification[i] == 6:  # Building
            z[i] = np.random.uniform(10, 50) + np.random.normal(0, 0.5)
        elif classification[i] == 2:  # Ground
            z[i] = np.random.uniform(0, 2) + np.random.normal(0, 0.2)
        elif classification[i] == 3:  # Vegetation
            z[i] = np.random.uniform(2, 15) + np.random.normal(0, 1.0)
        elif classification[i] == 9:  # Water
            z[i] = np.random.uniform(0, 1) + np.random.normal(0, 0.1)
        else:  # Unclassified
            z[i] = np.random.uniform(0, 30)
    
    # Create LAS data
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.classification = classification
    
    # Add intensity
    intensity = np.random.randint(0, 65535, num_points).astype(np.uint16)
    las.intensity = intensity
    
    # Add return information
    las.return_number = np.ones(num_points, dtype=np.uint8)
    las.number_of_returns = np.ones(num_points, dtype=np.uint8)
    
    # Add RGB colors if requested
    if add_rgb:
        print("Adding RGB colors...")
        # Buildings: gray
        # Ground: brown
        # Vegetation: green
        # Water: blue
        # Unclassified: random
        
        red = np.zeros(num_points, dtype=np.uint16)
        green = np.zeros(num_points, dtype=np.uint16)
        blue = np.zeros(num_points, dtype=np.uint16)
        
        for i in range(num_points):
            if classification[i] == 6:  # Building - gray
                val = np.random.randint(30000, 50000)
                red[i] = val
                green[i] = val
                blue[i] = val
            elif classification[i] == 2:  # Ground - brown
                red[i] = np.random.randint(25000, 40000)
                green[i] = np.random.randint(15000, 30000)
                blue[i] = np.random.randint(5000, 15000)
            elif classification[i] == 3:  # Vegetation - green
                red[i] = np.random.randint(5000, 20000)
                green[i] = np.random.randint(30000, 55000)
                blue[i] = np.random.randint(5000, 20000)
            elif classification[i] == 9:  # Water - blue
                red[i] = np.random.randint(5000, 15000)
                green[i] = np.random.randint(10000, 25000)
                blue[i] = np.random.randint(35000, 55000)
            else:  # Unclassified - random
                red[i] = np.random.randint(10000, 50000)
                green[i] = np.random.randint(10000, 50000)
                blue[i] = np.random.randint(10000, 50000)
        
        las.red = red
        las.green = green
        las.blue = blue
    
    # Add infrared if requested
    if add_infrared and hasattr(las, 'nir'):
        print("Adding infrared channel...")
        # Vegetation has high infrared reflectance
        nir = np.zeros(num_points, dtype=np.uint16)
        for i in range(num_points):
            if classification[i] == 3:  # Vegetation
                nir[i] = np.random.randint(40000, 60000)
            else:
                nir[i] = np.random.randint(10000, 30000)
        las.nir = nir
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write LAZ file
    print(f"Writing to {output_path}...")
    las.write(output_path)
    
    # Print statistics
    print(f"\n✅ Generated LAZ file:")
    print(f"   Points: {num_points:,}")
    print(f"   Extent: {extent}x{extent}m")
    print(f"   Classes: {np.unique(classification)}")
    print(f"   RGB: {'Yes' if add_rgb else 'No'}")
    print(f"   Infrared: {'Yes' if add_infrared else 'No'}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample LAZ files for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample_laz"),
        help="Output directory for LAZ files"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=3,
        help="Number of sample files to generate"
    )
    parser.add_argument(
        "--points-per-file",
        type=int,
        default=50000,
        help="Number of points per file"
    )
    parser.add_argument(
        "--add-rgb",
        action="store_true",
        default=True,
        help="Add RGB colors"
    )
    parser.add_argument(
        "--add-infrared",
        action="store_true",
        help="Add infrared channel"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sample LAZ Generator")
    print("=" * 60)
    print()
    
    # Generate multiple files with different characteristics
    configs = [
        {
            "name": "small_dense",
            "num_points": args.points_per_file,
            "extent": 50.0,
            "seed": 42
        },
        {
            "name": "medium_sparse",
            "num_points": args.points_per_file * 2,
            "extent": 100.0,
            "seed": 43
        },
        {
            "name": "large_urban",
            "num_points": args.points_per_file * 3,
            "extent": 150.0,
            "seed": 44,
            "classification_labels": [6, 2, 1]  # Mostly buildings and ground
        }
    ]
    
    for i, config in enumerate(configs[:args.num_files], 1):
        print(f"[{i}/{args.num_files}] Generating {config['name']}.laz...")
        
        output_path = args.output_dir / f"{config['name']}.laz"
        
        generate_sample_laz(
            output_path=output_path,
            num_points=config.get("num_points", args.points_per_file),
            extent=config.get("extent", 100.0),
            classification_labels=config.get("classification_labels"),
            add_rgb=args.add_rgb,
            add_infrared=args.add_infrared,
            seed=config.get("seed", 42)
        )
    
    print("=" * 60)
    print(f"✅ Generated {args.num_files} sample LAZ files")
    print(f"📁 Output directory: {args.output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
