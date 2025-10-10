#!/usr/bin/env python3
"""
Convert NPZ patch files back to LAZ format.

This script reads NPZ files created by the IGN LiDAR HD processor and converts them
back to LAZ point cloud format for visualization and further processing.

Usage:
    python convert_npz_to_laz.py <input.npz> [output.laz]
    python convert_npz_to_laz.py <input_directory> [output_directory]

Author: IGN LiDAR HD Team
Date: October 10, 2025
"""

import sys
import numpy as np
import laspy
from pathlib import Path
import argparse
from typing import Optional


def convert_npz_to_laz(npz_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convert a single NPZ file to LAZ format.
    
    Args:
        npz_path: Path to input NPZ file
        output_path: Path to output LAZ file (optional, auto-generated if not provided)
        
    Returns:
        Path to the created LAZ file
    """
    print(f"Loading NPZ file: {npz_path}")
    
    # Load NPZ data
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        raise
    
    # Print available keys for debugging
    print(f"Available keys in NPZ: {list(data.keys())}")
    
    # Check if this is a metadata-only file
    if list(data.keys()) == ['metadata'] or (len(data.keys()) == 1 and 'metadata' in data):
        print("\n⚠️  This appears to be a metadata-only NPZ file.")
        print("    These files only contain patch metadata without actual point cloud data.")
        
        # Display the metadata content
        try:
            metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
            print("\nMetadata contents:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  (Could not parse metadata: {e})")
        
        raise ValueError(
            "\nCannot convert metadata-only NPZ files to LAZ format.\n"
            "These files were likely created with save_metadata=True but without actual point data.\n"
            "To generate full NPZ files with point clouds, ensure your processing pipeline includes:\n"
            "  - output.save_metadata=True\n"
            "  - Actual point cloud processing (not just metadata extraction)"
        )
    
    # Determine which coordinate field to use
    if 'points' in data:
        coords = data['points']
    elif 'coords' in data:
        coords = data['coords']
    else:
        raise ValueError("NPZ file must contain either 'points' or 'coords' field")
    
    num_points = coords.shape[0]
    print(f"Number of points: {num_points}")
    
    # Create LAZ header
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = [0.001, 0.001, 0.001]  # 1mm precision
    header.offsets = [
        np.min(coords[:, 0]),
        np.min(coords[:, 1]),
        np.min(coords[:, 2])
    ]
    
    # Create LasData object
    las = laspy.LasData(header)
    
    # Set XYZ coordinates
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]
    
    # Set intensity if available
    if 'intensity' in data:
        intensity = data['intensity']
        # Normalize to uint16 range if needed
        if intensity.max() <= 1.0:
            # Normalized [0, 1] -> [0, 65535]
            las.intensity = (intensity * 65535).astype(np.uint16)
        else:
            las.intensity = intensity.astype(np.uint16)
        print("Added intensity field")
    else:
        las.intensity = np.zeros(num_points, dtype=np.uint16)
    
    # Set RGB if available
    if 'rgb' in data:
        rgb = data['rgb']
        # Normalize to uint16 range if needed
        if rgb.max() <= 1.0:
            # Normalized [0, 1] -> [0, 65535]
            las.red = (rgb[:, 0] * 65535).astype(np.uint16)
            las.green = (rgb[:, 1] * 65535).astype(np.uint16)
            las.blue = (rgb[:, 2] * 65535).astype(np.uint16)
        else:
            las.red = rgb[:, 0].astype(np.uint16)
            las.green = rgb[:, 1].astype(np.uint16)
            las.blue = rgb[:, 2].astype(np.uint16)
        print("Added RGB fields")
    
    # Set classification if available
    if 'classification' in data:
        las.classification = data['classification'].astype(np.uint8)
        print("Added classification field")
    elif 'labels' in data:
        # Use labels as classification
        las.classification = data['labels'].astype(np.uint8)
        print("Added classification field (from labels)")
    else:
        las.classification = np.zeros(num_points, dtype=np.uint8)
    
    # Set return number (default to 1 if not available)
    if 'return_number' in data:
        las.return_number = data['return_number'].astype(np.uint8)
    else:
        las.return_number = np.ones(num_points, dtype=np.uint8)
    
    # Set number of returns (default to 1 if not available)
    if 'number_of_returns' in data:
        las.number_of_returns = data['number_of_returns'].astype(np.uint8)
    else:
        las.number_of_returns = np.ones(num_points, dtype=np.uint8)
    
    # Determine output path
    if output_path is None:
        output_path = npz_path.with_suffix('.laz')
    
    # Write LAZ file
    print(f"Writing LAZ file: {output_path}")
    las.write(str(output_path))
    print(f"Successfully converted {npz_path.name} -> {output_path.name}")
    print(f"  Points: {num_points}")
    print(f"  Bounds: X=[{las.x.min():.2f}, {las.x.max():.2f}] "
          f"Y=[{las.y.min():.2f}, {las.y.max():.2f}] "
          f"Z=[{las.z.min():.2f}, {las.z.max():.2f}]")
    
    return output_path


def convert_directory(input_dir: Path, output_dir: Optional[Path] = None) -> list:
    """
    Convert all NPZ files in a directory to LAZ format.
    
    Args:
        input_dir: Directory containing NPZ files
        output_dir: Output directory (optional, uses input_dir if not provided)
        
    Returns:
        List of converted LAZ file paths
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NPZ files
    npz_files = list(input_dir.glob("*.npz"))
    
    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return []
    
    print(f"Found {len(npz_files)} NPZ files to convert")
    
    converted = []
    metadata_only = []
    failed = []
    
    for npz_file in npz_files:
        try:
            output_path = output_dir / npz_file.with_suffix('.laz').name
            convert_npz_to_laz(npz_file, output_path)
            converted.append(output_path)
            print()
        except ValueError as e:
            # Check if it's a metadata-only file
            if "metadata-only" in str(e):
                metadata_only.append(npz_file)
                print(f"⚠️  Skipping metadata-only file: {npz_file.name}")
            else:
                failed.append(npz_file)
                print(f"❌ Error converting {npz_file.name}: {e}")
            print()
        except Exception as e:
            failed.append(npz_file)
            print(f"❌ Error converting {npz_file.name}: {e}")
            print()
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"✅ Successfully converted: {len(converted)}/{len(npz_files)} files")
    if metadata_only:
        print(f"⚠️  Metadata-only files skipped: {len(metadata_only)}")
        print(f"   (These files only contain metadata without point cloud data)")
    if failed:
        print(f"❌ Failed conversions: {len(failed)}")
    print("="*60)
    return converted


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert NPZ patch files to LAZ format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_npz_to_laz.py patch_0001.npz
  python convert_npz_to_laz.py patch_0001.npz output.laz
  
  # Convert directory
  python convert_npz_to_laz.py patches/ output_laz/
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input NPZ file or directory containing NPZ files'
    )
    
    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        default=None,
        help='Output LAZ file or directory (optional)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Process based on input type
    try:
        if input_path.is_file():
            if input_path.suffix != '.npz':
                print(f"Error: Input file must have .npz extension")
                sys.exit(1)
            convert_npz_to_laz(input_path, output_path)
        elif input_path.is_dir():
            convert_directory(input_path, output_path)
        else:
            print(f"Error: Invalid input path: {input_path}")
            sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
