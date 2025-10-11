#!/usr/bin/env python3
"""
Convert NPZ patch files back to LAZ format with restored original coordinates.

This script reads NPZ files and restores the original LAMB93 coordinates
from the metadata, making the LAZ files readable in CloudCompare and other tools.

Usage:
    python convert_npz_to_laz_with_coords.py <input_directory> [output_directory]

Author: IGN LiDAR HD Team
Date: October 10, 2025
"""

import sys
import numpy as np
import laspy
from pathlib import Path
from typing import Optional


def convert_npz_to_laz_with_coords(npz_path: Path, output_path: Optional[Path] = None, overwrite: bool = False) -> Path:
    """
    Convert a single NPZ file to LAZ format with restored coordinates.
    
    Args:
        npz_path: Path to input NPZ file
        output_path: Path to output LAZ file (optional, auto-generated if not provided)
        overwrite: Whether to overwrite existing LAZ files
        
    Returns:
        Path to the created LAZ file
    """
    print(f"Processing: {npz_path.name}")
    
    # Determine output path
    if output_path is None:
        output_path = npz_path.with_suffix('.laz')
    
    # Check if output exists and skip if not overwriting
    if output_path.exists() and not overwrite:
        print(f"  ⚠️  Skipping (file exists): {output_path.name}")
        return output_path
    
    # Load NPZ data
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  ❌ Error loading NPZ file: {e}")
        raise
    
    # Get metadata
    if 'metadata' not in data:
        print(f"  ⚠️  Warning: No metadata found, using normalized coordinates")
        metadata = {}
    else:
        metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
    
    # Get coordinates
    if 'points' in data:
        coords = data['points'].copy()
    elif 'coords' in data:
        coords = data['coords'].copy()
    else:
        raise ValueError("NPZ file must contain either 'points' or 'coords' field")
    
    num_points = coords.shape[0]
    
    # Extract tile coordinates from filename
    # Filename format: LHD_FXX_0649_6863_PTS_O_LAMB93_IGN69_hybrid_patch_0000_aug_0.npz
    filename_parts = npz_path.stem.split('_')
    try:
        tile_x = int(filename_parts[2])  # e.g., 0649
        tile_y = int(filename_parts[3])  # e.g., 6863
        # LAMB93 tiles are 1km x 1km, convert to meters
        # Tile center is at (tile_x * 1000 + 500, tile_y * 1000 + 500)
        tile_center_x = tile_x * 1000 + 500
        tile_center_y = tile_y * 1000 + 500
        has_tile_coords = True
    except (IndexError, ValueError):
        has_tile_coords = False
    
    # Restore original coordinates
    if 'centroid' in metadata and has_tile_coords:
        # Metadata centroid is relative to tile center
        centroid = np.array(metadata['centroid'])
        # Add both tile center and local centroid
        coords[:, 0] = coords[:, 0] + centroid[0] + tile_center_x
        coords[:, 1] = coords[:, 1] + centroid[1] + tile_center_y
        coords[:, 2] = coords[:, 2] + centroid[2]
        print(f"  ✓ Restored LAMB93 coordinates (Tile {tile_x}_{tile_y} + centroid)")
    elif has_tile_coords:
        # No centroid, but we have tile coords - use tile center as offset
        coords[:, 0] = coords[:, 0] + tile_center_x
        coords[:, 1] = coords[:, 1] + tile_center_y
        print(f"  ✓ Applied tile offset (Tile {tile_x}_{tile_y}, no centroid)")
    else:
        print(f"  ⚠️  Cannot extract tile coordinates from filename, using normalized coordinates")
    
    # Create LAZ header with proper coordinate system
    # Point format 3 includes RGB
    header = laspy.LasHeader(point_format=3, version="1.4")
    
    # Set appropriate scales for LAMB93 coordinates (1mm precision)
    header.scales = [0.001, 0.001, 0.001]
    
    # Set offsets to min coordinates for better precision
    header.offsets = [
        np.floor(np.min(coords[:, 0])),
        np.floor(np.min(coords[:, 1])),
        np.floor(np.min(coords[:, 2]))
    ]
    
    # Create LasData object
    las = laspy.LasData(header)
    
    # Set XYZ coordinates (now in original LAMB93)
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]
    
    # Set intensity if available
    if 'intensity' in data:
        intensity = data['intensity']
        if intensity.max() <= 1.0:
            las.intensity = (intensity * 65535).astype(np.uint16)
        else:
            las.intensity = intensity.astype(np.uint16)
    else:
        las.intensity = np.zeros(num_points, dtype=np.uint16)
    
    # Set RGB if available
    if 'rgb' in data:
        rgb = data['rgb']
        if rgb.max() <= 1.0:
            # Normalized [0, 1] -> [0, 65535]
            las.red = (rgb[:, 0] * 65535).astype(np.uint16)
            las.green = (rgb[:, 1] * 65535).astype(np.uint16)
            las.blue = (rgb[:, 2] * 65535).astype(np.uint16)
        else:
            las.red = rgb[:, 0].astype(np.uint16)
            las.green = rgb[:, 1].astype(np.uint16)
            las.blue = rgb[:, 2].astype(np.uint16)
    
    # Set classification if available
    if 'classification' in data:
        las.classification = data['classification'].astype(np.uint8)
    elif 'labels' in data:
        las.classification = data['labels'].astype(np.uint8)
    else:
        las.classification = np.zeros(num_points, dtype=np.uint8)
    
    # Set return number (default to 1)
    if 'return_number' in data:
        las.return_number = data['return_number'].astype(np.uint8)
    else:
        las.return_number = np.ones(num_points, dtype=np.uint8)
    
    # Set number of returns (default to 1)
    if 'number_of_returns' in data:
        las.number_of_returns = data['number_of_returns'].astype(np.uint8)
    else:
        las.number_of_returns = np.ones(num_points, dtype=np.uint8)
    
    # Add extra dimensions for computed features
    extra_dims_added = []
    
    # Normals (3 dimensions)
    if 'normals' in data:
        normals = data['normals']
        if normals.shape[0] == num_points and normals.shape[1] == 3:
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_x", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_y", type=np.float32))
            las.add_extra_dim(laspy.ExtraBytesParams(name="normal_z", type=np.float32))
            las.normal_x = normals[:, 0].astype(np.float32)
            las.normal_y = normals[:, 1].astype(np.float32)
            las.normal_z = normals[:, 2].astype(np.float32)
            extra_dims_added.extend(['normal_x', 'normal_y', 'normal_z'])
    
    # Geometric features (scalar)
    geometric_features = {
        'curvature': 'curvature',
        'planarity': 'planarity',
        'linearity': 'linearity',
        'sphericity': 'sphericity',
        'verticality': 'verticality',
        'height': 'height'
    }
    
    for npz_key, las_name in geometric_features.items():
        if npz_key in data:
            feature_data = data[npz_key]
            if len(feature_data) == num_points:
                las.add_extra_dim(laspy.ExtraBytesParams(name=las_name, type=np.float32))
                setattr(las, las_name, feature_data.astype(np.float32))
                extra_dims_added.append(las_name)
    
    # Radiometric features
    radiometric_features = {
        'nir': 'nir',
        'ndvi': 'ndvi',
        'intensity': 'intensity_norm'  # Normalized intensity if different from standard
    }
    
    for npz_key, las_name in radiometric_features.items():
        if npz_key in data:
            feature_data = data[npz_key]
            if len(feature_data) == num_points:
                las.add_extra_dim(laspy.ExtraBytesParams(name=las_name, type=np.float32))
                setattr(las, las_name, feature_data.astype(np.float32))
                extra_dims_added.append(las_name)
    
    # Write LAZ file
    las.write(str(output_path))
    
    if extra_dims_added:
        print(f"    ✓ Added {len(extra_dims_added)} extra dimensions: {', '.join(extra_dims_added[:5])}{', ...' if len(extra_dims_added) > 5 else ''}")
    
    print(f"  ✓ Created: {output_path.name}")
    print(f"    Points: {num_points:,}")
    print(f"    Bounds: X=[{las.x.min():.2f}, {las.x.max():.2f}] "
          f"Y=[{las.y.min():.2f}, {las.y.max():.2f}] "
          f"Z=[{las.z.min():.2f}, {las.z.max():.2f}]")
    
    return output_path


def convert_directory(input_dir: Path, output_dir: Optional[Path] = None, overwrite: bool = False) -> list:
    """
    Convert all NPZ files in a directory to LAZ format.
    
    Args:
        input_dir: Directory containing NPZ files
        output_dir: Output directory (optional, uses input_dir if not provided)
        overwrite: Whether to overwrite existing LAZ files
        
    Returns:
        List of converted LAZ file paths
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all NPZ files
    npz_files = sorted(list(input_dir.glob("*.npz")))
    
    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return []
    
    print(f"Found {len(npz_files)} NPZ files to convert")
    print(f"Overwrite mode: {'ON' if overwrite else 'OFF'}")
    print()
    
    converted = []
    skipped = []
    failed = []
    
    for i, npz_file in enumerate(npz_files, 1):
        print(f"[{i}/{len(npz_files)}] ", end="")
        try:
            output_path = output_dir / npz_file.with_suffix('.laz').name
            
            # Check if should skip
            if output_path.exists() and not overwrite:
                print(f"Skipping: {npz_file.name} (already exists)")
                skipped.append(npz_file)
                continue
            
            convert_npz_to_laz_with_coords(npz_file, output_path, overwrite)
            converted.append(output_path)
        except Exception as e:
            failed.append(npz_file)
            print(f"  ❌ Error: {e}")
        print()
    
    # Summary
    print("\n" + "="*70)
    print("CONVERSION SUMMARY")
    print("="*70)
    print(f"✅ Successfully converted: {len(converted)}/{len(npz_files)} files")
    if skipped:
        print(f"⏭️  Skipped (already exist): {len(skipped)}")
    if failed:
        print(f"❌ Failed conversions: {len(failed)}")
        for f in failed:
            print(f"   - {f.name}")
    print("="*70)
    
    return converted


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python convert_npz_to_laz_with_coords.py <input_directory> [output_directory] [--overwrite]")
        print("\nThis script converts NPZ files to LAZ with restored LAMB93 coordinates.")
        print("By default, existing LAZ files are skipped. Use --overwrite to regenerate all files.")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    overwrite = '--overwrite' in sys.argv
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f"Error: Input must be a directory")
        sys.exit(1)
    
    # Process directory
    try:
        convert_directory(input_path, output_path, overwrite)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
