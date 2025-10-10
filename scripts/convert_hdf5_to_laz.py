#!/usr/bin/env python3
"""
Convert HDF5 patch files back to LAZ format.

This script reads HDF5 files created by the IGN LiDAR HD processor and converts them
back to LAZ point cloud format for visualization and further processing.

Usage:
    python convert_hdf5_to_laz.py <input.h5> [output.laz]
    python convert_hdf5_to_laz.py <input_directory> [output_directory]

Author: IGN LiDAR HD Team
Date: October 10, 2025
"""

import sys
import numpy as np
import laspy
import h5py
from pathlib import Path
import argparse
from typing import Optional, List, Dict, Any


def inspect_hdf5_structure(h5_path: Path) -> Dict[str, Any]:
    """
    Inspect the structure of an HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        Dictionary with file structure information
    """
    structure = {
        'keys': [],
        'datasets': {},
        'groups': [],
        'attributes': {}
    }
    
    with h5py.File(h5_path, 'r') as f:
        # Get top-level keys
        structure['keys'] = list(f.keys())
        
        # Get attributes
        structure['attributes'] = dict(f.attrs)
        
        # Inspect datasets and groups
        def inspect_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                structure['datasets'][name] = {
                    'shape': obj.shape,
                    'dtype': obj.dtype,
                    'size': obj.size
                }
            elif isinstance(obj, h5py.Group):
                structure['groups'].append(name)
        
        f.visititems(inspect_item)
    
    return structure


def convert_hdf5_to_laz(h5_path: Path, output_path: Optional[Path] = None, 
                        verbose: bool = True) -> Path:
    """
    Convert a single HDF5 file to LAZ format.
    
    Args:
        h5_path: Path to input HDF5 file
        output_path: Path to output LAZ file (optional, auto-generated if not provided)
        verbose: Whether to print detailed information
        
    Returns:
        Path to the created LAZ file
    """
    if verbose:
        print(f"Loading HDF5 file: {h5_path}")
    
    # Open HDF5 file
    try:
        h5_file = h5py.File(h5_path, 'r')
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        raise
    
    try:
        # Print available keys for debugging
        if verbose:
            print(f"Available keys in HDF5: {list(h5_file.keys())}")
        
        # Check if this is a metadata-only file
        if 'metadata' in h5_file.keys() and len(h5_file.keys()) == 1:
            print("\n⚠️  This appears to be a metadata-only HDF5 file.")
            print("    These files only contain patch metadata without actual point cloud data.")
            
            # Display the metadata content
            try:
                metadata_group = h5_file['metadata']
                print("\nMetadata contents:")
                for key in metadata_group.attrs:
                    print(f"  {key}: {metadata_group.attrs[key]}")
            except Exception as e:
                print(f"  (Could not parse metadata: {e})")
            
            h5_file.close()
            raise ValueError(
                "\nCannot convert metadata-only HDF5 files to LAZ format.\n"
                "These files were likely created with save_metadata=True but without actual point data.\n"
                "To generate full HDF5 files with point clouds, ensure your processing pipeline includes:\n"
                "  - output.save_metadata=True\n"
                "  - Actual point cloud processing (not just metadata extraction)"
            )
        
        # Determine which coordinate field to use
        # Common field names: 'points', 'coords', 'xyz', 'coordinates'
        coords = None
        coord_keys = ['points', 'coords', 'xyz', 'coordinates', 'point_cloud']
        
        for key in coord_keys:
            if key in h5_file:
                coords = h5_file[key][:]
                break
        
        if coords is None:
            available_keys = list(h5_file.keys())
            h5_file.close()
            raise ValueError(
                f"HDF5 file must contain coordinate data.\n"
                f"Expected one of: {coord_keys}\n"
                f"Found keys: {available_keys}"
            )
        
        # Ensure coords is 2D with shape (N, 3)
        if coords.ndim == 1:
            # Might be flattened, try to reshape
            if coords.shape[0] % 3 == 0:
                coords = coords.reshape(-1, 3)
            else:
                h5_file.close()
                raise ValueError(f"Invalid coordinate data shape: {coords.shape}")
        elif coords.ndim != 2 or coords.shape[1] != 3:
            h5_file.close()
            raise ValueError(f"Coordinate data must be (N, 3), got {coords.shape}")
        
        num_points = coords.shape[0]
        if verbose:
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
        intensity_keys = ['intensity', 'intensities', 'Intensity']
        for key in intensity_keys:
            if key in h5_file:
                intensity = h5_file[key][:]
                # Normalize to uint16 range if needed
                if intensity.max() <= 1.0:
                    # Normalized [0, 1] -> [0, 65535]
                    las.intensity = (intensity * 65535).astype(np.uint16)
                else:
                    las.intensity = intensity.astype(np.uint16)
                if verbose:
                    print("Added intensity field")
                break
        else:
            las.intensity = np.zeros(num_points, dtype=np.uint16)
        
        # Set RGB if available
        rgb_keys = ['rgb', 'colors', 'RGB', 'color']
        for key in rgb_keys:
            if key in h5_file:
                rgb = h5_file[key][:]
                
                # Ensure RGB is (N, 3)
                if rgb.ndim == 1 and rgb.shape[0] == num_points * 3:
                    rgb = rgb.reshape(num_points, 3)
                elif rgb.shape != (num_points, 3):
                    print(f"Warning: RGB shape {rgb.shape} doesn't match points {num_points}, skipping")
                    continue
                
                # Normalize to uint16 range if needed
                if rgb.max() <= 1.0:
                    # Normalized [0, 1] -> [0, 65535]
                    las.red = (rgb[:, 0] * 65535).astype(np.uint16)
                    las.green = (rgb[:, 1] * 65535).astype(np.uint16)
                    las.blue = (rgb[:, 2] * 65535).astype(np.uint16)
                elif rgb.max() <= 255:
                    # [0, 255] -> [0, 65535]
                    las.red = (rgb[:, 0] * 257).astype(np.uint16)
                    las.green = (rgb[:, 1] * 257).astype(np.uint16)
                    las.blue = (rgb[:, 2] * 257).astype(np.uint16)
                else:
                    las.red = rgb[:, 0].astype(np.uint16)
                    las.green = rgb[:, 1].astype(np.uint16)
                    las.blue = rgb[:, 2].astype(np.uint16)
                if verbose:
                    print("Added RGB fields")
                break
        
        # Set classification if available
        class_keys = ['classification', 'classifications', 'labels', 'label', 'class']
        for key in class_keys:
            if key in h5_file:
                classification = h5_file[key][:]
                las.classification = classification.astype(np.uint8)
                if verbose:
                    print(f"Added classification field (from {key})")
                break
        else:
            las.classification = np.zeros(num_points, dtype=np.uint8)
        
        # Set return number (default to 1 if not available)
        return_keys = ['return_number', 'return_num', 'returns']
        for key in return_keys:
            if key in h5_file:
                las.return_number = h5_file[key][:].astype(np.uint8)
                break
        else:
            las.return_number = np.ones(num_points, dtype=np.uint8)
        
        # Set number of returns (default to 1 if not available)
        num_return_keys = ['number_of_returns', 'num_returns', 'n_returns']
        for key in num_return_keys:
            if key in h5_file:
                las.number_of_returns = h5_file[key][:].astype(np.uint8)
                break
        else:
            las.number_of_returns = np.ones(num_points, dtype=np.uint8)
        
        # Add extra dimensions for features if present
        feature_keys = ['features', 'normals', 'curvature', 'planarity', 'verticality',
                       'horizontality', 'density', 'height', 'eigenvalues']
        
        for key in feature_keys:
            if key in h5_file:
                feature_data = h5_file[key][:]
                
                # Handle different feature shapes
                if feature_data.ndim == 1:
                    # Single feature value per point
                    try:
                        las.add_extra_dim(laspy.ExtraBytesParams(
                            name=key,
                            type=np.float32,
                            description=f"Feature: {key}"
                        ))
                        setattr(las, key, feature_data.astype(np.float32))
                        if verbose:
                            print(f"Added extra dimension: {key}")
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not add dimension {key}: {e}")
                
                elif feature_data.ndim == 2:
                    # Multi-dimensional features (e.g., normals)
                    n_dims = feature_data.shape[1]
                    
                    if key == 'normals' and n_dims == 3:
                        # Special handling for normals
                        try:
                            for i, suffix in enumerate(['x', 'y', 'z']):
                                dim_name = f'normal_{suffix}'
                                las.add_extra_dim(laspy.ExtraBytesParams(
                                    name=dim_name,
                                    type=np.float32,
                                    description=f"Surface normal {suffix.upper()}"
                                ))
                                setattr(las, dim_name, feature_data[:, i].astype(np.float32))
                            if verbose:
                                print(f"Added normal components (x, y, z)")
                        except Exception as e:
                            if verbose:
                                print(f"Warning: Could not add normals: {e}")
                    else:
                        # Generic multi-dimensional features
                        for i in range(min(n_dims, 10)):  # Limit to 10 dimensions
                            dim_name = f'{key}_{i}'
                            try:
                                las.add_extra_dim(laspy.ExtraBytesParams(
                                    name=dim_name,
                                    type=np.float32,
                                    description=f"{key} dimension {i}"
                                ))
                                setattr(las, dim_name, feature_data[:, i].astype(np.float32))
                            except Exception as e:
                                if verbose:
                                    print(f"Warning: Could not add {dim_name}: {e}")
                        if verbose:
                            print(f"Added {min(n_dims, 10)} dimensions from {key}")
        
    finally:
        h5_file.close()
    
    # Determine output path
    if output_path is None:
        output_path = h5_path.with_suffix('.laz')
    
    # Write LAZ file
    if verbose:
        print(f"Writing LAZ file: {output_path}")
    las.write(str(output_path))
    
    if verbose:
        print(f"Successfully converted {h5_path.name} -> {output_path.name}")
        print(f"  Points: {num_points}")
        print(f"  Bounds: X=[{las.x.min():.2f}, {las.x.max():.2f}] "
              f"Y=[{las.y.min():.2f}, {las.y.max():.2f}] "
              f"Z=[{las.z.min():.2f}, {las.z.max():.2f}]")
    
    return output_path


def convert_directory(input_dir: Path, output_dir: Optional[Path] = None,
                     verbose: bool = True) -> list:
    """
    Convert all HDF5 files in a directory to LAZ format.
    
    Args:
        input_dir: Directory containing HDF5 files
        output_dir: Output directory (optional, uses input_dir if not provided)
        verbose: Whether to print detailed information
        
    Returns:
        List of converted LAZ file paths
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all HDF5 files
    h5_files = list(input_dir.glob("*.h5")) + list(input_dir.glob("*.hdf5"))
    
    if not h5_files:
        print(f"No HDF5 files found in {input_dir}")
        return []
    
    print(f"Found {len(h5_files)} HDF5 files to convert")
    print("="*60)
    
    converted = []
    metadata_only = []
    failed = []
    
    for h5_file in h5_files:
        try:
            output_path = output_dir / h5_file.with_suffix('.laz').name
            convert_hdf5_to_laz(h5_file, output_path, verbose=verbose)
            converted.append(output_path)
            if verbose:
                print()
        except ValueError as e:
            # Check if it's a metadata-only file
            if "metadata-only" in str(e):
                metadata_only.append(h5_file)
                if verbose:
                    print(f"⚠️  Skipping metadata-only file: {h5_file.name}")
            else:
                failed.append(h5_file)
                print(f"❌ Error converting {h5_file.name}: {e}")
            if verbose:
                print()
        except Exception as e:
            failed.append(h5_file)
            print(f"❌ Error converting {h5_file.name}: {e}")
            if verbose:
                print()
    
    # Summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"✅ Successfully converted: {len(converted)}/{len(h5_files)} files")
    if metadata_only:
        print(f"⚠️  Metadata-only files skipped: {len(metadata_only)}")
        print(f"   (These files only contain metadata without point cloud data)")
    if failed:
        print(f"❌ Failed conversions: {len(failed)}")
        for f in failed[:5]:  # Show first 5 failures
            print(f"   - {f.name}")
        if len(failed) > 5:
            print(f"   ... and {len(failed) - 5} more")
    print("="*60)
    return converted


def inspect_file(h5_path: Path):
    """
    Inspect and display HDF5 file structure.
    
    Args:
        h5_path: Path to HDF5 file
    """
    print(f"Inspecting: {h5_path}")
    print("="*60)
    
    try:
        structure = inspect_hdf5_structure(h5_path)
        
        print("\n📋 Top-level keys:")
        for key in structure['keys']:
            print(f"  - {key}")
        
        print("\n📊 Datasets:")
        for name, info in structure['datasets'].items():
            print(f"  - {name}")
            print(f"    Shape: {info['shape']}, Type: {info['dtype']}, Size: {info['size']:,}")
        
        if structure['groups']:
            print("\n📁 Groups:")
            for group in structure['groups']:
                print(f"  - {group}")
        
        if structure['attributes']:
            print("\n🏷️  Attributes:")
            for key, value in structure['attributes'].items():
                print(f"  - {key}: {value}")
        
    except Exception as e:
        print(f"Error inspecting file: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 patch files to LAZ format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python convert_hdf5_to_laz.py patch_0001.h5
  python convert_hdf5_to_laz.py patch_0001.h5 output.laz
  
  # Convert directory
  python convert_hdf5_to_laz.py patches/ output_laz/
  
  # Inspect HDF5 structure
  python convert_hdf5_to_laz.py --inspect patch_0001.h5
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input HDF5 file or directory containing HDF5 files'
    )
    
    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        default=None,
        help='Output LAZ file or directory (optional)'
    )
    
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect HDF5 file structure without conversion'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    verbose = not args.quiet
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Inspect mode
    if args.inspect:
        if input_path.is_file():
            inspect_file(input_path)
        else:
            print(f"Error: --inspect requires a file, not a directory")
            sys.exit(1)
        return
    
    # Process based on input type
    try:
        if input_path.is_file():
            if input_path.suffix not in ['.h5', '.hdf5']:
                print(f"Error: Input file must have .h5 or .hdf5 extension")
                sys.exit(1)
            convert_hdf5_to_laz(input_path, output_path, verbose=verbose)
        elif input_path.is_dir():
            convert_directory(input_path, output_path, verbose=verbose)
        else:
            print(f"Error: Invalid input path: {input_path}")
            sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
