#!/usr/bin/env python3
"""
Script to verify NPZ files contain proper point cloud data after fix.

This script checks:
1. What keys are present in the NPZ file
2. The size and shape of each array
3. Whether actual point cloud data is present (not just metadata)
"""

import numpy as np
import sys
from pathlib import Path


def check_npz_file(npz_path):
    """Check contents of an NPZ file."""
    print(f"\n{'='*70}")
    print(f"Checking: {npz_path.name}")
    print(f"{'='*70}")
    
    # Get file size
    file_size = npz_path.stat().st_size
    print(f"File size: {file_size} bytes ({file_size/1024:.2f} KB)")
    
    # Load NPZ file
    npz = np.load(npz_path, allow_pickle=True)
    
    print(f"\nKeys in file: {list(npz.keys())}")
    
    total_data_bytes = 0
    has_points = False
    has_features = False
    has_labels = False
    
    for key in npz.keys():
        data = npz[key]
        print(f"\n  {key}:")
        
        if data.dtype == object:
            # Likely metadata
            print(f"    Type: object (metadata)")
            print(f"    Shape: {data.shape}")
            if data.size == 1:
                meta = data.item()
                if isinstance(meta, dict):
                    print(f"    Metadata keys: {list(meta.keys())}")
                    if 'num_points' in meta:
                        print(f"    num_points: {meta['num_points']}")
        else:
            # Actual data arrays
            print(f"    Shape: {data.shape}")
            print(f"    Dtype: {data.dtype}")
            print(f"    Size: {data.size} elements")
            print(f"    Memory: {data.nbytes} bytes ({data.nbytes/1024:.2f} KB)")
            total_data_bytes += data.nbytes
            
            # Check data range
            if data.size > 0 and np.issubdtype(data.dtype, np.number):
                print(f"    Range: [{data.min():.4f}, {data.max():.4f}]")
                print(f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            
            # Identify important arrays
            if 'point' in key.lower():
                has_points = True
            if 'feature' in key.lower():
                has_features = True
            if 'label' in key.lower():
                has_labels = True
    
    npz.close()
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total data: {total_data_bytes} bytes ({total_data_bytes/1024:.2f} KB)")
    print(f"  Has points: {'✓' if has_points else '✗'}")
    print(f"  Has features: {'✓' if has_features else '✗'}")
    print(f"  Has labels: {'✓' if has_labels else '✗'}")
    
    # Verdict
    if total_data_bytes < 1000:
        print(f"\n  ⚠️  WARNING: File appears to contain only metadata (< 1KB of data)")
        print(f"      This indicates the bug is still present!")
        return False
    else:
        print(f"\n  ✓ File contains actual point cloud data")
        return True


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # Check specific file
        npz_path = Path(sys.argv[1])
        if not npz_path.exists():
            print(f"Error: File not found: {npz_path}")
            sys.exit(1)
        check_npz_file(npz_path)
    else:
        # Check all NPZ files in the default directory
        default_dir = Path("/mnt/c/Users/Simon/ign/patch_1st_training/urban_dense")
        if not default_dir.exists():
            print(f"Error: Directory not found: {default_dir}")
            print(f"Usage: {sys.argv[0]} <path_to_npz_file>")
            sys.exit(1)
        
        npz_files = sorted(default_dir.glob("*.npz"))
        if not npz_files:
            print(f"No NPZ files found in {default_dir}")
            sys.exit(1)
        
        print(f"Found {len(npz_files)} NPZ files in {default_dir}")
        print(f"Checking first 3 files as samples...")
        
        results = []
        for npz_file in npz_files[:3]:
            result = check_npz_file(npz_file)
            results.append((npz_file.name, result))
        
        print(f"\n{'='*70}")
        print(f"Overall Summary:")
        print(f"{'='*70}")
        for filename, ok in results:
            status = "✓ OK" if ok else "✗ BROKEN"
            print(f"  {status}: {filename}")


if __name__ == "__main__":
    main()
