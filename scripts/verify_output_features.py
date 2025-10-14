#!/usr/bin/env python3
"""
Verify output files for NIR and NDVI features.
Checks both NPZ and LAZ files for completeness.
"""

import sys
from pathlib import Path
import numpy as np

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    print("⚠️  laspy not available - will only check NPZ files")

def check_npz_file(npz_path: Path) -> dict:
    """Check NPZ file for required features."""
    results = {
        'file': npz_path.name,
        'format': 'NPZ',
        'has_nir': False,
        'has_ndvi': False,
        'has_rgb': False,
        'issues': []
    }
    
    try:
        data = np.load(npz_path)
        
        # Check for RGB
        if 'rgb' in data:
            results['has_rgb'] = True
        else:
            results['issues'].append('RGB missing')
        
        # Check for NIR
        if 'nir' in data:
            nir = data['nir']
            if nir is not None and isinstance(nir, np.ndarray) and nir.size > 0:
                results['has_nir'] = True
                results['nir_range'] = f"[{nir.min():.3f}, {nir.max():.3f}]"
            else:
                results['issues'].append('NIR is None or empty')
        else:
            results['issues'].append('NIR missing')
        
        # Check for NDVI
        if 'ndvi' in data:
            ndvi = data['ndvi']
            if ndvi is not None and isinstance(ndvi, np.ndarray) and ndvi.size > 0:
                results['has_ndvi'] = True
                results['ndvi_range'] = f"[{ndvi.min():.3f}, {ndvi.max():.3f}]"
            else:
                results['issues'].append('NDVI is None or empty')
        else:
            results['issues'].append('NDVI missing')
        
        # Check for geometric feature artifacts
        for feat_name in ['planarity', 'linearity', 'roof_score']:
            if feat_name in data:
                feat = data[feat_name]
                if isinstance(feat, np.ndarray) and feat.size > 0:
                    nan_count = np.isnan(feat).sum()
                    inf_count = np.isinf(feat).sum()
                    if nan_count > 0:
                        results['issues'].append(f'{feat_name} has {nan_count} NaNs')
                    if inf_count > 0:
                        results['issues'].append(f'{feat_name} has {inf_count} Infs')
        
        results['total_features'] = len(data.keys())
        
    except Exception as e:
        results['issues'].append(f'Error reading NPZ: {e}')
    
    return results


def check_laz_file(laz_path: Path) -> dict:
    """Check LAZ file for required features."""
    results = {
        'file': laz_path.name,
        'format': 'LAZ',
        'has_nir': False,
        'has_ndvi': False,
        'has_rgb': False,
        'issues': []
    }
    
    if not LASPY_AVAILABLE:
        results['issues'].append('laspy not available')
        return results
    
    try:
        las = laspy.read(laz_path)
        extra_dims = list(las.point_format.extra_dimension_names)
        
        # Check for RGB (standard fields)
        if 'red' in las.point_format.dimension_names:
            results['has_rgb'] = True
        else:
            results['issues'].append('RGB missing')
        
        # Check for NIR (can be in standard fields for format 8, or extra dimensions)
        if 'nir' in las.point_format.dimension_names:
            # Point format 8 has NIR as standard field
            results['has_nir'] = True
            nir = las.nir
            # NIR in format 8 is uint16, normalize for display
            results['nir_range'] = f"[{nir.min()/65535:.3f}, {nir.max()/65535:.3f}]"
            results['nir_location'] = 'standard field'
        elif 'nir' in extra_dims:
            # Other formats have NIR as extra dimension
            results['has_nir'] = True
            nir = las['nir']
            results['nir_range'] = f"[{nir.min():.3f}, {nir.max():.3f}]"
            results['nir_location'] = 'extra dimension'
        else:
            results['issues'].append('NIR missing')
        
        # Check for NDVI
        if 'ndvi' in extra_dims:
            results['has_ndvi'] = True
            ndvi = las['ndvi']
            results['ndvi_range'] = f"[{ndvi.min():.3f}, {ndvi.max():.3f}]"
        else:
            results['issues'].append('NDVI missing')
        
        # Check for geometric feature artifacts
        for feat_name in ['planarity', 'linearity', 'roof_score']:
            if feat_name in extra_dims:
                feat = las[feat_name]
                nan_count = np.isnan(feat).sum()
                inf_count = np.isinf(feat).sum()
                if nan_count > 0:
                    results['issues'].append(f'{feat_name} has {nan_count} NaNs')
                if inf_count > 0:
                    results['issues'].append(f'{feat_name} has {inf_count} Infs')
        
        results['total_extra_dims'] = len(extra_dims)
        
    except Exception as e:
        results['issues'].append(f'Error reading LAZ: {e}')
    
    return results


def print_results(results: dict):
    """Print verification results."""
    status = '✅' if len(results['issues']) == 0 else '❌'
    print(f"\n{status} {results['file']} ({results['format']})")
    
    # Feature status
    rgb_status = '✅' if results['has_rgb'] else '❌'
    nir_status = '✅' if results['has_nir'] else '❌'
    ndvi_status = '✅' if results['has_ndvi'] else '❌'
    
    print(f"   RGB:  {rgb_status}")
    print(f"   NIR:  {nir_status} {results.get('nir_range', '')}")
    print(f"   NDVI: {ndvi_status} {results.get('ndvi_range', '')}")
    
    if 'total_features' in results:
        print(f"   Total features: {results['total_features']}")
    if 'total_extra_dims' in results:
        print(f"   Total extra dims: {results['total_extra_dims']}")
    
    # Issues
    if results['issues']:
        print(f"   Issues:")
        for issue in results['issues']:
            print(f"     - {issue}")


def main():
    """Main verification function."""
    if len(sys.argv) < 2:
        print("Usage: python verify_output_features.py <output_directory>")
        print("\nExample:")
        print("  python verify_output_features.py /mnt/c/Users/Simon/ign/versailles/output")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"❌ Directory not found: {output_dir}")
        sys.exit(1)
    
    print("="*80)
    print("OUTPUT FEATURE VERIFICATION")
    print("="*80)
    print(f"Checking: {output_dir}")
    
    # Find files
    npz_files = sorted(list(output_dir.glob("*.npz")))[-5:]  # Last 5 NPZ files
    laz_files = sorted(list(output_dir.glob("*.laz")))[-5:]  # Last 5 LAZ files
    
    if not npz_files and not laz_files:
        print("\n❌ No output files found!")
        sys.exit(1)
    
    # Check NPZ files
    if npz_files:
        print(f"\n📦 Checking {len(npz_files)} most recent NPZ files...")
        npz_results = []
        for npz_file in npz_files:
            results = check_npz_file(npz_file)
            npz_results.append(results)
            print_results(results)
    
    # Check LAZ files
    if laz_files and LASPY_AVAILABLE:
        print(f"\n📦 Checking {len(laz_files)} most recent LAZ files...")
        laz_results = []
        for laz_file in laz_files:
            results = check_laz_file(laz_file)
            laz_results.append(results)
            print_results(results)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if npz_files:
        npz_ok = sum(1 for r in npz_results if len(r['issues']) == 0)
        print(f"NPZ: {npz_ok}/{len(npz_results)} files OK")
        
        # Check if NDVI is missing in all files
        ndvi_missing_count = sum(1 for r in npz_results if not r['has_ndvi'])
        if ndvi_missing_count == len(npz_results):
            print("\n⚠️  NDVI is missing in ALL NPZ files!")
            print("   This suggests the config fix hasn't been applied yet.")
            print("   Run: pip install -e . && ign-lidar-hd process --config-file <your_config>")
        
    if laz_files and LASPY_AVAILABLE:
        laz_ok = sum(1 for r in laz_results if len(r['issues']) == 0)
        print(f"LAZ: {laz_ok}/{len(laz_results)} files OK")
        
        # Check if NDVI is missing in all files
        ndvi_missing_count = sum(1 for r in laz_results if not r['has_ndvi'])
        if ndvi_missing_count == len(laz_results):
            print("\n⚠️  NDVI is missing in ALL LAZ files!")
            print("   This suggests the config fix hasn't been applied yet.")
            print("   Run: pip install -e . && ign-lidar-hd process --config-file <your_config>")


if __name__ == '__main__':
    main()
