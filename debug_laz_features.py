#!/usr/bin/env python3
"""
Debug script to check LAZ files for features.
Helps diagnose missing features in enriched LAZ files.
"""

import logging
import sys
from pathlib import Path
import laspy
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def check_laz_features(laz_path: Path) -> dict:
    """
    Check a LAZ file for features and report findings.
    
    Args:
        laz_path: Path to LAZ file
        
    Returns:
        dict with diagnostic information
    """
    logger.info(f"Checking LAZ file: {laz_path.name}")
    logger.info(f"  Full path: {laz_path}")
    
    try:
        las = laspy.read(str(laz_path))
    except Exception as e:
        logger.error(f"  ❌ Failed to read LAZ file: {e}")
        return {"error": str(e)}
    
    info = {
        "file": laz_path.name,
        "num_points": len(las.points),
        "point_format": las.header.point_format.id,
        "version": f"{las.header.version.major}.{las.header.version.minor}",
        "has_rgb": las.header.point_format.id in [2, 3, 5, 7, 8, 10],
        "standard_dims": [],
        "extra_dims": {},
        "issues": []
    }
    
    # Check standard dimensions
    logger.info(f"  📊 Point count: {info['num_points']:,}")
    logger.info(f"  📝 Point format: {info['point_format']} (LAS {info['version']})")
    
    standard_dims = ['x', 'y', 'z', 'intensity', 'classification', 'return_number']
    logger.info(f"\n  ✅ Standard dimensions:")
    for dim in standard_dims:
        if dim in las.point_format.dimension_names:
            info['standard_dims'].append(dim)
            logger.info(f"    - {dim}: ✓")
    
    # Check RGB
    if info['has_rgb']:
        logger.info(f"    - RGB: ✓ (format {info['point_format']} supports RGB)")
        if 'red' in las.point_format.dimension_names:
            red = las.red
            green = las.green
            blue = las.blue
            logger.info(f"      Red:   min={np.min(red)}, max={np.max(red)}")
            logger.info(f"      Green: min={np.min(green)}, max={np.max(green)}")
            logger.info(f"      Blue:  min={np.min(blue)}, max={np.max(blue)}")
    else:
        logger.info(f"    - RGB: ✗ (format {info['point_format']} does not support RGB)")
    
    # Check extra dimensions (FEATURES)
    extra_dims = las.point_format.extra_dimension_names
    logger.info(f"\n  🎨 Extra dimensions (computed features):")
    
    if extra_dims:
        logger.info(f"    Found {len(extra_dims)} extra dimensions:")
        for dim in extra_dims:
            try:
                values = getattr(las, dim)
                info['extra_dims'][dim] = {
                    "dtype": str(values.dtype),
                    "shape": values.shape,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
                logger.info(f"    - {dim:20s}: {values.dtype} {values.shape}")
                logger.info(f"      └─ range=[{np.min(values):.4f}, {np.max(values):.4f}], "
                           f"mean={np.mean(values):.4f}, std={np.std(values):.4f}")
            except Exception as e:
                info['issues'].append(f"Failed to read {dim}: {e}")
                logger.warning(f"    - {dim}: ⚠️  Failed to read: {e}")
    else:
        logger.warning(f"    ⚠️  NO EXTRA DIMENSIONS FOUND!")
        logger.warning(f"    This file has no computed features (normals, curvature, etc.)")
        info['issues'].append("No extra dimensions found - features missing!")
    
    # Expected features for enriched LAZ
    expected_features = [
        'normal_x', 'normal_y', 'normal_z',
        'curvature', 'height',
        'planarity', 'linearity', 'sphericity', 'verticality'
    ]
    
    logger.info(f"\n  🔍 Feature check:")
    missing_features = []
    for feat in expected_features:
        if feat in extra_dims:
            logger.info(f"    - {feat:20s}: ✓")
        else:
            logger.info(f"    - {feat:20s}: ✗ MISSING")
            missing_features.append(feat)
    
    if missing_features:
        info['issues'].append(f"Missing features: {', '.join(missing_features)}")
    
    # Summary
    logger.info(f"\n  📋 Summary:")
    logger.info(f"    Standard dimensions: {len(info['standard_dims'])}")
    logger.info(f"    Extra dimensions:    {len(extra_dims)}")
    logger.info(f"    Has RGB:             {info['has_rgb']}")
    logger.info(f"    Missing features:    {len(missing_features)}")
    
    if info['issues']:
        logger.warning(f"\n  ⚠️  Issues found:")
        for issue in info['issues']:
            logger.warning(f"    - {issue}")
    else:
        logger.info(f"\n  ✅ All expected features present!")
    
    return info


def main():
    """Main function to check LAZ files."""
    if len(sys.argv) < 2:
        print("Usage: python debug_laz_features.py <laz_file_or_directory>")
        print("\nExamples:")
        print("  python debug_laz_features.py output/enriched/tile_1234_5678_enriched.laz")
        print("  python debug_laz_features.py output/enriched/")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)
    
    # Collect LAZ files
    if path.is_file():
        laz_files = [path]
    else:
        laz_files = list(path.rglob("*.laz"))
        if not laz_files:
            logger.error(f"No LAZ files found in: {path}")
            sys.exit(1)
    
    logger.info(f"="*70)
    logger.info(f"LAZ Feature Diagnostic Tool")
    logger.info(f"="*70)
    logger.info(f"Found {len(laz_files)} LAZ file(s) to check\n")
    
    # Check each file
    results = []
    for i, laz_file in enumerate(laz_files, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"File {i}/{len(laz_files)}")
        logger.info(f"{'='*70}")
        result = check_laz_features(laz_file)
        results.append(result)
    
    # Overall summary
    logger.info(f"\n{'='*70}")
    logger.info(f"OVERALL SUMMARY")
    logger.info(f"{'='*70}")
    
    files_with_features = sum(1 for r in results if len(r.get('extra_dims', {})) > 0)
    files_with_issues = sum(1 for r in results if len(r.get('issues', [])) > 0)
    
    logger.info(f"Total files checked:       {len(results)}")
    logger.info(f"Files with features:       {files_with_features}")
    logger.info(f"Files WITHOUT features:    {len(results) - files_with_features}")
    logger.info(f"Files with issues:         {files_with_issues}")
    
    if files_with_issues > 0:
        logger.warning(f"\n⚠️  {files_with_issues} file(s) have issues!")
        logger.warning(f"Review the detailed output above for more information.")
    else:
        logger.info(f"\n✅ All files appear to be correctly formatted!")


if __name__ == "__main__":
    main()
