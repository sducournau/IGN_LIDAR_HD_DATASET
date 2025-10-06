#!/usr/bin/env python3
"""
Feature Verification Module

This module provides functionality to verify geometric features, RGB, and NIR
values in enriched LAZ files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import laspy

logger = logging.getLogger(__name__)


class FeatureVerifier:
    """Verifies features in LAZ files."""
    
    def __init__(self, verbose: bool = True, show_samples: bool = False):
        """
        Initialize the feature verifier.
        
        Args:
            verbose: If True, print detailed information
            show_samples: If True, display sample points
        """
        self.verbose = verbose
        self.show_samples = show_samples
    
    def verify_file(self, laz_path: Path) -> Dict:
        """
        Verify all features in a LAZ file.
        
        Args:
            laz_path: Path to the LAZ file
            
        Returns:
            Dictionary with verification results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Analyzing: {laz_path.name}")
            print('='*80)
        
        las = laspy.read(str(laz_path))
        n_points = len(las.points)
        
        if self.verbose:
            print(f"Total points: {n_points:,}\n")
        
        results = {
            'file': laz_path.name,
            'n_points': n_points,
            'rgb': False,
            'nir': False,
            'linearity': False,
            'planarity': False,
            'sphericity': False,
            'anisotropy': False,
            'roughness': False,
            'warnings': []
        }
        
        # Check all features
        self._check_rgb(las, n_points, results)
        self._check_nir(las, n_points, results)
        self._check_linearity(las, n_points, results)
        self._check_geometric_features(las, n_points, results)
        
        # Show sample points if requested
        if self.show_samples and self.verbose:
            self._show_samples(las, n_points)
        
        if self.verbose:
            print('='*80)
        
        return results
    
    def _check_rgb(self, las, n_points: int, results: Dict):
        """Check RGB values."""
        if self.verbose:
            print("1. RGB VALUES CHECK")
            print('-'*80)
        
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            results['rgb'] = True
            rgb_r = (las.red / 257).astype(np.uint8)
            rgb_g = (las.green / 257).astype(np.uint8)
            rgb_b = (las.blue / 257).astype(np.uint8)
            
            rgb_unique = len(np.unique(np.column_stack([rgb_r, rgb_g, rgb_b]), axis=0))
            gray_count = ((rgb_r == 128) & (rgb_g == 128) & (rgb_b == 128)).sum()
            
            if self.verbose:
                print(f"✓ RGB channels present")
                print(f"  Red:   min={rgb_r.min():3d}, max={rgb_r.max():3d}, "
                      f"mean={rgb_r.mean():6.2f}, std={rgb_r.std():6.2f}")
                print(f"  Green: min={rgb_g.min():3d}, max={rgb_g.max():3d}, "
                      f"mean={rgb_g.mean():6.2f}, std={rgb_g.std():6.2f}")
                print(f"  Blue:  min={rgb_b.min():3d}, max={rgb_b.max():3d}, "
                      f"mean={rgb_b.mean():6.2f}, std={rgb_b.std():6.2f}")
                print(f"  Unique RGB combinations: {rgb_unique:,}")
            
            if rgb_unique < 100:
                warning = "Very few unique RGB values! RGB fetch may have failed."
                results['warnings'].append(warning)
                if self.verbose:
                    print(f"  ⚠️  WARNING: {warning}")
            
            if gray_count > n_points * 0.5:
                warning = f"{100*gray_count/n_points:.1f}% of points are default gray (128,128,128)"
                results['warnings'].append(warning)
                if self.verbose:
                    print(f"  ⚠️  WARNING: {warning}")
            elif self.verbose:
                print(f"  ✓ RGB values look good")
        else:
            if self.verbose:
                print("✗ RGB channels NOT present")
        
        if self.verbose:
            print()
    
    def _check_nir(self, las, n_points: int, results: Dict):
        """Check NIR (infrared) values."""
        if self.verbose:
            print("2. NIR (INFRARED) VALUES CHECK")
            print('-'*80)
        
        if hasattr(las, 'nir'):
            results['nir'] = True
            nir = las.nir
            nir_mode = np.bincount(nir).argmax()
            nir_mode_count = (nir == nir_mode).sum()
            
            if self.verbose:
                print(f"✓ NIR channel present")
                print(f"  Range: {nir.min()} - {nir.max()}")
                print(f"  Mean: {nir.mean():.2f}, Std: {nir.std():.2f}")
                print(f"  Unique NIR values: {len(np.unique(nir)):,}")
                print(f"  Most common value: {nir_mode} (appears {100*nir_mode_count/len(nir):.2f}%)")
            
            if nir_mode == 128 and nir_mode_count > n_points * 0.5:
                warning = f"{100*nir_mode_count/n_points:.1f}% of points have default NIR value (128)"
                results['warnings'].append(warning)
                if self.verbose:
                    print(f"  ⚠️  WARNING: {warning}")
            elif self.verbose:
                print(f"  ✓ NIR values look good")
        else:
            if self.verbose:
                print("✗ NIR channel NOT present")
        
        if self.verbose:
            print()
    
    def _check_linearity(self, las, n_points: int, results: Dict):
        """Check linearity feature."""
        if self.verbose:
            print("3. LINEARITY CHECK")
            print('-'*80)
        
        if hasattr(las, 'linearity'):
            results['linearity'] = True
            linearity = las.linearity
            
            if self.verbose:
                print(f"✓ Linearity present")
                print(f"  Range: {linearity.min():.6f} - {linearity.max():.6f}")
                print(f"  Mean: {linearity.mean():.6f}, Std: {linearity.std():.6f}")
                print(f"  Non-zero count: {(linearity > 0).sum():,} ({100*(linearity > 0).sum()/n_points:.2f}%)")
            
            if linearity.max() > 1.0:
                warning = f"Linearity values exceed 1.0! Max = {linearity.max()}"
                results['warnings'].append(warning)
                if self.verbose:
                    print(f"  ✗ ERROR: {warning}")
            elif linearity.min() < 0.0:
                warning = f"Linearity values below 0.0! Min = {linearity.min()}"
                results['warnings'].append(warning)
                if self.verbose:
                    print(f"  ✗ ERROR: {warning}")
            elif self.verbose:
                print(f"  ✓ Linearity values in valid range [0, 1]")
        else:
            if self.verbose:
                print("✗ Linearity NOT present")
        
        if self.verbose:
            print()
    
    def _check_geometric_features(self, las, n_points: int, results: Dict):
        """Check other geometric features."""
        if self.verbose:
            print("4. OTHER GEOMETRIC FEATURES CHECK")
            print('-'*80)
        
        geometric_features = ['planarity', 'sphericity', 'anisotropy', 'roughness']
        
        for feat_name in geometric_features:
            if hasattr(las, feat_name):
                results[feat_name] = True
                feat = getattr(las, feat_name)
                non_zero = (feat > 1e-6).sum()
                
                if self.verbose:
                    print(f"✓ {feat_name:12s}: min={feat.min():.4f}, max={feat.max():.4f}, "
                          f"mean={feat.mean():.4f}, non-zero={100*non_zero/n_points:.1f}%")
                
                if feat.max() > 1.0:
                    warning = f"{feat_name} exceeds 1.0! (max={feat.max():.4f})"
                    results['warnings'].append(warning)
                    if self.verbose:
                        print(f"  ⚠️  WARNING: {warning}")
                
                if feat.min() < 0.0:
                    warning = f"{feat_name} below 0.0! (min={feat.min():.4f})"
                    results['warnings'].append(warning)
                    if self.verbose:
                        print(f"  ⚠️  WARNING: {warning}")
            else:
                if self.verbose:
                    print(f"✗ {feat_name} NOT present")
        
        if self.verbose:
            print()
    
    def _show_samples(self, las, n_points: int):
        """Display sample points."""
        print("5. SAMPLE POINTS")
        print('-'*80)
        
        sample_idx = np.random.choice(n_points, min(10, n_points), replace=False)
        print(f"Random sample of {len(sample_idx)} points:\n")
        
        for i in sample_idx:
            line = f"  Point {i:8d}: "
            
            if hasattr(las, 'red'):
                rgb_r = int(las.red[i] / 257)
                rgb_g = int(las.green[i] / 257)
                rgb_b = int(las.blue[i] / 257)
                line += f"RGB=({rgb_r:3d},{rgb_g:3d},{rgb_b:3d}) "
            
            if hasattr(las, 'nir'):
                line += f"NIR={las.nir[i]:3d} "
            
            if hasattr(las, 'linearity'):
                line += f"L={las.linearity[i]:.4f} "
            
            if hasattr(las, 'planarity'):
                line += f"P={las.planarity[i]:.4f} "
            
            if hasattr(las, 'sphericity'):
                line += f"S={las.sphericity[i]:.4f}"
            
            print(line)
        
        print()
    
    def print_summary(self, all_results: List[Dict]):
        """
        Print summary of multiple file verifications.
        
        Args:
            all_results: List of verification result dictionaries
        """
        if not all_results or len(all_results) <= 1:
            return
        
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        total_files = len(all_results)
        
        # Count features
        feature_counts = {
            'rgb': sum(1 for r in all_results if r['rgb']),
            'nir': sum(1 for r in all_results if r['nir']),
            'linearity': sum(1 for r in all_results if r['linearity']),
            'planarity': sum(1 for r in all_results if r['planarity']),
            'sphericity': sum(1 for r in all_results if r['sphericity']),
            'anisotropy': sum(1 for r in all_results if r['anisotropy']),
            'roughness': sum(1 for r in all_results if r['roughness']),
        }
        
        print(f"Files verified: {total_files}")
        print(f"\nFeature presence:")
        for feat, count in feature_counts.items():
            status = "✓" if count == total_files else "⚠️"
            print(f"  {status} {feat:12s}: {count}/{total_files} files ({100*count/total_files:.1f}%)")
        
        # Count warnings
        total_warnings = sum(len(r['warnings']) for r in all_results)
        if total_warnings > 0:
            print(f"\n⚠️  Total warnings: {total_warnings}")
            files_with_warnings = sum(1 for r in all_results if r['warnings'])
            print(f"  Files with warnings: {files_with_warnings}/{total_files}")
        else:
            print(f"\n✓ No warnings detected")
        
        print("="*80)


def verify_laz_files(
    input_path: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    max_files: Optional[int] = None,
    verbose: bool = True,
    show_samples: bool = False
) -> List[Dict]:
    """
    Verify features in LAZ file(s).
    
    Args:
        input_path: Single LAZ file to verify
        input_dir: Directory containing LAZ files to verify
        max_files: Maximum number of files to verify
        verbose: If True, print detailed information
        show_samples: If True, display sample points
        
    Returns:
        List of verification result dictionaries
    """
    # Find LAZ files
    laz_files = []
    
    if input_path:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        laz_files = [input_path]
    
    elif input_dir:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Search recursively for LAZ files
        laz_files = list(input_dir.rglob("*.laz")) + list(input_dir.rglob("*.LAZ"))
        
        if not laz_files:
            raise FileNotFoundError(f"No LAZ files found in {input_dir}")
        
        logger.info(f"Found {len(laz_files)} LAZ files")
        
        # Limit to max_files if specified
        if max_files and len(laz_files) > max_files:
            logger.info(f"Limiting to first {max_files} files")
            laz_files = laz_files[:max_files]
    
    else:
        raise ValueError("Either input_path or input_dir must be specified")
    
    # Verify files
    verifier = FeatureVerifier(verbose=verbose, show_samples=show_samples)
    all_results = []
    
    for laz_file in laz_files:
        try:
            result = verifier.verify_file(laz_file)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {laz_file.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Print summary
    if verbose:
        verifier.print_summary(all_results)
    
    return all_results
