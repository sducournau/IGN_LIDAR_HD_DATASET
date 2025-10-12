#!/usr/bin/env python3
"""
Fix Enriched LAZ Files - Post-processing tool to correct data quality issues

This script fixes common issues found in enriched LAZ files:
1. Recalculate NDVI from NIR/Red if NIR data is available
2. Cap extreme eigenvalue outliers
3. Recompute derived features that depend on eigenvalues
4. Add validation and reporting

Usage:
    python fix_enriched_laz.py --input file.laz --output file_fixed.laz [options]
    
Author: Simon Ducournau
Date: October 12, 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import laspy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LAZFixer:
    """Fix data quality issues in enriched LAZ files."""
    
    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        fix_ndvi: bool = True,
        cap_eigenvalues: float = 100.0,
        recompute_derived: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the LAZ fixer.
        
        Args:
            input_path: Path to input LAZ file
            output_path: Path to output LAZ file (if None, will add _fixed suffix)
            fix_ndvi: Whether to recalculate NDVI
            cap_eigenvalues: Maximum allowed eigenvalue (values above will be capped)
            recompute_derived: Whether to recompute derived features
            verbose: Enable verbose logging
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self._default_output_path()
        self.fix_ndvi = fix_ndvi
        self.cap_eigenvalues = cap_eigenvalues
        self.recompute_derived = recompute_derived
        self.verbose = verbose
        
        self.issues_found = []
        self.fixes_applied = []
        
    def _default_output_path(self) -> Path:
        """Generate default output path with _fixed suffix."""
        stem = self.input_path.stem
        suffix = self.input_path.suffix
        return self.input_path.parent / f"{stem}_fixed{suffix}"
    
    def run(self) -> bool:
        """
        Run the fixing process.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("LAZ FILE FIXER - Starting")
        logger.info("=" * 80)
        logger.info(f"Input:  {self.input_path}")
        logger.info(f"Output: {self.output_path}")
        logger.info("")
        
        try:
            # Load LAZ file
            logger.info("Loading LAZ file...")
            las = laspy.read(str(self.input_path))
            logger.info(f"✓ Loaded {len(las.points):,} points (format {las.point_format.id})")
            
            # Analyze issues
            logger.info("\n" + "=" * 80)
            logger.info("ANALYZING ISSUES")
            logger.info("=" * 80)
            self._analyze_issues(las)
            
            if not self.issues_found:
                logger.info("✓ No issues found! File appears to be clean.")
                return True
            
            # Apply fixes
            logger.info("\n" + "=" * 80)
            logger.info("APPLYING FIXES")
            logger.info("=" * 80)
            las = self._apply_fixes(las)
            
            # Validate results
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATING RESULTS")
            logger.info("=" * 80)
            self._validate_results(las)
            
            # Save fixed file
            logger.info("\n" + "=" * 80)
            logger.info("SAVING FIXED FILE")
            logger.info("=" * 80)
            logger.info(f"Writing to: {self.output_path}")
            las.write(str(self.output_path))
            logger.info(f"✓ File saved successfully")
            
            # Print summary
            self._print_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def _analyze_issues(self, las: laspy.LasData) -> None:
        """Analyze the LAZ file for common issues."""
        
        # Check NDVI
        if hasattr(las, 'ndvi'):
            ndvi = np.array(las.ndvi)
            if np.all(np.abs(ndvi + 1.0) < 0.001):
                self.issues_found.append({
                    'type': 'ndvi_broken',
                    'severity': 'critical',
                    'description': 'All NDVI values are -1.0',
                    'affected_points': len(ndvi)
                })
                logger.warning("🔴 NDVI: All values are -1.0 (BROKEN)")
                
                # Check if we can fix it
                if hasattr(las, 'nir') and hasattr(las, 'red'):
                    nir = np.array(las.nir)
                    if np.max(nir) > 0.001:
                        logger.info("   ℹ️  NIR data available - NDVI can be recalculated")
                    else:
                        logger.warning("   ⚠️  NIR data is zero - cannot fix NDVI")
                else:
                    logger.warning("   ⚠️  NIR/Red data missing - cannot fix NDVI")
        
        # Check eigenvalues
        if hasattr(las, 'eigenvalue_1'):
            eig1 = np.array(las.eigenvalue_1)
            
            severe_outliers = np.sum(eig1 > 10000)
            moderate_outliers = np.sum(eig1 > 1000)
            mild_outliers = np.sum(eig1 > self.cap_eigenvalues)
            
            if severe_outliers > 0:
                self.issues_found.append({
                    'type': 'eigenvalue_extreme',
                    'severity': 'critical',
                    'description': f'Eigenvalues > 10,000',
                    'affected_points': severe_outliers,
                    'max_value': np.max(eig1)
                })
                logger.warning(f"🔴 Eigenvalues: {severe_outliers:,} points > 10,000 "
                             f"(max: {np.max(eig1):.2f})")
            
            if moderate_outliers > 0:
                self.issues_found.append({
                    'type': 'eigenvalue_high',
                    'severity': 'high',
                    'description': f'Eigenvalues > 1,000',
                    'affected_points': moderate_outliers
                })
                logger.warning(f"🟠 Eigenvalues: {moderate_outliers:,} points > 1,000")
            
            if mild_outliers > severe_outliers:
                logger.info(f"🟡 Eigenvalues: {mild_outliers:,} points > {self.cap_eigenvalues} "
                          f"(will be capped)")
        
        # Check derived features
        derived_features = [
            ('change_curvature', 100),
            ('omnivariance', 100),
            ('surface_roughness', 10)
        ]
        
        for feat_name, threshold in derived_features:
            if hasattr(las, feat_name):
                values = np.array(getattr(las, feat_name))
                outliers = np.sum(values > threshold)
                if outliers > 0:
                    self.issues_found.append({
                        'type': f'{feat_name}_outliers',
                        'severity': 'medium',
                        'description': f'{feat_name} > {threshold}',
                        'affected_points': outliers,
                        'max_value': np.max(values)
                    })
                    logger.warning(f"🟡 {feat_name}: {outliers:,} points > {threshold} "
                                 f"(max: {np.max(values):.2f})")
        
        logger.info(f"\nTotal issues found: {len(self.issues_found)}")
    
    def _apply_fixes(self, las: laspy.LasData) -> laspy.LasData:
        """Apply fixes to the LAZ file."""
        
        # Fix 1: Recalculate NDVI if possible
        if self.fix_ndvi and any(issue['type'] == 'ndvi_broken' for issue in self.issues_found):
            las = self._fix_ndvi(las)
        
        # Fix 2: Cap eigenvalues
        eigenvalue_issues = [issue for issue in self.issues_found 
                           if 'eigenvalue' in issue['type']]
        if eigenvalue_issues:
            las = self._cap_eigenvalues(las)
        
        # Fix 3: Recompute derived features
        if self.recompute_derived and eigenvalue_issues:
            las = self._recompute_derived_features(las)
        
        return las
    
    def _fix_ndvi(self, las: laspy.LasData) -> laspy.LasData:
        """Attempt to recalculate NDVI from NIR and Red channels."""
        logger.info("\n[1/3] Fixing NDVI...")
        
        if not (hasattr(las, 'nir') and hasattr(las, 'red')):
            logger.warning("   ⚠️  Cannot fix NDVI: NIR or Red data missing")
            return las
        
        nir = np.array(las.nir, dtype=np.float32)
        red = np.array(las.red, dtype=np.float32)
        
        # Check if NIR is in uint16 range and normalize
        if np.max(nir) > 1.0:
            logger.info(f"   Normalizing NIR from uint16 range (max={np.max(nir):.0f})")
            nir = nir / 65535.0
        
        if np.max(red) > 1.0:
            logger.info(f"   Normalizing Red from uint16 range (max={np.max(red):.0f})")
            red = red / 65535.0
        
        # Check if NIR data is actually present
        if np.max(nir) < 0.001:
            logger.warning("   ⚠️  Cannot fix NDVI: NIR data is essentially zero")
            return las
        
        # Compute NDVI
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.clip(ndvi, -1, 1)
        
        # Update LAZ file
        las.ndvi = ndvi.astype(np.float32)
        
        # Log statistics
        valid_mask = np.isfinite(ndvi)
        logger.info(f"   ✓ NDVI recalculated:")
        logger.info(f"     Range: [{np.min(ndvi[valid_mask]):.3f}, {np.max(ndvi[valid_mask]):.3f}]")
        logger.info(f"     Mean: {np.mean(ndvi[valid_mask]):.3f}")
        logger.info(f"     Std:  {np.std(ndvi[valid_mask]):.3f}")
        
        self.fixes_applied.append('NDVI recalculated')
        return las
    
    def _cap_eigenvalues(self, las: laspy.LasData) -> laspy.LasData:
        """Cap extreme eigenvalues to reasonable limits."""
        logger.info(f"\n[2/3] Capping eigenvalues (max={self.cap_eigenvalues})...")
        
        capped_counts = {}
        
        for i in [1, 2, 3]:
            attr_name = f'eigenvalue_{i}'
            if hasattr(las, attr_name):
                values = np.array(getattr(las, attr_name), dtype=np.float32)
                original_max = np.max(values)
                
                # Cap values
                capped_mask = values > self.cap_eigenvalues
                capped_count = np.sum(capped_mask)
                
                if capped_count > 0:
                    values[capped_mask] = self.cap_eigenvalues
                    setattr(las, attr_name, values)
                    capped_counts[attr_name] = capped_count
                    
                    logger.info(f"   ✓ {attr_name}: capped {capped_count:,} points "
                              f"(max was {original_max:.2f})")
        
        # Also cap sum_eigenvalues if it exists
        if hasattr(las, 'sum_eigenvalues'):
            values = np.array(las.sum_eigenvalues, dtype=np.float32)
            max_sum = 3 * self.cap_eigenvalues
            capped_mask = values > max_sum
            capped_count = np.sum(capped_mask)
            
            if capped_count > 0:
                values[capped_mask] = max_sum
                las.sum_eigenvalues = values
                logger.info(f"   ✓ sum_eigenvalues: capped {capped_count:,} points "
                          f"(max={max_sum:.2f})")
        
        if capped_counts:
            self.fixes_applied.append(f"Capped eigenvalues: {sum(capped_counts.values())} points")
        
        return las
    
    def _recompute_derived_features(self, las: laspy.LasData) -> laspy.LasData:
        """Recompute features that depend on eigenvalues."""
        logger.info("\n[3/3] Recomputing derived features...")
        
        if not all(hasattr(las, f'eigenvalue_{i}') for i in [1, 2, 3]):
            logger.warning("   ⚠️  Cannot recompute: eigenvalues missing")
            return las
        
        # Get eigenvalues
        lambda1 = np.array(las.eigenvalue_1, dtype=np.float32)
        lambda2 = np.array(las.eigenvalue_2, dtype=np.float32)
        lambda3 = np.array(las.eigenvalue_3, dtype=np.float32)
        
        epsilon = 1e-8
        sum_lambdas = lambda1 + lambda2 + lambda3 + epsilon
        
        features_recomputed = []
        
        # Recompute anisotropy
        if hasattr(las, 'anisotropy'):
            anisotropy = (lambda1 - lambda3) / lambda1.clip(epsilon, None)
            las.anisotropy = anisotropy.astype(np.float32)
            features_recomputed.append('anisotropy')
        
        # Recompute planarity
        if hasattr(las, 'planarity'):
            planarity = (lambda2 - lambda3) / lambda1.clip(epsilon, None)
            las.planarity = planarity.astype(np.float32)
            features_recomputed.append('planarity')
        
        # Recompute linearity
        if hasattr(las, 'linearity'):
            linearity = (lambda1 - lambda2) / lambda1.clip(epsilon, None)
            las.linearity = linearity.astype(np.float32)
            features_recomputed.append('linearity')
        
        # Recompute sphericity
        if hasattr(las, 'sphericity'):
            sphericity = lambda3 / lambda1.clip(epsilon, None)
            las.sphericity = sphericity.astype(np.float32)
            features_recomputed.append('sphericity')
        
        # Recompute omnivariance
        if hasattr(las, 'omnivariance'):
            omnivariance = np.cbrt(lambda1 * lambda2 * lambda3)
            las.omnivariance = omnivariance.astype(np.float32)
            features_recomputed.append('omnivariance')
        
        # Recompute eigenentropy
        if hasattr(las, 'eigenentropy'):
            with np.errstate(divide='ignore', invalid='ignore'):
                e1 = lambda1 / sum_lambdas
                e2 = lambda2 / sum_lambdas
                e3 = lambda3 / sum_lambdas
                eigenentropy = -(e1 * np.log(e1 + epsilon) + 
                               e2 * np.log(e2 + epsilon) + 
                               e3 * np.log(e3 + epsilon))
                eigenentropy = np.nan_to_num(eigenentropy, nan=0.0, posinf=0.0, neginf=0.0)
            las.eigenentropy = eigenentropy.astype(np.float32)
            features_recomputed.append('eigenentropy')
        
        # Recompute change_curvature
        if hasattr(las, 'change_curvature'):
            change_curvature = lambda3 / sum_lambdas
            las.change_curvature = change_curvature.astype(np.float32)
            features_recomputed.append('change_curvature')
        
        # Note: surface_roughness might need neighborhood info, skip for now
        
        logger.info(f"   ✓ Recomputed {len(features_recomputed)} features:")
        for feat in features_recomputed:
            logger.info(f"     - {feat}")
        
        self.fixes_applied.append(f"Recomputed {len(features_recomputed)} derived features")
        return las
    
    def _validate_results(self, las: laspy.LasData) -> None:
        """Validate that fixes were successful."""
        
        all_passed = True
        
        # Check NDVI
        if hasattr(las, 'ndvi'):
            ndvi = np.array(las.ndvi)
            all_minus_one = np.all(np.abs(ndvi + 1.0) < 0.001)
            if all_minus_one:
                logger.warning("⚠️  NDVI still broken (all -1.0)")
                all_passed = False
            else:
                logger.info("✓ NDVI: Values are valid")
        
        # Check eigenvalues
        if hasattr(las, 'eigenvalue_1'):
            eig1 = np.array(las.eigenvalue_1)
            max_eig = np.max(eig1)
            if max_eig > self.cap_eigenvalues * 1.01:  # Allow small tolerance
                logger.warning(f"⚠️  Eigenvalues: Max is {max_eig:.2f} > {self.cap_eigenvalues}")
                all_passed = False
            else:
                logger.info(f"✓ Eigenvalues: All values ≤ {self.cap_eigenvalues}")
        
        # Check derived features
        checks = [
            ('change_curvature', 1.0),
            ('omnivariance', self.cap_eigenvalues),
        ]
        
        for feat_name, max_expected in checks:
            if hasattr(las, feat_name):
                values = np.array(getattr(las, feat_name))
                max_val = np.max(values)
                if max_val > max_expected * 1.5:  # Allow some tolerance
                    logger.warning(f"⚠️  {feat_name}: Max is {max_val:.2f} > {max_expected:.2f}")
                    all_passed = False
                else:
                    logger.info(f"✓ {feat_name}: Max is {max_val:.2f}")
        
        if all_passed:
            logger.info("\n✅ All validation checks passed!")
        else:
            logger.warning("\n⚠️  Some validation checks failed")
    
    def _print_summary(self) -> None:
        """Print summary of operations."""
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Issues found:  {len(self.issues_found)}")
        logger.info(f"Fixes applied: {len(self.fixes_applied)}")
        
        if self.fixes_applied:
            logger.info("\nFixes applied:")
            for fix in self.fixes_applied:
                logger.info(f"  ✓ {fix}")
        
        logger.info(f"\n✅ Processing complete!")
        logger.info(f"Output file: {self.output_path}")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix data quality issues in enriched LAZ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all issues with default settings
  python fix_enriched_laz.py --input enriched.laz
  
  # Fix with custom eigenvalue cap
  python fix_enriched_laz.py --input enriched.laz --cap-eigenvalues 50
  
  # Only cap eigenvalues, don't recompute derived features
  python fix_enriched_laz.py --input enriched.laz --no-recompute
  
  # Specify custom output path
  python fix_enriched_laz.py --input enriched.laz --output fixed.laz
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input LAZ file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output LAZ file path (default: input_fixed.laz)'
    )
    
    parser.add_argument(
        '--fix-ndvi',
        action='store_true',
        default=True,
        help='Attempt to recalculate NDVI from NIR/Red (default: True)'
    )
    
    parser.add_argument(
        '--no-fix-ndvi',
        dest='fix_ndvi',
        action='store_false',
        help='Do not attempt to fix NDVI'
    )
    
    parser.add_argument(
        '--cap-eigenvalues',
        type=float,
        default=100.0,
        help='Maximum allowed eigenvalue (default: 100.0)'
    )
    
    parser.add_argument(
        '--recompute-derived',
        action='store_true',
        default=True,
        help='Recompute derived features from eigenvalues (default: True)'
    )
    
    parser.add_argument(
        '--no-recompute',
        dest='recompute_derived',
        action='store_false',
        help='Do not recompute derived features'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Create fixer and run
    fixer = LAZFixer(
        input_path=args.input,
        output_path=args.output,
        fix_ndvi=args.fix_ndvi,
        cap_eigenvalues=args.cap_eigenvalues,
        recompute_derived=args.recompute_derived,
        verbose=args.verbose
    )
    
    success = fixer.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
