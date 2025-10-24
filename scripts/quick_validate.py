#!/usr/bin/env python3
"""
Quick validation script for classification results.

Runs both diagnostic and visualization, then provides a simple PASS/FAIL
with key metrics (unclassified %, building coverage %).

Usage:
    python scripts/quick_validate.py <las_file> [output_dir]

Author: Classification Quality Audit - V2
Date: October 24, 2025
"""
import sys
import os
import subprocess
from pathlib import Path

def run_diagnostic(las_file):
    """Run diagnose_classification.py and parse results."""
    try:
        result = subprocess.run(
            ['python', 'scripts/diagnose_classification.py', las_file],
            capture_output=True,
            text=True
        )
        
        output = result.stdout
        
        # Parse key metrics from output
        building_pct = None
        unclass_pct = None
        
        for line in output.split('\n'):
            if 'Class  6' in line and 'Building' in line:
                # Extract percentage
                parts = line.split('(')
                if len(parts) > 1:
                    pct_str = parts[-1].split('%')[0]
                    try:
                        building_pct = float(pct_str)
                    except:
                        pass
            
            if 'Class  1' in line and 'Unclassified' in line:
                parts = line.split('(')
                if len(parts) > 1:
                    pct_str = parts[-1].split('%')[0]
                    try:
                        unclass_pct = float(pct_str)
                    except:
                        pass
        
        return {
            'success': result.returncode == 0,
            'output': output,
            'building_pct': building_pct,
            'unclass_pct': unclass_pct
        }
    except Exception as e:
        return {
            'success': False,
            'output': str(e),
            'building_pct': None,
            'unclass_pct': None
        }


def run_visualization(las_file, output_path):
    """Run visualize_classification.py."""
    try:
        result = subprocess.run(
            ['python', 'scripts/visualize_classification.py', las_file, output_path],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def evaluate_results(building_pct, unclass_pct):
    """Evaluate results and determine PASS/FAIL."""
    
    issues = []
    warnings = []
    passed = True
    
    # Building classification check
    if building_pct is None:
        issues.append("❌ Unable to determine building classification rate")
        passed = False
    elif building_pct < 1.0:
        issues.append(f"❌ CRITICAL: Building classification rate too low ({building_pct:.2f}%)")
        issues.append("   Expected: >5% for typical areas")
        passed = False
    elif building_pct < 5.0:
        warnings.append(f"⚠️  WARNING: Building classification rate low ({building_pct:.2f}%)")
        warnings.append("   Expected: >5% for typical areas")
        warnings.append("   Consider applying V3 configuration")
    elif building_pct < 10.0:
        warnings.append(f"🟡 Building classification moderate ({building_pct:.2f}%)")
        warnings.append("   Room for improvement with V3 config")
    else:
        # Good
        pass
    
    # Unclassified rate check
    if unclass_pct is None:
        issues.append("❌ Unable to determine unclassified rate")
        passed = False
    elif unclass_pct > 30.0:
        issues.append(f"❌ CRITICAL: Unclassified rate too high ({unclass_pct:.2f}%)")
        issues.append("   Expected: <15% after V2 fixes")
        issues.append("   Apply V2 configuration if not already applied")
        passed = False
    elif unclass_pct > 20.0:
        warnings.append(f"⚠️  WARNING: Unclassified rate high ({unclass_pct:.2f}%)")
        warnings.append("   Expected: <15% after V2 fixes")
        warnings.append("   Consider V3 configuration for more aggressive classification")
    elif unclass_pct > 15.0:
        warnings.append(f"🟡 Unclassified rate moderate ({unclass_pct:.2f}%)")
        warnings.append("   Acceptable, but V3 could improve further")
    else:
        # Good
        pass
    
    return {
        'passed': passed and len(warnings) == 0,
        'passed_with_warnings': passed and len(warnings) > 0,
        'failed': not passed,
        'issues': issues,
        'warnings': warnings
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/quick_validate.py <las_file> [output_dir]")
        print("\nExample:")
        print("  python scripts/quick_validate.py output/v2_fixed/tile_enriched.laz")
        print("  python scripts/quick_validate.py output/v2_fixed/tile_enriched.laz validation_results")
        sys.exit(1)
    
    las_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "validation_results"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🔍 QUICK VALIDATION - Classification Quality Check")
    print("="*80)
    print()
    print(f"Input file: {las_file}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Run diagnostic
    print("📊 Running diagnostic analysis...")
    diag_results = run_diagnostic(las_file)
    
    if not diag_results['success']:
        print(f"❌ Diagnostic failed: {diag_results['output']}")
        sys.exit(1)
    
    # Save diagnostic output
    diag_output_file = os.path.join(output_dir, "diagnostic_report.txt")
    with open(diag_output_file, 'w') as f:
        f.write(diag_results['output'])
    print(f"   ✅ Diagnostic report saved: {diag_output_file}")
    
    # Run visualization
    print("📈 Generating visualization...")
    viz_output_file = os.path.join(output_dir, "classification_visualization.png")
    viz_success = run_visualization(las_file, viz_output_file)
    
    if viz_success:
        print(f"   ✅ Visualization saved: {viz_output_file}")
    else:
        print(f"   ⚠️  Visualization may have issues (check {viz_output_file})")
    
    print()
    print("="*80)
    print("📋 KEY METRICS")
    print("="*80)
    
    building_pct = diag_results['building_pct']
    unclass_pct = diag_results['unclass_pct']
    
    if building_pct is not None:
        if building_pct >= 10:
            icon = "✅"
        elif building_pct >= 5:
            icon = "🟡"
        else:
            icon = "❌"
        print(f"{icon} Building Classification: {building_pct:.2f}%")
    else:
        print(f"❌ Building Classification: Unable to determine")
    
    if unclass_pct is not None:
        if unclass_pct <= 15:
            icon = "✅"
        elif unclass_pct <= 20:
            icon = "🟡"
        else:
            icon = "❌"
        print(f"{icon} Unclassified Rate: {unclass_pct:.2f}%")
    else:
        print(f"❌ Unclassified Rate: Unable to determine")
    
    print()
    print("="*80)
    print("📊 EVALUATION")
    print("="*80)
    
    eval_results = evaluate_results(building_pct, unclass_pct)
    
    if eval_results['passed']:
        print("✅ PASSED - Classification quality is good!")
        print()
        print("Next steps:")
        print("  1. Review visualization to visually verify results")
        print("  2. Process full dataset if satisfied")
        print("  3. Document results for reference")
    
    elif eval_results['passed_with_warnings']:
        print("⚠️  PASSED WITH WARNINGS - Classification acceptable but could be improved")
        print()
        if eval_results['warnings']:
            print("Warnings:")
            for warning in eval_results['warnings']:
                print(f"  {warning}")
        print()
        print("Next steps:")
        print("  1. Review warnings above")
        print("  2. Consider applying V3 config for improvement")
        print("  3. Or proceed with current results if acceptable")
    
    else:
        print("❌ FAILED - Classification quality issues detected")
        print()
        if eval_results['issues']:
            print("Issues:")
            for issue in eval_results['issues']:
                print(f"  {issue}")
        print()
        if eval_results['warnings']:
            print("Warnings:")
            for warning in eval_results['warnings']:
                print(f"  {warning}")
        print()
        print("Recommended actions:")
        print("  1. Review detailed diagnostic report")
        print("  2. Apply V2 configuration if not already applied:")
        print("     config_asprs_bdtopo_cadastre_cpu_fixed.yaml")
        print("  3. If already using V2, try V3 configuration:")
        print("     config_asprs_bdtopo_cadastre_cpu_v3.yaml")
        print("  4. Check ground truth data availability")
        print("  5. Verify DTM computation (RGE ALTI)")
    
    print()
    print("="*80)
    print("📁 VALIDATION RESULTS")
    print("="*80)
    print(f"Location: {output_dir}/")
    print(f"  - diagnostic_report.txt       Detailed analysis")
    print(f"  - classification_visualization.png  Visual results")
    print()
    print("For detailed guidance, see:")
    print("  - docs/QUICK_START_V2.md")
    print("  - docs/CLASSIFICATION_AUDIT_CORRECTION.md")
    print("="*80)
    
    # Create summary file
    summary_file = os.path.join(output_dir, "validation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("QUICK VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input File: {las_file}\n")
        f.write(f"Timestamp: {os.popen('date').read().strip()}\n\n")
        
        f.write("KEY METRICS:\n")
        f.write(f"  Building Classification: {building_pct:.2f}%\n" if building_pct else "  Building Classification: N/A\n")
        f.write(f"  Unclassified Rate: {unclass_pct:.2f}%\n" if unclass_pct else "  Unclassified Rate: N/A\n")
        f.write("\n")
        
        f.write("RESULT: ")
        if eval_results['passed']:
            f.write("✅ PASSED\n")
        elif eval_results['passed_with_warnings']:
            f.write("⚠️  PASSED WITH WARNINGS\n")
        else:
            f.write("❌ FAILED\n")
        f.write("\n")
        
        if eval_results['issues']:
            f.write("ISSUES:\n")
            for issue in eval_results['issues']:
                f.write(f"  {issue}\n")
            f.write("\n")
        
        if eval_results['warnings']:
            f.write("WARNINGS:\n")
            for warning in eval_results['warnings']:
                f.write(f"  {warning}\n")
            f.write("\n")
        
        f.write("See detailed reports for more information.\n")
    
    print(f"✅ Summary saved: {summary_file}")
    print()
    
    # Exit with appropriate code
    if eval_results['passed'] or eval_results['passed_with_warnings']:
        sys.exit(0)
    else:
        sys.exit(1)
