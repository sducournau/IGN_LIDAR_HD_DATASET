#!/usr/bin/env python3
"""
Batch testing script for V2 classification fixes.

Processes multiple tiles with both original and V2 configurations,
runs diagnostics, and generates a comprehensive comparison report.

Usage:
    python scripts/test_v2_fixes.py <input_dir> <output_dir> [--tiles TILE1 TILE2 ...]

Author: Classification Quality Audit - V2
Date: October 24, 2025
"""
import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
import argparse

def find_tiles(input_dir):
    """Find all LAS/LAZ tiles in input directory."""
    tiles = []
    for ext in ['*.las', '*.laz']:
        tiles.extend(Path(input_dir).glob(ext))
    return sorted(tiles)


def process_tile(tile_path, config_path, output_dir, label):
    """Process a single tile with given configuration."""
    print(f"  Processing with {label} config...")
    
    output_subdir = Path(output_dir) / label
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ign-lidar-hd', 'process',
        '-c', str(config_path),
        f'input_dir={tile_path.parent}',
        f'output_dir={output_subdir}'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        success = result.returncode == 0
        
        # Find output file
        output_file = None
        for f in output_subdir.glob('*_enriched.laz'):
            output_file = f
            break
        
        return {
            'success': success,
            'output_file': str(output_file) if output_file else None,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output_file': None,
            'error': 'Processing timeout (>1 hour)'
        }
    except Exception as e:
        return {
            'success': False,
            'output_file': None,
            'error': str(e)
        }


def run_diagnostic(las_file):
    """Run diagnostic script and extract metrics."""
    try:
        result = subprocess.run(
            ['python', 'scripts/diagnose_classification.py', str(las_file)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout
        
        # Parse metrics
        building_pct = None
        unclass_pct = None
        
        for line in output.split('\n'):
            if 'Class  6' in line and 'Building' in line:
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
            'building_pct': building_pct,
            'unclass_pct': unclass_pct,
            'output': output
        }
    except Exception as e:
        return {
            'success': False,
            'building_pct': None,
            'unclass_pct': None,
            'error': str(e)
        }


def generate_comparison(original_file, v2_file, output_dir, tile_name):
    """Generate comparison visualization."""
    comparison_dir = Path(output_dir) / 'comparisons' / tile_name
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ['python', 'scripts/compare_classifications.py',
             str(original_file), str(v2_file), str(comparison_dir)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        return result.returncode == 0
    except:
        return False


def generate_html_report(results, output_path):
    """Generate HTML report of all results."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>V2 Fixes Batch Test Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .summary {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .summary-item {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .summary-item .label {
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
        }
        .summary-item .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .status-pass {
            color: #27ae60;
            font-weight: bold;
        }
        .status-fail {
            color: #e74c3c;
            font-weight: bold;
        }
        .status-warning {
            color: #f39c12;
            font-weight: bold;
        }
        .metric-improved {
            color: #27ae60;
        }
        .metric-declined {
            color: #e74c3c;
        }
        .tile-section {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            background-color: #fafafa;
        }
        .comparison-link {
            display: inline-block;
            margin: 10px 0;
            padding: 8px 15px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .comparison-link:hover {
            background-color: #2980b9;
        }
        .error-box {
            background-color: #fee;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 V2 Fixes - Batch Test Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Total Tiles Tested:</strong> {total_tiles}</p>
"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tiles = len(results)
    
    # Calculate summary statistics
    successful_original = sum(1 for r in results if r.get('original_success'))
    successful_v2 = sum(1 for r in results if r.get('v2_success'))
    
    avg_building_improvement = 0
    avg_unclass_improvement = 0
    improved_count = 0
    
    for r in results:
        if (r.get('original_building_pct') is not None and 
            r.get('v2_building_pct') is not None):
            building_change = r['v2_building_pct'] - r['original_building_pct']
            unclass_change = r['v2_unclass_pct'] - r['original_unclass_pct']
            
            if building_change > 0 and unclass_change < 0:
                improved_count += 1
                avg_building_improvement += building_change
                avg_unclass_improvement += unclass_change
    
    if improved_count > 0:
        avg_building_improvement /= improved_count
        avg_unclass_improvement /= improved_count
    
    html += f"""
        <div class="summary">
            <h2>📊 Summary Statistics</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">Tiles Processed</div>
                    <div class="value">{total_tiles}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Original Success Rate</div>
                    <div class="value">{successful_original}/{total_tiles}</div>
                </div>
                <div class="summary-item">
                    <div class="label">V2 Success Rate</div>
                    <div class="value">{successful_v2}/{total_tiles}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Tiles Improved</div>
                    <div class="value">{improved_count}/{total_tiles}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Avg Building Increase</div>
                    <div class="value class="metric-improved">+{avg_building_improvement:.2f}%</div>
                </div>
                <div class="summary-item">
                    <div class="label">Avg Unclass Decrease</div>
                    <div class="value class="metric-improved">{avg_unclass_improvement:.2f}%</div>
                </div>
            </div>
        </div>

        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Tile</th>
                    <th>Original<br/>Building %</th>
                    <th>V2<br/>Building %</th>
                    <th>Change</th>
                    <th>Original<br/>Unclass %</th>
                    <th>V2<br/>Unclass %</th>
                    <th>Change</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for r in results:
        tile_name = r['tile_name']
        
        # Building metrics
        orig_build = r.get('original_building_pct')
        v2_build = r.get('v2_building_pct')
        build_change = ""
        if orig_build is not None and v2_build is not None:
            change = v2_build - orig_build
            change_pct = (change / orig_build * 100) if orig_build > 0 else 0
            build_change = f'<span class="{"metric-improved" if change > 0 else "metric-declined"}">{change:+.2f}% ({change_pct:+.1f}%)</span>'
        
        # Unclass metrics
        orig_unclass = r.get('original_unclass_pct')
        v2_unclass = r.get('v2_unclass_pct')
        unclass_change = ""
        if orig_unclass is not None and v2_unclass is not None:
            change = v2_unclass - orig_unclass
            change_pct = (change / orig_unclass * 100) if orig_unclass > 0 else 0
            unclass_change = f'<span class="{"metric-improved" if change < 0 else "metric-declined"}">{change:+.2f}% ({change_pct:+.1f}%)</span>'
        
        # Overall status
        status = "N/A"
        status_class = "status-warning"
        if orig_build is not None and v2_build is not None:
            if v2_build > orig_build * 1.5 and v2_unclass < orig_unclass * 0.6:
                status = "✅ EXCELLENT"
                status_class = "status-pass"
            elif v2_build > orig_build * 1.2 and v2_unclass < orig_unclass * 0.8:
                status = "✅ GOOD"
                status_class = "status-pass"
            elif v2_build > orig_build or v2_unclass < orig_unclass:
                status = "🟡 MODERATE"
                status_class = "status-warning"
            else:
                status = "❌ MINIMAL"
                status_class = "status-fail"
        
        html += f"""
                <tr>
                    <td><strong>{tile_name}</strong></td>
                    <td>{f"{orig_build:.2f}%" if orig_build is not None else "N/A"}</td>
                    <td>{f"{v2_build:.2f}%" if v2_build is not None else "N/A"}</td>
                    <td>{build_change if build_change else "N/A"}</td>
                    <td>{f"{orig_unclass:.2f}%" if orig_unclass is not None else "N/A"}</td>
                    <td>{f"{v2_unclass:.2f}%" if v2_unclass is not None else "N/A"}</td>
                    <td>{unclass_change if unclass_change else "N/A"}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>

        <h2>🔍 Individual Tile Details</h2>
"""
    
    for r in results:
        tile_name = r['tile_name']
        has_comparison = r.get('comparison_generated', False)
        
        html += f"""
        <div class="tile-section">
            <h3>📄 {tile_name}</h3>
"""
        
        if not r.get('original_success'):
            html += f"""
            <div class="error-box">
                <strong>❌ Original Processing Failed</strong><br/>
                {r.get('original_error', 'Unknown error')}
            </div>
"""
        
        if not r.get('v2_success'):
            html += f"""
            <div class="error-box">
                <strong>❌ V2 Processing Failed</strong><br/>
                {r.get('v2_error', 'Unknown error')}
            </div>
"""
        
        if has_comparison:
            comparison_path = f"comparisons/{tile_name}/comparison_visualization.png"
            report_path = f"comparisons/{tile_name}/comparison_report.txt"
            
            html += f"""
            <p><strong>Comparison Files:</strong></p>
            <a href="{comparison_path}" class="comparison-link">📊 View Visualization</a>
            <a href="{report_path}" class="comparison-link">📝 View Report</a>
"""
        
        html += """
        </div>
"""
    
    html += f"""
        <div class="footer">
            <p>Generated by IGN LiDAR HD Classification - Batch Testing Script</p>
            <p>V2 Fixes Quality Audit - October 24, 2025</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html.format(
            timestamp=timestamp,
            total_tiles=total_tiles
        ))
    
    print(f"✅ HTML report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch test V2 classification fixes on multiple tiles'
    )
    parser.add_argument('input_dir', help='Directory containing input tiles')
    parser.add_argument('output_dir', help='Directory for output results')
    parser.add_argument('--tiles', nargs='+', help='Specific tiles to test (optional)')
    parser.add_argument('--original-config', 
                       default='examples/config_asprs_bdtopo_cadastre_cpu_optimized.yaml',
                       help='Original configuration file')
    parser.add_argument('--v2-config',
                       default='examples/config_asprs_bdtopo_cadastre_cpu_fixed.yaml',
                       help='V2 fixed configuration file')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip processing, only run diagnostics on existing outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🔬 V2 Fixes - Batch Testing")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Original config: {args.original_config}")
    print(f"V2 config: {args.v2_config}")
    print()
    
    # Find tiles
    if args.tiles:
        tiles = [Path(args.input_dir) / t for t in args.tiles]
    else:
        tiles = find_tiles(args.input_dir)
    
    if not tiles:
        print("❌ No tiles found!")
        sys.exit(1)
    
    print(f"Found {len(tiles)} tile(s) to test")
    print()
    
    results = []
    
    for i, tile in enumerate(tiles, 1):
        tile_name = tile.stem
        print(f"[{i}/{len(tiles)}] Processing: {tile_name}")
        print("-"*80)
        
        result = {
            'tile_name': tile_name,
            'tile_path': str(tile)
        }
        
        # Process with original config
        if not args.skip_processing:
            print("  🔄 Processing with ORIGINAL config...")
            original_result = process_tile(tile, args.original_config, output_dir, 'original')
            result['original_success'] = original_result['success']
            result['original_file'] = original_result.get('output_file')
            if not original_result['success']:
                result['original_error'] = original_result.get('error', 'Processing failed')
                print(f"    ❌ Failed: {result['original_error']}")
            else:
                print(f"    ✅ Success: {result['original_file']}")
        else:
            # Look for existing output
            original_file = output_dir / 'original' / f'{tile_name}_enriched.laz'
            if original_file.exists():
                result['original_success'] = True
                result['original_file'] = str(original_file)
                print(f"    ✅ Using existing: {original_file}")
            else:
                result['original_success'] = False
                print(f"    ❌ No existing output found")
        
        # Process with V2 config
        if not args.skip_processing:
            print("  🔄 Processing with V2 config...")
            v2_result = process_tile(tile, args.v2_config, output_dir, 'v2')
            result['v2_success'] = v2_result['success']
            result['v2_file'] = v2_result.get('output_file')
            if not v2_result['success']:
                result['v2_error'] = v2_result.get('error', 'Processing failed')
                print(f"    ❌ Failed: {result['v2_error']}")
            else:
                print(f"    ✅ Success: {result['v2_file']}")
        else:
            # Look for existing output
            v2_file = output_dir / 'v2' / f'{tile_name}_enriched.laz'
            if v2_file.exists():
                result['v2_success'] = True
                result['v2_file'] = str(v2_file)
                print(f"    ✅ Using existing: {v2_file}")
            else:
                result['v2_success'] = False
                print(f"    ❌ No existing output found")
        
        # Run diagnostics on both
        if result.get('original_success') and result.get('original_file'):
            print("  📊 Running diagnostic on ORIGINAL...")
            diag = run_diagnostic(result['original_file'])
            result['original_building_pct'] = diag.get('building_pct')
            result['original_unclass_pct'] = diag.get('unclass_pct')
            print(f"    Building: {diag.get('building_pct'):.2f}%, Unclassified: {diag.get('unclass_pct'):.2f}%")
        
        if result.get('v2_success') and result.get('v2_file'):
            print("  📊 Running diagnostic on V2...")
            diag = run_diagnostic(result['v2_file'])
            result['v2_building_pct'] = diag.get('building_pct')
            result['v2_unclass_pct'] = diag.get('unclass_pct')
            print(f"    Building: {diag.get('building_pct'):.2f}%, Unclassified: {diag.get('unclass_pct'):.2f}%")
        
        # Generate comparison
        if (result.get('original_success') and result.get('v2_success') and
            result.get('original_file') and result.get('v2_file')):
            print("  📈 Generating comparison...")
            comparison_success = generate_comparison(
                result['original_file'],
                result['v2_file'],
                output_dir,
                tile_name
            )
            result['comparison_generated'] = comparison_success
            if comparison_success:
                print(f"    ✅ Comparison saved")
            else:
                print(f"    ⚠️  Comparison may have issues")
        
        results.append(result)
        print()
    
    # Generate summary report
    print("="*80)
    print("📊 Generating summary report...")
    
    # Save JSON results
    json_path = output_dir / 'batch_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ JSON results saved: {json_path}")
    
    # Generate HTML report
    html_path = output_dir / 'batch_test_report.html'
    generate_html_report(results, html_path)
    
    print()
    print("="*80)
    print("✅ Batch testing complete!")
    print(f"📁 Results: {output_dir}/")
    print(f"📄 Report: {html_path}")
    print("="*80)


if __name__ == "__main__":
    main()
