#!/usr/bin/env python3
"""
Automated Code Duplication Analysis Tool

This script analyzes the IGN LiDAR HD codebase to detect:
1. Duplicate function definitions
2. Similar code blocks
3. Import complexity
4. Module coupling

Usage:
    python scripts/analyze_duplication.py
    python scripts/analyze_duplication.py --output report.json
    python scripts/analyze_duplication.py --module features
"""

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
import sys


class FunctionAnalyzer(ast.NodeVisitor):
    """Analyzes Python AST to find function definitions and their signatures."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.functions = []
        self.imports = []
        self.classes = []
        
    def visit_FunctionDef(self, node):
        """Record function definition."""
        # Get function signature
        args = [arg.arg for arg in node.args.args]
        
        self.functions.append({
            'name': node.name,
            'line': node.lineno,
            'args': args,
            'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'file': str(self.filepath)
        })
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Record class definition."""
        self.classes.append({
            'name': node.name,
            'line': node.lineno,
            'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
            'file': str(self.filepath)
        })
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Record import statements."""
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'type': 'import'
            })
    
    def visit_ImportFrom(self, node):
        """Record from...import statements."""
        for alias in node.names:
            self.imports.append({
                'module': node.module,
                'name': alias.name,
                'alias': alias.asname,
                'line': node.lineno,
                'type': 'from'
            })


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        analyzer = FunctionAnalyzer(filepath)
        analyzer.visit(tree)
        
        return {
            'file': str(filepath),
            'lines': len(content.splitlines()),
            'functions': analyzer.functions,
            'classes': analyzer.classes,
            'imports': analyzer.imports,
        }
    except Exception as e:
        print(f"❌ Error analyzing {filepath}: {e}", file=sys.stderr)
        return None


def find_duplicate_functions(analyses: List[Dict]) -> Dict[str, List[Dict]]:
    """Find functions with same name across multiple files."""
    function_map = defaultdict(list)
    
    for analysis in analyses:
        if not analysis:
            continue
        for func in analysis['functions']:
            function_map[func['name']].append({
                'file': analysis['file'],
                'line': func['line'],
                'args': func['args'],
                'decorators': func['decorators']
            })
    
    # Filter to only duplicates
    duplicates = {
        name: locations 
        for name, locations in function_map.items() 
        if len(locations) > 1
    }
    
    return duplicates


def analyze_module_coupling(analyses: List[Dict]) -> Dict[str, Dict]:
    """Analyze import dependencies between modules."""
    coupling = defaultdict(lambda: {'imports': [], 'imported_by': []})
    
    for analysis in analyses:
        if not analysis:
            continue
        
        module = analysis['file']
        
        for imp in analysis['imports']:
            # Only track internal imports
            if 'ign_lidar' in imp.get('module', ''):
                target = imp['module']
                coupling[module]['imports'].append(target)
                coupling[target]['imported_by'].append(module)
    
    return dict(coupling)


def find_oversized_modules(analyses: List[Dict], threshold: int = 800) -> List[Dict]:
    """Find modules that exceed size threshold."""
    oversized = []
    
    for analysis in analyses:
        if not analysis:
            continue
        
        if analysis['lines'] > threshold:
            oversized.append({
                'file': analysis['file'],
                'lines': analysis['lines'],
                'functions': len(analysis['functions']),
                'classes': len(analysis['classes']),
                'over_by': analysis['lines'] - threshold
            })
    
    return sorted(oversized, key=lambda x: x['lines'], reverse=True)


def calculate_complexity_metrics(analyses: List[Dict]) -> Dict:
    """Calculate various complexity metrics."""
    total_lines = sum(a['lines'] for a in analyses if a)
    total_functions = sum(len(a['functions']) for a in analyses if a)
    total_classes = sum(len(a['classes']) for a in analyses if a)
    total_files = len([a for a in analyses if a])
    
    # Find files with most imports
    import_counts = []
    for analysis in analyses:
        if not analysis:
            continue
        import_counts.append({
            'file': analysis['file'],
            'import_count': len(analysis['imports'])
        })
    
    import_counts.sort(key=lambda x: x['import_count'], reverse=True)
    
    return {
        'total_files': total_files,
        'total_lines': total_lines,
        'total_functions': total_functions,
        'total_classes': total_classes,
        'avg_lines_per_file': total_lines / total_files if total_files > 0 else 0,
        'avg_functions_per_file': total_functions / total_files if total_files > 0 else 0,
        'most_imports': import_counts[:10]
    }


def generate_report(analyses: List[Dict], output_file: str = None):
    """Generate comprehensive analysis report."""
    print("=" * 80)
    print("IGN LiDAR HD - Code Duplication Analysis Report")
    print("=" * 80)
    print()
    
    # 1. Complexity Metrics
    print("📊 COMPLEXITY METRICS")
    print("-" * 80)
    metrics = calculate_complexity_metrics(analyses)
    print(f"Total Files:         {metrics['total_files']}")
    print(f"Total Lines:         {metrics['total_lines']:,}")
    print(f"Total Functions:     {metrics['total_functions']}")
    print(f"Total Classes:       {metrics['total_classes']}")
    print(f"Avg Lines/File:      {metrics['avg_lines_per_file']:.1f}")
    print(f"Avg Functions/File:  {metrics['avg_functions_per_file']:.1f}")
    print()
    
    # 2. Duplicate Functions
    print("🔄 DUPLICATE FUNCTIONS")
    print("-" * 80)
    duplicates = find_duplicate_functions(analyses)
    
    high_priority = []
    for name, locations in duplicates.items():
        if len(locations) >= 3:  # 3+ implementations
            high_priority.append((name, locations))
    
    print(f"Total duplicate function names: {len(duplicates)}")
    print(f"High priority (3+ locations):   {len(high_priority)}")
    print()
    
    if high_priority:
        print("Top 10 Most Duplicated Functions:")
        for name, locations in sorted(high_priority, key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"  • {name} ({len(locations)} locations):")
            for loc in locations[:3]:  # Show first 3
                try:
                    file_short = Path(loc['file']).relative_to(Path.cwd())
                except (ValueError, TypeError):
                    file_short = Path(loc['file']).name
                print(f"    - {file_short}:{loc['line']} (args: {', '.join(loc['args'])})")
            if len(locations) > 3:
                print(f"    ... and {len(locations) - 3} more")
    print()
    
    # 3. Oversized Modules
    print("📏 OVERSIZED MODULES (>800 LOC)")
    print("-" * 80)
    oversized = find_oversized_modules(analyses, threshold=800)
    print(f"Modules exceeding threshold: {len(oversized)}")
    print()
    
    if oversized:
        print("Largest Modules:")
        for i, mod in enumerate(oversized[:10], 1):
            try:
                file_short = Path(mod['file']).relative_to(Path.cwd())
            except (ValueError, TypeError):
                file_short = Path(mod['file']).name
            print(f"  {i}. {file_short}")
            print(f"     Lines: {mod['lines']:,} (+{mod['over_by']} over limit)")
            print(f"     Functions: {mod['functions']}, Classes: {mod['classes']}")
    print()
    
    # 4. Module Coupling
    print("🔗 MODULE COUPLING")
    print("-" * 80)
    coupling = analyze_module_coupling(analyses)
    
    # Find highly coupled modules
    high_coupling = []
    for module, data in coupling.items():
        total_coupling = len(data['imports']) + len(data['imported_by'])
        if total_coupling > 10:
            high_coupling.append((module, data, total_coupling))
    
    high_coupling.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Highly coupled modules (>10 connections): {len(high_coupling)}")
    print()
    
    if high_coupling:
        print("Top 5 Most Coupled Modules:")
        for module, data, total in high_coupling[:5]:
            file_short = Path(module).relative_to(Path.cwd())
            print(f"  • {file_short}")
            print(f"    Imports: {len(data['imports'])}, Imported by: {len(data['imported_by'])}")
    print()
    
    # 5. Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    if len(high_priority) > 5:
        recommendations.append(
            f"🔴 CRITICAL: {len(high_priority)} functions have 3+ implementations. "
            "Create unified implementations in core modules."
        )
    
    if len(oversized) > 3:
        recommendations.append(
            f"🟡 MODERATE: {len(oversized)} modules exceed 800 LOC. "
            "Consider splitting into smaller, focused modules."
        )
    
    if len(high_coupling) > 5:
        recommendations.append(
            f"🟡 MODERATE: {len(high_coupling)} modules are highly coupled. "
            "Review dependencies and consider decoupling."
        )
    
    if not recommendations:
        recommendations.append("✅ Code structure looks good!")
    
    for rec in recommendations:
        print(f"  {rec}")
    print()
    
    # 6. Save detailed report
    if output_file:
        report_data = {
            'metrics': metrics,
            'duplicates': {name: locs for name, locs in duplicates.items()},
            'oversized': oversized,
            'coupling': {str(k): v for k, v in coupling.items()},
            'recommendations': recommendations
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Detailed report saved to: {output_file}")
        print()
    
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze code duplication in IGN LiDAR HD package'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON report file',
        default=None
    )
    parser.add_argument(
        '--module', '-m',
        help='Analyze specific module only (e.g., features, core)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Find all Python files
    base_path = Path('ign_lidar')
    
    if args.module:
        pattern = f"{args.module}/**/*.py"
        base_path = base_path / args.module
    else:
        pattern = "**/*.py"
    
    python_files = list(base_path.glob(pattern))
    
    print(f"🔍 Analyzing {len(python_files)} Python files in {base_path}...")
    print()
    
    # Analyze all files
    analyses = []
    for filepath in python_files:
        if '__pycache__' in str(filepath):
            continue
        
        analysis = analyze_file(filepath)
        if analysis:
            analyses.append(analysis)
    
    # Generate report
    generate_report(analyses, output_file=args.output)


if __name__ == '__main__':
    main()
