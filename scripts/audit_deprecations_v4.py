#!/usr/bin/env python3
"""
Deprecation Audit Script for v4.0.0

This script scans the entire codebase to identify all deprecated code that needs
to be removed for v4.0.0 release.

Usage:
    python scripts/audit_deprecations_v4.py
    python scripts/audit_deprecations_v4.py --output deprecation_report.json
"""

import ast
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import argparse


@dataclass
class DeprecationItem:
    """Information about a deprecated item"""
    file_path: str
    line_number: int
    item_type: str  # 'function', 'class', 'import', 'parameter', 'warning'
    item_name: str
    deprecation_message: str
    removal_version: str = "4.0.0"
    replacement: str = ""
    severity: str = "medium"  # 'low', 'medium', 'high'


class DeprecationAuditor:
    """Audits codebase for deprecated items"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.deprecations: List[DeprecationItem] = []
        
        # Patterns to search for
        self.patterns = {
            'deprecation_warning': re.compile(
                r'DeprecationWarning|warnings\.warn.*deprecated|@deprecated',
                re.IGNORECASE
            ),
            'will_be_removed': re.compile(
                r'will be removed in v?4\.0|removed in v?4\.0',
                re.IGNORECASE
            ),
            'deprecated_comment': re.compile(
                r'#.*DEPRECATED|#.*deprecated',
                re.IGNORECASE
            ),
            'v3_backward_compat': re.compile(
                r'backward compatibility|legacy|v3\.x support',
                re.IGNORECASE
            ),
        }
    
    def scan_file(self, file_path: Path) -> None:
        """Scan a Python file for deprecated items"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Scan line by line
            for i, line in enumerate(lines, 1):
                self._check_line(file_path, i, line, content)
            
            # Try AST parsing for deeper analysis
            try:
                tree = ast.parse(content)
                self._check_ast(file_path, tree, lines)
            except SyntaxError:
                pass  # Skip files with syntax errors
                
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
    
    def _check_line(self, file_path: Path, line_num: int, line: str, full_content: str) -> None:
        """Check a single line for deprecation markers"""
        relative_path = str(file_path.relative_to(self.root_dir))
        
        # Check for DeprecationWarning
        if self.patterns['deprecation_warning'].search(line):
            # Extract context
            context_lines = full_content.split('\n')[max(0, line_num-3):line_num+2]
            context = '\n'.join(context_lines)
            
            # Try to extract message
            message_match = re.search(r'["\']([^"\']*deprecated[^"\']*)["\']', line, re.IGNORECASE)
            message = message_match.group(1) if message_match else line.strip()
            
            self.deprecations.append(DeprecationItem(
                file_path=relative_path,
                line_number=line_num,
                item_type='warning',
                item_name='DeprecationWarning',
                deprecation_message=message[:200],
                severity='high'
            ))
        
        # Check for "will be removed in v4.0"
        if self.patterns['will_be_removed'].search(line):
            # Extract function/class name if possible
            name_match = re.search(r'(?:def|class)\s+(\w+)', line)
            item_name = name_match.group(1) if name_match else 'unknown'
            
            self.deprecations.append(DeprecationItem(
                file_path=relative_path,
                line_number=line_num,
                item_type='marked_for_removal',
                item_name=item_name,
                deprecation_message=line.strip()[:200],
                severity='high'
            ))
        
        # Check for deprecated comments
        if self.patterns['deprecated_comment'].search(line):
            # Try to get function/class name from surrounding context
            context = full_content.split('\n')[max(0, line_num-2):line_num+1]
            name_match = re.search(r'(?:def|class)\s+(\w+)', '\n'.join(context))
            item_name = name_match.group(1) if name_match else 'unknown'
            
            self.deprecations.append(DeprecationItem(
                file_path=relative_path,
                line_number=line_num,
                item_type='deprecated_comment',
                item_name=item_name,
                deprecation_message=line.strip()[:200],
                severity='medium'
            ))
        
        # Check for backward compatibility code
        if self.patterns['v3_backward_compat'].search(line):
            self.deprecations.append(DeprecationItem(
                file_path=relative_path,
                line_number=line_num,
                item_type='backward_compatibility',
                item_name='v3.x support',
                deprecation_message=line.strip()[:200],
                severity='medium'
            ))
    
    def _check_ast(self, file_path: Path, tree: ast.AST, lines: List[str]) -> None:
        """Use AST to find deprecated items"""
        relative_path = str(file_path.relative_to(self.root_dir))
        
        for node in ast.walk(tree):
            # Check for deprecated decorators
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and 'deprecat' in decorator.id.lower():
                        self.deprecations.append(DeprecationItem(
                            file_path=relative_path,
                            line_number=node.lineno,
                            item_type='function' if isinstance(node, ast.FunctionDef) else 'class',
                            item_name=node.name,
                            deprecation_message=f"@deprecated decorator found",
                            severity='high'
                        ))
    
    def scan_directory(self, directory: Path = None) -> None:
        """Scan all Python files in directory"""
        if directory is None:
            directory = self.root_dir
        
        # Scan ign_lidar/ directory
        ign_lidar_dir = directory / 'ign_lidar'
        if ign_lidar_dir.exists():
            for py_file in ign_lidar_dir.rglob('*.py'):
                if '__pycache__' not in str(py_file):
                    self.scan_file(py_file)
        
        # Also scan tests for deprecated usage
        tests_dir = directory / 'tests'
        if tests_dir.exists():
            for py_file in tests_dir.rglob('*.py'):
                if '__pycache__' not in str(py_file):
                    self.scan_file(py_file)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate deprecation report"""
        # Group by file
        by_file: Dict[str, List[DeprecationItem]] = {}
        for item in self.deprecations:
            if item.file_path not in by_file:
                by_file[item.file_path] = []
            by_file[item.file_path].append(item)
        
        # Group by severity
        by_severity = {
            'high': [d for d in self.deprecations if d.severity == 'high'],
            'medium': [d for d in self.deprecations if d.severity == 'medium'],
            'low': [d for d in self.deprecations if d.severity == 'low'],
        }
        
        # Group by type
        by_type: Dict[str, List[DeprecationItem]] = {}
        for item in self.deprecations:
            if item.item_type not in by_type:
                by_type[item.item_type] = []
            by_type[item.item_type].append(item)
        
        return {
            'total_deprecations': len(self.deprecations),
            'by_severity': {
                'high': len(by_severity['high']),
                'medium': len(by_severity['medium']),
                'low': len(by_severity['low']),
            },
            'by_type': {k: len(v) for k, v in by_type.items()},
            'by_file': {k: len(v) for k, v in by_file.items()},
            'details': {
                'by_file': {k: [asdict(d) for d in v] for k, v in by_file.items()},
                'by_severity': {k: [asdict(d) for d in v] for k, v in by_severity.items()},
                'by_type': {k: [asdict(d) for d in v] for k, v in by_type.items()},
            }
        }
    
    def print_summary(self) -> None:
        """Print summary to console"""
        report = self.generate_report()
        
        print("\n" + "="*80)
        print("DEPRECATION AUDIT REPORT - v4.0.0")
        print("="*80)
        print(f"\nTotal Deprecations Found: {report['total_deprecations']}")
        
        print("\n📊 By Severity:")
        for severity, count in report['by_severity'].items():
            emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[severity]
            print(f"  {emoji} {severity.upper()}: {count}")
        
        print("\n📋 By Type:")
        for item_type, count in sorted(report['by_type'].items(), key=lambda x: -x[1]):
            print(f"  • {item_type}: {count}")
        
        print(f"\n📁 Files Affected: {len(report['by_file'])}")
        
        print("\n🔴 HIGH PRIORITY Items (must be addressed for v4.0):")
        high_priority = report['details']['by_severity']['high']
        if high_priority:
            for i, item in enumerate(high_priority[:10], 1):  # Show first 10
                print(f"\n  {i}. {item['file_path']}:{item['line_number']}")
                print(f"     Type: {item['item_type']}, Item: {item['item_name']}")
                print(f"     Message: {item['deprecation_message'][:100]}...")
            
            if len(high_priority) > 10:
                print(f"\n  ... and {len(high_priority) - 10} more high priority items")
        else:
            print("  None found!")
        
        print("\n" + "="*80)
        print("Next Steps:")
        print("1. Review all HIGH priority items")
        print("2. Create issues for each deprecated module/function")
        print("3. Update V4_IMPLEMENTATION_CHECKLIST.md with findings")
        print("4. Begin removal in v4.0-dev branch")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Audit deprecated code for v4.0.0')
    parser.add_argument('--output', '-o', help='Output JSON file path', default=None)
    parser.add_argument('--root', '-r', help='Root directory', default='.')
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    print(f"🔍 Scanning {root_dir} for deprecated code...")
    
    auditor = DeprecationAuditor(root_dir)
    auditor.scan_directory(root_dir)
    
    # Print summary
    auditor.print_summary()
    
    # Save to JSON if requested
    if args.output:
        report = auditor.generate_report()
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Detailed report saved to: {output_path}")


if __name__ == '__main__':
    main()
