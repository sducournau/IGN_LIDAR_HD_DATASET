#!/usr/bin/env python3
"""
Docusaurus Translation Status Checker
Compares English and French documentation files to identify translation needs.

Usage:
    python check_translation_status.py [--detailed] [--json]
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

class TranslationChecker:
    def __init__(self, docs_dir: str = "docs", i18n_dir: str = "i18n/fr/docusaurus-plugin-content-docs/current"):
        self.docs_dir = docs_dir
        self.i18n_dir = i18n_dir
        
    def get_all_md_files(self, base_path: str) -> List[str]:
        """Get all markdown files recursively"""
        md_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.md'):
                    rel_path = os.path.relpath(os.path.join(root, file), base_path)
                    md_files.append(rel_path)
        return sorted(md_files)
    
    def analyze_file(self, rel_path: str) -> Dict:
        """Analyze a single file pair"""
        en_path = os.path.join(self.docs_dir, rel_path)
        fr_path = os.path.join(self.i18n_dir, rel_path)
        
        result = {
            'path': rel_path,
            'en_exists': os.path.exists(en_path),
            'fr_exists': os.path.exists(fr_path),
            'status': 'unknown'
        }
        
        if not result['en_exists']:
            result['status'] = 'en_missing'
            return result
            
        if not result['fr_exists']:
            result['status'] = 'fr_missing'
            return result
        
        # Read file stats
        en_stat = os.stat(en_path)
        fr_stat = os.stat(fr_path)
        
        with open(en_path, 'r', encoding='utf-8') as f:
            en_content = f.read()
            result['en_lines'] = len(en_content.split('\n'))
            result['en_size'] = len(en_content)
            result['en_mtime'] = datetime.fromtimestamp(en_stat.st_mtime)
        
        with open(fr_path, 'r', encoding='utf-8') as f:
            fr_content = f.read()
            result['fr_lines'] = len(fr_content.split('\n'))
            result['fr_size'] = len(fr_content)
            result['fr_mtime'] = datetime.fromtimestamp(fr_stat.st_mtime)
            
            # Check for placeholder markers
            if any(marker in fr_content for marker in ['✅ TRADUCTION COMPLÈTE', 'placeholder', 'à traduire', 'needs translation']):
                result['has_placeholder'] = True
            else:
                result['has_placeholder'] = False
        
        # Calculate time difference in hours
        time_diff = (result['en_mtime'] - result['fr_mtime']).total_seconds() / 3600
        result['time_diff_hours'] = time_diff
        result['time_diff_days'] = int(time_diff / 24)
        
        # Calculate size difference
        line_diff = abs(result['en_lines'] - result['fr_lines'])
        result['line_diff'] = line_diff
        result['line_diff_pct'] = (line_diff / max(result['en_lines'], 1)) * 100
        
        # Determine status
        if result['has_placeholder']:
            result['status'] = 'placeholder'
        elif time_diff > 24:  # EN is more than 1 day newer
            result['status'] = 'en_newer'
        elif time_diff < -24:  # FR is more than 1 day newer
            result['status'] = 'fr_newer'
        else:
            result['status'] = 'up_to_date'
        
        # Flag significant size differences
        if line_diff > 50 and result['line_diff_pct'] > 20:
            result['significant_diff'] = True
        else:
            result['significant_diff'] = False
        
        return result
    
    def analyze_all(self) -> Dict:
        """Analyze all files and return statistics"""
        en_files = self.get_all_md_files(self.docs_dir)
        
        results = []
        stats = {
            'total': len(en_files),
            'up_to_date': 0,
            'en_newer': 0,
            'fr_newer': 0,
            'fr_missing': 0,
            'placeholder': 0,
            'significant_diff': 0
        }
        
        for rel_path in en_files:
            result = self.analyze_file(rel_path)
            results.append(result)
            
            # Update stats
            status = result['status']
            if status in stats:
                stats[status] += 1
            
            if result.get('significant_diff', False):
                stats['significant_diff'] += 1
        
        # Calculate completion rate
        good_statuses = ['up_to_date', 'fr_newer']
        completed = sum(stats.get(s, 0) for s in good_statuses)
        stats['completion_rate'] = (completed / stats['total'] * 100) if stats['total'] > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'files': results
        }
    
    def print_report(self, data: Dict, detailed: bool = False):
        """Print human-readable report"""
        stats = data['statistics']
        
        print("=" * 80)
        print("DOCUSAURUS FRENCH TRANSLATION STATUS REPORT")
        print("=" * 80)
        print(f"Generated: {data['timestamp']}")
        print()
        
        print("📊 SUMMARY STATISTICS")
        print("-" * 80)
        print(f"Total English files:        {stats['total']}")
        print(f"Up-to-date (±1 day):        {stats['up_to_date']} ✅")
        print(f"EN newer (>1 day):          {stats['en_newer']} ⚠️")
        print(f"FR newer (>1 day):          {stats['fr_newer']} ℹ️")
        print(f"Missing French translation: {stats['fr_missing']} ❌")
        print(f"Placeholder files:          {stats['placeholder']} 📝")
        print(f"Significant size diff:      {stats['significant_diff']} 📏")
        print(f"")
        print(f"📈 Completion Rate: {stats['completion_rate']:.1f}%")
        print()
        
        # Show files needing attention
        files = data['files']
        
        critical = [f for f in files if f['status'] in ['fr_missing', 'placeholder']]
        if critical:
            print("🔴 CRITICAL - Files Needing Immediate Attention")
            print("-" * 80)
            for f in critical:
                print(f"   {f['path']}")
                print(f"      Status: {f['status']}")
            print()
        
        needs_update = [f for f in files if f['status'] == 'en_newer']
        if needs_update:
            print("⚠️  WARNING - English Files Are Newer")
            print("-" * 80)
            for f in needs_update[:10]:  # Show first 10
                print(f"   {f['path']}")
                print(f"      EN: {f['en_lines']:4d} lines, {f['en_mtime'].strftime('%Y-%m-%d')}")
                print(f"      FR: {f['fr_lines']:4d} lines, {f['fr_mtime'].strftime('%Y-%m-%d')}")
                print(f"      ⏱️  {f['time_diff_days']} day(s) difference")
            print()
        
        size_diff = [f for f in files if f.get('significant_diff', False)]
        if size_diff and detailed:
            print("📏 SIGNIFICANT SIZE DIFFERENCES")
            print("-" * 80)
            for f in sorted(size_diff, key=lambda x: x['line_diff_pct'], reverse=True)[:15]:
                indicator = "⬆️ FR" if f['fr_lines'] > f['en_lines'] else "⬇️ FR"
                print(f"   {indicator} {f['path']}")
                print(f"      EN: {f['en_lines']:4d} | FR: {f['fr_lines']:4d} | "
                      f"Diff: {f['line_diff']:4d} lines ({f['line_diff_pct']:.0f}%)")
            print()
        
        print("=" * 80)
        
        # Recommendations
        if stats['fr_missing'] > 0 or stats['placeholder'] > 0:
            print("\n🎯 RECOMMENDED ACTIONS:")
            print("   1. Translate missing files immediately")
            print("   2. Replace placeholder files with complete translations")
        elif stats['en_newer'] > 0:
            print("\n🎯 RECOMMENDED ACTIONS:")
            print("   1. Review and update files where EN is newer")
            print("   2. Check if updates are minor (dates) or substantial (content)")
        else:
            print("\n✅ ALL FILES ARE UP-TO-DATE!")
            print("   Translation is in excellent condition.")
        
        if stats['significant_diff'] > 5:
            print(f"\nℹ️  NOTE: {stats['significant_diff']} files have significant size differences.")
            print("   This may indicate:")
            print("   - FR has extra explanations (good!)")
            print("   - FR is missing content (needs review)")
            print("   - EN has been expanded (FR needs update)")
            print("   Run with --detailed flag to see full list.")
        
        print()

def main():
    parser = argparse.ArgumentParser(description='Check French translation status')
    parser.add_argument('--detailed', action='store_true', help='Show detailed size differences')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--output', type=str, help='Save report to file')
    
    args = parser.parse_args()
    
    checker = TranslationChecker()
    data = checker.analyze_all()
    
    if args.json:
        output = json.dumps(data, indent=2, default=str)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)
    else:
        checker.print_report(data, detailed=args.detailed)
        
        if args.output:
            # Save human-readable report
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            checker.print_report(data, detailed=args.detailed)
            report = sys.stdout.getvalue()
            
            sys.stdout = old_stdout
            
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")

if __name__ == '__main__':
    main()
