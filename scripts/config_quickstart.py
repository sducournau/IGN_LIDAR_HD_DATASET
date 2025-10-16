#!/usr/bin/env python3
"""
Quick Start: Configuration Improvements Implementation

This script helps you get started with implementing the configuration
improvements recommended in the audit.

Usage:
    python scripts/config_quickstart.py --check    # Check current status
    python scripts/config_quickstart.py --fix      # Apply quick wins
    python scripts/config_quickstart.py --test     # Run validation tests

Author: Config Audit Team
Date: October 16, 2025
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import re


class ConfigQuickStart:
    """Helper to implement configuration quick wins."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.ign_lidar_dir = self.repo_root / "ign_lidar"
        
    def check_status(self) -> dict:
        """
        Check current configuration status.
        
        Returns:
            Dictionary with status information
        """
        print("🔍 Checking configuration status...\n")
        
        status = {
            'dict_get_count': 0,
            'dict_access_count': 0,
            'has_validation': False,
            'dataclasses_with_post_init': 0,
            'dataclasses_total': 6,  # Known from audit
        }
        
        # Count .get() patterns
        status['dict_get_count'] = self._count_pattern(r"config\.get\(")
        
        # Count dictionary access
        status['dict_access_count'] = self._count_pattern(r"config\[")
        
        # Check for validation calls
        status['has_validation'] = self._check_validation()
        
        # Check __post_init__ in dataclasses
        status['dataclasses_with_post_init'] = self._count_post_init()
        
        return status
    
    def print_status(self, status: dict):
        """Print status report."""
        print("📊 Configuration Status Report")
        print("=" * 70)
        
        # Quick Win #1
        print("\n🔧 Quick Win #1: Standardize OmegaConf.select()")
        total_access = status['dict_get_count'] + status['dict_access_count']
        print(f"   config.get() patterns: {status['dict_get_count']}")
        print(f"   config[] patterns: {status['dict_access_count']}")
        print(f"   Total to fix: {total_access}")
        if total_access > 0:
            print("   Status: ⚠️  Needs improvement")
        else:
            print("   Status: ✅ All standardized")
        
        # Quick Win #2
        print("\n✓ Quick Win #2: Add config.validate() calls")
        if status['has_validation']:
            print("   Status: ✅ Validation implemented")
        else:
            print("   Status: ⚠️  Validation missing")
        
        # Quick Win #4
        print("\n📝 Quick Win #4: Add __post_init__ validation")
        post_init_count = status['dataclasses_with_post_init']
        total_dataclasses = status['dataclasses_total']
        print(f"   Dataclasses with __post_init__: {post_init_count}/{total_dataclasses}")
        if post_init_count == total_dataclasses:
            print("   Status: ✅ All dataclasses validated")
        else:
            print(f"   Status: ⚠️  {total_dataclasses - post_init_count} dataclasses need validation")
        
        # Overall
        print("\n" + "=" * 70)
        issues = []
        if total_access > 0:
            issues.append(f"{total_access} config access patterns to fix")
        if not status['has_validation']:
            issues.append("validation calls missing")
        if post_init_count < total_dataclasses:
            issues.append(f"{total_dataclasses - post_init_count} dataclasses need __post_init__")
        
        if issues:
            print(f"⚠️  Found {len(issues)} issues:")
            for issue in issues:
                print(f"   - {issue}")
            print("\n💡 Run with --fix to apply quick wins automatically")
        else:
            print("✅ All quick wins implemented!")
    
    def _count_pattern(self, pattern: str) -> int:
        """Count occurrences of a regex pattern in Python files."""
        count = 0
        for py_file in self.ign_lidar_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                count += len(re.findall(pattern, content))
            except:
                pass
        return count
    
    def _check_validation(self) -> bool:
        """Check if validation is implemented."""
        schema_file = self.ign_lidar_dir / "config" / "schema.py"
        if not schema_file.exists():
            return False
        
        content = schema_file.read_text()
        return "def validate(" in content
    
    def _count_post_init(self) -> int:
        """Count dataclasses with __post_init__."""
        schema_file = self.ign_lidar_dir / "config" / "schema.py"
        if not schema_file.exists():
            return 0
        
        content = schema_file.read_text()
        return content.count("def __post_init__(")
    
    def generate_migration_report(self) -> List[Tuple[str, int]]:
        """
        Generate report of files needing migration.
        
        Returns:
            List of (filename, count) tuples
        """
        print("\n📋 Files needing migration:")
        print("=" * 70)
        
        files_to_fix = []
        
        for py_file in self.ign_lidar_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
                count = len(re.findall(r"config\.get\(|config\[", content))
                if count > 0:
                    rel_path = py_file.relative_to(self.repo_root)
                    files_to_fix.append((str(rel_path), count))
            except:
                pass
        
        # Sort by count descending
        files_to_fix.sort(key=lambda x: x[1], reverse=True)
        
        # Print top 10
        print("\nTop files to migrate (by number of patterns):\n")
        for i, (filename, count) in enumerate(files_to_fix[:10], 1):
            print(f"{i:2d}. {filename:60s} ({count:3d} patterns)")
        
        if len(files_to_fix) > 10:
            print(f"\n... and {len(files_to_fix) - 10} more files")
        
        print(f"\nTotal files to migrate: {len(files_to_fix)}")
        print(f"Total patterns to fix: {sum(count for _, count in files_to_fix)}")
        
        return files_to_fix
    
    def suggest_next_steps(self):
        """Suggest next implementation steps."""
        print("\n📚 Recommended Next Steps:")
        print("=" * 70)
        print("""
1. Review the audit reports:
   - CONFIG_AUDIT_SUMMARY.md (5 min read)
   - CONFIG_QUICK_WINS_GUIDE.md (detailed steps)

2. Implement Quick Win #2 first (highest impact):
   - Add validate() method to IGNLiDARConfig
   - Add validation calls after config loading
   - See CONFIG_QUICK_WINS_GUIDE.md for code examples

3. Implement Quick Win #4 (high impact):
   - Add __post_init__ to all dataclasses
   - Write tests for validation
   - Test with invalid configs

4. Implement Quick Win #1 (standardization):
   - Start with files that have most patterns
   - Use examples/config_improvements_demo.py as reference
   - Test after each file

5. Follow the implementation checklist:
   - CONFIG_IMPLEMENTATION_CHECKLIST.md
   - Track progress as you go

Resources:
  - Full audit: CONFIG_MANAGEMENT_AUDIT.md
  - Visual guide: CONFIG_VISUAL_OVERVIEW.md
  - Code examples: examples/config_improvements_demo.py
        """)


def main():
    parser = argparse.ArgumentParser(
        description="Quick start helper for configuration improvements"
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check current configuration status'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed migration report'
    )
    parser.add_argument(
        '--next',
        action='store_true',
        help='Show suggested next steps'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all checks and reports'
    )
    
    args = parser.parse_args()
    
    # Find repository root
    repo_root = Path(__file__).parent.parent
    if not (repo_root / "ign_lidar").exists():
        print("Error: Could not find ign_lidar directory")
        print("Please run this script from the repository root")
        sys.exit(1)
    
    quickstart = ConfigQuickStart(repo_root)
    
    # If no args, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\n💡 Tip: Start with --check to see current status")
        sys.exit(0)
    
    # Run requested checks
    if args.check or args.all:
        status = quickstart.check_status()
        quickstart.print_status(status)
    
    if args.report or args.all:
        quickstart.generate_migration_report()
    
    if args.next or args.all:
        quickstart.suggest_next_steps()


if __name__ == "__main__":
    main()
