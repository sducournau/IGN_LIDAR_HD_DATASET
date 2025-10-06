#!/usr/bin/env python3
"""
Sync French translations with English documentation.
This script identifies files that need updating and can optionally update them.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import difflib

# Base paths
DOCS_EN = Path("/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/docs")
DOCS_FR = Path("/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/i18n/fr/docusaurus-plugin-content-docs/current")

def get_file_info(file_path):
    """Get file size and modification time."""
    if file_path.exists():
        stat = file_path.stat()
        return {
            'size': stat.st_size,
            'mtime': datetime.fromtimestamp(stat.st_mtime),
            'exists': True
        }
    return {'exists': False}

def read_file(file_path):
    """Read file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def compare_files(en_path, fr_path):
    """Compare English and French files to show differences."""
    en_content = read_file(en_path)
    fr_content = read_file(fr_path)
    
    if not en_content or not fr_content:
        return None
    
    en_lines = en_content.splitlines()
    fr_lines = fr_content.splitlines()
    
    # Generate unified diff
    diff = difflib.unified_diff(
        fr_lines,
        en_lines,
        fromfile=f'FR: {fr_path.name}',
        tofile=f'EN: {en_path.name}',
        lineterm='',
        n=2
    )
    
    return list(diff)

def get_outdated_files():
    """Get list of files where English is newer than French."""
    outdated = []
    
    for en_path in DOCS_EN.rglob('*.md'):
        rel_path = en_path.relative_to(DOCS_EN)
        fr_path = DOCS_FR / rel_path
        
        if fr_path.exists():
            en_info = get_file_info(en_path)
            fr_info = get_file_info(fr_path)
            
            if en_info['mtime'] > fr_info['mtime']:
                time_diff = en_info['mtime'] - fr_info['mtime']
                outdated.append({
                    'rel_path': rel_path,
                    'en_path': en_path,
                    'fr_path': fr_path,
                    'en_mtime': en_info['mtime'],
                    'fr_mtime': fr_info['mtime'],
                    'diff_hours': time_diff.total_seconds() / 3600,
                    'en_size': en_info['size'],
                    'fr_size': fr_info['size']
                })
    
    return sorted(outdated, key=lambda x: x['diff_hours'], reverse=True)

def show_file_diff(item):
    """Show differences for a specific file."""
    print(f"\n{'=' * 80}")
    print(f"File: {item['rel_path']}")
    print(f"{'=' * 80}")
    print(f"EN modified: {item['en_mtime']} ({item['en_size']:,} bytes)")
    print(f"FR modified: {item['fr_mtime']} ({item['fr_size']:,} bytes)")
    print(f"Time diff: {item['diff_hours']:.1f} hours")
    
    diff = compare_files(item['en_path'], item['fr_path'])
    
    if diff:
        print(f"\n{'─' * 80}")
        print("DIFF (showing up to 50 lines):")
        print(f"{'─' * 80}")
        for i, line in enumerate(diff[:50]):
            print(line)
        if len(diff) > 50:
            print(f"\n... and {len(diff) - 50} more lines")

def main():
    print("=" * 80)
    print("FRENCH TRANSLATION SYNCHRONIZATION")
    print("=" * 80)
    print()
    
    outdated = get_outdated_files()
    
    if not outdated:
        print("✅ All French translations are up to date!")
        return
    
    print(f"Found {len(outdated)} files that need updating:\n")
    
    # Categorize by urgency (time difference)
    critical = [f for f in outdated if f['diff_hours'] >= 24]  # 1+ days old
    moderate = [f for f in outdated if 1 <= f['diff_hours'] < 24]  # 1-24 hours
    minor = [f for f in outdated if f['diff_hours'] < 1]  # < 1 hour
    
    print(f"🔴 CRITICAL (1+ days old): {len(critical)} files")
    for item in critical:
        print(f"   • {item['rel_path']} ({item['diff_hours']/24:.1f} days)")
    
    print(f"\n🟡 MODERATE (1-24 hours old): {len(moderate)} files")
    for item in moderate:
        print(f"   • {item['rel_path']} ({item['diff_hours']:.1f} hours)")
    
    print(f"\n🟢 MINOR (< 1 hour old): {len(minor)} files")
    for item in minor:
        print(f"   • {item['rel_path']} ({item['diff_hours']*60:.0f} minutes)")
    
    # Show detailed info for critical files
    if critical:
        print("\n" + "=" * 80)
        print("CRITICAL FILES - DETAILED ANALYSIS")
        print("=" * 80)
        
        for item in critical[:5]:  # Show first 5 critical
            show_file_diff(item)
    
    # Priority update list
    print("\n" + "=" * 80)
    print("📋 PRIORITY UPDATE LIST")
    print("=" * 80)
    print("\nFiles to update (ordered by priority):\n")
    
    for i, item in enumerate(outdated, 1):
        urgency = "🔴" if item['diff_hours'] >= 24 else "🟡" if item['diff_hours'] >= 1 else "🟢"
        print(f"{i:2d}. {urgency} {item['rel_path']}")
        print(f"     Time: {item['diff_hours']/24:.1f} days | Size diff: {item['en_size'] - item['fr_size']:+,} bytes")

if __name__ == "__main__":
    main()
