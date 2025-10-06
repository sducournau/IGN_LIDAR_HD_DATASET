#!/usr/bin/env python3
"""
Compare English and French documentation to identify missing translations
and files that need updating.
"""

import os
from pathlib import Path
from datetime import datetime

# Base paths
DOCS_EN = Path("/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/docs")
DOCS_FR = Path("/mnt/d/Users/Simon/OneDrive/Documents/GitHub/IGN_LIDAR_HD_DATASET/website/i18n/fr/docusaurus-plugin-content-docs/current")

def get_all_md_files(base_path):
    """Get all markdown files relative to base path."""
    md_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.md'):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(base_path)
                md_files.append(rel_path)
    return sorted(md_files)

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

def main():
    print("=" * 80)
    print("DOCUSAURUS TRANSLATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Get all English files
    en_files = get_all_md_files(DOCS_EN)
    fr_files = get_all_md_files(DOCS_FR)
    
    print(f"📊 Statistics:")
    print(f"   English files: {len(en_files)}")
    print(f"   French files:  {len(fr_files)}")
    print()
    
    # Convert to sets for comparison
    en_set = set(en_files)
    fr_set = set(fr_files)
    
    # Files missing in French
    missing_in_fr = en_set - fr_set
    
    # Extra files in French (not in English)
    extra_in_fr = fr_set - en_set
    
    # Common files (for comparison)
    common_files = en_set & fr_set
    
    print("=" * 80)
    print("🚨 MISSING FRENCH TRANSLATIONS")
    print("=" * 80)
    if missing_in_fr:
        for file in sorted(missing_in_fr):
            en_path = DOCS_EN / file
            info = get_file_info(en_path)
            print(f"\n❌ {file}")
            print(f"   Size: {info['size']:,} bytes")
            print(f"   Modified: {info['mtime']}")
    else:
        print("✅ All English files have French translations!")
    
    print()
    print("=" * 80)
    print("📝 EXTRA FRENCH FILES (not in English)")
    print("=" * 80)
    if extra_in_fr:
        for file in sorted(extra_in_fr):
            fr_path = DOCS_FR / file
            info = get_file_info(fr_path)
            print(f"\n⚠️  {file}")
            print(f"   Size: {info['size']:,} bytes")
            print(f"   Modified: {info['mtime']}")
    else:
        print("✅ No extra French files")
    
    print()
    print("=" * 80)
    print("🔄 FILES POTENTIALLY NEEDING UPDATE")
    print("=" * 80)
    print("(English file modified more recently than French)")
    print()
    
    needs_update = []
    for file in sorted(common_files):
        en_path = DOCS_EN / file
        fr_path = DOCS_FR / file
        
        en_info = get_file_info(en_path)
        fr_info = get_file_info(fr_path)
        
        # Check if English is newer
        if en_info['mtime'] > fr_info['mtime']:
            time_diff = en_info['mtime'] - fr_info['mtime']
            needs_update.append({
                'file': file,
                'en_mtime': en_info['mtime'],
                'fr_mtime': fr_info['mtime'],
                'diff_days': time_diff.days,
                'en_size': en_info['size'],
                'fr_size': fr_info['size']
            })
    
    if needs_update:
        for item in sorted(needs_update, key=lambda x: x['diff_days'], reverse=True):
            print(f"\n⏰ {item['file']}")
            print(f"   EN modified: {item['en_mtime']} ({item['en_size']:,} bytes)")
            print(f"   FR modified: {item['fr_mtime']} ({item['fr_size']:,} bytes)")
            print(f"   Age diff: {item['diff_days']} days")
            
            # Size difference
            size_diff = item['en_size'] - item['fr_size']
            if abs(size_diff) > 100:
                print(f"   Size diff: {size_diff:+,} bytes")
    else:
        print("✅ All French translations are up to date!")
    
    print()
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"Total English files:       {len(en_files)}")
    print(f"Total French files:        {len(fr_files)}")
    print(f"Missing translations:      {len(missing_in_fr)}")
    print(f"Extra French files:        {len(extra_in_fr)}")
    print(f"Files needing update:      {len(needs_update)}")
    print(f"Up-to-date translations:   {len(common_files) - len(needs_update)}")
    print()
    
    # Generate action items
    if missing_in_fr or needs_update:
        print("=" * 80)
        print("✅ ACTION ITEMS")
        print("=" * 80)
        
        if missing_in_fr:
            print("\n1. CREATE MISSING TRANSLATIONS:")
            for file in sorted(missing_in_fr)[:5]:  # Show first 5
                print(f"   - {file}")
            if len(missing_in_fr) > 5:
                print(f"   ... and {len(missing_in_fr) - 5} more")
        
        if needs_update:
            print("\n2. UPDATE OUTDATED TRANSLATIONS:")
            for item in sorted(needs_update, key=lambda x: x['diff_days'], reverse=True)[:5]:
                print(f"   - {item['file']} ({item['diff_days']} days old)")
            if len(needs_update) > 5:
                print(f"   ... and {len(needs_update) - 5} more")
        
        print()

if __name__ == "__main__":
    main()
