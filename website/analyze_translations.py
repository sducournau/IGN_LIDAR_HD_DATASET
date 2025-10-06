#!/usr/bin/env python3
"""
Comprehensive analysis of English vs French documentation.
Identifies missing files, outdated translations, and content differences.
"""

import os
from pathlib import Path
import hashlib
import re
from datetime import datetime

def get_file_hash(file_path):
    """Get MD5 hash of file content (excluding frontmatter dates)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove dates from frontmatter for comparison
            content = re.sub(r'date:.*?\n', '', content)
            content = re.sub(r'last_update:.*?\n', '', content)
            return hashlib.md5(content.encode()).hexdigest()
    except Exception as e:
        return None

def extract_title(file_path):
    """Extract title from frontmatter."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'^---\n.*?title:\s*["\']?(.*?)["\']?\n', content, re.MULTILINE | re.DOTALL)
            if match:
                return match.group(1).strip()
    except:
        pass
    return None

def has_translation_notice(file_path):
    """Check if file has translation notice marker."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return "🇫🇷 TRADUCTION" in content or "TRADUCTION FRANÇAISE" in content
    except:
        return False

def get_file_stats(file_path):
    """Get word count and other stats."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove code blocks
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            words = len(content.split())
            lines = len(content.split('\n'))
            return {'words': words, 'lines': lines}
    except:
        return {'words': 0, 'lines': 0}

def main():
    en_dir = Path("docs")
    fr_dir = Path("i18n/fr/docusaurus-plugin-content-docs/current")
    
    print("=" * 80)
    print("📚 DOCUSAURUS DOCUMENTATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Get all English files
    en_files = set()
    en_file_map = {}
    for en_file in en_dir.rglob("*.md"):
        rel_path = str(en_file.relative_to(en_dir))
        en_files.add(rel_path)
        en_file_map[rel_path] = en_file
    
    # Get all French files
    fr_files = set()
    fr_file_map = {}
    for fr_file in fr_dir.rglob("*.md"):
        rel_path = str(fr_file.relative_to(fr_dir))
        fr_files.add(rel_path)
        fr_file_map[rel_path] = fr_file
    
    print(f"📊 Total English files: {len(en_files)}")
    print(f"📊 Total French files: {len(fr_files)}")
    print()
    
    # Missing in French
    missing_in_fr = sorted(en_files - fr_files)
    if missing_in_fr:
        print(f"❌ MISSING IN FRENCH ({len(missing_in_fr)} files):")
        print("-" * 80)
        for file in missing_in_fr:
            en_path = en_file_map[file]
            title = extract_title(en_path) or "No title"
            stats = get_file_stats(en_path)
            print(f"   📄 {file}")
            print(f"      Title: {title}")
            print(f"      Size: {stats['words']} words, {stats['lines']} lines")
            print()
    
    # Extra in French (not in English - might be outdated)
    extra_in_fr = sorted(fr_files - en_files)
    if extra_in_fr:
        print(f"⚠️  EXTRA IN FRENCH (not in English) ({len(extra_in_fr)} files):")
        print("-" * 80)
        for file in extra_in_fr:
            print(f"   📄 {file}")
        print()
    
    # Files that exist in both - check translation status
    common_files = sorted(en_files & fr_files)
    print(f"🔍 COMPARING COMMON FILES ({len(common_files)} files):")
    print("-" * 80)
    print()
    
    needs_translation = []
    partial_translation = []
    translated = []
    
    for file in common_files:
        en_path = en_file_map[file]
        fr_path = fr_file_map[file]
        
        en_stats = get_file_stats(en_path)
        fr_stats = get_file_stats(fr_path)
        
        has_notice = has_translation_notice(fr_path)
        
        # Heuristic: if word counts are very similar, might be untranslated
        word_diff_ratio = abs(en_stats['words'] - fr_stats['words']) / max(en_stats['words'], 1)
        
        if word_diff_ratio < 0.05:  # Less than 5% difference
            needs_translation.append((file, en_stats, fr_stats, "Very similar word count - likely untranslated"))
        elif has_notice:
            partial_translation.append((file, en_stats, fr_stats, "Has translation notice"))
        else:
            # Check for English-heavy content
            with open(fr_path, 'r', encoding='utf-8') as f:
                fr_content = f.read().lower()
                english_words = ['processing', 'feature', 'configuration', 'example', 'usage', 'installation']
                french_words = ['traitement', 'fonctionnalité', 'configuration', 'exemple', 'utilisation', 'installation']
                
                en_count = sum(fr_content.count(word) for word in english_words)
                fr_count = sum(fr_content.count(word) for word in french_words)
                
                if en_count > fr_count * 1.5:
                    needs_translation.append((file, en_stats, fr_stats, "English content detected"))
                else:
                    translated.append((file, en_stats, fr_stats))
    
    # Print results
    if needs_translation:
        print(f"❌ NEEDS TRANSLATION ({len(needs_translation)} files):")
        print("-" * 80)
        for file, en_stats, fr_stats, reason in needs_translation:
            print(f"   📄 {file}")
            print(f"      Reason: {reason}")
            print(f"      EN: {en_stats['words']} words | FR: {fr_stats['words']} words")
            print()
    
    if partial_translation:
        print(f"🔄 PARTIAL TRANSLATION ({len(partial_translation)} files):")
        print("-" * 80)
        for file, en_stats, fr_stats, reason in partial_translation:
            print(f"   📄 {file}")
            print(f"      Reason: {reason}")
            print(f"      EN: {en_stats['words']} words | FR: {fr_stats['words']} words")
            print()
    
    print(f"✅ FULLY TRANSLATED ({len(translated)} files):")
    print("-" * 80)
    for file, en_stats, fr_stats in translated[:10]:  # Show first 10
        print(f"   📄 {file}")
    if len(translated) > 10:
        print(f"   ... and {len(translated) - 10} more")
    print()
    
    # Summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Missing in French:        {len(missing_in_fr)}")
    print(f"Needs Translation:        {len(needs_translation)}")
    print(f"Partial Translation:      {len(partial_translation)}")
    print(f"Fully Translated:         {len(translated)}")
    print(f"Extra in French:          {len(extra_in_fr)}")
    print()
    print(f"Total Translation Work:   {len(missing_in_fr) + len(needs_translation) + len(partial_translation)} files")
    print("=" * 80)
    
    # Create action items
    print()
    print("📝 ACTION ITEMS:")
    print("-" * 80)
    print()
    
    all_to_translate = missing_in_fr + [f for f, _, _, _ in needs_translation] + [f for f, _, _, _ in partial_translation]
    
    if all_to_translate:
        print("Files that need translation (priority order):")
        for i, file in enumerate(all_to_translate, 1):
            print(f"{i:2d}. {file}")
    
    print()

if __name__ == "__main__":
    main()
