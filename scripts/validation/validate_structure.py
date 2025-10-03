#!/usr/bin/env python3
"""
Verify Directory Structure and Metadata Preservation

This script validates that the enriched output directory:
1. Preserves the same directory structure as input
2. Contains all metadata files (.json, .txt)
3. Has enriched LAZ files in the correct locations
"""

import sys
from pathlib import Path
from collections import defaultdict


def validate_structure(input_dir: Path, output_dir: Path):
    """Validate that output preserves input structure."""
    
    print(f"\n{'='*70}")
    print("DIRECTORY STRUCTURE VALIDATION")
    print(f"{'='*70}\n")
    
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Get all files
    input_laz = sorted(input_dir.rglob("*.laz"))
    input_json = sorted(input_dir.rglob("*.json"))
    input_txt = sorted(input_dir.glob("*.txt"))
    
    output_laz = sorted(output_dir.rglob("*.laz"))
    output_json = sorted(output_dir.rglob("*.json"))
    output_txt = sorted(output_dir.glob("*.txt"))
    
    # Statistics
    print(f"📊 File Counts:")
    print(f"  Input LAZ:    {len(input_laz)}")
    print(f"  Output LAZ:   {len(output_laz)}")
    print(f"  Input JSON:   {len(input_json)}")
    print(f"  Output JSON:  {len(output_json)}")
    print(f"  Input TXT:    {len(input_txt)}")
    print(f"  Output TXT:   {len(output_txt)}")
    print()
    
    # Check directory structure preservation
    print(f"📁 Directory Structure:")
    input_dirs = set()
    output_dirs = set()
    
    for laz in input_laz:
        rel_dir = laz.parent.relative_to(input_dir)
        if str(rel_dir) != '.':
            input_dirs.add(str(rel_dir))
    
    for laz in output_laz:
        rel_dir = laz.parent.relative_to(output_dir)
        if str(rel_dir) != '.':
            output_dirs.add(str(rel_dir))
    
    # Sort for display
    input_dirs = sorted(input_dirs)
    output_dirs = sorted(output_dirs)
    
    print(f"  Input subdirectories:  {len(input_dirs)}")
    for d in input_dirs:
        status = "✓" if d in output_dirs else "✗"
        print(f"    {status} {d}")
    
    missing_dirs = set(input_dirs) - set(output_dirs)
    extra_dirs = set(output_dirs) - set(input_dirs)
    
    if missing_dirs:
        print(f"\n  ⚠️  Missing directories in output:")
        for d in missing_dirs:
            print(f"    - {d}")
    
    if extra_dirs:
        print(f"\n  ℹ️  Extra directories in output:")
        for d in extra_dirs:
            print(f"    - {d}")
    
    print()
    
    # Check metadata preservation
    print(f"📄 Metadata Files:")
    
    # Root-level metadata
    root_meta_input = {f.name for f in input_json if f.parent == input_dir}
    root_meta_input.update({f.name for f in input_txt if f.parent == input_dir})
    
    root_meta_output = {f.name for f in output_json if f.parent == output_dir}
    root_meta_output.update({f.name for f in output_txt if f.parent == output_dir})
    
    print(f"  Root-level metadata files:")
    for meta in sorted(root_meta_input):
        status = "✓" if meta in root_meta_output else "✗"
        print(f"    {status} {meta}")
    
    # Per-file metadata
    input_laz_names = {laz.stem for laz in input_laz}
    input_json_names = {js.stem for js in input_json}
    output_json_names = {js.stem for js in output_json}
    
    laz_with_metadata = input_laz_names & input_json_names
    metadata_preserved = laz_with_metadata & output_json_names
    metadata_missing = laz_with_metadata - output_json_names
    
    print(f"\n  Per-file JSON metadata:")
    print(f"    Input LAZ files with JSON:  {len(laz_with_metadata)}")
    print(f"    Preserved in output:        {len(metadata_preserved)} "
          f"({100*len(metadata_preserved)/max(1,len(laz_with_metadata)):.1f}%)")
    
    if metadata_missing:
        print(f"    ⚠️  Missing metadata for {len(metadata_missing)} files")
        if len(metadata_missing) <= 10:
            for name in sorted(metadata_missing):
                print(f"      - {name}.json")
        else:
            for name in sorted(list(metadata_missing)[:5]):
                print(f"      - {name}.json")
            print(f"      ... and {len(metadata_missing)-5} more")
    
    print()
    
    # Check LAZ files
    print(f"🗃️  LAZ Files:")
    input_laz_rel = {laz.relative_to(input_dir) for laz in input_laz}
    output_laz_rel = {laz.relative_to(output_dir) for laz in output_laz}
    
    processed = input_laz_rel & output_laz_rel
    pending = input_laz_rel - output_laz_rel
    
    print(f"  Total input files:     {len(input_laz)}")
    print(f"  Processed (enriched):  {len(processed)} "
          f"({100*len(processed)/max(1,len(input_laz)):.1f}%)")
    print(f"  Pending:               {len(pending)}")
    
    if pending and len(pending) <= 10:
        print(f"\n  Pending files:")
        for p in sorted(pending):
            print(f"    - {p}")
    
    print()
    
    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    issues = []
    
    if len(output_laz) < len(input_laz):
        issues.append(
            f"⚠️  Only {len(output_laz)}/{len(input_laz)} LAZ files processed"
        )
    
    if missing_dirs:
        issues.append(
            f"⚠️  {len(missing_dirs)} directories missing in output"
        )
    
    if metadata_missing:
        issues.append(
            f"⚠️  {len(metadata_missing)} JSON metadata files not copied"
        )
    
    if len(root_meta_output) < len(root_meta_input):
        missing = len(root_meta_input) - len(root_meta_output)
        issues.append(
            f"⚠️  {missing} root-level metadata files not copied"
        )
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ All checks passed!")
        print("  • Directory structure preserved")
        print("  • All metadata files copied")
        print("  • All LAZ files processed")
    
    print()
    
    return len(issues) == 0


def main():
    if len(sys.argv) != 3:
        print("Usage: python validate_structure.py <input_dir> <output_dir>")
        print("\nExample:")
        print("  python validate_structure.py \\")
        print("    /mnt/c/Users/Simon/ign/raw_tiles \\")
        print("    /mnt/c/Users/Simon/ign/pre_tiles")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    success = validate_structure(input_dir, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
