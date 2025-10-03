#!/usr/bin/env python3
"""
Verify RGB enrichment is working correctly in LAZ files.

This script checks that:
1. RGB fields exist
2. RGB values are in correct range (16-bit)
3. Point format supports native RGB
4. Colors are diverse (not all same color)

Usage:
    python verify_rgb_enrichment.py /path/to/enriched_file.laz
"""

import sys
from pathlib import Path
import numpy as np


def verify_rgb_enrichment(laz_file: Path) -> bool:
    """
    Verify that RGB enrichment was successful.
    
    Args:
        laz_file: Path to enriched LAZ file
        
    Returns:
        True if RGB is valid, False otherwise
    """
    try:
        import laspy
    except ImportError:
        print("❌ ERROR: laspy not installed")
        print("   Install with: pip install laspy")
        return False
    
    print(f"\n{'='*70}")
    print(f"RGB ENRICHMENT VERIFICATION")
    print(f"{'='*70}\n")
    print(f"📂 File: {laz_file}")
    
    # Read file
    try:
        las = laspy.read(str(laz_file))
        print(f"✅ File loaded: {len(las.points):,} points")
    except Exception as e:
        print(f"❌ ERROR: Cannot read file: {e}")
        return False
    
    # Check point format
    point_format = las.header.point_format.id
    print(f"📋 Point format: {point_format}")
    
    rgb_supported_formats = [2, 3, 5, 6, 7, 8, 10]
    if point_format in rgb_supported_formats:
        print(f"✅ Point format {point_format} supports native RGB")
    else:
        print(f"⚠️  Point format {point_format} does not support native RGB")
        print(f"   Supported formats: {rgb_supported_formats}")
        return False
    
    # Check RGB fields exist
    has_red = hasattr(las, 'red')
    has_green = hasattr(las, 'green')
    has_blue = hasattr(las, 'blue')
    
    print(f"\n🎨 RGB Fields:")
    print(f"   Red:   {'✅ Present' if has_red else '❌ Missing'}")
    print(f"   Green: {'✅ Present' if has_green else '❌ Missing'}")
    print(f"   Blue:  {'✅ Present' if has_blue else '❌ Missing'}")
    
    if not (has_red and has_green and has_blue):
        print(f"\n❌ ERROR: RGB fields are missing!")
        return False
    
    # Check RGB values
    red = np.array(las.red)
    green = np.array(las.green)
    blue = np.array(las.blue)
    
    print(f"\n📊 RGB Statistics:")
    print(f"   Red   - min: {red.min():5d}, max: {red.max():5d}, "
          f"mean: {red.mean():7.1f}")
    print(f"   Green - min: {green.min():5d}, max: {green.max():5d}, "
          f"mean: {green.mean():7.1f}")
    print(f"   Blue  - min: {blue.min():5d}, max: {blue.max():5d}, "
          f"mean: {blue.mean():7.1f}")
    
    # Check if RGB is 16-bit (0-65535)
    max_val = max(red.max(), green.max(), blue.max())
    if max_val > 65535:
        print(f"\n⚠️  WARNING: RGB values exceed 16-bit range!")
        print(f"   Max value: {max_val}")
        return False
    elif max_val <= 255:
        print(f"\n⚠️  WARNING: RGB values are in 8-bit range (0-255)")
        print(f"   Expected 16-bit range (0-65535)")
        print(f"   File may display but colors will be dark")
    else:
        print(f"\n✅ RGB values are in correct 16-bit range")
    
    # Check if all RGB values are zero
    if red.max() == 0 and green.max() == 0 and blue.max() == 0:
        print(f"\n❌ ERROR: All RGB values are zero!")
        print(f"   RGB augmentation failed or colors are all black")
        return False
    
    # Check color diversity
    num_unique_colors = len(np.unique(
        np.column_stack([red, green, blue]), axis=0
    ))
    color_diversity = (num_unique_colors / len(las.points)) * 100
    
    print(f"\n🌈 Color Diversity:")
    print(f"   Unique colors: {num_unique_colors:,}")
    print(f"   Diversity: {color_diversity:.2f}%")
    
    if color_diversity < 0.1:
        print(f"   ⚠️  Very low diversity - most points same color")
    elif color_diversity < 1:
        print(f"   ⚠️  Low diversity - limited color variation")
    else:
        print(f"   ✅ Good diversity - rich color variation")
    
    # Sample some RGB values
    print(f"\n📸 Sample Colors (first 5 points):")
    for i in range(min(5, len(las.points))):
        r, g, b = red[i], green[i], blue[i]
        r8 = int(r / 256)  # Convert to 8-bit for display
        g8 = int(g / 256)
        b8 = int(b / 256)
        print(f"   Point {i+1}: RGB({r:5d}, {g:5d}, {b:5d}) = "
              f"RGB8({r8:3d}, {g8:3d}, {b8:3d})")
    
    # Final verdict
    print(f"\n{'='*70}")
    if max_val > 255 and color_diversity > 0.1:
        print(f"✅ RGB ENRICHMENT SUCCESSFUL!")
        print(f"{'='*70}\n")
        print(f"🚀 CloudCompare Usage:")
        print(f"   1. Open file in CloudCompare")
        print(f"   2. Colors should display automatically")
        print(f"   3. If not: Edit → Colors → RGB")
        print(f"   4. Adjust: Display → Point Size (for better visibility)")
        print()
        return True
    else:
        print(f"⚠️  RGB ENRICHMENT NEEDS ATTENTION")
        print(f"{'='*70}\n")
        if max_val <= 255:
            print(f"Issue: RGB values are 8-bit instead of 16-bit")
            print(f"Solution: Re-run enrichment with latest code")
        if color_diversity < 0.1:
            print(f"Issue: Very low color diversity")
            print(f"Solution: Check if orthophoto fetch was successful")
        print()
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_rgb_enrichment.py <enriched_file.laz>")
        sys.exit(1)
    
    laz_file = Path(sys.argv[1])
    
    if not laz_file.exists():
        print(f"❌ ERROR: File not found: {laz_file}")
        sys.exit(1)
    
    if not laz_file.suffix.lower() in ['.laz', '.las']:
        print(f"⚠️  WARNING: File does not have .laz or .las extension")
    
    success = verify_rgb_enrichment(laz_file)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
