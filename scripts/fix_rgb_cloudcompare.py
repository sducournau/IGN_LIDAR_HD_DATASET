"""
Fix RGB colors in LAZ files for CloudCompare compatibility.

This script diagnoses and fixes RGB color issues in enriched LAZ files.

CloudCompare expects:
- RGB values in 16-bit format (0-65535)
- Native RGB fields (not extra dimensions)
- Point format that supports RGB (2, 3, 5, 6, 7, 8, or 10)
"""

import sys
import logging
from pathlib import Path
import numpy as np

try:
    import laspy
except ImportError:
    print("ERROR: laspy not installed. Install with: pip install laspy")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_rgb(laz_file: Path):
    """
    Diagnose RGB color issues in LAZ file.
    
    Args:
        laz_file: Path to LAZ file to diagnose
    """
    logger.info(f"Diagnosing: {laz_file.name}")
    logger.info("=" * 60)
    
    # Load LAZ
    las = laspy.read(str(laz_file))
    
    # Check point format
    point_format = las.header.point_format.id
    logger.info(f"Point Format: {point_format}")
    
    # Check if format supports RGB
    rgb_formats = [2, 3, 5, 6, 7, 8, 10]
    if point_format in rgb_formats:
        logger.info("✓ Point format supports RGB natively")
    else:
        logger.warning(f"✗ Point format {point_format} does NOT support RGB natively")
        logger.warning("  CloudCompare may not display RGB colors")
    
    # Check for RGB fields
    has_red = hasattr(las, 'red') and las.red is not None
    has_green = hasattr(las, 'green') and las.green is not None
    has_blue = hasattr(las, 'blue') and las.blue is not None
    
    logger.info(f"Has red field: {has_red}")
    logger.info(f"Has green field: {has_green}")
    logger.info(f"Has blue field: {has_blue}")
    
    if not (has_red and has_green and has_blue):
        logger.error("✗ Missing RGB fields!")
        return False
    
    # Check RGB value ranges
    red_min, red_max = las.red.min(), las.red.max()
    green_min, green_max = las.green.min(), las.green.max()
    blue_min, blue_max = las.blue.min(), las.blue.max()
    
    logger.info(f"Red range:   {red_min} - {red_max}")
    logger.info(f"Green range: {green_min} - {green_max}")
    logger.info(f"Blue range:  {blue_min} - {blue_max}")
    
    # Check if values look correct
    max_value = max(red_max, green_max, blue_max)
    
    if max_value == 0:
        logger.error("✗ All RGB values are 0 (black)!")
        return False
    elif max_value <= 255:
        logger.warning("⚠ RGB values are 8-bit (0-255)")
        logger.warning("  Should be 16-bit (0-65535) for CloudCompare")
        logger.warning("  Needs conversion!")
        return False
    elif max_value < 65000:
        logger.warning(f"⚠ Max RGB value is {max_value} (expected ~65535)")
        logger.warning("  Values may be incorrectly scaled")
    else:
        logger.info(f"✓ RGB values are 16-bit ({max_value})")
    
    # Check RGB data type
    logger.info(f"Red dtype: {las.red.dtype}")
    
    if las.red.dtype != np.uint16:
        logger.warning(f"⚠ RGB dtype is {las.red.dtype}, should be uint16")
        return False
    
    # Sample some points to check distribution
    sample_size = min(100, len(las.red))
    sample_indices = np.random.choice(len(las.red), sample_size, replace=False)
    
    non_zero_count = np.sum(
        (las.red[sample_indices] > 0) |
        (las.green[sample_indices] > 0) |
        (las.blue[sample_indices] > 0)
    )
    
    logger.info(f"Sample: {non_zero_count}/{sample_size} points have non-zero RGB")
    
    if non_zero_count == 0:
        logger.error("✗ All sampled points are black!")
        return False
    elif non_zero_count < sample_size * 0.5:
        logger.warning(f"⚠ Only {non_zero_count}/{sample_size} points have color")
    else:
        logger.info("✓ RGB colors look good!")
    
    logger.info("=" * 60)
    return True


def fix_rgb(laz_file: Path, output_file: Path = None):
    """
    Fix RGB colors in LAZ file for CloudCompare compatibility.
    
    Args:
        laz_file: Path to input LAZ file
        output_file: Path to output LAZ file (default: overwrite input)
    """
    logger.info(f"Fixing RGB in: {laz_file.name}")
    
    # Load LAZ
    las = laspy.read(str(laz_file))
    
    # Check if needs fixing
    needs_fix = False
    
    # Check if RGB values are 8-bit (need conversion to 16-bit)
    if hasattr(las, 'red') and las.red is not None:
        max_value = max(las.red.max(), las.green.max(), las.blue.max())
        if max_value <= 255:
            logger.info("Converting 8-bit RGB to 16-bit...")
            # Convert 8-bit (0-255) to 16-bit (0-65535)
            # Use 257 multiplier to get full range: 255 * 257 = 65535
            las.red = (las.red.astype(np.uint16) * 257)
            las.green = (las.green.astype(np.uint16) * 257)
            las.blue = (las.blue.astype(np.uint16) * 257)
            needs_fix = True
            logger.info("✓ Converted to 16-bit RGB")
    
    # Check point format
    point_format = las.header.point_format.id
    rgb_formats = [2, 3, 5, 6, 7, 8, 10]
    
    if point_format not in rgb_formats:
        logger.info(f"Converting point format {point_format} to format 2 (with RGB)...")
        
        # Create new LAS with RGB-compatible format
        from laspy import LasHeader, LasData
        
        # Create header with format 2 (supports RGB)
        header = LasHeader(version=las.header.version, point_format=2)
        header.offsets = las.header.offsets
        header.scales = las.header.scales
        
        # Create new LAS data
        las_new = LasData(header)
        
        # Copy all fields
        las_new.x = las.x
        las_new.y = las.y
        las_new.z = las.z
        las_new.intensity = las.intensity
        las_new.return_number = las.return_number
        las_new.number_of_returns = las.number_of_returns
        las_new.scan_direction_flag = las.scan_direction_flag
        las_new.edge_of_flight_line = las.edge_of_flight_line
        las_new.classification = las.classification
        las_new.synthetic = las.synthetic
        las_new.key_point = las.key_point
        las_new.withheld = las.withheld
        las_new.scan_angle_rank = las.scan_angle_rank
        las_new.user_data = las.user_data
        las_new.point_source_id = las.point_source_id
        
        # Copy RGB if exists
        if hasattr(las, 'red'):
            las_new.red = las.red
            las_new.green = las.green
            las_new.blue = las.blue
        
        las = las_new
        needs_fix = True
        logger.info("✓ Converted to point format 2")
    
    # Save
    if needs_fix:
        if output_file is None:
            output_file = laz_file
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        las.write(output_file)
        logger.info(f"✓ Saved to: {output_file}")
        
        # Verify
        logger.info("Verifying fixed file...")
        if diagnose_rgb(output_file):
            logger.info("✓ RGB colors fixed successfully!")
        else:
            logger.error("✗ Fix may not have worked correctly")
    else:
        logger.info("No fixes needed - RGB looks good!")


def batch_fix(input_dir: Path, output_dir: Path = None):
    """
    Fix RGB in all LAZ files in directory.
    
    Args:
        input_dir: Input directory with LAZ files
        output_dir: Output directory (default: overwrite input files)
    """
    laz_files = list(input_dir.glob("*.laz"))
    
    if not laz_files:
        logger.error(f"No LAZ files found in {input_dir}")
        return
    
    logger.info(f"Found {len(laz_files)} LAZ files")
    logger.info("")
    
    for laz_file in laz_files:
        if output_dir:
            output_file = output_dir / laz_file.name
        else:
            output_file = None
        
        try:
            fix_rgb(laz_file, output_file)
            logger.info("")
        except Exception as e:
            logger.error(f"Failed to fix {laz_file.name}: {e}")
            logger.info("")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix RGB colors in LAZ files for CloudCompare"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input LAZ file or directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file or directory (default: overwrite input)"
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Only diagnose, don't fix"
    )
    
    args = parser.parse_args()
    
    if args.input.is_file():
        # Single file
        if args.diagnose_only:
            diagnose_rgb(args.input)
        else:
            fix_rgb(args.input, args.output)
    elif args.input.is_dir():
        # Directory
        if args.diagnose_only:
            for laz_file in args.input.glob("*.laz"):
                diagnose_rgb(laz_file)
                print()
        else:
            batch_fix(args.input, args.output)
    else:
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)
