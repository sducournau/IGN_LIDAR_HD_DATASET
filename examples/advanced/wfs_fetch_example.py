"""
Example: Using WFS Ground Truth Fetcher with Custom Retry Configuration

This example demonstrates how to use the IGNGroundTruthFetcher with
custom retry configuration for robust WFS data fetching.

The fetcher includes:
- Automatic retry with exponential backoff
- Cache validation and management
- Detailed error reporting
- Type-safe return values (empty GeoDataFrame instead of None)
"""

from pathlib import Path
from ign_lidar.io.wfs_ground_truth import IGNGroundTruthFetcher
from ign_lidar.io.wfs_fetch_result import RetryConfig


def basic_usage():
    """Basic usage with default settings."""
    print("=" * 70)
    print("BASIC USAGE: Default Retry Configuration")
    print("=" * 70)

    # Initialize fetcher with cache directory
    fetcher = IGNGroundTruthFetcher(cache_dir=Path("./cache/ground_truth"))

    # Define bounding box (example: Versailles area)
    # Format: (xmin, ymin, xmax, ymax) in Lambert 93 (EPSG:2154)
    bbox = (650000, 6860000, 651000, 6861000)

    print(f"\nFetching ground truth for bbox: {bbox}")

    # Fetch buildings
    buildings = fetcher.fetch_buildings(bbox)
    print(f"✅ Buildings: {len(buildings)} features")
    if len(buildings) > 0:
        print(f"   Columns: {list(buildings.columns)}")

    # Fetch roads
    roads = fetcher.fetch_roads(bbox)
    print(f"✅ Roads: {len(roads)} features")

    # Fetch water bodies
    water = fetcher.fetch_water(bbox)
    print(f"✅ Water: {len(water)} features")

    # Fetch vegetation
    vegetation = fetcher.fetch_vegetation(bbox)
    print(f"✅ Vegetation: {len(vegetation)} features")

    print("\nNote: Empty results return empty GeoDataFrame (not None)")
    print(f"Type of empty result: {type(vegetation)}")
    print(f"Length of empty result: {len(vegetation)}")


def custom_retry_config():
    """Using custom retry configuration for special cases."""
    print("\n" + "=" * 70)
    print("CUSTOM RETRY: Aggressive Retry for Unreliable Network")
    print("=" * 70)

    # Initialize fetcher
    fetcher = IGNGroundTruthFetcher(cache_dir=Path("./cache/ground_truth"))

    # Access the internal fetch method with custom retry config
    # (for demonstration - normally you'd use the high-level methods)
    from ign_lidar.io.wfs_fetch_result import fetch_with_retry, RetryConfig

    # Create custom retry configuration
    aggressive_retry = RetryConfig(
        max_retries=10,  # More retries for unreliable connections
        initial_delay=0.5,  # Start with shorter delay
        max_delay=60.0,  # Allow longer max delay
        backoff_factor=2.0,  # Exponential backoff (0.5, 1, 2, 4, 8, 16, 32, 60)
        retry_on_timeout=True,
        retry_on_network_error=True,
    )

    print("\nCustom retry configuration:")
    print(f"  Max retries: {aggressive_retry.max_retries}")
    print(f"  Initial delay: {aggressive_retry.initial_delay}s")
    print(f"  Max delay: {aggressive_retry.max_delay}s")
    print(f"  Backoff factor: {aggressive_retry.backoff_factor}x")

    # Show delay progression
    print("\n  Delay progression:")
    for attempt in range(aggressive_retry.max_retries):
        delay = aggressive_retry.get_delay(attempt)
        print(f"    Attempt {attempt + 1}: {delay:.1f}s")


def conservative_retry_config():
    """Conservative retry for production environments."""
    print("\n" + "=" * 70)
    print("CONSERVATIVE RETRY: Production Environment")
    print("=" * 70)

    # Conservative settings: fail faster, less network load
    conservative_retry = RetryConfig(
        max_retries=2,  # Only 2 retries
        initial_delay=5.0,  # Longer initial delay
        max_delay=10.0,  # Short max delay
        backoff_factor=1.5,  # Gentle backoff
        retry_on_timeout=True,
        retry_on_network_error=False,  # Don't retry network errors
    )

    print("\nConservative retry configuration:")
    print(f"  Max retries: {conservative_retry.max_retries}")
    print(f"  Initial delay: {conservative_retry.initial_delay}s")
    print(f"  Retry on network errors: {conservative_retry.retry_on_network_error}")

    print("\n  Delay progression:")
    for attempt in range(conservative_retry.max_retries):
        delay = conservative_retry.get_delay(attempt)
        print(f"    Attempt {attempt + 1}: {delay:.1f}s")


def cache_management():
    """Understanding cache behavior."""
    print("\n" + "=" * 70)
    print("CACHE MANAGEMENT: Validation and Cleanup")
    print("=" * 70)

    from ign_lidar.io.wfs_fetch_result import validate_cache_file

    cache_dir = Path("./cache/ground_truth")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a test cache file
    test_cache = cache_dir / "test_cache.geojson"

    print(f"\nCache validation checks:")
    print("  1. File exists")
    print("  2. File size > 0")
    print("  3. File age (optional)")
    print("  4. File header readable")

    # Validate cache without age limit
    is_valid = validate_cache_file(test_cache)
    print(f"\n✅ Cache valid (no age limit): {is_valid}")

    # Validate cache with age limit (7 days)
    is_valid_age = validate_cache_file(test_cache, max_age_days=7)
    print(f"✅ Cache valid (< 7 days): {is_valid_age}")

    print("\nCache benefits:")
    print("  - Reduces WFS server load")
    print("  - Faster repeated queries")
    print("  - Works offline with cached data")
    print("  - Automatic validation prevents corruption")


def error_handling():
    """Understanding error handling and return values."""
    print("\n" + "=" * 70)
    print("ERROR HANDLING: Type-Safe Return Values")
    print("=" * 70)

    fetcher = IGNGroundTruthFetcher(cache_dir=Path("./cache/ground_truth"))

    # Even if no features found, we get empty GeoDataFrame (not None)
    bbox = (0, 0, 1, 1)  # Tiny bbox, likely no features
    result = fetcher.fetch_buildings(bbox)

    print(f"\nReturn value for empty result:")
    print(f"  Type: {type(result)}")
    print(f"  Length: {len(result)}")
    print(f"  Is None: {result is None}")
    print(f"  Empty check: len(result) == 0 → {len(result) == 0}")

    print("\nBenefits of empty GeoDataFrame:")
    print("  ✅ Type-safe (always GeoDataFrame)")
    print("  ✅ No None checks needed")
    print("  ✅ Can chain operations safely")
    print("  ✅ Consistent API (len(gdf) always works)")

    # Example: Safe chaining
    buildings = fetcher.fetch_buildings((650000, 6860000, 651000, 6861000))

    # These operations work even if buildings is empty
    filtered = buildings[buildings.area > 100] if len(buildings) > 0 else buildings
    print(f"\n  Filtered buildings: {len(filtered)}")

    # Safe to iterate (empty loop if no data)
    for idx, building in buildings.head(3).iterrows():
        print(f"  Building {idx}: {building.geometry.area:.1f} m²")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 70)
    print(" WFS Ground Truth Fetcher - Examples")
    print(" Demonstrating Robust Retry Logic and Cache Management")
    print("*" * 70)

    try:
        # Run examples
        basic_usage()
        custom_retry_config()
        conservative_retry_config()
        cache_management()
        error_handling()

        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except ImportError as e:
        print(f"\n❌ Missing dependencies: {e}")
        print("Install with: pip install geopandas shapely requests")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
