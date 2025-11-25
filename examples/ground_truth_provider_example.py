"""
Ground Truth Provider - Unified Interface Examples

This file demonstrates how to use the GroundTruthProvider for unified
ground truth operations. The GroundTruthProvider consolidates three
previously separate classes into a single, easy-to-use interface.

Before running this example, ensure you have the required dependencies:
    pip install shapely geopandas laspy

Examples included:
1. High-level convenience API (recommended)
2. Prefetch & cache workflow
3. Low-level component access
4. Cache management
5. Error handling

Author: IGN LiDAR HD Development Team
Version: 1.0.0
"""

import numpy as np
from pathlib import Path
import logging

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the unified ground truth provider
from ign_lidar.core.ground_truth_provider import GroundTruthProvider, get_provider


def example_1_high_level_api():
    """
    Example 1: High-Level Convenience API (RECOMMENDED)

    This is the recommended approach for most use cases. It provides
    a simple, intuitive interface without needing to know about
    sub-components.

    Use case:
    - Single tile processing
    - Quick prototyping
    - Most common scenarios
    """
    print("\n" + "=" * 70)
    print("Example 1: High-Level Convenience API")
    print("=" * 70)

    # Get the global provider instance
    gt = get_provider()

    print(f"Provider: {gt}")
    print(f"Cache stats: {gt.get_cache_stats()}")

    # Define bounding box for area of interest
    bbox = (100.0, 50.0, 150.0, 100.0)  # (minx, miny, maxx, maxy)

    print(f"\nFetching ground truth for bbox: {bbox}")

    try:
        # Fetch all ground truth features (buildings, roads, vegetation, water, etc.)
        features = gt.fetch_all_features(bbox)

        print(f"Fetched features: {list(features.keys())}")

        # Create sample point cloud
        n_points = 1000
        points = np.random.uniform(
            low=[bbox[0], bbox[1], 0],
            high=[bbox[2], bbox[3], 50],
            size=(n_points, 3),
        )

        print(f"Created {n_points} sample points")

        # Label points with ground truth
        labels = gt.label_points(points, features)

        print(f"Labeled {len(labels)} points")
        print(f"Unique classes: {np.unique(labels)}")

        # Check cache statistics
        stats = gt.get_cache_stats()
        print(f"Cache stats: {stats}")

    except Exception as e:
        logger.error(f"Error in high-level API example: {e}")
        print("Note: This example requires shapely/geopandas and WFS connectivity")


def example_2_prefetch_workflow():
    """
    Example 2: Prefetch & Cache Workflow

    This approach is useful when processing multiple tiles. Prefetch
    ground truth data first, then process tiles while data is cached.

    Use case:
    - Batch processing multiple tiles
    - Performance optimization
    - Memory management
    """
    print("\n" + "=" * 70)
    print("Example 2: Prefetch & Cache Workflow")
    print("=" * 70)

    # Reset provider for clean example
    GroundTruthProvider.reset_instance()
    gt = GroundTruthProvider(cache_enabled=True)

    print(f"Provider cache enabled: {gt._cache_enabled}")

    # Simulate multiple tiles to process
    bboxes = [
        (100.0, 50.0, 150.0, 100.0),
        (150.0, 50.0, 200.0, 100.0),
        (100.0, 100.0, 150.0, 150.0),
    ]

    print(f"\nProcessing {len(bboxes)} tiles")

    # Phase 1: Prefetch all ground truth data
    print("\nPhase 1: Prefetching ground truth data...")

    for i, bbox in enumerate(bboxes, 1):
        try:
            features = gt.fetch_all_features(bbox)
            print(f"  Tile {i}: Fetched {len(features)} feature types")
        except Exception as e:
            logger.warning(f"Could not prefetch tile {i}: {e}")

    # Check cache after prefetch
    stats = gt.get_cache_stats()
    print(f"\nCache after prefetch: {stats['size']} items cached")

    # Phase 2: Process tiles (data already cached)
    print("\nPhase 2: Processing tiles with cached data...")

    for i, bbox in enumerate(bboxes, 1):
        try:
            # This will use cached data from Phase 1
            features = gt.fetch_all_features(bbox)

            # Label points
            points = np.random.uniform(
                low=[bbox[0], bbox[1], 0],
                high=[bbox[2], bbox[3], 50],
                size=(100, 3),
            )
            labels = gt.label_points(points, features)

            print(f"  Tile {i}: Labeled {len(labels)} points (from cache)")
        except Exception as e:
            logger.warning(f"Could not process tile {i}: {e}")


def example_3_low_level_access():
    """
    Example 3: Low-Level Component Access

    For advanced use cases, you can access individual sub-components
    directly. This provides fine-grained control but requires understanding
    each component's API.

    Use case:
    - Advanced optimization
    - Custom workflows
    - Debugging
    - Deep integration
    """
    print("\n" + "=" * 70)
    print("Example 3: Low-Level Component Access")
    print("=" * 70)

    gt = GroundTruthProvider()

    print("\nAvailable sub-components:")
    print(f"  - fetcher: WFS data fetching")
    print(f"  - manager: Cache & prefetch management")
    print(f"  - optimizer: Spatial labeling optimization")

    try:
        # Access fetcher directly
        print("\nAccessing fetcher component...")
        fetcher = gt.fetcher
        print(f"Fetcher: {type(fetcher).__name__}")

        # Access manager directly
        print("\nAccessing manager component...")
        manager = gt.manager
        if manager:
            print(f"Manager: {type(manager).__name__}")
        else:
            print("Manager: Not available")

        # Access optimizer directly
        print("\nAccessing optimizer component...")
        optimizer = gt.optimizer
        if optimizer:
            print(f"Optimizer: {type(optimizer).__name__}")
        else:
            print("Optimizer: Not available")

        print("\nNote: Low-level components provide direct access to")
        print("original functionality for advanced use cases")

    except ImportError as e:
        logger.warning(f"Could not access component: {e}")


def example_4_cache_management():
    """
    Example 4: Cache Management

    Demonstrates how to manage the internal cache for performance
    optimization and memory management.

    Use case:
    - Memory management
    - Testing
    - Cache optimization
    - Debugging
    """
    print("\n" + "=" * 70)
    print("Example 4: Cache Management")
    print("=" * 70)

    # Reset and create new provider with cache
    GroundTruthProvider.reset_instance()
    gt = GroundTruthProvider(cache_enabled=True)

    print(f"Initial cache state: {gt.get_cache_stats()}")

    # Simulate cache usage
    print("\nSimulating cache usage...")

    for i in range(3):
        try:
            bbox = (100.0 + i * 50, 50.0, 150.0 + i * 50, 100.0)
            features = gt.fetch_all_features(bbox)
            print(f"  Fetch {i+1}: bbox {bbox}")
        except Exception:
            pass

    # Check cache status
    stats = gt.get_cache_stats()
    print(f"\nCache after fetches: {stats['size']} items")
    print(f"Cached items: {stats['keys']}")

    # Clear cache
    print("\nClearing cache...")
    gt.clear_cache()

    stats = gt.get_cache_stats()
    print(f"Cache after clear: {stats['size']} items")


def example_5_error_handling():
    """
    Example 5: Error Handling

    Demonstrates proper error handling when using the provider.

    Use case:
    - Graceful degradation
    - Logging
    - Recovery strategies
    """
    print("\n" + "=" * 70)
    print("Example 5: Error Handling")
    print("=" * 70)

    gt = GroundTruthProvider()

    bbox = (100.0, 50.0, 150.0, 100.0)
    points = np.random.rand(100, 3)

    print(f"Attempting to fetch features for bbox: {bbox}")

    try:
        # This may fail if dependencies not available or WFS unavailable
        features = gt.fetch_all_features(bbox)
        print(f"✓ Successfully fetched {len(features)} feature types")

        # Try labeling
        labels = gt.label_points(points, features)
        print(f"✓ Successfully labeled {len(labels)} points")

    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        print("  → Missing required dependencies (shapely, geopandas)")
        print("  → Install with: pip install shapely geopandas")

    except ConnectionError as e:
        logger.error(f"✗ Connection error: {e}")
        print("  → Could not connect to WFS service")
        print("  → Check network connectivity")

    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        print(f"  → Error type: {type(e).__name__}")


def example_6_singleton_pattern():
    """
    Example 6: Singleton Pattern

    The GroundTruthProvider uses the singleton pattern to ensure
    only one instance exists per process.

    Use case:
    - Understanding implementation details
    - Testing
    - Multi-threaded environments
    """
    print("\n" + "=" * 70)
    print("Example 6: Singleton Pattern")
    print("=" * 70)

    # Reset for clean example
    GroundTruthProvider.reset_instance()

    # Create first instance
    gt1 = GroundTruthProvider(cache_enabled=True)
    print(f"Created gt1: {id(gt1)}")

    # Create second instance
    gt2 = GroundTruthProvider(cache_enabled=False)  # Cache setting ignored
    print(f"Created gt2: {id(gt2)}")

    # Verify they're the same object
    print(f"\ngt1 is gt2: {gt1 is gt2}")
    print(f"Cache enabled in both: {gt1._cache_enabled}")

    # Using module-level get_provider function
    gt3 = get_provider()
    print(f"Created gt3 via get_provider: {id(gt3)}")
    print(f"gt1 is gt3: {gt1 is gt3}")

    print("\nAll three references point to the same singleton instance!")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Ground Truth Provider - Comprehensive Examples")
    print("=" * 70)

    # Run examples
    example_1_high_level_api()
    example_2_prefetch_workflow()
    example_3_low_level_access()
    example_4_cache_management()
    example_5_error_handling()
    example_6_singleton_pattern()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
