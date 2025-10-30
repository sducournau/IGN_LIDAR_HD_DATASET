"""
Test for v3.0.5 priority fix in reclassifier.

Validates that the reversed priority order ensures buildings overwrite roads,
not the other way around.
"""

import numpy as np
import pytest


def test_priority_order_is_reversed():
    """Test that priority order uses get_priority_order_for_iteration directly."""
    from ign_lidar.core.classification.reclassifier import Reclassifier
    from ign_lidar.core.classification.priorities import (
        get_priority_order_for_iteration,
    )

    # Create reclassifier instance
    reclassifier = Reclassifier(chunk_size=10000, show_progress=False)

    # Get priority order from centralized function (already lowest → highest)
    expected_priority = get_priority_order_for_iteration()

    # Get the priority order used by reclassifier
    reclassifier_features = [feat for feat, _ in reclassifier.priority_order]

    # Verify that it uses the centralized order directly (no double reversal)
    assert (
        reclassifier_features == expected_priority
    ), "Priority order should match get_priority_order_for_iteration()"

    # Verify specific ordering: buildings should come AFTER roads in iteration
    # (so buildings overwrite roads)
    building_idx = None
    road_idx = None

    for idx, feature in enumerate(reclassifier_features):
        if feature == "buildings":
            building_idx = idx
        elif feature == "roads":
            road_idx = idx

    if building_idx is not None and road_idx is not None:
        assert (
            building_idx > road_idx
        ), "Buildings should be processed AFTER roads to overwrite them"


def test_priority_order_critical_features():
    """Test that critical features are in correct order."""
    from ign_lidar.core.classification.reclassifier import Reclassifier

    reclassifier = Reclassifier(chunk_size=10000, show_progress=False)
    reclassifier_features = [feat for feat, _ in reclassifier.priority_order]

    # Buildings should be processed late (high priority)
    # Roads/parking should be processed early (low priority)
    feature_positions = {feat: idx for idx, feat in enumerate(reclassifier_features)}

    # Check if features exist before comparing
    if "buildings" in feature_positions and "roads" in feature_positions:
        assert (
            feature_positions["buildings"] > feature_positions["roads"]
        ), "Buildings must have higher priority than roads"

    if "buildings" in feature_positions and "parking" in feature_positions:
        assert (
            feature_positions["buildings"] > feature_positions["parking"]
        ), "Buildings must have higher priority than parking"

    if "buildings" in feature_positions and "sports" in feature_positions:
        assert (
            feature_positions["buildings"] > feature_positions["sports"]
        ), "Buildings must have higher priority than sports"


def test_priority_order_consistency():
    """Test that multiple instances have consistent priority order."""
    from ign_lidar.core.classification.reclassifier import Reclassifier

    # Create two instances
    reclassifier1 = Reclassifier(chunk_size=10000, show_progress=False)
    reclassifier2 = Reclassifier(chunk_size=50000, show_progress=False)

    # Extract priority orders
    order1 = [feat for feat, _ in reclassifier1.priority_order]
    order2 = [feat for feat, _ in reclassifier2.priority_order]

    # Should be identical regardless of chunk_size
    assert order1 == order2, "Priority order should be consistent across instances"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
