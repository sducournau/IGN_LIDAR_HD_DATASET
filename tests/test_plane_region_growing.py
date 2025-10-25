"""Test plane region growing and segmentation."""

import numpy as np
import pytest

from ign_lidar.core.classification.plane_detection import PlaneDetector, PlaneType


class TestPlaneRegionGrowing:
    """Test the region growing segmentation implementation."""

    def test_single_horizontal_plane_segmentation(self):
        """Test segmentation of a single horizontal plane."""
        detector = PlaneDetector(use_spatial_coherence=True)

        # Create horizontal plane (10x10 grid at z=5.0)
        x, y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
        points = np.column_stack(
            [x.ravel(), y.ravel(), np.full(2500, 5.0) + np.random.normal(0, 0.02, 2500)]
        )

        # Normals pointing up
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        normals += np.random.normal(0, 0.05, normals.shape)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        planarity = np.ones(len(points)) * 0.95

        # Segment
        segments = detector._segment_with_region_growing(
            points, normals, planarity, np.arange(len(points)), PlaneType.HORIZONTAL
        )

        # Should detect at least one plane
        assert len(segments) >= 1, "Should detect at least one plane"
        # Most points should be in detected planes
        total_segmented = sum(seg.n_points for seg in segments)
        assert total_segmented >= 0.8 * len(
            points
        ), "At least 80% of points should be segmented"

    def test_two_separated_horizontal_planes(self):
        """Test segmentation of two separated horizontal planes."""
        detector = PlaneDetector(use_spatial_coherence=True)

        # Create two horizontal planes at different heights
        # Plane 1: z=5.0
        x1, y1 = np.meshgrid(np.linspace(0, 5, 30), np.linspace(0, 5, 30))
        points1 = np.column_stack(
            [x1.ravel(), y1.ravel(), np.full(900, 5.0) + np.random.normal(0, 0.02, 900)]
        )

        # Plane 2: z=8.0 (separated by 5m in X)
        x2, y2 = np.meshgrid(np.linspace(10, 15, 30), np.linspace(0, 5, 30))
        points2 = np.column_stack(
            [x2.ravel(), y2.ravel(), np.full(900, 8.0) + np.random.normal(0, 0.02, 900)]
        )

        points = np.vstack([points1, points2])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        normals += np.random.normal(0, 0.05, normals.shape)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        planarity = np.ones(len(points)) * 0.95

        # Segment
        segments = detector._segment_with_region_growing(
            points, normals, planarity, np.arange(len(points)), PlaneType.HORIZONTAL
        )

        # Should detect two separate planes
        assert len(segments) >= 2, "Should detect at least two separate planes"

        # Check height separation
        heights = [seg.height_mean for seg in segments]
        height_range = max(heights) - min(heights)
        assert (
            height_range > 2.0
        ), "Detected planes should have significant height difference"

    def test_normal_similarity_filtering(self):
        """Test that points with dissimilar normals are filtered."""
        detector = PlaneDetector(use_spatial_coherence=True)

        # Create plane with mixed normals
        x, y = np.meshgrid(np.linspace(0, 10, 40), np.linspace(0, 10, 40))
        points = np.column_stack([x.ravel(), y.ravel(), np.full(1600, 5.0)])

        # Half points have upward normals, half have random normals
        normals = np.zeros((len(points), 3), dtype=float)
        normals[:800] = [0.0, 0.0, 1.0]  # Coherent normals
        normals[800:] = np.random.randn(800, 3)  # Random normals
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        planarity = np.ones(len(points)) * 0.90

        # Segment
        segments = detector._segment_with_region_growing(
            points, normals, planarity, np.arange(len(points)), PlaneType.HORIZONTAL
        )

        # Should filter out points with dissimilar normals
        if segments:
            total_accepted = sum(seg.n_points for seg in segments)
            # Most accepted points should be from coherent half
            assert (
                total_accepted < len(points) * 0.7
            ), "Should filter points with dissimilar normals"

    def test_min_points_threshold(self):
        """Test that clusters below min_points are rejected."""
        detector = PlaneDetector(use_spatial_coherence=True, min_points_per_plane=100)

        # Create small cluster (50 points)
        points = np.random.rand(50, 3) * 2.0
        normals = np.tile([0.0, 0.0, 1.0], (50, 1))
        planarity = np.ones(50) * 0.95

        # Segment
        segments = detector._segment_with_region_growing(
            points, normals, planarity, np.arange(50), PlaneType.HORIZONTAL
        )

        # Should reject cluster (too few points)
        assert len(segments) == 0, "Should reject clusters below min_points threshold"

    def test_fallback_to_single_segment(self):
        """Test fallback when spatial coherence is disabled."""
        detector = PlaneDetector(use_spatial_coherence=False)

        # Create two separated planes
        x1, y1 = np.meshgrid(np.linspace(0, 5, 20), np.linspace(0, 5, 20))
        points1 = np.column_stack([x1.ravel(), y1.ravel(), np.full(400, 5.0)])

        x2, y2 = np.meshgrid(np.linspace(10, 15, 20), np.linspace(0, 5, 20))
        points2 = np.column_stack([x2.ravel(), y2.ravel(), np.full(400, 5.0)])

        points = np.vstack([points1, points2])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        planarity = np.ones(len(points)) * 0.95

        # Segment without spatial coherence
        segments = detector._segment_planes(
            points, normals, planarity, np.arange(len(points)), PlaneType.HORIZONTAL
        )

        # Should create single segment (fallback)
        assert (
            len(segments) == 1
        ), "Should create single segment when spatial coherence disabled"
        assert segments[0].n_points == len(
            points
        ), "Single segment should contain all points"

    def test_vertical_plane_segmentation(self):
        """Test segmentation with different eps for vertical planes."""
        detector = PlaneDetector(use_spatial_coherence=True)

        # Create vertical plane (wall)
        x, z = np.meshgrid(np.linspace(0, 10, 40), np.linspace(0, 5, 30))
        points = np.column_stack(
            [x.ravel(), np.full(1200, 5.0) + np.random.normal(0, 0.05, 1200), z.ravel()]
        )

        # Normals pointing in Y direction (perpendicular to wall)
        normals = np.tile([0.0, 1.0, 0.0], (len(points), 1))
        normals += np.random.normal(0, 0.05, normals.shape)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        planarity = np.ones(len(points)) * 0.90

        # Segment as vertical plane
        segments = detector._segment_with_region_growing(
            points, normals, planarity, np.arange(len(points)), PlaneType.VERTICAL
        )

        # Should detect vertical plane(s)
        assert len(segments) >= 1, "Should detect vertical plane"
        total_segmented = sum(seg.n_points for seg in segments)
        assert total_segmented >= 0.7 * len(points), "Most points should be segmented"

    def test_create_plane_segment_properties(self):
        """Test that plane segment properties are computed correctly."""
        detector = PlaneDetector()

        # Create simple plane
        points = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5], [1, 1, 5]], dtype=float)
        normals = np.tile([0.0, 0.0, 1.0], (4, 1))
        planarity = np.ones(4) * 0.95
        indices = np.arange(4)

        segment = detector._create_plane_segment(
            points, normals, planarity, indices, PlaneType.HORIZONTAL, segment_id=42
        )

        # Check properties
        assert segment.plane_type == PlaneType.HORIZONTAL
        assert segment.id == 42
        assert segment.n_points == 4
        assert np.allclose(segment.centroid, [0.5, 0.5, 5.0])
        assert np.allclose(segment.normal, [0, 0, 1])
        assert segment.height_mean == 5.0
        assert segment.height_std == 0.0
        assert segment.area > 0  # Should compute some area
        assert abs(segment.orientation_angle) < 5.0  # Nearly horizontal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
