"""
Unit Tests for Data Augmentation at ENRICH Phase

Tests verify that:
1. Raw point augmentation works correctly
2. Features are recomputed on augmented geometry
3. Feature-geometry consistency is maintained
"""

import pytest
import numpy as np
from pathlib import Path

from ign_lidar.utils import augment_raw_points


class TestAugmentationEnrich:
    """Test data augmentation at ENRICH phase."""
    
    def test_augment_raw_points_shape(self):
        """Test that augmentation returns correct shapes (with dropout)."""
        N = 10000
        points = np.random.randn(N, 3).astype(np.float32)
        intensity = np.random.rand(N).astype(np.float32)
        return_number = np.random.randint(1, 4, N).astype(np.float32)
        classification = np.random.randint(1, 20, N).astype(np.uint8)
        
        (points_aug, intensity_aug,
         return_number_aug, classification_aug) = augment_raw_points(
            points, intensity, return_number, classification
        )
        
        # After dropout (5-15%), should have fewer points
        assert len(points_aug) < N
        assert len(points_aug) > N * 0.85  # At least 85% retained
        
        # All arrays should have same length
        assert len(intensity_aug) == len(points_aug)
        assert len(return_number_aug) == len(points_aug)
        assert len(classification_aug) == len(points_aug)
        
        # Correct shapes
        assert points_aug.shape[1] == 3
        assert intensity_aug.ndim == 1
        assert return_number_aug.ndim == 1
        assert classification_aug.ndim == 1
    
    def test_augment_raw_points_rotation(self):
        """Test that rotation is applied correctly."""
        # Create a simple point cloud along X-axis
        N = 1000
        points = np.zeros((N, 3), dtype=np.float32)
        points[:, 0] = np.linspace(0, 10, N)  # Points along X
        points[:, 2] = 5.0  # Fixed Z
        
        intensity = np.ones(N, dtype=np.float32)
        return_number = np.ones(N, dtype=np.float32)
        classification = np.ones(N, dtype=np.uint8)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        (points_aug, _, _, _) = augment_raw_points(
            points, intensity, return_number, classification
        )
        
        # After rotation, points should not be aligned with X-axis
        # (except by chance, but with jitter/dropout unlikely)
        x_variance = np.var(points_aug[:, 0])
        y_variance = np.var(points_aug[:, 1])
        
        # Y should have non-zero variance after rotation
        assert y_variance > 0.01
        
        # Z should still be roughly preserved (rotation around Z-axis)
        # But jitter adds noise
        assert np.abs(np.mean(points_aug[:, 2]) - 5.0) < 1.0
    
    def test_augment_raw_points_jitter(self):
        """Test that jitter is applied."""
        # Create identical points
        N = 1000
        points = np.ones((N, 3), dtype=np.float32) * 5.0
        
        intensity = np.ones(N, dtype=np.float32)
        return_number = np.ones(N, dtype=np.float32)
        classification = np.ones(N, dtype=np.uint8)
        
        np.random.seed(42)
        
        (points_aug, _, _, _) = augment_raw_points(
            points, intensity, return_number, classification
        )
        
        # After jitter, points should be different
        variance = np.var(points_aug, axis=0)
        
        # All dimensions should have some variance from jitter
        assert np.all(variance > 0.001)
    
    def test_augment_raw_points_scaling(self):
        """Test that scaling is applied."""
        N = 1000
        points = np.random.randn(N, 3).astype(np.float32) * 10.0
        
        intensity = np.ones(N, dtype=np.float32)
        return_number = np.ones(N, dtype=np.float32)
        classification = np.ones(N, dtype=np.uint8)
        
        np.random.seed(42)
        
        original_scale = np.std(points)
        
        (points_aug, _, _, _) = augment_raw_points(
            points, intensity, return_number, classification
        )
        
        augmented_scale = np.std(points_aug)
        
        # Scale should be different (0.95-1.05 range + jitter effect)
        assert augmented_scale != original_scale
        # Should be within reasonable range
        assert 0.9 * original_scale < augmented_scale < 1.1 * original_scale
    
    def test_augment_raw_points_dropout(self):
        """Test that dropout removes 5-15% of points."""
        N = 10000
        points = np.random.randn(N, 3).astype(np.float32)
        intensity = np.random.rand(N).astype(np.float32)
        return_number = np.ones(N, dtype=np.float32)
        classification = np.ones(N, dtype=np.uint8)
        
        # Run multiple times to check dropout range
        dropout_ratios = []
        for seed in range(10):
            np.random.seed(seed)
            (points_aug, _, _, _) = augment_raw_points(
                points, intensity, return_number, classification
            )
            dropout_ratio = 1.0 - (len(points_aug) / N)
            dropout_ratios.append(dropout_ratio)
        
        # Dropout should be in 5-15% range
        assert all(0.05 <= r <= 0.15 for r in dropout_ratios)
        
        # Should have some variation
        assert max(dropout_ratios) - min(dropout_ratios) > 0.02
    
    def test_augment_raw_points_preserves_classification(self):
        """Test that classification codes are preserved."""
        N = 1000
        points = np.random.randn(N, 3).astype(np.float32)
        intensity = np.random.rand(N).astype(np.float32)
        return_number = np.random.randint(1, 4, N).astype(np.float32)
        
        # Create specific classification distribution
        classification = np.array(
            [2] * 300 +     # Ground
            [6] * 500 +     # Building
            [5] * 200,      # Vegetation
            dtype=np.uint8
        )
        
        np.random.seed(42)
        
        (_, _, _, classification_aug) = augment_raw_points(
            points, intensity, return_number, classification
        )
        
        # Classification codes should still be 2, 5, or 6
        unique_classes = np.unique(classification_aug)
        assert all(c in [2, 5, 6] for c in unique_classes)
        
        # Should have all three classes (if dropout not too aggressive)
        assert len(unique_classes) >= 2  # At least 2 classes remain
    
    def test_augmentation_creates_different_results(self):
        """Test that multiple augmentations produce different results."""
        N = 1000
        points = np.random.randn(N, 3).astype(np.float32)
        intensity = np.random.rand(N).astype(np.float32)
        return_number = np.ones(N, dtype=np.float32)
        classification = np.ones(N, dtype=np.uint8)
        
        # Generate multiple augmentations
        augmentations = []
        for seed in range(5):
            np.random.seed(seed)
            (points_aug, _, _, _) = augment_raw_points(
                points, intensity, return_number, classification
            )
            augmentations.append(points_aug)
        
        # All augmentations should be different
        for i in range(len(augmentations)):
            for j in range(i + 1, len(augmentations)):
                # Cannot directly compare (different lengths due to dropout)
                # Compare first 100 points
                n_compare = min(100, len(augmentations[i]),
                               len(augmentations[j]))
                diff = np.mean(
                    np.abs(
                        augmentations[i][:n_compare] -
                        augmentations[j][:n_compare]
                    )
                )
                # Should be significantly different
                assert diff > 0.1


class TestFeatureGeometryConsistency:
    """
    Test that features computed on augmented geometry are consistent.
    
    This tests the CONCEPT, not full pipeline (which is integration test).
    """
    
    def test_normals_change_with_rotation(self):
        """
        Test that normals change when geometry is rotated.
        
        This is the KEY improvement: features must be recomputed
        after augmentation.
        """
        from ign_lidar.features import compute_normals_optimized
        
        # Create a simple planar surface (XY plane)
        N = 1000
        points = np.random.rand(N, 3).astype(np.float32) * 10
        points[:, 2] = 0  # Flat plane at Z=0
        
        # Compute normals on original
        normals_original, _ = compute_normals_optimized(points, k=10)
        
        # Average normal should point up (0, 0, 1)
        avg_normal_orig = np.mean(normals_original, axis=0)
        assert np.abs(avg_normal_orig[2]) > 0.9  # Mostly Z-direction
        
        # Rotate 90° around X-axis (plane becomes vertical)
        angle = np.pi / 2
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)
        
        points_rotated = points @ rotation_matrix.T
        
        # Recompute normals on rotated geometry
        normals_rotated, _ = compute_normals_optimized(points_rotated, k=10)
        
        # Average normal should now point in Y-direction
        avg_normal_rot = np.mean(normals_rotated, axis=0)
        
        # After 90° rotation around X, Z becomes Y
        assert np.abs(avg_normal_rot[1]) > 0.9  # Mostly Y-direction
        
        # This demonstrates why features MUST be recomputed!
        # If we just rotated the normals, we'd have the wrong result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
