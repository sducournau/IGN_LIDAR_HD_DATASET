"""
Test Feature Validation and Artifact Detection

Tests the validation logic for geometric features at tile boundaries.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ign_lidar.features.features_boundary import BoundaryAwareFeatureComputer


def test_valid_features():
    """Test that valid features pass validation."""
    print("\n=== Test 1: Valid Features ===")
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    # Create realistic feature values
    n_points = 1000
    boundary_mask = np.random.rand(n_points) < 0.1  # 10% boundary points
    
    features = {
        'normals': np.random.randn(n_points, 3),
        'curvature': np.random.rand(n_points) * 0.3,
        'planarity': np.random.rand(n_points) * 0.7 + 0.15,  # [0.15, 0.85]
        'linearity': np.random.rand(n_points) * 0.6,  # [0, 0.6]
        'sphericity': np.random.rand(n_points) * 0.3,
        'verticality': np.random.rand(n_points) * 0.8,
        'boundary_mask': boundary_mask,
        'num_boundary_points': np.sum(boundary_mask)
    }
    
    validated = computer._validate_features(features, boundary_mask)
    
    assert 'planarity' in validated, "Valid planarity should not be dropped"
    assert 'linearity' in validated, "Valid linearity should not be dropped"
    assert 'verticality' in validated, "Valid verticality should not be dropped"
    
    print("✓ All valid features passed validation")


def test_linearity_artifact():
    """Test detection of linearity scan line artifact."""
    print("\n=== Test 2: Linearity Artifact (Scan Lines) ===")
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    n_points = 1000
    boundary_mask = np.random.rand(n_points) < 0.1
    
    # Create artificial linearity with high mean + low variance
    linearity = np.ones(n_points) * 0.85  # Very high, constant
    linearity[boundary_mask] = 0.85 + np.random.randn(np.sum(boundary_mask)) * 0.02
    
    features = {
        'normals': np.random.randn(n_points, 3),
        'curvature': np.random.rand(n_points) * 0.3,
        'planarity': np.random.rand(n_points) * 0.5,
        'linearity': linearity,
        'sphericity': np.random.rand(n_points) * 0.3,
        'verticality': np.random.rand(n_points) * 0.8,
        'boundary_mask': boundary_mask,
        'num_boundary_points': np.sum(boundary_mask)
    }
    
    validated = computer._validate_features(features, boundary_mask)
    
    assert 'linearity' not in validated, "Artifacted linearity should be dropped"
    assert 'planarity' in validated, "Valid planarity should remain"
    
    print("✓ Linearity artifact detected and dropped")


def test_planarity_discontinuity():
    """Test detection of planarity discontinuity."""
    print("\n=== Test 3: Planarity Discontinuity ===")
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    n_points = 1000
    boundary_mask = np.random.rand(n_points) < 0.1
    
    # Create planarity with very high variance (discontinuity)
    planarity = np.random.rand(n_points) * 0.5
    boundary_indices = np.where(boundary_mask)[0]
    planarity[boundary_indices[:len(boundary_indices)//2]] = 0.05  # Low
    planarity[boundary_indices[len(boundary_indices)//2:]] = 0.95  # High
    
    features = {
        'normals': np.random.randn(n_points, 3),
        'curvature': np.random.rand(n_points) * 0.3,
        'planarity': planarity,
        'linearity': np.random.rand(n_points) * 0.5,
        'sphericity': np.random.rand(n_points) * 0.3,
        'verticality': np.random.rand(n_points) * 0.8,
        'boundary_mask': boundary_mask,
        'num_boundary_points': np.sum(boundary_mask)
    }
    
    validated = computer._validate_features(features, boundary_mask)
    
    assert 'planarity' not in validated, "Discontinuous planarity should be dropped"
    assert 'linearity' in validated, "Valid linearity should remain"
    
    print("✓ Planarity discontinuity detected and dropped")


def test_verticality_bimodal():
    """Test detection of verticality bimodal extreme pattern."""
    print("\n=== Test 4: Verticality Bimodal Extreme ===")
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    n_points = 1000
    boundary_mask = np.random.rand(n_points) < 0.1
    
    # Create bimodal verticality (all at extremes)
    verticality = np.random.rand(n_points) * 0.5
    boundary_indices = np.where(boundary_mask)[0]
    verticality[boundary_indices[:len(boundary_indices)//2]] = 0.01  # Near 0
    verticality[boundary_indices[len(boundary_indices)//2:]] = 0.98  # Near 1
    
    features = {
        'normals': np.random.randn(n_points, 3),
        'curvature': np.random.rand(n_points) * 0.3,
        'planarity': np.random.rand(n_points) * 0.5,
        'linearity': np.random.rand(n_points) * 0.5,
        'sphericity': np.random.rand(n_points) * 0.3,
        'verticality': verticality,
        'boundary_mask': boundary_mask,
        'num_boundary_points': np.sum(boundary_mask)
    }
    
    validated = computer._validate_features(features, boundary_mask)
    
    assert 'verticality' not in validated, "Bimodal verticality should be dropped"
    assert 'planarity' in validated, "Valid planarity should remain"
    
    print("✓ Verticality bimodal extreme detected and dropped")


def test_invalid_values():
    """Test detection of NaN and Inf values."""
    print("\n=== Test 5: Invalid Values (NaN, Inf) ===")
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    n_points = 1000
    boundary_mask = np.random.rand(n_points) < 0.1
    
    # Create features with NaN
    planarity = np.random.rand(n_points) * 0.5
    boundary_indices = np.where(boundary_mask)[0]
    if len(boundary_indices) > 0:
        planarity[boundary_indices[0]] = np.nan
    
    features = {
        'normals': np.random.randn(n_points, 3),
        'curvature': np.random.rand(n_points) * 0.3,
        'planarity': planarity,
        'linearity': np.random.rand(n_points) * 0.5,
        'sphericity': np.random.rand(n_points) * 0.3,
        'verticality': np.random.rand(n_points) * 0.8,
        'boundary_mask': boundary_mask,
        'num_boundary_points': np.sum(boundary_mask)
    }
    
    validated = computer._validate_features(features, boundary_mask)
    
    assert 'planarity' not in validated, "Features with NaN should be dropped"
    assert 'linearity' in validated, "Valid linearity should remain"
    
    print("✓ NaN values detected and feature dropped")


def test_no_boundary_points():
    """Test that validation is skipped when no boundary points."""
    print("\n=== Test 6: No Boundary Points ===")
    
    computer = BoundaryAwareFeatureComputer(k_neighbors=20)
    
    n_points = 1000
    boundary_mask = np.zeros(n_points, dtype=bool)  # No boundary points
    
    # Even create "bad" features
    features = {
        'normals': np.random.randn(n_points, 3),
        'curvature': np.random.rand(n_points) * 0.3,
        'planarity': np.ones(n_points) * 0.85,  # Would fail if checked
        'linearity': np.ones(n_points) * 0.85,  # Would fail if checked
        'sphericity': np.random.rand(n_points) * 0.3,
        'verticality': np.random.rand(n_points) * 0.8,
        'boundary_mask': boundary_mask,
        'num_boundary_points': 0
    }
    
    validated = computer._validate_features(features, boundary_mask)
    
    # All features should pass (validation skipped)
    assert 'planarity' in validated, "Should skip validation when no boundary"
    assert 'linearity' in validated, "Should skip validation when no boundary"
    
    print("✓ Validation correctly skipped with no boundary points")


if __name__ == '__main__':
    print("="*60)
    print("Feature Validation Test Suite")
    print("="*60)
    
    try:
        test_valid_features()
        test_linearity_artifact()
        test_planarity_discontinuity()
        test_verticality_bimodal()
        test_invalid_values()
        test_no_boundary_points()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
