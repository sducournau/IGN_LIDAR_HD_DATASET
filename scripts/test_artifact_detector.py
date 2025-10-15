#!/usr/bin/env python3
"""
Test script for artifact detector.

Tests the artifact detection functionality with synthetic data.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ign_lidar.preprocessing import ArtifactDetector, ArtifactDetectorConfig


def create_synthetic_data_with_artifacts():
    """Create synthetic point cloud with artificial scan line artifacts."""
    np.random.seed(42)
    
    # Create regular grid
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    xx, yy = np.meshgrid(x, y)
    
    coords = np.column_stack([
        xx.flatten(),
        yy.flatten(),
        np.random.randn(10000) * 0.5 + 10  # Z with noise
    ])
    
    # Create feature with artificial dash line artifacts
    # Base value
    feature_clean = np.ones(len(coords)) * 0.5
    
    # Add scan line artifacts (stripes perpendicular to X, parallel to Y)
    y_values = coords[:, 1]
    stripe_pattern = np.sin(y_values * 0.5) * 0.3  # Wavy stripes
    
    # Add random dash lines (high variance bands)
    for y_pos in [20, 40, 60, 80]:
        mask = np.abs(y_values - y_pos) < 2
        stripe_pattern[mask] += np.random.randn(np.sum(mask)) * 0.5
    
    feature_with_artifacts = feature_clean + stripe_pattern
    
    # Clip to valid range
    feature_with_artifacts = np.clip(feature_with_artifacts, 0, 1)
    
    # Also create a clean feature for comparison
    feature_no_artifacts = feature_clean + np.random.randn(len(coords)) * 0.05
    
    return coords, feature_with_artifacts, feature_no_artifacts


def test_artifact_detection():
    """Test artifact detection on synthetic data."""
    print("="*80)
    print("ARTIFACT DETECTOR TEST")
    print("="*80)
    
    # Create synthetic data
    print("\n1. Creating synthetic data with artificial scan line artifacts...")
    coords, feature_bad, feature_good = create_synthetic_data_with_artifacts()
    print(f"   Created {len(coords)} points")
    
    # Initialize detector
    print("\n2. Initializing artifact detector...")
    config = ArtifactDetectorConfig()
    config.show_dash_lines = True
    detector = ArtifactDetector(config)
    print("   ✓ Detector initialized")
    
    # Test on feature with artifacts
    print("\n3. Testing on feature WITH artifacts (simulated scan lines)...")
    metrics_bad = detector.detect_spatial_artifacts(
        coords,
        feature_bad,
        feature_name="planarity_with_artifacts"
    )
    
    print(f"   CV_X: {metrics_bad.cv_x:.4f}")
    print(f"   CV_Y: {metrics_bad.cv_y:.4f}")
    print(f"   Max CV: {metrics_bad.max_cv:.4f}")
    print(f"   Severity: {metrics_bad.severity}")
    print(f"   Has Artifacts: {metrics_bad.has_artifacts}")
    print(f"   Recommended Action: {metrics_bad.recommended_action}")
    
    # Test on clean feature
    print("\n4. Testing on feature WITHOUT artifacts (clean)...")
    metrics_good = detector.detect_spatial_artifacts(
        coords,
        feature_good,
        feature_name="planarity_clean"
    )
    
    print(f"   CV_X: {metrics_good.cv_x:.4f}")
    print(f"   CV_Y: {metrics_good.cv_y:.4f}")
    print(f"   Max CV: {metrics_good.max_cv:.4f}")
    print(f"   Severity: {metrics_good.severity}")
    print(f"   Has Artifacts: {metrics_good.has_artifacts}")
    print(f"   Recommended Action: {metrics_good.recommended_action}")
    
    # Test dash line detection
    print("\n5. Testing dash line detection...")
    dash_lines = detector.detect_dash_lines(coords, feature_bad, metrics_bad)
    print(f"   Detected {len(dash_lines)} dash lines at Y positions:")
    for y_pos in dash_lines[:5]:
        print(f"     - Y = {y_pos:.2f}")
    if len(dash_lines) > 5:
        print(f"     ... and {len(dash_lines) - 5} more")
    
    # Test visualization creation
    print("\n6. Testing visualization (without saving)...")
    try:
        # Create output directory
        output_dir = Path(__file__).parent.parent / "data" / "test_output" / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualization
        vis_path = output_dir / "test_artifact_visualization.png"
        detector.visualize_artifacts(
            coords,
            feature_bad,
            "planarity_with_artifacts",
            metrics_bad,
            output_path=vis_path,
            show=False
        )
        print(f"   ✓ Visualization saved to: {vis_path}")
    except Exception as e:
        print(f"   ⚠ Visualization failed (may need matplotlib): {e}")
    
    # Test field dropping logic
    print("\n7. Testing field dropping recommendations...")
    results = {
        'planarity_bad': metrics_bad,
        'planarity_good': metrics_good
    }
    drop_list = detector.get_fields_to_drop(results, threshold=0.30)
    print(f"   Fields to drop (threshold=0.30): {drop_list}")
    
    # Validation
    print("\n8. Validating results...")
    assert metrics_bad.has_artifacts, "Failed: Bad feature should have artifacts"
    assert not metrics_good.has_artifacts, "Failed: Good feature should not have artifacts"
    assert metrics_bad.cv_y > metrics_good.cv_y, "Failed: Bad feature should have higher CV"
    assert 'planarity_bad' in drop_list, "Failed: Bad feature should be in drop list"
    assert 'planarity_good' not in drop_list, "Failed: Good feature should not be in drop list"
    print("   ✓ All validations passed")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Artifact detection working correctly")
    print(f"✅ Feature with artifacts: CV_Y={metrics_bad.cv_y:.4f}, Action={metrics_bad.recommended_action}")
    print(f"✅ Feature without artifacts: CV_Y={metrics_good.cv_y:.4f}, Action={metrics_good.recommended_action}")
    print(f"✅ Dash line detection found {len(dash_lines)} artifacts")
    print(f"✅ Field dropping logic working (recommended drop: {drop_list})")
    print("="*80)
    
    return True


if __name__ == '__main__':
    try:
        success = test_artifact_detection()
        if success:
            print("\n✨ All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
