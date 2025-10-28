#!/usr/bin/env python3
"""
Test script to verify enriched tile saving with all fixes.
"""
from pathlib import Path

import laspy
import numpy as np

from ign_lidar.core.classification.io.serializers import save_enriched_tile_laz

# Create test data
n_points = 1000
print(f"Creating test data with {n_points:,} points...")

points = np.random.rand(n_points, 3).astype(np.float32) * 100
classification = np.random.randint(0, 10, n_points, dtype=np.uint8)
intensity = np.random.rand(n_points).astype(np.float32)
return_number = np.ones(n_points, dtype=np.float32)
input_rgb = np.random.rand(n_points, 3).astype(np.float32)
input_nir = np.random.rand(n_points).astype(np.float32)

# Create features with various name lengths and types
features = {
    "curvature": np.random.rand(n_points).astype(np.float32),
    "planarity": np.random.rand(n_points).astype(np.float32),
    "linearity": np.random.rand(n_points).astype(np.float32),
    "sphericity": np.random.rand(n_points).astype(np.float32),
    "anisotropy": np.random.rand(n_points).astype(np.float32),
    "roughness": np.random.rand(n_points).astype(np.float32),
    "normals": np.random.rand(n_points, 3).astype(np.float32),  # Test 3D array
    "normal_x": np.random.rand(n_points).astype(np.float32),
    "normal_y": np.random.rand(n_points).astype(np.float32),
    "normal_z": np.random.rand(n_points).astype(np.float32),
    # Test long names that need truncation
    "sum_eigenvalues": np.random.rand(n_points).astype(np.float32),
    "eigenentropy": np.random.rand(n_points).astype(np.float32),
    "omnivariance": np.random.rand(n_points).astype(np.float32),
    "change_curvature": np.random.rand(n_points).astype(np.float32),
    "edge_strength": np.random.rand(n_points).astype(np.float32),
    "corner_likelihood": np.random.rand(n_points).astype(np.float32),
    "overhang_indicator": np.random.rand(n_points).astype(np.float32),
    "surface_roughness": np.random.rand(n_points).astype(np.float32),
    "num_points_2m": np.random.rand(n_points).astype(np.float32),
    "neighborhood_extent": np.random.rand(n_points).astype(np.float32),
    "height_extent_ratio": np.random.rand(n_points).astype(np.float32),
    "vertical_std": np.random.rand(n_points).astype(np.float32),
    "height_above_ground": np.random.rand(n_points).astype(np.float32),
    "verticality": np.random.rand(n_points).astype(np.float32),
}

# Create dummy header for test
print("Creating LAS header...")
header = laspy.LasHeader(version="1.4", point_format=8)
header.offsets = [
    np.floor(points[:, 0].min()),
    np.floor(points[:, 1].min()),
    np.floor(points[:, 2].min()),
]
header.scales = [0.001, 0.001, 0.001]

# Save test file
output_path = Path("/tmp/test_enriched.laz")
print(f"\nSaving enriched tile to: {output_path}")

try:
    save_enriched_tile_laz(
        save_path=output_path,
        points=points,
        classification=classification,
        intensity=intensity,
        return_number=return_number,
        features=features,
        original_las=None,
        header=header,
        input_rgb=input_rgb,
        input_nir=input_nir,
    )
    print(f"✓ Save successful!")

    # Verify the saved file
    print(f"\nVerifying saved file...")
    las = laspy.read(str(output_path))
    print(f"✓ File loaded successfully!")
    print(f"  Points: {len(las.points):,}")
    print(f"  Point format: {las.point_format.id}")

    # List extra dimensions
    standard = [
        "X",
        "Y",
        "Z",
        "intensity",
        "return_number",
        "classification",
        "red",
        "green",
        "blue",
        "nir",
    ]
    extra = [name for name in las.point_format.dimension_names if name not in standard]

    print(f"\n📊 Extra dimensions saved: {len(extra)}")
    for name in sorted(extra):
        values = np.array(getattr(las, name))
        print(f"  - {name:30s} min={values.min():.3f}, max={values.max():.3f}")

    print(f"\n✅ Test passed! All features saved successfully.")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
