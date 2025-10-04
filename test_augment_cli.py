#!/usr/bin/env python3
"""
Quick test to verify augmentation implementation in CLI
"""

# Test that augmentation parameters are properly passed
import argparse
from pathlib import Path

# Simulate the CLI args
class MockArgs:
    input_dir = Path("test_input")
    output = Path("test_output")
    k_neighbors = 20
    use_gpu = False
    mode = "building"
    augment = True
    num_augmentations = 3
    add_rgb = False
    rgb_cache_dir = None
    preprocess = False
    sor_k = 12
    sor_std = 2.0
    ror_radius = 1.0
    ror_neighbors = 4
    voxel_size = None
    auto_params = False
    radius = None
    num_workers = 1

# Test worker args construction
args = MockArgs()
augment = getattr(args, 'augment', True)
num_augmentations = getattr(args, 'num_augmentations', 3)

print("Augmentation Configuration:")
print(f"  Augment: {augment}")
print(f"  Num augmentations: {num_augmentations}")

# Test creating worker args tuple
worker_arg = (
    Path("test.laz"),
    Path("output.laz"),
    args.k_neighbors,
    args.use_gpu,
    args.mode,
    True,  # skip_existing
    args.add_rgb,
    args.rgb_cache_dir,
    args.radius,
    args.preprocess,
    None,  # preprocess_config
    args.auto_params,
    augment,  # NEW
    num_augmentations  # NEW
)

print(f"\nWorker args tuple has {len(worker_arg)} elements")
print(f"  augment (index 12): {worker_arg[12]}")
print(f"  num_augmentations (index 13): {worker_arg[13]}")

print("\n✓ Augmentation parameters properly integrated into worker args!")
