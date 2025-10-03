"""
Example: Using GPU Acceleration with LiDARProcessor

This example shows how to use GPU acceleration when processing
LiDAR tiles into ML-ready patches.

Requirements:
- NVIDIA GPU with CUDA support
- CuPy: pip install cupy-cuda11x (adjust for your CUDA version)
- Optional: RAPIDS cuML for additional acceleration

Performance:
- GPU typically provides 4-10x speedup over CPU
- Most effective on large tiles (>1M points)
"""

from pathlib import Path
from ign_lidar.processor import LiDARProcessor

# Configure processor with GPU acceleration
processor = LiDARProcessor(
    lod_level='LOD2',
    patch_size=150.0,
    num_points=16384,
    augment=True,
    num_augmentations=3,
    include_extra_features=True,  # Building-specific features
    use_gpu=True  # ⚡ Enable GPU acceleration
)

# Process a single tile
input_laz = Path("data/raw/tile_0501_6320.laz")
output_dir = Path("data/patches")
output_dir.mkdir(parents=True, exist_ok=True)

print("Processing with GPU acceleration...")
num_patches = processor.process_tile(input_laz, output_dir)
print(f"Created {num_patches} patches")

# Note: If GPU is not available, the processor will automatically
# fall back to CPU with a warning message. No code changes needed!

# Batch processing with GPU
print("\nBatch processing multiple tiles...")
input_dir = Path("data/raw")
laz_files = list(input_dir.glob("*.laz"))

for i, laz_file in enumerate(laz_files, 1):
    print(f"[{i}/{len(laz_files)}] Processing {laz_file.name}...")
    num_patches = processor.process_tile(
        laz_file,
        output_dir,
        tile_idx=i,
        total_tiles=len(laz_files)
    )
    print(f"  Created {num_patches} patches")

print("\n✅ Done! All tiles processed with GPU acceleration")
