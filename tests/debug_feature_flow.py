#!/usr/bin/env python3
"""
Quick test to process one tile with debug logging to trace feature flow
"""

import logging
import sys

# Enable DEBUG logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s %(name)-40s %(message)s',
    stream=sys.stdout
)

# Now import and run
from ign_lidar.core.processor import LiDARProcessor
import numpy as np

print("="*80)
print("DEBUGGING FEATURE FLOW - PROCESSING ONE TILE")
print("="*80)

# Minimal config for quick test
processor = LiDARProcessor(
    lod_level='LOD3',
    processing_mode='patches_only',
    patch_size=50.0,
    patch_overlap=0.1,
    num_points=8192,  # Small for speed
    include_extra_features=True,
    feature_mode='full',  # FULL feature mode
    k_neighbors=20,
    use_gpu=True,
    use_gpu_chunked=True,
    gpu_batch_size=500000,
    architecture='hybrid',
    output_format='npz'
)

# Process one tile
input_dir = "/mnt/c/Users/Simon/ign/versailles/input"
output_dir = "/tmp/feature_debug_test"

print(f"\nProcessing from: {input_dir}")
print(f"Output to: {output_dir}")
print("\nLook for DEBUG messages showing feature counts...")
print("="*80)

try:
    processor.process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        max_tiles=1  # Only process ONE tile for debugging
    )
    
    print("\n" + "="*80)
    print("✅ Processing completed!")
    print("="*80)
    
    # Check the output
    import os
    npz_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    print(f"\nGenerated {len(npz_files)} NPZ files")
    
    if npz_files:
        # Load and inspect first file
        test_file = os.path.join(output_dir, npz_files[0])
        data = np.load(test_file, allow_pickle=True)
        
        print(f"\nInspecting: {npz_files[0]}")
        print(f"Arrays in NPZ: {len(data.files)}")
        print(f"Features array shape: {data['features'].shape}")
        
        if 'metadata' in data.files:
            metadata = data['metadata'].item()
            if isinstance(metadata, dict) and 'feature_names' in metadata:
                print(f"Feature names in metadata: {metadata['feature_names']}")
                print(f"Number of named features: {len(metadata['feature_names'])}")
        
        data.close()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
