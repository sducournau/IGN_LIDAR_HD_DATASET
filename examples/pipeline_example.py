#!/usr/bin/env python3
"""
Example: Run complete pipeline from YAML configuration

This example demonstrates how to execute the full LiDAR processing pipeline
using YAML configuration files for parameter management.
"""

import subprocess
from pathlib import Path


def run_full_pipeline():
    """
    Run complete pipeline: download → enrich → patch
    """
    config_file = Path("config_examples/pipeline_full.yaml")
    
    print("=" * 70)
    print("🚀 Running Full Pipeline from YAML Configuration")
    print("=" * 70)
    print(f"Configuration: {config_file}")
    print()
    
    cmd = [
        "ign-lidar-hd", "pipeline",
        str(config_file)
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed with code {result.returncode}")


def run_enrich_only():
    """
    Run only enrichment stage
    """
    config_file = Path("config_examples/pipeline_enrich.yaml")
    
    print("=" * 70)
    print("⚙️  Running Enrich-Only Pipeline")
    print("=" * 70)
    print(f"Configuration: {config_file}")
    print()
    
    cmd = [
        "ign-lidar-hd", "pipeline",
        str(config_file)
    ]
    
    subprocess.run(cmd)


def run_patch_only():
    """
    Run only patch creation stage
    """
    config_file = Path("config_examples/pipeline_patch.yaml")
    
    print("=" * 70)
    print("📦 Running Patch-Only Pipeline")
    print("=" * 70)
    print(f"Configuration: {config_file}")
    print()
    
    cmd = [
        "ign-lidar-hd", "pipeline",
        str(config_file)
    ]
    
    subprocess.run(cmd)


def create_custom_config():
    """
    Create a custom configuration programmatically
    """
    import yaml
    
    config = {
        'global': {
            'num_workers': 6,
        },
        'enrich': {
            'input_dir': 'data/my_tiles',
            'output': 'data/my_enriched',
            'mode': 'full',
            'k_neighbors': 15,
            'use_gpu': True,
            'add_rgb': True,
            'rgb_cache_dir': 'cache/rgb',
        },
        'patch': {
            'input_dir': 'data/my_enriched',
            'output': 'data/my_patches',
            'lod_level': 'LOD3',
            'patch_size': 100.0,
            'num_points': 8192,
            'augment': True,
            'num_augmentations': 5,
        },
    }
    
    output_path = Path("my_custom_pipeline.yaml")
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created custom configuration: {output_path}")
    print(f"  Edit the file and run: ign-lidar-hd pipeline {output_path}")
    
    return output_path


def create_example_configs():
    """
    Create example configuration files using CLI
    """
    print("Creating example configuration files...")
    
    # Create full pipeline example
    subprocess.run([
        "ign-lidar-hd", "pipeline",
        "pipeline_full_example.yaml",
        "--create-example", "full"
    ])
    
    # Create enrich-only example
    subprocess.run([
        "ign-lidar-hd", "pipeline",
        "pipeline_enrich_example.yaml",
        "--create-example", "enrich"
    ])
    
    # Create patch-only example
    subprocess.run([
        "ign-lidar-hd", "pipeline",
        "pipeline_patch_example.yaml",
        "--create-example", "patch"
    ])
    
    print("\n✓ Example configurations created!")
    print("  - pipeline_full_example.yaml")
    print("  - pipeline_enrich_example.yaml")
    print("  - pipeline_patch_example.yaml")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "full":
            run_full_pipeline()
        elif mode == "enrich":
            run_enrich_only()
        elif mode == "patch":
            run_patch_only()
        elif mode == "create":
            create_custom_config()
        elif mode == "examples":
            create_example_configs()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python pipeline_example.py [full|enrich|patch|create|examples]")
    else:
        print("Pipeline Example Usage:")
        print("  python pipeline_example.py full      # Run full pipeline")
        print("  python pipeline_example.py enrich    # Run enrich only")
        print("  python pipeline_example.py patch     # Run patch only")
        print("  python pipeline_example.py create    # Create custom config")
        print("  python pipeline_example.py examples  # Generate example configs")
