#!/usr/bin/env python3
"""
Multi-Scale Dataset Merger for LOD3 Training

This script merges patches from different scales (50m, 100m, 150m) into a
unified training dataset for multi-scale learning.

Usage:
    python merge_multiscale_dataset.py --output patches_multiscale

Features:
- Weighted sampling from each scale
- Automatic train/val/test split
- Symbolic linking (memory efficient) or copying
- Dataset statistics and visualization
"""

import numpy as np
from pathlib import Path
import random
import argparse
import json
from collections import defaultdict
from typing import List, Tuple, Dict


def collect_patches(patch_dir: Path) -> List[Path]:
    """Collect all NPZ patches from a directory."""
    return list(patch_dir.glob('*.npz'))


def merge_multiscale_datasets(
    base_dir: str = '/mnt/c/Users/Simon/ign',
    patch_dirs: List[str] = ['patches_50m', 'patches_100m', 'patches_150m'],
    output_dir: str = 'patches_multiscale',
    scale_weights: List[float] = [0.4, 0.3, 0.3],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_symlinks: bool = True,
    seed: int = 42
) -> Dict:
    """
    Merge patches from different scales into a unified training dataset.
    
    Args:
        base_dir: Base directory containing patch folders
        patch_dirs: List of directories containing patches at different scales
        output_dir: Output directory for merged dataset
        scale_weights: Relative proportions of each scale in final dataset
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        use_symlinks: Use symbolic links instead of copying files
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with dataset statistics
    """
    random.seed(seed)
    np.random.seed(seed)
    
    base_path = Path(base_dir)
    output_path = base_path / output_dir
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("Multi-Scale Dataset Merger")
    print("=" * 80)
    
    all_patches = []
    scale_stats = defaultdict(lambda: {'count': 0, 'size_mb': 0.0})
    
    # Collect patches from each scale
    for patch_dir, weight in zip(patch_dirs, scale_weights):
        patch_path = base_path / patch_dir
        
        if not patch_path.exists():
            print(f"⚠️  Warning: {patch_path} does not exist, skipping...")
            continue
            
        patches = collect_patches(patch_path)
        scale_name = patch_dir.split('_')[-1]
        
        print(f"\n📂 Processing {scale_name} patches from {patch_dir}:")
        print(f"   Found: {len(patches)} patches")
        
        # Calculate target sample size based on weight
        min_weight = min(scale_weights)
        target_samples = int(len(patches) * weight / min_weight)
        n_samples = min(target_samples, len(patches))
        
        # Sample patches
        sampled = random.sample(patches, n_samples)
        
        # Collect statistics
        total_size = sum(p.stat().st_size for p in sampled)
        scale_stats[scale_name]['count'] = n_samples
        scale_stats[scale_name]['size_mb'] = total_size / (1024 * 1024)
        
        print(f"   Sampled: {n_samples} patches ({weight*100:.1f}% weight)")
        print(f"   Size: {scale_stats[scale_name]['size_mb']:.2f} MB")
        
        all_patches.extend([(p, scale_name) for p in sampled])
    
    # Shuffle for randomization
    random.shuffle(all_patches)
    
    print(f"\n📊 Total patches in multi-scale dataset: {len(all_patches)}")
    for scale, stats in scale_stats.items():
        percentage = (stats['count'] / len(all_patches)) * 100
        print(f"   - {scale} patches: {stats['count']} ({percentage:.1f}%)")
    
    # Split into train/val/test
    n_total = len(all_patches)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    train_patches = all_patches[:n_train]
    val_patches = all_patches[n_train:n_train + n_val]
    test_patches = all_patches[n_train + n_val:]
    
    print(f"\n📋 Dataset Split:")
    print(f"   Train: {n_train} patches ({train_split*100:.0f}%)")
    print(f"   Val:   {n_val} patches ({val_split*100:.0f}%)")
    print(f"   Test:  {n_test} patches ({test_split*100:.0f}%)")
    
    # Create links/copies
    print(f"\n💾 Creating {'symbolic links' if use_symlinks else 'copies'}...")
    
    splits = {
        'train': train_patches,
        'val': val_patches,
        'test': test_patches
    }
    
    for split_name, patch_list in splits.items():
        split_dir = output_path / split_name
        
        for i, (patch_path, scale) in enumerate(patch_list):
            # Generate unique filename with scale information
            tile_name = patch_path.stem.split('_patch_')[0]
            patch_id = patch_path.stem.split('_patch_')[-1].split('_aug_')[0]
            aug_id = patch_path.stem.split('_aug_')[-1] if '_aug_' in patch_path.stem else 'base'
            
            output_name = f"{tile_name}_scale{scale}_{split_name}_p{patch_id}_aug{aug_id}.npz"
            output_file = split_dir / output_name
            
            # Create symlink or copy
            if use_symlinks:
                try:
                    if output_file.exists():
                        output_file.unlink()
                    output_file.symlink_to(patch_path.absolute())
                except Exception as e:
                    print(f"⚠️  Failed to create symlink, copying instead: {e}")
                    import shutil
                    shutil.copy2(patch_path, output_file)
            else:
                import shutil
                shutil.copy2(patch_path, output_file)
        
        print(f"   ✓ {split_name}: {len(patch_list)} patches")
    
    # Save metadata
    metadata = {
        'total_patches': len(all_patches),
        'scales': dict(scale_stats),
        'splits': {
            'train': n_train,
            'val': n_val,
            'test': n_test
        },
        'scale_weights': dict(zip([d.split('_')[-1] for d in patch_dirs], scale_weights)),
        'source_directories': patch_dirs,
        'seed': seed
    }
    
    metadata_file = output_path / 'dataset_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Merged dataset saved to: {output_dir}")
    print(f"📄 Metadata saved to: {metadata_file}")
    print("=" * 80)
    
    return metadata


def visualize_scale_distribution(metadata: Dict):
    """Print a simple visualization of scale distribution."""
    print("\n📈 Scale Distribution:")
    print("-" * 60)
    
    for scale, stats in metadata['scales'].items():
        count = stats['count']
        total = metadata['total_patches']
        percentage = (count / total) * 100
        bar_length = int(percentage / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        
        print(f"{scale:>5}: {bar} {count:>5} patches ({percentage:>5.1f}%)")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Merge multi-scale LOD3 training patches'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/mnt/c/Users/Simon/ign',
        help='Base directory containing patch folders'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='patches_multiscale',
        help='Output directory name'
    )
    parser.add_argument(
        '--scales',
        type=str,
        nargs='+',
        default=['patches_50m', 'patches_100m', 'patches_150m'],
        help='Scale directories to merge'
    )
    parser.add_argument(
        '--weights',
        type=float,
        nargs='+',
        default=[0.4, 0.3, 0.3],
        help='Relative weights for each scale'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Training split ratio'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Test split ratio'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of creating symlinks'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Validate splits
    total_split = args.train_split + args.val_split + args.test_split
    if not np.isclose(total_split, 1.0):
        print(f"⚠️  Warning: Split ratios sum to {total_split:.2f}, not 1.0")
        print("   Adjusting splits to sum to 1.0...")
        scale_factor = 1.0 / total_split
        args.train_split *= scale_factor
        args.val_split *= scale_factor
        args.test_split *= scale_factor
    
    # Merge datasets
    metadata = merge_multiscale_datasets(
        base_dir=args.base_dir,
        patch_dirs=args.scales,
        output_dir=args.output,
        scale_weights=args.weights,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        use_symlinks=not args.copy,
        seed=args.seed
    )
    
    # Visualize results
    visualize_scale_distribution(metadata)
    
    print("\n🎯 Next Steps:")
    print("   1. Verify dataset with: ls -lh", Path(args.base_dir) / args.output)
    print("   2. Start training with your multi-scale dataset")
    print("   3. Monitor training metrics across different scales")


if __name__ == "__main__":
    main()
