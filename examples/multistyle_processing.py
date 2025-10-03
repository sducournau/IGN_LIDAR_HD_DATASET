"""
Example: Processing with Multi-Label Architectural Styles

This example demonstrates how to process LiDAR tiles with multi-label
architectural style encoding for enhanced training data.
"""

import numpy as np
from pathlib import Path
from ign_lidar.processor import LiDARProcessor
from ign_lidar.architectural_styles import ARCHITECTURAL_STYLES

def example_multihot_processing():
    """
    Process tiles with multi-hot architectural style encoding.
    
    This creates patches with [N, 13] style features representing
    multiple architectural styles with their weights.
    """
    print("=" * 70)
    print("Processing with Multi-Hot Architectural Styles")
    print("=" * 70)
    
    # Configure processor
    processor = LiDARProcessor(
        lod_level='LOD2',
        patch_size=150.0,
        k_neighbors=20,
        augment=True,
        num_augmentations=3,
        include_architectural_style=True,
        style_encoding='multihot'  # Multi-label encoding
    )
    
    # Process directory
    input_dir = Path('/path/to/raw_tiles')
    output_dir = Path('/path/to/dataset_multihot')
    
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Style encoding: multihot [N, 13]")
    print()
    
    num_patches = processor.process_directory(
        input_dir,
        output_dir,
        num_workers=4
    )
    
    print(f"\n✅ Created {num_patches} patches with multi-hot style encoding")
    print(f"   Each patch has architectural_style: [N, 13] array")


def example_constant_processing():
    """
    Process tiles with constant (single) architectural style encoding.
    
    This creates patches with [N] style features representing
    the dominant architectural style.
    """
    print("=" * 70)
    print("Processing with Constant Architectural Style")
    print("=" * 70)
    
    # Configure processor
    processor = LiDARProcessor(
        lod_level='LOD2',
        patch_size=150.0,
        k_neighbors=20,
        include_architectural_style=True,
        style_encoding='constant'  # Single style (legacy compatible)
    )
    
    # Process directory
    input_dir = Path('/path/to/raw_tiles')
    output_dir = Path('/path/to/dataset_constant')
    
    print(f"\nInput: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Style encoding: constant [N]")
    print()
    
    num_patches = processor.process_directory(
        input_dir,
        output_dir,
        num_workers=4
    )
    
    print(f"\n✅ Created {num_patches} patches with constant style encoding")
    print(f"   Each patch has architectural_style: [N] array")


def example_analyze_patch():
    """
    Analyze architectural style distribution in a processed patch.
    """
    print("=" * 70)
    print("Analyzing Architectural Style in Patch")
    print("=" * 70)
    
    # Load a patch
    patch_file = Path('/path/to/dataset/patch_0001.npz')
    
    if not patch_file.exists():
        print(f"⚠️  Patch file not found: {patch_file}")
        return
    
    data = np.load(patch_file)
    
    print(f"\nPatch: {patch_file.name}")
    print(f"Points: {len(data['points']):,}")
    print()
    
    # Check style encoding
    arch_style = data['architectural_style']
    
    if arch_style.ndim == 1:
        # Constant encoding [N]
        print("Style encoding: constant")
        style_id = int(arch_style[0])
        style_name = ARCHITECTURAL_STYLES.get(style_id, "unknown")
        print(f"Style ID: {style_id}")
        print(f"Style name: {style_name}")
        
    elif arch_style.ndim == 2:
        # Multi-hot encoding [N, 13]
        print("Style encoding: multihot")
        print(f"Shape: {arch_style.shape}")
        print()
        
        # Analyze style distribution
        mean_weights = np.mean(arch_style, axis=0)
        
        print("Style Distribution:")
        print("-" * 50)
        for style_id, weight in enumerate(mean_weights):
            if weight > 0.01:  # Only show if > 1%
                style_name = ARCHITECTURAL_STYLES.get(style_id, "unknown")
                print(f"  {style_name:20s}: {weight*100:5.1f}%")


def example_pytorch_dataloader():
    """
    Example PyTorch DataLoader for multi-label architectural styles.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class LiDARStyleDataset(Dataset):
        """Dataset with architectural style features."""
        
        def __init__(self, dataset_dir: Path):
            self.files = list(dataset_dir.glob('*.npz'))
            
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            data = np.load(self.files[idx])
            
            # Load geometry
            points = data['points']
            normals = data['normals']
            
            # Load features
            curvature = data['curvature'][:, None]
            intensity = data['intensity'][:, None]
            height = data['height'][:, None]
            
            # Load architectural style
            arch_style = data['architectural_style']
            
            # Handle both encodings
            if arch_style.ndim == 1:
                # Constant: convert to one-hot
                arch_style_onehot = np.zeros((len(points), 13), dtype=np.float32)
                arch_style_onehot[np.arange(len(points)), arch_style] = 1.0
                arch_style = arch_style_onehot
            
            # Concatenate all features
            features = np.hstack([
                points,          # [N, 3]
                normals,         # [N, 3]
                curvature,       # [N, 1]
                intensity,       # [N, 1]
                height,          # [N, 1]
                arch_style       # [N, 13]
            ])  # [N, 22]
            
            labels = data['labels']
            
            return (
                torch.from_numpy(features).float(),
                torch.from_numpy(labels).long()
            )
    
    print("=" * 70)
    print("PyTorch DataLoader with Architectural Styles")
    print("=" * 70)
    
    # Create dataset
    dataset = LiDARStyleDataset(Path('/path/to/dataset'))
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    print(f"\nDataset size: {len(dataset)} patches")
    print(f"Feature dimensions: 22 (3 xyz + 3 normals + 3 geo + 13 style)")
    print()
    
    # Test loading
    features, labels = next(iter(dataloader))
    print(f"Batch features shape: {features.shape}")  # [B, N, 22]
    print(f"Batch labels shape: {labels.shape}")      # [B, N]
    
    # Extract style features
    style_features = features[:, :, -13:]  # Last 13 dimensions
    print(f"Style features shape: {style_features.shape}")  # [B, N, 13]
    
    # Analyze style distribution in first batch
    mean_style_dist = style_features.mean(dim=(0, 1))
    print("\nStyle distribution in batch:")
    for style_id, weight in enumerate(mean_style_dist):
        if weight > 0.01:
            style_name = ARCHITECTURAL_STYLES.get(style_id, "unknown")
            print(f"  {style_name:20s}: {weight*100:5.1f}%")


def example_cli_usage():
    """
    Example CLI commands for processing with architectural styles.
    """
    print("=" * 70)
    print("CLI Usage Examples")
    print("=" * 70)
    
    print("\n1. Process with multi-hot encoding:")
    print("-" * 70)
    print("""
python -m ign_lidar.cli process \\
  --input-dir /path/to/raw_tiles \\
  --output /path/to/dataset_multihot \\
  --k-neighbors 20 \\
  --include-architectural-style \\
  --style-encoding multihot \\
  --num-workers 4 \\
  --patch-size 150.0
    """)
    
    print("\n2. Process with constant encoding (default):")
    print("-" * 70)
    print("""
python -m ign_lidar.cli process \\
  --input-dir /path/to/raw_tiles \\
  --output /path/to/dataset_constant \\
  --k-neighbors 20 \\
  --include-architectural-style \\
  --style-encoding constant \\
  --num-workers 4
    """)
    
    print("\n3. Migrate existing metadata to multi-label:")
    print("-" * 70)
    print("""
python scripts/maintenance/migrate_architectural_styles.py \\
  --input-dir /path/to/raw_tiles \\
  --dry-run

# After verification, apply migration:
python scripts/maintenance/migrate_architectural_styles.py \\
  --input-dir /path/to/raw_tiles
    """)
    
    print("\n4. Analyze style distribution:")
    print("-" * 70)
    print("""
python scripts/maintenance/migrate_architectural_styles.py \\
  --input-dir /path/to/raw_tiles \\
  --analyze
    """)


if __name__ == '__main__':
    print("\nMulti-Label Architectural Styles - Examples\n")
    
    # Uncomment to run examples
    # example_multihot_processing()
    # example_constant_processing()
    # example_analyze_patch()
    # example_pytorch_dataloader()
    example_cli_usage()
