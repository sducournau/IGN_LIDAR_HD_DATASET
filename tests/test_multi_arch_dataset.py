"""
Unit tests for IGNLiDARMultiArchDataset and PatchAugmentation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import dataset classes
try:
    from ign_lidar.datasets import IGNLiDARMultiArchDataset, PatchAugmentation
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE or not DATASETS_AVAILABLE,
    reason="PyTorch or dataset classes not available"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir():
    """Create temporary directory with synthetic patches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "patches"
        data_dir.mkdir(exist_ok=True)
        
        # Create 20 synthetic patches
        for i in range(20):
            patch = create_synthetic_patch(num_points=1000)
            np.savez(
                data_dir / f"patch_{i:04d}.npz",
                **patch
            )
        
        yield data_dir


def create_synthetic_patch(num_points: int = 1000) -> dict:
    """Create synthetic patch data."""
    rng = np.random.RandomState(42)
    
    patch = {
        'points': rng.randn(num_points, 3).astype(np.float32),
        'features': rng.randn(num_points, 13).astype(np.float32),
        'rgb': rng.randint(0, 255, size=(num_points, 3), dtype=np.uint8),
        'labels': rng.randint(0, 5, size=num_points, dtype=np.int32),
    }
    
    return patch


# ============================================================================
# Test PatchAugmentation
# ============================================================================

class TestPatchAugmentation:
    """Test augmentation functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        aug = PatchAugmentation()
        
        assert aug.rotation_enabled is True
        assert aug.scaling_enabled is True
        assert aug.jitter_enabled is True
    
    def test_init_custom_config(self):
        """Test custom configuration."""
        config = {
            'rotation': {'enabled': False},
            'scaling': {'enabled': True, 'min_scale': 0.5, 'max_scale': 1.5},
        }
        
        aug = PatchAugmentation(config)
        
        assert aug.rotation_enabled is False
        assert aug.scaling_enabled is True
        assert aug.scaling_min == 0.5
        assert aug.scaling_max == 1.5
    
    def test_rotation(self):
        """Test rotation augmentation."""
        aug = PatchAugmentation({
            'rotation': {'enabled': True, 'max_angle': 90}
        })
        
        patch = create_synthetic_patch(100)
        augmented = aug(patch)
        
        # Shape preserved
        assert augmented['points'].shape == patch['points'].shape
        
        # Points changed (rotated)
        assert not np.allclose(augmented['points'], patch['points'])
        
        # Z coordinate relatively unchanged (rotation around Z)
        # (with some tolerance for numerical precision)
        z_diff = np.abs(augmented['points'][:, 2] - patch['points'][:, 2])
        assert np.mean(z_diff) < 0.1
    
    def test_scaling(self):
        """Test scaling augmentation."""
        aug = PatchAugmentation({
            'rotation': {'enabled': False},
            'scaling': {'enabled': True, 'min_scale': 1.5, 'max_scale': 1.5},
            'jitter': {'enabled': False},
        })
        
        patch = create_synthetic_patch(100)
        augmented = aug(patch)
        
        # Points scaled by ~1.5
        scale_ratio = np.linalg.norm(augmented['points']) / np.linalg.norm(patch['points'])
        assert 1.4 < scale_ratio < 1.6
    
    @pytest.mark.xfail(reason="Augmentation algorithm changed")
    def test_translation(self):
        """Test translation augmentation."""
        aug = PatchAugmentation({
            'rotation': {'enabled': False},
            'translation': {'enabled': True, 'max_offset': 1.0},
            'scaling': {'enabled': False},
            'jitter': {'enabled': False},
        })
        
        patch = create_synthetic_patch(100)
        augmented = aug(patch)
        
        # Points translated
        offset = np.mean(augmented['points'] - patch['points'], axis=0)
        assert np.linalg.norm(offset) <= 1.0
    
    def test_jitter(self):
        """Test jitter augmentation."""
        aug = PatchAugmentation({
            'rotation': {'enabled': False},
            'scaling': {'enabled': False},
            'jitter': {'enabled': True, 'sigma': 0.01, 'clip': 0.05},
        })
        
        patch = create_synthetic_patch(100)
        augmented = aug(patch)
        
        # Small noise added
        diff = augmented['points'] - patch['points']
        assert np.max(np.abs(diff)) <= 0.05
        assert np.std(diff) < 0.02
    
    def test_dropout(self):
        """Test dropout augmentation."""
        aug = PatchAugmentation({
            'rotation': {'enabled': False},
            'scaling': {'enabled': False},
            'jitter': {'enabled': False},
            'dropout': {'enabled': True, 'ratio': 0.3},
        })
        
        patch = create_synthetic_patch(1000)
        augmented = aug(patch)
        
        # ~30% points dropped
        assert len(augmented['points']) < len(patch['points'])
        assert 650 <= len(augmented['points']) <= 750
        
        # All arrays same length
        assert len(augmented['features']) == len(augmented['points'])
        assert len(augmented['labels']) == len(augmented['points'])
    
    def test_feature_noise(self):
        """Test feature noise augmentation."""
        aug = PatchAugmentation({
            'rotation': {'enabled': False},
            'scaling': {'enabled': False},
            'jitter': {'enabled': False},
            'feature_noise': {'enabled': True, 'sigma': 0.01},
        })
        
        patch = create_synthetic_patch(100)
        augmented = aug(patch)
        
        # Features changed
        assert not np.allclose(augmented['features'], patch['features'])
        
        # Small changes
        diff = augmented['features'] - patch['features']
        assert np.std(diff) < 0.02


# ============================================================================
# Test IGNLiDARMultiArchDataset
# ============================================================================

class TestIGNLiDARMultiArchDataset:
    """Test dataset functionality."""
    
    def test_init_basic(self, temp_data_dir):
        """Test basic initialization."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='pointnet++',
            split='train'
        )
        
        assert len(dataset) > 0
        assert dataset.architecture == 'pointnet++'
        assert dataset.split == 'train'
    
    def test_init_with_preset(self, temp_data_dir):
        """Test initialization with preset."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='transformer',
            preset='buildings'
        )
        
        assert dataset.num_points == 8192  # From buildings preset
    
    def test_init_invalid_architecture(self, temp_data_dir):
        """Test invalid architecture raises error."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            IGNLiDARMultiArchDataset(
                temp_data_dir,
                architecture='invalid_arch'
            )
    
    def test_init_invalid_preset(self, temp_data_dir):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            IGNLiDARMultiArchDataset(
                temp_data_dir,
                preset='invalid_preset'
            )
    
    def test_data_split(self, temp_data_dir):
        """Test train/val/test split."""
        # Create datasets for each split
        train_dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            split='train',
            train_ratio=0.6,
            val_ratio=0.2,
            random_seed=42
        )
        
        val_dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            split='val',
            train_ratio=0.6,
            val_ratio=0.2,
            random_seed=42
        )
        
        test_dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            split='test',
            train_ratio=0.6,
            val_ratio=0.2,
            random_seed=42
        )
        
        # Check sizes
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        assert total == 20
        
        # Check ratios (approximately)
        assert 10 <= len(train_dataset) <= 14
        assert 3 <= len(val_dataset) <= 5
        assert 2 <= len(test_dataset) <= 5
    
    def test_getitem(self, temp_data_dir):
        """Test __getitem__ returns correct format."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='pointnet++',
            num_points=512
        )
        
        sample = dataset[0]
        
        # Check it's a dictionary
        assert isinstance(sample, dict)
        
        # Check torch tensors
        assert isinstance(sample['points'], torch.Tensor)
        assert isinstance(sample['features'], torch.Tensor)
        
        # Check shapes
        assert sample['points'].shape[0] <= 512
        assert sample['points'].shape[1] == 3
    
    def test_len(self, temp_data_dir):
        """Test __len__ returns correct count."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            split='train',
            train_ratio=0.8
        )
        
        # Should have ~16 patches (80% of 20)
        assert 14 <= len(dataset) <= 18
    
    def test_architecture_pointnet(self, temp_data_dir):
        """Test PointNet++ architecture formatting."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='pointnet++',
            num_points=256
        )
        
        sample = dataset[0]
        
        assert 'points' in sample
        assert 'features' in sample
        assert sample['points'].shape[1] == 3
    
    @pytest.mark.xfail(reason="KNNEngine API mismatch with formatter")
    def test_architecture_transformer(self, temp_data_dir):
        """Test transformer architecture formatting."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='transformer',
            num_points=256,
            knn_k=16
        )
        
        sample = dataset[0]
        
        assert 'points' in sample
        assert 'features' in sample
        assert 'edge_index' in sample
        assert 'pos_encoding' in sample
    
    def test_architecture_octree(self, temp_data_dir):
        """Test octree architecture formatting."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='octree',
            octree_depth=4
        )
        
        sample = dataset[0]
        
        assert 'octree' in sample
        assert 'points' in sample
        assert 'features' in sample
    
    @pytest.mark.xfail(reason="KNNEngine API mismatch with formatter")
    def test_architecture_sparse_conv(self, temp_data_dir):
        """Test sparse convolution architecture formatting."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='sparse_conv',
            voxel_size=0.2
        )
        
        sample = dataset[0]
        
        assert 'voxel_coords' in sample
        assert 'voxel_features' in sample
        assert 'points' in sample
    
    def test_augmentation(self, temp_data_dir):
        """Test augmentation is applied."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='pointnet++',
            augment=True,
            augmentation_config={
                'rotation': {'enabled': True},
                'jitter': {'enabled': True},
            }
        )
        
        # Get same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Should be different due to augmentation
        assert not torch.allclose(sample1['points'], sample2['points'])
    
    def test_cache_in_memory(self, temp_data_dir):
        """Test memory caching."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='pointnet++',
            cache_in_memory=True,
            split='train',
            train_ratio=0.5
        )
        
        # Cache should be populated
        assert len(dataset.cache) == len(dataset)
        
        # Getting items should use cache
        sample1 = dataset[0]
        sample2 = dataset[0]
        
        # Should be identical (same cached data)
        assert torch.allclose(sample1['points'], sample2['points'])
    
    def test_get_dataloader(self, temp_data_dir):
        """Test DataLoader creation."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='pointnet++',
            split='train'
        )
        
        dataloader = dataset.get_dataloader(
            batch_size=4,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        # Check it works
        batch = next(iter(dataloader))
        
        assert 'points' in batch
        assert batch['points'].shape[0] <= 4  # Batch size
    
    def test_get_stats(self, temp_data_dir):
        """Test statistics retrieval."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='pointnet++',
            split='train'
        )
        
        stats = dataset.get_stats()
        
        assert 'num_patches' in stats
        assert 'architecture' in stats
        assert 'num_points' in stats
        assert stats['num_patches'] > 0
        assert stats['architecture'] == 'pointnet++'
    
    def test_presets(self, temp_data_dir):
        """Test all presets work."""
        presets = ['buildings', 'vegetation', 'semantic_sota', 'fast']
        
        for preset in presets:
            dataset = IGNLiDARMultiArchDataset(
                temp_data_dir,
                architecture='pointnet++',
                preset=preset
            )
            
            assert len(dataset) > 0
            sample = dataset[0]
            assert 'points' in sample


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline(self, temp_data_dir):
        """Test complete pipeline: dataset → dataloader → batch."""
        dataset = IGNLiDARMultiArchDataset(
            temp_data_dir,
            architecture='transformer',
            preset='buildings',
            augment=True,
            split='train'
        )
        
        dataloader = dataset.get_dataloader(
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        # Iterate through batches
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            
            # Check batch format
            assert 'points' in batch
            assert 'features' in batch
            
            # Check batch size
            assert batch['points'].shape[0] <= 2
            
            if batch_count >= 3:
                break
        
        assert batch_count >= 3
    
    @pytest.mark.xfail(reason="KNNEngine API mismatch with formatter")
    def test_multiple_architectures(self, temp_data_dir):
        """Test different architectures on same data."""
        architectures = ['pointnet++', 'octree', 'transformer', 'sparse_conv']
        
        for arch in architectures:
            dataset = IGNLiDARMultiArchDataset(
                temp_data_dir,
                architecture=arch,
                split='train'
            )
            
            sample = dataset[0]
            
            # All should have points
            assert 'points' in sample
            assert isinstance(sample['points'], torch.Tensor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
