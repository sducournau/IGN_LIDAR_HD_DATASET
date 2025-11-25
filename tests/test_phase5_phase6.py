"""
Tests for Phase 5 & 6: PyTorch Integration and Distributed Processing

Tests cover:
- Tensor conversion between NumPy and PyTorch
- GPU inference pipeline
- Multi-GPU coordination
- Distributed feature computation
- Data loading and partitioning
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Optional

# Optional imports for GPU tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)


@pytest.fixture
def sample_features():
    """Generate sample feature array."""
    return np.random.randn(1000, 38).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample label array."""
    return np.random.randint(0, 15, size=1000)


@pytest.fixture
def sample_point_cloud():
    """Generate sample point cloud."""
    return np.random.randn(10000, 3).astype(np.float32)


class TestTensorConverter:
    """Tests for PyTorch tensor conversion."""

    @pytest.mark.unit
    def test_numpy_to_tensor_cpu(self, sample_features):
        """Test conversion from NumPy to tensor on CPU."""
        from ign_lidar.features.pytorch_integration import TensorConverter

        converter = TensorConverter(device='cpu', tensor_dtype='float32')
        tensor = converter.numpy_to_tensor(sample_features)

        assert torch.is_tensor(tensor)
        assert tensor.device.type == 'cpu'
        assert tensor.dtype == torch.float32
        assert tensor.shape == sample_features.shape

    @pytest.mark.unit
    def test_tensor_to_numpy(self, sample_features):
        """Test conversion from tensor back to NumPy."""
        from ign_lidar.features.pytorch_integration import TensorConverter

        converter = TensorConverter(device='cpu')
        tensor = converter.numpy_to_tensor(sample_features)
        recovered = converter.tensor_to_numpy(tensor)

        assert isinstance(recovered, np.ndarray)
        np.testing.assert_allclose(recovered, sample_features, rtol=1e-5)

    @pytest.mark.unit
    def test_batch_conversion(self, sample_features):
        """Test batch conversion of multiple arrays."""
        from ign_lidar.features.pytorch_integration import TensorConverter

        converter = TensorConverter(device='cpu')
        arrays = [sample_features, sample_features]
        tensors = converter.batch_numpy_to_tensor(arrays)

        assert len(tensors) == 2
        assert all(torch.is_tensor(t) for t in tensors)

    @pytest.mark.unit
    def test_stack_tensors(self, sample_features):
        """Test stacking multiple tensors."""
        from ign_lidar.features.pytorch_integration import TensorConverter

        converter = TensorConverter(device='cpu')
        tensors = converter.batch_numpy_to_tensor(
            [sample_features, sample_features]
        )
        stacked = converter.stack_tensors(tensors)

        assert stacked.shape[0] == 2
        assert stacked.shape[1:] == sample_features.shape


class TestGPUInference:
    """Tests for GPU inference pipeline."""

    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(38, 64)
                self.fc2 = nn.Linear(64, 15)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        return SimpleModel()

    @pytest.mark.unit
    def test_gpu_inference_cpu(self, sample_features, simple_model):
        """Test GPU inference on CPU."""
        from ign_lidar.features.pytorch_integration import GPUInference

        inference = GPUInference(
            simple_model,
            batch_size=256,
            device='cpu'
        )

        predictions = inference.predict(sample_features)
        assert predictions.shape[0] == sample_features.shape[0]
        assert predictions.shape[1] == 15

    @pytest.mark.unit
    def test_inference_with_confidence(self, sample_features, simple_model):
        """Test inference with confidence filtering."""
        from ign_lidar.features.pytorch_integration import GPUInference

        inference = GPUInference(
            simple_model,
            batch_size=256,
            device='cpu'
        )

        predictions, confidences = inference.predict_with_confidence(
            sample_features,
            confidence_threshold=0.5
        )

        assert len(predictions) <= len(sample_features)
        assert len(confidences) == len(predictions)
        assert np.all(confidences >= 0.5)

    @pytest.mark.unit
    def test_model_loader(self, simple_model, tmp_path):
        """Test model loading and saving."""
        from ign_lidar.features.pytorch_integration import ModelLoader

        model_path = tmp_path / "test_model.pt"

        # Save model
        ModelLoader.save_model(
            simple_model,
            model_path,
            metadata={'version': '1.0'}
        )

        assert model_path.exists()

        # Load model
        loaded_model = ModelLoader.load_model(
            model_path,
            model_class=simple_model.__class__,
            device='cpu'
        )

        assert loaded_model is not None


class TestGPUManager:
    """Tests for GPU coordination."""

    @pytest.mark.unit
    def test_gpu_manager_init(self):
        """Test GPU manager initialization."""
        from ign_lidar.optimization.distributed_processor import GPUManager

        manager = GPUManager(verbose=False)
        assert manager is not None

    @pytest.mark.unit
    def test_get_available_gpus(self):
        """Test getting available GPUs."""
        from ign_lidar.optimization.distributed_processor import GPUManager

        manager = GPUManager(verbose=False)
        gpus = manager.get_available_gpus()
        assert isinstance(gpus, list)

    @pytest.mark.unit
    def test_memory_usage(self):
        """Test memory usage tracking."""
        from ign_lidar.optimization.distributed_processor import GPUManager

        manager = GPUManager(verbose=False)
        usage = manager.get_gpu_memory_usage()
        assert isinstance(usage, dict)


class TestMultiGPUProcessor:
    """Tests for multi-GPU processing."""

    @pytest.mark.unit
    def test_multi_gpu_processor_init(self):
        """Test multi-GPU processor initialization."""
        from ign_lidar.optimization.distributed_processor import MultiGPUProcessor

        processor = MultiGPUProcessor(num_gpus=1, verbose=False)
        assert processor.num_gpus >= 0

    @pytest.mark.unit
    def test_round_robin_partition(self):
        """Test round-robin partitioning."""
        from ign_lidar.optimization.distributed_processor import MultiGPUProcessor

        processor = MultiGPUProcessor(num_gpus=1, verbose=False)
        data = list(range(10))
        partitions = processor._partition_round_robin(data)

        assert len(partitions) == processor.num_gpus
        total = sum(len(p) for p in partitions)
        assert total == len(data)

    @pytest.mark.unit
    def test_balanced_partition(self):
        """Test balanced partitioning."""
        from ign_lidar.optimization.distributed_processor import MultiGPUProcessor

        processor = MultiGPUProcessor(num_gpus=1, verbose=False)
        data = list(range(10))
        partitions = processor._partition_balanced(data)

        assert len(partitions) == processor.num_gpus
        total = sum(len(p) for p in partitions)
        assert total == len(data)


class TestDistributedFeatureCompute:
    """Tests for distributed feature computation."""

    @pytest.mark.unit
    def test_distributed_compute_init(self):
        """Test initialization."""
        from ign_lidar.optimization.distributed_processor import DistributedFeatureCompute

        compute = DistributedFeatureCompute(num_gpus=1)
        assert compute is not None

    @pytest.mark.unit
    def test_spatial_partitioning(self, sample_point_cloud):
        """Test spatial partitioning."""
        from ign_lidar.optimization.distributed_processor import DistributedFeatureCompute

        compute = DistributedFeatureCompute(num_gpus=1)
        partitions = compute._partition_point_cloud(
            sample_point_cloud,
            strategy='spatial',
            chunk_size=1000
        )

        assert len(partitions) > 0
        total_points = sum(len(p) for p in partitions)
        assert total_points == len(sample_point_cloud)

    @pytest.mark.unit
    def test_balanced_partitioning(self, sample_point_cloud):
        """Test balanced partitioning."""
        from ign_lidar.optimization.distributed_processor import DistributedFeatureCompute

        compute = DistributedFeatureCompute(num_gpus=1)
        partitions = compute._partition_point_cloud(
            sample_point_cloud,
            strategy='balanced',
            chunk_size=1000
        )

        assert len(partitions) > 0
        total_points = sum(len(p) for p in partitions)
        assert total_points == len(sample_point_cloud)


class TestDistributedDataLoader:
    """Tests for distributed data loading."""

    @pytest.mark.unit
    def test_data_loader_init(self, sample_features):
        """Test data loader initialization."""
        from ign_lidar.optimization.distributed_processor import DistributedDataLoader

        loader = DistributedDataLoader(
            sample_features,
            num_workers=2,
            batch_size=32,
            num_ranks=1,
            rank=0
        )

        assert loader is not None

    @pytest.mark.unit
    def test_data_loader_iteration(self, sample_features):
        """Test iterating over data loader."""
        from ign_lidar.optimization.distributed_processor import DistributedDataLoader

        loader = DistributedDataLoader(
            sample_features.tolist(),
            num_workers=2,
            batch_size=32,
            num_ranks=1,
            rank=0,
            shuffle=False
        )

        batches = list(loader)
        assert len(batches) > 0
        assert len(batches[0]) <= 32


class TestPhase5Phase6Integration:
    """Integration tests for Phase 5 & 6."""

    @pytest.mark.integration
    def test_pytorch_and_distributed_together(self, sample_features):
        """Test PyTorch and distributed modules work together."""
        from ign_lidar.features.pytorch_integration import TensorConverter
        from ign_lidar.optimization.distributed_processor import GPUManager

        converter = TensorConverter(device='cpu')
        manager = GPUManager(verbose=False)

        tensor = converter.numpy_to_tensor(sample_features)
        assert torch.is_tensor(tensor)

        gpus = manager.get_available_gpus()
        assert isinstance(gpus, list)

    @pytest.mark.integration
    def test_full_pipeline_simulation(self, sample_features, sample_labels):
        """Simulate full pipeline: convert → inference → distribute."""
        from ign_lidar.features.pytorch_integration import (
            TensorConverter, GPUInference
        )
        from ign_lidar.optimization.distributed_processor import (
            DistributedDataLoader
        )

        # Step 1: Convert features
        converter = TensorConverter(device='cpu')
        feature_tensor = converter.numpy_to_tensor(sample_features)

        # Step 2: Create distributed loader
        loader = DistributedDataLoader(
            sample_features.tolist(),
            batch_size=64,
            num_ranks=1,
            rank=0
        )

        batches = list(loader)
        assert len(batches) > 0


class TestPerformancePhase5Phase6:
    """Performance benchmarks for Phase 5 & 6."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_tensor_conversion_throughput(self, sample_features):
        """Test tensor conversion throughput."""
        from ign_lidar.features.pytorch_integration import TensorConverter
        import time

        converter = TensorConverter(device='cpu')

        start = time.time()
        for _ in range(100):
            tensor = converter.numpy_to_tensor(sample_features)
            converter.tensor_to_numpy(tensor)
        elapsed = time.time() - start

        throughput_ksamples_per_sec = (100 * len(sample_features)) / (elapsed * 1000)
        assert throughput_ksamples_per_sec > 10  # At least 10k samples/sec

    @pytest.mark.performance
    @pytest.mark.slow
    def test_partitioning_speed(self, sample_point_cloud):
        """Test partitioning speed."""
        from ign_lidar.optimization.distributed_processor import DistributedFeatureCompute
        import time

        compute = DistributedFeatureCompute(num_gpus=1)

        start = time.time()
        for _ in range(10):
            partitions = compute._partition_point_cloud(
                sample_point_cloud,
                strategy='spatial'
            )
        elapsed = time.time() - start

        # Should be very fast
        assert elapsed < 1.0  # Less than 1 second for 10 iterations


class TestErrorHandling:
    """Error handling and edge cases."""

    @pytest.mark.unit
    def test_invalid_device(self):
        """Test handling of invalid device."""
        from ign_lidar.features.pytorch_integration import TensorConverter

        # Should fall back to CPU
        converter = TensorConverter(device='invalid_device')
        assert converter.device.type in ['cpu', 'cuda', 'mps']

    @pytest.mark.unit
    def test_empty_array(self):
        """Test handling of empty array."""
        from ign_lidar.features.pytorch_integration import TensorConverter

        converter = TensorConverter(device='cpu')
        empty = np.array([], dtype=np.float32)

        tensor = converter.numpy_to_tensor(empty)
        assert tensor.numel() == 0

    @pytest.mark.unit
    def test_mismatched_partition_sizes(self, sample_point_cloud):
        """Test partitioning with edge case sizes."""
        from ign_lidar.optimization.distributed_processor import DistributedFeatureCompute

        compute = DistributedFeatureCompute(num_gpus=1)

        # Chunk size larger than data
        partitions = compute._partition_point_cloud(
            sample_point_cloud,
            strategy='spatial',
            chunk_size=100000
        )

        assert len(partitions) == 1
        assert len(partitions[0]) == len(sample_point_cloud)
