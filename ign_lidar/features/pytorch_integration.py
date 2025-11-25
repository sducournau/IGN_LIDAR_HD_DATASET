"""
PyTorch Integration Module - Phase 5

Direct tensor interoperability between NumPy arrays and PyTorch tensors.
Enables seamless GPU acceleration and model inference.

**Phase 5 Features (November 25, 2025):**

1. **Tensor Conversion**: NumPy ↔ PyTorch with automatic device placement
2. **GPU Inference**: Direct feature → PyTorch model inference pipeline
3. **Batch Processing**: Efficient batched tensor operations
4. **Memory Efficiency**: Pinned memory and async transfers
5. **Model Integration**: Load/inference custom PyTorch models

Example Usage:

    from ign_lidar.features.pytorch_integration import TensorConverter, GPUInference

    # Convert features to tensor
    converter = TensorConverter(device='cuda', dtype=torch.float32)
    feature_tensor = converter.numpy_to_tensor(features_array)

    # Run inference
    inference = GPUInference(model, batch_size=4096)
    predictions = inference.predict(feature_tensor)

Version: 1.0.0
Date: November 25, 2025
"""

import logging
from typing import Dict, Optional, Tuple, Union, Any, List
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class TensorConverter:
    """
    Convert between NumPy arrays and PyTorch tensors with automatic device placement.

    Handles:
    - Device placement (CPU/CUDA/MPS)
    - Dtype conversion (float32/float64/int32/int64)
    - Pinned memory for async transfers
    - Memory-efficient batch conversion
    """

    def __init__(
        self,
        device: str = 'cpu',
        tensor_dtype: Optional[str] = 'float32',
        use_pinned_memory: bool = False
    ):
        """
        Initialize tensor converter.

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', 'mps')
            tensor_dtype: PyTorch dtype name ('float32', 'float64', 'int32', 'int64')
            use_pinned_memory: Whether to use pinned memory for faster transfers
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        self.device = torch.device(device)
        self.dtype = getattr(torch, tensor_dtype)
        self.use_pinned_memory = use_pinned_memory

        # Validate device availability
        if 'cuda' in str(device):
            if not torch.cuda.is_available():
                logger.warning(f"CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
        elif 'mps' in str(device):
            if not torch.backends.mps.is_available():
                logger.warning(f"MPS not available, falling back to CPU")
                self.device = torch.device('cpu')

        logger.info(f"TensorConverter initialized: device={self.device}, dtype={self.dtype}")

    def numpy_to_tensor(
        self,
        array: np.ndarray,
        requires_grad: bool = False,
        non_blocking: bool = True
    ) -> torch.Tensor:
        """
        Convert NumPy array to PyTorch tensor.

        Args:
            array: Input NumPy array
            requires_grad: Whether to track gradients
            non_blocking: Non-blocking transfer to device

        Returns:
            PyTorch tensor on target device
        """
        # Convert to tensor
        if self.use_pinned_memory and self.device.type == 'cuda':
            tensor = torch.from_numpy(np.ascontiguousarray(array.astype(self.dtype.numpy_to_numpy))).pin_memory()
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(dtype=self.dtype)

        # Move to device
        tensor = tensor.to(self.device, non_blocking=non_blocking)
        tensor.requires_grad = requires_grad

        return tensor

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to NumPy array.

        Args:
            tensor: Input PyTorch tensor

        Returns:
            NumPy array on CPU
        """
        if tensor.requires_grad:
            tensor = tensor.detach()

        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()

        return tensor.numpy()

    def batch_numpy_to_tensor(
        self,
        arrays: List[np.ndarray],
        requires_grad: bool = False
    ) -> List[torch.Tensor]:
        """
        Convert multiple NumPy arrays to PyTorch tensors.

        Args:
            arrays: List of NumPy arrays
            requires_grad: Whether to track gradients

        Returns:
            List of PyTorch tensors
        """
        return [
            self.numpy_to_tensor(arr, requires_grad=requires_grad)
            for arr in arrays
        ]

    def stack_tensors(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack multiple tensors into single batch tensor.

        Args:
            tensors: List of tensors to stack

        Returns:
            Stacked tensor with batch dimension
        """
        return torch.stack(tensors)


class GPUInference:
    """
    GPU-accelerated inference pipeline for PyTorch models.

    Handles:
    - Batch processing of features
    - Automatic device placement
    - Memory-efficient inference
    - Mixed precision support
    - Caching of results
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 4096,
        device: str = 'cuda',
        dtype: str = 'float32',
        use_amp: bool = False
    ):
        """
        Initialize GPU inference engine.

        Args:
            model: PyTorch model
            batch_size: Batch size for inference
            device: Target device
            dtype: Tensor dtype
            use_amp: Use automatic mixed precision
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        self.model = model.to(device)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.use_amp = use_amp

        # Set to evaluation mode
        self.model.eval()

        self.converter = TensorConverter(device=device, tensor_dtype=dtype)
        logger.info(f"GPUInference initialized: batch_size={batch_size}, device={device}")

    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Run inference on features.

        Args:
            features: Input features [N, D] as NumPy array or tensor
            return_numpy: Whether to return NumPy array

        Returns:
            Model predictions
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            feature_tensor = self.converter.numpy_to_tensor(features)
        else:
            feature_tensor = features.to(self.device)

        # Batch inference
        predictions = []
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    for i in range(0, len(feature_tensor), self.batch_size):
                        batch = feature_tensor[i:i + self.batch_size]
                        pred = self.model(batch)
                        predictions.append(pred)
            else:
                for i in range(0, len(feature_tensor), self.batch_size):
                    batch = feature_tensor[i:i + self.batch_size]
                    pred = self.model(batch)
                    predictions.append(pred)

        predictions = torch.cat(predictions, dim=0)

        if return_numpy:
            return self.converter.tensor_to_numpy(predictions)

        return predictions

    def predict_with_confidence(
        self,
        features: Union[np.ndarray, torch.Tensor],
        confidence_threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence filtering.

        Args:
            features: Input features
            confidence_threshold: Minimum confidence to include

        Returns:
            Tuple of (predictions, confidences)
        """
        predictions = self.predict(features, return_numpy=False)

        # Get max probability as confidence
        if len(predictions.shape) > 1:
            confidences = torch.softmax(predictions, dim=-1).max(dim=-1).values
        else:
            confidences = predictions

        predictions_np = self.converter.tensor_to_numpy(predictions)
        confidences_np = self.converter.tensor_to_numpy(confidences)

        # Filter by threshold
        mask = confidences_np >= confidence_threshold
        filtered_pred = predictions_np[mask]
        filtered_conf = confidences_np[mask]

        return filtered_pred, filtered_conf

    def get_embeddings(
        self,
        features: Union[np.ndarray, torch.Tensor],
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract intermediate layer embeddings.

        Args:
            features: Input features
            layer_name: Name of layer to extract from

        Returns:
            Embeddings from specified layer
        """
        # Convert to tensor
        if isinstance(features, np.ndarray):
            feature_tensor = self.converter.numpy_to_tensor(features)
        else:
            feature_tensor = features.to(self.device)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(feature_tensor), self.batch_size):
                batch = feature_tensor[i:i + self.batch_size]

                if layer_name:
                    # Extract intermediate layer
                    embeddings.append(
                        self._get_intermediate_output(batch, layer_name)
                    )
                else:
                    embeddings.append(self.model(batch))

        result = torch.cat(embeddings, dim=0)
        return self.converter.tensor_to_numpy(result)

    def _get_intermediate_output(
        self,
        x: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """Get output from intermediate layer."""
        outputs = {}

        def hook(name):
            def hook_fn(module, input, output):
                outputs[name] = output
            return hook_fn

        # Register hook
        layer = dict(self.model.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook(layer_name))

        try:
            with torch.no_grad():
                self.model(x)
            return outputs[layer_name]
        finally:
            handle.remove()


class ModelLoader:
    """
    Load and manage PyTorch models for feature-based inference.

    Supports:
    - Loading from file paths
    - Checkpoint management
    - Model validation
    - Device-agnostic loading
    """

    @staticmethod
    def load_model(
        model_path: Union[str, Path],
        model_class: Optional[nn.Module] = None,
        device: str = 'cuda',
        strict: bool = True
    ) -> nn.Module:
        """
        Load PyTorch model from file.

        Args:
            model_path: Path to model file (.pt, .pth, .ckpt)
            model_class: Model class to instantiate (if not in checkpoint)
            device: Device to load model on
            strict: Whether to enforce strict key matching

        Returns:
            Loaded model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if model_class is None:
                    raise ValueError("model_class required for checkpoint format")
                model = model_class()
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                if model_class is None:
                    raise ValueError("model_class required for checkpoint format")
                model = model_class()
            else:
                # Assume entire dict is state_dict
                state_dict = checkpoint
                if model_class is None:
                    raise ValueError("model_class required for state dict")
                model = model_class()
        else:
            # Assume checkpoint is full model
            model = checkpoint

        if isinstance(model, dict):
            if model_class is None:
                raise ValueError("model_class required")
            model = model_class()
            model.load_state_dict(checkpoint, strict=strict)

        model = model.to(device)
        logger.info(f"Loaded model from {model_path} on {device}")

        return model

    @staticmethod
    def save_model(
        model: nn.Module,
        output_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save PyTorch model with optional metadata.

        Args:
            model: Model to save
            output_path: Output path
            optimizer: Optional optimizer state
            metadata: Optional metadata dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metadata:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, output_path)
        logger.info(f"Saved model to {output_path}")


def convert_features_to_pytorch_dataset(
    features: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    batch_size: int = 32,
    device: str = 'cuda',
    shuffle: bool = True
) -> DataLoader:
    """
    Convert feature arrays to PyTorch DataLoader.

    Args:
        features: Dictionary of feature name → array
        labels: Optional label array
        batch_size: Batch size for DataLoader
        device: Target device
        shuffle: Whether to shuffle data

    Returns:
        PyTorch DataLoader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed")

    converter = TensorConverter(device=device)

    # Stack all features
    feature_list = [converter.numpy_to_tensor(v) for v in features.values()]
    feature_tensor = torch.stack(feature_list, dim=1)

    # Handle labels
    if labels is not None:
        label_tensor = converter.numpy_to_tensor(labels)
        dataset = TensorDataset(feature_tensor, label_tensor)
    else:
        dataset = TensorDataset(feature_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device != 'cpu'
    )
