"""
Data augmentation for IGN LiDAR HD patches.

Provides various augmentation techniques for point cloud data:
- Random rotation
- Random scaling
- Random translation
- Random jitter
- Random dropout
- Feature noise
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PatchAugmentation:
    """
    Data augmentation for point cloud patches.
    
    Examples:
        >>> aug = PatchAugmentation({
        ...     'rotation': {'enabled': True, 'max_angle': 180},
        ...     'scaling': {'enabled': True, 'min_scale': 0.9, 'max_scale': 1.1},
        ...     'jitter': {'enabled': True, 'sigma': 0.01}
        ... })
        >>> 
        >>> augmented_patch = aug(patch)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Parse configuration
        self.rotation_enabled = self.config.get('rotation', {}).get('enabled', False)
        self.rotation_max_angle = self.config.get('rotation', {}).get('max_angle', 180)
        
        self.scaling_enabled = self.config.get('scaling', {}).get('enabled', False)
        self.scaling_min = self.config.get('scaling', {}).get('min_scale', 0.8)
        self.scaling_max = self.config.get('scaling', {}).get('max_scale', 1.2)
        
        self.translation_enabled = self.config.get('translation', {}).get('enabled', False)
        self.translation_max = self.config.get('translation', {}).get('max_offset', 0.5)
        
        self.jitter_enabled = self.config.get('jitter', {}).get('enabled', False)
        self.jitter_sigma = self.config.get('jitter', {}).get('sigma', 0.01)
        self.jitter_clip = self.config.get('jitter', {}).get('clip', 0.05)
        
        self.dropout_enabled = self.config.get('dropout', {}).get('enabled', False)
        self.dropout_ratio = self.config.get('dropout', {}).get('ratio', 0.1)
        
        self.feature_noise_enabled = self.config.get('feature_noise', {}).get('enabled', False)
        self.feature_noise_sigma = self.config.get('feature_noise', {}).get('sigma', 0.01)
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default augmentation configuration."""
        return {
            'rotation': {
                'enabled': True,
                'max_angle': 180,  # degrees around Z axis
            },
            'scaling': {
                'enabled': True,
                'min_scale': 0.9,
                'max_scale': 1.1,
            },
            'translation': {
                'enabled': False,
                'max_offset': 0.5,  # meters
            },
            'jitter': {
                'enabled': True,
                'sigma': 0.01,
                'clip': 0.05,
            },
            'dropout': {
                'enabled': False,
                'ratio': 0.1,
            },
            'feature_noise': {
                'enabled': False,
                'sigma': 0.01,
            },
        }
    
    def __call__(self, patch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply augmentation to a patch.
        
        Args:
            patch: Input patch dictionary
            
        Returns:
            Augmented patch dictionary
        """
        # Copy patch to avoid modifying original
        patch = {k: v.copy() if isinstance(v, np.ndarray) else v 
                 for k, v in patch.items()}
        
        # Extract points (XYZ)
        if 'points' not in patch:
            logger.warning("No 'points' key found in patch, skipping augmentation")
            return patch
        
        points = patch['points']
        
        # Apply transformations
        if self.rotation_enabled:
            points = self._rotate(points)
        
        if self.scaling_enabled:
            points = self._scale(points)
        
        if self.translation_enabled:
            points = self._translate(points)
        
        if self.jitter_enabled:
            points = self._jitter(points)
        
        if self.dropout_enabled:
            points, mask = self._dropout(points)
            # Apply mask to other features too
            patch = self._apply_mask(patch, mask)
        
        # Update points
        patch['points'] = points
        
        # Apply feature noise
        if self.feature_noise_enabled and 'features' in patch:
            patch['features'] = self._add_feature_noise(patch['features'])
        
        return patch
    
    def _rotate(self, points: np.ndarray) -> np.ndarray:
        """
        Random rotation around Z axis.
        
        Args:
            points: Point cloud [N, 3]
            
        Returns:
            Rotated points [N, 3]
        """
        # Random angle in radians
        angle = np.random.uniform(-self.rotation_max_angle, self.rotation_max_angle)
        angle_rad = np.deg2rad(angle)
        
        # Rotation matrix around Z axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Apply rotation
        rotated = points @ rotation_matrix.T
        
        return rotated
    
    def _scale(self, points: np.ndarray) -> np.ndarray:
        """
        Random uniform scaling.
        
        Args:
            points: Point cloud [N, 3]
            
        Returns:
            Scaled points [N, 3]
        """
        scale = np.random.uniform(self.scaling_min, self.scaling_max)
        return points * scale
    
    def _translate(self, points: np.ndarray) -> np.ndarray:
        """
        Random translation.
        
        Args:
            points: Point cloud [N, 3]
            
        Returns:
            Translated points [N, 3]
        """
        offset = np.random.uniform(
            -self.translation_max,
            self.translation_max,
            size=(3,)
        ).astype(np.float32)
        
        return points + offset
    
    def _jitter(self, points: np.ndarray) -> np.ndarray:
        """
        Add random Gaussian noise to points.
        
        Args:
            points: Point cloud [N, 3]
            
        Returns:
            Jittered points [N, 3]
        """
        noise = np.random.normal(0, self.jitter_sigma, size=points.shape).astype(np.float32)
        noise = np.clip(noise, -self.jitter_clip, self.jitter_clip)
        
        return points + noise
    
    def _dropout(self, points: np.ndarray) -> tuple:
        """
        Randomly drop points.
        
        Args:
            points: Point cloud [N, 3]
            
        Returns:
            Tuple of (dropped points, mask)
        """
        n = len(points)
        n_keep = int(n * (1 - self.dropout_ratio))
        
        # Random indices to keep
        indices = np.random.choice(n, size=n_keep, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True
        
        return points[mask], mask
    
    def _apply_mask(self, patch: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply dropout mask to all features in patch.
        
        Args:
            patch: Patch dictionary
            mask: Boolean mask
            
        Returns:
            Masked patch
        """
        for key, value in patch.items():
            if isinstance(value, np.ndarray) and len(value) == len(mask):
                patch[key] = value[mask]
        
        return patch
    
    def _add_feature_noise(self, features: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to features.
        
        Args:
            features: Feature array [N, F]
            
        Returns:
            Noisy features [N, F]
        """
        noise = np.random.normal(0, self.feature_noise_sigma, size=features.shape).astype(np.float32)
        return features + noise
    
    def __repr__(self) -> str:
        """String representation."""
        enabled = []
        if self.rotation_enabled:
            enabled.append('rotation')
        if self.scaling_enabled:
            enabled.append('scaling')
        if self.translation_enabled:
            enabled.append('translation')
        if self.jitter_enabled:
            enabled.append('jitter')
        if self.dropout_enabled:
            enabled.append('dropout')
        if self.feature_noise_enabled:
            enabled.append('feature_noise')
        
        return f"PatchAugmentation(enabled={enabled})"
