"""
Base formatter for point cloud data.

Provides common functionality for all architecture-specific formatters.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path


class BaseFormatter:
    """
    Base class for point cloud formatters.
    
    Provides common methods for feature extraction, normalization,
    and data preparation for deep learning architectures.
    """
    
    def __init__(
        self,
        num_points: int = 16384,
        normalize: bool = True,
        standardize_features: bool = True
    ):
        """
        Initialize base formatter.
        
        Args:
            num_points: Number of points per patch
            normalize: Whether to normalize XYZ coordinates
            standardize_features: Whether to standardize features (mean=0, std=1)
        """
        self.num_points = num_points
        self.normalize = normalize
        self.standardize_features = standardize_features
    
    def _normalize_xyz(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize XYZ coordinates to unit sphere.
        
        Args:
            points: [N, 3] array of XYZ coordinates
            
        Returns:
            Normalized points in unit sphere
        """
        # Center at origin
        centroid = points.mean(axis=0, keepdims=True)
        points_centered = points - centroid
        
        # Scale to unit sphere
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        if max_dist > 1e-8:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
            
        return points_normalized.astype(np.float32)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Standardize features (mean=0, std=1).
        
        Args:
            features: [N, C] array of features
            
        Returns:
            Standardized features
        """
        if features.shape[1] == 0:
            return features
            
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        
        features_norm = (features - mean) / std
        return features_norm.astype(np.float32)
    
    def _normalize_rgb(self, rgb: np.ndarray, mode: str = 'standard') -> np.ndarray:
        """
        Normalize RGB values.
        
        Args:
            rgb: [N, 3] RGB array
            mode: Normalization mode ('standard', 'minmax', or 'none')
            
        Returns:
            Normalized RGB
        """
        if mode == 'standard':
            # Standardize: mean=0, std=1
            mean = rgb.mean(axis=0, keepdims=True)
            std = rgb.std(axis=0, keepdims=True) + 1e-8
            return ((rgb - mean) / std).astype(np.float32)
        elif mode == 'minmax':
            # Min-max to [0, 1]
            return (rgb / 255.0).astype(np.float32)
        else:
            # No normalization
            return rgb.astype(np.float32)
    
    def _build_feature_matrix(
        self,
        patch: Dict[str, np.ndarray],
        use_rgb: bool = True,
        use_infrared: bool = True,
        use_geometric: bool = True,
        use_radiometric: bool = False,
        use_contextual: bool = False
    ) -> np.ndarray:
        """
        Build feature matrix from patch data with multi-modal features.
        
        Features included (configurable):
        - RGB (3): R, G, B colors from orthophotos
        - Infrared (2): NIR + NDVI for vegetation
        - Geometric (13): normals, curvature, planarity, verticality, etc.
        - Radiometric (5): intensity, return_number, etc.
        - Contextual (6): local_density, height_stats, etc.
        
        Args:
            patch: Dict with point cloud data
            use_*: Flags to enable/disable feature groups
            
        Returns:
            features: [N, C] array with concatenated features
        """
        features = []
        
        # 1. RGB features (3 channels) 🎨
        if use_rgb and 'rgb' in patch:
            rgb = patch['rgb']  # [N, 3]
            # Normalize to [0, 1]
            rgb_norm = self._normalize_rgb(rgb, mode='minmax')
            features.append(rgb_norm)
        
        # 2. Infrared features (2 channels) 🌡️
        if use_infrared:
            if 'nir' in patch:
                nir = patch['nir']  # [N, 1]
                nir_norm = (nir.astype(np.float32) / 255.0)
                features.append(nir_norm)
            
            if 'ndvi' in patch:
                ndvi = patch['ndvi']  # [N, 1] ∈ [-1, 1]
                features.append(ndvi.astype(np.float32))
        
        # 3. Geometric features (13 channels) 📐
        if use_geometric:
            geom_features = []
            
            # Normals (3)
            if 'normals' in patch:
                geom_features.append(patch['normals'].astype(np.float32))
            
            # Scalar geometric features
            scalar_geom = [
                'curvature',           # (1)
                'planarity',           # (1)
                'linearity',           # (1)
                'sphericity',          # (1)
                'verticality',         # (1)
                'local_density',       # (1)
                'height_above_ground'  # (1)
            ]
            
            for feat_name in scalar_geom:
                if feat_name in patch:
                    feat = patch[feat_name]
                    if feat.ndim == 1:
                        feat = feat[:, np.newaxis]  # [N] → [N, 1]
                    geom_features.append(feat.astype(np.float32))
            
            # Eigenvalues (3)
            if 'eigenvalues' in patch:
                geom_features.append(patch['eigenvalues'].astype(np.float32))
            
            if geom_features:
                features.append(np.concatenate(geom_features, axis=1))
        
        # 4. Radiometric features (5 channels) 📊
        if use_radiometric:
            radio_features = []
            
            radio_keys = [
                'intensity',      # (1)
                'return_number',  # (1)
                'num_returns',    # (1)
                'scan_angle',     # (1)
                'classification'  # (1)
            ]
            
            for feat_name in radio_keys:
                if feat_name in patch:
                    feat = patch[feat_name]
                    if feat.ndim == 1:
                        feat = feat[:, np.newaxis]
                    # Normalize
                    if feat_name == 'intensity':
                        feat = feat.astype(np.float32) / 255.0
                    elif feat_name in ['return_number', 'num_returns']:
                        feat = feat.astype(np.float32) / 5.0  # Max 5 returns
                    elif feat_name == 'scan_angle':
                        feat = feat.astype(np.float32) / 90.0  # Max ±90°
                    radio_features.append(feat)
            
            if radio_features:
                features.append(np.concatenate(radio_features, axis=1))
        
        # 5. Contextual features (6 channels) 🏗️
        if use_contextual:
            context_features = []
            
            context_keys = [
                'height_range',       # (1)
                'height_std',         # (1)
                'num_neighbors',      # (1)
                'distance_to_ground', # (1)
                'relative_height'     # (1)
            ]
            
            for feat_name in context_keys:
                if feat_name in patch:
                    feat = patch[feat_name]
                    if feat.ndim == 1:
                        feat = feat[:, np.newaxis]
                    context_features.append(feat.astype(np.float32))
            
            if context_features:
                features.append(np.concatenate(context_features, axis=1))
        
        # Concatenate all features
        if not features:
            # Fallback: use zeros if no features
            return np.zeros((len(patch['points']), 1), dtype=np.float32)
        
        feature_matrix = np.concatenate(features, axis=1)
        return feature_matrix.astype(np.float32)
    
    def _extract_metadata(self, patch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Extract metadata from patch.
        
        Args:
            patch: Patch dictionary
            
        Returns:
            Metadata dict
        """
        metadata = {
            'num_points': len(patch['points']),
            'has_rgb': 'rgb' in patch,
            'has_nir': 'nir' in patch,
            'has_ndvi': 'ndvi' in patch,
            'has_normals': 'normals' in patch,
            'has_labels': 'labels' in patch,
        }
        
        # Bounding box
        if 'points' in patch:
            points = patch['points']
            metadata['bbox_min'] = points.min(axis=0).tolist()
            metadata['bbox_max'] = points.max(axis=0).tolist()
            metadata['centroid'] = points.mean(axis=0).tolist()
        
        return metadata
    
    def format_patch(self, patch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Format patch for specific architecture.
        
        To be implemented by subclasses.
        
        Args:
            patch: Input patch dictionary
            
        Returns:
            Formatted patch
        """
        raise NotImplementedError("Subclasses must implement format_patch()")
