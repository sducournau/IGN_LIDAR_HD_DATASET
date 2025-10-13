"""
Base formatter for point cloud data.

Provides common functionality for all architecture-specific formatters.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
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
    
    def _normalize_features(self, features: np.ndarray, exclude_normals: bool = True) -> np.ndarray:
        """
        Standardize features (mean=0, std=1).
        
        IMPORTANT: Normals (first 3 dimensions if present) should NOT be standardized
        as they are already unit vectors with geometric meaning. Standardizing them
        causes quantization artifacts and stripe patterns in derived features.
        
        Args:
            features: [N, C] array of features
            exclude_normals: If True, skip standardization for first 3 dims (normals)
            
        Returns:
            Standardized features
        """
        if features.shape[1] == 0:
            return features
        
        # Check if first 3 dimensions are likely normals (unit vectors)
        # Normals should have values roughly in [-1, 1] and norm close to 1
        likely_normals = False
        if exclude_normals and features.shape[1] >= 3:
            first_3_cols = features[:, :3]
            norms = np.linalg.norm(first_3_cols, axis=1)
            # Check if most norms are close to 1 (within 20% tolerance)
            if np.mean(np.abs(norms - 1.0) < 0.2) > 0.5:
                likely_normals = True
        
        if likely_normals:
            # Standardize only non-normal features (dims 3+)
            if features.shape[1] > 3:
                # Keep normals as-is
                normals = features[:, :3].copy()
                other_features = features[:, 3:]
                
                # Standardize other features
                mean = other_features.mean(axis=0, keepdims=True)
                std = other_features.std(axis=0, keepdims=True) + 1e-8
                other_features_norm = (other_features - mean) / std
                
                # Concatenate back
                features_norm = np.concatenate([normals, other_features_norm], axis=1)
            else:
                # Only normals, no standardization needed
                features_norm = features.copy()
        else:
            # Standard normalization for all features
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
        use_contextual: bool = False,
        return_feature_names: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Build feature matrix from patch data with multi-modal features.
        
        Features included (configurable):
        - RGB (3): R, G, B colors from orthophotos
        - Infrared (2): NIR + NDVI for vegetation
        - Geometric (40+): ALL computed geometric features in full mode:
          * Normals (3): normal_x, normal_y, normal_z
          * Shape descriptors (6): planarity, linearity, sphericity, anisotropy, roughness, omnivariance
          * Curvature (2): curvature, change_curvature
          * Eigenvalues (5): eigenvalue_1/2/3, sum_eigenvalues, eigenentropy
          * Heights (3): height_above_ground, vertical_std, z_normalized
          * Building scores (3): verticality, wall_score, roof_score
          * Density (5): density, local_density, num_points_2m, neighborhood_extent, height_extent_ratio
          * Architectural (4): edge_strength, corner_likelihood, overhang_indicator, surface_roughness
          * Additional (6): z_absolute, z_from_ground, z_from_median, distance_to_center, horizontality, local_roughness
        - Radiometric (5): intensity, return_number, etc. (optional)
        - Contextual (6): local_density, height_stats, etc. (optional)
        
        Args:
            patch: Dict with point cloud data
            use_*: Flags to enable/disable feature groups
            return_feature_names: If True, return (features, feature_names) tuple
            
        Returns:
            features: [N, C] array with concatenated features (C varies based on mode: ~12-50+ features)
            OR (features, feature_names) if return_feature_names=True
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # DEBUG: Log what's in the patch
        patch_keys = [k for k in patch.keys() if not k.startswith('_')]
        logger.debug(f"  📊 DEBUG _build_feature_matrix: Building features from {len(patch_keys)} arrays in patch")
        
        features = []
        feature_names = []
        
        # 1. RGB features (3 channels) 🎨
        if use_rgb and 'rgb' in patch:
            rgb = patch['rgb']  # [N, 3]
            # Normalize to [0, 1]
            rgb_norm = self._normalize_rgb(rgb, mode='minmax')
            features.append(rgb_norm)
            feature_names.extend(['red', 'green', 'blue'])
        
        # 2. Infrared features (2 channels) 🌡️
        if use_infrared:
            if 'nir' in patch:
                nir = patch['nir']  # [N] or [N, 1]
                if nir.ndim == 1:
                    nir = nir[:, np.newaxis]  # Ensure [N, 1]
                nir_norm = (nir.astype(np.float32) / 255.0)
                features.append(nir_norm)
                feature_names.append('nir')
            
            if 'ndvi' in patch:
                ndvi = patch['ndvi']  # [N] or [N, 1] ∈ [-1, 1]
                if ndvi.ndim == 1:
                    ndvi = ndvi[:, np.newaxis]  # Ensure [N, 1]
                features.append(ndvi.astype(np.float32))
                feature_names.append('ndvi')
        
        # 3. Geometric features (ALL computed features) 📐
        if use_geometric:
            geom_features = []
            
            # Normals (3)
            if 'normals' in patch:
                geom_features.append(patch['normals'].astype(np.float32))
                feature_names.extend(['normal_x', 'normal_y', 'normal_z'])
            
            # ALL scalar geometric features (in consistent order for reproducibility)
            # This ensures FULL mode includes all 35+ computed features
            scalar_geom_ordered = [
                # Core shape descriptors (6)
                'planarity', 'linearity', 'sphericity',
                'anisotropy', 'roughness', 'omnivariance',
                
                # Curvature features (2)
                'curvature', 'change_curvature',
                
                # Eigenvalue features (5)
                'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
                'sum_eigenvalues', 'eigenentropy',
                
                # Height features (3)
                'height_above_ground', 'vertical_std', 'z_normalized',
                
                # Building scores (3)
                'verticality', 'wall_score', 'roof_score',
                
                # Density features (4)
                'density', 'local_density', 'num_points_2m', 
                'neighborhood_extent', 'height_extent_ratio',
                
                # Architectural features (4)
                'edge_strength', 'corner_likelihood', 
                'overhang_indicator', 'surface_roughness',
                
                # Additional height features (3)
                'z_absolute', 'z_from_ground', 'z_from_median',
                'distance_to_center', 'local_roughness',
                'horizontality',
            ]
            
            for feat_name in scalar_geom_ordered:
                if feat_name in patch:
                    feat = patch[feat_name]
                    if feat.ndim == 1:
                        feat = feat[:, np.newaxis]  # [N] → [N, 1]
                    geom_features.append(feat.astype(np.float32))
                    feature_names.append(feat_name)
            
            # Eigenvalues (3) - if stored as separate array
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
            fallback = np.zeros((len(patch['points']), 1), dtype=np.float32)
            if return_feature_names:
                return fallback, ['zero_placeholder']
            return fallback
        
        feature_matrix = np.concatenate(features, axis=1)
        
        if return_feature_names:
            return feature_matrix.astype(np.float32), feature_names
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
