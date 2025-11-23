"""
Hybrid formatter for deep learning.

Formats IGN LiDAR HD patches for hybrid/ensemble models.
Saves all architecture formats in a single file for maximum flexibility.

GPU Acceleration: Set use_gpu=True for 10-20x speedup on KNN graph construction.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

from .base_formatter import BaseFormatter

# Import centralized GPU manager
from ...core.gpu import GPUManager

logger = logging.getLogger(__name__)

# Check GPU availability using centralized manager
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    import cupy as cp
    if _gpu_manager.cuml_available:
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    else:
        cuNearestNeighbors = None
else:
    cp = None
    cuNearestNeighbors = None


class HybridFormatter(BaseFormatter):
    """
    Format patches for hybrid/ensemble models.
    
    Returns a single comprehensive dictionary containing:
    - All architecture-specific formats (PointNet++, Octree, Transformer, Sparse Conv)
    - Common metadata
    - Individual feature access (RGB, NIR, normals, etc.)
    
    This allows training ensemble models or easily switching between architectures
    without regenerating patches.
    """

    def __init__(
        self,
        num_points: int = 16384,
        octree_depth: int = 6,
        knn_k: int = 32,
        voxel_size: float = 0.1,
        normalize: bool = True,
        standardize_features: bool = True,
        # Feature flags
        use_rgb: bool = True,
        use_infrared: bool = False,
        use_geometric: bool = True,
        use_radiometric: bool = False,
        use_contextual: bool = False
    ):
        """
        Initialize hybrid formatter.
        
        Args:
            num_points: Target number of points
            octree_depth: Depth for octree structure
            knn_k: Number of neighbors for KNN graph
            voxel_size: Voxel size for sparse convolutions
            normalize: Normalize XYZ coordinates
            standardize_features: Standardize features (mean=0, std=1)
            use_rgb: Include RGB features
            use_infrared: Include infrared/NDVI features
            use_geometric: Include geometric features
            use_radiometric: Include radiometric features
            use_contextual: Include contextual features
        """
        super().__init__(
            num_points=num_points,
            normalize=normalize,
            standardize_features=standardize_features
        )
        
        self.octree_depth = octree_depth
        self.knn_k = knn_k
        self.voxel_size = voxel_size
        
        # Feature flags
        self.use_rgb = use_rgb
        self.use_infrared = use_infrared
        self.use_geometric = use_geometric
        self.use_radiometric = use_radiometric
        self.use_contextual = use_contextual

    def format_patch(self, patch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Format patch for hybrid/ensemble models.
        
        Returns a comprehensive dictionary with all formats and features.
        
        Args:
            patch: Input patch dictionary
            
        Returns:
            Dictionary containing:
            - points: [N, 3] XYZ coordinates (normalized)
            - features: [N, C] concatenated features
            - labels: [N] point labels
            - rgb: [N, 3] RGB colors (if available)
            - nir: [N, 1] near-infrared (if available)
            - ndvi: [N, 1] NDVI (if available)
            - normals: [N, 3] normal vectors (if available)
            - octree: octree structure (if spatial index available)
            - knn_graph: [N, K, 2] KNN graph (if spatial index available)
            - metadata: comprehensive metadata
        """
        # DEBUG: Log features received in patch
        import logging
        logger = logging.getLogger(__name__)
        feature_keys = [k for k in patch.keys() if not k.startswith('_')]
        logger.debug(f"  📊 DEBUG HybridFormatter: Received patch with {len(feature_keys)} feature arrays")
        logger.debug(f"      Keys: {', '.join(sorted(feature_keys)[:15])}...")
        
        points = patch['points']
        features, feature_names = self._build_feature_matrix(
            patch,
            use_rgb=self.use_rgb,
            use_infrared=self.use_infrared,
            use_geometric=self.use_geometric,
            use_radiometric=self.use_radiometric,
            use_contextual=self.use_contextual,
            return_feature_names=True
        )
        
        # Sample points if needed
        if len(points) > self.num_points:
            indices = self._fps_sampling(points, self.num_points)
            points = points[indices]
            features = features[indices]
            labels = patch.get('labels')
            if labels is not None:
                labels = labels[indices]
        else:
            labels = patch.get('labels')
            
        # Normalize coordinates
        if self.normalize:
            points_norm = self._normalize_xyz(points)
        else:
            points_norm = points.astype(np.float32)
            
        # Standardize features
        if self.standardize_features:
            features_norm = self._normalize_features(features)
        else:
            features_norm = features
            
        # Build comprehensive output
        metadata = self._extract_metadata(patch)
        metadata['feature_names'] = feature_names  # Add feature names to metadata
        metadata['num_features'] = len(feature_names)
        
        output = {
            # Core data (required)
            'points': points_norm,
            'features': features_norm,
            'labels': labels if labels is not None else np.zeros(len(points), dtype=np.int32),
            
            # Metadata
            'metadata': metadata,
        }
        
        # Add individual features for flexible access
        if self.use_rgb and 'rgb' in patch:
            rgb = patch['rgb']
            if len(rgb) > self.num_points and len(points) == self.num_points:
                rgb = rgb[indices]
            output['rgb'] = rgb.astype(np.float32)
            
        if self.use_infrared:
            if 'nir' in patch:
                nir = patch['nir']
                if len(nir) > self.num_points and len(points) == self.num_points:
                    nir = nir[indices]
                output['nir'] = nir.astype(np.float32)
                
            if 'ndvi' in patch:
                ndvi = patch['ndvi']
                if len(ndvi) > self.num_points and len(points) == self.num_points:
                    ndvi = ndvi[indices]
                output['ndvi'] = ndvi.astype(np.float32)
                
        if self.use_geometric:
            if 'normals' in patch:
                normals = patch['normals']
                if len(normals) > self.num_points and len(points) == self.num_points:
                    normals = normals[indices]
                output['normals'] = normals.astype(np.float32)
                
            # Include other geometric features
            for geo_feat in ['curvature', 'planarity', 'linearity', 'sphericity', 
                           'verticality', 'eigenvalues', 'height_above_ground']:
                if geo_feat in patch:
                    feat = patch[geo_feat]
                    if len(feat) > self.num_points and len(points) == self.num_points:
                        feat = feat[indices]
                    output[geo_feat] = feat.astype(np.float32)
                    
        # Add spatial structures if available
        if 'octree' in patch:
            output['octree'] = patch['octree']
            
        if 'knn_graph' in patch:
            knn_graph = patch['knn_graph']
            if len(knn_graph) > self.num_points and len(points) == self.num_points:
                # Need to rebuild KNN graph for sampled points
                output['knn_graph'] = self._build_knn_graph(points_norm, self.knn_k)
            else:
                output['knn_graph'] = knn_graph
        elif self.use_geometric:
            # Build KNN graph for transformer architectures
            output['knn_graph'] = self._build_knn_graph(points_norm, self.knn_k)
            
        # Add voxel representation for sparse convolutions
        if self.voxel_size > 0:
            voxel_coords, voxel_features, voxel_labels = self._voxelize(
                points_norm, features_norm, 
                labels if labels is not None else np.zeros(len(points), dtype=np.int32)
            )
            output['voxel_coords'] = voxel_coords
            output['voxel_features'] = voxel_features
            output['voxel_labels'] = voxel_labels
            
        return output
        
    def _build_knn_graph_gpu(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        GPU-accelerated KNN graph using KNNEngine (unified API).
        
        Args:
            points: [N, 3] point coordinates
            k: number of neighbors
            
        Returns:
            [N, K, 2] edge indices
            
        Note:
            Now uses KNNEngine for automatic backend selection (FAISS-GPU, cuML, etc.)
            This replaces the manual cuML implementation with the unified API.
        """
        from ...optimization import KNNEngine
        
        # Use KNNEngine with GPU backend (auto-selects FAISS-GPU or cuML)
        engine = KNNEngine()
        
        # Query k+1 neighbors (includes self)
        distances, indices = engine.query(points, k=k+1, use_gpu=True)
        
        # Build edge list (skip self-connection at index 0)
        n_points = len(points)
        edges = np.zeros((n_points, k, 2), dtype=np.int32)
        
        for j in range(k):
            edges[:, j, 0] = np.arange(n_points)
            edges[:, j, 1] = indices[:, j+1]  # Skip j=0 (self)
        
        return edges

    def _build_knn_graph(self, points: np.ndarray, k: int, use_gpu: bool = False) -> np.ndarray:
        """
        Build KNN graph using unified KNNEngine API.
        
        **GPU Acceleration**: Set use_gpu=True for 10-20x speedup on large point clouds.
        
        Args:
            points: [N, 3] point coordinates
            k: number of neighbors
            use_gpu: Use GPU acceleration (default False)
            
        Returns:
            [N, K, 2] edge indices
            
        Note:
            Now uses KNNEngine for automatic backend selection.
            This provides better performance and avoids code duplication.
        """
        from ...optimization import KNNEngine
        
        # Use unified KNNEngine (auto-selects optimal backend)
        engine = KNNEngine()
        
        try:
            # Query k+1 neighbors (includes self)
            distances, indices = engine.query(points, k=k+1, use_gpu=use_gpu)
        except Exception as e:
            logger.warning(f"KNN engine failed ({e}), using fallback")
            # Fallback to sklearn
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points)
            distances, indices = nbrs.kneighbors(points)
        
        # Build edge list (skip self-connection at index 0)
        n_points = len(points)
        edges = np.zeros((n_points, k, 2), dtype=np.int32)
        
        for j in range(k):
            edges[:, j, 0] = np.arange(n_points)
            edges[:, j, 1] = indices[:, j+1]  # Skip j=0 (self)
                
        return edges
        
    def _voxelize(
        self, 
        points: np.ndarray, 
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxelize point cloud for sparse convolutions.
        
        Args:
            points: [N, 3] coordinates
            features: [N, C] features
            labels: [N] labels
            
        Returns:
            voxel_coords: [M, 3] voxel coordinates
            voxel_features: [M, C] averaged features
            voxel_labels: [M] majority labels
        """
        # Quantize coordinates
        voxel_coords = np.floor(points / self.voxel_size).astype(np.int32)
        
        # Find unique voxels
        unique_coords, inverse_indices = np.unique(
            voxel_coords, axis=0, return_inverse=True
        )
        
        # Average features per voxel
        num_voxels = len(unique_coords)
        voxel_features = np.zeros((num_voxels, features.shape[1]), dtype=np.float32)
        voxel_labels = np.zeros(num_voxels, dtype=np.int32)
        
        for i in range(num_voxels):
            mask = inverse_indices == i
            voxel_features[i] = features[mask].mean(axis=0)
            # Majority vote for labels
            if labels[mask].size > 0:
                voxel_labels[i] = np.bincount(labels[mask]).argmax()
                
        return unique_coords, voxel_features, voxel_labels
