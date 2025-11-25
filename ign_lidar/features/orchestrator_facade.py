"""
Simplified Feature Orchestrator Facade - High-Level Interface

This module provides a simplified, user-friendly interface to the Feature Orchestrator,
following the same pattern as ClassificationEngine and GroundTruthProvider.

The facade wraps the existing FeatureOrchestrator to:
1. Reduce API complexity for common use cases
2. Provide sensible defaults
3. Hide internal implementation details
4. Offer both high-level (easy) and low-level (powerful) APIs

Usage:

    # High-level API (recommended for most users)
    from ign_lidar.features.orchestrator_facade import FeatureOrchestrationService
    
    service = FeatureOrchestrationService(config)
    features = service.compute_features(points, classification)
    
    # Advanced usage (full control)
    features_advanced = service.compute_with_mode(
        points,
        classification,
        mode='LOD3',
        use_gpu=True,
        use_rgb=True
    )

Architecture:

    FeatureOrchestrationService (Facade)
    └── FeatureOrchestrator (Wrapped)
        ├── RGB Fetcher
        ├── Infrared Fetcher
        └── Feature Computer (Strategy-based)

Benefits:

    ✓ Simpler API for common workflows
    ✓ Sensible defaults (LOD2, CPU by default, smart GPU detection)
    ✓ Progressive disclosure of complexity
    ✓ Full backward compatibility with FeatureOrchestrator
    ✓ Better documentation through clear parameter names

Version: 1.0.0
Date: November 25, 2025
"""

import logging
from typing import Dict, Optional, Tuple, Any
import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class FeatureOrchestrationService:
    """
    Simplified facade for feature computation operations.

    This class provides a high-level, easy-to-use interface to the complex
    FeatureOrchestrator. It handles:
    - Configuration management with sensible defaults
    - Resource initialization
    - Feature computation coordination
    - Progress reporting
    - Caching and optimization

    Example (Recommended):
        >>> config = OmegaConf.load("config.yaml")
        >>> service = FeatureOrchestrationService(config)
        >>> features = service.compute_features(points, classification)

    Example (Advanced):
        >>> features = service.compute_with_mode(
        ...     points=points,
        ...     classification=classification,
        ...     mode='LOD3',
        ...     use_gpu=True,
        ...     k_neighbors=50
        ... )
    """

    def __init__(
        self,
        config: DictConfig,
        progress_callback: Optional[callable] = None,
        verbose: bool = False,
    ):
        """
        Initialize Feature Orchestration Service.

        Args:
            config: Hydra/OmegaConf configuration with 'processor' and 'features' sections
            progress_callback: Optional callback(progress: float, message: str)
            verbose: Enable detailed logging

        Raises:
            ValueError: If configuration is invalid
            ImportError: If required dependencies are missing
        """
        self.config = config
        self.progress_callback = progress_callback
        self.verbose = verbose

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Lazy initialization of the underlying orchestrator
        self._orchestrator = None
        self._initialized = False

        logger.debug(
            "FeatureOrchestrationService initialized (lazy initialization enabled)"
        )

    @property
    def orchestrator(self):
        """
        Lazy-load the underlying FeatureOrchestrator.

        Returns:
            FeatureOrchestrator: The wrapped orchestrator instance

        Note:
            The orchestrator is only initialized on first access to reduce
            startup time when not all features are needed.
        """
        if self._orchestrator is None:
            try:
                from .orchestrator import FeatureOrchestrator

                logger.debug("Lazy-loading FeatureOrchestrator")
                self._orchestrator = FeatureOrchestrator(
                    self.config, self.progress_callback
                )
                self._initialized = True
            except ImportError as e:
                logger.error(f"Failed to import FeatureOrchestrator: {e}")
                raise

        return self._orchestrator

    # ========================================================================
    # High-Level API (Recommended for most users)
    # ========================================================================

    def compute_features(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        nir: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute features with sensible defaults.

        HIGH-LEVEL API: Recommended for most use cases.

        This method automatically selects the best strategy (CPU/GPU) based on
        available hardware and data size. It uses LOD2 feature mode by default.

        Args:
            points: Point cloud array [N, 3] with (x, y, z)
            classification: Classification labels [N]
            rgb: Optional RGB data [H, W, 3] for spectral features
            nir: Optional NIR data [H, W] for vegetation indices

        Returns:
            Dictionary of computed features:
            {
                'normals': [N, 3],
                'curvature': [N],
                'density': [N],
                'eigenvalues': [N, 3],
                ... (additional features based on mode)
            }

        Example:
            >>> points = np.random.rand(10000, 3)
            >>> classification = np.random.randint(0, 6, 10000)
            >>> features = service.compute_features(points, classification)
            >>> print(f"Computed {len(features)} feature types")

        Note:
            This method uses LOD2 mode (12 features) by default, which provides
            a good balance between speed and accuracy. Use compute_with_mode()
            for different feature sets.
        """
        logger.info(
            f"Computing features for {len(points)} points (LOD2 mode, auto strategy)"
        )

        try:
            # Delegate to orchestrator with default parameters
            features = self.orchestrator.compute_features(
                points=points,
                classification=classification,
                rgb=rgb,
                nir=nir,
            )

            logger.info(f"Successfully computed {len(features)} feature types")
            return features

        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            raise

    def compute_with_mode(
        self,
        points: np.ndarray,
        classification: np.ndarray,
        mode: str = "lod2",
        use_gpu: Optional[bool] = None,
        use_rgb: bool = False,
        use_infrared: bool = False,
        k_neighbors: int = 30,
        search_radius: float = 3.0,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Compute features with explicit mode and parameters.

        HIGH-LEVEL API: Full control with clear parameter names.

        Args:
            points: Point cloud array [N, 3]
            classification: Classification labels [N]
            mode: Feature mode ('minimal', 'lod2', 'lod3', 'asprs', 'full')
            use_gpu: Force GPU (True) or CPU (False), or auto-detect (None)
            use_rgb: Include RGB spectral features
            use_infrared: Include NIR/vegetation features
            k_neighbors: Number of neighbors for geometric features
            search_radius: Search radius in meters for neighbor queries
            **kwargs: Additional parameters passed to orchestrator

        Returns:
            Dictionary of computed features

        Example:
            >>> features = service.compute_with_mode(
            ...     points=points,
            ...     classification=classification,
            ...     mode='LOD3',
            ...     use_gpu=True,
            ...     use_rgb=True,
            ...     k_neighbors=50
            ... )

        Note:
            - Mode determines which features are computed
            - GPU acceleration is recommended for >100k points
            - RGB/NIR data enhances spectral features significantly
        """
        logger.info(
            f"Computing features (mode={mode}, gpu={use_gpu}, "
            f"rgb={use_rgb}, infrared={use_infrared})"
        )

        try:
            # Build parameters dict
            params = {
                "points": points,
                "classification": classification,
                "mode": mode.upper(),
                "use_rgb": use_rgb,
                "use_infrared": use_infrared,
                "k_neighbors": k_neighbors,
                "search_radius": search_radius,
            }

            if use_gpu is not None:
                params["use_gpu"] = use_gpu

            params.update(kwargs)

            # Delegate to orchestrator
            features = self.orchestrator.compute_features(**params)

            logger.info(f"Successfully computed {len(features)} feature types")
            return features

        except Exception as e:
            logger.error(f"Feature computation with mode '{mode}' failed: {e}")
            raise

    # ========================================================================
    # Low-Level API (For advanced users)
    # ========================================================================

    def get_orchestrator(self):
        """
        Get direct access to underlying FeatureOrchestrator.

        LOW-LEVEL API: For advanced users who need full control.

        Returns:
            FeatureOrchestrator: The wrapped orchestrator instance

        Example:
            >>> orch = service.get_orchestrator()
            >>> # Access internal methods directly
            >>> normals = orch._compute_geometric_features(points)
        """
        return self.orchestrator

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_feature_modes(self) -> Dict[str, str]:
        """
        Get available feature modes with descriptions.

        Returns:
            Dictionary mapping mode names to descriptions

        Example:
            >>> modes = service.get_feature_modes()
            >>> for mode, description in modes.items():
            ...     print(f"{mode}: {description}")
        """
        return {
            "minimal": "8 essential features (fastest)",
            "lod2": "12 features for building LOD2",
            "lod3": "38 features for building LOD3",
            "asprs": "25 features for ASPRS classification",
            "full": "All available features",
        }

    def get_optimization_info(self) -> Dict[str, Any]:
        """
        Get information about optimization choices.

        Returns:
            Dictionary with:
            {
                'strategy': 'CPU'|'GPU'|'GPU_CHUNKED'|'BOUNDARY',
                'gpu_available': bool,
                'use_rgb': bool,
                'use_infrared': bool,
                'batch_mode': bool,
                ...
            }

        Example:
            >>> info = service.get_optimization_info()
            >>> print(f"Using strategy: {info['strategy']}")
        """
        try:
            return {
                "strategy": getattr(
                    self.orchestrator, "strategy_name", "UNKNOWN"
                ),
                "gpu_available": getattr(
                    self.orchestrator, "gpu_available", False
                ),
                "feature_mode": getattr(
                    self.orchestrator, "feature_mode", "UNKNOWN"
                ),
                "initialized": self._initialized,
            }
        except Exception as e:
            logger.warning(f"Could not get optimization info: {e}")
            return {}

    def clear_cache(self):
        """
        Clear internal feature cache.

        Use this to free memory after processing large batches.

        Example:
            >>> service.compute_features(points1, labels1)
            >>> service.clear_cache()  # Free memory
            >>> service.compute_features(points2, labels2)
        """
        if self._orchestrator is not None:
            try:
                self.orchestrator.clear_cache()
                logger.debug("Feature cache cleared")
            except AttributeError:
                logger.debug("Orchestrator does not support cache clearing")

    def get_performance_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics from feature computation.

        Returns:
            Dictionary with performance information, or None if unavailable

        Example:
            >>> metrics = service.get_performance_summary()
            >>> if metrics:
            ...     print(f"Average time per point: {metrics['avg_time_per_point']:.3f}ms")
        """
        if self._orchestrator is None:
            return None

        try:
            return self.orchestrator.get_performance_summary()
        except AttributeError:
            logger.debug("Orchestrator does not provide performance summary")
            return None

    def __repr__(self) -> str:
        """String representation."""
        initialized_str = "initialized" if self._initialized else "lazy"
        return f"FeatureOrchestrationService({initialized_str})"
