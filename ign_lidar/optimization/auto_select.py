"""
Automatic Optimization Selection

Automatically selects and applies the best available ground truth optimization.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OptimizationLevel:
    """Available optimization levels."""
    GPU = "gpu"
    VECTORIZED = "vectorized"
    STRTREE = "strtree"
    PREFILTER = "prefilter"
    ORIGINAL = "original"


def check_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    try:
        import cupy as cp
        _ = cp.array([1.0])
        return True
    except Exception:
        return False


def check_cuspatial_available() -> bool:
    """Check if cuSpatial is available."""
    try:
        import cuspatial
        return True
    except ImportError:
        return False


def check_geopandas_available() -> bool:
    """Check if GeoPandas is available."""
    try:
        import geopandas
        return True
    except ImportError:
        return False


def check_strtree_available() -> bool:
    """Check if STRtree is available."""
    try:
        from shapely.strtree import STRtree
        return True
    except ImportError:
        return False


def auto_optimize(force_level: Optional[str] = None, verbose: bool = True) -> str:
    """
    Automatically select and apply the best available optimization.
    
    Args:
        force_level: Force specific optimization level
        verbose: Print detailed information
        
    Returns:
        Name of applied optimization level
    """
    if verbose:
        logger.info("="*80)
        logger.info("GROUND TRUTH OPTIMIZATION AUTO-SELECT")
        logger.info("="*80)
    
    # Check available optimizations
    has_gpu = check_gpu_available()
    has_geopandas = check_geopandas_available()
    has_strtree = check_strtree_available()
    
    # Determine best optimization
    if force_level:
        selected_level = force_level
        if verbose:
            logger.info(f"Forced optimization: {selected_level}")
    else:
        if has_gpu:
            selected_level = OptimizationLevel.GPU
            if verbose:
                logger.info("Auto-selected: GPU acceleration (fastest)")
        elif has_geopandas:
            selected_level = OptimizationLevel.VECTORIZED
            if verbose:
                logger.info("Auto-selected: Vectorized (GeoPandas)")
        elif has_strtree:
            selected_level = OptimizationLevel.STRTREE
            if verbose:
                logger.info("Auto-selected: STRtree spatial indexing")
        else:
            selected_level = OptimizationLevel.PREFILTER
            if verbose:
                logger.info("Auto-selected: Pre-filtering")
    
    # Apply optimization
    try:
        if selected_level == OptimizationLevel.GPU:
            from .gpu import patch_advanced_classifier
            patch_advanced_classifier()
            speedup = "100-1000×"
        
        elif selected_level == OptimizationLevel.VECTORIZED:
            from .vectorized import patch_advanced_classifier
            patch_advanced_classifier()
            speedup = "30-100×"
        
        elif selected_level == OptimizationLevel.STRTREE:
            from .strtree import patch_advanced_classifier
            patch_advanced_classifier()
            speedup = "10-30×"
        
        elif selected_level == OptimizationLevel.PREFILTER:
            from .prefilter import patch_classifier
            patch_classifier()
            speedup = "2-5×"
        
        else:
            if verbose:
                logger.info("Using original implementation (no optimization)")
            speedup = "1×"
        
        if verbose:
            logger.info(f"✅ Optimization applied: {selected_level}")
            logger.info(f"   Expected speedup: {speedup}")
            logger.info("="*80)
        
        return selected_level
    
    except Exception as e:
        logger.error(f"Failed to apply {selected_level} optimization: {e}")
        
        # Try fallback
        if selected_level == OptimizationLevel.GPU and has_geopandas:
            return auto_optimize(force_level=OptimizationLevel.VECTORIZED, verbose=verbose)
        elif selected_level == OptimizationLevel.VECTORIZED and has_strtree:
            return auto_optimize(force_level=OptimizationLevel.STRTREE, verbose=verbose)
        elif selected_level == OptimizationLevel.STRTREE:
            return auto_optimize(force_level=OptimizationLevel.PREFILTER, verbose=verbose)
        else:
            logger.warning("All optimizations failed, using original implementation")
            return OptimizationLevel.ORIGINAL
