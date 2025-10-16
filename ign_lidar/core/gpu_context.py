"""
GPU Context Management for Multiprocessing
Handles CUDA context initialization in worker processes
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def initialize_gpu_context_for_worker() -> bool:
    """
    Initialize GPU context safely for a worker process.
    Should be called at the start of each worker process.
    
    Returns:
        bool: True if GPU context was initialized successfully, False otherwise
    """
    try:
        # Import here to avoid issues if CuPy is not available
        import cupy as cp
        
        # Force CUDA context initialization
        device = cp.cuda.Device()
        device.use()
        
        # Test basic operation to ensure context is working
        test_array = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        result = cp.asnumpy(test_array)
        
        # Clear test array to free memory
        del test_array
        cp.get_default_memory_pool().free_all_blocks()
        
        logger.info("✅ GPU context initialized successfully in worker process")
        return True
        
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize GPU context in worker: {e}")
        logger.warning("💻 Worker will fall back to CPU processing")
        return False

def disable_gpu_for_multiprocessing():
    """
    Set environment variables to disable GPU for multiprocessing.
    This prevents CUDA initialization issues.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logger.info("🚫 GPU disabled for multiprocessing mode")

def worker_init_func():
    """
    Worker initialization function for multiprocessing.Pool.
    Call this with initializer parameter in Pool constructor.
    """
    # Try to initialize GPU context for this worker
    initialize_gpu_context_for_worker()