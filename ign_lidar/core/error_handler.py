"""
Error Handling for IGN LiDAR HD Processing
Provides detailed, actionable error messages with recovery suggestions.
Version: 2.0.0
"""

import logging
from typing import Optional, Dict, Any
import sys

# GPU availability check (centralized via GPUManager)
from .gpu import GPUManager
_gpu_manager = GPUManager()
GPU_AVAILABLE = _gpu_manager.gpu_available

if GPU_AVAILABLE:
    import cupy as cp
else:
    cp = None

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for processing errors with detailed messages."""
    
    def __init__(
        self,
        message: str,
        suggestions: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.suggestions = suggestions or []
        self.context = context or {}
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with context and suggestions."""
        parts = ["\n" + "="*70]
        parts.append(f"ERROR: {self.message}")
        parts.append("="*70)
        
        if self.context:
            parts.append("\nContext:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")
        
        if self.suggestions:
            parts.append("\nSuggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                parts.append(f"  {i}. {suggestion}")
        
        parts.append("="*70 + "\n")
        return "\n".join(parts)


class GPUMemoryError(ProcessingError):
    """GPU out of memory error with detailed diagnostics and recovery guidance."""
    
    @staticmethod
    def from_cuda_error(
        error: Exception,
        current_vram_gb: Optional[float] = None,
        total_vram_gb: Optional[float] = None,
        chunk_size: Optional[int] = None,
        num_points: Optional[int] = None
    ) -> 'GPUMemoryError':
        """Create error from CUDA exception with comprehensive context and solutions."""
        context = {}
        
        if current_vram_gb is not None and total_vram_gb is not None:
            context['VRAM Usage'] = (
                f"{current_vram_gb:.1f}GB / {total_vram_gb:.1f}GB "
                f"({100*current_vram_gb/total_vram_gb:.0f}% full)"
            )
            context['VRAM Free'] = f"{total_vram_gb - current_vram_gb:.1f}GB"
            
            # Memory pressure indicator
            pressure = current_vram_gb / total_vram_gb
            if pressure > 0.95:
                context['Memory Pressure'] = "🔴 CRITICAL (>95%)"
            elif pressure > 0.85:
                context['Memory Pressure'] = "🟠 HIGH (>85%)"
            elif pressure > 0.70:
                context['Memory Pressure'] = "🟡 MODERATE (>70%)"
            else:
                context['Memory Pressure'] = "🟢 LOW (<70%)"
        
        if chunk_size is not None:
            context['Current Chunk Size'] = f"{chunk_size:,} points"
            
            # Suggest new chunk size (50% reduction)
            new_chunk_size = int(chunk_size * 0.5)
            context['Suggested Chunk Size'] = f"{new_chunk_size:,} points (-50%)"
        
        if num_points is not None:
            context['Total Points'] = f"{num_points:,}"
            if chunk_size:
                num_chunks = (num_points + chunk_size - 1) // chunk_size
                context['Number of Chunks'] = num_chunks
                
                # Estimate new chunk count with reduced size
                if chunk_size > 0:
                    new_chunk_size = int(chunk_size * 0.5)
                    new_num_chunks = (num_points + new_chunk_size - 1) // new_chunk_size
                    context['New Chunk Count'] = f"{new_num_chunks} (was {num_chunks})"
        
        suggestions = [
            "🔧 Quick Fixes (choose one):",
            "",
            "1️⃣  Reduce chunk size (RECOMMENDED):",
            "   • Config file: Set smaller 'chunk_size' value",
            "   • Command line: --chunk-size 500000",
            f"   • Try: {int(chunk_size * 0.5):,} points (50% of current)" if chunk_size else "",
            "",
            "2️⃣  Switch to CPU mode:",
            "   • Remove --use-gpu flag",
            "   • Or set use_gpu: false in config.yaml",
            "   • Slower but no memory limits",
            "",
            "3️⃣  Free GPU memory:",
            "   • Close other GPU applications",
            "   • Check: nvidia-smi (kill process if needed)",
            "   • Use: fuser -v /dev/nvidia*",
            "",
            "4️⃣  Process fewer files:",
            "   • Reduce num_workers in config",
            "   • Process one tile at a time",
            "   • Use sequential processing",
            "",
            "5️⃣  Hardware upgrade:",
            "   • Current GPU may be insufficient",
            "   • Recommended: ≥8GB VRAM",
            "   • Ideal: ≥16GB VRAM for large tiles",
            "",
            "📊 Memory Estimation Guide:",
            "   • 1M points ≈ 500MB VRAM",
            "   • 5M points ≈ 2.5GB VRAM",
            "   • 10M points ≈ 5GB VRAM",
            "",
            "🔍 Debug Mode:",
            "   • Enable: --verbose",
            "   • Shows memory usage per operation",
            "   • Helps identify memory peaks"
        ]
        
        # Filter empty suggestions
        suggestions = [s for s in suggestions if s.strip() or not s]
        
        return GPUMemoryError(
            message="🚫 GPU out of memory (VRAM limit exceeded)",
            suggestions=suggestions,
            context=context
        )


class GPUNotAvailableError(ProcessingError):
    """GPU requested but not available with detailed installation guidance."""
    
    @staticmethod
    def create(reason: str = "Unknown") -> 'GPUNotAvailableError':
        """Create error with comprehensive diagnostics and installation guidance."""
        context = {
            'Reason': reason,
            'CuPy Available': GPU_AVAILABLE,
            'Python Version': sys.version.split()[0]
        }
        
        # Determine CUDA version if nvidia-smi available
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                context['NVIDIA Driver'] = result.stdout.strip()
        except Exception:
            context['NVIDIA Driver'] = 'Not detected'
        
        suggestions = []
        
        if not GPU_AVAILABLE:
            # CuPy not installed - provide detailed installation guide
            suggestions.extend([
                "📦 Install CuPy for your CUDA version:",
                "   • CUDA 11.x: pip install cupy-cuda11x",
                "   • CUDA 12.x: pip install cupy-cuda12x",
                "   • Auto-detect: pip install cupy",
                "",
                "🔍 Check your CUDA version:",
                "   • Run: nvidia-smi",
                "   • Look for 'CUDA Version: X.Y'",
                "",
                "🐧 Install NVIDIA drivers (if missing):",
                "   • Ubuntu/Debian: sudo apt install nvidia-driver-535",
                "   • Windows: Download from nvidia.com/drivers",
                "   • Verify: nvidia-smi should show GPU info",
                "",
                "🔄 Alternative: Use CPU mode",
                "   • Remove --use-gpu flag from command",
                "   • Or set use_gpu: false in config.yaml",
                "",
                "📚 Full guide: docs/guides/gpu-setup.md"
            ])
        else:
            # CuPy installed but GPU not detected
            suggestions.extend([
                "🔍 Diagnose GPU availability:",
                "   • Run: nvidia-smi",
                "   • Should show GPU name and memory",
                "",
                "⚡ Possible issues:",
                "   • GPU being used by another process",
                "   • Driver version mismatch with CUDA",
                "   • CUDA_VISIBLE_DEVICES set incorrectly",
                "",
                "🔧 Quick fixes:",
                "   • Restart Python interpreter",
                "   • Check: echo $CUDA_VISIBLE_DEVICES",
                "   • Kill other GPU processes",
                "",
                "🔄 Fallback to CPU:",
                "   • Remove --use-gpu flag",
                "   • Or set use_gpu: false in config",
                "",
                "🆘 Still broken? Report issue:",
                "   • GitHub: github.com/sducournau/IGN_LIDAR_HD_DATASET/issues",
                "   • Include output of nvidia-smi"
            ])
        
        return GPUNotAvailableError(
            message="🚫 GPU requested but not available",
            suggestions=suggestions,
            context=context
        )


class MemoryPressureError(ProcessingError):
    """System under memory pressure."""
    
    @staticmethod
    def create(
        available_ram_gb: float,
        swap_used_percent: float,
        required_ram_gb: Optional[float] = None
    ) -> 'MemoryPressureError':
        """Create error with memory diagnostics."""
        context = {
            'Available RAM': f"{available_ram_gb:.1f}GB",
            'Swap Usage': f"{swap_used_percent:.0f}%"
        }
        
        if required_ram_gb is not None:
            context['Required RAM'] = f"{required_ram_gb:.1f}GB"
            context['RAM Deficit'] = (
                f"{max(0, required_ram_gb - available_ram_gb):.1f}GB"
            )
        
        suggestions = [
            "Close other applications to free RAM",
            "Reduce number of workers: --workers 1",
            "Process files one at a time",
            "Use smaller chunk sizes",
            "Add more RAM to your system",
            "Process on a machine with more RAM"
        ]
        
        return MemoryPressureError(
            message="System under memory pressure",
            suggestions=suggestions,
            context=context
        )


class FileProcessingError(ProcessingError):
    """Error processing specific file."""
    
    @staticmethod
    def create(
        file_path: str,
        error: Exception,
        stage: str = "Unknown"
    ) -> 'FileProcessingError':
        """Create error for file processing failure."""
        context = {
            'File': file_path,
            'Stage': stage,
            'Error Type': type(error).__name__,
            'Error Message': str(error)
        }
        
        suggestions = [
            "Verify file exists and is readable",
            "Check file format is valid LAZ/LAS",
            "Try processing with --force flag",
            "Check disk space available",
            "Verify file is not corrupted"
        ]
        
        if "permission" in str(error).lower():
            suggestions.insert(0, "Check file permissions")
        
        if "corrupt" in str(error).lower() or "invalid" in str(error).lower():
            suggestions.insert(0, "File may be corrupted - try redownloading")
        
        return FileProcessingError(
            message=f"Failed to process file: {file_path}",
            suggestions=suggestions,
            context=context
        )


class ConfigurationError(ProcessingError):
    """Invalid configuration error."""
    
    @staticmethod
    def create(
        parameter: str,
        value: Any,
        reason: str,
        valid_range: Optional[str] = None
    ) -> 'ConfigurationError':
        """Create error for invalid configuration."""
        context = {
            'Parameter': parameter,
            'Value': str(value),
            'Reason': reason
        }
        
        if valid_range:
            context['Valid Range'] = valid_range
        
        suggestions = [
            f"Correct the {parameter} parameter",
            "Check configuration file syntax",
            "Use default configuration: Remove custom config",
            "See documentation for valid parameter ranges"
        ]
        
        return ConfigurationError(
            message=f"Invalid configuration: {parameter}",
            suggestions=suggestions,
            context=context
        )


class FeatureComputationError(ProcessingError):
    """Error during feature computation."""
    
    @staticmethod
    def create(
        feature_name: str,
        error: Exception,
        num_points: Optional[int] = None,
        stage: str = "computation"
    ) -> 'FeatureComputationError':
        """Create error for feature computation failure."""
        context = {
            'Feature': feature_name,
            'Stage': stage,
            'Error Type': type(error).__name__,
            'Error Message': str(error)
        }
        
        if num_points is not None:
            context['Number of Points'] = f"{num_points:,}"
        
        suggestions = [
            "Check point cloud has sufficient density",
            "Verify k_neighbors parameter is appropriate",
            "Try reducing feature complexity",
            "Check for NaN or Inf values in input data",
            "Enable verbose logging for more details"
        ]
        
        # Add specific suggestions based on error type
        error_str = str(error).lower()
        if "memory" in error_str:
            suggestions.insert(0, "Reduce chunk size or use GPU chunked mode")
        elif "index" in error_str or "dimension" in error_str:
            suggestions.insert(0, "Verify point cloud dimensions and format")
        
        return FeatureComputationError(
            message=f"Failed to compute feature: {feature_name}",
            suggestions=suggestions,
            context=context
        )


class CacheError(ProcessingError):
    """Error with caching system."""
    
    @staticmethod
    def create(
        cache_type: str,
        operation: str,
        error: Exception,
        cache_path: Optional[str] = None
    ) -> 'CacheError':
        """Create error for cache operation failure."""
        context = {
            'Cache Type': cache_type,
            'Operation': operation,
            'Error Type': type(error).__name__,
            'Error Message': str(error)
        }
        
        if cache_path:
            context['Cache Path'] = cache_path
        
        suggestions = [
            "Clear cache and retry: rm -rf cache_directory",
            "Check disk space availability",
            "Verify write permissions for cache directory",
            "Disable caching temporarily to proceed",
            "Check for corrupted cache files"
        ]
        
        # Permission-specific suggestions
        if "permission" in str(error).lower():
            suggestions.insert(0, "Grant write permissions: chmod 755 cache_dir")
        
        return CacheError(
            message=f"Cache {operation} failed for {cache_type}",
            suggestions=suggestions,
            context=context
        )


class DataFetchError(ProcessingError):
    """Error fetching external data (RGB, NIR, ground truth)."""
    
    @staticmethod
    def create(
        data_type: str,
        error: Exception,
        url: Optional[str] = None,
        retry_count: int = 0
    ) -> 'DataFetchError':
        """Create error for data fetching failure."""
        context = {
            'Data Type': data_type,
            'Error Type': type(error).__name__,
            'Error Message': str(error),
            'Retry Count': retry_count
        }
        
        if url:
            context['URL'] = url
        
        suggestions = [
            "Check internet connectivity",
            "Verify service is available (IGN WFS, orthophoto)",
            "Try again later if service is temporarily down",
            "Check firewall/proxy settings",
            "Use cached data if available"
        ]
        
        # Network-specific suggestions
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        if "timeout" in error_str or "timeout" in error_type:
            suggestions.insert(0, "Increase timeout value in configuration")
        elif "404" in error_str or "not found" in error_str:
            suggestions.insert(0, "Verify data is available for this region")
        elif "connection" in error_str:
            suggestions.insert(0, "Check network connection and DNS resolution")
        
        return DataFetchError(
            message=f"Failed to fetch {data_type} data",
            suggestions=suggestions,
            context=context
        )


class InitializationError(ProcessingError):
    """Error during component initialization."""
    
    @staticmethod
    def create(
        component: str,
        error: Exception,
        dependencies: Optional[list] = None
    ) -> 'InitializationError':
        """Create error for initialization failure."""
        context = {
            'Component': component,
            'Error Type': type(error).__name__,
            'Error Message': str(error)
        }
        
        if dependencies:
            context['Required Dependencies'] = ', '.join(dependencies)
        
        suggestions = [
            "Check all required dependencies are installed",
            "Verify configuration is valid",
            "Check system resources are available",
            "Review error logs for details",
            "Try using default configuration"
        ]
        
        # Import-specific suggestions
        if isinstance(error, ImportError):
            suggestions.insert(
                0, 
                "Install missing dependencies: pip install -r requirements.txt"
            )
            if dependencies:
                dep_str = ' '.join(dependencies)
                suggestions.insert(1, f"Install specific packages: pip install {dep_str}")
        
        return InitializationError(
            message=f"Failed to initialize {component}",
            suggestions=suggestions,
            context=context
        )


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory information."""
    if not GPU_AVAILABLE or cp is None:
        return {}
    
    try:
        mempool = cp.get_default_memory_pool()
        device = cp.cuda.Device()
        free_mem, total_mem = device.mem_info
        
        return {
            'used_gb': mempool.used_bytes() / (1024**3),
            'total_gb': mempool.total_bytes() / (1024**3),
            'free_gb': free_mem / (1024**3),
            'device_total_gb': total_mem / (1024**3)
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return {}


def format_memory_error(
    error: Exception,
    **kwargs
) -> GPUMemoryError:
    """Format CUDA memory error with context."""
    mem_info = get_gpu_memory_info()
    
    return GPUMemoryError.from_cuda_error(
        error=error,
        current_vram_gb=mem_info.get('used_gb'),
        total_vram_gb=mem_info.get('device_total_gb'),
        **kwargs
    )


def handle_gpu_error(
    error: Exception,
    operation: str = "GPU operation",
    **context
) -> ProcessingError:
    """
    Handle GPU errors and convert to detailed error messages.
    
    Args:
        error: Original exception
        operation: Description of operation that failed
        **context: Additional context (chunk_size, num_points, etc.)
    
    Returns:
        ProcessingError with detailed message
    """
    error_str = str(error).lower()
    
    # CUDA out of memory
    if "out of memory" in error_str or "cudaerror" in error_str:
        return format_memory_error(error, **context)
    
    # GPU not available
    if "cuda" in error_str and "not available" in error_str:
        return GPUNotAvailableError.create(reason=str(error))
    
    # Generic GPU error
    return ProcessingError(
        message=f"{operation} failed: {error}",
        suggestions=[
            "Try using CPU mode: Remove --use-gpu flag",
            "Check GPU availability: nvidia-smi",
            "Verify CUDA installation",
            "Check error logs for details"
        ],
        context=context
    )


# Convenience functions for common errors

def raise_gpu_memory_error(**kwargs):
    """Raise GPU memory error with context."""
    mem_info = get_gpu_memory_info()
    raise GPUMemoryError.from_cuda_error(
        error=RuntimeError("GPU out of memory"),
        current_vram_gb=mem_info.get('used_gb'),
        total_vram_gb=mem_info.get('device_total_gb'),
        **kwargs
    )


def raise_gpu_not_available_error(reason: str = "Unknown"):
    """Raise GPU not available error."""
    raise GPUNotAvailableError.create(reason=reason)


def raise_memory_pressure_error(**kwargs):
    """Raise memory pressure error."""
    raise MemoryPressureError.create(**kwargs)


def warn_gpu_fallback(reason: str):
    """Log warning about GPU fallback to CPU."""
    logger.warning(
        f"\n{'='*70}\n"
        f"⚠️  GPU processing unavailable: {reason}\n"
        f"   Falling back to CPU mode (will be slower)\n"
        f"{'='*70}"
    )
