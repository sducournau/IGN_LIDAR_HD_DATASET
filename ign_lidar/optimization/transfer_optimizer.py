"""
Transfer Optimizer - Minimize CPU↔GPU Data Transfers

This module provides tools to analyze and optimize data transfers between
CPU and GPU, reducing one of the main bottlenecks in GPU-accelerated pipelines.

Key Features:
- Transfer tracking and profiling
- Automatic caching recommendations
- Bandwidth utilization analysis
- Transfer elimination strategies

Author: IGN LiDAR HD Team
Date: November 23, 2025
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

# ✅ NEW (v3.5.2): Centralized GPU imports via GPUManager
from ign_lidar.core.gpu import GPUManager

gpu = GPUManager()
HAS_CUPY = gpu.gpu_available

if HAS_CUPY:
    cp = gpu.get_cupy()
else:
    cp = None


@dataclass
class TransferEvent:
    """Single data transfer event."""
    timestamp: float
    direction: str  # 'upload' or 'download'
    size_mb: float
    duration_ms: float
    bandwidth_gbps: float
    data_key: Optional[str] = None
    cached: bool = False


@dataclass
class TransferProfile:
    """Transfer profile for a processing session."""
    total_uploads_mb: float = 0.0
    total_downloads_mb: float = 0.0
    total_upload_time_ms: float = 0.0
    total_download_time_ms: float = 0.0
    num_uploads: int = 0
    num_downloads: int = 0
    upload_events: List[TransferEvent] = field(default_factory=list)
    download_events: List[TransferEvent] = field(default_factory=list)
    redundant_uploads: int = 0  # Uploads that could have been cached
    redundant_uploads_mb: float = 0.0
    
    @property
    def total_transfer_mb(self) -> float:
        """Total data transferred (upload + download)."""
        return self.total_uploads_mb + self.total_downloads_mb
    
    @property
    def total_transfer_time_ms(self) -> float:
        """Total transfer time."""
        return self.total_upload_time_ms + self.total_download_time_ms
    
    @property
    def avg_upload_bandwidth_gbps(self) -> float:
        """Average upload bandwidth in GB/s."""
        if self.total_upload_time_ms == 0:
            return 0.0
        return (self.total_uploads_mb / 1024) / (self.total_upload_time_ms / 1000)
    
    @property
    def avg_download_bandwidth_gbps(self) -> float:
        """Average download bandwidth in GB/s."""
        if self.total_download_time_ms == 0:
            return 0.0
        return (self.total_downloads_mb / 1024) / (self.total_download_time_ms / 1000)
    
    @property
    def cache_efficiency(self) -> float:
        """Percentage of redundant uploads that could be cached."""
        if self.num_uploads == 0:
            return 0.0
        return (self.redundant_uploads / self.num_uploads) * 100


class TransferOptimizer:
    """
    Analyzes and optimizes CPU↔GPU data transfers.
    
    This class tracks all data transfers, identifies patterns, and provides
    recommendations for reducing transfer overhead through caching and
    data reuse strategies.
    
    Example:
        >>> from ign_lidar.optimization import TransferOptimizer
        >>> 
        >>> optimizer = TransferOptimizer(enable_profiling=True)
        >>> 
        >>> # Track transfers during processing
        >>> with optimizer.profile_transfer('upload', 'points_data'):
        ...     points_gpu = cp.asarray(points_cpu)
        >>> 
        >>> # Get recommendations
        >>> report = optimizer.get_report()
        >>> optimizer.print_recommendations()
    """
    
    def __init__(self, enable_profiling: bool = True):
        """
        Initialize transfer optimizer.
        
        Args:
            enable_profiling: Enable detailed transfer profiling
        """
        self.enable_profiling = enable_profiling
        self.profile = TransferProfile()
        self.data_access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.enabled = enable_profiling  # Work with or without CuPy for testing
        
        if not HAS_CUPY:
            logger.debug("CuPy not available - transfer optimization running in CPU mode")
        
        if self.enabled:
            logger.info("Transfer optimizer initialized")
    
    def track_upload(
        self,
        size_mb: float,
        duration_ms: float,
        data_key: Optional[str] = None,
        cached: bool = False
    ) -> None:
        """
        Track a CPU→GPU upload.
        
        Args:
            size_mb: Size of data transferred in MB
            duration_ms: Transfer duration in milliseconds
            data_key: Optional key to identify the data
            cached: Whether this data was retrieved from cache
        """
        if not self.enabled:
            return
        
        bandwidth_gbps = (size_mb / 1024) / (duration_ms / 1000) if duration_ms > 0 else 0.0
        
        event = TransferEvent(
            timestamp=time.time(),
            direction='upload',
            size_mb=size_mb,
            duration_ms=duration_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_key=data_key,
            cached=cached
        )
        
        self.profile.upload_events.append(event)
        self.profile.total_uploads_mb += size_mb
        self.profile.total_upload_time_ms += duration_ms
        self.profile.num_uploads += 1
        
        # Track access patterns for caching recommendations
        if data_key:
            self.data_access_patterns[data_key].append(time.time())
            
            # Check if this is a redundant upload (accessed before)
            if len(self.data_access_patterns[data_key]) > 1 and not cached:
                self.profile.redundant_uploads += 1
                self.profile.redundant_uploads_mb += size_mb
    
    def track_download(
        self,
        size_mb: float,
        duration_ms: float,
        data_key: Optional[str] = None
    ) -> None:
        """
        Track a GPU→CPU download.
        
        Args:
            size_mb: Size of data transferred in MB
            duration_ms: Transfer duration in milliseconds
            data_key: Optional key to identify the data
        """
        if not self.enabled:
            return
        
        bandwidth_gbps = (size_mb / 1024) / (duration_ms / 1000) if duration_ms > 0 else 0.0
        
        event = TransferEvent(
            timestamp=time.time(),
            direction='download',
            size_mb=size_mb,
            duration_ms=duration_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_key=data_key
        )
        
        self.profile.download_events.append(event)
        self.profile.total_downloads_mb += size_mb
        self.profile.total_download_time_ms += duration_ms
        self.profile.num_downloads += 1
        
        if data_key:
            self.data_access_patterns[data_key].append(time.time())
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get comprehensive transfer analysis report.
        
        Returns:
            Dictionary with transfer statistics and recommendations
        """
        report = {
            'total_uploads': self.profile.num_uploads,
            'total_downloads': self.profile.num_downloads,
            'total_upload_mb': self.profile.total_uploads_mb,
            'total_download_mb': self.profile.total_downloads_mb,
            'total_transfer_mb': self.profile.total_transfer_mb,
            'upload_time_ms': self.profile.total_upload_time_ms,
            'download_time_ms': self.profile.total_download_time_ms,
            'total_time_ms': self.profile.total_transfer_time_ms,
            'avg_upload_bandwidth_gbps': self.profile.avg_upload_bandwidth_gbps,
            'avg_download_bandwidth_gbps': self.profile.avg_download_bandwidth_gbps,
            'redundant_uploads': self.profile.redundant_uploads,
            'redundant_uploads_mb': self.profile.redundant_uploads_mb,
            'cache_efficiency': self.profile.cache_efficiency,
            'potential_savings_mb': self.profile.redundant_uploads_mb,
            'potential_savings_pct': (self.profile.redundant_uploads_mb / max(self.profile.total_uploads_mb, 1)) * 100
        }
        
        # Identify hot data (frequently accessed)
        hot_data = []
        for key, accesses in self.data_access_patterns.items():
            if len(accesses) >= 3:  # Accessed 3+ times
                hot_data.append({
                    'key': key,
                    'accesses': len(accesses),
                    'should_cache': True
                })
        
        report['hot_data'] = sorted(hot_data, key=lambda x: x['accesses'], reverse=True)
        
        return report
    
    def print_report(self) -> None:
        """Print formatted transfer analysis report."""
        if not self.enabled:
            logger.info("Transfer optimizer not enabled")
            return
        
        report = self.get_report()
        
        logger.info("=" * 70)
        logger.info("GPU TRANSFER ANALYSIS REPORT")
        logger.info("=" * 70)
        logger.info(f"Total Transfers: {report['total_uploads'] + report['total_downloads']}")
        logger.info(f"  ↑ Uploads:   {report['total_uploads']:4d} ({report['total_upload_mb']:8.1f} MB)")
        logger.info(f"  ↓ Downloads: {report['total_downloads']:4d} ({report['total_download_mb']:8.1f} MB)")
        logger.info("")
        logger.info(f"Transfer Time:")
        logger.info(f"  Upload:   {report['upload_time_ms']:8.1f} ms")
        logger.info(f"  Download: {report['download_time_ms']:8.1f} ms")
        logger.info(f"  Total:    {report['total_time_ms']:8.1f} ms")
        logger.info("")
        logger.info(f"Bandwidth:")
        logger.info(f"  Upload:   {report['avg_upload_bandwidth_gbps']:6.2f} GB/s")
        logger.info(f"  Download: {report['avg_download_bandwidth_gbps']:6.2f} GB/s")
        logger.info("")
        logger.info(f"Caching Opportunities:")
        logger.info(f"  Redundant uploads: {report['redundant_uploads']} ({report['redundant_uploads_mb']:.1f} MB)")
        logger.info(f"  Cache efficiency:  {report['cache_efficiency']:.1f}%")
        logger.info(f"  Potential savings: {report['potential_savings_mb']:.1f} MB ({report['potential_savings_pct']:.1f}%)")
        
        if report['hot_data']:
            logger.info("")
            logger.info(f"Hot Data (accessed 3+ times):")
            for item in report['hot_data'][:5]:  # Top 5
                logger.info(f"  - {item['key']}: {item['accesses']} accesses → CACHE")
        
        logger.info("=" * 70)
    
    def print_recommendations(self) -> None:
        """Print optimization recommendations."""
        if not self.enabled:
            return
        
        report = self.get_report()
        
        logger.info("=" * 70)
        logger.info("OPTIMIZATION RECOMMENDATIONS")
        logger.info("=" * 70)
        
        recommendations = []
        
        # Check cache efficiency
        if report['cache_efficiency'] < 50 and report['redundant_uploads'] > 5:
            recommendations.append(
                f"⚠️  LOW CACHE EFFICIENCY ({report['cache_efficiency']:.1f}%)\n"
                f"   → Enable GPUArrayCache for frequently accessed data\n"
                f"   → Potential savings: {report['potential_savings_mb']:.1f} MB"
            )
        
        # Check bandwidth utilization
        if report['avg_upload_bandwidth_gbps'] < 5.0:
            recommendations.append(
                f"⚠️  LOW UPLOAD BANDWIDTH ({report['avg_upload_bandwidth_gbps']:.1f} GB/s)\n"
                f"   → Use pinned memory for faster transfers\n"
                f"   → Consider asynchronous transfers with CUDA streams"
            )
        
        # Check for excessive transfers
        transfer_ratio = report['total_transfer_mb'] / max(report['total_upload_mb'], 1)
        if transfer_ratio > 2.0:
            recommendations.append(
                f"⚠️  EXCESSIVE TRANSFERS (ratio: {transfer_ratio:.1f}x)\n"
                f"   → Minimize round-trips: keep data on GPU longer\n"
                f"   → Batch operations to reduce transfer frequency"
            )
        
        # Hot data recommendations
        if report['hot_data']:
            keys = ', '.join([item['key'] for item in report['hot_data'][:3]])
            recommendations.append(
                f"💡 CACHE THESE ARRAYS: {keys}\n"
                f"   → These are accessed {report['hot_data'][0]['accesses']}+ times\n"
                f"   → Use GPUArrayCache.get_or_upload()"
            )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
                logger.info("")
        else:
            logger.info("✅ No major issues detected - transfers are well optimized!")
        
        logger.info("=" * 70)
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.profile = TransferProfile()
        self.data_access_patterns.clear()


def create_transfer_optimizer(enable_profiling: bool = True) -> TransferOptimizer:
    """
    Factory function to create a transfer optimizer.
    
    Args:
        enable_profiling: Enable detailed transfer profiling
        
    Returns:
        Configured TransferOptimizer instance
    """
    return TransferOptimizer(enable_profiling=enable_profiling)
