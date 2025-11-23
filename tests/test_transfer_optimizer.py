"""
Tests for TransferOptimizer (November 23, 2025)

Validates GPU transfer tracking and optimization recommendations.
"""

import pytest
import numpy as np
from ign_lidar.optimization import TransferOptimizer, create_transfer_optimizer


def test_transfer_optimizer_creation():
    """Test that TransferOptimizer can be created."""
    optimizer = TransferOptimizer(enable_profiling=True)
    assert optimizer is not None
    assert hasattr(optimizer, 'profile')
    assert hasattr(optimizer, 'track_upload')
    assert hasattr(optimizer, 'track_download')


def test_factory_function():
    """Test factory function."""
    optimizer = create_transfer_optimizer(enable_profiling=True)
    assert isinstance(optimizer, TransferOptimizer)


def test_track_upload():
    """Test upload tracking."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Track a mock upload
    optimizer.track_upload(
        size_mb=10.0,
        duration_ms=5.0,
        data_key='test_data',
        cached=False
    )
    
    assert optimizer.profile.num_uploads == 1
    assert optimizer.profile.total_uploads_mb == 10.0
    assert optimizer.profile.total_upload_time_ms == 5.0
    assert len(optimizer.profile.upload_events) == 1


def test_track_download():
    """Test download tracking."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Track a mock download
    optimizer.track_download(
        size_mb=8.0,
        duration_ms=4.0,
        data_key='result_data'
    )
    
    assert optimizer.profile.num_downloads == 1
    assert optimizer.profile.total_downloads_mb == 8.0
    assert optimizer.profile.total_download_time_ms == 4.0
    assert len(optimizer.profile.download_events) == 1


def test_redundant_upload_detection():
    """Test detection of redundant uploads."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # First upload - not redundant
    optimizer.track_upload(
        size_mb=10.0,
        duration_ms=5.0,
        data_key='points',
        cached=False
    )
    
    assert optimizer.profile.redundant_uploads == 0
    
    # Second upload of same data - should be flagged as redundant
    optimizer.track_upload(
        size_mb=10.0,
        duration_ms=5.0,
        data_key='points',
        cached=False
    )
    
    assert optimizer.profile.redundant_uploads == 1
    assert optimizer.profile.redundant_uploads_mb == 10.0


def test_cache_efficiency_calculation():
    """Test cache efficiency metric."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Simulate 10 uploads, 5 redundant
    for i in range(10):
        optimizer.track_upload(
            size_mb=5.0,
            duration_ms=2.0,
            data_key=f'data_{i % 5}',  # Only 5 unique keys
            cached=False
        )
    
    # Should have 5 redundant uploads (second access to each of the 5 keys)
    assert optimizer.profile.redundant_uploads == 5
    assert optimizer.profile.cache_efficiency == 50.0  # 5/10 = 50%


def test_bandwidth_calculation():
    """Test bandwidth calculations."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Upload 1024 MB in 1000 ms = 1 GB/s
    optimizer.track_upload(size_mb=1024.0, duration_ms=1000.0)
    
    # Download 512 MB in 500 ms = 1 GB/s
    optimizer.track_download(size_mb=512.0, duration_ms=500.0)
    
    assert optimizer.profile.avg_upload_bandwidth_gbps == pytest.approx(1.0, rel=0.01)
    assert optimizer.profile.avg_download_bandwidth_gbps == pytest.approx(1.0, rel=0.01)


def test_hot_data_identification():
    """Test identification of frequently accessed data."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Access some data multiple times
    for _ in range(5):
        optimizer.track_upload(size_mb=10.0, duration_ms=5.0, data_key='hot_data')
    
    optimizer.track_upload(size_mb=10.0, duration_ms=5.0, data_key='cold_data')
    
    report = optimizer.get_report()
    
    # 'hot_data' accessed 5 times should be identified as hot
    assert len(report['hot_data']) == 1
    assert report['hot_data'][0]['key'] == 'hot_data'
    assert report['hot_data'][0]['accesses'] == 5
    assert report['hot_data'][0]['should_cache'] is True


def test_report_generation():
    """Test comprehensive report generation."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Simulate some transfers
    optimizer.track_upload(size_mb=100.0, duration_ms=50.0, data_key='data1')
    optimizer.track_upload(size_mb=100.0, duration_ms=50.0, data_key='data1', cached=False)  # Redundant
    optimizer.track_download(size_mb=50.0, duration_ms=25.0)
    
    report = optimizer.get_report()
    
    assert report['total_uploads'] == 2
    assert report['total_downloads'] == 1
    assert report['total_upload_mb'] == 200.0
    assert report['total_download_mb'] == 50.0
    assert report['total_transfer_mb'] == 250.0
    assert report['redundant_uploads'] == 1
    assert report['redundant_uploads_mb'] == 100.0
    assert report['potential_savings_pct'] == 50.0  # 100/200 = 50%


def test_reset():
    """Test resetting optimizer state."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Add some data
    optimizer.track_upload(size_mb=10.0, duration_ms=5.0)
    optimizer.track_download(size_mb=5.0, duration_ms=2.0)
    
    assert optimizer.profile.num_uploads == 1
    assert optimizer.profile.num_downloads == 1
    
    # Reset
    optimizer.reset()
    
    assert optimizer.profile.num_uploads == 0
    assert optimizer.profile.num_downloads == 0
    assert len(optimizer.profile.upload_events) == 0
    assert len(optimizer.profile.download_events) == 0
    assert len(optimizer.data_access_patterns) == 0


def test_print_functions():
    """Test that print functions don't crash."""
    optimizer = TransferOptimizer(enable_profiling=True)
    
    # Add some data
    optimizer.track_upload(size_mb=100.0, duration_ms=50.0, data_key='data1')
    optimizer.track_upload(size_mb=100.0, duration_ms=50.0, data_key='data1')  # Redundant
    optimizer.track_download(size_mb=50.0, duration_ms=25.0)
    
    # Should not raise
    optimizer.print_report()
    optimizer.print_recommendations()


def test_disabled_profiling():
    """Test that optimizer works when profiling is disabled."""
    optimizer = TransferOptimizer(enable_profiling=False)
    
    # These should not raise, just do nothing
    optimizer.track_upload(size_mb=10.0, duration_ms=5.0)
    optimizer.track_download(size_mb=5.0, duration_ms=2.0)
    
    # Profile should remain empty
    assert optimizer.profile.num_uploads == 0
    assert optimizer.profile.num_downloads == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
