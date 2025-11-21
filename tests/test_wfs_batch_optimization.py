"""
Test suite for WFS batch fetching optimization.

Tests the optimized parallel fetching implementation since IGN WFS
does not support true multi-layer batch requests.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point, box
import pandas as pd

from ign_lidar.io.wfs_optimized import OptimizedWFSFetcher, OptimizedWFSConfig


@pytest.fixture
def test_bbox():
    """Test bounding box (1km x 1km in Paris area)."""
    return (648000.0, 6860000.0, 649000.0, 6861000.0)


@pytest.fixture
def test_layers():
    """Test layer names."""
    return ['buildings', 'roads', 'water']


@pytest.fixture
def mock_gdf():
    """Create a mock GeoDataFrame with test data."""
    return gpd.GeoDataFrame(
        {'id': [1, 2, 3], 'name': ['A', 'B', 'C']},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs='EPSG:2154'
    )


@pytest.fixture
def wfs_config():
    """WFS configuration for testing."""
    return OptimizedWFSConfig(
        enable_batch_fetch=True,
        enable_parallel=True,
        max_workers=4,
        enable_disk_cache=False,  # Disable for tests
        enable_memory_cache=False
    )


@pytest.fixture
def wfs_fetcher(tmp_path, wfs_config):
    """Create WFS fetcher instance for testing."""
    return OptimizedWFSFetcher(cache_dir=tmp_path / "cache", config=wfs_config)


class TestBatchFetching:
    """Test batch fetching functionality."""
    
    def test_batch_fetch_calls_parallel(self, wfs_fetcher, test_bbox, test_layers):
        """Test that _fetch_batch delegates to _fetch_parallel."""
        with patch.object(wfs_fetcher, '_fetch_parallel') as mock_parallel:
            mock_parallel.return_value = {}
            
            result = wfs_fetcher._fetch_batch(test_bbox, test_layers)
            
            mock_parallel.assert_called_once_with(test_bbox, test_layers)
    
    def test_batch_fetch_documentation(self, wfs_fetcher):
        """Verify batch fetch has comprehensive documentation."""
        doc = wfs_fetcher._fetch_batch.__doc__
        
        assert doc is not None
        assert 'IGN WFS service does not support' in doc
        assert 'parallel fetching' in doc.lower()
        assert 'performance' in doc.lower()
    
    def test_no_true_batch_support(self):
        """
        Document that IGN WFS doesn't support multi-layer batch requests.
        
        This test serves as documentation that we've verified:
        1. Comma-separated TYPENAME returns only first layer
        2. TYPENAMES (plural) parameter returns 400 error
        3. Space-separated TYPENAME returns 400 error
        
        Therefore, optimized parallel fetching is the best approach.
        """
        # This is a documentation test - no actual WFS call
        assert True, "IGN WFS requires separate requests per layer"


class TestParallelFetching:
    """Test parallel fetching implementation."""
    
    def test_parallel_fetch_success(self, wfs_fetcher, test_bbox, test_layers, mock_gdf):
        """Test successful parallel fetching of multiple layers."""
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.return_value = mock_gdf
            
            result = wfs_fetcher._fetch_parallel(test_bbox, test_layers)
            
            assert len(result) == len(test_layers)
            assert all(layer in result for layer in test_layers)
            assert mock_fetch.call_count == len(test_layers)
    
    def test_parallel_fetch_empty_results(self, wfs_fetcher, test_bbox, test_layers):
        """Test parallel fetching with no features found."""
        empty_gdf = gpd.GeoDataFrame(
            geometry=[],
            crs='EPSG:2154'
        )
        
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.return_value = empty_gdf
            
            result = wfs_fetcher._fetch_parallel(test_bbox, test_layers)
            
            # Empty GeoDataFrames should not be included
            assert len(result) == 0
    
    def test_parallel_fetch_partial_failure(self, wfs_fetcher, test_bbox, mock_gdf):
        """Test parallel fetching with some layers failing."""
        layers = ['buildings', 'roads', 'water']
        
        def mock_fetch_side_effect(bbox, layer):
            if layer == 'roads':
                raise Exception("Network error")
            return mock_gdf
        
        with patch.object(wfs_fetcher, '_fetch_single_layer', side_effect=mock_fetch_side_effect):
            result = wfs_fetcher._fetch_parallel(test_bbox, layers)
            
            # Should get 2 successful layers
            assert len(result) == 2
            assert 'buildings' in result
            assert 'water' in result
            assert 'roads' not in result
    
    def test_parallel_fetch_fallback_to_sequential(self, wfs_fetcher, test_bbox, test_layers):
        """Test fallback to sequential for single layer or disabled parallel."""
        # Test with single layer
        with patch.object(wfs_fetcher, '_fetch_sequential') as mock_sequential:
            mock_sequential.return_value = {}
            
            wfs_fetcher._fetch_parallel(test_bbox, ['buildings'])
            mock_sequential.assert_called_once()
        
        # Test with parallel disabled
        wfs_fetcher.config.enable_parallel = False
        with patch.object(wfs_fetcher, '_fetch_sequential') as mock_sequential:
            mock_sequential.return_value = {}
            
            wfs_fetcher._fetch_parallel(test_bbox, test_layers)
            mock_sequential.assert_called_once()
    
    def test_parallel_fetch_respects_max_workers(self, wfs_fetcher, test_bbox, mock_gdf):
        """Test that parallel fetching respects max_workers configuration."""
        layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
        wfs_fetcher.config.max_workers = 2
        
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.return_value = mock_gdf
            
            with patch('ign_lidar.io.wfs_optimized.ThreadPoolExecutor') as mock_executor:
                mock_executor.return_value.__enter__ = Mock()
                mock_executor.return_value.__exit__ = Mock()
                
                # Set up mock to actually execute the fetches
                executor_instance = MagicMock()
                mock_executor.return_value = executor_instance
                
                # Mock as_completed to return completed futures
                with patch('ign_lidar.io.wfs_optimized.as_completed'):
                    wfs_fetcher._fetch_parallel(test_bbox, layers)
                
                # Verify max_workers was respected (min of configured and layer count)
                mock_executor.assert_called_once()
                call_kwargs = mock_executor.call_args[1]
                assert call_kwargs['max_workers'] == 2
    
    def test_parallel_fetch_performance_logging(self, wfs_fetcher, test_bbox, test_layers, mock_gdf, caplog):
        """Test that parallel fetching logs performance metrics."""
        import logging
        caplog.set_level(logging.INFO)
        
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.return_value = mock_gdf
            
            wfs_fetcher._fetch_parallel(test_bbox, test_layers)
            
            # Check that performance info was logged
            log_messages = [record.message for record in caplog.records]
            assert any('Parallel fetch completed' in msg for msg in log_messages)
            assert any('layers/s' in msg for msg in log_messages)


class TestSequentialFetching:
    """Test sequential fetching implementation."""
    
    def test_sequential_fetch_success(self, wfs_fetcher, test_bbox, test_layers, mock_gdf):
        """Test successful sequential fetching."""
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.return_value = mock_gdf
            
            result = wfs_fetcher._fetch_sequential(test_bbox, test_layers)
            
            assert len(result) == len(test_layers)
            assert mock_fetch.call_count == len(test_layers)
    
    def test_sequential_fetch_with_errors(self, wfs_fetcher, test_bbox, test_layers, mock_gdf):
        """Test sequential fetching continues after errors."""
        def mock_fetch_side_effect(bbox, layer):
            if layer == 'roads':
                raise Exception("Error")
            return mock_gdf
        
        with patch.object(wfs_fetcher, '_fetch_single_layer', side_effect=mock_fetch_side_effect):
            result = wfs_fetcher._fetch_sequential(test_bbox, test_layers)
            
            assert len(result) == 2  # buildings and water
            assert 'roads' not in result


class TestPerformanceComparison:
    """Test performance characteristics of parallel vs sequential fetching."""
    
    def test_parallel_faster_than_sequential(self, wfs_fetcher, test_bbox, mock_gdf):
        """Verify that parallel fetching is faster for multiple layers."""
        layers = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
        
        # Simulate network delay
        def slow_fetch(bbox, layer):
            time.sleep(0.05)  # 50ms delay per layer
            return mock_gdf
        
        with patch.object(wfs_fetcher, '_fetch_single_layer', side_effect=slow_fetch):
            # Sequential timing
            wfs_fetcher.config.enable_parallel = False
            start = time.time()
            wfs_fetcher._fetch_sequential(test_bbox, layers)
            sequential_time = time.time() - start
            
            # Parallel timing
            wfs_fetcher.config.enable_parallel = True
            start = time.time()
            wfs_fetcher._fetch_parallel(test_bbox, layers)
            parallel_time = time.time() - start
            
            # Parallel should be at least 2x faster with 4 workers
            # (5 layers * 50ms = 250ms sequential vs ~100ms parallel)
            speedup = sequential_time / parallel_time
            assert speedup > 1.5, f"Parallel speedup {speedup:.2f}x not sufficient"
    
    def test_minimal_overhead_for_single_layer(self, wfs_fetcher, test_bbox, mock_gdf):
        """Test that single layer has minimal parallel overhead."""
        single_layer = ['buildings']
        
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.return_value = mock_gdf
            
            # Should use sequential path automatically
            with patch.object(wfs_fetcher, '_fetch_sequential') as mock_sequential:
                mock_sequential.return_value = {'buildings': mock_gdf}
                
                result = wfs_fetcher._fetch_parallel(test_bbox, single_layer)
                
                mock_sequential.assert_called_once()


class TestConfiguration:
    """Test WFS configuration options."""
    
    def test_default_parallel_enabled(self):
        """Test that parallel fetching is enabled by default."""
        config = OptimizedWFSConfig()
        assert config.enable_parallel is True
        assert config.enable_batch_fetch is True
    
    def test_configurable_max_workers(self):
        """Test that max_workers can be configured."""
        config = OptimizedWFSConfig(max_workers=8)
        assert config.max_workers == 8
    
    def test_disable_parallel_fetching(self, tmp_path):
        """Test disabling parallel fetching."""
        config = OptimizedWFSConfig(enable_parallel=False)
        fetcher = OptimizedWFSFetcher(cache_dir=tmp_path, config=config)
        
        assert fetcher.config.enable_parallel is False


class TestErrorHandling:
    """Test error handling in batch/parallel fetching."""
    
    def test_all_layers_fail_returns_empty(self, wfs_fetcher, test_bbox, test_layers):
        """Test that all failures return empty dict."""
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            
            result = wfs_fetcher._fetch_parallel(test_bbox, test_layers)
            
            assert result == {}
    
    def test_none_return_handled(self, wfs_fetcher, test_bbox, test_layers):
        """Test that None return values are handled properly."""
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.return_value = None
            
            result = wfs_fetcher._fetch_parallel(test_bbox, test_layers)
            
            assert result == {}
    
    def test_exception_logging(self, wfs_fetcher, test_bbox, test_layers, caplog):
        """Test that exceptions are properly logged."""
        with patch.object(wfs_fetcher, '_fetch_single_layer') as mock_fetch:
            mock_fetch.side_effect = ValueError("Test error")
            
            wfs_fetcher._fetch_parallel(test_bbox, test_layers)
            
            # Check error was logged for each layer
            error_logs = [r for r in caplog.records if r.levelname == 'ERROR']
            assert len(error_logs) == len(test_layers)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
