"""
Tests for WFS fetch result handling and retry logic.

This validates error handling, retry mechanisms, and cache validation
for BD Topo data fetching.
"""

import pytest
import time
from pathlib import Path
import tempfile
import geopandas as gpd
from shapely.geometry import Point

from ign_lidar.io.wfs_fetch_result import (
    FetchStatus,
    FetchResult,
    RetryConfig,
    fetch_with_retry,
    validate_cache_file,
)


class TestFetchResult:
    """Test FetchResult dataclass."""

    def test_success_result(self):
        """Test successful fetch result."""
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
        result = FetchResult(status=FetchStatus.SUCCESS, data=gdf)

        assert result.success
        assert result.has_data
        assert len(result.data) == 1
        assert result.error is None

    def test_cache_hit_result(self):
        """Test cache hit result."""
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})
        result = FetchResult(
            status=FetchStatus.CACHE_HIT, data=gdf, cache_hit=True
        )

        assert result.success
        assert result.has_data
        assert result.cache_hit

    def test_error_result(self):
        """Test error result."""
        result = FetchResult(
            status=FetchStatus.NETWORK_ERROR,
            error="Connection refused",
            retry_count=3,
        )

        assert not result.success
        assert not result.has_data
        assert result.error == "Connection refused"
        assert result.retry_count == 3

    def test_empty_result(self):
        """Test empty result (no features)."""
        gdf = gpd.GeoDataFrame({"geometry": []})
        result = FetchResult(status=FetchStatus.EMPTY_RESULT, data=gdf)

        assert result.success  # Status is "success-like"
        assert not result.has_data  # But no data available
        assert len(result.data) == 0

    def test_repr_success(self):
        """Test string representation for success."""
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0), Point(1, 1)]})
        result = FetchResult(
            status=FetchStatus.SUCCESS, data=gdf, elapsed_time=1.5
        )

        repr_str = repr(result)
        assert "success" in repr_str
        assert "features=2" in repr_str
        assert "1.50s" in repr_str

    def test_repr_error(self):
        """Test string representation for error."""
        result = FetchResult(
            status=FetchStatus.NETWORK_ERROR,
            error="Timeout",
            retry_count=2,
        )

        repr_str = repr(result)
        assert "network_error" in repr_str
        assert "Timeout" in repr_str
        assert "retries=2" in repr_str


class TestRetryConfig:
    """Test RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 10.0
        assert config.backoff_factor == 2.0
        assert config.retry_on_timeout
        assert config.retry_on_network_error

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=5.0,
            backoff_factor=1.5,
        )

        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 5.0
        assert config.backoff_factor == 1.5

    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0, max_delay=10.0, backoff_factor=2.0
        )

        assert config.get_delay(0) == 1.0  # 1.0 * 2^0
        assert config.get_delay(1) == 2.0  # 1.0 * 2^1
        assert config.get_delay(2) == 4.0  # 1.0 * 2^2
        assert config.get_delay(3) == 8.0  # 1.0 * 2^3
        assert config.get_delay(4) == 10.0  # Capped at max_delay


class TestFetchWithRetry:
    """Test fetch_with_retry function."""

    def test_successful_fetch_first_try(self):
        """Test successful fetch on first attempt."""
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})

        def fetch_func():
            return gdf

        result = fetch_with_retry(fetch_func)

        assert result.success
        assert result.has_data
        assert result.retry_count == 0
        assert result.elapsed_time > 0

    def test_successful_fetch_after_retry(self):
        """Test successful fetch after retry."""
        attempt_count = [0]
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})

        def fetch_func():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise ConnectionError("Network error")
            return gdf

        config = RetryConfig(max_retries=3, initial_delay=0.1)
        result = fetch_with_retry(fetch_func, retry_config=config)

        assert result.success
        assert result.retry_count == 1  # Succeeded on second attempt

    def test_all_retries_exhausted(self):
        """Test when all retries are exhausted."""

        def fetch_func():
            raise ConnectionError("Network error")

        config = RetryConfig(max_retries=2, initial_delay=0.1)
        result = fetch_with_retry(fetch_func, retry_config=config)

        assert not result.success
        assert result.status == FetchStatus.NETWORK_ERROR
        assert result.retry_count == 2
        assert "Network error" in result.error

    def test_timeout_retry(self):
        """Test retry on timeout."""
        attempt_count = [0]
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]})

        def fetch_func():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise TimeoutError("Request timeout")
            return gdf

        config = RetryConfig(
            max_retries=3, initial_delay=0.1, retry_on_timeout=True
        )
        result = fetch_with_retry(fetch_func, retry_config=config)

        assert result.success
        assert result.retry_count == 1

    def test_no_retry_on_timeout_when_disabled(self):
        """Test no retry on timeout when disabled."""

        def fetch_func():
            raise TimeoutError("Request timeout")

        config = RetryConfig(max_retries=3, retry_on_timeout=False)
        result = fetch_with_retry(fetch_func, retry_config=config)

        assert not result.success
        assert result.retry_count == 0  # No retries attempted

    def test_non_retryable_error(self):
        """Test that non-retryable errors don't retry."""

        def fetch_func():
            raise ValueError("Invalid parameter")

        config = RetryConfig(max_retries=3, initial_delay=0.1)
        result = fetch_with_retry(fetch_func, retry_config=config)

        assert not result.success
        assert result.retry_count == 0  # ValueError is not retried
        assert "Error: Invalid parameter" in result.error

    def test_none_result(self):
        """Test when fetch returns None."""

        def fetch_func():
            return None

        result = fetch_with_retry(fetch_func)

        assert not result.success
        assert result.status == FetchStatus.INVALID_RESPONSE
        assert "returned None" in result.error

    def test_empty_geodataframe(self):
        """Test when fetch returns empty GeoDataFrame."""

        def fetch_func():
            return gpd.GeoDataFrame({"geometry": []})

        result = fetch_with_retry(fetch_func)

        assert result.status == FetchStatus.EMPTY_RESULT
        assert len(result.data) == 0


class TestValidateCacheFile:
    """Test cache file validation."""

    def test_nonexistent_file(self):
        """Test validation of non-existent file."""
        path = Path("/tmp/nonexistent_file.geojson")
        assert not validate_cache_file(path)

    def test_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cache_path = Path(f.name)

        try:
            assert not validate_cache_file(cache_path)
        finally:
            cache_path.unlink()

    def test_valid_file(self):
        """Test validation of valid file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Some cache data here...")
            cache_path = Path(f.name)

        try:
            assert validate_cache_file(cache_path)
        finally:
            cache_path.unlink()

    def test_file_age_check(self):
        """Test file age validation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Some cache data")
            cache_path = Path(f.name)

        try:
            # Modify file time to be old
            old_time = time.time() - (10 * 24 * 3600)  # 10 days ago
            cache_path.touch()
            import os

            os.utime(cache_path, (old_time, old_time))

            # Should fail with max_age_days=5
            assert not validate_cache_file(cache_path, max_age_days=5)

            # Should pass with max_age_days=15
            assert validate_cache_file(cache_path, max_age_days=15)

        finally:
            cache_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
