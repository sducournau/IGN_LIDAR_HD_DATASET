"""
Enhanced WFS Fetch Result Handling

This module provides structured error handling and retry logic for BD Topo
WFS fetching operations, improving robustness of ground truth data retrieval.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
from pathlib import Path
import time
import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


class FetchStatus(Enum):
    """Status of a WFS fetch operation."""

    SUCCESS = "success"
    CACHE_HIT = "cache_hit"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESULT = "empty_result"
    CACHE_ERROR = "cache_error"


@dataclass
class FetchResult:
    """
    Structured result of a WFS fetch operation.

    This provides explicit success/failure information and allows
    callers to handle different failure modes appropriately.

    Attributes:
        status: Status of the fetch operation
        data: GeoDataFrame if successful, None otherwise
        error: Error message if failed, None otherwise
        cache_hit: Whether data came from cache
        retry_count: Number of retries attempted
        elapsed_time: Time taken for operation (seconds)
    """

    status: FetchStatus
    data: Optional[gpd.GeoDataFrame] = None
    error: Optional[str] = None
    cache_hit: bool = False
    retry_count: int = 0
    elapsed_time: float = 0.0

    @property
    def success(self) -> bool:
        """Check if fetch was successful."""
        return self.status in (FetchStatus.SUCCESS, FetchStatus.CACHE_HIT)

    @property
    def has_data(self) -> bool:
        """Check if data is available."""
        return self.data is not None and len(self.data) > 0

    def __repr__(self) -> str:
        """String representation."""
        if self.success:
            n_features = len(self.data) if self.data is not None else 0
            cache_str = " (cached)" if self.cache_hit else ""
            return (
                f"FetchResult(status={self.status.value}, "
                f"features={n_features}{cache_str}, "
                f"time={self.elapsed_time:.2f}s)"
            )
        else:
            return (
                f"FetchResult(status={self.status.value}, "
                f"error='{self.error}', retries={self.retry_count})"
            )


class RetryConfig:
    """Configuration for retry logic."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        backoff_factor: float = 2.0,
        retry_on_timeout: bool = True,
        retry_on_network_error: bool = True,
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Exponential backoff multiplier
            retry_on_timeout: Retry on timeout errors
            retry_on_network_error: Retry on network errors
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retry_on_timeout = retry_on_timeout
        self.retry_on_network_error = retry_on_network_error

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.initial_delay * (self.backoff_factor**attempt)
        return min(delay, self.max_delay)


def fetch_with_retry(
    fetch_func: Callable[[], gpd.GeoDataFrame],
    retry_config: Optional[RetryConfig] = None,
    operation_name: str = "fetch",
) -> FetchResult:
    """
    Execute a fetch operation with automatic retry logic.

    Args:
        fetch_func: Function that performs the fetch operation
        retry_config: Retry configuration (default: 3 retries)
        operation_name: Name of operation for logging

    Returns:
        FetchResult with status and data

    Example:
        >>> def fetch_buildings():
        ...     return fetcher._fetch_wfs_layer("BATIMENT", bbox)
        >>> result = fetch_with_retry(fetch_buildings)
        >>> if result.success:
        ...     process(result.data)
        >>> else:
        ...     logger.error(f"Failed: {result.error}")
    """
    if retry_config is None:
        retry_config = RetryConfig()

    start_time = time.time()
    last_error = None
    retry_count = 0

    for attempt in range(retry_config.max_retries + 1):
        try:
            logger.debug(
                f"Attempting {operation_name} "
                f"(attempt {attempt + 1}/{retry_config.max_retries + 1})"
            )

            data = fetch_func()

            elapsed = time.time() - start_time

            # Check if data is valid
            if data is None:
                return FetchResult(
                    status=FetchStatus.INVALID_RESPONSE,
                    error="Fetch returned None",
                    retry_count=retry_count,
                    elapsed_time=elapsed,
                )

            if len(data) == 0:
                return FetchResult(
                    status=FetchStatus.EMPTY_RESULT,
                    data=data,
                    retry_count=retry_count,
                    elapsed_time=elapsed,
                )

            # Success
            return FetchResult(
                status=FetchStatus.SUCCESS,
                data=data,
                retry_count=retry_count,
                elapsed_time=elapsed,
            )

        except TimeoutError as e:
            last_error = f"Timeout: {str(e)}"
            should_retry = (
                retry_config.retry_on_timeout and attempt < retry_config.max_retries
            )
            if not should_retry:
                break
            retry_count += 1
            delay = retry_config.get_delay(attempt)
            logger.warning(
                f"{operation_name} timeout, retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{retry_config.max_retries})"
            )
            time.sleep(delay)

        except (ConnectionError, IOError) as e:
            last_error = f"Network error: {str(e)}"
            if (
                not retry_config.retry_on_network_error
                or attempt >= retry_config.max_retries
            ):
                break
            retry_count += 1
            delay = retry_config.get_delay(attempt)
            logger.warning(
                f"{operation_name} network error, retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{retry_config.max_retries})"
            )
            time.sleep(delay)

        except Exception as e:
            # Other errors don't retry
            last_error = f"Error: {str(e)}"
            break

    # All retries exhausted or non-retryable error
    elapsed = time.time() - start_time
    return FetchResult(
        status=FetchStatus.NETWORK_ERROR,
        error=last_error,
        retry_count=retry_count,
        elapsed_time=elapsed,
    )


def validate_cache_file(cache_path: Path, max_age_days: Optional[int] = None) -> bool:
    """
    Validate that a cache file exists and is not corrupted.

    Args:
        cache_path: Path to cache file
        max_age_days: Maximum age of cache file (days), None = no limit

    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False

    # Check file size
    if cache_path.stat().st_size == 0:
        logger.warning(f"Cache file is empty: {cache_path}")
        return False

    # Check age if specified
    if max_age_days is not None:
        age_seconds = time.time() - cache_path.stat().st_mtime
        age_days = age_seconds / (24 * 3600)
        if age_days > max_age_days:
            logger.info(
                f"Cache file too old ({age_days:.1f} days "
                f"> {max_age_days} days): {cache_path}"
            )
            return False

    # Try to read file (basic validation)
    try:
        # Just check if we can open it (don't fully parse to save time)
        with open(cache_path, "rb") as f:
            header = f.read(100)
            if len(header) == 0:
                logger.warning(f"Cache file header is empty: {cache_path}")
                return False
    except Exception as e:
        logger.warning(f"Cache file appears corrupted: {cache_path} - {e}")
        return False

    return True


__all__ = [
    "FetchStatus",
    "FetchResult",
    "RetryConfig",
    "fetch_with_retry",
    "validate_cache_file",
]
