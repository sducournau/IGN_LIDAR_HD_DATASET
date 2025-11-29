"""Tests for WFS Rate Limiter and Circuit Breaker

Tests the rate limiting, circuit breaker, and concurrency limiting
functionality for WFS requests.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from ign_lidar.io.wfs_rate_limiter import (
    TokenBucketRateLimiter,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    ConcurrencyLimiter,
    WFSRateLimiter,
)


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiting."""
    
    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=10)
        
        # Should allow burst
        for _ in range(10):
            assert limiter.try_acquire()
        
        # Should block
        assert not limiter.try_acquire()
    
    def test_refill_over_time(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketRateLimiter(rate=5.0, capacity=5)
        
        # Consume all tokens
        for _ in range(5):
            assert limiter.try_acquire()
        
        # Should be empty
        assert not limiter.try_acquire()
        
        # Wait for refill (0.2s = 1 token at 5/s)
        time.sleep(0.25)
        
        # Should have 1 token now
        assert limiter.try_acquire()
    
    def test_blocking_acquire(self):
        """Test blocking acquire."""
        limiter = TokenBucketRateLimiter(rate=5.0, capacity=1)
        
        # Use first token
        assert limiter.acquire(timeout=0.1)
        
        # Second acquire should wait
        start = time.time()
        result = limiter.acquire(timeout=0.5)
        elapsed = time.time() - start
        
        assert result  # Should succeed after waiting
        assert elapsed >= 0.15  # Should have waited ~0.2s


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_closed_state_allows_requests(self):
        """Test that closed circuit allows requests."""
        breaker = CircuitBreaker()
        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.allow_request()
    
    def test_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)
        
        # Record failures
        for i in range(3):
            assert breaker.get_state() == CircuitState.CLOSED
            breaker.record_failure()
        
        # Should be open now
        assert breaker.get_state() == CircuitState.OPEN
        assert not breaker.allow_request()
    
    def test_half_open_after_timeout(self):
        """Test circuit goes to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.2,
        )
        breaker = CircuitBreaker(config)
        
        # Trip circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.get_state() == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.25)
        
        # Should allow request now (half-open)
        assert breaker.allow_request()
        assert breaker.get_state() == CircuitState.HALF_OPEN
    
    def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        breaker = CircuitBreaker(config)
        
        # Trip circuit
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.15)
        
        # Move to half-open
        assert breaker.allow_request()
        assert breaker.get_state() == CircuitState.HALF_OPEN
        
        # Record successes
        breaker.record_success()
        assert breaker.get_state() == CircuitState.HALF_OPEN
        
        breaker.record_success()
        assert breaker.get_state() == CircuitState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
        )
        breaker = CircuitBreaker(config)
        
        # Trip circuit
        breaker.record_failure()
        breaker.record_failure()
        time.sleep(0.15)
        
        # Move to half-open
        assert breaker.allow_request()
        assert breaker.get_state() == CircuitState.HALF_OPEN
        
        # Failure should reopen
        breaker.record_failure()
        assert breaker.get_state() == CircuitState.OPEN
    
    def test_manual_reset(self):
        """Test manual circuit reset."""
        breaker = CircuitBreaker()
        
        # Trip circuit
        for _ in range(5):
            breaker.record_failure()
        
        assert breaker.is_open()
        
        # Manual reset
        breaker.reset()
        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.allow_request()


class TestConcurrencyLimiter:
    """Test concurrency limiting."""
    
    def test_limits_concurrent_requests(self):
        """Test that concurrency is limited."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        
        # First two should succeed
        limiter.__enter__()
        limiter.__enter__()
        assert limiter.get_active_count() == 2
        
        # Third should block
        # (we can't test blocking easily without threads)
        
        # Release one
        limiter.__exit__(None, None, None)
        assert limiter.get_active_count() == 1
    
    def test_context_manager(self):
        """Test context manager usage."""
        limiter = ConcurrencyLimiter(max_concurrent=3)
        
        with limiter:
            assert limiter.get_active_count() == 1
            with limiter:
                assert limiter.get_active_count() == 2
            assert limiter.get_active_count() == 1
        
        assert limiter.get_active_count() == 0


class TestWFSRateLimiter:
    """Test comprehensive WFS rate limiter."""
    
    def test_successful_request(self):
        """Test successful request execution."""
        limiter = WFSRateLimiter(
            requests_per_second=10.0,
            burst_capacity=10,
            max_concurrent=3,
            enable_circuit_breaker=True,
        )
        
        result = limiter.execute(lambda: "success")
        assert result == "success"
        
        stats = limiter.get_stats()
        assert stats['requests_made'] == 1
        assert stats['requests_succeeded'] == 1
        assert stats['requests_failed'] == 0
    
    def test_failed_request_opens_circuit(self):
        """Test that failures open circuit breaker."""
        limiter = WFSRateLimiter(
            requests_per_second=10.0,
            enable_circuit_breaker=True,
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
        )
        
        # Cause failures
        for _ in range(3):
            try:
                limiter.execute(lambda: (_ for _ in ()).throw(Exception("Test error")))
            except Exception:
                pass
        
        # Circuit should be open
        stats = limiter.get_stats()
        assert stats['circuit_breaker_state'] == 'open'
        assert stats['circuit_breaker_trips'] == 1
        
        # Next request should be blocked
        result = limiter.execute(lambda: "blocked")
        assert result is None
        
        stats = limiter.get_stats()
        assert stats['requests_blocked'] > 0
    
    def test_rate_limiting_blocks_fast_requests(self):
        """Test that rate limiting works."""
        limiter = WFSRateLimiter(
            requests_per_second=2.0,
            burst_capacity=2,
            max_concurrent=10,
            enable_circuit_breaker=False,
        )
        
        # First 2 should succeed immediately
        for _ in range(2):
            result = limiter.execute(lambda: "ok", timeout=0.1)
            assert result == "ok"
        
        # Next should timeout (rate limited)
        result = limiter.execute(lambda: "blocked", timeout=0.1)
        assert result is None
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        limiter = WFSRateLimiter(
            requests_per_second=10.0,
            enable_circuit_breaker=True,
        )
        
        # Successful request
        limiter.execute(lambda: "ok")
        
        # Failed request
        try:
            limiter.execute(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass
        
        stats = limiter.get_stats()
        assert stats['requests_made'] == 2
        assert stats['requests_succeeded'] == 1
        assert stats['requests_failed'] == 1
    
    def test_reset(self):
        """Test reset functionality."""
        limiter = WFSRateLimiter(enable_circuit_breaker=True)
        
        # Trip circuit
        for _ in range(5):
            try:
                limiter.execute(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass
        
        # Reset
        limiter.reset()
        
        stats = limiter.get_stats()
        assert stats['requests_made'] == 0
        assert stats['circuit_breaker_state'] == 'closed'


@pytest.mark.slow
class TestIntegration:
    """Integration tests for rate limiter."""
    
    def test_concurrent_requests_limited(self):
        """Test that concurrent requests are properly limited."""
        limiter = WFSRateLimiter(
            requests_per_second=20.0,
            max_concurrent=3,
            enable_circuit_breaker=False,
        )
        
        active_count = []
        lock = threading.Lock()
        
        def slow_request():
            with lock:
                active_count.append(limiter.concurrency_limiter.get_active_count())
            time.sleep(0.1)
            return "done"
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(
                target=lambda: limiter.execute(slow_request)
            )
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        # Should never have more than 3 concurrent
        assert max(active_count) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
