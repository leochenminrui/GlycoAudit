"""Rate limiting utilities for the glyco mirror."""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiolimiter import AsyncLimiter


class RateLimiter:
    """Global rate limiter for HTTP requests."""

    def __init__(self, requests_per_second: float = 1.0):
        """Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second.
        """
        self.requests_per_second = requests_per_second
        self._limiter = AsyncLimiter(max_rate=requests_per_second, time_period=1.0)
        self._last_request_time: dict[str, float] = {}

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        await self._limiter.acquire()

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[None]:
        """Context manager for rate limiting."""
        await self.acquire()
        yield

    async def wait_for_host(self, host: str, min_delay: float = 0.5) -> None:
        """Wait for host-specific rate limiting.

        Some hosts may require additional per-host delays.

        Args:
            host: Hostname being accessed.
            min_delay: Minimum delay between requests to same host.
        """
        now = time.time()
        last_time = self._last_request_time.get(host, 0)
        elapsed = now - last_time

        if elapsed < min_delay:
            await asyncio.sleep(min_delay - elapsed)

        self._last_request_time[host] = time.time()


class SyncRateLimiter:
    """Synchronous rate limiter for HTTP requests."""

    def __init__(self, requests_per_second: float = 1.0):
        """Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second.
        """
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time: float = 0
        self._host_times: dict[str, float] = {}

    def wait(self) -> None:
        """Wait to respect rate limit."""
        now = time.time()
        elapsed = now - self._last_request_time

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self._last_request_time = time.time()

    def wait_for_host(self, host: str, min_delay: float = 0.5) -> None:
        """Wait for host-specific rate limiting.

        Args:
            host: Hostname being accessed.
            min_delay: Minimum delay between requests to same host.
        """
        now = time.time()
        last_time = self._host_times.get(host, 0)
        elapsed = now - last_time

        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)

        self._host_times[host] = time.time()


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter: Whether to add random jitter.

    Returns:
        Delay in seconds.
    """
    delay = min(base_delay * (2 ** attempt), max_delay)

    if jitter:
        # Add random jitter (0.5x to 1.5x)
        delay = delay * (0.5 + random.random())

    return delay


def get_retry_delays(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> list[float]:
    """Get list of retry delays for all attempts.

    Args:
        max_retries: Maximum number of retries.
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.

    Returns:
        List of delays for each retry attempt.
    """
    return [
        exponential_backoff(i, base_delay, max_delay, jitter=True)
        for i in range(max_retries)
    ]
