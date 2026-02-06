"""
Rate Limiter for OpenAI API Calls

Prevents cost overruns and API abuse by enforcing:
1. Request rate limits (requests/minute, requests/hour)
2. Cost limits ($ per hour)
3. Sliding window tracking

Usage:
    from decant.rate_limiter import RateLimiter

    limiter = RateLimiter(
        requests_per_minute=20,
        requests_per_hour=500,
        cost_limit_per_hour=5.0
    )

    # Before making API call
    if limiter.check_and_increment():
        response = client.chat.completions.create(...)
        limiter.record_cost(cost=0.02)
    else:
        raise RateLimitError("Rate limit exceeded")
"""

import time
import logging
from collections import deque
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 20
    requests_per_hour: int = 500
    cost_limit_per_hour: float = 5.0  # USD

    # Warning thresholds (percentage of limit)
    warning_threshold: float = 0.8  # Warn at 80% of limit


@dataclass
class RequestRecord:
    """Record of a single API request."""

    timestamp: float
    cost: float = 0.0
    model: str = "unknown"
    tokens: int = 0


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class RateLimiter:
    """
    Sliding window rate limiter for OpenAI API calls.

    Thread-safe for concurrent use (uses time-based windows, not locks).
    """

    def __init__(
        self,
        requests_per_minute: int = 20,
        requests_per_hour: int = 500,
        cost_limit_per_hour: float = 5.0
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute (default: 20)
            requests_per_hour: Maximum requests per hour (default: 500)
            cost_limit_per_hour: Maximum cost per hour in USD (default: $5.00)
        """
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            cost_limit_per_hour=cost_limit_per_hour
        )

        # Sliding windows (deque for efficient popleft)
        self.requests_minute: deque = deque()  # Last 60 seconds
        self.requests_hour: deque = deque()    # Last 3600 seconds

        # Statistics
        self.total_requests: int = 0
        self.total_cost: float = 0.0
        self.session_start: float = time.time()

        logger.info(
            f"RateLimiter initialized: "
            f"{self.config.requests_per_minute} req/min, "
            f"{self.config.requests_per_hour} req/hour, "
            f"${self.config.cost_limit_per_hour}/hour"
        )

    def _clean_windows(self, current_time: float):
        """Remove expired records from sliding windows."""

        # Clean minute window (60 seconds)
        minute_cutoff = current_time - 60
        while self.requests_minute and self.requests_minute[0].timestamp < minute_cutoff:
            self.requests_minute.popleft()

        # Clean hour window (3600 seconds)
        hour_cutoff = current_time - 3600
        while self.requests_hour and self.requests_hour[0].timestamp < hour_cutoff:
            self.requests_hour.popleft()

    def check_limits(self) -> Dict[str, Any]:
        """
        Check current usage against limits.

        Returns:
            dict with:
                - allowed (bool): Whether request is allowed
                - reason (str): Reason if not allowed
                - usage (dict): Current usage stats
        """
        current_time = time.time()
        self._clean_windows(current_time)

        # Count requests in windows
        requests_in_minute = len(self.requests_minute)
        requests_in_hour = len(self.requests_hour)

        # Calculate cost in last hour
        cost_in_hour = sum(r.cost for r in self.requests_hour)

        # Check limits
        limits_status = {
            'allowed': True,
            'reason': '',
            'usage': {
                'requests_per_minute': requests_in_minute,
                'requests_per_hour': requests_in_hour,
                'cost_per_hour': cost_in_hour,
                'total_requests': self.total_requests,
                'total_cost': self.total_cost,
                'session_duration': current_time - self.session_start
            }
        }

        # Check minute limit
        if requests_in_minute >= self.config.requests_per_minute:
            limits_status['allowed'] = False
            limits_status['reason'] = (
                f"Minute limit exceeded: {requests_in_minute}/{self.config.requests_per_minute} requests. "
                f"Wait {60 - (current_time - self.requests_minute[0].timestamp):.0f}s."
            )
            logger.warning(limits_status['reason'])
            return limits_status

        # Check hour limit
        if requests_in_hour >= self.config.requests_per_hour:
            limits_status['allowed'] = False
            limits_status['reason'] = (
                f"Hour limit exceeded: {requests_in_hour}/{self.config.requests_per_hour} requests. "
                f"Wait {3600 - (current_time - self.requests_hour[0].timestamp):.0f}s."
            )
            logger.warning(limits_status['reason'])
            return limits_status

        # Check cost limit
        if cost_in_hour >= self.config.cost_limit_per_hour:
            limits_status['allowed'] = False
            limits_status['reason'] = (
                f"Cost limit exceeded: ${cost_in_hour:.2f}/${self.config.cost_limit_per_hour:.2f} in last hour. "
                f"Wait {3600 - (current_time - self.requests_hour[0].timestamp):.0f}s."
            )
            logger.warning(limits_status['reason'])
            return limits_status

        # Check warning thresholds
        if requests_in_minute >= self.config.requests_per_minute * self.config.warning_threshold:
            logger.warning(
                f"⚠️  Approaching minute limit: {requests_in_minute}/{self.config.requests_per_minute}"
            )

        if cost_in_hour >= self.config.cost_limit_per_hour * self.config.warning_threshold:
            logger.warning(
                f"⚠️  Approaching cost limit: ${cost_in_hour:.2f}/${self.config.cost_limit_per_hour:.2f}"
            )

        return limits_status

    def check_and_increment(self) -> bool:
        """
        Check if request is allowed and increment counters.

        Returns:
            True if request allowed, False otherwise

        Raises:
            RateLimitError: If rate limit exceeded
        """
        status = self.check_limits()

        if not status['allowed']:
            raise RateLimitError(status['reason'])

        # Record request
        current_time = time.time()
        record = RequestRecord(timestamp=current_time)

        self.requests_minute.append(record)
        self.requests_hour.append(record)
        self.total_requests += 1

        logger.debug(
            f"Request allowed: {status['usage']['requests_per_minute']}/{self.config.requests_per_minute} req/min, "
            f"{status['usage']['requests_per_hour']}/{self.config.requests_per_hour} req/hour"
        )

        return True

    def record_cost(
        self,
        cost: float,
        model: str = "unknown",
        tokens: int = 0
    ):
        """
        Record cost of most recent request.

        Args:
            cost: Cost in USD
            model: Model name
            tokens: Number of tokens used
        """
        if not self.requests_hour:
            logger.warning("No recent request to record cost for")
            return

        # Update most recent request
        latest_request = self.requests_hour[-1]
        latest_request.cost = cost
        latest_request.model = model
        latest_request.tokens = tokens

        self.total_cost += cost

        logger.debug(f"Recorded cost: ${cost:.4f} (model: {model}, tokens: {tokens})")

    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        current_time = time.time()
        self._clean_windows(current_time)

        requests_in_minute = len(self.requests_minute)
        requests_in_hour = len(self.requests_hour)
        cost_in_hour = sum(r.cost for r in self.requests_hour)

        return {
            'requests_per_minute': requests_in_minute,
            'requests_per_hour': requests_in_hour,
            'cost_per_hour': cost_in_hour,
            'total_requests': self.total_requests,
            'total_cost': self.total_cost,
            'session_duration_minutes': (current_time - self.session_start) / 60,
            'limits': {
                'requests_per_minute_limit': self.config.requests_per_minute,
                'requests_per_hour_limit': self.config.requests_per_hour,
                'cost_per_hour_limit': self.config.cost_limit_per_hour
            },
            'utilization': {
                'minute': f"{(requests_in_minute / self.config.requests_per_minute) * 100:.1f}%",
                'hour': f"{(requests_in_hour / self.config.requests_per_hour) * 100:.1f}%",
                'cost': f"{(cost_in_hour / self.config.cost_limit_per_hour) * 100:.1f}%"
            }
        }

    def reset(self):
        """Reset all counters and windows."""
        self.requests_minute.clear()
        self.requests_hour.clear()
        self.total_requests = 0
        self.total_cost = 0.0
        self.session_start = time.time()
        logger.info("Rate limiter reset")

    def wait_if_needed(self, max_wait_seconds: int = 60) -> float:
        """
        Wait if rate limit would be exceeded.

        Args:
            max_wait_seconds: Maximum time to wait (default: 60s)

        Returns:
            Seconds waited

        Raises:
            RateLimitError: If wait time exceeds max_wait_seconds
        """
        status = self.check_limits()

        if status['allowed']:
            return 0.0

        # Calculate wait time from reason string
        import re
        match = re.search(r'Wait (\d+)s', status['reason'])
        if match:
            wait_time = int(match.group(1))

            if wait_time > max_wait_seconds:
                raise RateLimitError(
                    f"Required wait time ({wait_time}s) exceeds maximum ({max_wait_seconds}s)"
                )

            logger.info(f"Rate limit reached. Waiting {wait_time}s...")
            time.sleep(wait_time + 1)  # Add 1s buffer
            return wait_time + 1

        return 0.0


# Global rate limiter instance
_global_limiter: Optional[RateLimiter] = None


def get_global_limiter(
    requests_per_minute: int = 20,
    requests_per_hour: int = 500,
    cost_limit_per_hour: float = 5.0
) -> RateLimiter:
    """
    Get or create global rate limiter instance.

    Singleton pattern for easy integration across modules.
    """
    global _global_limiter

    if _global_limiter is None:
        _global_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            cost_limit_per_hour=cost_limit_per_hour
        )

    return _global_limiter


# Export key classes and functions
__all__ = [
    'RateLimiter',
    'RateLimitError',
    'RateLimitConfig',
    'get_global_limiter'
]
