"""
Standardized Error Handling for Decant

Provides consistent error handling patterns across all modules.
"""

import logging
from typing import Optional, Callable, Any, TypeVar, Dict
from functools import wraps
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')


class DecantError(Exception):
    """Base exception for Decant application."""
    pass


class LLMError(DecantError):
    """LLM-related errors (API failures, validation failures)."""
    pass


class DataValidationError(DecantError):
    """Data validation errors."""
    pass


class FileOperationError(DecantError):
    """File operation errors."""
    pass


def handle_llm_error(error: Exception, operation: str, fallback_value: Any = None) -> Any:
    """
    Standardized LLM error handling.

    Args:
        error: Exception that occurred
        operation: Description of operation
        fallback_value: Value to return on error

    Returns:
        fallback_value if error is recoverable, otherwise raises
    """
    error_type = type(error).__name__

    # Validation errors - log and return fallback
    if isinstance(error, ValidationError):
        logger.error(f"LLM response validation failed during {operation}: {error}")
        return fallback_value

    # JSON decode errors - log and return fallback
    if error_type == "JSONDecodeError":
        logger.error(f"Invalid JSON from LLM during {operation}: {error}")
        return fallback_value

    # API rate limits - log warning (retries should handle this)
    if "rate limit" in str(error).lower() or error_type == "RateLimitError":
        logger.warning(f"Rate limit hit during {operation}: {error}")
        # Don't return fallback - let retry logic handle it
        raise LLMError(f"Rate limit during {operation}") from error

    # Other API errors - log and potentially retry
    if "api" in error_type.lower():
        logger.error(f"API error during {operation}: {error}")
        raise LLMError(f"API error during {operation}") from error

    # Unknown errors - log and raise
    logger.error(f"Unexpected error during {operation}: {error_type} - {error}")
    raise LLMError(f"Unexpected error during {operation}") from error


def safe_execute(
    func: Callable[..., T],
    fallback_value: T,
    error_message: str = "Operation failed"
) -> Callable[..., T]:
    """
    Decorator for safe function execution with fallback.

    Args:
        func: Function to execute
        fallback_value: Value to return on error
        error_message: Error message to log

    Returns:
        Wrapped function that returns fallback on error
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{error_message}: {type(e).__name__} - {e}")
            return fallback_value

    return wrapper


def validate_llm_response(
    response: Dict,
    expected_keys: list,
    operation: str
) -> bool:
    """
    Validate LLM response has expected structure.

    Args:
        response: LLM response dict
        expected_keys: List of required keys
        operation: Operation name for logging

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response, dict):
        logger.error(f"LLM response is not a dict during {operation}: {type(response)}")
        return False

    missing_keys = [key for key in expected_keys if key not in response]
    if missing_keys:
        logger.error(f"LLM response missing keys during {operation}: {missing_keys}")
        return False

    return True


class ErrorContext:
    """Context manager for consistent error handling."""

    def __init__(self, operation: str, fallback_value: Any = None):
        self.operation = operation
        self.fallback_value = fallback_value
        self.error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            logger.error(f"Error in {self.operation}: {exc_type.__name__} - {exc_val}")

            # Return True to suppress exception (return fallback)
            # Return False to propagate exception
            if self.fallback_value is not None:
                return True  # Suppress exception
        return False


# Export key functions and classes
__all__ = [
    'DecantError',
    'LLMError',
    'DataValidationError',
    'FileOperationError',
    'handle_llm_error',
    'safe_execute',
    'validate_llm_response',
    'ErrorContext'
]
