"""Utility functions for input sanitization, caching, and data validation."""

import hashlib
import json
import logging
import re
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_text_input(text: str, max_length: int = 5000) -> str:
    """Sanitize user text input to prevent prompt injection."""
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    dangerous_patterns = [
        r'ignore[\s\.\,\:\;]+previous',
        r'ignore[\s\.\,\:\;]+all',
        r'ignore[\s\.\,\:\;]+the',
        r'ignore[\s\.\,\:\;]+above',
        r'disregard[\s\.\,\:\;]+previous',
        r'forget[\s\.\,\:\;]+previous',

        # System/role markers
        r'system[\s\.\,\:\;]*:',
        r'assistant[\s\.\,\:\;]*:',
        r'user[\s\.\,\:\;]*:',
        r'\[INST\]',
        r'\[/INST\]',
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'<\|system\|>',
        r'<\|assistant\|>',

        # Direct instruction injection
        r'new\s+instruction',
        r'override\s+instruction',
        r'system\s+message',
        r'you\s+are\s+now',
        r'act\s+as',
        r'pretend\s+to\s+be',

        # Multi-line injection patterns
        r'===+\s*system',
        r'---+\s*system',
        r'```\s*system',

        # Extract/reveal patterns
        r'print\s+your',
        r'reveal\s+your',
        r'what\s+are\s+your',
        r'show\s+me\s+your',
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove excessive newlines (keep max 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove non-printable characters (except newlines, tabs)
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

    # Strip leading/trailing whitespace
    text = text.strip()

    if len(text) > 0:
        suspicious_indicators = [
            'extract', 'override', 'bypass', 'jailbreak',
            'instruction set', 'new role', 'prompt'
        ]
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in suspicious_indicators):
            logger.warning(f"Suspicious input detected (contains: {[i for i in suspicious_indicators if i in text_lower]})")

    return text


def validate_image_upload(file_bytes: bytes, max_size_mb: int = 10) -> bool:
    """Validate uploaded image file by size and magic bytes."""
    # Check size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        logger.warning(f"Image too large: {size_mb:.2f}MB > {max_size_mb}MB")
        return False

    # Check magic bytes for common image formats
    magic_bytes = {
        b'\xFF\xD8\xFF': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF',
        b'RIFF': 'WEBP',  # Needs additional check
    }

    for magic, format_name in magic_bytes.items():
        if file_bytes.startswith(magic):
            logger.info(f"Valid {format_name} image detected")
            return True

    logger.warning("Unknown or invalid image format")
    return False


class LLMCache:
    """File-based cache for LLM responses with TTL expiry."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        if cache_dir is None:
            cache_dir = Path.cwd() / '.cache' / 'llm'

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

        logger.info(f"LLM cache initialized at {self.cache_dir}")

    def _get_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        # Include model and kwargs in hash
        cache_input = {
            'prompt': prompt,
            'model': model,
            **kwargs
        }
        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(self, prompt: str, model: str, **kwargs) -> Optional[Any]:
        """Get cached response if available and not expired."""
        cache_key = self._get_cache_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check if expired
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > self.ttl:
            logger.debug(f"Cache expired for key {cache_key[:8]}...")
            cache_file.unlink()
            return None

        # Load cached response
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            logger.info(f"Cache HIT for key {cache_key[:8]}...")
            return cached_data['response']
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None

    def set(self, prompt: str, model: str, response: Any, **kwargs) -> None:
        """Store response in cache."""
        cache_key = self._get_cache_key(prompt, model, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                'prompt': prompt[:200],  # Store truncated prompt for debugging
                'model': model,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Cache SET for key {cache_key[:8]}...")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")

    def clear(self) -> None:
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")


# Global cache instance
_llm_cache = LLMCache()


def with_llm_cache(func: Callable) -> Callable:
    """Decorator to cache LLM API call results."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract prompt from args/kwargs
        # Assumes prompt is first non-self argument
        if len(args) > 1:
            prompt = str(args[1])
        elif 'prompt' in kwargs:
            prompt = kwargs['prompt']
        elif 'tasting_notes' in kwargs:
            prompt = kwargs['tasting_notes']
        elif 'wine_name' in kwargs:
            prompt = kwargs['wine_name']
        else:
            # Can't cache without a prompt identifier
            return func(*args, **kwargs)

        # Try to get from cache
        from decant.config import OPENAI_MODEL
        cached_result = _llm_cache.get(prompt, OPENAI_MODEL)
        if cached_result is not None:
            return cached_result

        # Call function
        result = func(*args, **kwargs)

        # Store in cache
        _llm_cache.set(prompt, OPENAI_MODEL, result)

        return result

    return wrapper


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide with zero-division protection."""
    if denominator == 0:
        logger.warning(f"Division by zero: {numerator}/{denominator}, returning {default}")
        return default
    return numerator / denominator


def handle_api_error(error: Exception, operation: str) -> None:
    """Log API errors with context."""
    error_type = type(error).__name__
    logger.error(f"API error during {operation}: {error_type} - {str(error)}")


def validate_wine_features(features: dict) -> bool:
    """Validate wine feature dict has required keys in valid range."""
    required_features = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']

    for feature in required_features:
        if feature not in features:
            logger.error(f"Missing required feature: {feature}")
            return False

        value = features[feature]
        if not isinstance(value, (int, float)):
            logger.error(f"Feature {feature} must be numeric, got {type(value)}")
            return False

        if not (1.0 <= value <= 10.0):
            logger.error(f"Feature {feature} out of range [1-10]: {value}")
            return False

    return True





from decant.constants import AlgorithmConstants

class Constants:
    """Legacy constants wrapper. Use AlgorithmConstants directly."""

    # Import all constants from new module
    EXPONENTIAL_ALPHA = AlgorithmConstants.EXPONENTIAL_ALPHA
    BAYESIAN_ALPHA = EXPONENTIAL_ALPHA

    ACIDITY_BODY_EPSILON = AlgorithmConstants.ACIDITY_BODY_EPSILON
    ACIDITY_BODY_WEIGHT = AlgorithmConstants.ACIDITY_BODY_WEIGHT

    COLOR_MATCH_BONUS = AlgorithmConstants.COLOR_MATCH_BONUS
    SWEETNESS_MATCH_BONUS = AlgorithmConstants.SWEETNESS_MATCH_BONUS
    SPARKLING_MATCH_BONUS = AlgorithmConstants.SPARKLING_MATCH_BONUS

    LLM_CACHE_TTL_HOURS = AlgorithmConstants.LLM_CACHE_TTL_HOURS
    MAX_TEXT_INPUT_LENGTH = AlgorithmConstants.MAX_TEXT_INPUT_LENGTH
    MAX_IMAGE_SIZE_MB = AlgorithmConstants.MAX_IMAGE_SIZE_MB

    LIKELIHOOD_STRONG_MATCH = 75.0
    LIKELIHOOD_WORTH_TRYING = 60.0
    LIKELIHOOD_EXPLORE = 45.0
