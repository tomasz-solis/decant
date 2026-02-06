"""
Decant Constants and Enums

Centralized constants, enums, and magic values to eliminate string duplication
and improve type safety.
"""

from enum import Enum
from typing import Tuple
from pydantic import BaseModel, Field, confloat


# =======================
# WINE ATTRIBUTE ENUMS
# =======================

class WineColor(str, Enum):
    """Wine color categories."""
    WHITE = "White"
    RED = "Red"
    ROSE = "Rosé"
    ORANGE = "Orange"


class Sweetness(str, Enum):
    """Wine sweetness levels."""
    DRY = "Dry"
    MEDIUM_DRY = "Medium-Dry"
    MEDIUM_SWEET = "Medium-Sweet"
    SWEET = "Sweet"


class Verdict(Enum):
    """Match verdicts with thresholds and display strings."""
    STRONG_MATCH = ("💙 Strong Match", 75.0)
    WORTH_TRYING = ("🧡 Worth Trying", 60.0)
    EXPLORE = ("🟡 Explore", 45.0)
    DIFFERENT_STYLE = ("⚪ Different Style", 0.0)
    FIRST_WINE = ("🔍 First Wine", 0.0)

    def __init__(self, display: str, threshold: float):
        self.display = display
        self.threshold = threshold

    @classmethod
    def from_score(cls, likelihood_score: float, n_samples: int = 0) -> 'Verdict':
        """Get verdict from likelihood score."""
        if n_samples == 0:
            return cls.FIRST_WINE

        if likelihood_score >= cls.STRONG_MATCH.threshold:
            return cls.STRONG_MATCH
        elif likelihood_score >= cls.WORTH_TRYING.threshold:
            return cls.WORTH_TRYING
        elif likelihood_score >= cls.EXPLORE.threshold:
            return cls.EXPLORE
        else:
            return cls.DIFFERENT_STYLE


# =======================
# COLUMN NAME CONSTANTS
# =======================

class ColumnNames:
    """CSV column names to avoid string hardcoding."""

    # Core identification
    WINE_NAME = "wine_name"
    PRODUCER = "producer"
    VINTAGE = "vintage"

    # User preferences
    LIKED = "liked"
    SCORE = "score"
    NOTES = "notes"
    PRICE = "price"

    # Geography
    COUNTRY = "country"
    REGION = "region"

    # Wine attributes
    WINE_COLOR = "wine_color"
    IS_SPARKLING = "is_sparkling"
    IS_NATURAL = "is_natural"
    SWEETNESS = "sweetness"

    # Core 5 flavor features
    ACIDITY = "acidity"
    MINERALITY = "minerality"
    FRUITINESS = "fruitiness"
    TANNIN = "tannin"
    BODY = "body"

    # Derived features (computed)
    STRUCTURE_SCORE = "structure_score"
    ACIDITY_BODY_RATIO = "acidity_body_ratio"
    PALATE_SCORE = "palate_score"

    @classmethod
    def feature_columns(cls) -> list:
        """Get list of core 5 feature columns."""
        return [cls.ACIDITY, cls.MINERALITY, cls.FRUITINESS, cls.TANNIN, cls.BODY]


# =======================
# ALGORITHM CONSTANTS
# =======================

class AlgorithmConstants:
    """
    Algorithm constants with documentation.

    Previously in utils.Constants - moved here for better organization.
    """

    # EXPONENTIAL CONFIDENCE
    # α = 0.4 chosen as balanced coefficient
    # - Lower (0.2-0.3): More conservative, slower confidence growth
    # - Higher (0.5-0.6): More aggressive, faster confidence growth
    # - 0.4: Balanced - reaches 70% confidence at 3 wines, 86% at 5 wines
    # Validated via cross-validation in notebooks/03_exponential_decay_analysis.ipynb
    EXPONENTIAL_ALPHA = 0.4

    # FEATURE ENGINEERING
    # Small epsilon to prevent division by zero in acidity/body ratio
    # 0.1 chosen as minimal offset that doesn't significantly affect results
    ACIDITY_BODY_EPSILON = 0.1

    # PALATE FORMULA WEIGHTS
    # Acidity/Body ratio multiplied by 2 in palate score calculation
    # Empirically determined to give appropriate weight to this derived feature
    ACIDITY_BODY_WEIGHT = 2.0

    # STYLE MATCHING BONUSES
    # Bonus points added to palate score for exact style matches
    # Used in In-Context Learning example selection
    COLOR_MATCH_BONUS = 5.0
    SWEETNESS_MATCH_BONUS = 3.0
    SPARKLING_MATCH_BONUS = 2.0

    # CACHING
    LLM_CACHE_TTL_HOURS = 24  # Cache LLM responses for 24 hours

    # INPUT LIMITS
    MAX_TEXT_INPUT_LENGTH = 5000  # Max characters for tasting notes
    MAX_IMAGE_SIZE_MB = 10        # Max image upload size

    # RETRY CONFIGURATION
    MAX_RETRIES = 3
    RETRY_MIN_WAIT_SECONDS = 2
    RETRY_MAX_WAIT_SECONDS = 10
    RETRY_MULTIPLIER = 1


# =======================
# LLM RESPONSE VALIDATION SCHEMAS
# =======================

class TechnicalProfile(BaseModel):
    """Validation schema for LLM-extracted technical wine profile."""

    acidity: confloat(ge=1.0, le=10.0) = Field(..., description="Acidity level (1-10)")
    fruitiness: confloat(ge=1.0, le=10.0) = Field(..., description="Fruitiness level (1-10)")
    body: confloat(ge=1.0, le=10.0) = Field(..., description="Body level (1-10)")
    minerality: confloat(ge=1.0, le=10.0) = Field(..., description="Minerality level (1-10)")
    tannin: confloat(ge=1.0, le=10.0) = Field(..., description="Tannin level (1-10)")


class WineMetadata(BaseModel):
    """Validation schema for wine metadata from LLM."""

    name: str = Field(..., description="Wine name")
    region: str = Field(..., description="Wine region")
    style: str = Field(..., description="Wine style description")


class LLMWineAnalysis(BaseModel):
    """Complete validation schema for LLM wine analysis response."""

    wine_metadata: WineMetadata
    technical_profile: TechnicalProfile
    sommelier_verdict: str = Field(..., description="One sentence technical summary")


class ImageExtractionResponse(BaseModel):
    """Validation schema for image-based wine extraction."""

    wine_name: str = Field(..., min_length=1)
    producer: str = Field(..., min_length=1)
    vintage: int = Field(..., ge=1900, le=2100)
    tasting_notes: str = Field(..., min_length=10)
    overall_score: confloat(ge=1.0, le=10.0)
    price_eur: confloat(ge=0.0)

    # Geography
    country: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)

    # Attributes
    wine_color: WineColor
    is_sparkling: bool
    is_natural: bool
    sweetness: Sweetness

    # Features
    acidity: confloat(ge=1.0, le=10.0)
    minerality: confloat(ge=1.0, le=10.0)
    fruitiness: confloat(ge=1.0, le=10.0)
    tannin: confloat(ge=1.0, le=10.0)
    body: confloat(ge=1.0, le=10.0)


# =======================
# FEATURE RANGE CONSTANTS
# =======================

class FeatureRanges:
    """Valid ranges for wine features."""

    MIN_FEATURE_VALUE = 1.0
    MAX_FEATURE_VALUE = 10.0

    MIN_VINTAGE = 1900
    MAX_VINTAGE = 2100

    MIN_SCORE = 1.0
    MAX_SCORE = 10.0

    MIN_PRICE = 0.0
    MAX_PRICE = 10000.0  # Reasonable upper limit


# =======================
# FILE PATH CONSTANTS
# =======================

class FilePaths:
    """Standard file paths used in the application."""

    DATA_DIR = "data"
    PROCESSED_DIR = "data/processed"
    RAW_DIR = "data/raw"
    WINE_IMAGES_DIR = "data/wine_images"
    CACHE_DIR = ".cache"
    LLM_CACHE_DIR = ".cache/llm"

    HISTORY_CSV = "data/history.csv"
    WINE_FEATURES_CSV = "data/processed/wine_features.csv"


# =======================
# UI CONSTANTS
# =======================

class UIConstants:
    """UI-related constants."""

    # Color schemes for radar charts (dark mode)
    WINE_COLORS_CHART = {
        WineColor.WHITE: {'primary': '#FFD700', 'fill': 'rgba(255, 215, 0, 0.4)', 'emoji': '⚪'},
        WineColor.RED: {'primary': '#8B0000', 'fill': 'rgba(139, 0, 0, 0.4)', 'emoji': '🔴'},
        WineColor.ROSE: {'primary': '#FF69B4', 'fill': 'rgba(255, 105, 180, 0.4)', 'emoji': '🌸'},
        WineColor.ORANGE: {'primary': '#FF8C00', 'fill': 'rgba(255, 140, 0, 0.4)', 'emoji': '🟠'}
    }

    # Feature display names
    FEATURE_LABELS = {
        'acidity': 'Acidity',
        'minerality': 'Minerality',
        'fruitiness': 'Fruitiness',
        'tannin': 'Tannin',
        'body': 'Body'
    }

    # Feature emojis
    FEATURE_EMOJIS = {
        'acidity': '⚡',
        'minerality': '💎',
        'fruitiness': '🍇',
        'tannin': '🌰',
        'body': '💪'
    }
