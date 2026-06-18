"""Constants, enums, and validation schemas for Decant."""

from enum import Enum

from pydantic import BaseModel, Field


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
    """LEGACY: not currently the source of truth for verdict assignment.

    The active verdict logic lives in
    `decant.palate_engine.PalateEngine.calculate_match` and considers
    `palate_match` and `confidence_factor` as separate dimensions
    rather than multiplying them. This enum is kept around in case
    a future caller wants a single-threshold mapping, but the labels
    and thresholds here may drift from the active logic.

    If you're trying to understand what verdict the user sees, read
    palate_engine.calculate_match, not this enum.
    """
    STRONG_MATCH = ("Strong Match", 60.0)
    WORTH_TRYING = ("Worth Trying", 50.0)
    EXPLORE = ("Explore", 40.0)
    DIFFERENT_STYLE = ("Different Style", 0.0)
    FIRST_WINE = ("First Wine", 0.0)

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


class AlgorithmConstants:
    """Algorithm tuning parameters."""

    EXPONENTIAL_ALPHA = 0.4        # Confidence decay coefficient
    ACIDITY_BODY_EPSILON = 0.1     # Division-by-zero guard for acidity/body ratio
    ACIDITY_BODY_WEIGHT = 2.0      # Weight for acidity/body ratio in palate score

    COLOR_MATCH_BONUS = 5.0
    SWEETNESS_MATCH_BONUS = 3.0
    SPARKLING_MATCH_BONUS = 2.0

    LLM_CACHE_TTL_HOURS = 24
    MAX_TEXT_INPUT_LENGTH = 5000
    MAX_IMAGE_SIZE_MB = 10

    MAX_RETRIES = 3
    RETRY_MIN_WAIT_SECONDS = 2
    RETRY_MAX_WAIT_SECONDS = 10
    RETRY_MULTIPLIER = 1


class TechnicalProfile(BaseModel):
    """Validation schema for LLM-extracted technical wine profile."""

    acidity: float = Field(..., ge=1.0, le=10.0, description="Acidity level (1-10)")
    fruitiness: float = Field(..., ge=1.0, le=10.0, description="Fruitiness level (1-10)")
    body: float = Field(..., ge=1.0, le=10.0, description="Body level (1-10)")
    minerality: float = Field(..., ge=1.0, le=10.0, description="Minerality level (1-10)")
    tannin: float = Field(..., ge=1.0, le=10.0, description="Tannin level (1-10)")


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
    overall_score: float = Field(..., ge=1.0, le=10.0)
    price_eur: float = Field(..., ge=0.0)

    # Geography
    country: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)

    # Attributes
    wine_color: WineColor
    is_sparkling: bool
    is_natural: bool
    sweetness: Sweetness

    # Features
    acidity: float = Field(..., ge=1.0, le=10.0)
    minerality: float = Field(..., ge=1.0, le=10.0)
    fruitiness: float = Field(..., ge=1.0, le=10.0)
    tannin: float = Field(..., ge=1.0, le=10.0)
    body: float = Field(..., ge=1.0, le=10.0)


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


class UIConstants:
    """UI-related constants."""

    # Per-wine-color chart palette. Plotly cannot read CSS variables, so
    # components.py imports this mapping directly to avoid a second
    # hand-maintained copy.
    #
    # These intentionally diverge from the muted CSS feature-bar tokens
    # (--gold/--wine/--rose/--orange). On the consolidated palate radar
    # all colours overlay in one plot, and the brand tokens are all warm
    # and close in value, so their translucent fills blended into one
    # indistinguishable mass. Here the hues are pulled apart (a brighter
    # yellow-gold, a vivid red, a clearly lighter pink) and each colour
    # carries a distinct marker `symbol`, so traces stay separable where
    # they overlap - including for colour-blind readers. Fills are kept
    # light so stacked overlaps don't muddy.
    WINE_COLORS_CHART = {
        WineColor.WHITE: {'primary': '#C9A227', 'fill': 'rgba(201, 162, 39, 0.12)', 'symbol': 'circle'},
        WineColor.RED: {'primary': '#B0142F', 'fill': 'rgba(176, 20, 47, 0.12)', 'symbol': 'diamond'},
        WineColor.ROSE: {'primary': '#E879A6', 'fill': 'rgba(232, 121, 166, 0.12)', 'symbol': 'square'},
        WineColor.ORANGE: {'primary': '#D2691E', 'fill': 'rgba(210, 105, 30, 0.12)', 'symbol': 'triangle-up'}
    }

    # Feature display names
    FEATURE_LABELS = {
        'acidity': 'Acidity',
        'minerality': 'Minerality',
        'fruitiness': 'Fruitiness',
        'tannin': 'Tannin',
        'body': 'Body'
    }
