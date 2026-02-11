"""Pydantic schemas for Decant data validation."""

from datetime import date as DateType
from typing import List, Optional
from pydantic import BaseModel, Field


class TastingNotes(BaseModel):
    """Structured tasting notes for a wine."""

    appearance: str = Field(..., description="Visual characteristics of the wine")
    nose: str = Field(..., description="Aromatic profile")
    palate: str = Field(..., description="Flavor profile and mouthfeel")
    overall: str = Field(..., description="Overall impression")


class WineScores(BaseModel):
    """Numerical scores for wine attributes."""

    acidity: int = Field(..., ge=1, le=10, description="Acidity level (1-10)")
    body: int = Field(..., ge=1, le=10, description="Body weight (1-10)")
    complexity: int = Field(..., ge=1, le=10, description="Complexity (1-10)")
    finish: int = Field(..., ge=1, le=10, description="Finish length (1-10)")
    overall: float = Field(..., ge=1.0, le=10.0, description="Overall score (1-10)")


class Wine(BaseModel):
    """Complete wine profile with tasting data."""

    wine_id: str = Field(..., description="Unique identifier for the wine")
    producer: str = Field(..., description="Wine producer/winery name")
    vintage: int = Field(..., ge=1900, le=2100, description="Vintage year")
    price_usd: float = Field(..., gt=0, description="Price in USD")
    tasting_notes: TastingNotes
    scores: WineScores
    varietal: Optional[str] = Field(None, description="Grape varietal(s)")
    appellation: Optional[str] = Field(None, description="Appellation or denomination")
    liked: Optional[bool] = Field(None, description="Whether the taster liked this wine")


class ComparisonContext(BaseModel):
    """Context for a wine comparison tasting."""

    occasion: str = Field(..., description="Occasion or purpose of tasting")
    food_pairing: Optional[str] = Field(None, description="Food paired with wines")
    temperature_celsius: Optional[float] = Field(None, ge=-5, le=25, description="Serving temperature")
    glassware: Optional[str] = Field(None, description="Type of glassware used")
    location: Optional[str] = Field(None, description="Tasting location")


class WineComparison(BaseModel):
    """A structured comparison between multiple wines."""

    comparison_id: str = Field(..., description="Unique identifier for comparison")
    date: DateType = Field(..., description="Date of comparison")
    wine_type: str = Field(..., description="Type/style of wine being compared")
    region: str = Field(..., description="Wine region")
    wines: List[Wine] = Field(..., min_length=2, description="Wines being compared")
    comparison_notes: str = Field(..., description="Comparative analysis")
    recommendation: str = Field(..., description="Final recommendation")
    context: ComparisonContext
    taster: Optional[str] = Field(None, description="Name of taster")


class TastingHistory(BaseModel):
    """Collection of wine comparisons."""

    comparisons: List[WineComparison] = Field(default_factory=list)


class WineRecommendationRequest(BaseModel):
    """Request schema for wine recommendations via API."""

    preferences: List[str] = Field(..., min_length=1, description="Wine preferences or requirements")
    budget_usd: Optional[float] = Field(None, gt=0, description="Maximum budget in USD")
    occasion: Optional[str] = Field(None, description="Occasion or use case")
    food_pairing: Optional[str] = Field(None, description="Food to pair with")


class WineRecommendationResponse(BaseModel):
    """Response schema for wine recommendations."""

    recommendations: List[Wine]
    reasoning: str = Field(..., description="Explanation for recommendations")
    alternatives: Optional[List[Wine]] = Field(None, description="Alternative options")


class WineFeatures(BaseModel):
    """LLM-extracted wine features - core 5 features only.

    Matches simplified history.csv schema for automated data entry.
    """

    # Core 5 flavor features (1-10 scale, allows decimals like 5.5, 6.7)
    acidity: float = Field(..., ge=1.0, le=10.0, description="Perceived acidity level (1=low, 10=high)")
    minerality: float = Field(..., ge=1.0, le=10.0, description="Mineral character (1=low, 10=high)")
    fruitiness: float = Field(..., ge=1.0, le=10.0, description="Fruit intensity (1=low, 10=high)")
    tannin: float = Field(..., ge=1.0, le=10.0, description="Tannin level (1=low, 10=high)")
    body: float = Field(..., ge=1.0, le=10.0, description="Body weight (1=light, 10=full)")

    # Metadata
    reasoning: Optional[str] = Field(None, description="Brief explanation of feature extraction")


class WineExtraction(BaseModel):
    """Complete wine data extraction from name or image.

    All fields auto-extracted for HIGH-DIMENSIONAL TASTE SPACE + GEOGRAPHY.
    Zero-friction logging - AI does all the homework.
    """

    wine_name: str = Field(..., description="Full wine name with vintage")
    producer: str = Field(..., description="Producer/winery name")
    vintage: int = Field(..., ge=1900, le=2100, description="Vintage year")
    notes: str = Field(..., description="Professional tasting notes")
    score: float = Field(..., ge=1.0, le=10.0, description="Quality score 1-10")

    # WINE ORIGIN (AUTO-EXTRACTED)
    country: str = Field(..., description="Country of origin (e.g., Spain, France, Italy)")
    region: str = Field(..., description="Wine region (e.g., Rías Baixas, Bordeaux, Tuscany)")

    # HIGH-DIMENSIONAL WINE ATTRIBUTES (AI-inferred)
    wine_color: str = Field(..., description="Wine color: White, Red, Rosé, or Orange")
    is_sparkling: bool = Field(..., description="True if sparkling/champagne/cava/prosecco")
    is_natural: bool = Field(..., description="True if natural/organic/biodynamic wine")
    sweetness: str = Field(..., description="Sweetness level: Dry, Medium-Dry, Medium-Sweet, or Sweet")

    # Core 5 flavor features (allows decimals like 5.5, 6.7)
    acidity: float = Field(..., ge=1.0, le=10.0, description="Acidity level")
    minerality: float = Field(..., ge=1.0, le=10.0, description="Minerality level")
    fruitiness: float = Field(..., ge=1.0, le=10.0, description="Fruitiness level")
    tannin: float = Field(..., ge=1.0, le=10.0, description="Tannin level")
    body: float = Field(..., ge=1.0, le=10.0, description="Body level")
