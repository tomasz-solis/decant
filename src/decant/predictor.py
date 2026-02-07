"""
Wine preference predictor with error handling, caching, and retry logic.

This is the improved version of predictor.py with:
- Comprehensive error handling
- LLM response caching
- Retry logic with exponential backoff
- Input sanitization
- Consistent division by zero protection
- Documented magic numbers via Constants
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError, APIError
from pydantic import BaseModel, Field, confloat
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from decant.schema import WineFeatures, WineExtraction
from decant.palate_engine import PalateEngine, PalateScore
from decant.config import OPENAI_MODEL, OPENAI_TEMPERATURE
from decant.constants import AlgorithmConstants, ColumnNames
from decant.palate_formula import add_palate_features_to_dataframe
from decant.error_handling import handle_llm_error, LLMError
from decant.rate_limiter import get_global_limiter, RateLimitError as RLError
from decant.utils import (
    sanitize_text_input,
    safe_divide,
    handle_api_error,
    validate_wine_features,
    Constants,  # Legacy support
    logger,
    _llm_cache
)


class PalateMatch(BaseModel):
    """LLM output for wine recommendation with structured validation."""

    match_score: confloat(ge=0, le=100) = Field(
        ...,
        description="Match score from 0-100 indicating palate compatibility"
    )
    qualitative_analysis: str = Field(
        ...,
        description="Detailed analysis of how the wine compares to preferred wines"
    )
    key_alignment: str = Field(
        ...,
        description="What aspects align with preferences"
    )
    key_concerns: str = Field(
        ...,
        description="What aspects might not align with preferences"
    )
    recommendation: str = Field(
        ...,
        description="Final recommendation (e.g., 'Strong Match', 'Avoid', 'Worth Trying')"
    )


class VinoPredictor:
    """
    Wine preference predictor using In-Context Learning.

    Features:
    - Error handling with retries
    - LLM response caching
    - Input sanitization
    - Comprehensive logging
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize predictor with wine features data.

        Args:
            data_path: Path to wine_features.csv. If None, uses default location.
        """
        # Load environment variables
        load_dotenv()

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)

        # Initialize rate limiter
        self.rate_limiter = get_global_limiter(
            requests_per_minute=20,
            requests_per_hour=500,
            cost_limit_per_hour=5.0
        )
        logger.info("Rate limiter initialized for API cost protection")

        # Load wine features data
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "wine_features.csv"

        try:
            self.df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.df)} wines from history")
        except FileNotFoundError:
            logger.warning(f"Wine features file not found at {data_path}, starting with empty history")
            self.df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading wine features: {e}")
            self.df = pd.DataFrame()

        # Select context examples
        self._refresh_context()

    def _refresh_context(self):
        """Refresh ICL context with latest history data (self-learning)."""
        try:
            # Reload data to get latest entries
            data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "wine_features.csv"
            if data_path.exists():
                self.df = pd.read_csv(data_path)

            # Select fresh context examples
            self.liked_examples = self._select_liked_examples()
            self.disliked_examples = self._select_disliked_examples()

            logger.info(f"Context: {len(self.liked_examples)} liked + {len(self.disliked_examples)} disliked wines")
        except Exception as e:
            logger.error(f"Error refreshing context: {e}")
            self.liked_examples = pd.DataFrame()
            self.disliked_examples = pd.DataFrame()

    def _select_liked_examples(self, top_n: int = 3, target_wine: Optional[dict] = None) -> pd.DataFrame:
        """
        Select top N liked wines based on palate formula.

        Args:
            top_n: Number of examples to return
            target_wine: Optional dict with wine_color, sweetness, is_sparkling for exact matching

        Returns:
            DataFrame with top liked wines
        """
        if len(self.df) == 0:
            return pd.DataFrame()

        liked_df = self.df[self.df['liked'] == True].copy()

        if len(liked_df) == 0:
            return pd.DataFrame()

        # REFACTORED: Use centralized palate formula instead of duplicating logic
        liked_df = add_palate_features_to_dataframe(liked_df)

        # EXACT STYLE MATCHING: Prioritize wines with same attributes
        if target_wine and 'wine_color' in liked_df.columns:
            # Bonus points for matching color
            if 'wine_color' in target_wine and 'wine_color' in liked_df.columns:
                color_match = liked_df['wine_color'] == target_wine['wine_color']
                liked_df.loc[color_match, 'palate_score'] += Constants.COLOR_MATCH_BONUS

            # Bonus for matching sweetness
            if 'sweetness' in target_wine and 'sweetness' in liked_df.columns:
                sweetness_match = liked_df['sweetness'] == target_wine['sweetness']
                liked_df.loc[sweetness_match, 'palate_score'] += Constants.SWEETNESS_MATCH_BONUS

            # Bonus for matching sparkling
            if 'is_sparkling' in target_wine and 'is_sparkling' in liked_df.columns:
                sparkling_match = liked_df['is_sparkling'] == target_wine['is_sparkling']
                liked_df.loc[sparkling_match, 'palate_score'] += Constants.SPARKLING_MATCH_BONUS

        # Sort by palate score and take top N
        top_liked = liked_df.nlargest(min(top_n, len(liked_df)), 'palate_score')

        return top_liked

    def _select_disliked_examples(self, bottom_n: int = 2) -> pd.DataFrame:
        """Select bottom N disliked wines based on palate formula."""
        if len(self.df) == 0:
            return pd.DataFrame()

        disliked_df = self.df[self.df['liked'] == False].copy()

        if len(disliked_df) == 0:
            return pd.DataFrame()

        # REFACTORED: Use centralized palate formula
        disliked_df = add_palate_features_to_dataframe(disliked_df)

        # Take wines with lowest palate score
        return disliked_df.nsmallest(min(bottom_n, len(disliked_df)), 'palate_score')

    def _build_context_prompt(self, features: WineFeatures) -> str:
        """
        Build ICL prompt with examples and new wine features.

        COMPRESSED VERSION - Removed redundant instructions.

        Args:
            features: Extracted features for the new wine

        Returns:
            Formatted prompt with context examples
        """
        prompt = """You are a wine sommelier analyzing palate compatibility based on tasting history.

## LIKED WINES ✓
"""

        # Add liked examples with derived features
        for idx, (_, wine) in enumerate(self.liked_examples.iterrows(), 1):
            acidity_body_ratio = safe_divide(
                wine['acidity'],
                wine['body'] + Constants.ACIDITY_BODY_EPSILON,
                default=wine['acidity']
            )
            prompt += f"""
{idx}. {wine['producer']} ({wine.get('vintage', 'NV')}) - ${wine['price_usd']:.2f}
   Acidity: {wine['acidity']}/10, Minerality: {wine['minerality']}/10, Fruitiness: {wine['fruitiness']}/10
   Tannin: {wine['tannin']}/10, Body: {wine['body']}/10, A/B Ratio: {acidity_body_ratio:.2f}
"""

        # Add disliked examples if available
        if len(self.disliked_examples) > 0:
            prompt += "\n## DISLIKED WINES ✗\n"

            for idx, (_, wine) in enumerate(self.disliked_examples.iterrows(), 1):
                acidity_body_ratio = safe_divide(
                    wine['acidity'],
                    wine['body'] + Constants.ACIDITY_BODY_EPSILON,
                    default=wine['acidity']
                )
                prompt += f"""
{idx}. {wine['producer']} ({wine.get('vintage', 'NV')}) - ${wine['price_usd']:.2f}
   Acidity: {wine['acidity']}/10, Minerality: {wine['minerality']}/10, Fruitiness: {wine['fruitiness']}/10
   Tannin: {wine['tannin']}/10, Body: {wine['body']}/10, A/B Ratio: {acidity_body_ratio:.2f}
"""

        # Add new wine to evaluate
        acidity_body_ratio_new = safe_divide(
            features.acidity,
            features.body + Constants.ACIDITY_BODY_EPSILON,
            default=features.acidity
        )
        prompt += f"""
---

## NEW WINE TO EVALUATE:
Acidity: {features.acidity}/10, Minerality: {features.minerality}/10, Fruitiness: {features.fruitiness}/10
Tannin: {features.tannin}/10, Body: {features.body}/10, A/B Ratio: {acidity_body_ratio_new:.2f}

Analyze how this wine compares to preferences. Focus on:
1. Patterns in liked vs disliked wines (especially Acidity/Body ratio)
2. Feature alignment with preferred profile
3. Potential concerns

Provide JSON with: match_score (0-100), qualitative_analysis, key_alignment, key_concerns, recommendation
"""

        return prompt

    def extract_wine_data(self, wine_name: str) -> WineExtraction:
        """
        Extract complete wine data from just a wine name using LLM.

        Args:
            wine_name: Wine name (e.g., "Fefiñanes Albariño 2022")

        Returns:
            WineExtraction object with all fields populated

        Raises:
            LLMError: If extraction fails after retries
        """
        # Sanitize input
        wine_name = sanitize_text_input(wine_name)

        # Build context from liked wines
        context = ""
        if len(self.liked_examples) > 0:
            context = "\n\n## USER'S TASTE PROFILE (Recent Liked Wines):\n"
            for _, wine in self.liked_examples.tail(3).iterrows():
                context += f"- {wine['producer']}: Acidity {wine['acidity']}/10, Minerality {wine['minerality']}/10\n"

        prompt = f"""Extract COMPLETE HIGH-DIMENSIONAL wine information from this wine name.

{context}

WINE NAME: {wine_name}

BE AGGRESSIVE in inferring all attributes using your encyclopedic wine knowledge.

## REQUIRED FIELDS:

### Basic Info
1. **wine_name**: Full name with vintage (e.g., "Fefiñanes Albariño 2022")
2. **producer**: Winery name
3. **vintage**: Year (or use current year if not specified)
4. **notes**: Professional tasting notes based on typical characteristics
5. **score**: Your quality rating 1-10 based on wine knowledge and reputation

### WINE ORIGIN (MANDATORY - NEVER LEAVE BLANK)
6. **country**: Country of origin - REQUIRED
7. **region**: Specific wine region/appellation - REQUIRED

### HIGH-DIMENSIONAL ATTRIBUTES
8. **wine_color**: MUST be one of: "White", "Red", "Rosé", "Orange"
9. **is_sparkling**: Boolean (True/False)
10. **is_natural**: Boolean (True/False)
11. **sweetness**: MUST be one of: "Dry", "Medium-Dry", "Medium-Sweet", "Sweet"

### Core 5 Flavor Features (1-10 scale)
12. **acidity**: 1-10 (crisp/tart = high)
13. **minerality**: 1-10 (stony/saline = high)
14. **fruitiness**: 1-10 (fruit-forward = high)
15. **tannin**: 1-10 (grippy = high, whites typically 1-3)
16. **body**: 1-10 (full-bodied = high)

⚠️ **NEVER DEFAULT TO 5/10** - Use regional and varietal knowledge to infer accurate values.

Return JSON only with these exact field names.
"""

        try:
            messages = [
                {"role": "system", "content": "You are a wine expert with encyclopedic knowledge of wines, producers, and regions. Return JSON only."},
                {"role": "user", "content": prompt}
            ]

            response = self._call_openai_with_retry(messages)
            extraction = WineExtraction(**response)
            logger.info(f"Extracted wine data: {extraction.wine_name} ({extraction.producer})")
            return extraction

        except Exception as e:
            error_msg = f"Failed to extract wine data from name '{wine_name}': {str(e)}"
            logger.error(error_msg)
            raise LLMError(error_msg) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError)),
        reraise=True
    )
    def _call_openai_with_retry(self, messages: list, response_format: dict = None) -> dict:
        """
        Call OpenAI API with automatic retry on transient errors.

        Args:
            messages: Chat messages
            response_format: Response format specification

        Returns:
            Parsed JSON response

        Raises:
            OpenAIError: If all retries fail
        """
        try:
            # SECURITY FIX: Check rate limits before API call
            try:
                self.rate_limiter.check_and_increment()
            except RLError as e:
                logger.error(f"Rate limit exceeded: {e}")
                # Get current stats for user feedback
                stats = self.rate_limiter.get_stats()
                raise RateLimitError(
                    f"API rate limit exceeded. Current usage: "
                    f"{stats['requests_per_minute']}/{stats['limits']['requests_per_minute_limit']} req/min, "
                    f"{stats['requests_per_hour']}/{stats['limits']['requests_per_hour_limit']} req/hour, "
                    f"${stats['cost_per_hour']:.2f}/${stats['limits']['cost_per_hour_limit']:.2f}/hour. "
                    f"Please wait before retrying."
                )

            logger.debug("Calling OpenAI API...")
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format=response_format or {"type": "json_object"},
                temperature=OPENAI_TEMPERATURE
            )

            # Record cost (estimate based on model pricing)
            # GPT-4o pricing: ~$0.005 per 1K input tokens, ~$0.015 per 1K output tokens
            if hasattr(completion, 'usage'):
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
                estimated_cost = (input_tokens * 0.000005) + (output_tokens * 0.000015)
                self.rate_limiter.record_cost(
                    cost=estimated_cost,
                    model=OPENAI_MODEL,
                    tokens=input_tokens + output_tokens
                )
                logger.debug(f"API cost: ${estimated_cost:.4f} ({input_tokens + output_tokens} tokens)")

            response_content = completion.choices[0].message.content
            logger.debug("OpenAI API call successful")
            return json.loads(response_content)

        except RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying... ({e})")
            raise
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from API")
        except Exception as e:
            handle_api_error(e, "OpenAI API call")
            raise

    def extract_features(self, tasting_notes: str) -> Optional[WineFeatures]:
        """
        Extract numerical features from tasting notes using LLM.

        WITH CACHING: Identical tasting notes return cached features.

        Args:
            tasting_notes: Free-form wine tasting notes

        Returns:
            Validated WineFeatures object or None on error
        """
        # Sanitize input
        tasting_notes = sanitize_text_input(tasting_notes, max_length=Constants.MAX_TEXT_INPUT_LENGTH)

        if not tasting_notes:
            logger.error("Empty tasting notes after sanitization")
            return None

        # Check cache first
        cache_key = f"features_{tasting_notes}"
        cached = _llm_cache.get(cache_key, OPENAI_MODEL)
        if cached:
            logger.info("Using cached feature extraction")
            return WineFeatures(**cached)

        prompt = f"""Analyze these wine tasting notes and extract numerical features on a 1-10 scale:

TASTING NOTES:
{tasting_notes}

Rate each feature:
- Acidity: 1=low/flat, 10=very high/crisp
- Minerality: 1=none, 10=very mineral/saline
- Fruitiness: 1=subtle, 10=very fruity
- Tannin: 1=soft/none, 10=firm/grippy
- Body: 1=light, 10=full-bodied

Return as JSON with fields: acidity, minerality, fruitiness, tannin, body, reasoning
"""

        try:
            messages = [
                {"role": "system", "content": "You are a wine expert extracting features from tasting notes. Return JSON only."},
                {"role": "user", "content": prompt}
            ]

            data = self._call_openai_with_retry(messages)
            features = WineFeatures(**data)

            # Cache result
            _llm_cache.set(cache_key, OPENAI_MODEL, data)

            logger.info(f"Features extracted: Acid={features.acidity}, Mineral={features.minerality}")
            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def calculate_palate_score(
        self,
        wine_features: dict,
        wine_color: str
    ) -> PalateScore:
        """
        Calculate dual-metric palate score using PalateEngine.

        Args:
            wine_features: Dict with acidity, fruitiness, body, tannin, minerality
            wine_color: Wine color for color-specific matching

        Returns:
            PalateScore with both metrics and detailed explanation
        """
        # Validate features
        if not validate_wine_features(wine_features):
            logger.error("Invalid wine features")
            # Return neutral score
            return PalateScore(
                palate_match=50.0,
                likelihood_score=50.0,
                n_samples=0,
                confidence_factor=0.0,
                verdict="⚠️ Invalid Features",
                explanation="Feature validation failed"
            )

        try:
            # Initialize PalateEngine with current history
            engine = PalateEngine(self.df)

            # Calculate dual-metric score
            score = engine.calculate_match(wine_features, wine_color)

            return score
        except Exception as e:
            logger.error(f"Error calculating palate score: {e}")
            # Return neutral score on error
            return PalateScore(
                palate_match=50.0,
                likelihood_score=50.0,
                n_samples=0,
                confidence_factor=0.0,
                verdict="⚠️ Calculation Error",
                explanation=f"Error: {str(e)}"
            )
