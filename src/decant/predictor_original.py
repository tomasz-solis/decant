"""
Wine preference predictor using In-Context Learning (ICL).

This module implements a sophisticated wine recommendation system that uses
few-shot learning with OpenAI to predict palate compatibility.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from decant.schema import WineFeatures, WineExtraction
from decant.palate_engine import PalateEngine, PalateScore
from decant.config import OPENAI_MODEL, OPENAI_TEMPERATURE


class PalateMatch(BaseModel):
    """LLM output for wine recommendation with structured validation."""

    match_score: float = Field(
        ...,
        ge=0,
        le=100,
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

    Loads historical wine data and uses top liked/disliked examples
    as context for predicting preference on new wines.
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

        # Load wine features data
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "wine_features.csv"

        self.df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(self.df)} wines from history")

        # Select context examples (will be refreshed on each prediction for self-learning)
        self._refresh_context()

    def _refresh_context(self):
        """Refresh ICL context with latest history data (self-learning)."""
        # Reload data to get latest entries
        data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "wine_features.csv"
        if data_path.exists():
            self.df = pd.read_csv(data_path)

        # Select fresh context examples
        self.liked_examples = self._select_liked_examples()
        self.disliked_examples = self._select_disliked_examples()

        print(f"✓ Context: {len(self.liked_examples)} liked + {len(self.disliked_examples)} disliked wines")

    def _select_liked_examples(self, top_n: int = 3, target_wine: Optional[dict] = None) -> pd.DataFrame:
        """
        Select top N liked wines based on palate formula.

        EXACT STYLE MATCHING: If target_wine provided, prioritize same color/sweetness/sparkling.

        Args:
            top_n: Number of examples to return
            target_wine: Optional dict with wine_color, sweetness, is_sparkling for exact matching
        """
        liked_df = self.df[self.df['liked'] == True].copy()

        if len(liked_df) == 0:
            return pd.DataFrame()

        # Calculate derived features for palate formula
        liked_df['structure_score'] = liked_df['acidity'] + liked_df['minerality']
        liked_df['acidity_body_ratio'] = liked_df['acidity'] / (liked_df['body'] + 0.1)  # Avoid div by zero
        liked_df['palate_score'] = liked_df['structure_score'] + (liked_df['acidity_body_ratio'] * 2)

        # EXACT STYLE MATCHING: Prioritize wines with same attributes
        if target_wine and 'wine_color' in liked_df.columns:
            # Boost palate score for exact style matches
            exact_matches = liked_df.copy()

            # Bonus points for matching color
            if 'wine_color' in target_wine and 'wine_color' in exact_matches.columns:
                color_match = exact_matches['wine_color'] == target_wine['wine_color']
                exact_matches.loc[color_match, 'palate_score'] += 5

            # Bonus for matching sweetness
            if 'sweetness' in target_wine and 'sweetness' in exact_matches.columns:
                sweetness_match = exact_matches['sweetness'] == target_wine['sweetness']
                exact_matches.loc[sweetness_match, 'palate_score'] += 3

            # Bonus for matching sparkling
            if 'is_sparkling' in target_wine and 'is_sparkling' in exact_matches.columns:
                sparkling_match = exact_matches['is_sparkling'] == target_wine['is_sparkling']
                exact_matches.loc[sparkling_match, 'palate_score'] += 2

            liked_df = exact_matches

        # Sort by palate score and take top N
        top_liked = liked_df.nlargest(min(top_n, len(liked_df)), 'palate_score')

        return top_liked

    def _select_disliked_examples(self, bottom_n: int = 2) -> pd.DataFrame:
        """Select bottom N disliked wines based on palate formula."""
        disliked_df = self.df[self.df['liked'] == False].copy()

        if len(disliked_df) == 0:
            return pd.DataFrame()

        # Calculate derived features for palate formula
        disliked_df['structure_score'] = disliked_df['acidity'] + disliked_df['minerality']
        disliked_df['acidity_body_ratio'] = disliked_df['acidity'] / (disliked_df['body'] + 0.1)
        disliked_df['palate_score'] = disliked_df['structure_score'] + (disliked_df['acidity_body_ratio'] * 2)

        # Take wines with lowest palate score
        return disliked_df.nsmallest(min(bottom_n, len(disliked_df)), 'palate_score')

    def _build_context_prompt(self, features: WineFeatures) -> str:
        """
        Build ICL prompt with examples and new wine features.

        Args:
            features: Extracted features for the new wine

        Returns:
            Formatted prompt with context examples
        """
        prompt = """You are a wine sommelier analyzing palate compatibility.

I have a specific wine preference profile. Based on my tasting history, predict how well a new wine will match my palate.

## MY PREFERENCE PROFILE (from tasting history):

### WINES I LOVED ✓
"""

        # Add liked examples with derived features
        for idx, (_, wine) in enumerate(self.liked_examples.iterrows(), 1):
            acidity_body_ratio = wine['acidity'] / (wine['body'] + 0.1)
            prompt += f"""
{idx}. {wine['producer']} ({wine.get('vintage', 'NV')}) - ${wine['price_usd']:.2f}
   - Acidity: {wine['acidity']}/10
   - Minerality: {wine['minerality']}/10
   - Fruitiness: {wine['fruitiness']}/10
   - Tannin: {wine['tannin']}/10
   - Body: {wine['body']}/10
   - Acidity/Body Ratio: {acidity_body_ratio:.2f} (higher = more crisp/refreshing)
"""

        # Add disliked examples if available with derived features
        if len(self.disliked_examples) > 0:
            prompt += "\n### WINES I DISLIKED ✗\n"

            for idx, (_, wine) in enumerate(self.disliked_examples.iterrows(), 1):
                acidity_body_ratio = wine['acidity'] / (wine['body'] + 0.1)
                prompt += f"""
{idx}. {wine['producer']} ({wine.get('vintage', 'NV')}) - ${wine['price_usd']:.2f}
   - Acidity: {wine['acidity']}/10
   - Minerality: {wine['minerality']}/10
   - Fruitiness: {wine['fruitiness']}/10
   - Tannin: {wine['tannin']}/10
   - Body: {wine['body']}/10
   - Acidity/Body Ratio: {acidity_body_ratio:.2f}
"""

        # Add new wine to evaluate with derived features
        acidity_body_ratio_new = features.acidity / (features.body + 0.1)
        prompt += f"""

---

## NEW WINE TO EVALUATE:

Features extracted from tasting notes:
- Acidity: {features.acidity}/10
- Minerality: {features.minerality}/10
- Fruitiness: {features.fruitiness}/10
- Tannin: {features.tannin}/10
- Body: {features.body}/10
- Acidity/Body Ratio: {acidity_body_ratio_new:.2f}

---

## YOUR TASK:

Analyze how this wine compares to my preferences. Consider:

1. **Pattern Recognition**: What patterns do you see in the wines I loved vs. disliked? Pay special attention to the Acidity/Body ratio as a key preference indicator.
2. **Feature Alignment**: How do this wine's features align with my preferred profile?
3. **Derived Features**: Compare the structure score (acidity + minerality) and acidity/body ratio to my preferred wines.
4. **Trade-offs**: Are there any concerning aspects that might not work for my palate?

Provide:
- A match_score (0-100) indicating palate compatibility
- A qualitative_analysis explaining the reasoning
- key_alignment: What matches my preferences
- key_concerns: What might not work for me
- recommendation: Your verdict (e.g., "Strong Buy", "Worth Trying", "Skip")
"""

        return prompt

    def extract_features(self, tasting_notes: str) -> WineFeatures:
        """
        Extract numerical features from tasting notes using LLM.

        Args:
            tasting_notes: Free-form wine tasting notes

        Returns:
            Validated WineFeatures object
        """
        prompt = f"""
Analyze these wine tasting notes and extract numerical features on a 1-10 scale:

TASTING NOTES:
{tasting_notes}

Rate each feature:
- Acidity: 1=low/flat, 10=very high/crisp
- Minerality: 1=none, 10=very mineral/saline
- Fruitiness: 1=subtle, 10=very fruity
- Tannin: 1=soft/none, 10=firm/grippy
- Body: 1=light, 10=full-bodied

Look for keywords:
- Acidity: "crisp", "fresh", "zesty", "bright", "tart"
- Minerality: "mineral", "saline", "stony", "flinty", "chalky", "steely"
- Fruitiness: fruit names, "fruity", "ripe", "juicy"
- Tannin: "tannic", "grippy", "structured", "firm", "velvety"
- Body: "light", "medium", "full", "weight", "texture"
"""

        try:
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a wine expert extracting features from tasting notes. Return JSON only."},
                    {"role": "user", "content": prompt + "\n\nReturn as JSON with fields: acidity, minerality, fruitiness, tannin, body, reasoning"}
                ],
                response_format={"type": "json_object"},
                temperature=OPENAI_TEMPERATURE
            )

            import json
            data = json.loads(completion.choices[0].message.content)
            return WineFeatures(**data)

        except Exception as e:
            print(f"⚠️  Error extracting features: {e}")
            # Re-raise the exception - don't return fake generic values
            # The caller should handle the failure appropriately
            raise Exception(f"Failed to extract wine features: {str(e)}")

    def extract_wine_data(self, wine_name: str) -> WineExtraction:
        """
        Extract complete wine data from just a wine name.

        Uses LLM knowledge to fill in producer, vintage, tasting notes,
        score, and all 5 flavor features automatically.

        Args:
            wine_name: Wine name (e.g., "Fefiñanes Albariño 2022")

        Returns:
            WineExtraction object with all fields populated
        """
        # Build self-learning context from recent liked wines
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
   - Use producer location, wine style, or grape variety knowledge
   - Examples:
     * Fefiñanes → Spain
     * Domaine de la Romanée-Conti → France
     * Gaja → Italy
     * Ridge Vineyards → USA
   - NEVER return "Unknown" - use your encyclopedic knowledge

7. **region**: Specific wine region/appellation - REQUIRED
   - Look for DO, DOCa, AOC, AVA indicators in wine name
   - Use encyclopedic wine knowledge to identify region from producer
   - Examples:
     * Fefiñanes Albariño → "Rías Baixas" (Albariño = Rías Baixas DO)
     * Château Margaux → "Bordeaux"
     * Antinori Chianti → "Tuscany"
     * Ridge Monte Bello → "Santa Cruz Mountains"
     * "Ribera del Duero" in name → "Ribera del Duero"
     * Rioja producer → "Rioja"
   - If specific DO/AOC not in name, use producer's PRIMARY region
   - NEVER return "Unknown" - research the producer if needed

### HIGH-DIMENSIONAL ATTRIBUTES (AGGRESSIVE INFERENCE REQUIRED)

8. **wine_color**: MUST be one of: "White", "Red", "Rosé", "Orange"
   - Use grape variety, region, and producer to determine
   - Examples: Albariño = White, Pinot Noir = Red, Tempranillo = Red

9. **is_sparkling**: Boolean (True/False)
   - True if: Champagne, Cava, Prosecco, Espumante, Crémant, Sekt, or "Sparkling" in name
   - False for all still wines

10. **is_natural**: Boolean (True/False)
   - True if: Label mentions "natural", "organic", "biodynamic", "bio", "nature"
   - True if: Known natural wine producer (e.g., La Stoppa, Frank Cornelissen)
   - False if conventional/industrial

11. **sweetness**: MUST be one of: "Dry", "Medium-Dry", "Medium-Sweet", "Sweet"
   - Use REGION + STYLE inference:
     * Rías Baixas Albariño = "Dry"
     * German Kabinett = "Medium-Sweet"
     * German Spätlese = "Medium-Sweet"
     * German Auslese = "Sweet"
     * Champagne Brut = "Dry"
     * Champagne Demi-Sec = "Medium-Sweet"
     * Sauternes = "Sweet"
     * Moscato d'Asti = "Sweet"
     * Most still table wines = "Dry" unless specified

### Core 5 Flavor Features (1-10 scale)
12. **acidity**: 1-10 (crisp/tart = high)
13. **minerality**: 1-10 (stony/saline = high)
14. **fruitiness**: 1-10 (fruit-forward = high)
15. **tannin**: 1-10 (grippy = high, whites typically 1-3)
16. **body**: 1-10 (full-bodied = high)

## CRITICAL: ZERO-DEFAULT BAN (5/10 Ban)

⚠️ **YOU MUST NEVER DEFAULT TO 5/10** ⚠️

If data is missing, you MUST infer values based on the wine style, region, and grape variety:

**Style-Based Inference Rules:**

- **Albariño (Rías Baixas)**: acidity=8-9, minerality=8-9, fruitiness=7, tannin=1-2, body=5-6
- **Riesling (Germany)**: acidity=9, minerality=8, fruitiness=8, tannin=1, body=4-5
- **Chardonnay (Burgundy)**: acidity=7, minerality=7, fruitiness=6, tannin=2, body=7-8
- **Tempranillo (Rioja)**: acidity=6, minerality=5, fruitiness=7, tannin=7-8, body=8
- **Pinot Noir (Burgundy)**: acidity=7, minerality=7, fruitiness=8, tannin=5-6, body=6
- **Cabernet Sauvignon (Napa)**: acidity=6, minerality=4, fruitiness=8, tannin=8-9, body=9
- **Manzanilla (Jerez)**: acidity=8, minerality=9, fruitiness=3, tannin=1, body=3
- **Champagne (Brut)**: acidity=9, minerality=7, fruitiness=6, tannin=1, body=5

**General Regional Inference:**
- **Atlantic/Coastal wines** (Galicia, Loire, Chablis): HIGH acidity + minerality
- **Mediterranean wines** (Rioja, Tuscany): MEDIUM acidity, HIGHER body
- **New World wines** (California, Australia): HIGHER fruitiness, FULLER body
- **Fortified/Sherry**: HIGH acidity or alcohol, SPECIFIC styles (dry to sweet)

USE YOUR ENCYCLOPEDIC KNOWLEDGE. NEVER use 5/10 as a lazy default.

## INFERENCE RULES:

**Sweetness Inference Logic:**
- If wine name contains "Brut", "Sec", "Dry", "Trocken" → "Dry"
- If wine name contains "Kabinett", "Feinherb" → "Medium-Dry"
- If wine name contains "Spätlese", "Demi-Sec", "Halbtrocken" → "Medium-Sweet"
- If wine name contains "Auslese", "Sauternes", "Moscato", "Dulce", "Dolce" → "Sweet"
- Default for table wines: "Dry"

**Color Inference:**
- Albariño, Verdejo, Sauvignon Blanc, Chardonnay, Riesling → "White"
- Tempranillo, Merlot, Cabernet, Pinot Noir, Syrah → "Red"
- Check producer's typical style if unclear

Use your knowledge. BE AGGRESSIVE. Make the best inference even with limited info.
"""

        try:
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a wine expert with encyclopedic knowledge of wines, producers, and regions. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=OPENAI_TEMPERATURE
            )

            import json
            data = json.loads(completion.choices[0].message.content)
            extraction = WineExtraction(**data)
            print(f"✓ Extracted: {extraction.wine_name} ({extraction.producer})")
            return extraction

        except Exception as e:
            print(f"⚠️  Error extracting wine data: {e}")
            # Re-raise - don't return fake generic values
            raise Exception(f"Failed to extract wine data from name: {str(e)}")

    def predict_match(self, tasting_notes: str) -> Tuple[WineFeatures, PalateMatch]:
        """
        Predict palate match for a wine based on tasting notes.

        Args:
            tasting_notes: Free-form wine tasting notes

        Returns:
            Tuple of (extracted features, palate match prediction)
        """
        # Refresh context for self-learning
        self._refresh_context()

        # Step 1: Extract features with Pydantic validation
        print("🔍 Extracting wine features...")
        features = self.extract_features(tasting_notes)
        print(f"✓ Features extracted: Acid={features.acidity}, Mineral={features.minerality}, Fruit={features.fruitiness}")

        # Step 2: Build ICL prompt with context
        print("🧠 Analyzing palate compatibility with In-Context Learning...")
        context_prompt = self._build_context_prompt(features)

        # Step 3: Get match prediction with JSON parsing
        try:
            completion = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert sommelier analyzing wine palate compatibility. Return JSON only with fields: match_score, qualitative_analysis, recommended."},
                    {"role": "user", "content": context_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=OPENAI_TEMPERATURE
            )

            import json
            data = json.loads(completion.choices[0].message.content)
            match = PalateMatch(**data)
            print(f"✓ Match score: {match.match_score}/100")

            return features, match

        except Exception as e:
            print(f"⚠️  Error predicting match: {e}")
            # Return conservative default
            return features, PalateMatch(
                match_score=50.0,
                qualitative_analysis=f"Error during analysis: {str(e)}",
                key_alignment="Unable to analyze",
                key_concerns="API error occurred",
                recommendation="Unable to Evaluate"
            )

    def calculate_likelihood(self, wine_features: dict, wine_color: str, wine_style: dict = None) -> Tuple[int, str, str]:
        """
        Calculate likelihood to like based on top 3 liked wines of same color.
        HARDENED: Forces category-based matching and style-based inference.

        Args:
            wine_features: Dict with acidity, minerality, fruitiness, tannin, body
            wine_color: Wine color (White, Red, Rosé, Orange)
            wine_style: Optional dict with region, grape, sweetness for inference

        Returns:
            Tuple of (likelihood_score 0-100, verdict, region_context)
        """
        feature_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']

        # STEP 1: STYLE-BASED INFERENCE (if features missing)
        # Never use 5/10 defaults - infer from wine style
        wine_vec = []
        for col in feature_cols:
            val = wine_features.get(col, 0)
            if val == 0 and wine_style:
                # Infer from style
                val = self._infer_feature_from_style(col, wine_color, wine_style)
            wine_vec.append(val)

        # If still have zeros after inference, calculate from color baseline
        if any(v == 0 for v in wine_vec):
            # Use average of this color as baseline
            color_wines = self.df[self.df['wine_color'] == wine_color]
            if len(color_wines) > 0:
                for i, col in enumerate(feature_cols):
                    if wine_vec[i] == 0:
                        wine_vec[i] = int(color_wines[col].mean())

        # STEP 2: FORCED CATEGORY MATCH - Get baseline from same color
        same_color_liked = self.df[
            (self.df['liked'] == True) &
            (self.df['wine_color'] == wine_color)
        ]

        # TIER 1: Same color liked wines
        if len(same_color_liked) > 0:
            baseline_source = f"{wine_color} wines"
            wine_color_fallback = False
        else:
            # TIER 2: All liked wines (forced fallback)
            same_color_liked = self.df[self.df['liked'] == True]

            if len(same_color_liked) == 0:
                # TIER 3: ALL wines of same color (even disliked)
                same_color_liked = self.df[self.df['wine_color'] == wine_color]

                if len(same_color_liked) == 0:
                    # TIER 4: Truly no history - use generic baseline
                    return 50, "🔍 First Wine", "This will establish your baseline"

                baseline_source = f"all {wine_color} wines in database"
                wine_color_fallback = True
            else:
                baseline_source = "all wine styles"
                wine_color_fallback = True

        # Calculate palate score and get top 3 (or all if less than 3)
        same_color_liked = same_color_liked.copy()
        same_color_liked['structure_score'] = (
            same_color_liked['acidity'] + same_color_liked['minerality']
        )
        same_color_liked['acidity_body_ratio'] = (
            same_color_liked['acidity'] / (same_color_liked['body'] + 0.1)
        )
        same_color_liked['palate_score'] = (
            same_color_liked['structure_score'] +
            (same_color_liked['acidity_body_ratio'] * 2)
        )

        # Get top wines (up to 3)
        n_wines = min(3, len(same_color_liked))
        top_wines = same_color_liked.nlargest(n_wines, 'palate_score')

        # STEP 3: Calculate similarity
        similarities = []
        for _, ref_wine in top_wines.iterrows():
            ref_vec = [ref_wine[col] for col in feature_cols]
            # Euclidean distance
            distance = sum((a - b) ** 2 for a, b in zip(wine_vec, ref_vec)) ** 0.5
            similarities.append(distance)

        avg_distance = sum(similarities) / len(similarities)

        # Convert distance to 0-100% score (lower distance = higher score)
        # Max distance is ~22.4 (sqrt(5 * 10^2)), normalize to 0-1, invert
        normalized = avg_distance / 22.4
        likelihood = int((1 - normalized) * 100)
        likelihood = max(0, min(100, likelihood))

        # Generate verdict
        if likelihood >= 80:
            verdict = "💙 Strong Match"
        elif likelihood >= 60:
            verdict = "🧡 Worth Trying"
        else:
            verdict = "🟡 Different Style"

        # Get region context from top wine
        top_region = top_wines.iloc[0]['region'] if 'region' in top_wines.columns else "Unknown"

        # Build context message
        if wine_color_fallback:
            region_context = f"Compared to {baseline_source}"
        else:
            if top_region != "Unknown":
                region_context = f"Matches your {top_region} profile"
            else:
                region_context = f"Based on {len(top_wines)} {wine_color} wines"

        return likelihood, verdict, region_context

    def _infer_feature_from_style(self, feature: str, wine_color: str, wine_style: dict) -> int:
        """
        Infer feature value from wine style/region/grape.
        NEVER returns 5 - always makes an educated guess.

        Args:
            feature: Feature name (acidity, minerality, etc.)
            wine_color: Wine color
            wine_style: Dict with region, grape, sweetness, etc.

        Returns:
            Inferred value 1-10
        """
        region = wine_style.get('region', '').lower()
        grape = wine_style.get('grape', '').lower()
        sweetness = wine_style.get('sweetness', 'Dry').lower()

        # REGION-BASED INFERENCE
        # Atlantic/Coastal regions: HIGH acidity + minerality
        atlantic_regions = ['rías baixas', 'albariño', 'galicia', 'loire', 'chablis', 'txakoli']
        if any(r in region for r in atlantic_regions):
            if feature == 'acidity':
                return 9
            elif feature == 'minerality':
                return 9
            elif feature == 'fruitiness':
                return 7
            elif feature == 'tannin':
                return 1 if wine_color == 'White' else 3
            elif feature == 'body':
                return 5

        # Mediterranean regions: MEDIUM acidity, HIGHER body
        mediterranean = ['rioja', 'ribera', 'tuscany', 'priorat', 'barolo']
        if any(r in region for r in mediterranean):
            if feature == 'acidity':
                return 6
            elif feature == 'minerality':
                return 5
            elif feature == 'fruitiness':
                return 7
            elif feature == 'tannin':
                return 8 if wine_color == 'Red' else 2
            elif feature == 'body':
                return 8

        # Sherry/Fortified: HIGH acidity or specific profiles
        if 'jerez' in region or 'sherry' in region or 'manzanilla' in region:
            if feature == 'acidity':
                return 8
            elif feature == 'minerality':
                return 9
            elif feature == 'fruitiness':
                return 3
            elif feature == 'tannin':
                return 1
            elif feature == 'body':
                return 3

        # GRAPE-BASED INFERENCE
        if 'albariño' in grape or 'albariño' in region:
            if feature == 'acidity':
                return 8
            elif feature == 'minerality':
                return 8
            elif feature == 'fruitiness':
                return 7
            elif feature == 'tannin':
                return 1
            elif feature == 'body':
                return 5

        if 'riesling' in grape:
            if feature == 'acidity':
                return 9
            elif feature == 'minerality':
                return 8
            elif feature == 'fruitiness':
                return 8
            elif feature == 'tannin':
                return 1
            elif feature == 'body':
                return 4

        if 'tempranillo' in grape or 'tempranillo' in region:
            if feature == 'acidity':
                return 6
            elif feature == 'minerality':
                return 5
            elif feature == 'fruitiness':
                return 7
            elif feature == 'tannin':
                return 7
            elif feature == 'body':
                return 8

        # COLOR-BASED DEFAULTS (last resort)
        if wine_color == 'White':
            defaults = {'acidity': 7, 'minerality': 6, 'fruitiness': 7, 'tannin': 1, 'body': 5}
        elif wine_color == 'Red':
            defaults = {'acidity': 6, 'minerality': 5, 'fruitiness': 7, 'tannin': 6, 'body': 7}
        elif wine_color == 'Rosé':
            defaults = {'acidity': 7, 'minerality': 5, 'fruitiness': 8, 'tannin': 2, 'body': 5}
        else:  # Orange
            defaults = {'acidity': 7, 'minerality': 6, 'fruitiness': 6, 'tannin': 4, 'body': 6}

        return defaults.get(feature, 6)  # Never 5

    def calculate_palate_score(
        self,
        wine_features: dict,
        wine_color: str
    ) -> PalateScore:
        """
        Calculate dual-metric palate score using PalateEngine

        This method provides:
        - Metric A (Palate Match): Cosine similarity for flavor alignment
        - Metric B (Likelihood Score): Bayesian-adjusted confidence score

        Args:
            wine_features: Dict with acidity, fruitiness, body, tannin, minerality
            wine_color: Wine color for color-specific matching

        Returns:
            PalateScore with both metrics and detailed explanation

        Example:
            >>> predictor = VinoPredictor()
            >>> features = {'acidity': 8, 'fruitiness': 7, 'body': 5, 'tannin': 1, 'minerality': 9}
            >>> score = predictor.calculate_palate_score(features, 'White')
            >>> print(f"Alignment: {score.palate_match}%, Likelihood: {score.likelihood_score}%")
        """
        # Initialize PalateEngine with current history
        engine = PalateEngine(self.df)

        # Calculate dual-metric score
        score = engine.calculate_match(wine_features, wine_color)

        return score
