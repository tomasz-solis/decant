"""OpenAI Vision call: extract wine attributes from a photo.

This module owns the prompt and response parsing for the
"upload a wine label, get structured features back" flow that powers
Tab 1 (Add Wine).

Streamlit coupling: this module calls `st.warning`, `st.error`,
`st.info` directly to surface partial-failure and validation issues
inline as the extraction progresses. That's a small smell - services
shouldn't touch the UI layer - but it preserves the original
behaviour during the Phase 3 refactor. A future pass should refactor
to return a result + warnings list and have the caller render them.

The OpenAI `client` is passed in by the caller rather than constructed
here, so the app's single client instance is reused (matches the old
behaviour where the client was a module-global in `app.py`).
"""

from __future__ import annotations

import base64
import json

import streamlit as st
from openai import OpenAI

from decant.config import OPENAI_MODEL, OPENAI_TEMPERATURE
from decant.services.data_access import normalize as _ensure_wine_df


def extract_complete_wine_data(image_file, history_df, client: OpenAI):
    """Extract wine data from a label photo via OpenAI Vision.

    Args:
        image_file: Streamlit-uploaded file (has .name and .read()).
        history_df: Recent history, used to build a few-shot context
            block in the prompt.
        client: OpenAI client instance. Caller owns the lifecycle so
            the same client is reused across the app.

    Returns:
        dict with the wine fields the app stores, or None on failure.
        Surfaces warnings/errors via Streamlit directly.
    """
    try:
        # Convert image to base64
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_type = "image/jpeg" if image_file.name.lower().endswith('.jpg') or image_file.name.lower().endswith('.jpeg') else "image/png"

        # Build self-learning context from liked wines
        context = ""
        safe_history_df = _ensure_wine_df(history_df)
        if not safe_history_df.empty:
            liked_wines = safe_history_df[safe_history_df['liked'].eq(True)].tail(5)
            if len(liked_wines) > 0:
                context = "\n\nUSER'S TASTE PROFILE (Recent Liked Wines):\n"
                for _, wine in liked_wines.iterrows():
                    context += f"- {wine.get('wine_name', 'Unknown')}: Acidity {wine.get('acidity', 0)}/10, Minerality {wine.get('minerality', 0)}/10\n"

        # Extraction prompt (JSON format)
        prompt = f"""Analyze this wine bottle and extract complete wine information.

{context}

## INSTRUCTIONS:

Use your encyclopedic wine knowledge to infer ALL flavor attributes based on:
1. The specific PRODUCER and their documented house style
2. The REGION'S terroir and typical characteristics
3. The GRAPE VARIETY and its typical profile
4. The VINTAGE (if known) and its conditions

DO NOT use generic defaults like 5.0. Each wine is unique - use your training data about this specific producer, region, and style.

General regional tendencies (but always adjust for the specific producer):
- Atlantic regions (Galicia, Loire, Chablis): Tend toward higher acidity + minerality
- Mediterranean regions (Rioja, Tuscany, Rhône): Tend toward more body, moderate acidity
- High-altitude wines: Often have higher acidity, more elegance
- Warm climate wines: Often have riper fruit, fuller body

Remember: Producers within the same region can be VERY different. Use your knowledge of the specific producer.

Return JSON with these exact fields:

{{
  "wine_name": "Full name with vintage",
  "producer": "Winery name",
  "vintage": 2021,
  "notes": "Professional tasting notes based on this producer's style",
  "score": 7.5,
  "price": 15.0,
  "country": "Spain",
  "region": "Bierzo",
  "wine_color": "Red",
  "is_sparkling": false,
  "is_natural": false,
  "sweetness": "Dry",
  "acidity": 7.5,
  "minerality": 7.0,
  "fruitiness": 8.0,
  "tannin": 6.5,
  "body": 6.5
}}

CONSTRAINTS:
- Whites typically have tannin 1-3, reds 5-9
- wine_color: MUST be "White", "Red", "Rosé", or "Orange"
- sweetness: MUST be "Dry", "Medium-Dry", "Medium-Sweet", or "Sweet"
- Use ACTUAL wine knowledge, not formula-based inference

Return ONLY valid JSON."""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "system",
                "content": "You are a wine expert with encyclopedic knowledge. Return JSON only."
            }, {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
                ]
            }],
            response_format={"type": "json_object"},
            max_completion_tokens=800,
            temperature=OPENAI_TEMPERATURE
        )

        content = response.choices[0].message.content

        # Parse JSON response
        raw_data = json.loads(content)

        # Ensure liked field exists (user will set)
        raw_data['liked'] = None

        # Map price_eur to price if needed
        if 'price_eur' in raw_data and 'price' not in raw_data:
            raw_data['price'] = raw_data['price_eur']

        # Ensure all required fields exist with reasonable fallbacks
        if 'price' not in raw_data:
            raw_data['price'] = 0.0

        # Validate extracted data before returning
        from pydantic import ValidationError
        from decant.constants import WineColor, Sweetness

        # Validate critical fields
        validation_errors = []

        # Validate wine_name
        if not raw_data.get('wine_name') or len(raw_data['wine_name']) < 1:
            validation_errors.append("wine_name is empty or too short")

        # Validate producer
        if not raw_data.get('producer') or len(raw_data['producer']) < 1:
            validation_errors.append("producer is empty or too short")

        # Validate vintage - the LLM legitimately returns None for non-vintage
        # wines (NV champagne, sparkling, some natural wines). Treat None as
        # "no year known" rather than an error.
        vintage = raw_data.get('vintage')
        if vintage is None:
            raw_data['vintage'] = 0
        elif vintage < 1900 or vintage > 2100:
            st.warning(f"Invalid vintage {vintage}, setting to 0")
            raw_data['vintage'] = 0

        # Validate features are in range. None values are coerced to mid-scale
        # (5.0) with a warning, since the predictor expects numeric values.
        for feature in ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']:
            value = raw_data.get(feature)
            if value is None:
                st.warning(f"{feature} not extracted, defaulting to 5.0")
                raw_data[feature] = 5.0
            elif value < 1.0 or value > 10.0:
                st.warning(f"{feature} value {value} out of range [1-10], clamping")
                raw_data[feature] = max(1.0, min(10.0, value))

        # Validate score. None defaults to 5.0 (mid-scale, no opinion).
        score = raw_data.get('score')
        if score is None:
            raw_data['score'] = 5.0
        elif score < 1.0 or score > 10.0:
            st.warning(f"Score {score} out of range [1-10], clamping")
            raw_data['score'] = max(1.0, min(10.0, score))

        # Validate wine_color
        valid_colors = [c.value for c in WineColor]
        if raw_data.get('wine_color') not in valid_colors:
            st.warning(f"Invalid wine color '{raw_data.get('wine_color')}', defaulting to 'White'")
            raw_data['wine_color'] = 'White'

        # Validate sweetness
        valid_sweetness = [s.value for s in Sweetness]
        if raw_data.get('sweetness') not in valid_sweetness:
            st.warning(f"Invalid sweetness '{raw_data.get('sweetness')}', defaulting to 'Dry'")
            raw_data['sweetness'] = 'Dry'

        # If critical validation errors, return None
        if validation_errors:
            st.error(f"Critical validation errors: {', '.join(validation_errors)}")
            st.info("Please verify the image and try again, or enter data manually.")
            return None

        return raw_data

    except json.JSONDecodeError as je:
        st.error(f"LLM returned invalid JSON: {je}")
        st.info("Please try again or enter data manually.")
        return None
    except ValidationError as ve:
        st.error(f"Validation error: {ve}")
        st.info("Please try again or enter data manually.")
        return None
    except Exception as e:
        st.error(f"Error extracting wine data: {str(e)}")
        st.info("Please try again or enter data manually.")
        return None
