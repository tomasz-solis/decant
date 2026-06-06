"""Infer a wine's flavour profile from its name and region.

Used for text-entered wines, where the user types a name (e.g.
"Fefiñanes Albariño 2022") and we have no label photo to read the
profile from. A single LLM call estimates the five flavour features.

This must run exactly once per wine, at extraction time - NOT at
display time. Running it inside a Streamlit render path means it
re-fires on every rerun, and because the OpenAI API is not truly
deterministic (even at temperature 0 with a fixed seed, the `seed`
parameter is best-effort), each call returns slightly different
numbers. That made the palate score visibly unstable: the same wine
scored 83.0% on one render and 84.9% on the next. Inferring once and
freezing the result in the wine record fixes that at the source.

The function is deliberately free of Streamlit state. It takes a
client and returns data; the caller decides what to store.
"""

from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI

from decant.config import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_SEED


_FEATURE_KEYS = ("acidity", "fruitiness", "body", "tannin", "minerality")

_INFERENCE_PROMPT = """Role: You are a Master Sommelier and Data Engineer specializing in quantitative viticulture.

Task: Provide a precise, technical flavor profile for the wine: {wine_name} from {region}.

Objective: Your output will be used to calculate a vector-space similarity model. Consistency in your scoring logic is mandatory.

Scoring Guidelines (Scale 1.0 - 10.0):
- Acidity: 1.0 (Flat/Flabby) to 10.0 (High Tartaric/Piercing)
- Fruitiness: 1.0 (Earth-driven/Savory) to 10.0 (Primary Fruit Bomb/Jammy)
- Body: 1.0 (Light/Watery) to 10.0 (Full/Viscous/Heavy)
- Tannin: 1.0 (No structure/Silk) to 10.0 (Aggressive/Gripping/Astringent)
- Minerality: 1.0 (Clean/Fruit-only) to 10.0 (Stony/Saline/Chalky)

Requirements:
1. Use your internal knowledge of this specific producer, vintage, and regional style.
2. Avoid "safe" middle-ground scores (like 5.0) unless truly warranted.
3. Provide the output ONLY as a JSON object for programmatic parsing.

Desired JSON Structure:
{{
  "wine_metadata": {{
    "name": "{wine_name}",
    "region": "{region}",
    "style": "Regional style description"
  }},
  "technical_profile": {{
    "acidity": float,
    "fruitiness": float,
    "body": float,
    "tannin": float,
    "minerality": float
  }},
  "sommelier_verdict": "One sentence technical summary of the structure."
}}"""


def infer_features_from_text(
    wine_name: str,
    region: str,
    client: OpenAI,
) -> Optional[dict[str, float]]:
    """Estimate the five flavour features for a named wine.

    Args:
        wine_name: The wine name as entered by the user.
        region: Region string, or "Unknown" if not known.
        client: An OpenAI client.

    Returns:
        A dict with keys acidity, fruitiness, body, tannin, minerality
        (floats, rounded to one decimal place so sub-decimal LLM jitter
        can't perturb the downstream score), or None if the call failed
        or returned an unusable structure. The caller is expected to
        fall back to manual entry when None is returned.

    Notes:
        Features are rounded to 1 dp on purpose. Even with a successful
        call, two inferences of the same wine can differ in the second
        decimal; rounding removes that noise from the stored value. The
        real fix for instability is calling this once (see module
        docstring), but rounding is cheap insurance.
    """
    prompt = _INFERENCE_PROMPT.format(wine_name=wine_name, region=region)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=OPENAI_TEMPERATURE,
            seed=OPENAI_SEED,
        )
        result = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, KeyError, AttributeError, IndexError):
        return None
    except Exception:
        # Network, auth, rate-limit, etc. Caller falls back to manual entry.
        return None

    profile = result.get("technical_profile")
    if not isinstance(profile, dict):
        return None

    features: dict[str, float] = {}
    for key in _FEATURE_KEYS:
        value = profile.get(key)
        if value is None:
            return None
        try:
            features[key] = round(float(value), 1)
        except (TypeError, ValueError):
            return None

    return features
