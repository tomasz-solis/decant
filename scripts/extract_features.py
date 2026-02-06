#!/usr/bin/env python3
"""
Feature extraction script for Decant.

Reads wine tasting data from history.json, uses OpenAI to extract numerical
features from tasting notes, and outputs a CSV with features + liked labels.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decant.schema import TastingHistory, Wine, WineFeatures


def load_tasting_history(file_path: Path) -> TastingHistory:
    """Load and validate tasting history from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    try:
        history = TastingHistory(**data)
        print(f"✓ Loaded {len(history.comparisons)} comparisons")
        return history
    except ValidationError as e:
        print(f"✗ Validation error in {file_path}:")
        print(e)
        sys.exit(1)


def extract_features_with_llm(wine: Wine, client: OpenAI) -> WineFeatures:
    """
    Use OpenAI to extract numerical features from tasting notes.

    Uses structured outputs with Pydantic schema for reliable validation.
    """
    # Combine all tasting notes into a comprehensive description
    tasting_text = f"""
    Wine: {wine.producer} {wine.vintage}

    Appearance: {wine.tasting_notes.appearance}
    Nose: {wine.tasting_notes.nose}
    Palate: {wine.tasting_notes.palate}
    Overall: {wine.tasting_notes.overall}
    """

    prompt = f"""
    Analyze the following wine tasting notes and extract numerical features.
    Rate each feature on a scale of 1-10:

    - Acidity: How acidic/crisp is the wine? (1=low, 10=very high)
    - Minerality: Mineral/saline/stony character (1=none, 10=very high)
    - Fruitiness: Fruit intensity and presence (1=subtle, 10=very fruity)
    - Tannin: Tannin structure (1=soft/none, 10=firm/grippy)
    - Body: Weight and texture (1=light, 10=full-bodied)

    {tasting_text}

    Extract these features based on the descriptors in the notes. Look for keywords:
    - Acidity: "crisp", "fresh", "zesty", "tart", "bright"
    - Minerality: "mineral", "saline", "sea", "stony", "flinty", "chalky"
    - Fruitiness: fruit names, "fruity", "ripe", "juicy"
    - Tannin: "tannic", "grippy", "structured", "firm", "velvety", "smooth"
    - Body: "light", "medium", "full", "weight", "texture"
    """

    try:
        # Use structured outputs with Pydantic schema
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are a wine expert analyzing tasting notes to extract numerical features."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format=WineFeatures,
            temperature=0.3  # Lower temperature for more consistent extractions
        )

        features = completion.choices[0].message.parsed
        print(f"  ✓ Extracted features for {wine.producer}")
        return features

    except Exception as e:
        print(f"  ✗ Error extracting features for {wine.producer}: {e}")
        # Return default mid-range values on error
        return WineFeatures(
            acidity=5,
            minerality=5,
            fruitiness=5,
            tannin=5,
            body=5,
            reasoning=f"Error during extraction: {str(e)}"
        )


def create_feature_dataframe(history: TastingHistory, client: OpenAI) -> pd.DataFrame:
    """
    Extract features for all wines and create a DataFrame.

    Returns:
        DataFrame with columns: wine_id, producer, vintage, price_usd,
                                acidity, minerality, fruitiness, tannin, body, liked
    """
    rows = []

    for comparison in history.comparisons:
        print(f"\nProcessing comparison: {comparison.comparison_id}")

        for wine in comparison.wines:
            # Extract features using LLM
            features = extract_features_with_llm(wine, client)

            # Create row with wine metadata + features + liked label
            row = {
                "wine_id": wine.wine_id,
                "producer": wine.producer,
                "vintage": wine.vintage,
                "price_usd": wine.price_usd,
                "acidity": features.acidity,
                "minerality": features.minerality,
                "fruitiness": features.fruitiness,
                "tannin": features.tannin,
                "body": features.body,
                "liked": wine.liked if wine.liked is not None else False,
                "extraction_reasoning": features.reasoning or ""
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n✓ Created feature matrix with {len(df)} wines")
    return df


def main():
    """Main execution function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "history.json"
    output_file = project_root / "data" / "processed" / "wine_features.csv"

    # Load environment variables from .env file
    load_dotenv(project_root / ".env")

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("✗ Error: OPENAI_API_KEY environment variable not set")
        print("  Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized")

    # Load tasting history
    print(f"\nLoading data from {input_file}...")
    history = load_tasting_history(input_file)

    # Extract features
    print("\nExtracting features with OpenAI...")
    df = create_feature_dataframe(history, client)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n✓ Features saved to {output_file}")

    # Display summary statistics
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    print(df[["acidity", "minerality", "fruitiness", "tannin", "body"]].describe())
    print(f"\nLiked distribution:")
    print(df["liked"].value_counts())
    print("="*60)


if __name__ == "__main__":
    main()
