#!/usr/bin/env python3
"""
Migrate wine history from JSON to CSV format.

Converts data/history.json into a flat CSV structure with all wine details
including tasting notes, scores, and extracted features.
"""

import sys
import json
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decant.schema import TastingHistory


def flatten_tasting_notes(tasting_notes_obj):
    """Convert tasting notes object to a single text string."""
    return (
        f"Appearance: {tasting_notes_obj.appearance}. "
        f"Nose: {tasting_notes_obj.nose}. "
        f"Palate: {tasting_notes_obj.palate}. "
        f"Overall: {tasting_notes_obj.overall}"
    )


def migrate_json_to_csv():
    """Convert JSON history to CSV format."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    json_path = project_root / "data" / "history.json"
    csv_path = project_root / "data" / "history.csv"

    print("🔄 Starting migration from JSON to CSV...")
    print(f"   Reading: {json_path}")
    print(f"   Writing: {csv_path}")
    print()

    # Load and validate JSON data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        history = TastingHistory(**data)
        print(f"✓ Loaded {len(history.comparisons)} comparisons")
    except FileNotFoundError:
        print(f"✗ Error: {json_path} not found")
        print("  Make sure you have wine data in data/history.json")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading JSON: {e}")
        sys.exit(1)

    # Extract wines into flat structure
    wines_data = []

    for comparison in history.comparisons:
        for wine in comparison.wines:
            # Combine tasting notes into single text
            notes_text = flatten_tasting_notes(wine.tasting_notes)

            # Create wine name
            wine_name = f"{wine.producer} {wine.vintage}"

            # Extract data
            wine_row = {
                'wine_name': wine_name,
                'producer': wine.producer,
                'vintage': wine.vintage,
                'notes': notes_text,
                'score': wine.scores.overall,
                'liked': wine.liked if wine.liked is not None else False,
                'price': wine.price_usd,
                'acidity': wine.scores.acidity,
                'minerality': 0,  # Not in original scores, will be filled by feature extraction
                'fruitiness': 0,  # Not in original scores, will be filled by feature extraction
                'tannin': 0,      # Not in original scores, will be filled by feature extraction
                'body': wine.scores.body,
                # Additional useful fields
                'complexity': wine.scores.complexity,
                'finish': wine.scores.finish,
                'wine_type': comparison.wine_type,
                'region': comparison.region,
                'comparison_date': str(comparison.date)
            }

            wines_data.append(wine_row)

    # Create DataFrame
    df = pd.DataFrame(wines_data)

    # Ensure column order
    column_order = [
        'wine_name',
        'producer',
        'vintage',
        'notes',
        'score',
        'liked',
        'price',
        'acidity',
        'minerality',
        'fruitiness',
        'tannin',
        'body',
        'complexity',
        'finish',
        'wine_type',
        'region',
        'comparison_date'
    ]

    df = df[column_order]

    # Save to CSV
    df.to_csv(csv_path, index=False)

    print(f"✓ Migrated {len(df)} wines to CSV")
    print()
    print("📊 Summary:")
    print(f"   Total wines: {len(df)}")
    print(f"   Liked: {df['liked'].sum()}")
    print(f"   Disliked: {(~df['liked']).sum()}")
    print()
    print(f"✓ CSV saved to: {csv_path}")
    print()
    print("📝 Note: minerality, fruitiness, and tannin columns are set to 0")
    print("   Run 'python scripts/extract_features.py' to populate these with AI-extracted values")
    print()

    # Display sample
    print("Sample data (first 3 rows):")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df[['wine_name', 'liked', 'price', 'acidity', 'body', 'score']].head(3))
    print("=" * 80)


if __name__ == "__main__":
    migrate_json_to_csv()
