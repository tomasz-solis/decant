#!/usr/bin/env python3
"""
Deployment Test - Simulate a complete Save operation.

Tests that new wine data can be written correctly with all 18 columns populated.
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

project_root = Path(__file__).parent.parent

print("=== DEPLOYMENT TEST ===\n")
print("Simulating a 'Save New Wine' operation...\n")

# Load current history
history_path = project_root / "data" / "history.csv"
df = pd.read_csv(history_path)

print(f"Current database: {len(df)} wines")
print(f"Columns: {len(df.columns)}\n")

# Simulate a new wine entry with ALL 18 fields
new_wine = {
    'wine_name': 'Test Wine Deployment 2025',
    'producer': 'Test Winery',
    'vintage': 2024,
    'notes': 'Deployment test wine - full profile',
    'score': 8.5,
    'liked': True,
    'price': 25.00,
    'country': 'Spain',
    'region': 'Ribera del Duero',
    'wine_color': 'Red',
    'is_sparkling': False,
    'is_natural': False,
    'sweetness': 'Dry',
    'acidity': 7,
    'minerality': 6,
    'fruitiness': 8,
    'tannin': 8,
    'body': 9
}

print("=== NEW WINE DATA ===")
print(f"Wine: {new_wine['wine_name']}")
print(f"Location: {new_wine['region']}, {new_wine['country']}")
print(f"Color: {new_wine['wine_color']} | Sweetness: {new_wine['sweetness']}")
print(f"Flavor Profile:")
print(f"  Acidity: {new_wine['acidity']}/10")
print(f"  Minerality: {new_wine['minerality']}/10")
print(f"  Fruitiness: {new_wine['fruitiness']}/10")
print(f"  Tannin: {new_wine['tannin']}/10")
print(f"  Body: {new_wine['body']}/10")
print()

# Verify all 18 expected columns are present
expected_cols = [
    'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
    'country', 'region',
    'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
    'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
]

missing_in_new = set(expected_cols) - set(new_wine.keys())
if missing_in_new:
    print(f"❌ FAILURE: Missing columns in new wine data: {missing_in_new}")
    sys.exit(1)

print("✅ All 18 columns present in new wine data")

# Append to dataframe
df_new = pd.DataFrame([new_wine])
df_combined = pd.concat([df, df_new], ignore_index=True)

print(f"✅ Combined dataframe: {len(df_combined)} wines (was {len(df)})")

# Verify no NaN values in critical fields
critical_fields = ['wine_name', 'country', 'region', 'acidity', 'body']
for field in critical_fields:
    if df_combined[field].isna().any():
        print(f"⚠️  Warning: NaN values found in {field}")
    else:
        print(f"✅ No NaN values in {field}")

# Create test save
test_path = project_root / "data" / "history_test_deployment.csv"
df_combined.to_csv(test_path, index=False)

print(f"\n✅ Test save successful: {test_path.name}")

# Verify we can read it back
df_verify = pd.read_csv(test_path)
print(f"✅ Read back verification: {len(df_verify)} wines, {len(df_verify.columns)} columns")

# Check the last row (our test wine)
last_row = df_verify.iloc[-1]
print("\n=== VERIFICATION OF SAVED DATA ===")
print(f"Wine Name: {last_row['wine_name']}")
print(f"Location: {last_row['region']}, {last_row['country']}")
print(f"Acidity: {last_row['acidity']}/10")
print(f"Body: {last_row['body']}/10")

# Verify all values match
errors = 0
for key, expected_value in new_wine.items():
    actual_value = last_row[key]
    if pd.isna(expected_value) and pd.isna(actual_value):
        continue  # Both NaN is OK
    if expected_value != actual_value:
        print(f"❌ Mismatch in {key}: expected {expected_value}, got {actual_value}")
        errors += 1

if errors == 0:
    print("\n✅✅✅ DEPLOYMENT TEST PASSED ✅✅✅")
    print("All 18 columns write and read correctly!")
    print(f"\nTest file saved at: {test_path}")
    print("You can delete this file or keep it for reference.")
else:
    print(f"\n❌ DEPLOYMENT TEST FAILED: {errors} mismatches found")
    sys.exit(1)
