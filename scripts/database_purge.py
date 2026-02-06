#!/usr/bin/env python3
"""
Total Database Purge and Cleanup

1. Remove wines with 'Unknown' or empty wine_name
2. Remove duplicates by wine_name
3. Fix Martín Códax 2022 specifically
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

project_root = Path(__file__).parent.parent
history_path = project_root / "data" / "history.csv"

print("=== TOTAL DATABASE PURGE ===\n")

# Load data
df = pd.read_csv(history_path)
print(f"Starting with: {len(df)} wines\n")

# Create backup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_path = project_root / "data" / f"history.csv.backup_purge_{timestamp}"
df.to_csv(backup_path, index=False)
print(f"✓ Backup saved to {backup_path.name}\n")

# STEP 1: Remove wines with 'Unknown' or empty wine_name
print("=== STEP 1: Remove Zombie Wines ===")
before_count = len(df)

# Check for zombies
zombie_mask = (
    df['wine_name'].isna() |
    (df['wine_name'] == '') |
    (df['wine_name'] == 'Unknown') |
    (df['wine_name'].astype(str).str.lower() == 'nan')
)

zombies = df[zombie_mask]
if len(zombies) > 0:
    print(f"Found {len(zombies)} zombie wines:")
    for idx, row in zombies.iterrows():
        wine = str(row['wine_name'])[:40]
        print(f"  - Row {idx}: '{wine}'")

    # Remove zombies
    df = df[~zombie_mask].copy()
    print(f"✓ Removed {before_count - len(df)} zombie wines\n")
else:
    print("✓ No zombie wines found\n")

# STEP 2: Remove duplicates
print("=== STEP 2: Remove Duplicates ===")
before_count = len(df)

# Find duplicates
duplicates = df[df.duplicated(subset=['wine_name'], keep='first')]
if len(duplicates) > 0:
    print(f"Found {len(duplicates)} duplicate wines:")
    for idx, row in duplicates.iterrows():
        wine = str(row['wine_name'])[:40]
        print(f"  - '{wine}'")

    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['wine_name'], keep='first')
    print(f"✓ Removed {before_count - len(df)} duplicates\n")
else:
    print("✓ No duplicates found\n")

# STEP 3: Fix Martín Códax 2022 specifically
print("=== STEP 3: Fix Martín Códax 2022 ===")

# Find Martín Códax row
martin_mask = df['wine_name'].str.contains('Martín Códax', case=False, na=False)
if martin_mask.any():
    martin_idx = df[martin_mask].index[0]

    print(f"Found at row {martin_idx}")
    print("Current values:")
    print(f"  Country: {df.at[martin_idx, 'country']}")
    print(f"  Region: {df.at[martin_idx, 'region']}")
    print(f"  Wine Color: {df.at[martin_idx, 'wine_color']}")
    print(f"  Sweetness: {df.at[martin_idx, 'sweetness']}")

    # Update values
    df.at[martin_idx, 'country'] = 'Spain'
    df.at[martin_idx, 'region'] = 'Rías Baixas'
    df.at[martin_idx, 'wine_color'] = 'White'
    df.at[martin_idx, 'sweetness'] = 'Dry'

    print("\nUpdated to:")
    print(f"  ✓ Country: Spain")
    print(f"  ✓ Region: Rías Baixas")
    print(f"  ✓ Wine Color: White")
    print(f"  ✓ Sweetness: Dry\n")
else:
    print("⚠️  Martín Códax 2022 not found in database\n")

# Save cleaned data
df.to_csv(history_path, index=False)
print(f"=== DATABASE PURGED ===")
print(f"Final count: {len(df)} wines")
print(f"Saved to: {history_path.name}\n")

# Print cleaned database for verification
print("=== CLEANED DATABASE ===")
print(f"{'#':<3} {'Wine Name':<40} {'Region':<20} {'Country':<10}")
print("-" * 80)
for idx, row in df.iterrows():
    wine = str(row['wine_name'])[:38]
    region = str(row['region'])[:18] if pd.notna(row['region']) else 'Unknown'
    country = str(row['country'])[:8] if pd.notna(row['country']) else 'Unknown'
    print(f"{idx+1:<3} {wine:<40} {region:<20} {country:<10}")

print(f"\n✅ Total: {len(df)} clean wines\n")
