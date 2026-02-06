#!/usr/bin/env python3
"""
Migrate CSV data to PostgreSQL database.

This script imports existing wine data from data/history.csv
into the PostgreSQL database on Supabase.

Usage:
    python scripts/migrate_csv_to_postgres.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from decant.database import init_database, add_wine, get_wine_count


def migrate_csv_to_postgres():
    """Migrate CSV data to PostgreSQL."""
    csv_path = Path("data/history.csv")

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("   Nothing to migrate.")
        return

    print("🚀 Starting migration from CSV to PostgreSQL...")

    # Initialize database schema
    print("📋 Initializing database schema...")
    try:
        init_database()
        print("✅ Database schema initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return

    # Check current wine count
    try:
        current_count = get_wine_count()
        print(f"📊 Current wines in database: {current_count}")
    except Exception as e:
        print(f"⚠️  Could not check current wine count: {e}")
        current_count = 0

    # Load CSV
    print(f"📂 Loading CSV from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} wines from CSV")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return

    # Migrate each wine
    success_count = 0
    error_count = 0

    print("\n🔄 Migrating wines...")
    for idx, row in df.iterrows():
        try:
            wine_data = {
                'wine_name': row['wine_name'],
                'producer': row.get('producer', 'Unknown'),
                'vintage': float(row['vintage']) if pd.notna(row.get('vintage')) else None,
                'notes': row.get('notes', ''),
                'score': float(row['score']),
                'liked': bool(row['liked']),
                'price': float(row['price']) if pd.notna(row.get('price')) else None,
                'country': row.get('country', 'Unknown'),
                'region': row.get('region', 'Unknown'),
                'wine_color': row.get('wine_color', 'White'),
                'is_sparkling': bool(row.get('is_sparkling', False)) if pd.notna(row.get('is_sparkling')) else False,
                'is_natural': bool(row.get('is_natural', False)) if pd.notna(row.get('is_natural')) else False,
                'sweetness': row.get('sweetness', 'Dry'),
                'acidity': float(row['acidity']) if pd.notna(row.get('acidity')) else 5.0,
                'minerality': float(row['minerality']) if pd.notna(row.get('minerality')) else 5.0,
                'fruitiness': float(row['fruitiness']) if pd.notna(row.get('fruitiness')) else 5.0,
                'tannin': float(row['tannin']) if pd.notna(row.get('tannin')) else 1.0,
                'body': float(row['body']) if pd.notna(row.get('body')) else 5.0,
            }

            add_wine(wine_data)
            success_count += 1
            print(f"   ✓ {wine_data['wine_name']}")

        except Exception as e:
            error_count += 1
            print(f"   ✗ {row.get('wine_name', 'Unknown')}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"📈 Migration Summary:")
    print(f"   ✅ Successfully migrated: {success_count} wines")
    if error_count > 0:
        print(f"   ❌ Errors: {error_count} wines")

    try:
        final_count = get_wine_count()
        print(f"   📊 Total wines in database: {final_count}")
    except Exception as e:
        print(f"   ⚠️  Could not verify final count: {e}")

    print(f"{'='*60}")
    print("\n🎉 Migration complete!")
    print("\n💡 Next steps:")
    print("   1. Test the app: streamlit run app.py")
    print("   2. Verify your wines appear correctly")
    print("   3. Backup your CSV if migration was successful")


if __name__ == "__main__":
    migrate_csv_to_postgres()
