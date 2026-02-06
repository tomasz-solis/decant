#!/usr/bin/env python3
"""
Migrate history.csv to include Country and Region columns.

Adds:
- country (string)
- region (string)

Preserves all existing data.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console

console = Console()


def migrate_geography():
    """Add country and region columns to history.csv."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    console.print("\n[bold]🌍 Geography Migration[/bold]\n")

    if not history_path.exists():
        console.print(f"[red]✗ history.csv not found at {history_path}[/red]")
        console.print("[yellow]Creating empty history.csv with geography schema[/yellow]")

        # Create empty CSV with geography schema
        columns = [
            'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
            'country', 'region',
            'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
            'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
        ]

        df = pd.DataFrame(columns=columns)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(history_path, index=False)

        console.print(f"[green]✓ Created empty history.csv with GEOGRAPHY schema (18 columns)[/green]\n")
        return

    # Load existing data
    df = pd.read_csv(history_path)
    console.print(f"[dim]Loaded {len(df)} wines from history.csv[/dim]")
    console.print(f"[dim]Current columns: {', '.join(df.columns.tolist())}[/dim]\n")

    # Check if migration needed
    has_country = 'country' in df.columns
    has_region = 'region' in df.columns

    if has_country and has_region:
        console.print("[green]✓ Geography columns already exist - no migration needed[/green]\n")
        return

    console.print(f"[yellow]Adding geography columns: country, region[/yellow]\n")

    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = project_root / "data" / f"history.csv.backup_{timestamp}"
    df.to_csv(backup_path, index=False)
    console.print(f"[dim]✓ Backup saved to {backup_path.name}[/dim]")

    # Add new columns if they don't exist
    if 'country' not in df.columns:
        df['country'] = 'Unknown'
        console.print("[dim]✓ Added country column (default: 'Unknown')[/dim]")

    if 'region' not in df.columns:
        df['region'] = 'Unknown'
        console.print("[dim]✓ Added region column (default: 'Unknown')[/dim]")

    # Ensure column order
    desired_order = [
        'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
        'country', 'region',
        'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
        'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
    ]

    # Reorder columns (keep only columns that exist)
    df_ordered = df[[col for col in desired_order if col in df.columns]]

    # Save migrated data
    df_ordered.to_csv(history_path, index=False)

    console.print(f"\n[green]✓ Migration complete![/green]")
    console.print(f"[green]✓ {len(df_ordered)} wines saved with GEOGRAPHY schema (18 columns)[/green]")
    console.print(f"\n[bold]Schema:[/bold]")
    console.print(f"  Basic: wine_name, producer, vintage, notes, score, liked, price")
    console.print(f"  [cyan]Geography: country, region[/cyan]")
    console.print(f"  Dimensions: wine_color, is_sparkling, is_natural, sweetness")
    console.print(f"  Features: acidity, minerality, fruitiness, tannin, body")
    console.print("\n[bold yellow]📌 Next Step:[/bold yellow] Run enrich_geography.py to fill in country/region data\n")


if __name__ == "__main__":
    migrate_geography()
