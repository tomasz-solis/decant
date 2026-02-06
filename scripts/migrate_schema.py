#!/usr/bin/env python3
"""
Migrate history.csv to simplified schema.

This script:
1. Backs up existing history.csv
2. Removes extra columns (complexity, finish, wine_type, region, comparison_date)
3. Keeps only core 12 columns
4. Validates data types (liked=bool, price=float, score=float)
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console

console = Console()


def migrate_schema():
    """Migrate history.csv to simplified core schema."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    console.print("\n[bold]📦 Migrating history.csv to Simplified Schema[/bold]\n")

    if not history_path.exists():
        console.print(f"[red]✗ history.csv not found at {history_path}[/red]")
        console.print("[yellow]Creating empty history.csv with core schema[/yellow]")

        # Create empty CSV with core columns
        core_columns = ['wine_name', 'producer', 'vintage', 'notes', 'score', 'liked',
                        'price', 'acidity', 'minerality', 'fruitiness', 'tannin', 'body']

        df = pd.DataFrame(columns=core_columns)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(history_path, index=False)

        console.print(f"[green]✓ Created empty history.csv with core schema[/green]\n")
        return

    # Load existing data
    df = pd.read_csv(history_path)
    console.print(f"[dim]Loaded {len(df)} wines from history.csv[/dim]")
    console.print(f"[dim]Current columns: {', '.join(df.columns.tolist())}[/dim]\n")

    # Core columns we want to keep
    core_columns = ['wine_name', 'producer', 'vintage', 'notes', 'score', 'liked',
                    'price', 'acidity', 'minerality', 'fruitiness', 'tannin', 'body']

    # Check if migration is needed
    extra_columns = [col for col in df.columns if col not in core_columns]

    if not extra_columns:
        console.print("[green]✓ Schema already simplified - no migration needed[/green]\n")
        return

    console.print(f"[yellow]Found extra columns to remove: {', '.join(extra_columns)}[/yellow]\n")

    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = project_root / "data" / f"history.csv.backup_{timestamp}"
    df.to_csv(backup_path, index=False)
    console.print(f"[dim]✓ Backup saved to {backup_path.name}[/dim]")

    # Handle price column name variations
    if 'price' not in df.columns:
        if 'price_eur' in df.columns:
            df['price'] = df['price_eur']
            console.print("[dim]✓ Renamed price_eur to price[/dim]")
        elif 'price_usd' in df.columns:
            df['price'] = df['price_usd']
            console.print("[dim]✓ Renamed price_usd to price[/dim]")
        else:
            df['price'] = 0.0
            console.print("[yellow]⚠ No price column found - setting to 0.0[/yellow]")

    # Select only core columns
    df_simplified = df[[col for col in core_columns if col in df.columns]]

    # Add missing columns with defaults
    for col in core_columns:
        if col not in df_simplified.columns:
            if col == 'liked':
                df_simplified[col] = True
            elif col in ['price', 'score']:
                df_simplified[col] = 0.0
            elif col in ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']:
                df_simplified[col] = 5
            else:
                df_simplified[col] = ''

    # Validate data types
    console.print("\n[dim]Validating data types...[/dim]")

    try:
        df_simplified['liked'] = df_simplified['liked'].astype(bool)
        df_simplified['price'] = df_simplified['price'].astype(float)
        df_simplified['score'] = df_simplified['score'].astype(float)
        df_simplified['vintage'] = df_simplified['vintage'].astype(int)

        # Ensure feature columns are integers
        for col in ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']:
            df_simplified[col] = df_simplified[col].fillna(5).astype(int)

        console.print("[green]✓ Data types validated[/green]")

    except Exception as e:
        console.print(f"[red]✗ Error validating data types: {e}[/red]")
        console.print("[yellow]Check your data for invalid values[/yellow]")
        return

    # Save simplified schema
    df_simplified.to_csv(history_path, index=False)

    console.print(f"\n[green]✓ Migration complete![/green]")
    console.print(f"[green]✓ Removed {len(extra_columns)} extra columns[/green]")
    console.print(f"[green]✓ {len(df_simplified)} wines saved with core 12-column schema[/green]")
    console.print(f"\n[bold]Core columns:[/bold] {', '.join(core_columns)}\n")


if __name__ == "__main__":
    migrate_schema()
