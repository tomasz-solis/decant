#!/usr/bin/env python3
"""
Migrate history.csv to HIGH-DIMENSIONAL taste space schema.

Adds 4 new columns:
- wine_color (White, Red, Rosé, Orange)
- is_sparkling (Boolean)
- is_natural (Boolean)
- sweetness (Dry, Medium-Dry, Medium-Sweet, Sweet)

This is the ULTIMATE zero-friction upgrade for wine logging.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console

console = Console()


def migrate_to_high_dimensional():
    """Migrate to high-dimensional taste space schema."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    console.print("\n[bold]🌈 Migrating to HIGH-DIMENSIONAL Taste Space[/bold]\n")

    if not history_path.exists():
        console.print(f"[red]✗ history.csv not found at {history_path}[/red]")
        console.print("[yellow]Creating empty history.csv with high-dimensional schema[/yellow]")

        # Create empty CSV with high-dimensional schema
        columns = [
            'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
            'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
            'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
        ]

        df = pd.DataFrame(columns=columns)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(history_path, index=False)

        console.print(f"[green]✓ Created empty history.csv with HIGH-DIMENSIONAL schema (16 columns)[/green]\n")
        return

    # Load existing data
    df = pd.read_csv(history_path)
    console.print(f"[dim]Loaded {len(df)} wines from history.csv[/dim]")
    console.print(f"[dim]Current columns: {', '.join(df.columns.tolist())}[/dim]\n")

    # Check if migration needed
    new_columns = ['wine_color', 'is_sparkling', 'is_natural', 'sweetness']
    missing_columns = [col for col in new_columns if col not in df.columns]

    if not missing_columns:
        console.print("[green]✓ Already HIGH-DIMENSIONAL - no migration needed[/green]\n")
        return

    console.print(f"[yellow]Adding {len(missing_columns)} new dimensions: {', '.join(missing_columns)}[/yellow]\n")

    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = project_root / "data" / f"history.csv.backup_{timestamp}"
    df.to_csv(backup_path, index=False)
    console.print(f"[dim]✓ Backup saved to {backup_path.name}[/dim]")

    # Add new columns with intelligent defaults
    if 'wine_color' not in df.columns:
        # Infer from wine_name if possible, default to White
        df['wine_color'] = 'White'
        console.print("[dim]✓ Added wine_color (default: White)[/dim]")

    if 'is_sparkling' not in df.columns:
        df['is_sparkling'] = False
        console.print("[dim]✓ Added is_sparkling (default: False)[/dim]")

    if 'is_natural' not in df.columns:
        df['is_natural'] = False
        console.print("[dim]✓ Added is_natural (default: False)[/dim]")

    if 'sweetness' not in df.columns:
        df['sweetness'] = 'Dry'
        console.print("[dim]✓ Added sweetness (default: Dry)[/dim]")

    # Validate data types
    console.print("\n[dim]Validating data types...[/dim]")

    try:
        # Ensure boolean types
        df['is_sparkling'] = df['is_sparkling'].astype(bool)
        df['is_natural'] = df['is_natural'].astype(bool)
        df['liked'] = df['liked'].astype(bool)

        # Ensure numeric types
        df['price'] = df['price'].astype(float)
        df['score'] = df['score'].astype(float)
        df['vintage'] = df['vintage'].astype(int)

        # Ensure feature columns are integers
        for col in ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']:
            if col in df.columns:
                df[col] = df[col].fillna(5).astype(int)

        console.print("[green]✓ Data types validated[/green]")

    except Exception as e:
        console.print(f"[red]✗ Error validating data types: {e}[/red]")
        console.print("[yellow]Check your data for invalid values[/yellow]")
        return

    # Ensure column order
    desired_order = [
        'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
        'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
        'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
    ]

    # Keep only desired columns in desired order
    df_ordered = df[[col for col in desired_order if col in df.columns]]

    # Save high-dimensional schema
    df_ordered.to_csv(history_path, index=False)

    console.print(f"\n[green]✓ Migration complete![/green]")
    console.print(f"[green]✓ Added {len(missing_columns)} new dimensions[/green]")
    console.print(f"[green]✓ {len(df_ordered)} wines saved with HIGH-DIMENSIONAL schema (16 columns)[/green]")
    console.print(f"\n[bold]HIGH-DIMENSIONAL Columns:[/bold]")
    console.print(f"  Basic: wine_name, producer, vintage, notes, score, liked, price")
    console.print(f"  [cyan]Dimensions: wine_color, is_sparkling, is_natural, sweetness[/cyan]")
    console.print(f"  Features: acidity, minerality, fruitiness, tannin, body")
    console.print("\n[bold cyan]🎉 Welcome to the High-Dimensional Taste Space![/bold cyan]\n")


if __name__ == "__main__":
    migrate_to_high_dimensional()
