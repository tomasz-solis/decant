#!/usr/bin/env python3
"""
Sync wine_features.csv with history.csv

This script updates wine_features.csv to match the liked/disliked status
and other metadata from history.csv without re-extracting features via API.

Use this when you've manually edited history.csv and want to sync the files
without spending API credits.
"""

import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()


def sync_features():
    """
    Sync wine_features.csv with history.csv data.

    Uses simplified schema with core 12 columns.
    Creates wine_features.csv from history.csv if it doesn't exist.
    """
    project_root = Path(__file__).parent.parent

    history_path = project_root / "data" / "history.csv"
    features_path = project_root / "data" / "processed" / "wine_features.csv"

    console.print("\n[bold]🔄 Syncing Features with History[/bold]\n")

    # Check history exists
    if not history_path.exists():
        console.print(f"[red]✗ history.csv not found at {history_path}[/red]")
        return

    # Load history
    history_df = pd.read_csv(history_path)
    console.print(f"[dim]Loaded {len(history_df)} wines from history.csv[/dim]")

    # FULL schema columns (18 total: 16 + country + region)
    full_schema_columns = [
        'wine_name', 'producer', 'vintage', 'notes', 'score', 'liked', 'price',
        'country', 'region',
        'wine_color', 'is_sparkling', 'is_natural', 'sweetness',
        'acidity', 'minerality', 'fruitiness', 'tannin', 'body'
    ]

    # Keep only full schema columns
    history_df = history_df[[col for col in full_schema_columns if col in history_df.columns]]

    # If features doesn't exist, create it from history
    if not features_path.exists():
        console.print(f"[yellow]Creating wine_features.csv from history.csv[/yellow]")
        features_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(features_path, index=False)
        console.print(f"[green]✓ Created wine_features.csv with {len(history_df)} wines[/green]\n")
        return

    # Load existing features
    features_df = pd.read_csv(features_path)
    console.print(f"[dim]Loaded {len(features_df)} wines from wine_features.csv[/dim]\n")

    # Completely replace features with history (history is source of truth)
    console.print("[yellow]Replacing wine_features.csv with latest history.csv data[/yellow]")

    # Create backup
    backup_path = features_path.with_suffix('.csv.backup')
    features_df.to_csv(backup_path, index=False)
    console.print(f"[dim]Backup saved to {backup_path.name}[/dim]")

    # Save history as features (with HIGH-DIMENSIONAL validation)
    history_df['liked'] = history_df['liked'].astype(bool)
    history_df['price'] = history_df['price'].astype(float)
    history_df['score'] = history_df['score'].astype(float)

    # Validate high-dimensional boolean fields
    if 'is_sparkling' in history_df.columns:
        history_df['is_sparkling'] = history_df['is_sparkling'].astype(bool)
    if 'is_natural' in history_df.columns:
        history_df['is_natural'] = history_df['is_natural'].astype(bool)

    history_df.to_csv(features_path, index=False)

    console.print(f"\n[green]✓ wine_features.csv updated with {len(history_df)} wines[/green]")
    console.print(f"[green]✓ FULL schema synced (18 columns)[/green]")
    console.print(f"[cyan]✓ Includes: country, region, wine_color, is_sparkling, is_natural, sweetness[/cyan]\n")


if __name__ == "__main__":
    sync_features()
