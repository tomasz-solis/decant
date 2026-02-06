#!/usr/bin/env python3
"""
Wine Database Management Utility

Easily add, remove, and manage wines in your Decant database.
"""

import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()


def backup_files():
    """Create timestamped backups of data files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_root = Path(__file__).parent.parent

    history_path = project_root / "data" / "history.csv"
    features_path = project_root / "data" / "processed" / "wine_features.csv"

    backups = []

    if history_path.exists():
        backup_path = project_root / "data" / f"history.csv.backup_{timestamp}"
        shutil.copy(history_path, backup_path)
        backups.append(backup_path)

    if features_path.exists():
        backup_path = project_root / "data" / "processed" / f"wine_features.csv.backup_{timestamp}"
        shutil.copy(features_path, backup_path)
        backups.append(backup_path)

    return backups


def list_wines(detailed=False):
    """List all wines in the database with row numbers."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    if not history_path.exists():
        console.print("[red]✗ No history.csv found[/red]")
        return

    df = pd.read_csv(history_path)

    console.print("\n[bold]🍷 Wine Database[/bold]")
    console.print(f"Total wines: {len(df)}\n")

    if detailed:
        # Detailed view: Show each wine in a card format
        from rich.panel import Panel

        for idx, row in df.iterrows():
            wine_name = row.get('wine_name', 'Unknown')
            producer = row.get('producer', 'Unknown')
            vintage = row.get('vintage', 'NV')
            liked = "✅ Liked" if row['liked'] else "❌ Disliked"

            # Handle both price column names
            price_col = 'price' if 'price' in row else 'price_usd' if 'price_usd' in row else 'price_eur'
            price = f"€{row.get(price_col, 0):.2f}" if price_col in row else "N/A"

            acidity = int(row.get('acidity', 0))
            minerality = int(row.get('minerality', 0))
            fruitiness = int(row.get('fruitiness', 0))
            tannin = int(row.get('tannin', 0))
            body = int(row.get('body', 0))
            score = row.get('score', 0)
            region = row.get('region', 'Unknown')
            wine_type = row.get('wine_type', 'Unknown')

            details = f"""[cyan]{wine_name}[/cyan]
Producer: {producer} | Vintage: {vintage} | {liked}
Price: {price} | Score: {score:.1f}/10 | Type: {wine_type}
Region: {region}

[bold]Features:[/bold]
  Acidity: {acidity}/10  Minerality: {minerality}/10  Fruitiness: {fruitiness}/10
  Tannin: {tannin}/10  Body: {body}/10"""

            console.print(Panel(details, title=f"[bold]#{idx}[/bold]", border_style="magenta"))
            console.print()

    else:
        # Compact table view
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Wine Name", style="cyan", width=30)
        table.add_column("Producer", width=20)
        table.add_column("Vintage", justify="center", width=8)
        table.add_column("Liked", justify="center", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Acidity", justify="center", width=8)

        for idx, row in df.iterrows():
            wine_name = str(row.get('wine_name', 'Unknown'))[:30]
            producer = str(row.get('producer', 'Unknown'))[:20]
            vintage = str(row.get('vintage', 'NV'))
            liked = "✅" if row['liked'] else "❌"

            # Handle both price column names
            price_col = 'price' if 'price' in row else 'price_usd' if 'price_usd' in row else 'price_eur'
            price = f"€{row.get(price_col, 0):.2f}" if price_col in row else "N/A"

            acidity = f"{row.get('acidity', 0)}" if row.get('acidity', 0) > 0 else "N/A"

            table.add_row(
                str(idx),
                wine_name,
                producer,
                vintage,
                liked,
                price,
                f"{acidity}/10" if acidity != "N/A" else "N/A"
            )

        console.print(table)
        console.print()


def remove_wines(wine_ids, regenerate=True):
    """Remove wines by row numbers."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    if not history_path.exists():
        console.print("[red]✗ No history.csv found[/red]")
        return

    # Parse wine IDs
    if isinstance(wine_ids, str):
        ids_to_remove = [int(x.strip()) for x in wine_ids.split(',')]
    else:
        ids_to_remove = [wine_ids]

    # Load data
    df = pd.read_csv(history_path)

    # Validate IDs
    invalid_ids = [i for i in ids_to_remove if i >= len(df) or i < 0]
    if invalid_ids:
        console.print(f"[red]✗ Invalid row numbers: {invalid_ids}[/red]")
        console.print(f"   Valid range: 0-{len(df)-1}")
        return

    # Show wines to be removed
    console.print("\n[bold yellow]⚠ Wines to be removed:[/bold yellow]\n")
    for idx in ids_to_remove:
        wine = df.iloc[idx]
        wine_name = wine.get('wine_name', 'Unknown')
        console.print(f"  [{idx}] {wine_name}")

    console.print()

    # Confirm
    if not Confirm.ask("Proceed with removal?"):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Backup
    console.print("\n[dim]Creating backups...[/dim]")
    backups = backup_files()
    for backup in backups:
        console.print(f"[dim]  ✓ {backup.name}[/dim]")

    # Remove wines
    df = df.drop(ids_to_remove).reset_index(drop=True)
    df.to_csv(history_path, index=False)

    console.print(f"\n[green]✓ Removed {len(ids_to_remove)} wine(s) from history.csv[/green]")
    console.print(f"  Remaining wines: {len(df)}")

    # Regenerate features
    if regenerate:
        console.print("\n[dim]Regenerating features...[/dim]")
        regenerate_features()
    else:
        console.print("\n[yellow]⚠ Remember to run: python scripts/extract_features.py[/yellow]")


def remove_wines_by_name(name_pattern, regenerate=True):
    """Remove wines matching a name pattern."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    if not history_path.exists():
        console.print("[red]✗ No history.csv found[/red]")
        return

    df = pd.read_csv(history_path)

    # Find matches
    mask = df['wine_name'].str.contains(name_pattern, case=False, na=False)
    matches = df[mask]

    if len(matches) == 0:
        console.print(f"[yellow]No wines found matching '{name_pattern}'[/yellow]")
        return

    # Show matches
    console.print(f"\n[bold yellow]⚠ Found {len(matches)} wine(s) matching '{name_pattern}':[/bold yellow]\n")
    for idx, wine in matches.iterrows():
        wine_name = wine.get('wine_name', 'Unknown')
        console.print(f"  [{idx}] {wine_name}")

    console.print()

    # Confirm
    if not Confirm.ask("Remove these wines?"):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Backup
    console.print("\n[dim]Creating backups...[/dim]")
    backups = backup_files()
    for backup in backups:
        console.print(f"[dim]  ✓ {backup.name}[/dim]")

    # Remove wines
    df = df[~mask].reset_index(drop=True)
    df.to_csv(history_path, index=False)

    console.print(f"\n[green]✓ Removed {len(matches)} wine(s) from history.csv[/green]")
    console.print(f"  Remaining wines: {len(df)}")

    # Regenerate features
    if regenerate:
        console.print("\n[dim]Regenerating features...[/dim]")
        regenerate_features()
    else:
        console.print("\n[yellow]⚠ Remember to run: python scripts/extract_features.py[/yellow]")


def regenerate_features():
    """Run sync script to update wine_features.csv from history.csv."""
    import subprocess

    project_root = Path(__file__).parent.parent
    sync_script = project_root / "scripts" / "sync_features.py"

    if not sync_script.exists():
        console.print("[red]✗ sync_features.py not found[/red]")
        return

    try:
        result = subprocess.run(
            [sys.executable, str(sync_script)],
            cwd=project_root,
            capture_output=False,  # Show output directly
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            console.print("[green]✓ Features synced successfully[/green]")
        else:
            console.print("[red]✗ Feature sync failed[/red]")

    except subprocess.TimeoutExpired:
        console.print("[red]✗ Feature sync timed out[/red]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")


def add_wine_template():
    """Show template for adding wine manually."""
    console.print("\n[bold]📝 CSV Template for Adding Wines[/bold]\n")

    template = """Add this line to data/history.csv (all one line):

wine_name,producer,vintage,notes,score,liked,price,acidity,minerality,fruitiness,tannin,body,complexity,finish,wine_type,region,comparison_date
"Wine Name 2023","Producer Name",2023,"Tasting notes go here",8.0,True,25.50,0,0,0,0,0,7,8,"White Wine","Region Name",2026-02-05

Then run: python scripts/extract_features.py
"""

    console.print(template)


def main():
    parser = argparse.ArgumentParser(
        description="Decant Wine Database Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                          List all wines (compact table)
  %(prog)s --list --detailed               List with all features (card view)
  %(prog)s --remove 3                      Remove wine at row 3
  %(prog)s --remove 3,5,7                  Remove multiple wines
  %(prog)s --remove-name "Rombauer"        Remove wines by name
  %(prog)s --regenerate                    Regenerate features only
  %(prog)s --template                      Show CSV template
        """
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all wines in database'
    )

    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed view with all features (use with --list)'
    )

    parser.add_argument(
        '--remove', '-r',
        type=str,
        metavar='ID',
        help='Remove wine(s) by row number (comma-separated for multiple)'
    )

    parser.add_argument(
        '--remove-name', '-n',
        type=str,
        metavar='NAME',
        help='Remove wines matching name pattern'
    )

    parser.add_argument(
        '--regenerate', '-g',
        action='store_true',
        help='Regenerate features from history.csv'
    )

    parser.add_argument(
        '--template', '-t',
        action='store_true',
        help='Show CSV template for manual addition'
    )

    parser.add_argument(
        '--no-regenerate',
        action='store_true',
        help='Skip automatic feature regeneration after removal'
    )

    args = parser.parse_args()

    # Show header
    console.print("\n[bold magenta]🍷 Decant Wine Database Manager[/bold magenta]\n")

    # Execute commands
    if args.list:
        list_wines(detailed=args.detailed)

    elif args.remove:
        remove_wines(args.remove, regenerate=not args.no_regenerate)

    elif args.remove_name:
        remove_wines_by_name(args.remove_name, regenerate=not args.no_regenerate)

    elif args.regenerate:
        console.print("[dim]Regenerating features from history.csv...[/dim]\n")
        regenerate_features()

    elif args.template:
        add_wine_template()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
