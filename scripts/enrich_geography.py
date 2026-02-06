#!/usr/bin/env python3
"""
Batch Geography Enrichment Script

Automatically fills Country and Region for wines in history.csv where these fields are empty.
Uses AI to infer geography from wine_name and producer.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
from openai import OpenAI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decant.schema import WineExtraction

console = Console()
load_dotenv()


def enrich_single_wine(wine_name, producer, client):
    """
    Use AI to extract country and region from wine name and producer.

    Args:
        wine_name: Wine name
        producer: Producer/winery name
        client: OpenAI client

    Returns:
        Tuple of (country, region)
    """
    prompt = f"""Extract the COUNTRY and REGION for this wine using your encyclopedic wine knowledge.

WINE NAME: {wine_name}
PRODUCER: {producer}

MANDATORY: You MUST provide both Country and Region. NEVER return "Unknown".

Examples:
- Fefiñanes Albariño 2022 → Spain, Rías Baixas
- Château Margaux 2015 → France, Bordeaux
- Gaja Barbaresco 2018 → Italy, Piedmont
- Ridge Monte Bello 2019 → USA, Santa Cruz Mountains

Extract:
COUNTRY: [Country of origin - REQUIRED]
- Use producer location or wine style knowledge
- NEVER return "Unknown"

REGION: [Specific wine region/DO/AOC - REQUIRED]
- Look for appellations like DO, DOCa, AOC, AVA
- Use producer's primary region if specific DO not clear
- NEVER return "Unknown"

Spanish examples: Rías Baixas, Ribera del Duero, Rioja, Priorat, Bierzo
French examples: Bordeaux, Burgundy, Champagne, Loire Valley, Rhône Valley
Italian examples: Tuscany, Piedmont, Veneto, Sicily
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use full model for accuracy
            messages=[
                {"role": "system", "content": "You are a wine geography expert with encyclopedic knowledge of wine regions, DOs, AOCs, and producers worldwide."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )

        content = response.choices[0].message.content

        # Parse response
        import re
        country_match = re.search(r'COUNTRY:\s*(.+)', content, re.IGNORECASE)
        region_match = re.search(r'REGION:\s*(.+)', content, re.IGNORECASE)

        country = country_match.group(1).strip() if country_match else "Unknown"
        region = region_match.group(1).strip() if region_match else "Unknown"

        return country, region

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return "Unknown", "Unknown"


def enrich_geography():
    """Main enrichment function."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    console.print("\n[bold]🌍 Batch Geography Enrichment[/bold]\n")

    if not history_path.exists():
        console.print(f"[red]✗ history.csv not found at {history_path}[/red]")
        return

    # Load data
    df = pd.read_csv(history_path)
    console.print(f"[dim]Loaded {len(df)} wines from history.csv[/dim]\n")

    # Check if country/region columns exist
    if 'country' not in df.columns:
        df['country'] = 'Unknown'
    if 'region' not in df.columns:
        df['region'] = 'Unknown'

    # Find wines needing enrichment
    needs_enrichment = df[
        (df['country'].isna()) |
        (df['country'] == 'Unknown') |
        (df['country'] == '') |
        (df['region'].isna()) |
        (df['region'] == 'Unknown') |
        (df['region'] == '')
    ]

    if len(needs_enrichment) == 0:
        console.print("[green]✓ All wines already have geography data![/green]\n")
        return

    console.print(f"[yellow]Found {len(needs_enrichment)} wines needing geography enrichment[/yellow]\n")

    # Initialize OpenAI client
    client = OpenAI()

    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = project_root / "data" / f"history.csv.backup_{timestamp}"
    df.to_csv(backup_path, index=False)
    console.print(f"[dim]✓ Backup saved to {backup_path.name}[/dim]\n")

    # Process each wine
    enriched_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Enriching wines...", total=len(needs_enrichment))

        for idx in needs_enrichment.index:
            wine_name = str(df.at[idx, 'wine_name'])
            producer = str(df.at[idx, 'producer'])

            progress.update(task, description=f"Processing: {wine_name[:30]}...")

            # Extract geography
            country, region = enrich_single_wine(wine_name, producer, client)

            # Update dataframe
            df.at[idx, 'country'] = country
            df.at[idx, 'region'] = region

            enriched_count += 1

            console.print(f"  [cyan]{wine_name[:40]}[/cyan]")
            console.print(f"    → {region}, {country}")

            progress.advance(task)

    # Save enriched data
    df.to_csv(history_path, index=False)

    console.print(f"\n[green]✓ Enrichment complete![/green]")
    console.print(f"[green]✓ Updated {enriched_count} wines with geography data[/green]")
    console.print(f"[green]✓ Saved to {history_path.name}[/green]\n")

    # Show top regions
    regional_wines = df[(df['region'] != 'Unknown') & (df['region'].notna())]
    if len(regional_wines) > 0:
        console.print("[bold]🏆 Your Wine Regions:[/bold]")
        region_counts = regional_wines['region'].value_counts().head(5)
        for region, count in region_counts.items():
            console.print(f"  {region}: {count} wines")
        console.print()


if __name__ == "__main__":
    enrich_geography()
