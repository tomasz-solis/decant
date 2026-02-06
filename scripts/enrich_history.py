#!/usr/bin/env python3
"""
History Enrichment Loop

Scans history.csv for wines with missing flavor features (acidity, minerality,
fruitiness, tannin, body = 0) and uses AI to research and populate complete profiles.

Also fills in any missing Country/Region data simultaneously.
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

console = Console()
load_dotenv()


def extract_wine_profile(wine_name, producer, notes, client):
    """
    Use AI to extract complete wine profile including flavor features and geography.

    Args:
        wine_name: Wine name
        producer: Producer/winery name
        notes: Tasting notes (may be empty)
        client: OpenAI client

    Returns:
        Dictionary with all extracted fields
    """
    prompt = f"""Research this wine and provide a complete HIGH-DIMENSIONAL profile using your encyclopedic wine knowledge.

WINE NAME: {wine_name}
PRODUCER: {producer}
NOTES: {notes if notes and notes != 'nan' else 'Not provided'}

MANDATORY: Extract ALL fields. Use your wine knowledge to infer standard profiles for this wine style.

## WINE ORIGIN (MANDATORY - NEVER LEAVE BLANK)

COUNTRY: [Country of origin - REQUIRED]
- Use producer location, wine style, or grape variety knowledge
- Examples: Spain, France, Italy, USA, Germany, Portugal, Austria, Chile
- NEVER return "Unknown" - research the producer

REGION: [Specific wine region/appellation - REQUIRED]
- Look for DO, DOCa, AOC, AVA indicators in wine name
- Examples:
  * Albariño → Rías Baixas
  * Tempranillo Ribera → Ribera del Duero
  * Rioja label → Rioja
  * Bordeaux château → Bordeaux
  * Barolo → Piedmont
- Use producer's primary region if specific DO not clear
- NEVER return "Unknown"

## HIGH-DIMENSIONAL ATTRIBUTES

WINE_COLOR: [MUST be: "White", "Red", "Rosé", or "Orange"]
- Infer from grape variety, wine name, or style
- Examples: Albariño → White, Tempranillo → Red, Garnacha → Red/Rosé

IS_SPARKLING: [true or false]
- true if Champagne, Cava, Prosecco, Espumante, or "Brut" in name
- false for still wines

IS_NATURAL: [true or false]
- true if notes mention "organic", "bio", "natural", "biodynamic", "ECOLÓGICO"
- false otherwise

SWEETNESS: [MUST be: "Dry", "Medium-Dry", "Medium-Sweet", or "Sweet"]
- Infer from wine style:
  * Most table wines → "Dry"
  * Albariño/Verdejo/Sauvignon Blanc → "Dry"
  * German Kabinett → "Medium-Sweet"
  * Sauternes/Moscato/Ice Wine → "Sweet"
  * Champagne Brut/Cava Brut → "Dry"

## CORE 5 FLAVOR FEATURES (1-10 scale)

Rate each feature based on typical profile for this wine style:

ACIDITY: [1-10]
- High acidity (8-10): Albariño, Riesling, Sauvignon Blanc, Champagne
- Medium acidity (5-7): Chardonnay, Rioja, Pinot Noir
- Low acidity (1-4): Viognier, Merlot

MINERALITY: [1-10]
- High minerality (8-10): Albariño, Chablis, Riesling, Atlantic wines
- Medium minerality (5-7): Burgundy, Loire wines
- Low minerality (1-4): New World wines, oaked styles

FRUITINESS: [1-10]
- High fruit (8-10): New World wines, Garnacha, fruit-forward styles
- Medium fruit (5-7): Bordeaux, Rioja Reserva
- Low fruit (1-4): Aged wines, mineral-driven styles

TANNIN: [1-10 - whites typically 1-3, reds vary]
- High tannin (8-10): Barolo, young Bordeaux, Ribera del Duero
- Medium tannin (5-7): Rioja, Pinot Noir, Chianti
- Low tannin (1-4): Beaujolais, Garnacha, white wines

BODY: [1-10]
- Full body (8-10): Oaked wines, Tempranillo, Cabernet, Chardonnay Reserve
- Medium body (5-7): Rioja, Pinot Noir, unoaked whites
- Light body (1-4): Albariño, Pinot Grigio, Beaujolais

Use standard profiles for each wine region/style. Be precise and research-based."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use full model for accuracy
            messages=[
                {"role": "system", "content": "You are a master sommelier with encyclopedic knowledge of wines worldwide. You provide precise, research-based wine profiles using standard characteristics for each wine region and style."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )

        content = response.choices[0].message.content

        # Parse response
        import re

        def extract_field(text, field_name):
            match = re.search(f'{field_name}:\\s*(.+?)(?:\\n|$)', text, re.IGNORECASE)
            return match.group(1).strip() if match else None

        def extract_number(text, field_name):
            match = re.search(f'{field_name}:\\s*(\\d+)', text, re.IGNORECASE)
            return int(match.group(1)) if match else None

        def extract_boolean(text, field_name):
            match = re.search(f'{field_name}:\\s*(true|false)', text, re.IGNORECASE)
            return match.group(1).lower() == 'true' if match else False

        # Extract all fields
        result = {
            'country': extract_field(content, 'COUNTRY') or 'Unknown',
            'region': extract_field(content, 'REGION') or 'Unknown',
            'wine_color': extract_field(content, 'WINE_COLOR') or 'White',
            'is_sparkling': extract_boolean(content, 'IS_SPARKLING'),
            'is_natural': extract_boolean(content, 'IS_NATURAL'),
            'sweetness': extract_field(content, 'SWEETNESS') or 'Dry',
            'acidity': extract_number(content, 'ACIDITY') or 5,
            'minerality': extract_number(content, 'MINERALITY') or 5,
            'fruitiness': extract_number(content, 'FRUITINESS') or 5,
            'tannin': extract_number(content, 'TANNIN') or 3,
            'body': extract_number(content, 'BODY') or 5
        }

        return result

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return None


def enrich_history():
    """Main enrichment function."""
    project_root = Path(__file__).parent.parent
    history_path = project_root / "data" / "history.csv"

    console.print("\n[bold]🔬 History Enrichment Loop[/bold]\n")

    if not history_path.exists():
        console.print(f"[red]✗ history.csv not found at {history_path}[/red]")
        return

    # Load data
    df = pd.read_csv(history_path)
    console.print(f"[dim]Loaded {len(df)} wines from history.csv[/dim]\n")

    # Identify wines needing enrichment (missing flavor features)
    feature_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']

    needs_enrichment = df[
        df[feature_cols].isna().any(axis=1) |
        (df[feature_cols] == 0).all(axis=1)
    ]

    if len(needs_enrichment) == 0:
        console.print("[green]✓ All wines already have complete profiles![/green]\n")
        return

    console.print(f"[yellow]Found {len(needs_enrichment)} wines needing enrichment[/yellow]")
    console.print(f"[dim]Will extract: flavor features + geography[/dim]\n")

    # Initialize OpenAI client
    client = OpenAI()

    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = project_root / "data" / f"history.csv.backup_{timestamp}"
    df.to_csv(backup_path, index=False)
    console.print(f"[dim]✓ Backup saved to {backup_path.name}[/dim]\n")

    # Process each wine
    enriched_count = 0
    verification_examples = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Enriching wines...", total=len(needs_enrichment))

        for idx in needs_enrichment.index:
            wine_name = str(df.at[idx, 'wine_name'])
            producer = str(df.at[idx, 'producer']) if pd.notna(df.at[idx, 'producer']) else 'Unknown'
            notes = str(df.at[idx, 'notes']) if pd.notna(df.at[idx, 'notes']) else ''

            progress.update(task, description=f"Processing: {wine_name[:40]}...")

            # Extract complete profile
            profile = extract_wine_profile(wine_name, producer, notes, client)

            if profile:
                # Update all fields
                df.at[idx, 'country'] = profile['country']
                df.at[idx, 'region'] = profile['region']
                df.at[idx, 'wine_color'] = profile['wine_color']
                df.at[idx, 'is_sparkling'] = profile['is_sparkling']
                df.at[idx, 'is_natural'] = profile['is_natural']
                df.at[idx, 'sweetness'] = profile['sweetness']
                df.at[idx, 'acidity'] = profile['acidity']
                df.at[idx, 'minerality'] = profile['minerality']
                df.at[idx, 'fruitiness'] = profile['fruitiness']
                df.at[idx, 'tannin'] = profile['tannin']
                df.at[idx, 'body'] = profile['body']

                enriched_count += 1

                # Store first 2 for verification
                if len(verification_examples) < 2:
                    verification_examples.append({
                        'wine_name': wine_name,
                        'profile': profile
                    })

                # Display progress
                console.print(f"  [cyan]{wine_name[:50]}[/cyan]")
                console.print(f"    📍 {profile['region']}, {profile['country']}")
                console.print(f"    🎯 Acidity: {profile['acidity']}/10, Minerality: {profile['minerality']}/10, Fruit: {profile['fruitiness']}/10")
                console.print(f"    🎯 Tannin: {profile['tannin']}/10, Body: {profile['body']}/10")

            progress.advance(task)

    # Save enriched data
    df.to_csv(history_path, index=False)

    console.print(f"\n[green]✓ Enrichment complete![/green]")
    console.print(f"[green]✓ Updated {enriched_count} wines with complete profiles[/green]")
    console.print(f"[green]✓ Saved to {history_path.name}[/green]\n")

    # VERIFICATION: Show at least 2 wines with non-zero values
    if verification_examples:
        console.print("[bold]🔍 Verification - Sample Enriched Wines:[/bold]\n")
        for i, example in enumerate(verification_examples, 1):
            console.print(f"[bold cyan]{i}. {example['wine_name']}[/bold cyan]")
            console.print(f"   Location: {example['profile']['region']}, {example['profile']['country']}")
            console.print(f"   Color: {example['profile']['wine_color']} | Sweetness: {example['profile']['sweetness']}")
            console.print(f"   [green]Acidity: {example['profile']['acidity']}/10[/green]")
            console.print(f"   [green]Minerality: {example['profile']['minerality']}/10[/green]")
            console.print(f"   [green]Fruitiness: {example['profile']['fruitiness']}/10[/green]")
            console.print(f"   [green]Tannin: {example['profile']['tannin']}/10[/green]")
            console.print(f"   [green]Body: {example['profile']['body']}/10[/green]")
            console.print()

        console.print("[bold green]✅ Radar charts will now display with complete data![/bold green]\n")

    # Show enrichment summary
    feature_stats = df[feature_cols].describe()
    console.print("[bold]📊 Feature Statistics After Enrichment:[/bold]")
    console.print(f"  Acidity: avg {feature_stats.loc['mean', 'acidity']:.1f}/10")
    console.print(f"  Minerality: avg {feature_stats.loc['mean', 'minerality']:.1f}/10")
    console.print(f"  Fruitiness: avg {feature_stats.loc['mean', 'fruitiness']:.1f}/10")
    console.print(f"  Tannin: avg {feature_stats.loc['mean', 'tannin']:.1f}/10")
    console.print(f"  Body: avg {feature_stats.loc['mean', 'body']:.1f}/10\n")


if __name__ == "__main__":
    enrich_history()
