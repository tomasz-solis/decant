#!/usr/bin/env python3
"""
Test geography extraction with a single known wine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

load_dotenv()
console = Console()

def test_single_extraction():
    """Test extraction with a known wine."""

    client = OpenAI()

    # Test with a well-known wine
    wine_name = "Fefiñanes Albariño 2022"
    producer = "Fefiñanes"

    console.print(f"\n[bold]Testing Geography Extraction[/bold]\n")
    console.print(f"Wine: {wine_name}")
    console.print(f"Producer: {producer}\n")

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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a wine geography expert with encyclopedic knowledge of wine regions, DOs, AOCs, and producers worldwide."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1
        )

        content = response.choices[0].message.content

        console.print("[bold cyan]Raw API Response:[/bold cyan]")
        console.print(content)
        console.print()

        # Parse response
        import re
        country_match = re.search(r'COUNTRY:\s*(.+)', content, re.IGNORECASE)
        region_match = re.search(r'REGION:\s*(.+)', content, re.IGNORECASE)

        console.print(f"[bold]Country match:[/bold] {country_match}")
        console.print(f"[bold]Region match:[/bold] {region_match}")
        console.print()

        country = country_match.group(1).strip() if country_match else "Unknown"
        region = region_match.group(1).strip() if region_match else "Unknown"

        console.print(f"[green]✓ Extracted Country:[/green] {country}")
        console.print(f"[green]✓ Extracted Region:[/green] {region}\n")

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")

if __name__ == "__main__":
    test_single_extraction()
