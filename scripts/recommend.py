#!/usr/bin/env python3
"""
Wine recommendation script using In-Context Learning.

Analyzes wine tasting notes and predicts palate compatibility based on
historical preferences using few-shot learning.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from decant.predictor import VinoPredictor


def create_features_table(features, match_score):
    """Create a beautiful table showing wine features."""
    table = Table(
        title="Wine Wine Feature Analysis",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold white"
    )

    table.add_column("Feature", style="cyan", width=15)
    table.add_column("Score", justify="center", style="bold white", width=8)
    table.add_column("Bar", width=30)
    table.add_column("Assessment", style="dim white", width=20)

    # Define feature colors and assessments
    features_data = [
        ("Acidity", features.acidity, "", "High" if features.acidity >= 8 else "Moderate" if features.acidity >= 5 else "Low"),
        ("Minerality", features.minerality, "⬜", "High" if features.minerality >= 8 else "Moderate" if features.minerality >= 5 else "Low"),
        ("Fruitiness", features.fruitiness, "", "High" if features.fruitiness >= 8 else "Moderate" if features.fruitiness >= 5 else "Low"),
        ("Tannin", features.tannin, "", "High" if features.tannin >= 8 else "Moderate" if features.tannin >= 5 else "Low"),
        ("Body", features.body, "", "Full" if features.body >= 8 else "Medium" if features.body >= 5 else "Light"),
    ]

    for feature_name, score, icon, assessment in features_data:
        # Create visual bar
        filled = int(score)
        empty = 10 - filled
        bar = icon * filled + "" * empty

        # Color code the score
        if score >= 8:
            score_style = "bold green"
        elif score >= 5:
            score_style = "bold yellow"
        else:
            score_style = "bold red"

        table.add_row(
            feature_name,
            f"[{score_style}]{score}/10[/{score_style}]",
            bar,
            assessment
        )

    # Add derived scores (Palate Formula)
    table.add_section()
    structure_score = features.acidity + features.minerality
    acidity_body_ratio = features.acidity / (features.body + 0.1)
    palate_score = structure_score + (acidity_body_ratio * 2)

    table.add_row(
        "[bold]Structure Score[/bold]",
        f"[bold magenta]{structure_score}/20[/bold magenta]",
        "" * int(structure_score / 2),
        "Acid + Mineral"
    )

    table.add_row(
        "[bold]Acidity/Body[/bold]",
        f"[bold cyan]{acidity_body_ratio:.2f}[/bold cyan]",
        "" * min(int(acidity_body_ratio), 10),
        "Crispness Index"
    )

    table.add_row(
        "[bold]Palate Score[/bold]",
        f"[bold white]{palate_score:.1f}[/bold white]",
        "" * min(int(palate_score / 3), 10),
        "Overall Match"
    )

    return table


def create_verdict_panel(match, features):
    """Create a verdict panel with recommendation using consistent branding.

    Blue = Strong Buy (75+)
    Gold = Worth Trying (50-74)
    Yellow = Skip (<50)
    """
    # Determine color based on match score with consistent branding
    if match.match_score >= 75:
        color = "blue"
        emoji = ""
        border_style = "bold blue"
    elif match.match_score >= 50:
        color = "yellow"
        emoji = "Warning"
        border_style = "bold yellow"
    else:
        color = "dim yellow"
        emoji = "Warning:"
        border_style = "dim yellow"

    # Build verdict content
    content = f"""
[bold {color}]{emoji} {match.recommendation}[/bold {color}]

[bold white]Match Score:[/bold white] [{color}]{match.match_score:.0f}/100[/{color}]

[bold cyan]Qualitative Analysis:[/bold cyan]
{match.qualitative_analysis}

[bold green]OK Key Alignment:[/bold green]
{match.key_alignment}

[bold red]Warning: Key Concerns:[/bold red]
{match.key_concerns}
"""

    panel = Panel(
        content.strip(),
        title="Target Palate Match Verdict",
        border_style=border_style,
        box=box.DOUBLE,
        padding=(1, 2)
    )

    return panel


def main():
    """Main execution function."""
    console = Console()

    # Print header
    console.print()
    console.print(Panel.fit(
        "[bold white]Wine Decant Wine Recommender[/bold white]\n"
        "[dim]Powered by In-Context Learning & OpenAI[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Check if tasting notes provided
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red] python scripts/recommend.py \"<tasting notes>\"")
        console.print()
        console.print("[bold cyan]Example:[/bold cyan]")
        console.print('  python scripts/recommend.py "Bright acidity, citrus, mineral notes, crisp finish"')
        console.print()
        sys.exit(1)

    tasting_notes = " ".join(sys.argv[1:])

    # Show input
    console.print(Panel(
        f"[bold white]Tasting Notes:[/bold white]\n{tasting_notes}",
        title=" Input",
        border_style="dim white"
    ))
    console.print()

    try:
        # Initialize predictor
        with console.status("[bold cyan]Loading wine preference model...", spinner="dots"):
            predictor = VinoPredictor()

        console.print("[green]OK[/green] Model loaded successfully\n")

        # Get prediction
        with console.status("[bold cyan]Analyzing wine with In-Context Learning...", spinner="dots"):
            features, match = predictor.predict_match(tasting_notes)

        console.print("[green]OK[/green] Analysis complete\n")

        # Display results
        console.print(create_features_table(features, match.match_score))
        console.print()
        console.print(create_verdict_panel(match, features))
        console.print()

        # Add context info
        console.print(
            f"[dim]Analysis based on {len(predictor.liked_examples)} liked "
            f"and {len(predictor.disliked_examples)} disliked wines from your history[/dim]"
        )
        console.print()

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("\n[yellow]Make sure OPENAI_API_KEY is set in your .env file[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
