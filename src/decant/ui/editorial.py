"""Editorial HTML helpers for the Streamlit UI.

Streamlit's native widgets are useful for inputs, but the read-only
surfaces look much better when rendered as small, controlled HTML
blocks. These helpers keep that markup escaped and consistent.
"""

from __future__ import annotations

from html import escape
from typing import Mapping, Sequence

import pandas as pd
import streamlit as st


_FEATURES = (
    ("acidity", "Acidity"),
    ("minerality", "Minerality"),
    ("fruitiness", "Fruit"),
    ("tannin", "Tannin"),
    ("body", "Body"),
)


def html_text(value: object, fallback: str = "Unknown", max_chars: int | None = None) -> str:
    """Return HTML-escaped display text with null-ish values handled."""
    if value is None:
        text = fallback
    elif isinstance(value, float) and pd.isna(value):
        text = fallback
    else:
        text = str(value).strip() or fallback

    if text.lower() == "nan":
        text = fallback

    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "..."

    return escape(text)


def tone_class_for_wine_color(wine_color: object) -> str:
    """Return a stable CSS tone class for wine colour names."""
    text = str(wine_color or "").strip().lower()
    if text.startswith("red"):
        return "tone-red"
    if text.startswith("white"):
        return "tone-white"
    if text.startswith("orange"):
        return "tone-orange"
    if text.startswith("ros"):
        return "tone-rose"
    return "tone-neutral"


def format_score(value: object, suffix: str = "") -> str:
    """Format a score-ish number with one decimal place."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0.0" + suffix
    if pd.isna(number):
        return "0.0" + suffix
    return f"{number:.1f}{suffix}"


def format_price_eur(value: object) -> str:
    """Format a EUR price as HTML-safe display text."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if pd.isna(number):
        number = 0.0
    return f"&euro;{number:,.0f}"


def render_tab_heading(title: str, kicker: str, deck: str | None = None) -> None:
    """Render a compact editorial heading for a tab."""
    deck_html = f"<p>{html_text(deck)}</p>" if deck else ""
    st.markdown(
        (
            '<section class="tab-heading">'
            f"<span>{html_text(kicker)}</span>"
            f"<h2>{html_text(title)}</h2>"
            f"{deck_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_stat_grid(
    metrics: Sequence[tuple[str, str, str]],
    class_name: str = "editorial-stat-grid",
) -> None:
    """Render a responsive grid of stat tiles."""
    cards = []
    for label, value, note in metrics:
        cards.append(
            (
                '<article class="stat-tile">'
                f'<div class="stat-label">{html_text(label)}</div>'
                f'<div class="stat-value">{html_text(value)}</div>'
                f'<div class="stat-note">{html_text(note)}</div>'
                "</article>"
            )
        )

    # The column track follows the tile count (see .stat-grid-cols-N in
    # styles.py) so a 3-tile grid doesn't keep a 4th empty slot.
    column_class = f"stat-grid-cols-{len(cards)}"
    st.markdown(
        f"<div class=\"{escape(class_name)} {column_class}\">"
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_cellar_snapshot(history_df: pd.DataFrame) -> None:
    """Render the high-level cellar snapshot below the masthead."""
    if history_df is None or history_df.empty:
        return

    total = len(history_df)

    if "liked" in history_df.columns:
        liked = int(history_df["liked"].fillna(False).astype(bool).sum())
    else:
        liked = 0

    if "score" in history_df.columns:
        avg_score = pd.to_numeric(history_df["score"], errors="coerce").mean()
    else:
        avg_score = float("nan")

    if "region" in history_df.columns:
        region_text = history_df["region"].astype(str).str.strip()
        regions = history_df.loc[
            history_df["region"].notna()
            & (region_text != "")
            & (region_text.str.lower() != "unknown"),
            "region",
        ].nunique()
    else:
        regions = 0

    if "country" in history_df.columns:
        country_text = history_df["country"].astype(str).str.strip()
        countries = history_df.loc[
            history_df["country"].notna()
            & (country_text != "")
            & (country_text.str.lower() != "unknown"),
            "country",
        ].nunique()
    else:
        countries = 0

    hit_rate = f"{liked / total:.0%} hit rate" if total else "rated favorites"
    metrics = (
        ("Wines", f"{total:,}", "in the cellar"),
        ("Liked", f"{liked:,}", hit_rate),
        ("Average", format_score(avg_score), "score / 10"),
        ("Map", f"{regions:,}", f"regions, {countries:,} countries"),
    )
    render_stat_grid(metrics, class_name="cellar-snapshot")


def render_feature_profile(
    title: str,
    values: Mapping[str, object] | pd.Series,
    note: str | None = None,
    tone: object = None,
) -> None:
    """Render a single five-feature palate profile."""
    st.markdown(
        _feature_profile_html(title=title, values=values, note=note, tone=tone),
        unsafe_allow_html=True,
    )


def render_feature_profiles(
    profiles: Mapping[str, Mapping[str, object] | pd.Series],
    counts: Mapping[str, int] | None = None,
) -> None:
    """Render all wine-colour profiles in a magazine-style grid."""
    cards = []
    for wine_color, values in profiles.items():
        count = (counts or {}).get(wine_color)
        note = f"{count} liked wine{'s' if count != 1 else ''}" if count else None
        cards.append(
            _feature_profile_html(
                title=f"{wine_color} wines",
                values=values,
                note=note,
                tone=wine_color,
            )
        )

    if not cards:
        return

    st.markdown(
        f'<div class="feature-profile-grid feature-profile-grid-{len(cards)}">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def _feature_profile_html(
    title: str,
    values: Mapping[str, object] | pd.Series,
    note: str | None,
    tone: object,
) -> str:
    rows = []
    for key, label in _FEATURES:
        raw = values.get(key, 0) if hasattr(values, "get") else 0
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = 0.0
        if pd.isna(number):
            number = 0.0
        number = max(0.0, min(number, 10.0))
        width = max(4.0, number * 10.0)
        rows.append(
            (
                '<div class="feature-row">'
                f'<span class="feature-name">{html_text(label)}</span>'
                '<span class="feature-track">'
                f'<span class="feature-fill" style="width: {width:.1f}%;"></span>'
                "</span>"
                f'<span class="feature-number">{number:.1f}</span>'
                "</div>"
            )
        )

    note_html = f"<p>{html_text(note)}</p>" if note else ""
    tone_class = tone_class_for_wine_color(tone)
    return (
        f'<article class="feature-profile {tone_class}">'
        '<div class="feature-profile-head">'
        f"<h3>{html_text(title)}</h3>"
        f"{note_html}"
        "</div>"
        f'<div class="feature-rows">{"".join(rows)}</div>'
        "</article>"
    )


def render_ranked_list(
    rows: Sequence[tuple[str, str, str, str]],
    class_name: str = "ranked-list",
) -> None:
    """Render ranked editorial rows.

    Each row is (title, meta, value, value_label).
    """
    items = []
    for index, (title, meta, value, value_label) in enumerate(rows, start=1):
        # The unit label is optional - pass "" to show just the score.
        label_html = (
            f"<small>{html_text(value_label)}</small>" if value_label else ""
        )
        items.append(
            (
                '<li class="ranked-item">'
                f'<div class="ranked-index">{index:02d}</div>'
                '<div class="ranked-copy">'
                f'<div class="ranked-title">{html_text(title)}</div>'
                f'<div class="ranked-meta">{html_text(meta)}</div>'
                "</div>"
                '<div class="ranked-value">'
                f"<span>{html_text(value)}</span>"
                f"{label_html}"
                "</div>"
                "</li>"
            )
        )

    st.markdown(
        f"<ol class=\"{escape(class_name)}\">" + "".join(items) + "</ol>",
        unsafe_allow_html=True,
    )


def render_gallery_result_count(count: int) -> None:
    """Render the gallery result count without a bulky heading."""
    noun = "wine" if count == 1 else "wines"
    st.markdown(
        (
            '<div class="gallery-result-line">'
            f"<span>{count:,}</span>"
            f"<p>{noun} in view</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
