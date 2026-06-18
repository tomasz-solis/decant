"""Plotly chart components used across the app.

All functions in this module are pure: they take inputs (DataFrames,
arrays, dicts) and return Plotly figures. They do not call Streamlit
directly - the caller does the `st.plotly_chart(fig)` rendering.

This separation matters because Streamlit reruns the entire script on
every interaction, and Plotly figure construction is the most expensive
part of those reruns. Keeping these as pure functions makes them
straightforward to cache (`@st.cache_data`) if needed in future.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from decant.constants import UIConstants, WineColor
from decant.services.data_access import normalize as _ensure_wine_df


# Plotly figure theming - single source of truth for chart colours.
# Changing this dict re-themes every chart in the app. Values mirror the
# CSS palette in `ui/styles.py` so charts feel like part of the page.
# Keep these in sync if either side moves.
_THEME = {
    "bg": "#F6F3EC",            # ivory - matches --bg-primary
    "bg_card": "#FFFCF6",       # parchment - matches --card-bg
    "transparent": "rgba(0, 0, 0, 0)",
    "grid": "rgba(33, 26, 22, 0.12)",   # faint ink - radial/angular axes
    "text": "#211A16",          # near-ink - matches --text-primary
    "text_muted": "#7F7568",    # muted brown - matches --text-muted
    "accent": "#8A1F3D",        # bordeaux - primary accent (--terracotta)
    "olive": "#55614B",         # secondary accent for second-line series (--olive)
    "wine": "#7A1730",          # deep wine red for the main palate trace (--wine)
    "wine_fill": "rgba(122, 23, 48, 0.18)",  # translucent wine for fills (--wine-fill)
    "font_family": "Inter, system-ui, -apple-system, sans-serif",
}


_WINE_COLOR_CHART = {
    color.value: spec for color, spec in UIConstants.WINE_COLORS_CHART.items()
}


def _chart_palette_for(wine_color: str) -> dict[str, str]:
    """Return the Plotly palette for a wine colour name."""
    return _WINE_COLOR_CHART.get(wine_color, _WINE_COLOR_CHART[WineColor.WHITE.value])


def _scaled_marker_size(prices: pd.Series) -> pd.Series:
    """Scale wine prices into marker diameters that stay readable."""
    cleaned = pd.to_numeric(prices, errors="coerce").fillna(0).clip(lower=0, upper=120)
    return 10 + (cleaned / 120) * 30


def create_mini_radar_chart(liked_avg):
    """Create a small radar chart for sidebar palate fingerprint."""
    fig = go.Figure()

    categories = ['Acidity', 'Minerality', 'Fruitiness', 'Tannin', 'Body']
    values = liked_avg.tolist() + [liked_avg.iloc[0]]

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor=_THEME["wine_fill"],
        line=dict(color=_THEME["wine"], width=2),
        marker=dict(size=4, color=_THEME["wine"])
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=False,
                gridcolor=_THEME['grid'],
                linecolor=_THEME['grid'],
            ),
            angularaxis=dict(
                tickfont=dict(size=9, family=_THEME['font_family'], color=_THEME['text']),
                gridcolor=_THEME['grid'],
                linecolor=_THEME['grid'],
            )
        ),
        showlegend=False,
        height=200,
        font=dict(family=_THEME['font_family'], color=_THEME['text']),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=_THEME['transparent'],
        plot_bgcolor=_THEME['transparent'],
        hoverlabel=dict(
            bgcolor=_THEME['text'],
            font=dict(family=_THEME['font_family'], color=_THEME['bg_card']),
            bordercolor=_THEME['text'],
        ),
    )

    return fig


def create_decision_boundary_plot(df):
    """Create a 2D scatter plot showing decision boundary (Acidity vs Minerality)."""
    df = _ensure_wine_df(df)
    fig = go.Figure()

    required_cols = ["liked", "acidity", "minerality", "price", "wine_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if df.empty or missing_cols:
        return fig

    # Liked wines
    liked_df = df[df['liked'] == True]
    fig.add_trace(go.Scatter(
        x=liked_df['acidity'],
        y=liked_df['minerality'],
        mode='markers',
        marker=dict(
            size=_scaled_marker_size(liked_df['price']),
            color=_THEME['wine_fill'],
            line=dict(width=1.5, color=_THEME['wine']),
            sizemode='diameter',
            sizemin=8,
            opacity=0.88,
        ),
        name='Liked',
        text=liked_df['wine_name'],
        customdata=liked_df['price'],
        hovertemplate='<b>%{text}</b><br>Acidity: %{x}<br>Minerality: %{y}<br>Price: EUR %{customdata:.0f}<extra></extra>'
    ))

    # Disliked wines
    disliked_df = df[df['liked'] == False]
    if len(disliked_df) > 0:
        fig.add_trace(go.Scatter(
            x=disliked_df['acidity'],
            y=disliked_df['minerality'],
            mode='markers',
            marker=dict(
                size=_scaled_marker_size(disliked_df['price']),
                color='rgba(127, 117, 104, 0.22)',
                line=dict(width=1.5, color=_THEME['text_muted']),
                sizemode='diameter',
                sizemin=8,
                opacity=0.78,
            ),
            name='Disliked',
            text=disliked_df['wine_name'],
            customdata=disliked_df['price'],
            hovertemplate='<b>%{text}</b><br>Acidity: %{x}<br>Minerality: %{y}<br>Price: EUR %{customdata:.0f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text='Acidity vs Minerality',
            font=dict(size=18, family=_THEME['font_family'], color=_THEME['text']),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title='Acidity',
            range=[0, 11],
            showgrid=True,
            gridcolor=_THEME['grid'],
            zeroline=False,
            linecolor=_THEME['grid'],
            tickfont=dict(size=11, family=_THEME['font_family'], color=_THEME['text_muted']),
        ),
        yaxis=dict(
            title='Minerality',
            range=[0, 11],
            showgrid=True,
            gridcolor=_THEME['grid'],
            zeroline=False,
            linecolor=_THEME['grid'],
            tickfont=dict(size=11, family=_THEME['font_family'], color=_THEME['text_muted']),
        ),
        height=400,
        showlegend=True,
        font=dict(family=_THEME['font_family'], color=_THEME['text']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family=_THEME['font_family'], color=_THEME['text'])
        ),
        paper_bgcolor=_THEME['transparent'],
        plot_bgcolor=_THEME['transparent'],
        margin=dict(t=56, b=76, l=62, r=32),
        hoverlabel=dict(
            bgcolor=_THEME['text'],
            font=dict(family=_THEME['font_family'], color=_THEME['bg_card']),
            bordercolor=_THEME['text'],
        ),
    )

    return fig


def calculate_similarity(wine_features, target_features):
    """
    Calculate similarity using unified Palate Formula.

    Uses same derived features as predictor.py:
    - Structure score (acidity + minerality)
    - Acidity/Body ratio
    - Palate score (structure + acidity/body*2)

    Returns Euclidean distance weighted by palate formula.
    """
    import numpy as np

    # Calculate derived features for wine in history
    wine_structure = wine_features['acidity'] + wine_features['minerality']
    wine_acidity_body_ratio = wine_features['acidity'] / (wine_features['body'] + 0.1)
    wine_palate_score = wine_structure + (wine_acidity_body_ratio * 2)

    # Calculate derived features for target wine
    target_structure = target_features.acidity + target_features.minerality
    target_acidity_body_ratio = target_features.acidity / (target_features.body + 0.1)
    target_palate_score = target_structure + (target_acidity_body_ratio * 2)

    # Create feature vectors including both raw and derived features
    wine_vec = np.array([
        wine_features['acidity'],
        wine_features['minerality'],
        wine_features['fruitiness'],
        wine_features['tannin'],
        wine_features['body'],
        wine_structure / 2,  # Normalize structure score (0-20 -> 0-10)
        wine_acidity_body_ratio,
        wine_palate_score / 3  # Normalize palate score
    ])

    target_vec = np.array([
        target_features.acidity,
        target_features.minerality,
        target_features.fruitiness,
        target_features.tannin,
        target_features.body,
        target_structure / 2,
        target_acidity_body_ratio,
        target_palate_score / 3
    ])

    return np.linalg.norm(wine_vec - target_vec)


def create_master_radar(features, global_avg, color_avg, wine_color="White"):
    """
    Radar chart with 3 series:

    Series 1 (Dashed Grey): Global Average of all liked wines
    Series 2 (Solid Color): Style Target - liked wines of current color
    Series 3 (Bold Black/White Outline): Current wine being evaluated

    Args:
        features: Current wine features (WineFeatures object)
        global_avg: pandas Series with global average (all liked wines)
        color_avg: pandas Series with color-specific average (liked wines of same color)
        wine_color: Wine color for styling ('White', 'Red', 'Rosé', 'Orange')
    """
    fig = go.Figure()

    categories = ['Acidity', 'Minerality', 'Fruitiness', 'Tannin', 'Body']

    colors = _chart_palette_for(wine_color)

    # Safe extraction helper
    def safe_get(obj, attr):
        """Extract value, returning None for missing/zero/NaN."""
        try:
            val = getattr(obj, attr, None)
            return val if (val is not None and val != 0 and not pd.isna(val)) else None
        except:
            return None

    # SERIES 1: Global Average (Dashed muted line) - ALL liked wines
    if global_avg is not None and len(global_avg) > 0:
        try:
            global_vals = global_avg.fillna(0).replace(0, 5).tolist()
            global_vals = global_vals + [global_vals[0]]

            fig.add_trace(go.Scatterpolar(
                r=global_vals,
                theta=categories + [categories[0]],
                fill='none',
                line=dict(color=_THEME['text_muted'], width=1.8, dash='dash'),
                name='Your Global Average',
                marker=dict(size=5, symbol='circle', color=_THEME['text_muted'])
            ))
        except:
            pass

    # SERIES 2: Style Target (soft fill) - liked wines of SAME color
    if color_avg is not None and len(color_avg) > 0:
        try:
            color_vals = color_avg.fillna(0).replace(0, 5).tolist()
            color_vals = color_vals + [color_vals[0]]

            fig.add_trace(go.Scatterpolar(
                r=color_vals,
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor=colors['fill'],
                line=dict(color=colors['primary'], width=2.4),
                name=f'Your {wine_color} Target',
                marker=dict(size=6, symbol='circle', color=colors['primary'])
            ))
        except:
            pass

    # SERIES 3: Current Wine (ink outline)
    current_vals = [
        safe_get(features, 'acidity') or 5,
        safe_get(features, 'minerality') or 5,
        safe_get(features, 'fruitiness') or 5,
        safe_get(features, 'tannin') or 5,
        safe_get(features, 'body') or 5
    ]
    current_vals = current_vals + [current_vals[0]]

    fig.add_trace(go.Scatterpolar(
        r=current_vals,
        theta=categories + [categories[0]],
        fill='none',
        line=dict(
            color=_THEME['text'],
            width=3,
        ),
        name='Current Wine',
        marker=dict(size=7, symbol='circle', color=_THEME['text'])
    ))

    # Styling
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=True,
                tickfont=dict(size=11, family=_THEME['font_family'], color=_THEME['text_muted']),
                gridcolor=_THEME['grid'],
                linecolor=_THEME['grid'],
                tickvals=[0, 2, 4, 6, 8, 10],
            ),
            angularaxis=dict(
                tickfont=dict(size=13, family=_THEME['font_family'], color=_THEME['text']),
                gridcolor=_THEME['grid'],
                linecolor=_THEME['grid'],
            ),
            bgcolor=_THEME['transparent']
        ),
        showlegend=True,
        title=dict(
            text=f'{wine_color} Wine Profile',
            font=dict(size=18, family=_THEME['font_family'], color=_THEME['text']),
            x=0,
            xanchor='left'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, family=_THEME['font_family'], color=_THEME['text'])
        ),
        height=550,
        font=dict(family=_THEME['font_family'], color=_THEME['text']),
        paper_bgcolor=_THEME['transparent'],
        margin=dict(t=64, b=92, l=64, r=64),
        hoverlabel=dict(
            bgcolor=_THEME['text'],
            font=dict(family=_THEME['font_family'], color=_THEME['bg_card']),
            bordercolor=_THEME['text'],
        ),
    )

    return fig


def create_radar_chart(features, liked_avg, wine_color="White", disliked_avg=None):
    """
    Legacy radar function - calls Master Radar.
    Kept for backward compatibility with Tab 2.
    """
    # For Tab 2 single-color view, just show the color target
    return create_master_radar(features, None, liked_avg, wine_color)


def create_consolidated_palate_radar(color_profiles: dict):
    """
    ONE MASTER RADAR for Tab 2: Overlays all wine color profiles.

    High-contrast visualization showing liked wine averages for each color.

    Args:
        color_profiles: Dict with color names as keys, pandas Series as values
                       e.g. {'White': Series([8,7,7,1,5]), 'Red': Series([6,5,7,7,8])}

    Returns:
        Plotly figure with all color profiles overlaid
    """
    fig = go.Figure()

    categories = ['Acidity', 'Minerality', 'Fruitiness', 'Tannin', 'Body']

    # Add trace for each color profile.
    for wine_color, profile in color_profiles.items():
        if len(profile) > 0:
            colors = _chart_palette_for(wine_color)

            # Get values and close the polygon
            vals = profile.fillna(0).replace(0, 5).tolist()
            vals = vals + [vals[0]]

            # One trace per wine colour, overlaid. Distinct hue + marker
            # symbol per colour keeps them separable where they overlap;
            # the light fill keeps stacked areas from muddying.
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor=colors['fill'],
                line=dict(color=colors['primary'], width=2.6),
                name=f"{wine_color} Profile",
                marker=dict(
                    size=8,
                    symbol=colors.get('symbol', 'circle'),
                    color=colors['primary'],
                ),
                hovertemplate=f"<b>{wine_color}</b><br>" +
                             "%{theta}: %{r:.1f}/10<br>" +
                             "<extra></extra>"
            ))

    # Styling: clean and high contrast on the editorial light theme.
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=True,
                tickfont=dict(size=11, family=_THEME['font_family'], color=_THEME['text_muted']),
                gridcolor=_THEME['grid'],
                linecolor=_THEME['grid'],
                tickvals=[0, 2, 4, 6, 8, 10]
            ),
            angularaxis=dict(
                tickfont=dict(size=13, family=_THEME['font_family'], color=_THEME['text']),
                linewidth=1,
                gridcolor=_THEME['grid'],
                linecolor=_THEME['grid'],
            ),
            bgcolor=_THEME['transparent']
        ),
        showlegend=True,
        title=dict(
            text='Palate Map by Wine Color',
            font=dict(size=20, color=_THEME['text'], family=_THEME['font_family']),
            x=0,
            xanchor='left'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=13, family=_THEME['font_family'], color=_THEME['text']),
            bgcolor='rgba(0, 0, 0, 0)',
            borderwidth=0
        ),
        height=560,
        font=dict(family=_THEME['font_family'], color=_THEME['text']),
        paper_bgcolor=_THEME['transparent'],
        margin=dict(t=72, b=104, l=72, r=72),
        hoverlabel=dict(
            bgcolor=_THEME['text'],
            font=dict(family=_THEME['font_family'], color=_THEME['bg_card']),
            bordercolor=_THEME['text'],
        ),
    )

    return fig
