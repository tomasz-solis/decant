"""Plotly chart components used across the app.

All functions in this module are pure: they take inputs (DataFrames,
arrays, dicts) and return Plotly figures. They do not call Streamlit
directly — the caller does the `st.plotly_chart(fig)` rendering.

This separation matters because Streamlit reruns the entire script on
every interaction, and Plotly figure construction is the most expensive
part of those reruns. Keeping these as pure functions makes them
straightforward to cache (`@st.cache_data`) if needed in future.

`ensure_wine_df` is used in a couple of places to normalise input
DataFrames against the wine schema. It's imported here from the
top-level `app.py` for now; Phase 3 Chunk 2 will move it into
`decant.services.data_access` where the schema constants already
partially live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from decant.schema import WineFeatures


# Imported lazily from app.py to keep this module's dependency graph
# small until Chunk 2 moves ensure_wine_df into data_access.
def _ensure_wine_df(df: pd.DataFrame) -> pd.DataFrame:
    """Forward to app.ensure_wine_df at call time, to avoid a circular import.

    Will be replaced in Chunk 2 by a direct import from
    decant.services.data_access.
    """
    from app import ensure_wine_df  # noqa: PLC0415 — intentionally lazy
    return ensure_wine_df(df)


def create_mini_radar_chart(liked_avg):
    """Create a small radar chart for sidebar palate fingerprint."""
    fig = go.Figure()

    categories = ['Acidity', 'Minerality', 'Fruitiness', 'Tannin', 'Body']
    values = liked_avg.tolist() + [liked_avg.iloc[0]]

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(139, 0, 0, 0.4)',
        line=dict(color='#8B0000', width=2),
        marker=dict(size=4, color='#8B0000')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=False,
                gridcolor='rgba(255, 255, 255, 0.08)',
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='#E8E8EB'),
            )
        ),
        showlegend=False,
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='#0F0F12',
        plot_bgcolor='#0F0F12'
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
            size=liked_df['price'] * 1.5,  # Bubble size proportional to price
            color='rgba(56, 139, 253, 0.6)',
            line=dict(width=2, color='rgba(56, 139, 253, 1)'),
            sizemode='diameter',
            sizemin=4
        ),
        name='✓ Liked',
        text=liked_df['wine_name'],
        hovertemplate='<b>%{text}</b><br>Acidity: %{x}<br>Minerality: %{y}<br>Price: €%{marker.size:.0f}<extra></extra>'
    ))

    # Disliked wines
    disliked_df = df[df['liked'] == False]
    if len(disliked_df) > 0:
        fig.add_trace(go.Scatter(
            x=disliked_df['acidity'],
            y=disliked_df['minerality'],
            mode='markers',
            marker=dict(
                size=disliked_df['price'] * 1.5,
                color='rgba(248, 113, 113, 0.6)',
                line=dict(width=2, color='rgba(248, 113, 113, 1)'),
                sizemode='diameter',
                sizemin=4
            ),
            name='✗ Disliked',
            text=disliked_df['wine_name'],
            hovertemplate='<b>%{text}</b><br>Acidity: %{x}<br>Minerality: %{y}<br>Price: €%{marker.size:.0f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text='<b>Decision Boundary: Acidity vs Minerality</b>',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Acidity',
            range=[0, 11],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            title='Minerality',
            range=[0, 11],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        height=400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 240, 240, 0.3)'
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

    # Color schemes for dark mode (40% transparency for better visibility)
    style_colors = {
        'White': {'primary': '#FFD700', 'fill': 'rgba(255, 215, 0, 0.4)'},
        'Red': {'primary': '#8B0000', 'fill': 'rgba(139, 0, 0, 0.4)'},
        'Rosé': {'primary': '#FF8C69', 'fill': 'rgba(255, 140, 105, 0.4)'},
        'Orange': {'primary': '#FF8C00', 'fill': 'rgba(255, 140, 0, 0.4)'}
    }
    colors = style_colors.get(wine_color, style_colors['White'])

    # Safe extraction helper
    def safe_get(obj, attr):
        """Extract value, returning None for missing/zero/NaN."""
        try:
            val = getattr(obj, attr, None)
            return val if (val is not None and val != 0 and not pd.isna(val)) else None
        except:
            return None

    # SERIES 1: Global Average (Dashed Grey) - ALL liked wines
    if global_avg is not None and len(global_avg) > 0:
        try:
            global_vals = global_avg.fillna(0).replace(0, 5).tolist()
            global_vals = global_vals + [global_vals[0]]

            fig.add_trace(go.Scatterpolar(
                r=global_vals,
                theta=categories + [categories[0]],
                fill='none',
                line=dict(color='grey', width=2, dash='dash'),
                name='Your Global Average',
                marker=dict(size=6, symbol='circle', color='grey')
            ))
        except:
            pass

    # SERIES 2: Style Target (Solid Color Fill) - Liked wines of SAME color
    if color_avg is not None and len(color_avg) > 0:
        try:
            color_vals = color_avg.fillna(0).replace(0, 5).tolist()
            color_vals = color_vals + [color_vals[0]]

            fig.add_trace(go.Scatterpolar(
                r=color_vals,
                theta=categories + [categories[0]],
                fill='toself',  # Solid fill with 30% transparency
                fillcolor=colors['fill'],
                line=dict(color=colors['primary'], width=3),
                name=f'Your {wine_color} Target',
                marker=dict(size=8, symbol='diamond', color=colors['primary'])
            ))
        except:
            pass

    # SERIES 3: Current Wine (Bold Black/White Outline)
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
        fill='none',  # NO FILL - bold outline only
        line=dict(
            color='white',
            width=6,  # Extra bold
        ),
        name='🎯 Current Wine',
        marker=dict(size=12, symbol='star', color='white')
    ))

    # Styling
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=True,
                tickfont=dict(size=14, family='Arial Black', color='#E8E8EB'),
                gridcolor='rgba(255, 255, 255, 0.08)',
            ),
            angularaxis=dict(
                tickfont=dict(size=16, family='Arial Black', color='#E8E8EB'),
            ),
            bgcolor='rgba(15, 15, 18, 0.3)'
        ),
        showlegend=True,
        title=dict(
            text=f'<b>Master Radar: {wine_color} Wine Analysis</b>',
            font=dict(size=18, color=colors['primary']),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=12, color='#E8E8EB')
        ),
        height=550,
        paper_bgcolor='#0F0F12'
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

    # High-contrast color schemes for dark mode (40% transparency for fills)
    style_colors = {
        'White': {'primary': '#FFD700', 'fill': 'rgba(255, 215, 0, 0.4)', 'emoji': '⚪'},
        'Red': {'primary': '#8B0000', 'fill': 'rgba(139, 0, 0, 0.4)', 'emoji': '🔴'},
        'Rosé': {'primary': '#FF69B4', 'fill': 'rgba(255, 105, 180, 0.4)', 'emoji': '🌸'},
        'Orange': {'primary': '#FF8C00', 'fill': 'rgba(255, 140, 0, 0.4)', 'emoji': '🟠'}
    }

    # Add trace for each color profile
    for wine_color, profile in color_profiles.items():
        if len(profile) > 0:
            colors = style_colors.get(wine_color, style_colors['White'])

            # Get values and close the polygon
            vals = profile.fillna(0).replace(0, 5).tolist()
            vals = vals + [vals[0]]

            # Add colored fill trace with 30% transparency
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',  # Solid fill with transparency
                fillcolor=colors['fill'],
                line=dict(color=colors['primary'], width=3),
                name=f"{colors['emoji']} {wine_color} Profile",
                marker=dict(size=8, symbol='diamond', color=colors['primary']),
                hovertemplate=f"<b>{wine_color}</b><br>" +
                             "%{theta}: %{r:.1f}/10<br>" +
                             "<extra></extra>"
            ))

    # Styling - clean and high contrast for dark mode
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=True,
                tickfont=dict(size=14, family='Arial Black', color='#E8E8EB'),
                gridcolor='rgba(255, 255, 255, 0.08)',
                tickvals=[0, 2, 4, 6, 8, 10]
            ),
            angularaxis=dict(
                tickfont=dict(size=16, family='Arial Black', color='#E8E8EB'),
                linewidth=2,
                gridcolor='rgba(255, 255, 255, 0.08)'
            ),
            bgcolor='rgba(15, 15, 18, 0.3)'
        ),
        showlegend=True,
        title=dict(
            text='<b>🎯 Master Palate Radar: All Wine Profiles</b>',
            font=dict(size=20, color='#E8E8EB', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=13, family='Arial', color='#E8E8EB'),
            bgcolor='rgba(26, 26, 30, 0.9)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1
        ),
        height=600,
        paper_bgcolor='#0F0F12',
        margin=dict(t=80, b=100, l=80, r=80)
    )

    return fig

