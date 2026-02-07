"""
Decant - Taste, with confidence.
A Streamlit app for wine analytics and personalized recommendations using In-Context Learning.
"""

import sys
import os
import base64
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from decant import VinoPredictor
from decant.schema import WineExtraction
from decant.config import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_SEED
from decant import database as db
from decant.auth import setup_authentication
from pydantic import ValidationError

# Load environment variables
load_dotenv()

# AUTHENTICATION - Must be first Streamlit command!
username = setup_authentication()

# Initialize database (create tables if they don't exist)
try:
    db.init_database()
except Exception as e:
    # Table already exists or database not available - graceful fallback
    pass

# Detect Streamlit Cloud environment
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud" or os.getenv("STREAMLIT_SHARING_MODE") is not None

# Initialize OpenAI client for Vision API
# For Streamlit Cloud: use st.secrets, fallback to env vars for local dev
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found. Please set it in Streamlit Cloud secrets or .env file")
        st.stop()

client = OpenAI(api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="Decant - Taste, with confidence",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed on mobile for better UX
)

# 2026 Bento Dark Mode CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800;900&display=swap');

    /* 2026 Bento Dark Mode Color System */
    :root {
        --bg-primary: #0F0F12;
        --bg-secondary: #1A1A1E;
        --card-bg: rgba(255, 255, 255, 0.05);
        --wine-red: #800020;
        --accent-red: #8B0000;
        --accent-red-glow: rgba(128, 0, 32, 0.5);
        --text-primary: #E8E8EB;
        --text-secondary: #A0A0A8;
        --border-subtle: rgba(255, 255, 255, 0.1);
        --border-radius: 16px;
    }

    /* Global Typography - Geist with Inter fallback */
    * {
        font-family: 'Geist', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Global Background */
    body, .main, [data-testid="stAppViewContainer"] {
        background-color: #0F0F12 !important;
    }

    /* Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container */
    .main {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* 2026 Bento Glassmorphic Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }

    /* Hero Card with Wine Red Glow */
    .glass-card.glow {
        box-shadow: 0 8px 32px rgba(128, 0, 32, 0.5),
                    0 0 80px rgba(128, 0, 32, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    /* Radial Gradient Text Effect for Match Likelihood */
    .match-score-gradient {
        background: radial-gradient(circle at 30% 50%, #FF1744 0%, #800020 50%, #4A0012 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        letter-spacing: -3px;
        text-shadow: 0 0 40px rgba(255, 23, 68, 0.3);
    }

    /* Fallback for browsers without backdrop-filter */
    @supports not (backdrop-filter: blur(10px)) {
        .glass-card {
            background: rgba(26, 26, 30, 0.95);
        }
    }

    /* Header Styling */
    .main-title {
        font-size: 2.5em;
        font-weight: 700;
        text-align: center;
        color: var(--text-primary);
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1em;
        margin-bottom: 32px;
        font-weight: 500;
    }

    /* 2026 Bento Button Styling */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.1em;
        font-weight: 600;
        border-radius: 12px;
        margin: 10px 0;
        background: linear-gradient(135deg, var(--wine-red) 0%, var(--accent-red) 100%);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 16px rgba(128, 0, 32, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #A60028 0%, #D00000 100%);
        box-shadow: 0 6px 24px rgba(128, 0, 32, 0.6),
                    0 0 40px rgba(128, 0, 32, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed var(--accent-red);
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        background: var(--card-bg);
    }

    /* Dark Mode Overrides for Streamlit Components */
    .stSelectbox, .stMultiSelect, .stTextInput, .stNumberInput {
        color: var(--text-primary);
    }

    .stSelectbox > div > div {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
    }

    /* Plotly Chart Container */
    .js-plotly-plot {
        background-color: transparent !important;
    }

    /* Bento Grid Layout */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 24px 0;
    }

    /* Wine Gallery Card - Flex Container for Sticky Footer */
    .wine-card {
        padding: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        display: flex;
        flex-direction: column;
        height: 100%;
    }

    .wine-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 16px 48px rgba(128, 0, 32, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .wine-card h4 {
        font-weight: 600;
        line-height: 1.3;
    }

    .wine-card .badge {
        display: inline-block;
        transition: transform 0.2s;
    }

    .wine-card:hover .badge {
        transform: scale(1.05);
    }

    /* Wine Card Image - Strict 350px Container */
    .wine-card-img {
        height: 350px;
        width: 100%;
        object-fit: contain;
        background: #0a0a0a;
        border-radius: 8px;
    }

    .wine-card-img-placeholder {
        height: 350px;
        width: 100%;
        background: rgba(139, 0, 0, 0.1);
        border: 2px solid rgba(139, 0, 0, 0.3);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 60px;
    }

    /* Wine Card Content Wrapper - Full Height Flex */
    .wine-card-content {
        display: flex;
        flex-direction: column;
        height: 100%;
        gap: 12px;
    }

    /* Sticky Footer - Pushes Bottom Elements to Same Level */
    .wine-card-footer {
        margin-top: auto;
    }

    /* Icon Row */
    .icon-row {
        display: flex;
        gap: 8px;
        margin: 8px 0;
    }

    /* Seal of Approval */
    .seal-of-approval {
        position: relative;
        overflow: hidden;
    }

    .seal-of-approval::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(139, 0, 0, 0.2), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0%, 100% { left: -100%; }
        50% { left: 100%; }
    }

    /* Responsive Breakpoints - OPTIMIZED FOR IN-SHOP MOBILE USE */
    @media (max-width: 768px) {
        /* Compact header for more screen real estate */
        .main-title {
            font-size: 1.5em;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        }

        .subtitle {
            font-size: 0.85em;
            margin-bottom: 16px;
        }

        /* Tighter spacing on mobile */
        .glass-card {
            padding: 12px;
            margin: 8px 0;
            border-radius: 12px;
        }

        /* BIGGER BUTTONS for in-shop quick taps */
        .stButton > button {
            height: 56px !important;
            font-size: 1.1em !important;
            font-weight: 700 !important;
            border-radius: 12px;
        }

        /* Wine images - optimized for mobile viewing */
        .wine-card-img, .wine-card-img-placeholder {
            height: 240px;  /* Smaller on mobile to see more content */
        }

        /* Single column layout - essential for shop browsing */
        .bento-grid {
            grid-template-columns: 1fr !important;
            gap: 12px;
        }

        /* CRITICAL: Better touch targets (Apple HIG 44px minimum) */
        .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
            min-height: 48px !important;
        }

        /* File uploader - make it huge and obvious for quick photo capture */
        .stFileUploader {
            padding: 24px !important;
            margin: 16px 0 !important;
        }

        .stFileUploader label {
            font-size: 1.2em !important;
            font-weight: 600 !important;
        }

        /* Sidebar collapsed by default on mobile */
        [data-testid="stSidebar"] {
            min-width: 0;
        }

        [data-testid="stSidebar"][aria-expanded="false"] {
            margin-left: -21rem;
        }

        /* Main content full width on mobile */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }

        /* Columns stack vertically on mobile - CRITICAL for readability */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            min-width: 100% !important;
        }

        /* Force stacking for multi-column layouts (3+ columns) */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 8px !important;
        }

        /* Metrics more compact on mobile */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.03);
            padding: 8px;
            border-radius: 8px;
        }

        /* Tab navigation bigger for easier tapping */
        .stTabs [data-baseweb="tab-list"] button {
            min-height: 48px !important;
            font-size: 1em !important;
        }

        /* Image preview on mobile - contain to screen */
        img {
            max-width: 100% !important;
            height: auto !important;
        }
    }

    @media (min-width: 769px) and (max-width: 1200px) {
        .bento-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .wine-card-img, .wine-card-img-placeholder {
            height: 320px;  /* Medium height for tablets */
        }
    }

    @media (min-width: 1201px) {
        .bento-grid {
            grid-template-columns: repeat(4, 1fr);
        }
    }

    /* Landscape Mobile Optimization - For horizontal phone in shop */
    @media (max-width: 900px) and (orientation: landscape) {
        .main-title {
            font-size: 1.2em;
            margin-bottom: 2px;
        }
        .subtitle {
            font-size: 0.8em;
            margin-bottom: 8px;
        }
        .glass-card {
            padding: 8px;
            margin: 6px 0;
        }
        .stButton > button {
            height: 44px !important;
            font-size: 0.95em !important;
        }
        /* Compact metrics in landscape */
        [data-testid="stMetric"] {
            padding: 6px;
        }
    }

    /* Small phones (iPhone SE, etc) - Extra compact */
    @media (max-width: 375px) {
        .main-title {
            font-size: 1.3em;
        }
        .glass-card {
            padding: 10px;
        }
        .stButton > button {
            height: 52px !important;
            font-size: 1em !important;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load predictor with caching."""
    try:
        predictor = VinoPredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        st.info("💡 Make sure OPENAI_API_KEY is set in your .env file")
        return None


@st.cache_data
def load_wine_data(user_id: str = "admin"):
    """
    Load wine data from PostgreSQL database with fallback to CSV.

    Args:
        user_id: User ID to filter wines. Defaults to 'admin' for backward compatibility.

    Returns:
        DataFrame with wine data filtered by user
    """
    try:
        # Try loading from database first (filtered by user)
        df = db.get_all_wines(user_id=user_id)

        if df is not None and len(df) > 0:
            # NaN PROTECTION: Fill missing values before any logic runs
            # Numerical columns -> 0
            numerical_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body',
                             'score', 'price', 'vintage']
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

            # Boolean-like numerical columns (stored as float) -> 0
            for col in ['is_sparkling', 'is_natural']:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)

            # Text columns -> 'Unknown'
            text_cols = ['wine_name', 'producer', 'notes', 'country', 'region',
                        'wine_color', 'sweetness']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('Unknown').astype(str)

            # Boolean columns -> False
            if 'liked' in df.columns:
                df['liked'] = df['liked'].fillna(False)

            return df

    except Exception as e:
        st.warning(f"⚠️ Database unavailable, falling back to CSV: {str(e)}")

    # Fallback to CSV if database fails
    data_path = Path("data/processed/wine_features.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)

        # NaN PROTECTION: Fill missing values before any logic runs
        numerical_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body',
                         'score', 'price', 'vintage']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        for col in ['is_sparkling', 'is_natural']:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        text_cols = ['wine_name', 'producer', 'notes', 'country', 'region',
                    'wine_color', 'sweetness']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)

        if 'liked' in df.columns:
            df['liked'] = df['liked'].fillna(False)

        return df

    return None


def should_display_vintage(vintage_value):
    """Check if vintage should be displayed (not 0, None, or NaN)."""
    if vintage_value is None or pd.isna(vintage_value):
        return False
    try:
        vintage_int = int(vintage_value)
        return vintage_int > 0 and vintage_int < 2100
    except (ValueError, TypeError):
        return False


def get_wine_image_url(wine_name, producer):
    """Fetch wine bottle image URL from online sources."""
    try:
        # Use Vivino search URL as a fallback
        # Note: This returns a search page URL. For actual images, we'd need to scrape or use an API
        search_query = f"{producer} {wine_name}".replace(" ", "+")
        return f"https://www.vivino.com/search/wines?q={search_query}"
    except Exception:
        return None


def get_wine_image_path(wine_name):
    """Get the stored image path for a wine."""
    import re
    # Create safe filename from wine name
    safe_name = re.sub(r'[^\w\s-]', '', wine_name.lower())
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    image_dir = Path("data/wine_images")

    # Check for common image extensions
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        image_path = image_dir / f"{safe_name}{ext}"
        if image_path.exists():
            return str(image_path)
    return None


def save_wine_image(uploaded_file, wine_name):
    """Save uploaded wine image."""
    import re
    from PIL import Image
    import io

    try:
        # Create safe filename
        safe_name = re.sub(r'[^\w\s-]', '', wine_name.lower())
        safe_name = re.sub(r'[-\s]+', '_', safe_name)

        # Get file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in ['jpg', 'jpeg', 'png', 'webp']:
            file_ext = 'jpg'

        # Save path
        image_dir = Path("data/wine_images")
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{safe_name}.{file_ext}"

        # Open and resize image to reasonable size
        image = Image.open(uploaded_file)

        # Resize to max 800px width while maintaining aspect ratio
        max_width = 800
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        # Save optimized image
        image.save(image_path, quality=85, optimize=True)
        return str(image_path)
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None


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
    fig = go.Figure()

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
    MASTER RADAR with 3 series for Deep UI Alignment:

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

    # Safe extraction helper - NO 5/10 DEFAULTS
    def safe_get(obj, attr):
        """Extract value with NO generic defaults."""
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


def extract_complete_wine_data(image_file, history_df):
    """
    Extract wine data from photo using simplified schema.

    Returns dict with core 12 columns for history.csv.
    """
    try:
        # Convert image to base64
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_type = "image/jpeg" if image_file.name.lower().endswith('.jpg') or image_file.name.lower().endswith('.jpeg') else "image/png"

        # Build self-learning context from liked wines
        context = ""
        if history_df is not None and len(history_df) > 0:
            liked_wines = history_df[history_df['liked'] == True].tail(5)
            if len(liked_wines) > 0:
                context = "\n\nUSER'S TASTE PROFILE (Recent Liked Wines):\n"
                for _, wine in liked_wines.iterrows():
                    context += f"- {wine.get('wine_name', 'Unknown')}: Acidity {wine.get('acidity', 0)}/10, Minerality {wine.get('minerality', 0)}/10\n"

        # AGGRESSIVE HIGH-DIMENSIONAL extraction prompt
        prompt = f"""Analyze this wine bottle and extract COMPLETE HIGH-DIMENSIONAL wine information.

{context}

BE AGGRESSIVE - infer ALL attributes from the label and your wine knowledge.

## REQUIRED FIELDS:

**Basic Info:**
WINE_NAME: [Full name with vintage, e.g. "Fefiñanes Albariño 2022"]
PRODUCER: [Winery name]
VINTAGE: [Year]
TASTING_NOTES: [Professional tasting notes]
OVERALL_SCORE: [Your rating 1-10]
PRICE_EUR: [Estimated retail price in EUR]

**WINE ORIGIN (MANDATORY - NEVER LEAVE BLANK):**
COUNTRY: [Country of origin - REQUIRED]
- Use label text, producer location, or wine style knowledge
- Examples: Spain, France, Italy, USA, Germany, Portugal

REGION: [Specific wine region/appellation - REQUIRED]
- Look for DO, DOCa, AOC, AVA, or regional indicators
- Examples:
  * Albariño → Rías Baixas
  * Rioja label → Rioja
  * Bordeaux château → Bordeaux
  * Barolo → Piedmont
- If no specific region visible, use producer's primary region
- NEVER leave as "Unknown" - use your encyclopedic wine knowledge
- Common Spanish DOs: Rías Baixas, Ribera del Duero, Rioja, Priorat
- Common French AOCs: Bordeaux, Burgundy, Champagne, Rhône
- Common Italian DOCs: Tuscany, Piedmont, Veneto

**HIGH-DIMENSIONAL ATTRIBUTES (AGGRESSIVE INFERENCE):**

WINE_COLOR: [MUST be: "White", "Red", "Rosé", or "Orange"]
- Look at bottle color, label design, grape variety

IS_SPARKLING: [Boolean: true or false]
- true if Champagne, Cava, Prosecco, or "Espumante"
- Check for bubbles indication on label

IS_NATURAL: [Boolean: true or false]
- true if label shows "organic", "bio", "natural", "biodynamic"
- Check for certification badges

SWEETNESS: [MUST be: "Dry", "Medium-Dry", "Medium-Sweet", or "Sweet"]
- Infer from region/style:
  * Albariño/Verdejo → "Dry"
  * German Kabinett → "Medium-Sweet"
  * Sauternes/Moscato → "Sweet"
  * Champagne Brut → "Dry"
  * Default table wine → "Dry"

**CORE 5 FLAVOR FEATURES (1-10 scale):**
ACIDITY: [1-10]
MINERALITY: [1-10]
FRUITINESS: [1-10]
TANNIN: [1-10 - whites typically 1-3]
BODY: [1-10]

Use encyclopedic wine knowledge. Make aggressive inferences."""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{base64_image}"}}
                ]
            }],
            max_tokens=800,
            temperature=OPENAI_TEMPERATURE
        )

        content = response.choices[0].message.content

        # Parse fields
        import re
        from datetime import datetime

        def extract_field(text, field_name):
            match = re.search(f'{field_name}:\\s*(.+)', text, re.IGNORECASE)
            return match.group(1).strip() if match else ''

        def extract_number(text, field_name):
            match = re.search(f'{field_name}:\\s*(\\d+\\.?\\d*)', text, re.IGNORECASE)
            return float(match.group(1)) if match else 0  # Return 0 if not found - predictor will infer

        def extract_boolean(text, field_name):
            match = re.search(f'{field_name}:\\s*(true|false)', text, re.IGNORECASE)
            return match.group(1).lower() == 'true' if match else False

        # Extract raw data from LLM response
        raw_data = {
            'wine_name': extract_field(content, 'WINE_NAME'),
            'producer': extract_field(content, 'PRODUCER'),
            'vintage': int(extract_number(content, 'VINTAGE')) or 0,
            'notes': extract_field(content, 'TASTING_NOTES'),
            'score': extract_number(content, 'OVERALL_SCORE'),
            'liked': None,  # User will set
            'price': extract_number(content, 'PRICE_EUR') or 0.0,
            # WINE ORIGIN (AI-extracted)
            'country': extract_field(content, 'COUNTRY') or 'Unknown',
            'region': extract_field(content, 'REGION') or 'Unknown',
            # HIGH-DIMENSIONAL ATTRIBUTES
            'wine_color': extract_field(content, 'WINE_COLOR') or 'White',
            'is_sparkling': extract_boolean(content, 'IS_SPARKLING'),
            'is_natural': extract_boolean(content, 'IS_NATURAL'),
            'sweetness': extract_field(content, 'SWEETNESS') or 'Dry',
            # Core 5 flavor features
            'acidity': extract_number(content, 'ACIDITY') or 5.0,  # Default to mid-range
            'minerality': extract_number(content, 'MINERALITY') or 5.0,
            'fruitiness': extract_number(content, 'FRUITINESS') or 5.0,
            'tannin': extract_number(content, 'TANNIN') or 3.0,  # Lower default for whites
            'body': extract_number(content, 'BODY') or 5.0
        }

        # SECURITY FIX: Validate extracted data before returning
        from pydantic import ValidationError
        from decant.constants import WineColor, Sweetness

        # Validate critical fields
        validation_errors = []

        # Validate wine_name
        if not raw_data['wine_name'] or len(raw_data['wine_name']) < 1:
            validation_errors.append("wine_name is empty or too short")

        # Validate producer
        if not raw_data['producer'] or len(raw_data['producer']) < 1:
            validation_errors.append("producer is empty or too short")

        # Validate vintage
        if raw_data['vintage'] < 1900 or raw_data['vintage'] > 2100:
            st.warning(f"⚠️ Invalid vintage {raw_data['vintage']}, setting to 0")
            raw_data['vintage'] = 0

        # Validate features are in range
        for feature in ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']:
            value = raw_data[feature]
            if value < 1.0 or value > 10.0:
                st.warning(f"⚠️ {feature} value {value} out of range [1-10], clamping")
                raw_data[feature] = max(1.0, min(10.0, value))

        # Validate score
        if raw_data['score'] < 1.0 or raw_data['score'] > 10.0:
            st.warning(f"⚠️ Score {raw_data['score']} out of range [1-10], clamping")
            raw_data['score'] = max(1.0, min(10.0, raw_data['score']))

        # Validate wine_color
        valid_colors = [c.value for c in WineColor]
        if raw_data['wine_color'] not in valid_colors:
            st.warning(f"⚠️ Invalid wine color '{raw_data['wine_color']}', defaulting to 'White'")
            raw_data['wine_color'] = 'White'

        # Validate sweetness
        valid_sweetness = [s.value for s in Sweetness]
        if raw_data['sweetness'] not in valid_sweetness:
            st.warning(f"⚠️ Invalid sweetness '{raw_data['sweetness']}', defaulting to 'Dry'")
            raw_data['sweetness'] = 'Dry'

        # If critical validation errors, return None
        if validation_errors:
            st.error(f"🚨 Critical validation errors: {', '.join(validation_errors)}")
            st.info("💡 Please verify the image and try again, or enter data manually.")
            return None

        return raw_data

    except json.JSONDecodeError as je:
        st.error(f"🚨 LLM returned invalid JSON: {je}")
        st.info("💡 Please try again or enter data manually.")
        return None
    except ValidationError as ve:
        st.error(f"🚨 Validation error: {ve}")
        st.info("💡 Please try again or enter data manually.")
        return None
    except Exception as e:
        st.error(f"❌ Error extracting wine data: {str(e)}")
        st.info("💡 Please try again or enter data manually.")
        return None


def main():
    # Premium Dark Header
    st.markdown("""
<div class="glass-card" style="text-align: center; padding: 32px 20px; margin-bottom: 32px; background: linear-gradient(135deg, rgba(139, 0, 0, 0.1) 0%, rgba(26, 26, 30, 0.05) 100%);">
    <h1 class="main-title">🍷 Decant</h1>
    <p class="subtitle">Taste, with confidence.</p>
</div>
""", unsafe_allow_html=True)

    # Streamlit Cloud deployment warning (persistent at top)
    if IS_STREAMLIT_CLOUD:
        st.info(
            "ℹ️ **Running on Streamlit Cloud**: Your tasting history is stored in a CSV file. "
            "On the free tier, data will reset when the app restarts. "
            "Use the 📥 Download button in Analytics to backup your collection regularly."
        )

    # Create tabs for main navigation
    tab1, tab2, tab3 = st.tabs(["🍷 Add Wine", "📊 My Palate Maps", "🖼️ Wine Gallery"])

    # Sidebar
    with st.sidebar:
        st.header("📊 Palate Summary")

        # Load wine features data with caching
        df = load_wine_data(username)

        if df is not None:
            # 🌍 REGIONAL FILTER DROPDOWN
            if 'region' in df.columns:
                # Get unique regions (exclude Unknown)
                regions = df[
                    (df['region'] != 'Unknown') &
                    (df['region'].notna())
                ]['region'].unique()

                if len(regions) > 0:
                    regions_sorted = sorted(regions)
                    selected_region = st.selectbox(
                        "🌍 Filter by Region",
                        ["All Regions"] + list(regions_sorted),
                        key='region_filter'
                    )

                    # Apply filter if not "All Regions"
                    if selected_region != "All Regions":
                        df = df[df['region'] == selected_region]
                        st.caption(f"Showing: {selected_region}")

            st.markdown("---")

            # Calculate summary stats (possibly filtered)
            total_wines = len(df)
            liked_wines = df['liked'].sum()
            disliked_wines = total_wines - liked_wines

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Liked", liked_wines)
            with col2:
                st.metric("❌ Disliked", disliked_wines)

            st.metric("📝 Total Wines", total_wines)

            # Quick Stats only (Radar removed as requested)
            st.markdown("---")
            st.markdown("### 🧬 Palate Stats")

            feature_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']
            liked_df = df[df['liked'] == True]

            if len(liked_df) > 0:
                liked_avg = liked_df[feature_cols].mean()

                # Check if all values are 0 (no data yet)
                if liked_avg.sum() == 0:
                    st.caption("🔍 Add wines with flavor profiles to see your palate stats")
                else:
                    # Show average values only (no radar)
                    st.caption("Your ideal wine profile:")
                    feat_col1, feat_col2, feat_col3 = st.columns(3)
                    with feat_col1:
                        st.metric("⚡ Acid", f"{liked_avg['acidity']:.1f}")
                        st.metric("💎 Mineral", f"{liked_avg['minerality']:.1f}")
                    with feat_col2:
                        st.metric("🍇 Fruit", f"{liked_avg['fruitiness']:.1f}")
                        st.metric("🌰 Tannin", f"{liked_avg['tannin']:.1f}")
                    with feat_col3:
                        st.metric("💪 Body", f"{liked_avg['body']:.1f}")

            # 🌍 REGIONAL ANALYTICS
            st.markdown("---")
            st.markdown("### 🏆 Top Regions")

            if 'region' in liked_df.columns and 'country' in liked_df.columns:
                # Filter out Unknown regions
                regional_wines = liked_df[
                    (liked_df['region'] != 'Unknown') &
                    (liked_df['region'].notna())
                ]

                if len(regional_wines) > 0:
                    # Calculate average score by region
                    regional_stats = regional_wines.groupby('region').agg({
                        'score': 'mean',
                        'wine_name': 'count'
                    }).round(1)

                    regional_stats.columns = ['avg_score', 'count']
                    regional_stats = regional_stats.sort_values('avg_score', ascending=False)

                    # Show top 3 regions
                    for idx, (region, stats) in enumerate(regional_stats.head(3).iterrows()):
                        if idx == 0:
                            st.metric(
                                f"🥇 {region}",
                                f"{stats['avg_score']:.1f}/10",
                                f"{int(stats['count'])} wines"
                            )
                        else:
                            medal = "🥈" if idx == 1 else "🥉"
                            st.metric(
                                f"{medal} {region}",
                                f"{stats['avg_score']:.1f}/10",
                                f"{int(stats['count'])} wines"
                            )
                else:
                    st.caption("Log wines with regions to see analytics")

            # Show top liked wines with PROMINENT GEOGRAPHY
            st.markdown("---")
            st.markdown("### 🌟 Top Liked Wines")

            liked_df_sorted = liked_df.sort_values('score', ascending=False)

            for idx, row in liked_df_sorted.head(5).iterrows():
                wine_name = row.get('producer', row.get('wine_name', 'Unknown'))

                # Build geography string for display
                country = row.get('country', 'Unknown')
                region = row.get('region', 'Unknown')

                # Only show location if we have real data
                if region != 'Unknown' and country != 'Unknown':
                    location = f"📍 {region}, {country}"
                elif country != 'Unknown':
                    location = f"📍 {country}"
                else:
                    location = None

                with st.expander(f"{wine_name}"):
                    # Show location prominently at top (if available)
                    if location:
                        st.markdown(f"**{location}**")
                    st.write(f"**Score:** {row.get('score', 0):.1f}/10")
                    # Only show vintage if it's valid
                    if should_display_vintage(row.get('vintage')):
                        st.write(f"**Vintage:** {int(row.get('vintage'))}")
                    price_col = 'price' if 'price' in row else 'price_eur'
                    st.write(f"**Price:** €{row.get(price_col, 0):.2f}")
        else:
            st.warning("No wine data found. Run feature extraction first.")

        # DEBUG: Show raw data to verify NaN protection
        st.markdown("---")
        st.markdown("### 🔍 Debug Data")
        if df is not None:
            st.caption("First 5 wines (verify no NaNs):")
            st.dataframe(df.head(), width="stretch")
        else:
            st.caption("No data loaded")

        st.markdown("---")
        st.info("Decant uses AI to predict wine compatibility based on your tasting history.")

    # 🍷 TAB 1: Add Wine
    with tab1:
        st.markdown("### 🍷 Add Wine to Collection")
        st.caption("Enter wine name or upload a photo - AI extracts everything else")

        # Load history for self-learning context
        history_df = load_wine_data(username)

        # Input mode selection
        input_mode = st.radio(
            "Input Method",
            ["📝 Enter Wine Name", "📸 Upload Photo"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if input_mode == "📝 Enter Wine Name":
            # Text input mode - MOBILE-OPTIMIZED with voice input hint
            st.markdown("### 🍷 Enter Wine Name")
            st.caption("Type or use voice input (tap microphone on mobile keyboard)")

            wine_name_input = st.text_input(
                "Wine Name",
                placeholder="e.g., Fefiñanes Albariño 2022",
                help="💬 Mobile tip: Use voice input for faster entry!",
                label_visibility="collapsed"
            )

            if wine_name_input and st.button("🔍 CHECK THIS WINE", type="primary", width="stretch"):
                with st.spinner("🧠 AI is extracting wine details from name..."):
                    predictor = load_predictor()
                    if predictor:
                        extraction = predictor.extract_wine_data(wine_name_input)

                        # Convert to dict with HIGH-DIMENSIONAL attributes + GEOGRAPHY
                        wine_data = {
                            'wine_name': extraction.wine_name,
                            'producer': extraction.producer,
                            'vintage': extraction.vintage,
                            'notes': extraction.notes,
                            'score': float(extraction.score),
                            'liked': None,  # User will set
                            'price': 0.0,  # User will set
                            # WINE ORIGIN (AI-extracted)
                            'country': extraction.country,
                            'region': extraction.region,
                            # HIGH-DIMENSIONAL ATTRIBUTES (AI-inferred)
                            'wine_color': extraction.wine_color,
                            'is_sparkling': extraction.is_sparkling,
                            'is_natural': extraction.is_natural,
                            'sweetness': extraction.sweetness,
                            # Core 5 flavor features
                            'acidity': extraction.acidity,
                            'minerality': extraction.minerality,
                            'fruitiness': extraction.fruitiness,
                            'tannin': extraction.tannin,
                            'body': extraction.body
                        }

                        st.session_state['wine_data'] = wine_data
                        st.success("✅ Wine data extracted!")
                        st.rerun()

        else:
            # Photo upload mode - MOBILE-OPTIMIZED for in-shop use
            st.markdown("### 📸 Snap a Photo")
            st.caption("Point your camera at the wine label - AI does the rest!")

            uploaded_file = st.file_uploader(
                "Tap to open camera or choose photo",
                type=["jpg", "jpeg", "png"],
                help="📱 On mobile: Opens camera automatically | 💻 On desktop: Upload from files",
                label_visibility="visible",
                accept_multiple_files=False
            )

            if uploaded_file:
                # Show image preview
                st.image(uploaded_file, caption="Wine Bottle", width="stretch")

                # Auto-extract ALL data when photo is uploaded
                if 'wine_data' not in st.session_state or st.session_state.get('last_upload') != uploaded_file.name:
                    with st.spinner("🧠 AI is analyzing your wine... extracting all details"):
                        uploaded_file.seek(0)
                        wine_data = extract_complete_wine_data(uploaded_file, history_df)

                        if wine_data:
                            st.session_state['wine_data'] = wine_data
                            st.session_state['last_upload'] = uploaded_file.name
                            st.success("✅ Wine analyzed! All fields extracted automatically")
                            st.rerun()

        # Show extracted data if available
        if 'wine_data' in st.session_state:
            wine_data = st.session_state['wine_data']

            # Display wine name prominently with geography
            st.markdown(f"## 🍷 {wine_data['wine_name']}")

            # 🌍 BULLETPROOF LOCATION HEADER - Explicit mapping to prevent NaNs
            # Explicitly map from wine_data with fallbacks
            country = wine_data.get('country', None)
            region = wine_data.get('region', None)

            # Convert None, NaN, empty string, or 'nan' string to 'Unknown'
            if country is None or country == '' or str(country).lower() == 'nan' or (isinstance(country, float) and pd.isna(country)):
                country = 'Unknown'
            else:
                country = str(country)

            if region is None or region == '' or str(region).lower() == 'nan' or (isinstance(region, float) and pd.isna(region)):
                region = 'Unknown'
            else:
                region = str(region)

            # Display ONLY if we have real data (no "Unknown" placeholders)
            if country != 'Unknown' and region != 'Unknown':
                st.markdown(f"### 📍 {region}, {country}")
            elif country != 'Unknown':
                st.markdown(f"### 📍 {country}")

            # 🎯 VISUAL STYLE HEADER - Categorical Clarity
            wine_color = wine_data.get('wine_color', 'White')
            region = wine_data.get('region', 'Unknown')
            is_sparkling = wine_data.get('is_sparkling', False)
            sweetness = wine_data.get('sweetness', 'Dry')

            # Build style descriptor
            style_type = "Sparkling" if is_sparkling else "Still"
            style_full = f"{sweetness} {style_type}"

            # Color emojis (used in other sections, not for header)
            color_emoji = {"White": "⚪", "Red": "🔴", "Rosé": "🌸", "Orange": "🟠"}
            color_icon = color_emoji.get(wine_color, '⚪')

            # 🎯 PALATE MATCH VERDICT - Move to TOP (Deep UI Alignment requirement)
            if history_df is not None and len(history_df) > 0:
                from decant.predictor import VinoPredictor

                # Initialize predictor
                predictor = VinoPredictor()

                # Calculate likelihood - HARDENED with style-based inference
                wine_features_dict = {
                    'acidity': wine_data.get('acidity', 0),
                    'minerality': wine_data.get('minerality', 0),
                    'fruitiness': wine_data.get('fruitiness', 0),
                    'tannin': wine_data.get('tannin', 0),
                    'body': wine_data.get('body', 0)
                }

                # 🚨 If features not extracted from image, use OpenAI to infer with explanation
                feature_descriptions = {}
                if all(v == 0 for v in wine_features_dict.values()):
                    wine_name = wine_data.get('wine_name', '')
                    region = wine_data.get('region', 'Unknown')

                    # Ask OpenAI to rate AND explain each characteristic
                    st.info("ℹ️ Wine characteristics inferred from wine name and region (not extracted from label)")

                    # Cache key for consistent results
                    cache_key = f"{wine_name}_{region}".lower().replace(" ", "_")

                    # Check if we've already rated this wine
                    if 'wine_ratings_cache' not in st.session_state:
                        st.session_state['wine_ratings_cache'] = {}

                    if cache_key in st.session_state['wine_ratings_cache']:
                        # Use cached ratings for consistency
                        cached = st.session_state['wine_ratings_cache'][cache_key]
                        wine_features_dict = cached['features']
                        feature_descriptions = cached['descriptions']
                        wine_data.update({
                            'acidity': wine_features_dict['acidity'],
                            'fruitiness': wine_features_dict['fruitiness'],
                            'body': wine_features_dict['body'],
                            'minerality': wine_features_dict['minerality'],
                            'tannin': wine_features_dict['tannin']
                        })
                        st.caption("✓ Using cached ratings for consistency")
                    else:
                        # First time - get ratings from LLM
                        try:
                            # Nuclear-Grade Feature Extraction Prompt for Decision Science
                            inference_prompt = f"""Role: You are a Master Sommelier and Data Engineer specializing in quantitative viticulture.

Task: Provide a precise, technical flavor profile for the wine: {wine_name} from {region}.

Objective: Your output will be used to calculate a vector-space similarity model. Consistency in your scoring logic is mandatory.

Scoring Guidelines (Scale 1.0 - 10.0):
• Acidity: 1.0 (Flat/Flabby) to 10.0 (High Tartaric/Piercing)
• Fruitiness: 1.0 (Earth-driven/Savory) to 10.0 (Primary Fruit Bomb/Jammy)
• Body: 1.0 (Light/Watery) to 10.0 (Full/Viscous/Heavy)
• Tannin: 1.0 (No structure/Silk) to 10.0 (Aggressive/Gripping/Astringent)
• Minerality: 1.0 (Clean/Fruit-only) to 10.0 (Stony/Saline/Chalky)

Requirements:
1. Use your internal knowledge of this specific producer, vintage, and regional style.
2. Avoid "safe" middle-ground scores (like 5.0) unless truly warranted.
3. Provide the output ONLY as a JSON object for programmatic parsing.

Desired JSON Structure:
{{
  "wine_metadata": {{
    "name": "{wine_name}",
    "region": "{region}",
    "style": "Regional style description"
  }},
  "technical_profile": {{
    "acidity": float,
    "fruitiness": float,
    "body": float,
    "tannin": float,
    "minerality": float
  }},
  "sommelier_verdict": "One sentence technical summary of the structure."
}}"""

                            response = client.chat.completions.create(
                                model=OPENAI_MODEL,
                                messages=[
                                    {"role": "user", "content": inference_prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=OPENAI_TEMPERATURE,
                                seed=OPENAI_SEED
                            )

                            import json
                            from pydantic import ValidationError
                            from decant.constants import LLMWineAnalysis

                            # Parse JSON response
                            result = json.loads(response.choices[0].message.content)

                            # SECURITY FIX: Validate LLM response with Pydantic
                            try:
                                validated_response = LLMWineAnalysis.model_validate(result)

                                # Extract technical profile scores from validated response
                                profile = validated_response.technical_profile
                                wine_features_dict = {
                                    'acidity': float(profile.acidity),
                                    'fruitiness': float(profile.fruitiness),
                                    'body': float(profile.body),
                                    'minerality': float(profile.minerality),
                                    'tannin': float(profile.tannin)
                                }

                                # Use sommelier verdict as explanation for all features
                                sommelier_verdict = validated_response.sommelier_verdict
                                feature_descriptions = {
                                    'acidity': f"{profile.acidity}/10 - {sommelier_verdict}",
                                    'fruitiness': f"{profile.fruitiness}/10 - {sommelier_verdict}",
                                    'body': f"{profile.body}/10 - {sommelier_verdict}",
                                    'minerality': f"{profile.minerality}/10 - {sommelier_verdict}",
                                    'tannin': f"{profile.tannin}/10 - {sommelier_verdict}"
                                }

                                # Update wine_data with inferred values so they display correctly
                                wine_data['acidity'] = wine_features_dict['acidity']
                                wine_data['fruitiness'] = wine_features_dict['fruitiness']
                                wine_data['body'] = wine_features_dict['body']
                                wine_data['minerality'] = wine_features_dict['minerality']
                                wine_data['tannin'] = wine_features_dict['tannin']

                                # Cache the results for future consistency
                                st.session_state['wine_ratings_cache'][cache_key] = {
                                    'features': wine_features_dict,
                                    'descriptions': feature_descriptions
                                }

                            except ValidationError as ve:
                                # Validation failed - LLM returned invalid data
                                st.error(f"🚨 LLM returned invalid response structure: {ve}")
                                st.info("💡 Please enter features manually below.")
                                # Don't cache invalid results

                        except json.JSONDecodeError as je:
                            st.error(f"🚨 LLM returned invalid JSON: {je}")
                            st.info("💡 Please enter features manually below.")
                        except KeyError as ke:
                            st.error(f"🚨 LLM response missing required field: {ke}")
                            st.info("💡 Please enter features manually below.")
                        except Exception as e:
                            st.warning(f"⚠️ Could not infer wine characteristics: {str(e)}")
                            st.info("💡 Please enter features manually below.")
                            wine_features_dict = None

                # 🎯 PALATE ENGINE - SINGLE SOURCE OF TRUTH
                # Calculate palate score - display_match_score is THE ONLY variable for all UI
                # CRITICAL: display_match_score is extracted ONCE and used in:
                #   1. Hero Card (SOLE AUTHORITATIVE display)
                #   2. Liked toggle default
                palate_score = None
                display_match_score = None  # SINGLE SOURCE OF TRUTH - backend variable for all UI

                if wine_features_dict is not None:
                    palate_score = predictor.calculate_palate_score(
                        wine_features_dict,
                        wine_color
                    )
                    # SINGLE SOURCE OF TRUTH: Extract once, use everywhere
                    display_match_score = palate_score.likelihood_score
                    # If this is 69.8%, hero card will display 69.8% as the sole authority

                # 🎯 HERO CARD: Palate Recommendation Score (SOLE AUTHORITATIVE DISPLAY)
                # CHECK: Display score only if it exists AND is calculated (not None, not just initialized)
                if display_match_score is not None and palate_score is not None:
                    # DISPLAY: Show the actual calculated score (even if 0, it's a real calculation)
                    # MOBILE-OPTIMIZED: Larger text, clearer verdict for in-shop quick glance
                    st.markdown(f"""
<div class="glass-card glow" style="text-align: center; padding: 32px 24px; margin: 20px 0; position: relative;">
    <p style="color: #A0A0A8; margin: 0 0 12px 0; font-size: clamp(10px, 2.5vw, 12px); text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">
        Palate Recommendation Score
    </p>
    <div class="match-score-gradient" style="font-size: clamp(60px, 15vw, 80px); margin: 0; font-family: 'Geist', 'Inter', sans-serif; line-height: 1;">
        {display_match_score:.1f}%
    </div>
    <p style="color: #E8E8EB; margin: 12px 0 0 0; font-size: clamp(14px, 4vw, 18px); font-weight: 600;">{palate_score.verdict}</p>
</div>
""", unsafe_allow_html=True)

                    # Glassmorphic Bento Card - Calculation Breakdown
                    st.markdown(f"""<div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;"><p style="color: #A0A0A8; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin: 0 0 1rem 0;">🔍 How This Score is Calculated</p><div style="margin-bottom: 1rem;"><p style="color: #E8E8EB; font-weight: 700; font-size: 14px; margin: 0 0 4px 0;">Flavor Alignment: <span style="color: #800020;">{palate_score.palate_match:.1f}%</span></p><p style="color: #A0A0A8; font-size: 12px; margin: 0; line-height: 1.5;">How similar this wine's flavor profile is to wines you've enjoyed</p></div><div style="margin-bottom: 1rem;"><p style="color: #E8E8EB; font-weight: 700; font-size: 14px; margin: 0 0 4px 0;">Statistical Confidence: <span style="color: #800020;">{palate_score.confidence_factor*100:.0f}%</span></p><p style="color: #A0A0A8; font-size: 12px; margin: 0; line-height: 1.5;">Based on {palate_score.n_samples} wine(s) in your tasting history</p></div><div style="background: rgba(128, 0, 32, 0.1); border-radius: 8px; padding: 12px; margin: 1rem 0;"><p style="color: #A0A0A8; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 8px 0;">Formula</p><p style="font-family: 'Monaco', 'Courier New', monospace; font-size: 16px; color: #E8E8EB; margin: 0; letter-spacing: 1px; font-weight: 600;">{palate_score.palate_match:.1f}% × {palate_score.confidence_factor*100:.0f}% = {display_match_score:.1f}%</p></div><p style="color: #A0A0A8; font-size: 11px; margin: 12px 0 0 0; line-height: 1.6;">💡 Your recommendation improves as you rate more wines. Add <strong style="color: #E8E8EB;">{max(0, 10 - palate_score.n_samples)} more wine(s)</strong> to reach 95%+ confidence.</p></div>""", unsafe_allow_html=True)
                else:
                    # LOADING STATE: Show "Calculating..." text instead of 0%
                    st.markdown("""
<div class="glass-card glow" style="text-align: center; padding: 40px 30px; margin: 24px 0;">
    <p style="color: #A0A0A8; margin: 0 0 16px 0; font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">Palate Recommendation Score</p>
    <div class="match-score-gradient" style="font-size: 48px; margin: 16px 0; font-family: 'Geist', 'Inter', sans-serif;">
        Calculating...
    </div>
    <p style="color: #A0A0A8; margin: 16px 0 0 0; font-size: 14px;">Analysing your palate profile</p>
</div>
""", unsafe_allow_html=True)

                # Add visual separator
                st.markdown("---")

                # 📋 CLEAN PROFESSIONAL PRESENTATION - 2-Column Layout
                st.markdown("### 📋 Wine Profile")

                eval_col1, eval_col2 = st.columns(2)

                # LEFT COLUMN: Style, Origin, Vintage
                with eval_col1:
                    st.markdown("**🍷 Style & Origin**")
                    # Vertical bulleted list format - clean hierarchy
                    st.markdown(f"- **Type:** {wine_color}")
                    st.markdown(f"- **Style:** {style_full}")
                    # Show Appellation with region hierarchy
                    if region != 'Unknown' and country != 'Unknown':
                        st.markdown(f"- **Appellation:** {region} ({country})")
                    elif region != 'Unknown':
                        st.markdown(f"- **Appellation:** {region}")
                    elif country != 'Unknown':
                        st.markdown(f"- **Origin:** {country}")
                    if should_display_vintage(wine_data.get('vintage')):
                        st.markdown(f"- **Vintage:** {int(wine_data.get('vintage'))}")
                    if wine_data.get('producer'):
                        st.markdown(f"- **Producer:** {wine_data.get('producer')}")

                # RIGHT COLUMN: Tasting Notes & Verdict
                with eval_col2:
                    st.markdown("**📝 Tasting Notes & Verdict**")
                    notes = wine_data.get('notes', 'No tasting notes available')

                    # Display full notes with natural wrapping (no truncation)
                    st.markdown(f"_{notes}_")

                    # Why you'll like it - 1 sentence verdict
                    st.markdown("")  # spacing
                    if display_match_score is not None:
                        # Use display_match_score (SINGLE SOURCE OF TRUTH)
                        if display_match_score >= 75:
                            why_like = f"**💙 Why you'll like it:** This matches your preferred {wine_color.lower()} style perfectly."
                        elif display_match_score >= 60:
                            why_like = f"**🧡 Why try it:** Good compatibility with your palate, worth exploring."
                        else:
                            why_like = f"**🟡 Different:** This is a departure from your usual {wine_color.lower()} wines."
                        st.markdown(why_like)

                st.markdown("---")
            else:
                st.info("🔍 Add wines to your collection to see palate match predictions")
                st.markdown("---")

            # 95% PRE-POPULATED "STORE MODE" UI
            st.markdown("### 💾 Store Mode - Quick Log")
            st.caption("AI extracted everything - only 3 inputs needed from you!")

            # OPTIMIZED FORM: 3 inputs in one clean row [Score, Price, Like-Toggle]
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Score (slider for quick input)
                score_input = st.slider(
                    "⭐ Your Score",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(wine_data.get('score', 7.5)),
                    step=0.5,
                    help="How would you rate this wine?"
                )

            with col2:
                # Price - moved from Technical Details for better UX
                price_input = st.number_input(
                    "💶 Price (€)",
                    min_value=0.0,
                    value=float(wine_data.get('price', 0.0)),
                    step=0.50,
                    help="Retail price in EUR"
                )

            with col3:
                # Liked (toggle with smart default based on UNIFIED score)
                # Uses ONLY display_match_score (SINGLE SOURCE OF TRUTH)
                if display_match_score is not None:
                    liked_default = display_match_score >= 65
                else:
                    # Fallback for truly empty history: neutral default
                    liked_default = (score_input >= 7.0)

                liked_input = st.toggle(
                    "❤️ Did You Like It?",
                    value=liked_default,
                    help="Would you buy this again?"
                )

            # Advanced details in expander (AI-extracted technical data)
            with st.expander("⚙️ Technical Details & Edit Data (Optional)"):
                st.markdown("#### 🎯 Flavor Profile (0-10 Scale)")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("⚡ Acidity", f"{wine_data['acidity']}/10")
                with col2:
                    st.metric("💎 Minerality", f"{wine_data['minerality']}/10")
                with col3:
                    st.metric("🍇 Fruitiness", f"{wine_data['fruitiness']}/10")
                with col4:
                    st.metric("🌰 Tannin", f"{wine_data['tannin']}/10")
                with col5:
                    st.metric("💪 Body", f"{wine_data['body']}/10")

                # Show explanations if features were inferred (not extracted from image)
                if feature_descriptions:
                    st.markdown("")
                    st.markdown("**📝 Characteristic Explanations:**")
                    st.markdown(f"• **Acidity ({wine_data['acidity']}/10)**: {feature_descriptions.get('acidity', 'N/A')}")
                    st.markdown(f"• **Fruitiness ({wine_data['fruitiness']}/10)**: {feature_descriptions.get('fruitiness', 'N/A')}")
                    st.markdown(f"• **Body ({wine_data['body']}/10)**: {feature_descriptions.get('body', 'N/A')}")
                    st.markdown(f"• **Minerality ({wine_data['minerality']}/10)**: {feature_descriptions.get('minerality', 'N/A')}")
                    st.markdown(f"• **Tannin ({wine_data['tannin']}/10)**: {feature_descriptions.get('tannin', 'N/A')}")

                st.markdown("---")

                st.markdown("#### 📊 Full Technical Specifications")
                tech_col1, tech_col2 = st.columns(2)
                with tech_col1:
                    st.markdown(f"**Wine Color:** {wine_data.get('wine_color', 'White')}")
                    st.markdown(f"**Sparkling:** {'Yes' if wine_data.get('is_sparkling', False) else 'No'}")
                    st.markdown(f"**Natural:** {'Yes' if wine_data.get('is_natural', False) else 'No'}")
                with tech_col2:
                    st.markdown(f"**Sweetness:** {wine_data.get('sweetness', 'Dry')}")
                    st.markdown(f"**Producer:** {wine_data.get('producer', 'Unknown')}")
                    if should_display_vintage(wine_data.get('vintage')):
                        st.markdown(f"**Vintage:** {int(wine_data.get('vintage'))}")
                    else:
                        st.markdown(f"**Vintage:** NV")

            # Large, prominent Save button
            if st.button("💾 SAVE TO MY COLLECTION", type="primary", width="stretch"):
                # Validate and update user inputs
                try:
                    # Type validation with high-dimensional attributes
                    wine_data['score'] = float(score_input)
                    wine_data['liked'] = bool(liked_input)  # Ensure boolean
                    wine_data['price'] = float(price_input)  # Price is now always in Quick Log

                    # Input validation - catch invalid data early
                    validation_errors = []

                    if not wine_data.get('wine_name') or wine_data['wine_name'].strip() == '':
                        validation_errors.append("Wine name is required")

                    if wine_data['score'] < 1 or wine_data['score'] > 10:
                        validation_errors.append(f"Score must be 1-10 (got {wine_data['score']})")

                    if wine_data['price'] < 0:
                        validation_errors.append(f"Price cannot be negative (got {wine_data['price']})")

                    # Validate flavor features (must be 1-10)
                    for feature in ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']:
                        value = wine_data.get(feature, 0)
                        if value < 1 or value > 10:
                            validation_errors.append(f"{feature.capitalize()} must be 1-10 (got {value})")

                    if validation_errors:
                        st.error(f"🚫 Cannot save wine - please fix these issues:\n" + "\n".join(f"• {err}" for err in validation_errors))
                        st.stop()

                    # Validate high-dimensional fields
                    wine_data['is_sparkling'] = bool(wine_data.get('is_sparkling', False))
                    wine_data['is_natural'] = bool(wine_data.get('is_natural', False))

                    # Save to PostgreSQL database
                    row_data = {
                        'user_id': username,  # Add logged-in user
                        'wine_name': wine_data['wine_name'],
                        'producer': wine_data['producer'],
                        'vintage': wine_data['vintage'],
                        'notes': wine_data['notes'],
                        'score': wine_data['score'],
                        'liked': wine_data['liked'],
                        'price': wine_data['price'],
                        # WINE ORIGIN
                        'country': wine_data.get('country', 'Unknown'),
                        'region': wine_data.get('region', 'Unknown'),
                        # HIGH-DIMENSIONAL ATTRIBUTES
                        'wine_color': wine_data.get('wine_color', 'White'),
                        'is_sparkling': wine_data['is_sparkling'],
                        'is_natural': wine_data['is_natural'],
                        'sweetness': wine_data.get('sweetness', 'Dry'),
                        # Core 5 flavor features
                        'acidity': wine_data['acidity'],
                        'minerality': wine_data['minerality'],
                        'fruitiness': wine_data['fruitiness'],
                        'tannin': wine_data['tannin'],
                        'body': wine_data['body']
                    }

                    try:
                        # Try saving to database first (with loading indicator)
                        with st.spinner("💾 Saving wine to database..."):
                            db.add_wine(row_data)
                        st.success("✅ Wine saved to database!")

                    except Exception as db_error:
                        # Fallback to CSV if database fails
                        st.warning(f"⚠️ Database temporarily unavailable, saving locally")

                        csv_path = Path("data/history.csv")
                        if csv_path.exists():
                            df_history = pd.read_csv(csv_path)
                            for col in row_data.keys():
                                if col not in df_history.columns:
                                    df_history[col] = None
                            df_history = pd.concat([df_history, pd.DataFrame([row_data])], ignore_index=True)
                        else:
                            df_history = pd.DataFrame([row_data])

                        df_history['liked'] = df_history['liked'].astype(bool)
                        df_history['price'] = df_history['price'].astype(float)
                        df_history['score'] = df_history['score'].astype(float)
                        df_history.to_csv(csv_path, index=False)

                        if IS_STREAMLIT_CLOUD:
                            st.warning(
                                "⚠️ **CSV Fallback on Streamlit Cloud**: Data will be reset when app restarts. "
                                "Please check your database connection."
                            )

                    # Sync features
                    import subprocess
                    sync_script = Path("scripts/sync_features.py")
                    if sync_script.exists():
                        result = subprocess.run([sys.executable, str(sync_script)], cwd=Path.cwd(), capture_output=True, text=True)
                        if result.returncode != 0:
                            st.warning(f"⚠️ Sync script error: {result.stderr}")
                        else:
                            st.info("🔄 Features synced successfully")

                    # Clear cached data to force reload
                    load_wine_data.clear()

                    st.success(f"✅ Saved {wine_data['wine_name']} to your collection!")
                    st.balloons()

                    # Clear session state to start fresh
                    if 'wine_data' in st.session_state:
                        del st.session_state['wine_data']
                    if 'last_upload' in st.session_state:
                        del st.session_state['last_upload']

                    st.info("🍷 Ready for next wine! Add another above.")

                except ValueError as e:
                    st.error(f"Validation error: {str(e)}")
                    st.info("Please check that price is a valid number and liked is true/false")
                except Exception as e:
                    st.error(f"Error saving: {str(e)}")
                    st.info("Check that data/history.csv exists and is writable")

            else:
                # No data extracted yet
                st.info("👆 Enter a wine name or upload a photo to get started")

    # 📊 TAB 2: Wine Cellar - Palate Maps
    with tab2:
        st.markdown("## 📊 My Palate Maps")
        st.caption("Your ideal flavor profiles by wine color")

        # Data persistence controls (Download/Upload)
        col_data1, col_data2 = st.columns([1, 1])
        with col_data1:
            # Download button
            csv_path = Path("data/history.csv")
            if csv_path.exists():
                history_csv = csv_path.read_text()
                st.download_button(
                    label="📥 Download My Collection (CSV)",
                    data=history_csv,
                    file_name="decant_wine_history.csv",
                    mime="text/csv",
                    help="Backup your wine collection. Essential on Streamlit Cloud free tier!"
                )
            else:
                st.info("No history data yet. Add wines first!")

        with col_data2:
            # Upload button (restore from backup)
            uploaded_file = st.file_uploader(
                "📤 Restore from Backup",
                type=['csv'],
                help="Upload a previously downloaded CSV to restore your collection",
                key='restore_history'
            )

            if uploaded_file is not None:
                try:
                    # Read uploaded CSV
                    uploaded_df = pd.read_csv(uploaded_file)

                    # Validate schema (check for required columns)
                    required_cols = ['wine_name', 'score', 'liked']
                    missing_cols = [col for col in required_cols if col not in uploaded_df.columns]

                    if missing_cols:
                        st.error(f"❌ Invalid CSV: Missing columns {missing_cols}")
                    else:
                        # Merge with existing data (avoid duplicates by wine_name + vintage)
                        csv_path = Path("data/history.csv")
                        if csv_path.exists():
                            existing_df = pd.read_csv(csv_path)

                            # Create unique key for deduplication
                            existing_df['_key'] = existing_df['wine_name'] + '_' + existing_df.get('vintage', 'NV').astype(str)
                            uploaded_df['_key'] = uploaded_df['wine_name'] + '_' + uploaded_df.get('vintage', 'NV').astype(str)

                            # Keep only new wines from uploaded file
                            new_wines = uploaded_df[~uploaded_df['_key'].isin(existing_df['_key'])]

                            if len(new_wines) > 0:
                                # Drop temporary key column
                                new_wines = new_wines.drop(columns=['_key'])
                                existing_df = existing_df.drop(columns=['_key'])

                                # Append new wines
                                merged_df = pd.concat([existing_df, new_wines], ignore_index=True)
                                merged_df.to_csv(csv_path, index=False)

                                st.success(f"✅ Restored {len(new_wines)} new wines! ({len(existing_df)} existing wines kept)")
                                load_wine_data.clear()  # Clear cache
                            else:
                                st.info("✅ No new wines to add. All uploaded wines already exist!")
                        else:
                            # No existing data, just save uploaded file
                            uploaded_df.to_csv(csv_path, index=False)
                            st.success(f"✅ Restored {len(uploaded_df)} wines!")
                            load_wine_data.clear()  # Clear cache

                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")

        st.markdown("---")

        # Load data
        history_df = load_wine_data(username)

        if history_df is not None and len(history_df) > 0:
            # Get only liked wines
            liked_wines = history_df[history_df['liked'] == True]

            if len(liked_wines) == 0:
                st.warning("No liked wines yet. Add wines and mark them as liked to see your palate maps!")
            else:
                # Calculate color profiles for consolidation
                colors = ['White', 'Red', 'Rosé', 'Orange']
                feature_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']

                color_profiles = {}
                color_counts = {}

                for wine_color in colors:
                    color_wines = liked_wines[liked_wines['wine_color'] == wine_color]

                    if len(color_wines) > 0:
                        # Calculate ideal profile (average of liked wines)
                        ideal_profile = color_wines[feature_cols].mean()
                        color_profiles[wine_color] = ideal_profile
                        color_counts[wine_color] = len(color_wines)

                # Create ONE consolidated Master Radar with all color profiles overlaid
                if len(color_profiles) > 0:
                    st.markdown("### 🎯 Consolidated Master Radar")
                    st.caption("All your wine color profiles overlaid in one high-contrast chart")

                    # Display wine counts
                    count_text = " | ".join([f"{color}: {count} wines" for color, count in color_counts.items()])
                    st.caption(f"📊 {count_text}")

                    # Create and display consolidated radar
                    consolidated_radar = create_consolidated_palate_radar(color_profiles)
                    st.plotly_chart(consolidated_radar, width="stretch", key='consolidated_master_radar')

                    st.markdown("---")

                    # Summary metrics by color (in expandable section)
                    with st.expander("📊 View Detailed Metrics by Color"):
                        for wine_color in colors:
                            if wine_color in color_profiles:
                                st.markdown(f"#### {wine_color} Wines")
                                ideal_profile = color_profiles[wine_color]

                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.metric("⚡ Acidity", f"{ideal_profile['acidity']:.1f}/10")
                                with col2:
                                    st.metric("💎 Minerality", f"{ideal_profile['minerality']:.1f}/10")
                                with col3:
                                    st.metric("🍇 Fruitiness", f"{ideal_profile['fruitiness']:.1f}/10")
                                with col4:
                                    st.metric("🌰 Tannin", f"{ideal_profile['tannin']:.1f}/10")
                                with col5:
                                    st.metric("💪 Body", f"{ideal_profile['body']:.1f}/10")

                                st.markdown("---")
        else:
            st.info("No wine data available. Add wines to see your palate maps!")

    with tab3:
        st.markdown("## 🖼️ Wine Gallery")
        st.caption("Browse your complete wine collection with all details")

        # Load data
        gallery_df = load_wine_data(username)

        if gallery_df is not None and len(gallery_df) > 0:
            # Add search and filter options
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_query = st.text_input("🔍 Search wines", placeholder="Search by name, producer, region...")
            with col2:
                color_filter = st.selectbox("Filter by color", ["All"] + ["White", "Red", "Rosé", "Orange"])
            with col3:
                liked_filter = st.selectbox("Filter by preference", ["All", "Liked", "Not Liked"])

            # Apply filters
            filtered_df = gallery_df.copy()

            if search_query:
                search_lower = search_query.lower()
                filtered_df = filtered_df[
                    filtered_df['wine_name'].str.lower().str.contains(search_lower, na=False) |
                    filtered_df['producer'].str.lower().str.contains(search_lower, na=False) |
                    filtered_df['region'].str.lower().str.contains(search_lower, na=False)
                ]

            if color_filter != "All":
                filtered_df = filtered_df[filtered_df['wine_color'] == color_filter]

            if liked_filter == "Liked":
                filtered_df = filtered_df[filtered_df['liked'] == True]
            elif liked_filter == "Not Liked":
                filtered_df = filtered_df[filtered_df['liked'] == False]

            # Sort by score descending
            filtered_df = filtered_df.sort_values('score', ascending=False)

            st.markdown(f"### Found {len(filtered_df)} wines")

            # CSS to remove phantom rows between Streamlit columns
            st.markdown("""
<style>
/* Remove gaps between Streamlit column containers */
.block-container [data-testid="column"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.block-container .element-container {
    margin-bottom: 0 !important;
}

/* Wine Gallery Grid */
.wine-gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    grid-auto-rows: min-content;
    gap: 2.5rem;
    margin: 24px 0;
}

.wine-card-notes {
    display: -webkit-box;
    -webkit-line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
    color: #A0A0A8;
    font-size: 13px;
    line-height: 1.5;
    margin: 8px 0;
}

.icon-row {
    min-height: 24px;
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 8px 0;
}

.wine-card-footer {
    margin-top: auto;
}
</style>
""", unsafe_allow_html=True)

            # Render wine cards in a grid using dynamic columns
            # Create columns in batches to avoid empty boxes
            wines_list = list(filtered_df.iterrows())
            num_wines = len(wines_list)
            num_cols = 3  # Cards per row

            for batch_start in range(0, num_wines, num_cols):
                batch_end = min(batch_start + num_cols, num_wines)
                batch_size = batch_end - batch_start

                # Only create as many columns as we have wines in this batch
                # Use equal widths to prevent empty boxes
                cols = st.columns([1] * batch_size, gap="medium")

                for col_idx in range(batch_size):
                    wine_idx = batch_start + col_idx
                    _, wine = wines_list[wine_idx]

                    with cols[col_idx]:
                        wine_name = wine.get('wine_name', 'Unknown')

                        # Check for existing image
                        image_path = get_wine_image_path(wine_name)

                        # Wrap in glass-card
                        st.markdown('<div class="glass-card wine-card">', unsafe_allow_html=True)

                        # Image section
                        if image_path and Path(image_path).exists():
                            import base64
                            with open(image_path, "rb") as img_file:
                                img_data = base64.b64encode(img_file.read()).decode()
                                img_ext = image_path.split('.')[-1]
                                st.markdown(f'<img src="data:image/{img_ext};base64,{img_data}" class="wine-card-img" />', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="wine-card-img-placeholder">🍷</div>', unsafe_allow_html=True)

                        # ROW 1: Wine name
                        st.markdown(f"<h4 style='margin: 12px 0 4px 0; font-size: 16px;'>{wine_name[:40]}</h4>", unsafe_allow_html=True)

                        # ROW 2: Winery (smaller font)
                        vintage_display = f" • {int(wine.get('vintage'))}" if should_display_vintage(wine.get('vintage')) else ""
                        st.markdown(f"<p style='font-size: 13px; color: #A0A0A8; margin: 0 0 4px 0;'>{wine.get('producer', 'Unknown')[:30]}{vintage_display}</p>", unsafe_allow_html=True)

                        # ROW 3: Country (region)
                        st.markdown(f"<p style='font-size: 12px; color: #A0A0A8; margin: 0 0 8px 0;'>📍 {wine.get('region', wine.get('country', 'Unknown'))[:35]}</p>", unsafe_allow_html=True)

                        # ROW 4: Score (left) | Price (right)
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Score", f"{wine.get('score', 0):.1f}/10")
                        with m2:
                            st.metric("Price", f"€{wine.get('price', 0):.0f}")

                        # ROW 5: Icons
                        active_icons = []
                        if wine.get('liked'):
                            active_icons.append('<span class="badge" style="font-size: 16px;">❤️</span>')
                        if wine.get('is_sparkling'):
                            active_icons.append('<span class="badge" style="font-size: 16px;">✨</span>')
                        if wine.get('is_natural'):
                            active_icons.append('<span class="badge" style="font-size: 16px;">🌱</span>')

                        icons_content = " ".join(active_icons) if active_icons else "&nbsp;"
                        st.markdown(f'<div class="icon-row" style="min-height: 24px; margin: 8px 0;">{icons_content}</div>', unsafe_allow_html=True)

                        # ROW 6: Description (expandable)
                        notes = wine.get('notes', '')
                        if notes:
                            with st.expander("📝 Tasting Notes"):
                                st.markdown(f"_{notes}_")

                        # Upload section
                        with st.expander("📸 Upload Photo"):
                            uploaded_image = st.file_uploader(
                                "Choose bottle photo",
                                type=['jpg', 'jpeg', 'png', 'webp'],
                                key=f"upload_{wine_name}_{wine_idx}",
                                label_visibility="collapsed"
                            )

                            if uploaded_image:
                                if st.button("💾 Save Photo", key=f"save_{wine_name}_{wine_idx}"):
                                    saved_path = save_wine_image(uploaded_image, wine_name)
                                    if saved_path:
                                        st.success("✓ Photo saved!")
                                        st.rerun()

                            vivino_url = get_wine_image_url(wine_name, wine.get('producer', ''))
                            st.markdown(f"[🔍 Find on Vivino]({vivino_url})")

                        st.markdown('</div>', unsafe_allow_html=True)  # Close wine-card
        else:
            st.info("No wines in your collection yet. Add wines to see them here!")


if __name__ == "__main__":
    main()
