"""
Decant - Taste, with confidence.
A Streamlit app for wine analytics and personalized recommendations using In-Context Learning.
"""

import sys
import os
import base64
import json
from pathlib import Path
from typing import Optional

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
from pydantic import ValidationError
from decant.supabase_session import (
    current_user_email,
    get_anon_supabase,
    get_supabase_client,
    get_user_supabase,
    is_authenticated,
)
from decant.ui.auth_form import render_header_auth
from decant.ui.components import (
    calculate_similarity,
    create_consolidated_palate_radar,
    create_decision_boundary_plot,
    create_master_radar,
    create_mini_radar_chart,
    create_radar_chart,
)
from decant.services.image_storage import (
    get_wine_image_path,
    get_wine_image_url,
    save_wine_image,
)
from decant.services.vision_extract import extract_complete_wine_data
from decant.services.data_access import normalize as ensure_wine_df
from decant.wines_repo import list_wines as repo_list_wines, repo_add_wine

# Load environment variables
load_dotenv()


def check_required_supabase_secrets() -> None:
    """Fail fast when required Supabase secrets are missing.

    Phase 2: only four keys are required at startup. No section headers.
    The household account credentials are entered by the user at sign-in
    time, not stored in TOML.
    """
    if st.session_state.get("_supabase_startup_checked"):
        return

    required_keys = ["SUPABASE_URL", "SUPABASE_KEY", "CELLAR_ID", "OPENAI_API_KEY"]
    missing = []
    for key in required_keys:
        try:
            value = st.secrets[key]
        except (FileNotFoundError, KeyError):
            value = None
        if value is None or str(value).strip() == "":
            missing.append(key)

    if missing:
        st.error("❌ Missing required secret(s): " + ", ".join(missing))
        st.stop()

    st.session_state["_supabase_startup_checked"] = True


def is_debug_enabled() -> bool:
    """Return debug mode from secrets."""
    try:
        debug_value = st.secrets.get("DEBUG", False)
    except (FileNotFoundError, KeyError, AttributeError):
        return False

    if isinstance(debug_value, str):
        return debug_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(debug_value)


# Authentication: anonymous browsing is allowed by default. Sign-in is
# rendered via the top-right popover (see render_header_auth) and
# unlocks Tab 1 + any RLS-protected operation. There is no
# Streamlit-side username concept anymore — `is_authenticated()`
# checks the Supabase session.
check_required_supabase_secrets()
is_guest = not is_authenticated()
DEBUG_MODE = is_debug_enabled()

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
    initial_sidebar_state="auto"
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

    /* Responsive Breakpoints - Mobile */
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

        /* Larger buttons for touch targets */
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

        /* Touch targets (44px minimum per Apple HIG) */
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

        /* Columns stack vertically on mobile */
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
def get_predictor():
    """Build the VinoPredictor once per session.

    The expensive parts (OpenAI client, rate limiter) are cached.
    History is *not* cached here — call `refresh_predictor(df)` after
    loading wines to point the predictor at the current data.
    """
    try:
        return VinoPredictor(history_df=pd.DataFrame())
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        st.info("💡 Make sure OPENAI_API_KEY is set in your .env file")
        return None


def load_predictor(history_df: Optional[pd.DataFrame] = None):
    """Return a ready-to-use predictor with current history loaded.

    Kept for backward compatibility with existing call sites. New code
    should call `get_predictor()` directly and pass the DataFrame.
    """
    predictor = get_predictor()
    if predictor is not None and history_df is not None:
        predictor.refresh_context(history_df)
    return predictor



def load_wine_data():
    """Load wine data for the shared cellar.

    Uses the authenticated client when the user is signed in (write-capable
    session, RLS sees the user), or the anon client for guests (read-only,
    RLS must allow anon SELECT on the wines table).
    """
    try:
        if is_authenticated():
            sb = get_user_supabase()
        else:
            sb = get_anon_supabase()
        df = repo_list_wines(sb)
        return ensure_wine_df(df)
    except Exception as e:
        st.session_state.pop("_wine_df_empty_debug", None)
        st.error(f"❌ Supabase error while loading wines: {e}")
        return ensure_wine_df(None)


def clear_wine_data_cache() -> None:
    """Clear cached wine data when cache decorators are enabled."""
    clear_fn = getattr(load_wine_data, "clear", None)
    if callable(clear_fn):
        clear_fn()


def show_empty_data_diagnostics() -> None:
    """Show actionable hints when no wines are returned."""
    diagnostics = st.session_state.get("_wine_df_empty_debug")
    if not diagnostics:
        return

    configured_cellar_id = diagnostics.get("configured_cellar_id")
    accessible_cellar_ids = diagnostics.get("accessible_cellar_ids") or []
    probe_error = diagnostics.get("probe_error")

    if accessible_cellar_ids and configured_cellar_id not in accessible_cellar_ids:
        st.warning(
            "No rows matched the configured `CELLAR_ID`. "
            "Update `CELLAR_ID` in Streamlit Cloud secrets to one of the accessible values below."
        )
        st.code("\n".join(accessible_cellar_ids), language="text")
        return

    if probe_error:
        st.caption(f"Debug hint: unable to inspect accessible cellar IDs ({probe_error}).")


def should_display_vintage(vintage_value):
    """Check if vintage should be displayed (not 0, None, or NaN)."""
    if vintage_value is None or pd.isna(vintage_value):
        return False
    try:
        vintage_int = int(vintage_value)
        return vintage_int > 0 and vintage_int < 2100
    except (ValueError, TypeError):
        return False



def main():
    # Top header: title on the left, auth popover top-right.
    # The columns sit close so the layout reads as one header band,
    # not "title floating in space + button stranded on the right."
    contact_email = st.secrets.get("CONTACT_EMAIL", "tomasz.solis@gmail.com")
    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            "<h1 class='main-title' style='margin: 0 0 4px 0;'>🍷 Decant</h1>"
            "<p class='subtitle' style='margin: 0;'>Taste, with confidence.</p>",
            unsafe_allow_html=True,
        )
    with header_right:
        render_header_auth(contact_email=str(contact_email))

    st.markdown("---")

    # Guest mode banner
    if is_guest:
        st.info("👀 **Guest mode** — You can browse the collection. Log in to add wines.")

    # Streamlit Cloud deployment warning (persistent at top)
    if IS_STREAMLIT_CLOUD:
        st.info(
            "ℹ️ **Running on Streamlit Cloud**: Your tasting history is stored in a CSV file. "
            "On the free tier, data will reset when the app restarts. "
            "Use the 📥 Download button in Analytics to backup your collection regularly."
        )

    # Three tabs always created so Streamlit tab indexing stays stable.
    # Tab 1 (Add Wine) content is gated below by is_authenticated() — anonymous
    # users see a sign-in nudge inside the tab rather than a hidden tab.
    # This avoids the AttributeError that would come from `with None:` if we
    # tried to skip tab1 entirely, and keeps the layout consistent.
    tab1, tab2, tab3, tab4 = st.tabs([
        "🍷 Add Wine",
        "📊 My Palate Maps",
        "🏆 Stats",
        "🖼️ Wine Gallery",
    ])

    # 🍷 TAB 1: Add Wine — content is gated by is_authenticated() below.
    # We always create 3 tabs so Streamlit's tab indexing stays stable;
    # if the user isn't signed in, Tab 1 just shows a sign-in nudge.
    with tab1:
        # Auth gate: anonymous users see a sign-in nudge instead of the add UI.
        # This closes the OpenAI abuse vector — no Vision API or extraction
        # calls are reachable without a signed-in session.
        if not is_authenticated():
            st.markdown("### 🍷 Add Wine to Collection")
            st.info(
                "Sign in to add wines and use the AI extraction feature. "
                "Browsing the gallery and palate maps doesn't require an account."
            )
            st.caption("Use the **Sign in** button at the top right.")
        else:
            st.markdown("### 🍷 Add Wine to Collection")
            st.caption("Enter wine name or upload a photo - AI extracts everything else")

            # Load history for self-learning context
            history_df = ensure_wine_df(load_wine_data())

            # Input mode selection
            input_mode = st.radio(
                "Input Method",
                ["📝 Enter Wine Name", "📸 Upload Photo"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if input_mode == "📝 Enter Wine Name":
                # Text input mode
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

                            # Convert to dict
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
                # Photo upload mode
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
                            wine_data = extract_complete_wine_data(uploaded_file, history_df, client)

                            if wine_data:
                                st.session_state['wine_data'] = wine_data
                                st.session_state['last_upload'] = uploaded_file.name
                                # Store raw file bytes so we can save the photo later
                                uploaded_file.seek(0)
                                st.session_state['uploaded_photo_bytes'] = uploaded_file.read()
                                st.session_state['uploaded_photo_name'] = uploaded_file.name
                                st.success("✅ Wine analyzed! All fields extracted automatically")
                                st.rerun()

            # Show extracted data if available
            if 'wine_data' in st.session_state:
                wine_data = st.session_state['wine_data']

                # Display wine name prominently with geography
                st.markdown(f"## 🍷 {wine_data['wine_name']}")

                # Location header with NaN-safe fallbacks
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

                # Style header
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
                    # Reuse the cached predictor and point it at the latest history.
                    # history_df is already loaded by the caller (Supabase via load_history).
                    predictor = load_predictor(history_df=history_df)

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

                # Large, prominent Save button (login required)
                if is_guest:
                    st.warning("🔒 Log in to save wines to your collection")

                if st.button("💾 SAVE TO MY COLLECTION", type="primary", width="stretch", disabled=is_guest):
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

                        # Save to Supabase wines table (RLS-authenticated session)
                        row_data = {
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
                            with st.spinner("💾 Saving wine to Supabase..."):
                                repo_add_wine(get_user_supabase(), row_data)
                            st.success("✅ Wine saved to Supabase!")
                        except Exception as supabase_error:
                            st.error(f"❌ Supabase error while saving wine: {supabase_error}")
                            st.stop()

                        # Save uploaded photo if available
                        photo_bytes = st.session_state.get('uploaded_photo_bytes')
                        photo_name = st.session_state.get('uploaded_photo_name')
                        if photo_bytes and wine_data.get('wine_name'):
                            import io
                            photo_file = io.BytesIO(photo_bytes)
                            photo_file.name = photo_name or "photo.jpg"
                            saved_path = save_wine_image(photo_file, wine_data['wine_name'])
                            if saved_path:
                                st.info("📸 Photo saved")

                        # Clear cached data to force reload
                        clear_wine_data_cache()

                        st.success(f"✅ Saved {wine_data['wine_name']} to your collection!")
                        st.balloons()

                        # Clear session state to start fresh
                        for key in ['wine_data', 'last_upload', 'uploaded_photo_bytes', 'uploaded_photo_name']:
                            st.session_state.pop(key, None)

                        st.info("🍷 Ready for next wine! Add another above.")

                    except ValueError as e:
                        st.error(f"Validation error: {str(e)}")
                        st.info("Please check that price is a valid number and liked is true/false")
                    except Exception as e:
                        st.error(f"Error saving: {str(e)}")
                        st.info("Check Supabase configuration and RLS permissions")

                else:
                    # No data extracted yet
                    st.info("👆 Enter a wine name or upload a photo to get started")

    # 📊 TAB 2: Wine Cellar - Palate Maps
    with tab2:
        st.markdown("## 📊 My Palate Maps")
        st.caption("Your ideal flavor profiles by wine color")


        # Load data
        history_df = ensure_wine_df(load_wine_data())

        if not history_df.empty:
            # Get only liked wines
            if 'liked' in history_df.columns:
                liked_wines = history_df[history_df['liked'] == True]
            else:
                liked_wines = history_df.iloc[0:0].copy()

            if liked_wines.empty:
                st.warning("No liked wines yet. Add wines and mark them as liked to see your palate maps!")
            else:
                # Calculate color profiles for consolidation
                colors = ['White', 'Red', 'Rosé', 'Orange']
                feature_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']
                missing_feature_cols = [col for col in feature_cols if col not in liked_wines.columns]

                color_profiles = {}
                color_counts = {}

                if 'wine_color' not in liked_wines.columns:
                    st.caption("Missing fields for palate maps: wine_color")
                elif missing_feature_cols:
                    st.caption(f"Missing fields for palate maps: {', '.join(missing_feature_cols)}")
                else:
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

        # Data persistence controls (Download/Upload)
        col_data1, col_data2 = st.columns([1, 1])
        with col_data1:
            # Download from Supabase data already loaded
            tab2_df = ensure_wine_df(load_wine_data())
            if not tab2_df.empty:
                csv_data = tab2_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download My Collection (CSV)",
                    data=csv_data,
                    file_name="decant_wine_history.csv",
                    mime="text/csv",
                    help="Backup your wine collection"
                )
            else:
                st.info("No history data yet. Add wines first!")

        with col_data2:
            if is_guest:
                st.info("🔒 Log in to restore from backup")
            else:
                # Upload button (restore from backup)
                uploaded_file = st.file_uploader(
                    "📤 Restore from Backup",
                    type=['csv'],
                    help="Upload a previously downloaded CSV to restore your collection",
                    key='restore_history'
                )

                if uploaded_file is not None:
                    try:
                        uploaded_df = pd.read_csv(uploaded_file)

                        required_cols = ['wine_name', 'score', 'liked']
                        missing_cols = [col for col in required_cols if col not in uploaded_df.columns]

                        if missing_cols:
                            st.error(f"❌ Invalid CSV: Missing columns {missing_cols}")
                        else:
                            # Dedup against existing Supabase data
                            existing_df = ensure_wine_df(load_wine_data())

                            if not existing_df.empty:
                                existing_keys = (
                                    existing_df['wine_name'].astype(str) + '_'
                                    + existing_df['vintage'].fillna('NV').astype(str)
                                )
                                uploaded_keys = (
                                    uploaded_df['wine_name'].astype(str) + '_'
                                    + uploaded_df['vintage'].fillna('NV').astype(str) if 'vintage' in uploaded_df.columns
                                    else uploaded_df['wine_name'].astype(str) + '_NV'
                                )
                                new_wines = uploaded_df[~uploaded_keys.isin(existing_keys)]
                            else:
                                new_wines = uploaded_df

                            if len(new_wines) > 0:
                                sb = get_user_supabase()
                                imported = 0
                                for _, row in new_wines.iterrows():
                                    try:
                                        row_data = row.dropna().to_dict()
                                        repo_add_wine(sb, row_data)
                                        imported += 1
                                    except Exception as row_err:
                                        st.warning(f"⚠️ Skipped {row.get('wine_name', '?')}: {row_err}")
                                st.success(f"✅ Imported {imported} new wines!")
                                clear_wine_data_cache()
                            else:
                                st.info("✅ No new wines to add. All uploaded wines already exist!")

                    except Exception as e:
                        st.error(f"❌ Error reading CSV: {str(e)}")

        st.markdown("---")

    with tab3:
        st.markdown("## 🏆 Stats")
        st.caption("Your collection at a glance")

        # Load history fresh for this tab — cached at the load_wine_data level.
        df = ensure_wine_df(load_wine_data())

        # 🌍 REGIONAL FILTER DROPDOWN
        if not df.empty and 'region' in df.columns:
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

                if selected_region != "All Regions":
                    df = df[df['region'] == selected_region]
                    st.caption(f"Showing: {selected_region}")

        st.markdown("---")

        # Headline metrics (Liked / Disliked / Total).
        total_wines = len(df)
        has_liked_col = 'liked' in df.columns
        liked_wines = int(df['liked'].sum()) if has_liked_col else 0
        disliked_wines = max(total_wines - liked_wines, 0)
        liked_df = df[df['liked'] == True] if has_liked_col else df.iloc[0:0].copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Liked", liked_wines)
        with col2:
            st.metric("❌ Disliked", disliked_wines)
        with col3:
            st.metric("📝 Total", total_wines)

        # --- Palate Stats (your ideal flavour numbers) ---
        st.markdown("---")
        st.markdown("### 🧬 Palate Stats")

        feature_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']
        missing_feature_cols = [c for c in feature_cols if c not in liked_df.columns]

        if liked_df.empty:
            st.caption("🔍 Add wines with flavor profiles to see your palate stats")
        elif missing_feature_cols:
            st.caption(f"Missing fields for palate stats: {', '.join(missing_feature_cols)}")
        else:
            liked_avg = liked_df[feature_cols].mean()
            if liked_avg.sum() == 0:
                st.caption("🔍 Add wines with flavor profiles to see your palate stats")
            else:
                st.caption("Your ideal wine profile:")
                f1, f2, f3, f4, f5 = st.columns(5)
                with f1:
                    st.metric("⚡ Acid", f"{liked_avg['acidity']:.1f}")
                with f2:
                    st.metric("💎 Mineral", f"{liked_avg['minerality']:.1f}")
                with f3:
                    st.metric("🍇 Fruit", f"{liked_avg['fruitiness']:.1f}")
                with f4:
                    st.metric("🌰 Tannin", f"{liked_avg['tannin']:.1f}")
                with f5:
                    st.metric("💪 Body", f"{liked_avg['body']:.1f}")

        # --- Top Regions (top 3 by average score) ---
        st.markdown("---")
        st.markdown("### 🌍 Top Regions")

        regional_required_cols = ['region', 'country', 'score', 'wine_name']
        missing_regional_cols = [c for c in regional_required_cols if c not in liked_df.columns]

        if liked_df.empty:
            st.caption("Log wines with regions to see analytics")
        elif missing_regional_cols:
            st.caption(f"Missing fields for regional analytics: {', '.join(missing_regional_cols)}")
        else:
            regional_wines = liked_df[
                (liked_df['region'] != 'Unknown') &
                (liked_df['region'].notna())
            ]
            if len(regional_wines) > 0:
                regional_stats = regional_wines.groupby('region').agg({
                    'score': 'mean',
                    'wine_name': 'count'
                }).round(1)
                regional_stats.columns = ['avg_score', 'count']
                regional_stats = regional_stats.sort_values('avg_score', ascending=False)
                for idx, (region, stats) in enumerate(regional_stats.head(3).iterrows()):
                    medal = {0: '🥇', 1: '🥈', 2: '🥉'}.get(idx, f"#{idx + 1}")
                    rcol1, rcol2, rcol3 = st.columns([1, 5, 2])
                    with rcol1:
                        st.markdown(f"### {medal}")
                    with rcol2:
                        st.markdown(f"**{region}**")
                        st.caption(f"{int(stats['count'])} wines")
                    with rcol3:
                        st.metric("Avg score", f"{stats['avg_score']:.1f}/10")
            else:
                st.caption("No regional data yet")

        # --- Top Wines (top 3 by score) ---
        st.markdown("---")
        st.markdown("### 🍷 Top Wines")

        top_wines_df = liked_df if not liked_df.empty else df
        required_for_top = {'wine_name', 'score'}
        if top_wines_df.empty:
            st.caption("Add and rate wines to see your top picks.")
        elif not required_for_top.issubset(top_wines_df.columns):
            st.caption("Score column missing — can't rank wines yet.")
        else:
            top3 = top_wines_df.sort_values('score', ascending=False).head(3)
            for rank, (_, wine) in enumerate(top3.iterrows(), start=1):
                producer = wine.get('producer', '')
                vintage = wine.get('vintage')
                year = f" {int(vintage)}" if vintage and not pd.isna(vintage) and vintage > 0 else ""
                medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, f"#{rank}")
                wcol1, wcol2, wcol3 = st.columns([1, 6, 2])
                with wcol1:
                    st.markdown(f"### {medal}")
                with wcol2:
                    st.markdown(f"**{wine['wine_name']}**{year}")
                    if producer:
                        st.caption(producer)
                with wcol3:
                    st.metric("Score", f"{wine['score']:.1f}/10")

        # --- Debug (gated by DEBUG_MODE) ---
        if DEBUG_MODE:
            st.markdown("---")
            st.markdown("### 🔍 Debug Data")
            st.caption(f"Shape: {df.shape}")
            st.caption(f"Columns: {list(df.columns)}")
            missing_liked_debug = st.session_state.get("_wine_df_missing_liked_debug")
            if missing_liked_debug:
                st.caption(
                    "Loaded wines missing 'liked' column. "
                    f"rows type: {missing_liked_debug.get('rows_type')}"
                )
                st.caption(f"Source columns: {missing_liked_debug.get('columns')}")
            preview_rows = min(3, len(df))
            if preview_rows > 0:
                st.dataframe(df.head(preview_rows), width="stretch")
            else:
                st.caption("No rows to preview")

    with tab4:
        st.markdown("## 🖼️ Wine Gallery")
        st.caption("Browse your complete wine collection with all details")

        # Load data
        gallery_df = ensure_wine_df(load_wine_data())

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
            show_empty_data_diagnostics()


if __name__ == "__main__":
    main()