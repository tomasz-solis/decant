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
from pydantic import ValidationError
from decant.supabase_session import (
    current_user_email,
    get_anon_supabase,
    get_supabase_client,
    get_user_supabase,
    is_authenticated,
)
from decant.ui.auth_form import render_header_auth
from decant.ui.styles import apply_global_styles
from decant.services.data_access import normalize as ensure_wine_df
from decant.ui import tab_add_wine, tab_gallery, tab_palate_maps, tab_stats
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

# Apply the main app theme. Lives in decant.ui.styles so this file
# stays focused on app logic, not visual presentation.
apply_global_styles()


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

    # Four tabs always created so Streamlit tab indexing stays stable.
    # Tab 1 (Add Wine) content is gated inside the tab body — anonymous
    # users see a sign-in nudge inside the tab rather than a hidden tab.
    # This avoids the AttributeError that would come from `with None:` if we
    # tried to skip tab1 entirely, and keeps the layout consistent.
    tab1, tab2, tab3, tab4 = st.tabs([
        "🍷 Add Wine",
        "📊 My Palate Maps",
        "🏆 Stats",
        "🖼️ Wine Gallery",
    ])

    # All four tabs are now thin dispatch calls — bodies live in
    # decant.ui.tab_*. history_df is loaded once per tab to stay
    # conservative about post-write freshness; load_wine_data is
    # cached at the function level so the redundancy is cheap.
    with tab1:
        tab_add_wine.render(
            history_df=ensure_wine_df(load_wine_data()),
            predictor=load_predictor(),
            client=client,
            is_authenticated_now=is_authenticated(),
        )

    with tab2:
        tab_palate_maps.render(
            history_df=ensure_wine_df(load_wine_data()),
            is_guest=is_guest,
        )

    with tab3:
        tab_stats.render(
            history_df=ensure_wine_df(load_wine_data()),
            debug_mode=DEBUG_MODE,
        )

    with tab4:
        tab_gallery.render(history_df=ensure_wine_df(load_wine_data()))


if __name__ == "__main__":
    main()