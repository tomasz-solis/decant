"""
Decant - Taste, with confidence.
A Streamlit app for wine analytics and personalized recommendations using In-Context Learning.
"""

import sys
import os
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from decant import VinoPredictor
from decant.supabase_session import (
    get_anon_supabase,
    get_user_supabase,
    is_authenticated,
)
from decant.constants import FilePaths
from decant.ui.auth_form import render_header_auth
from decant.ui.editorial import render_cellar_snapshot
from decant.ui.styles import apply_global_styles
from decant.services.data_access import normalize as ensure_wine_df
from decant.ui import tab_add_wine, tab_gallery, tab_palate_maps, tab_stats
from decant.wines_repo import list_wines as repo_list_wines

# Load environment variables
load_dotenv()


def is_streamlit_cloud() -> bool:
    """Return True when running on Streamlit Cloud."""
    return (
        os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"
        or os.getenv("STREAMLIT_SHARING_MODE") is not None
    )


def read_secret(key: str) -> Optional[str]:
    """Read a Streamlit secret as a non-empty string, if present."""
    try:
        value = st.secrets[key]
    except (FileNotFoundError, KeyError):
        return None

    text = str(value).strip()
    return text or None


def check_required_supabase_secrets() -> None:
    """Check required secrets and enable local preview mode if absent.

    Streamlit Cloud remains strict because production should never boot
    with partial credentials. Local runs fall back to the bundled CSV so
    the read-only app can still be opened without secrets.
    """
    if st.session_state.get("_supabase_startup_checked"):
        return

    supabase_keys = ["SUPABASE_URL", "SUPABASE_KEY", "CELLAR_ID"]
    cloud_required_keys = supabase_keys + ["OPENAI_API_KEY"]
    missing_cloud_keys = [
        key for key in cloud_required_keys if read_secret(key) is None
    ]
    missing_supabase_keys = [key for key in supabase_keys if read_secret(key) is None]

    if missing_cloud_keys:
        if is_streamlit_cloud():
            st.error("Missing required secret(s): " + ", ".join(missing_cloud_keys))
            st.stop()

    if missing_supabase_keys:
        st.session_state["_decant_local_preview_missing_secrets"] = missing_supabase_keys
    else:
        st.session_state.pop("_decant_local_preview_missing_secrets", None)

    st.session_state["_supabase_startup_checked"] = True


def get_openai_api_key() -> Optional[str]:
    """Return the OpenAI key from Streamlit secrets or environment."""
    return read_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def is_debug_enabled() -> bool:
    """Return debug mode from secrets."""
    debug_value = read_secret("DEBUG")
    if debug_value is None:
        return False

    return debug_value.strip().lower() in {"1", "true", "yes", "on"}


# Authentication: anonymous browsing is allowed by default. Sign-in is
# rendered via the top-right popover (see render_header_auth) and
# unlocks Tab 1 + any RLS-protected operation. There is no
# Streamlit-side username concept anymore - `is_authenticated()`
# checks the Supabase session.
check_required_supabase_secrets()
is_guest = not is_authenticated()
DEBUG_MODE = is_debug_enabled()

# Detect Streamlit Cloud environment
IS_STREAMLIT_CLOUD = is_streamlit_cloud()

# Initialize OpenAI client for Vision API
# For Streamlit Cloud: use st.secrets, fallback to env vars for local dev
api_key = get_openai_api_key()
if api_key:
    # VinoPredictor still reads from the environment. Mirroring the
    # Streamlit secret here keeps local secrets.toml and Streamlit Cloud
    # secrets working without requiring a second .env entry.
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)
else:
    client = None

# Page configuration
st.set_page_config(
    page_title="Decant - Taste, with confidence",

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
    History is *not* cached here - call `refresh_predictor(df)` after
    loading wines to point the predictor at the current data.
    """
    try:
        return VinoPredictor(history_df=pd.DataFrame())
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        st.info("Make sure OPENAI_API_KEY is set in your .env file")
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
    missing_secrets = st.session_state.get("_decant_local_preview_missing_secrets")
    if missing_secrets:
        st.session_state["_wine_data_source"] = "local_missing_secrets"
        return load_local_wine_data()

    try:
        if is_authenticated():
            sb = get_user_supabase()
        else:
            sb = get_anon_supabase()
        df = repo_list_wines(sb)
        st.session_state["_wine_data_source"] = "supabase"
        return ensure_wine_df(df)
    except Exception as e:
        st.session_state.pop("_wine_df_empty_debug", None)
        fallback_df = load_local_wine_data()
        if not fallback_df.empty:
            st.session_state["_wine_data_source"] = "local_supabase_error"
            st.session_state["_wine_data_source_error"] = str(e)
            return fallback_df

        st.session_state["_wine_data_source"] = "empty_supabase_error"
        st.error(f"Supabase error while loading wines: {e}")
        return ensure_wine_df(None)


def load_local_wine_data() -> pd.DataFrame:
    """Load bundled CSV data for local read-only preview mode."""
    csv_path = Path(FilePaths.HISTORY_CSV)
    if not csv_path.exists():
        return ensure_wine_df(None)

    try:
        return ensure_wine_df(pd.read_csv(csv_path))
    except Exception as e:
        st.session_state["_wine_data_source_error"] = str(e)
        return ensure_wine_df(None)


def clear_wine_data_cache() -> None:
    """Clear cached wine data when cache decorators are enabled."""
    clear_fn = getattr(load_wine_data, "clear", None)
    if callable(clear_fn):
        clear_fn()


def main():
    # Top header: title on the left, auth popover top-right, with the
    # cellar hero photo as the band's background (styled in
    # decant.ui.styles via the :has(.app-masthead) block). The columns
    # sit close so the layout reads as one header band, not "title
    # floating in space + button stranded on the right."
    contact_email = read_secret("CONTACT_EMAIL") or "tomasz.solis@gmail.com"
    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            "<div class='app-masthead'>"
            "<p class='masthead-kicker'>Private cellar journal</p>"
            "<h1 class='main-title'>Decant</h1>"
            "<p class='subtitle'>Taste, with confidence.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with header_right:
        render_header_auth(contact_email=str(contact_email))

    # Guest mode banner
    if is_guest:
        st.info("**Guest mode** - You can browse the collection. Log in to add wines.")

    # Streamlit Cloud deployment warning (persistent at top)
    if IS_STREAMLIT_CLOUD:
        st.info(
            "**Running on Streamlit Cloud**: Your tasting history is stored in a CSV file. "
            "On the free tier, data will reset when the app restarts. "
            "Use the Download button in Analytics to backup your collection regularly."
        )

    history_df = ensure_wine_df(load_wine_data())
    wine_data_source = st.session_state.get("_wine_data_source")
    if wine_data_source == "local_missing_secrets":
        missing = st.session_state.get("_decant_local_preview_missing_secrets", [])
        st.info(
            "Local preview mode: using bundled CSV data because Supabase "
            f"secret(s) are missing: {', '.join(missing)}."
        )
    elif wine_data_source == "local_supabase_error":
        st.warning(
            "Supabase is not reachable from this local run, so Decant is "
            "showing bundled CSV data in read-only mode."
        )
        if DEBUG_MODE:
            st.caption(st.session_state.get("_wine_data_source_error", ""))

    if client is None and not is_guest:
        st.warning(
            "OPENAI_API_KEY is not configured, so AI wine extraction is disabled."
        )

    render_cellar_snapshot(history_df)

    # Four tabs always created so Streamlit tab indexing stays stable.
    # Tab 1 (Add Wine) content is gated inside the tab body - anonymous
    # users see a sign-in nudge inside the tab rather than a hidden tab.
    # This avoids the AttributeError that would come from `with None:` if we
    # tried to skip tab1 entirely, and keeps the layout consistent.
    tab1, tab2, tab3, tab4 = st.tabs([
        "Add Wine",
        "My Palate Maps",
        "Stats",
        "Wine Gallery",
    ])

    # All four tabs are now thin dispatch calls - bodies live in
    # decant.ui.tab_*. history_df is loaded once at the top of main()
    # so local connection/fallback messages appear only once per rerun.
    with tab1:
        tab_add_wine.render(
            history_df=history_df,
            predictor=load_predictor() if not is_guest and client is not None else None,
            client=client,
            is_authenticated_now=is_authenticated(),
        )

    with tab2:
        tab_palate_maps.render(
            history_df=history_df,
            is_guest=is_guest,
        )

    with tab3:
        tab_stats.render(
            history_df=history_df,
            debug_mode=DEBUG_MODE,
        )

    with tab4:
        tab_gallery.render(history_df=history_df)


if __name__ == "__main__":
    main()
