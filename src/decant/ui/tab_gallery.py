"""Wine Gallery tab body.

Browse the household's full wine collection. Pure read-only view —
no auth gate, no writes (except photo upload which is local). Used
by all visitors including anonymous ones.

The grid layout is rendered via `apply_gallery_styles()` (scoped CSS
in `decant.ui.styles`), with cards built using Streamlit's native
column primitives plus a thin `unsafe_allow_html` layer for the
`.glass-card` wrapper.

Call `render(history_df)` from inside `with tab4:` in `app.py`.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from decant.services.image_storage import (
    get_wine_image_path,
    get_wine_image_url,
    save_wine_image,
)
from decant.ui.helpers import should_display_vintage, show_empty_data_diagnostics
from decant.ui.styles import apply_gallery_styles


# Hard-coded for now. Could move to ui-config later if other tabs
# need similar palettes.
_COLOR_OPTIONS = ["All", "White", "Red", "Rosé", "Orange"]
_LIKED_OPTIONS = ["All", "Liked", "Not Liked"]
_CARDS_PER_ROW = 3


def render(history_df: pd.DataFrame) -> None:
    """Render the Wine Gallery tab body.

    Args:
        history_df: Already-normalised wine history (from
            `data_access.normalize`). The caller loads this once at
            the top of `main()` to avoid redundant Supabase round
            trips across tabs.
    """
    st.markdown("## 🖼️ Wine Gallery")
    st.caption("Browse your complete wine collection with all details")

    if history_df is None or len(history_df) == 0:
        st.info("No wines in your collection yet. Add wines to see them here!")
        show_empty_data_diagnostics()
        return

    filtered_df = _apply_filters(history_df)
    if filtered_df.empty:
        st.info("No wines match the current filters.")
        return

    st.markdown(f"### Found {len(filtered_df)} wines")
    apply_gallery_styles()

    _render_grid(filtered_df)


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render the filter row and return the filtered DataFrame.

    Three filters: a text search across name/producer/region, a wine
    color selector, and a like/dislike toggle. The result is sorted
    by score descending so the best matches surface first.
    """
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Search wines",
            placeholder="Search by name, producer, region...",
        )
    with col2:
        color_filter = st.selectbox("Filter by color", _COLOR_OPTIONS)
    with col3:
        liked_filter = st.selectbox("Filter by preference", _LIKED_OPTIONS)

    out = df.copy()

    if search_query:
        q = search_query.lower()
        out = out[
            out["wine_name"].str.lower().str.contains(q, na=False)
            | out["producer"].str.lower().str.contains(q, na=False)
            | out["region"].str.lower().str.contains(q, na=False)
        ]

    if color_filter != "All":
        out = out[out["wine_color"] == color_filter]

    if liked_filter == "Liked":
        out = out[out["liked"] == True]  # noqa: E712 — pandas needs ==, not `is`
    elif liked_filter == "Not Liked":
        out = out[out["liked"] == False]  # noqa: E712

    return out.sort_values("score", ascending=False)


def _render_grid(df: pd.DataFrame) -> None:
    """Render filtered wines as a card grid using batched columns.

    Streamlit doesn't have a "true" grid widget — we fake one by
    creating one row of N columns per batch. Batched (not one big
    columns call) so that incomplete final rows don't render empty
    boxes.
    """
    wines_list = list(df.iterrows())
    num_wines = len(wines_list)

    for batch_start in range(0, num_wines, _CARDS_PER_ROW):
        batch_end = min(batch_start + _CARDS_PER_ROW, num_wines)
        batch_size = batch_end - batch_start
        cols = st.columns([1] * batch_size, gap="medium")

        for col_idx in range(batch_size):
            wine_idx = batch_start + col_idx
            _, wine = wines_list[wine_idx]
            with cols[col_idx]:
                _render_wine_card(wine, wine_idx)


def _render_wine_card(wine: pd.Series, wine_idx: int) -> None:
    """Render a single wine card inside an already-active column."""
    wine_name = wine.get("wine_name", "Unknown")

    st.markdown('<div class="glass-card wine-card">', unsafe_allow_html=True)
    _render_card_image(wine_name)
    _render_card_meta(wine, wine_name)
    _render_card_metrics(wine)
    _render_card_icons(wine)
    _render_card_notes(wine)
    _render_card_upload(wine, wine_name, wine_idx)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_card_image(wine_name: str) -> None:
    """Render the wine's saved image or a placeholder if none exists."""
    image_path = get_wine_image_path(wine_name)
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        img_ext = image_path.split(".")[-1]
        st.markdown(
            f'<img src="data:image/{img_ext};base64,{img_data}" class="wine-card-img" />',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="wine-card-img-placeholder">🍷</div>',
            unsafe_allow_html=True,
        )


def _render_card_meta(wine: pd.Series, wine_name: str) -> None:
    """Name, producer + vintage, region/country."""
    st.markdown(
        f"<h4 style='margin: 12px 0 4px 0; font-size: 16px;'>{wine_name[:40]}</h4>",
        unsafe_allow_html=True,
    )

    vintage_value = wine.get("vintage")
    vintage_suffix = (
        f" • {int(vintage_value)}" if should_display_vintage(vintage_value) else ""
    )
    producer = (wine.get("producer", "Unknown") or "Unknown")[:30]
    st.markdown(
        f"<p style='font-size: 13px; color: #A0A0A8; margin: 0 0 4px 0;'>"
        f"{producer}{vintage_suffix}</p>",
        unsafe_allow_html=True,
    )

    location = wine.get("region") or wine.get("country") or "Unknown"
    st.markdown(
        f"<p style='font-size: 12px; color: #A0A0A8; margin: 0 0 8px 0;'>"
        f"📍 {str(location)[:35]}</p>",
        unsafe_allow_html=True,
    )


def _render_card_metrics(wine: pd.Series) -> None:
    """Score (left) and price (right) as Streamlit metrics."""
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Score", f"{wine.get('score', 0):.1f}/10")
    with m2:
        st.metric("Price", f"€{wine.get('price', 0):.0f}")


def _render_card_icons(wine: pd.Series) -> None:
    """Liked / sparkling / natural badge row.

    Always renders the row container even if no badges fire — keeps
    card heights consistent across the grid.
    """
    active_icons = []
    if wine.get("liked"):
        active_icons.append('<span class="badge" style="font-size: 16px;">❤️</span>')
    if wine.get("is_sparkling"):
        active_icons.append('<span class="badge" style="font-size: 16px;">✨</span>')
    if wine.get("is_natural"):
        active_icons.append('<span class="badge" style="font-size: 16px;">🌱</span>')

    icons_content = " ".join(active_icons) if active_icons else "&nbsp;"
    st.markdown(
        f'<div class="icon-row" style="min-height: 24px; margin: 8px 0;">'
        f"{icons_content}</div>",
        unsafe_allow_html=True,
    )


def _render_card_notes(wine: pd.Series) -> None:
    """Tasting notes inside a collapsible expander, if present."""
    notes = wine.get("notes", "")
    if notes:
        with st.expander("📝 Tasting Notes"):
            st.markdown(f"_{notes}_")


def _render_card_upload(wine: pd.Series, wine_name: str, wine_idx: int) -> None:
    """Photo upload + Vivino link, both inside an expander.

    Photo upload writes locally via `save_wine_image`. The Vivino
    link goes to an external search URL — useful when the user
    wants to look up something the household hasn't photographed yet.
    """
    with st.expander("📸 Upload Photo"):
        uploaded_image = st.file_uploader(
            "Choose bottle photo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"upload_{wine_name}_{wine_idx}",
            label_visibility="collapsed",
        )

        if uploaded_image:
            if st.button("💾 Save Photo", key=f"save_{wine_name}_{wine_idx}"):
                saved_path = save_wine_image(uploaded_image, wine_name)
                if saved_path:
                    st.success("✓ Photo saved!")
                    st.rerun()

        vivino_url = get_wine_image_url(wine_name, wine.get("producer", ""))
        st.markdown(f"[🔍 Find on Vivino]({vivino_url})")
