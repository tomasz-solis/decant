"""Wine Gallery tab body.

Browse the household's full wine collection. Anonymous users get a
pure read-only view. Authenticated users can also edit a wine's
metadata in place (vintage, producer, price, score, etc) via an
inline form on each card. Flavour features stay read-only — editing
them affects every downstream palate score, which needs its own
path with explicit user confirmation about the blast radius.

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
from decant.supabase_session import get_user_supabase, is_authenticated
from decant.ui.helpers import should_display_vintage, show_empty_data_diagnostics
from decant.ui.styles import apply_gallery_styles
from decant.wines_repo import repo_update_wine


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
    _render_card_edit(wine, wine_idx)
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


# ---- editable field set ----------------------------------------------
# Mirrors `wines_repo._EDITABLE_FIELDS`. Kept here as a separate
# constant so a UI-side dropdown change doesn't have to round-trip
# through the repo module — the actual security filter is enforced
# server-side by repo_update_wine.
_SWEETNESS_OPTIONS = ["Dry", "Off-Dry", "Semi-Sweet", "Sweet"]
_WINE_COLOR_OPTIONS = ["Red", "White", "Rosé", "Orange", "Sparkling"]


def _coerce_vintage(raw: str | int | float | None) -> int | None:
    """Parse a vintage input. Empty / 'NV' / non-numeric -> None.

    Vintage is stored as nullable in Supabase; the user UI accepts a
    free-form string so people can type 'NV' for non-vintage and we
    translate it to NULL.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw) if 1800 < raw < 2100 else None
    s = str(raw).strip().upper()
    if not s or s == "NV":
        return None
    try:
        n = int(s)
        return n if 1800 < n < 2100 else None
    except ValueError:
        return None


def _render_card_edit(wine: pd.Series, wine_idx: int) -> None:
    """Inline edit form for a wine's metadata.

    Only authenticated users see this. Guest mode keeps the gallery
    read-only as before. The form lives inside an expander so it
    doesn't clutter the card's resting state.

    Edits metadata fields only. Flavour features (acidity, body, etc)
    are intentionally absent — changing them affects every downstream
    palate score and needs an explicit blast-radius warning. See
    `wines_repo.repo_update_wine`.
    """
    if not is_authenticated():
        return

    wine_id = wine.get("id")
    if not wine_id:
        # `id` is the int4 primary key from Supabase, normalized to
        # int via data_access.normalize. 0 means "no id available"
        # (shouldn't happen for rows loaded from Supabase, but guard
        # anyway — we can't update a row without a primary key).
        return

    with st.expander("✏️ Edit details"):
        # Use a form so the user can change multiple fields and submit
        # them in one batch. Without `st.form`, each input change would
        # trigger a rerun and the in-progress edits would be lost.
        with st.form(f"edit_wine_{wine_idx}", clear_on_submit=False):
            # Name goes at the top, full-width. It's the wine's
            # identity, not just another attribute — corrections to
            # missing/wrong words in the original extraction live here.
            new_wine_name = st.text_input(
                "Name",
                value=str(wine.get("wine_name") or ""),
                key=f"edit_name_{wine_idx}",
            )

            # Column A: text inputs only (uniform height -> clean
            # vertical rhythm). Column B: pickers + numerics. The
            # three flag checkboxes get their own row below to avoid
            # the previous layout where stacked checkboxes vs a
            # slider made the columns look wonky.
            col_a, col_b = st.columns(2)
            with col_a:
                vintage_raw = wine.get("vintage")
                vintage_default = (
                    str(int(vintage_raw))
                    if should_display_vintage(vintage_raw)
                    else ""
                )
                new_vintage_str = st.text_input(
                    "Vintage",
                    value=vintage_default,
                    placeholder="e.g. 2021 or NV",
                    key=f"edit_vintage_{wine_idx}",
                )
                new_producer = st.text_input(
                    "Producer",
                    value=str(wine.get("producer") or ""),
                    key=f"edit_producer_{wine_idx}",
                )
                new_region = st.text_input(
                    "Region",
                    value=str(wine.get("region") or ""),
                    key=f"edit_region_{wine_idx}",
                )
                new_country = st.text_input(
                    "Country",
                    value=str(wine.get("country") or ""),
                    key=f"edit_country_{wine_idx}",
                )

            with col_b:
                current_color = wine.get("wine_color") or "Red"
                color_index = (
                    _WINE_COLOR_OPTIONS.index(current_color)
                    if current_color in _WINE_COLOR_OPTIONS
                    else 0
                )
                new_wine_color = st.selectbox(
                    "Colour",
                    _WINE_COLOR_OPTIONS,
                    index=color_index,
                    key=f"edit_color_{wine_idx}",
                )
                current_sweetness = wine.get("sweetness") or "Dry"
                sweetness_index = (
                    _SWEETNESS_OPTIONS.index(current_sweetness)
                    if current_sweetness in _SWEETNESS_OPTIONS
                    else 0
                )
                new_sweetness = st.selectbox(
                    "Sweetness",
                    _SWEETNESS_OPTIONS,
                    index=sweetness_index,
                    key=f"edit_sweetness_{wine_idx}",
                )
                new_score = st.slider(
                    "Score",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(wine.get("score") or 0.0),
                    step=0.1,
                    key=f"edit_score_{wine_idx}",
                )
                new_price = st.number_input(
                    "Price (€)",
                    min_value=0.0,
                    value=float(wine.get("price") or 0.0),
                    step=1.0,
                    key=f"edit_price_{wine_idx}",
                )

            # Flags row: three checkboxes side by side keeps the
            # whole form compact and reads as "the on/off attributes
            # of this wine" in a single horizontal beat.
            flag_a, flag_b, flag_c = st.columns(3)
            with flag_a:
                new_liked = st.checkbox(
                    "❤️ Liked",
                    value=bool(wine.get("liked")),
                    key=f"edit_liked_{wine_idx}",
                )
            with flag_b:
                new_sparkling = st.checkbox(
                    "✨ Sparkling",
                    value=bool(wine.get("is_sparkling")),
                    key=f"edit_sparkling_{wine_idx}",
                )
            with flag_c:
                new_natural = st.checkbox(
                    "🌱 Natural",
                    value=bool(wine.get("is_natural")),
                    key=f"edit_natural_{wine_idx}",
                )

            new_notes = st.text_area(
                "Tasting notes",
                value=str(wine.get("notes") or ""),
                key=f"edit_notes_{wine_idx}",
                height=80,
            )

            submitted = st.form_submit_button("💾 Save changes")

            if submitted:
                cleaned_name = new_wine_name.strip()
                if not cleaned_name:
                    st.error("❌ Name can't be empty.")
                    return

                fields = {
                    "wine_name": cleaned_name,
                    "vintage": _coerce_vintage(new_vintage_str),
                    "producer": new_producer.strip() or None,
                    "region": new_region.strip() or None,
                    "country": new_country.strip() or None,
                    "wine_color": new_wine_color,
                    "score": float(new_score),
                    "price": float(new_price),
                    "liked": bool(new_liked),
                    "is_sparkling": bool(new_sparkling),
                    "is_natural": bool(new_natural),
                    "sweetness": new_sweetness,
                    "notes": new_notes.strip() or None,
                }
                try:
                    result = repo_update_wine(
                        get_user_supabase(),
                        int(wine_id),
                        fields,
                    )
                except Exception as exc:
                    st.error(f"❌ Couldn't save changes: {exc}")
                    return

                if not result:
                    st.error(
                        "❌ Save returned no row — the wine may no longer "
                        "exist or you may not have permission to edit it."
                    )
                    return

                st.success("✅ Saved.")
                st.rerun()


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
