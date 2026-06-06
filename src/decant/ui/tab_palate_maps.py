"""Palate Maps tab body.

Two sections, in this order:

1. **Palate maps** - the consolidated radar chart and per-colour
   "ideal profile" metrics, computed from liked wines. Read-only,
   no auth gate.

2. **Data persistence** - download the collection as CSV (anyone),
   and restore from a previously-downloaded CSV (signed-in users
   only). The restore path writes to Supabase via the `wines_repo`.

`render(history_df, is_guest)` is called from inside `with tab2:` in
`app.py`. The Supabase client for writes is fetched lazily inside
the restore handler - it's only needed when a signed-in user
actually uploads a file, so binding it earlier would be wasteful.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from decant.ui.components import create_consolidated_palate_radar


# Same colour palette the rest of the app uses. Kept local because
# this is the only file that needs all four in order.
_WINE_COLORS = ["White", "Red", "Rosé", "Orange"]
_FEATURE_COLS = ["acidity", "minerality", "fruitiness", "tannin", "body"]


def render(history_df: pd.DataFrame, is_guest: bool) -> None:
    """Render the Palate Maps tab body.

    Args:
        history_df: Already-normalised wine history.
        is_guest: True if the visitor is not signed in. Gates the
            restore-from-backup upload widget.
    """
    st.markdown("## My Palate Maps")
    st.caption("Your ideal flavor profiles by wine color")

    _render_palate_maps(history_df)
    _render_data_persistence(history_df, is_guest)


def _render_palate_maps(history_df: pd.DataFrame) -> None:
    """Consolidated radar + per-colour ideal-profile metrics.

    Only wines marked `liked=True` contribute to the profiles. If the
    user has no liked wines yet, render an empty-state warning
    instead of an empty chart.
    """
    if history_df.empty:
        st.info("No wine data available. Add wines to see your palate maps!")
        return

    if "liked" in history_df.columns:
        liked_wines = history_df[history_df["liked"] == True]  # noqa: E712
    else:
        liked_wines = history_df.iloc[0:0].copy()

    if liked_wines.empty:
        st.warning(
            "No liked wines yet. Add wines and mark them as liked to see "
            "your palate maps!"
        )
        return

    missing_feature_cols = [c for c in _FEATURE_COLS if c not in liked_wines.columns]
    if "wine_color" not in liked_wines.columns:
        st.caption("Missing fields for palate maps: wine_color")
        return
    if missing_feature_cols:
        st.caption(
            f"Missing fields for palate maps: {', '.join(missing_feature_cols)}"
        )
        return

    color_profiles, color_counts = _compute_color_profiles(liked_wines)
    if not color_profiles:
        st.caption("No liked wines with complete feature data yet.")
        return

    # Consolidated radar first - gives the at-a-glance view before
    # the per-colour breakdowns.
    consolidated_radar = create_consolidated_palate_radar(color_profiles)
    st.plotly_chart(consolidated_radar, width='stretch')

    _render_per_color_metrics(color_profiles)


def _compute_color_profiles(
    liked_wines: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, int]]:
    """Group liked wines by wine_color and compute mean feature vectors."""
    profiles: dict[str, pd.Series] = {}
    counts: dict[str, int] = {}
    for wine_color in _WINE_COLORS:
        color_wines = liked_wines[liked_wines["wine_color"] == wine_color]
        if len(color_wines) > 0:
            profiles[wine_color] = color_wines[_FEATURE_COLS].mean()
            counts[wine_color] = len(color_wines)
    return profiles, counts


def _render_per_color_metrics(color_profiles: dict[str, pd.Series]) -> None:
    """Five-column ideal-profile metrics per wine colour."""
    for wine_color in _WINE_COLORS:
        if wine_color not in color_profiles:
            continue

        st.markdown(f"#### {wine_color} Wines")
        ideal = color_profiles[wine_color]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Acidity", f"{ideal['acidity']:.1f}/10")
        with col2:
            st.metric("Minerality", f"{ideal['minerality']:.1f}/10")
        with col3:
            st.metric("Fruitiness", f"{ideal['fruitiness']:.1f}/10")
        with col4:
            st.metric("Tannin", f"{ideal['tannin']:.1f}/10")
        with col5:
            st.metric("Body", f"{ideal['body']:.1f}/10")

        st.markdown("---")


def _render_data_persistence(history_df: pd.DataFrame, is_guest: bool) -> None:
    """Two-column row: download CSV (anyone) + restore from backup (auth)."""
    col_download, col_upload = st.columns([1, 1])

    with col_download:
        _render_download_button(history_df)

    with col_upload:
        if is_guest:
            st.info("Log in to restore from backup")
        else:
            _render_restore_form(history_df)

    st.markdown("---")


def _render_download_button(history_df: pd.DataFrame) -> None:
    """CSV download button. Shown to everyone; empty when no data."""
    if history_df.empty:
        st.info("No history data yet. Add wines first!")
        return

    csv_data = history_df.to_csv(index=False)
    st.download_button(
        label="Download My Collection (CSV)",
        data=csv_data,
        file_name="decant_wine_history.csv",
        mime="text/csv",
    )


def _render_restore_form(history_df: pd.DataFrame) -> None:
    """Upload a previously-downloaded CSV and merge non-duplicates.

    Dedup key is (wine_name + vintage). Existing rows in Supabase
    are looked up at submission time, not at render - the user may
    have added wines since the form was rendered.
    """
    uploaded_file = st.file_uploader(
        "Restore from Backup",
        type=["csv"],
        help="Upload a previously downloaded CSV to restore your collection",
        key="restore_history",
    )

    if uploaded_file is None:
        return

    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return

    required_cols = ["wine_name", "score", "liked"]
    missing_cols = [c for c in required_cols if c not in uploaded_df.columns]
    if missing_cols:
        st.error(f"Invalid CSV: Missing columns {missing_cols}")
        return

    new_wines = _dedup_against_existing(uploaded_df, history_df)

    if len(new_wines) == 0:
        st.info("No new wines to add. All uploaded wines already exist.")
        return

    _import_new_wines(new_wines)


def _dedup_against_existing(
    uploaded_df: pd.DataFrame, existing_df: pd.DataFrame
) -> pd.DataFrame:
    """Return uploaded rows that aren't already in existing.

    Dedup key is `wine_name + '_' + vintage`, with 'NV' filling in
    for missing vintages. Same key construction on both sides so
    comparison is symmetric.
    """
    if existing_df.empty:
        return uploaded_df

    existing_keys = (
        existing_df["wine_name"].astype(str)
        + "_"
        + existing_df["vintage"].fillna("NV").astype(str)
    )

    if "vintage" in uploaded_df.columns:
        uploaded_keys = (
            uploaded_df["wine_name"].astype(str)
            + "_"
            + uploaded_df["vintage"].fillna("NV").astype(str)
        )
    else:
        uploaded_keys = uploaded_df["wine_name"].astype(str) + "_NV"

    return uploaded_df[~uploaded_keys.isin(existing_keys)]


def _import_new_wines(new_wines: pd.DataFrame) -> None:
    """Write deduped rows to Supabase one at a time.

    Per-row try/except so a single bad row doesn't abort the import.
    The Supabase client and cache-clear are imported lazily to keep
    this module's import graph small - they're only used here.
    """
    from decant.supabase_session import get_user_supabase
    from decant.wines_repo import repo_add_wine

    sb = get_user_supabase()
    imported = 0
    for _, row in new_wines.iterrows():
        try:
            row_data = row.dropna().to_dict()
            repo_add_wine(sb, row_data)
            imported += 1
        except Exception as row_err:
            st.warning(f"Skipped {row.get('wine_name', '?')}: {row_err}")

    st.success(f"Imported {imported} new wines.")

    # Invalidate the load_wine_data cache so the next read sees the
    # writes. Imported lazily for the same reason as above.
    from app import clear_wine_data_cache
    clear_wine_data_cache()
