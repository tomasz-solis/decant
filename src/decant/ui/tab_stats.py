"""Stats tab: collection metrics, palate stats, top regions, top wines.

Read-only view. No auth gate. Region filter is local to this tab so
the rest of the app doesn't have to know about it.

`render(history_df, debug_mode=False)` is called from inside
`with tab3:` in `app.py`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from decant.ui.editorial import (
    format_score,
    render_feature_profile,
    render_ranked_list,
    render_stat_grid,
    render_tab_heading,
)


def render(history_df: pd.DataFrame, debug_mode: bool = False) -> None:
    """Render the Stats tab body.

    Args:
        history_df: Already-normalised wine history.
        debug_mode: When True, append a diagnostic section showing the
            DataFrame shape, columns, and a preview. Defaults to False
            so production renders stay clean.
    """
    render_tab_heading(
        "Stats",
        "Cellar index",
        "A clean read on what you drink, where it comes from, and what keeps winning.",
    )

    df = history_df

    # Regional filter dropdown.
    if not df.empty and 'region' in df.columns:
        # Get unique regions (exclude Unknown)
        regions = df[
            (df['region'] != 'Unknown') &
            (df['region'].notna())
        ]['region'].unique()

        if len(regions) > 0:
            regions_sorted = sorted(regions)
            selected_region = st.selectbox(
                "Filter by Region",
                ["All Regions"] + list(regions_sorted),
                key='region_filter'
            )

            if selected_region != "All Regions":
                df = df[df['region'] == selected_region]
                st.caption(f"Showing: {selected_region}")

    # Headline metrics (Liked / Disliked / Total).
    total_wines = len(df)
    has_liked_col = 'liked' in df.columns
    liked_wines = int(df['liked'].sum()) if has_liked_col else 0
    disliked_wines = max(total_wines - liked_wines, 0)
    liked_df = df[df['liked'].eq(True)] if has_liked_col else df.iloc[0:0].copy()

    like_share = f"{liked_wines / total_wines:.0%} of visible wines" if total_wines else "no visible wines"
    render_stat_grid(
        (
            ("Liked", f"{liked_wines:,}", like_share),
            ("Not liked", f"{disliked_wines:,}", "kept for context"),
            ("Visible", f"{total_wines:,}", "after filters"),
        ),
        class_name="editorial-stat-grid",
    )

    # --- Palate Stats (your ideal flavour numbers) ---
    st.markdown("---")
    st.markdown("### Palate Stats")

    feature_cols = ['acidity', 'minerality', 'fruitiness', 'tannin', 'body']
    missing_feature_cols = [c for c in feature_cols if c not in liked_df.columns]

    if liked_df.empty:
        st.caption("Add wines with flavor profiles to see your palate stats")
    elif missing_feature_cols:
        st.caption(f"Missing fields for palate stats: {', '.join(missing_feature_cols)}")
    else:
        liked_avg = liked_df[feature_cols].mean()
        if liked_avg.sum() == 0:
            st.caption("Add wines with flavor profiles to see your palate stats")
        else:
            render_feature_profile(
                "Ideal profile",
                liked_avg,
                note=f"{len(liked_df)} liked wine{'s' if len(liked_df) != 1 else ''}",
                tone="neutral",
            )

    st.markdown("---")
    region_col, wine_col = st.columns(2, gap="large")

    with region_col:
        st.markdown("### Top Regions")
        regional_required_cols = ['region', 'country', 'score', 'wine_name']
        missing_regional_cols = [
            c for c in regional_required_cols if c not in liked_df.columns
        ]

        if liked_df.empty:
            st.caption("Log wines with regions to see analytics")
        elif missing_regional_cols:
            st.caption(
                f"Missing fields for regional analytics: {', '.join(missing_regional_cols)}"
            )
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
                rows = [
                    (
                        str(region),
                        f"{int(stats['count'])} wine"
                        f"{'s' if int(stats['count']) != 1 else ''}",
                        format_score(stats["avg_score"]),
                        "",
                    )
                    for region, stats in regional_stats.head(3).iterrows()
                ]
                render_ranked_list(rows)
            else:
                st.caption("No regional data yet")

    with wine_col:
        st.markdown("### Top Wines")
        top_wines_df = liked_df if not liked_df.empty else df
        required_for_top = {'wine_name', 'score'}
        if top_wines_df.empty:
            st.caption("Add and rate wines to see your top picks.")
        elif not required_for_top.issubset(top_wines_df.columns):
            st.caption("Score column missing - can't rank wines yet.")
        else:
            top3 = top_wines_df.sort_values('score', ascending=False).head(3)
            rows = []
            for _, wine in top3.iterrows():
                producer = wine.get('producer', '')
                vintage = wine.get('vintage')
                has_year = vintage and not pd.isna(vintage) and vintage > 0
                year = f" {int(vintage)}" if has_year else ""
                meta = str(producer) if producer else "Unknown producer"
                rows.append(
                    (
                        f"{wine['wine_name']}{year}",
                        meta,
                        format_score(wine["score"]),
                        "",
                    )
                )
            render_ranked_list(rows)

    # --- Debug (gated by debug_mode) ---
    if debug_mode:
        st.markdown("---")
        st.markdown("### Debug Data")
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
