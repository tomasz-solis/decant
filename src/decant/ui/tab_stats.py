"""Stats tab: collection metrics, palate stats, top regions, top wines.

Read-only view. No auth gate. Region filter is local to this tab so
the rest of the app doesn't have to know about it.

`render(history_df, debug_mode=False)` is called from inside
`with tab3:` in `app.py`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(history_df: pd.DataFrame, debug_mode: bool = False) -> None:
    """Render the Stats tab body.

    Args:
        history_df: Already-normalised wine history.
        debug_mode: When True, append a diagnostic section showing the
            DataFrame shape, columns, and a preview. Defaults to False
            so production renders stay clean.
    """
    st.markdown("## Stats")
    st.caption("Your collection at a glance")

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

    st.markdown("---")

    # Headline metrics (Liked / Disliked / Total).
    total_wines = len(df)
    has_liked_col = 'liked' in df.columns
    liked_wines = int(df['liked'].sum()) if has_liked_col else 0
    disliked_wines = max(total_wines - liked_wines, 0)
    liked_df = df[df['liked'] == True] if has_liked_col else df.iloc[0:0].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Liked", liked_wines)
    with col2:
        st.metric("Disliked", disliked_wines)
    with col3:
        st.metric("Total", total_wines)

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
            st.caption("Your ideal wine profile:")
            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                st.metric("Acid", f"{liked_avg['acidity']:.1f}")
            with f2:
                st.metric("Mineral", f"{liked_avg['minerality']:.1f}")
            with f3:
                st.metric("Fruit", f"{liked_avg['fruitiness']:.1f}")
            with f4:
                st.metric("Tannin", f"{liked_avg['tannin']:.1f}")
            with f5:
                st.metric("Body", f"{liked_avg['body']:.1f}")

    # --- Top Regions (top 3 by average score) ---
    st.markdown("---")
    st.markdown("### Top Regions")

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
                medal = {0: '', 1: '', 2: ''}.get(idx, f"#{idx + 1}")
                rcol1, rcol2, rcol3 = st.columns([1, 5, 2])
                with rcol1:
                    # Medal: an inline icon, NOT a section heading. Rendering
                    # via `### {medal}` makes it an h3 - same heading level as
                    # the "Top Regions" section title above, which confuses
                    # the visual hierarchy.
                    st.markdown(
                        f"<div style='font-size: 2rem; line-height: 1; "
                        f"margin: 0.5rem 0;'>{medal}</div>",
                        unsafe_allow_html=True,
                    )
                with rcol2:
                    st.markdown(f"**{region}**")
                    st.caption(f"{int(stats['count'])} wines")
                with rcol3:
                    st.metric("Avg Score", f"{stats['avg_score']:.1f}/10")
        else:
            st.caption("No regional data yet")

    # --- Top Wines (top 3 by score) ---
    st.markdown("---")
    st.markdown("### Top Wines")

    top_wines_df = liked_df if not liked_df.empty else df
    required_for_top = {'wine_name', 'score'}
    if top_wines_df.empty:
        st.caption("Add and rate wines to see your top picks.")
    elif not required_for_top.issubset(top_wines_df.columns):
        st.caption("Score column missing - can't rank wines yet.")
    else:
        top3 = top_wines_df.sort_values('score', ascending=False).head(3)
        for rank, (_, wine) in enumerate(top3.iterrows(), start=1):
            producer = wine.get('producer', '')
            vintage = wine.get('vintage')
            year = f" {int(vintage)}" if vintage and not pd.isna(vintage) and vintage > 0 else ""
            medal = {1: '', 2: '', 3: ''}.get(rank, f"#{rank}")
            wcol1, wcol2, wcol3 = st.columns([1, 6, 2])
            with wcol1:
                # Medal as inline icon (see _render_regions for rationale).
                st.markdown(
                    f"<div style='font-size: 2rem; line-height: 1; "
                    f"margin: 0.5rem 0;'>{medal}</div>",
                    unsafe_allow_html=True,
                )
            with wcol2:
                st.markdown(f"**{wine['wine_name']}**{year}")
                if producer:
                    st.caption(producer)
            with wcol3:
                st.metric("Score", f"{wine['score']:.1f}/10")

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
