"""Small UI helpers shared across tab modules.

Both functions used to live in `app.py` and were referenced from
multiple tabs. Moving them here breaks a potential circular-import
problem and clarifies that they're presentation-layer utilities,
not app-level orchestration.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def should_display_vintage(vintage_value: Any) -> bool:
    """True if the vintage should be shown in the UI.

    Filters out the sentinels that the schema uses for "unknown
    vintage": None, NaN, 0 (the default), and any value outside the
    plausible wine-vintage range. Non-numeric strings also return
    False rather than raising.
    """
    if vintage_value is None or pd.isna(vintage_value):
        return False
    try:
        vintage_int = int(vintage_value)
        return 0 < vintage_int < 2100
    except (ValueError, TypeError):
        return False


def show_empty_data_diagnostics() -> None:
    """Surface actionable hints when the wines table comes back empty.

    Reads from `st.session_state["_wine_df_empty_debug"]`, which the
    data-access layer populates when a Supabase query returns zero
    rows. Renders a warning + the list of cellar IDs the current
    session can actually see, so a misconfigured `CELLAR_ID` is easy
    to spot and fix.

    Does nothing if no diagnostics are recorded.
    """
    diagnostics = st.session_state.get("_wine_df_empty_debug")
    if not diagnostics:
        return

    configured_cellar_id = diagnostics.get("configured_cellar_id")
    accessible_cellar_ids = diagnostics.get("accessible_cellar_ids") or []
    probe_error = diagnostics.get("probe_error")

    if accessible_cellar_ids and configured_cellar_id not in accessible_cellar_ids:
        st.warning(
            "No rows matched the configured `CELLAR_ID`. "
            "Update `CELLAR_ID` in Streamlit Cloud secrets to one of "
            "the accessible values below."
        )
        st.code("\n".join(accessible_cellar_ids), language="text")
        return

    if probe_error:
        st.caption(
            f"Debug hint: unable to inspect accessible cellar IDs ({probe_error})."
        )
