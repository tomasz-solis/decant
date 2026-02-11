"""Supabase repository helpers for wines table operations."""

from typing import Any

import pandas as pd
import streamlit as st
from supabase import Client


def _get_cellar_id() -> str:
    """Read shared cellar id from Streamlit secrets."""
    cellar_id = st.secrets["CELLAR_ID"]
    if not cellar_id:
        raise ValueError("CELLAR_ID is empty in Streamlit secrets")
    return cellar_id


def _is_debug_enabled() -> bool:
    """Return debug mode from secrets."""
    try:
        debug_value = st.secrets.get("DEBUG", False)
    except (FileNotFoundError, KeyError, AttributeError):
        return False

    if isinstance(debug_value, str):
        return debug_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(debug_value)


def list_wines(sb: Client) -> pd.DataFrame:
    """Return wines for the shared cellar ordered by newest first."""
    cellar_id = _get_cellar_id()
    res = (
        sb.table("wines")
        .select("*")
        .eq("cellar_id", cellar_id)
        .order("created_at", desc=True)
        .execute()
    )
    rows = res.data or []
    df = pd.DataFrame(rows)

    if _is_debug_enabled():
        if not df.empty and "liked" not in df.columns:
            st.session_state["_wine_df_missing_liked_debug"] = {
                "columns": list(df.columns),
                "rows_type": type(rows).__name__,
            }
        else:
            st.session_state.pop("_wine_df_missing_liked_debug", None)

    return df


def add_wine(sb: Client, payload: dict[str, Any]) -> dict[str, Any]:
    """Insert a wine row scoped to a cellar."""
    row_payload = {**payload}
    row_payload["cellar_id"] = _get_cellar_id()
    response = sb.table("wines").insert(row_payload).execute()
    data = response.data or []
    return data[0] if data else {}
