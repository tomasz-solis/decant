"""Supabase repository helpers for wines table operations."""

from typing import Any

import pandas as pd
import streamlit as st
from supabase import Client


def _normalize_secret_string(raw_value: Any, secret_name: str) -> str:
    """Normalize a string secret value and guard against common formatting issues."""
    if raw_value is None:
        raise ValueError(f"{secret_name} is missing in Streamlit secrets")

    value = str(raw_value).strip()

    # Handle accidental copied quotes inside TOML string values.
    quote_pairs = [
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    ]
    for left_quote, right_quote in quote_pairs:
        if value.startswith(left_quote) and value.endswith(right_quote) and len(value) >= 2:
            value = value[1:-1].strip()
            break

    if not value:
        raise ValueError(f"{secret_name} is empty in Streamlit secrets")
    return value


def _get_cellar_id() -> str:
    """Read shared cellar id from Streamlit secrets."""
    return _normalize_secret_string(st.secrets["CELLAR_ID"], "CELLAR_ID")


def _is_debug_enabled() -> bool:
    """Return debug mode from secrets."""
    try:
        debug_value = st.secrets.get("DEBUG", False)
    except (FileNotFoundError, KeyError, AttributeError):
        return False

    if isinstance(debug_value, str):
        return debug_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(debug_value)


def repo_list_wines(sb: Client) -> pd.DataFrame:
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

    if df.empty:
        diagnostics: dict[str, Any] = {
            "configured_cellar_id": cellar_id,
            "accessible_cellar_ids": [],
            "probe_error": None,
        }
        try:
            probe = sb.table("wines").select("cellar_id").limit(500).execute()
            probe_rows = probe.data or []
            diagnostics["accessible_cellar_ids"] = sorted(
                {
                    row.get("cellar_id")
                    for row in probe_rows
                    if row.get("cellar_id")
                }
            )
        except Exception as probe_error:
            diagnostics["probe_error"] = str(probe_error)
        st.session_state["_wine_df_empty_debug"] = diagnostics
    else:
        st.session_state.pop("_wine_df_empty_debug", None)

    if _is_debug_enabled():
        if not df.empty and "liked" not in df.columns:
            st.session_state["_wine_df_missing_liked_debug"] = {
                "columns": list(df.columns),
                "rows_type": type(rows).__name__,
            }
        else:
            st.session_state.pop("_wine_df_missing_liked_debug", None)

    return df


def repo_add_wine(sb: Client, row_data: dict[str, Any]) -> dict[str, Any]:
    """Insert a wine row scoped to a cellar."""
    row_data["cellar_id"] = _get_cellar_id()
    response = sb.table("wines").insert(row_data).execute()
    data = response.data or []
    return data[0] if data else {}


def list_wines(sb: Client) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return repo_list_wines(sb)
