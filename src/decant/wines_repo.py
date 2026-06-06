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


def repo_update_wine(
    sb: Client,
    wine_id: int,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Update an existing wine row's editable fields.

    Args:
        sb: Authenticated Supabase client.
        wine_id: The wine row's primary key (`id` column, int4).
        fields: Field name -> new value. Only metadata fields are
            allowed; see `_EDITABLE_FIELDS`. Unknown or disallowed
            fields are silently dropped to prevent accidentally
            updating flavour features or identity columns via this
            entry point.

    Returns:
        The updated row, or an empty dict if Supabase returned
        nothing (e.g. the row doesn't exist or doesn't belong to
        the caller's cellar).

    Notes:
        The query filters on both `id` AND `cellar_id` as a defense
        in depth - even if RLS were misconfigured, a caller couldn't
        update a wine outside their cellar via this function.

        Flavour features (acidity/fruitiness/body/tannin/minerality)
        are intentionally NOT editable here. Changing those affects
        every downstream palate-match calculation because they shift
        the population mean and ideal profile. If we ever expose
        feature editing, it needs its own path with explicit user
        confirmation about that blast radius.

        wine_name IS editable. Renaming is needed because the
        original extraction is sometimes wrong, and there's no other
        way to fix it without delete-and-re-add. Side-effects: the
        LLM cache keyed by name goes stale (harmless, expires
        normally); prior-tasting matching uses tokenised wine_name,
        so small corrections still match across the rename but a
        wholesale rename would stop matching prior history; the
        unique constraint on (user_id, wine_name, vintage) rejects
        a rename that collides with an existing wine.
    """
    allowed = {k: v for k, v in fields.items() if k in _EDITABLE_FIELDS}
    if not allowed:
        return {}

    cellar_id = _get_cellar_id()
    response = (
        sb.table("wines")
        .update(allowed)
        .eq("id", wine_id)
        .eq("cellar_id", cellar_id)
        .execute()
    )
    data = response.data or []
    return data[0] if data else {}


# Fields that `repo_update_wine` will accept. Everything else is
# silently dropped - see the docstring for why flavour features and
# id are excluded. wine_name IS editable here because the original
# extraction is sometimes wrong (e.g. missing '1er' in a Premier Cru
# name) and there's no other way to fix it. Renaming has a few
# side-effects: the LLM cache keyed by name goes stale (harmless,
# expires in 24h), prior-tasting matching still works as long as
# token overlap survives the rename, and the unique constraint on
# (user_id, wine_name, vintage) will reject a rename that collides
# with an existing wine.
_EDITABLE_FIELDS: frozenset[str] = frozenset({
    "wine_name",
    "vintage",
    "producer",
    "region",
    "country",
    "price",
    "score",
    "liked",
    "notes",
    "wine_color",
    "is_sparkling",
    "is_natural",
    "sweetness",
})


def list_wines(sb: Client) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    return repo_list_wines(sb)
