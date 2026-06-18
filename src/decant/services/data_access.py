"""Single source of truth for loading wine history from Supabase.

The repository layer (`wines_repo`) speaks raw Supabase. This module
normalises the result into a DataFrame with a guaranteed schema, so the
rest of the codebase can rely on column names, types, and defaults
without checking for missing fields.

The schema constants live here (not in app.py) because data access is
the layer that owns them. Anyone reading wine data goes through
`load_history`.
"""

from typing import Optional

import pandas as pd
from supabase import Client

from decant.wines_repo import repo_list_wines


EXPECTED_COLUMNS = [
    # Row identity from Supabase (int4 PRIMARY KEY). Preserved so
    # callers that need to update a specific row (e.g. the Gallery
    # edit form) can pass the primary key to repo_update_wine.
    # Defaults to 0 when absent - consumers check for truthiness
    # before using it (0 is treated as "no id available").
    "id",
    "wine_name",
    "producer",
    "vintage",
    "notes",
    "score",
    "liked",
    "price",
    "country",
    "region",
    "wine_color",
    "is_sparkling",
    "is_natural",
    "sweetness",
    "acidity",
    "minerality",
    "fruitiness",
    "tannin",
    "body",
]

NUMERIC_COLUMNS = [
    "id",
    "acidity",
    "minerality",
    "fruitiness",
    "tannin",
    "body",
    "score",
    "price",
    "vintage",
]

BOOL_COLUMNS = ["liked", "is_sparkling", "is_natural"]

TEXT_COLUMNS = [
    "wine_name",
    "producer",
    "notes",
    "country",
    "region",
    "wine_color",
    "sweetness",
]

DEFAULTS = {
    "id": 0,
    "wine_name": "Unknown",
    "producer": "Unknown",
    "vintage": 0.0,
    "notes": "",
    "score": 0.0,
    "liked": False,
    "price": 0.0,
    "country": "Unknown",
    "region": "Unknown",
    "wine_color": "Unknown",
    "is_sparkling": False,
    "is_natural": False,
    "sweetness": "Unknown",
    "acidity": 0.0,
    "minerality": 0.0,
    "fruitiness": 0.0,
    "tannin": 0.0,
    "body": 0.0,
}


def normalize(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a wine DataFrame with guaranteed columns and types.

    Handles None, empty, and malformed inputs (e.g., DataFrames built
    from list rows with RangeIndex columns). Missing columns are filled
    with sensible defaults from `DEFAULTS`. Numeric columns are coerced
    via pd.to_numeric with `errors="coerce"` and NaNs replaced by the
    default. Bool and text columns get analogous treatment.

    Prices are in EUR. The column is named `price` (currency-agnostic);
    the EUR convention is documented in the app, not encoded in the
    schema.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    # Guard against frames built from list rows where columns are 0/1/2...
    if isinstance(df.columns, pd.RangeIndex) or all(
        isinstance(col, (int, float)) for col in df.columns
    ):
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    out = df.copy()

    for col in EXPECTED_COLUMNS:
        if col not in out.columns:
            out[col] = DEFAULTS[col]

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(DEFAULTS[col])

    # `id` is int4 in Postgres, not float. After to_numeric it'd be
    # 5.0; cast back to int so .eq("id", 5) lands cleanly. The other
    # numerics stay as floats (acidity/score/price all benefit from
    # decimals).
    out["id"] = out["id"].astype(int)

    # pandas 2.x emits a FutureWarning when fillna() silently downcasts
    # an object column (here NaN/None -> bool). Opt into the future
    # no-downcast behaviour and do the bool cast explicitly: the result
    # is identical, just without the deprecation noise.
    with pd.option_context("future.no_silent_downcasting", True):
        for col in BOOL_COLUMNS:
            out[col] = out[col].fillna(False).astype(bool)

    for col in TEXT_COLUMNS:
        out[col] = out[col].fillna(DEFAULTS[col]).astype(str)

    # Stable column order so downstream consumers can rely on it.
    return out[EXPECTED_COLUMNS]


def load_history(sb: Client) -> pd.DataFrame:
    """Load full wine history from Supabase, normalised for downstream use.

    Returns an empty DataFrame with the correct schema if the table is
    empty or the query fails to return data. Errors from the Supabase
    client itself are propagated; callers decide how to render them.
    """
    raw = repo_list_wines(sb)
    return normalize(raw)
