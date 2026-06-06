"""Tests for decant.services.data_access.

Covers two responsibilities:
1. `normalize` - schema-stable DataFrames from messy inputs.
2. `load_history` - wrapping the Supabase repo with normalisation.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from decant.services.data_access import (
    DEFAULTS,
    EXPECTED_COLUMNS,
    load_history,
    normalize,
)


class TestNormalize:
    """Behaviour of the normalize() function on various inputs."""

    def test_none_returns_empty_with_schema(self):
        result = normalize(None)
        assert list(result.columns) == EXPECTED_COLUMNS
        assert len(result) == 0

    def test_non_dataframe_returns_empty_with_schema(self):
        result = normalize("not a dataframe")
        assert list(result.columns) == EXPECTED_COLUMNS
        assert len(result) == 0

    def test_empty_dataframe_returns_empty_with_schema(self):
        result = normalize(pd.DataFrame())
        assert list(result.columns) == EXPECTED_COLUMNS

    def test_well_formed_input_preserved(self):
        df = pd.DataFrame([{
            "wine_name": "Albariño",
            "producer": "Martín Códax",
            "vintage": 2022,
            "notes": "Crisp",
            "score": 8.0,
            "liked": True,
            "price": 18.5,
            "country": "Spain",
            "region": "Rías Baixas",
            "wine_color": "White",
            "is_sparkling": False,
            "is_natural": False,
            "sweetness": "Dry",
            "acidity": 9.0,
            "minerality": 8.0,
            "fruitiness": 7.0,
            "tannin": 1.0,
            "body": 5.0,
        }])
        result = normalize(df)
        assert len(result) == 1
        assert result.iloc[0]["wine_name"] == "Albariño"
        assert result.iloc[0]["liked"] is True or result.iloc[0]["liked"]
        assert result.iloc[0]["acidity"] == 9.0

    def test_missing_columns_filled_with_defaults(self):
        df = pd.DataFrame([{"wine_name": "Sketch", "liked": True}])
        result = normalize(df)
        assert result.iloc[0]["producer"] == DEFAULTS["producer"]
        assert result.iloc[0]["price"] == DEFAULTS["price"]
        assert result.iloc[0]["acidity"] == DEFAULTS["acidity"]

    def test_numeric_string_coerced(self):
        df = pd.DataFrame([{"wine_name": "X", "acidity": "8.5", "body": "7"}])
        result = normalize(df)
        assert result.iloc[0]["acidity"] == 8.5
        assert result.iloc[0]["body"] == 7.0

    def test_unparseable_numeric_becomes_default(self):
        df = pd.DataFrame([{"wine_name": "X", "acidity": "not a number"}])
        result = normalize(df)
        assert result.iloc[0]["acidity"] == DEFAULTS["acidity"]

    def test_bool_nan_becomes_false(self):
        df = pd.DataFrame([{"wine_name": "X", "liked": None}])
        result = normalize(df)
        assert result.iloc[0]["liked"] is False or result.iloc[0]["liked"] == False  # noqa: E712

    def test_rangeindex_columns_returns_empty(self):
        """Frames built from list rows have integer column indices."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        result = normalize(df)
        assert list(result.columns) == EXPECTED_COLUMNS
        assert len(result) == 0

    def test_id_column_preserved(self):
        """Supabase int4 primary key must survive normalize.

        Regression guard: the original schema had no id column, so
        normalize stripped it via the EXPECTED_COLUMNS allowlist.
        The Gallery edit feature needs the id to update a row, so
        id was added to the allowlist. This test pins that.
        """
        df = pd.DataFrame([{"id": 42, "wine_name": "X"}])
        result = normalize(df)
        assert "id" in result.columns
        assert result.iloc[0]["id"] == 42

    def test_id_coerced_to_int_not_float(self):
        """id is int4 in Postgres. pd.to_numeric defaults to float;
        we cast back to int so .eq("id", 5) lands cleanly instead of
        receiving 5.0.
        """
        df = pd.DataFrame([{"id": 42, "wine_name": "X"}])
        result = normalize(df)
        assert result.iloc[0]["id"] == 42
        # Pandas may use int64; pin that it's an integer type, not float.
        assert pd.api.types.is_integer_dtype(result["id"])

    def test_id_defaults_to_zero_when_missing(self):
        """If a frame somehow arrives without id (legacy callers,
        test fixtures), default to 0 - consumers treat that as
        'no id available' and skip the edit affordance.
        """
        df = pd.DataFrame([{"wine_name": "X"}])
        result = normalize(df)
        assert result.iloc[0]["id"] == 0


class TestLoadHistory:
    """load_history wires the repo to normalize. Don't re-test repo logic here."""

    def test_load_history_normalises_repo_output(self, monkeypatch):
        fake_df = pd.DataFrame([{
            "wine_name": "Test Wine",
            "liked": True,
            "acidity": "8",  # string, should be coerced
        }])
        monkeypatch.setattr(
            "decant.services.data_access.repo_list_wines",
            lambda sb: fake_df,
        )
        result = load_history(MagicMock())
        assert list(result.columns) == EXPECTED_COLUMNS
        assert result.iloc[0]["acidity"] == 8.0
        assert result.iloc[0]["producer"] == DEFAULTS["producer"]

    def test_load_history_handles_empty_repo(self, monkeypatch):
        monkeypatch.setattr(
            "decant.services.data_access.repo_list_wines",
            lambda sb: pd.DataFrame(),
        )
        result = load_history(MagicMock())
        assert len(result) == 0
        assert list(result.columns) == EXPECTED_COLUMNS


class TestSchemaSingleSource:
    """Phase 3 Chunk 2: the wine schema lives in data_access only.

    Before Chunk 2, app.py duplicated the schema constants. These
    tests pin that the consolidation actually consolidated - if
    someone re-introduces a duplicate, the tests fail.
    """

    def test_app_module_imports_normalize_from_data_access(self):
        """app.py exposes ensure_wine_df, but it's an alias for normalize."""
        import sys
        from pathlib import Path
        # Add repo root so `import app` works
        repo_root = Path(__file__).parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        # Importing app at test time would trigger Streamlit; instead, read
        # the source and check the import statement directly.
        app_source = (repo_root / "app.py").read_text()
        assert "from decant.services.data_access import normalize as ensure_wine_df" in app_source, (
            "app.py must import ensure_wine_df as an alias for "
            "data_access.normalize, not define its own."
        )

    def test_no_duplicate_schema_constants_in_app(self):
        """Catch regressions where someone re-adds the constants to app.py."""
        from pathlib import Path
        app_source = (Path(__file__).parent.parent / "app.py").read_text()
        forbidden_constants = [
            "EXPECTED_WINE_COLUMNS = [",
            "NUMERIC_WINE_COLUMNS = [",
            "BOOL_WINE_COLUMNS = [",
            "TEXT_WINE_COLUMNS = [",
            "DEFAULT_WINE_VALUES = {",
        ]
        for constant in forbidden_constants:
            assert constant not in app_source, (
                f"{constant!r} found in app.py - schema lives in "
                "decant.services.data_access only."
            )
