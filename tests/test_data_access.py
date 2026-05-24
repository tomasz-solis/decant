"""Tests for decant.services.data_access.

Covers two responsibilities:
1. `normalize` — schema-stable DataFrames from messy inputs.
2. `load_history` — wrapping the Supabase repo with normalisation.
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
