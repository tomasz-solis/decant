"""Tests for VinoPredictor's DataFrame-based constructor.

Phase 1 change: predictor no longer reads from disk. These tests pin
the new contract and catch regressions if anyone wires CSV reading
back in.
"""

import os
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _stub_openai_key():
    """Provide a dummy key so VinoPredictor.__init__ doesn't raise."""
    original = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key-not-real"
    yield
    if original is None:
        os.environ.pop("OPENAI_API_KEY", None)
    else:
        os.environ["OPENAI_API_KEY"] = original


@pytest.fixture
def empty_history():
    return pd.DataFrame()


@pytest.fixture
def sample_history():
    """3 liked + 2 disliked wines covering both colours."""
    return pd.DataFrame([
        {"wine_name": "Albariño A", "producer": "P1", "liked": True,
         "wine_color": "White", "acidity": 9, "minerality": 9, "fruitiness": 7,
         "tannin": 1, "body": 5, "price": 20.0, "vintage": 2022},
        {"wine_name": "Albariño B", "producer": "P2", "liked": True,
         "wine_color": "White", "acidity": 8, "minerality": 8, "fruitiness": 8,
         "tannin": 1, "body": 6, "price": 18.0, "vintage": 2021},
        {"wine_name": "Rioja", "producer": "P3", "liked": True,
         "wine_color": "Red", "acidity": 6, "minerality": 5, "fruitiness": 7,
         "tannin": 7, "body": 8, "price": 25.0, "vintage": 2018},
        {"wine_name": "Bad White", "producer": "P4", "liked": False,
         "wine_color": "White", "acidity": 3, "minerality": 2, "fruitiness": 9,
         "tannin": 1, "body": 8, "price": 12.0, "vintage": 2020},
        {"wine_name": "Bad Red", "producer": "P5", "liked": False,
         "wine_color": "Red", "acidity": 4, "minerality": 2, "fruitiness": 9,
         "tannin": 3, "body": 7, "price": 10.0, "vintage": 2019},
    ])


class TestConstructor:
    """The new DataFrame-based constructor."""

    def test_empty_history_initialises_cleanly(self, empty_history):
        from decant.predictor import VinoPredictor
        with patch("decant.predictor.OpenAI"), patch("decant.predictor.get_global_limiter"):
            predictor = VinoPredictor(history_df=empty_history)
            assert len(predictor.df) == 0
            assert len(predictor.liked_examples) == 0
            assert len(predictor.disliked_examples) == 0

    def test_history_populates_examples(self, sample_history):
        from decant.predictor import VinoPredictor
        with patch("decant.predictor.OpenAI"), patch("decant.predictor.get_global_limiter"):
            predictor = VinoPredictor(history_df=sample_history)
            assert len(predictor.df) == 5
            assert len(predictor.liked_examples) > 0
            assert len(predictor.disliked_examples) > 0

    def test_none_history_treated_as_empty(self):
        from decant.predictor import VinoPredictor
        with patch("decant.predictor.OpenAI"), patch("decant.predictor.get_global_limiter"):
            predictor = VinoPredictor(history_df=None)
            assert len(predictor.df) == 0


class TestRefreshContext:
    """Calling refresh_context with new data updates the example sets."""

    def test_refresh_context_swaps_data(self, empty_history, sample_history):
        from decant.predictor import VinoPredictor
        with patch("decant.predictor.OpenAI"), patch("decant.predictor.get_global_limiter"):
            predictor = VinoPredictor(history_df=empty_history)
            assert len(predictor.df) == 0

            predictor.refresh_context(sample_history)
            assert len(predictor.df) == 5
            assert len(predictor.liked_examples) > 0


class TestPromptBuilding:
    """Regression tests for the price_usd/dollar bugs fixed in Phase 1."""

    def test_prompt_uses_price_column_not_price_usd(self, sample_history):
        """Reading wine['price_usd'] would KeyError; the fix uses 'price'."""
        from decant.predictor import VinoPredictor
        from decant.schema import WineFeatures
        with patch("decant.predictor.OpenAI"), patch("decant.predictor.get_global_limiter"):
            predictor = VinoPredictor(history_df=sample_history)
            features = WineFeatures(
                acidity=8, minerality=7, fruitiness=7, tannin=2, body=5,
                reasoning="test",
            )
            # Should not raise KeyError on 'price_usd'
            prompt = predictor._build_context_prompt(features)
            assert isinstance(prompt, str)

    def test_prompt_uses_euro_symbol(self, sample_history):
        """EUR pricing - prompt should use € not $."""
        from decant.predictor import VinoPredictor
        from decant.schema import WineFeatures
        with patch("decant.predictor.OpenAI"), patch("decant.predictor.get_global_limiter"):
            predictor = VinoPredictor(history_df=sample_history)
            features = WineFeatures(
                acidity=8, minerality=7, fruitiness=7, tannin=2, body=5,
                reasoning="test",
            )
            prompt = predictor._build_context_prompt(features)
            assert "€" in prompt
            assert "$" not in prompt or prompt.count("$") < prompt.count("€")
