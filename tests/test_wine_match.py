"""Tests for decant.services.wine_match.

Pure-function module, so tests are straightforward set-based assertions
on the tokenizer and DataFrame-driven assertions on the matcher.
"""

from __future__ import annotations

import pandas as pd
import pytest

from decant.services.wine_match import (
    PriorTasting,
    find_prior_tasting,
    tokenize_wine,
)


class TestTokenizeWine:
    """Tokenization basics."""

    def test_lowercases(self):
        assert tokenize_wine("Drain Givry") == frozenset({"drain", "givry"})

    def test_strips_punctuation(self):
        assert tokenize_wine("Drain, Givry 1er Cru") == frozenset({"drain", "givry"})

    def test_drops_stopwords(self):
        # "domaine", "1er", "cru" are stopwords
        assert tokenize_wine("Domaine Drain Givry 1er Cru Crausot") == frozenset(
            {"drain", "givry", "crausot"}
        )

    def test_drops_short_tokens(self):
        # Threshold is < 2 chars dropped, so single chars go but 2-char stays.
        assert tokenize_wine("a b ch xy abc") == frozenset({"ch", "xy", "abc"})

    def test_handles_none(self):
        assert tokenize_wine(None) == frozenset()

    def test_handles_empty(self):
        assert tokenize_wine("") == frozenset()

    def test_handles_only_stopwords(self):
        assert tokenize_wine("Domaine Chateau Cru") == frozenset()

    def test_preserves_diacritics(self):
        # 'château' is in stopwords; 'côte' is not.
        result = tokenize_wine("Château Côte de Beaune")
        assert "côte" in result
        assert "beaune" in result
        assert "château" not in result


class TestFindPriorTasting:
    """End-to-end matching against a small history DataFrame."""

    @pytest.fixture
    def history(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "wine_name": "Givry 1er Cru Crausot",
                "producer": "Domaine Christophe Drain",
                "vintage": 2021,
                "score": 9.0,
                "liked": True,
            },
            {
                "wine_name": "Albariño Reserva",
                "producer": "Adega do Mar",
                "vintage": 2022,
                "score": 7.5,
                "liked": True,
            },
            {
                "wine_name": "Macon Villages",
                "producer": "Drain Frères",  # different producer than Christophe Drain
                "vintage": 2020,
                "score": 6.0,
                "liked": False,
            },
        ])

    def test_exact_match_same_vintage(self, history):
        result = find_prior_tasting(
            "Givry 1er Cru Crausot",
            "Domaine Christophe Drain",
            2021,
            history,
        )
        assert result is not None
        assert result.match_kind == "exact"
        assert result.vintage == 2021
        assert result.score == 9.0
        assert result.liked is True

    def test_different_vintage_match(self, history):
        # Same wine, different year
        result = find_prior_tasting(
            "Givry 1er Cru Crausot",
            "Domaine Christophe Drain",
            2022,
            history,
        )
        assert result is not None
        assert result.match_kind == "different_vintage"
        assert result.vintage == 2021  # the prior tasting's vintage
        assert result.score == 9.0

    def test_loose_producer_string_still_matches(self, history):
        # Just "Christophe Drain", history has "Domaine Christophe Drain"
        result = find_prior_tasting(
            "Givry Crausot",
            "Christophe Drain",
            2021,
            history,
        )
        assert result is not None
        assert result.match_kind == "exact"

    def test_different_producer_no_match(self, history):
        # Different producer entirely, even same wine name
        result = find_prior_tasting(
            "Macon Villages",
            "Jean-Paul Dupont",
            2020,
            history,
        )
        assert result is None

    def test_similar_producer_unrelated_wine_no_match(self, history):
        # Same producer (Christophe Drain) but a wine name with no token overlap
        result = find_prior_tasting(
            "Bourgogne Aligoté",
            "Domaine Christophe Drain",
            2021,
            history,
        )
        # Producer matches, but name has zero overlap with anything
        # in their history — no match.
        assert result is None

    def test_empty_history(self):
        result = find_prior_tasting(
            "Anything",
            "Anyone",
            2020,
            pd.DataFrame(),
        )
        assert result is None

    def test_none_history(self):
        result = find_prior_tasting("Anything", "Anyone", 2020, None)  # type: ignore[arg-type]
        assert result is None

    def test_unknown_vintage_on_candidate_is_not_exact(self, history):
        # Candidate vintage = 0 (schema sentinel for unknown)
        result = find_prior_tasting(
            "Givry 1er Cru Crausot",
            "Domaine Christophe Drain",
            0,
            history,
        )
        # Producer + name match -> should match, but as different_vintage
        # since we can't verify exact.
        assert result is not None
        assert result.match_kind == "different_vintage"

    def test_no_producer_falls_back_to_higher_name_threshold(self, history):
        # No producer given; name match must be stronger
        # "Givry 1er Cru Crausot" tokens vs "Givry 1er Cru Crausot" = 1.0, passes
        result = find_prior_tasting(
            "Givry 1er Cru Crausot",
            None,
            2021,
            history,
        )
        assert result is not None

        # Weaker name match without producer should fail (just "Givry"
        # vs "Givry 1er Cru Crausot" = 1/3 token overlap, below 0.7)
        result_weak = find_prior_tasting(
            "Givry",
            None,
            2021,
            history,
        )
        assert result_weak is None

    def test_exact_match_preferred_over_different_vintage(self):
        """When both 2020 and 2021 exist in history, asking about 2021 picks 2021."""
        history = pd.DataFrame([
            {
                "wine_name": "Givry Crausot",
                "producer": "Domaine Christophe Drain",
                "vintage": 2020,
                "score": 7.0,
                "liked": True,
            },
            {
                "wine_name": "Givry Crausot",
                "producer": "Domaine Christophe Drain",
                "vintage": 2021,
                "score": 9.0,
                "liked": True,
            },
        ])
        result = find_prior_tasting(
            "Givry Crausot",
            "Domaine Christophe Drain",
            2021,
            history,
        )
        assert result is not None
        assert result.match_kind == "exact"
        assert result.vintage == 2021
        assert result.score == 9.0
