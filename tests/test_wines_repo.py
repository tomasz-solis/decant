"""Tests for decant.wines_repo.repo_update_wine.

The Supabase client is mocked. We're testing the function's contract —
field filtering, the query filters, and what it returns — not the
network round-trip.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from decant.wines_repo import repo_update_wine


@pytest.fixture
def mock_sb():
    """A mock Supabase client whose .update() chain is fully recordable.

    Every fluent method returns the same mock so we can assert on the
    full chain of calls (.update -> .eq -> .eq -> .execute).
    """
    sb = MagicMock()
    chain = sb.table.return_value
    chain.update.return_value = chain
    chain.eq.return_value = chain
    # Default execute response: success, one row returned.
    chain.execute.return_value = MagicMock(data=[{"id": 42, "vintage": 2021}])
    return sb


@pytest.fixture(autouse=True)
def mock_cellar_id():
    """Pin the cellar id so we can assert on what gets filtered."""
    with patch("decant.wines_repo._get_cellar_id", return_value="cellar-test"):
        yield


class TestRepoUpdateWine:

    def test_updates_allowed_field(self, mock_sb):
        result = repo_update_wine(mock_sb, 42, {"vintage": 2021})

        mock_sb.table.assert_called_once_with("wines")
        chain = mock_sb.table.return_value
        chain.update.assert_called_once_with({"vintage": 2021})
        assert result == {"id": 42, "vintage": 2021}

    def test_filters_by_id_and_cellar(self, mock_sb):
        repo_update_wine(mock_sb, 42, {"vintage": 2021})

        chain = mock_sb.table.return_value
        # Both filters must be applied — id for the row, cellar_id as
        # a defense in depth against RLS misconfiguration.
        eq_calls = chain.eq.call_args_list
        assert ("id", 42) in [call.args for call in eq_calls]
        assert ("cellar_id", "cellar-test") in [call.args for call in eq_calls]

    def test_silently_drops_unknown_fields(self, mock_sb):
        repo_update_wine(mock_sb, 42, {
            "vintage": 2021,
            "made_up_field": "x",
            "another_garbage": 42,
        })

        chain = mock_sb.table.return_value
        # Only the known field is forwarded; the other two vanish.
        chain.update.assert_called_once_with({"vintage": 2021})

    def test_silently_drops_flavor_features(self, mock_sb):
        """Flavour features must NOT be editable via this entry point.

        Changing acidity/fruitiness/body/tannin/minerality affects
        every downstream palate score (they shift the population mean
        and ideal profile). If we ever expose feature editing, it
        needs its own path with an explicit blast-radius warning.
        """
        repo_update_wine(mock_sb, 42, {
            "vintage": 2021,
            "acidity": 9.5,
            "fruitiness": 6.0,
            "body": 5.0,
            "tannin": 4.0,
            "minerality": 8.0,
        })

        chain = mock_sb.table.return_value
        chain.update.assert_called_once_with({"vintage": 2021})

    def test_silently_drops_id_and_metadata_columns(self, mock_sb):
        """Row identity (id), cellar membership (cellar_id), and
        Supabase-managed columns (created_at, updated_at) are never
        editable via this path.

        wine_name IS editable — see test_renames_wine_name. That's a
        deliberate exception: the original LLM extraction sometimes
        misses words (e.g. '1er' in a Premier Cru name) and there's
        no other way to correct it. The trade-offs are documented in
        repo_update_wine's docstring.
        """
        repo_update_wine(mock_sb, 42, {
            "vintage": 2021,
            "id": 9999,
            "cellar_id": "different-cellar",
            "created_at": "2020-01-01",
            "updated_at": "2020-01-01",
        })

        chain = mock_sb.table.return_value
        # vintage round-trips; everything else gets dropped.
        chain.update.assert_called_once_with({"vintage": 2021})

    def test_renames_wine_name(self, mock_sb):
        """wine_name is in the editable allowlist (unlike id/cellar_id).

        Pins the deliberate exception so future readers don't see
        wine_name in _EDITABLE_FIELDS and think it's a mistake.
        """
        repo_update_wine(mock_sb, 42, {
            "wine_name": "Christophe Drain Givry 1er Crausot",
        })

        chain = mock_sb.table.return_value
        chain.update.assert_called_once_with({
            "wine_name": "Christophe Drain Givry 1er Crausot",
        })

    def test_returns_empty_when_no_allowed_fields(self, mock_sb):
        """If the caller passes only garbage, we never hit Supabase.

        Avoids a no-op UPDATE that would still cost a round-trip.
        """
        result = repo_update_wine(mock_sb, 42, {
            "made_up": "x",
            "acidity": 9.0,  # disallowed
        })

        assert result == {}
        mock_sb.table.assert_not_called()

    def test_returns_empty_when_supabase_returns_no_rows(self, mock_sb):
        """E.g. the wine doesn't exist or belongs to another cellar.

        The cellar_id filter would silently exclude rows the caller
        can't update; we return {} so the caller can surface a
        'not found' state to the user.
        """
        chain = mock_sb.table.return_value
        chain.execute.return_value = MagicMock(data=[])

        result = repo_update_wine(mock_sb, 9999, {"vintage": 2021})

        assert result == {}

    def test_accepts_all_editable_metadata_fields(self, mock_sb):
        """The full editable field set should round-trip in one call."""
        fields = {
            "wine_name": "Test Wine 1er Cru",
            "vintage": 2021,
            "producer": "Domaine Test",
            "region": "Burgundy",
            "country": "France",
            "price": 25.0,
            "score": 8.5,
            "liked": True,
            "notes": "Crisp, mineral",
            "wine_color": "White",
            "is_sparkling": False,
            "is_natural": True,
            "sweetness": "Dry",
        }
        repo_update_wine(mock_sb, 42, fields)

        chain = mock_sb.table.return_value
        chain.update.assert_called_once_with(fields)

    def test_accepts_partial_update(self, mock_sb):
        """Only the fields you pass should be sent."""
        repo_update_wine(mock_sb, 42, {"vintage": 2021, "notes": "Updated"})

        chain = mock_sb.table.return_value
        chain.update.assert_called_once_with({"vintage": 2021, "notes": "Updated"})
