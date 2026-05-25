"""Tests for decant.supabase_session.

The interesting behaviour to pin is session-state management: caching,
clearing on sign-out, and the is_authenticated / get_supabase_client
contract. The actual Supabase round-trip is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_st(monkeypatch):
    """Patch `st.session_state` and `st.secrets` so the module under test
    can run without a real Streamlit runtime.
    """
    fake_session = {}
    fake_secrets = {
        "SUPABASE_URL": "https://fake.supabase.co",
        "SUPABASE_KEY": "fake-anon-key",
    }

    class FakeSecrets:
        def __getitem__(self, key):
            if key not in fake_secrets:
                raise KeyError(key)
            return fake_secrets[key]

        def get(self, key, default=None):
            return fake_secrets.get(key, default)

    import streamlit as st
    monkeypatch.setattr(st, "session_state", fake_session)
    monkeypatch.setattr(st, "secrets", FakeSecrets())
    return fake_session


class TestIsAuthenticated:
    """The is_authenticated check is the gate for all write operations."""

    def test_false_when_no_session(self, fake_st):
        from decant.supabase_session import is_authenticated
        assert is_authenticated() is False

    def test_true_when_session_present(self, fake_st):
        from decant.supabase_session import is_authenticated, _SESSION_CLIENT_KEY
        fake_st[_SESSION_CLIENT_KEY] = MagicMock()
        assert is_authenticated() is True


class TestGetSupabaseClient:
    """Returns the cached client when present; raises when not."""

    def test_returns_cached_client(self, fake_st):
        from decant.supabase_session import get_supabase_client, _SESSION_CLIENT_KEY
        client = MagicMock(name="cached")
        fake_st[_SESSION_CLIENT_KEY] = client
        assert get_supabase_client() is client

    def test_raises_when_no_session(self, fake_st):
        from decant.supabase_session import get_supabase_client
        with pytest.raises(RuntimeError, match="No authenticated Supabase session"):
            get_supabase_client()


class TestSignIn:
    """Sign-in stores the client + email on success, returns error on failure."""

    def test_success_caches_client_and_email(self, fake_st):
        from decant.supabase_session import sign_in, _SESSION_CLIENT_KEY, _SESSION_USER_KEY

        fake_session = MagicMock()
        fake_session.access_token = "access"
        fake_session.refresh_token = "refresh"
        fake_response = MagicMock(session=fake_session)

        with patch("decant.supabase_session.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.auth.sign_in_with_password.return_value = fake_response
            mock_create.return_value = mock_client

            ok, err = sign_in("user@example.com", "password")

        assert ok is True
        assert err is None
        assert fake_st[_SESSION_CLIENT_KEY] is mock_client
        assert fake_st[_SESSION_USER_KEY] == "user@example.com"

    def test_failure_returns_safe_error_message(self, fake_st):
        """v2: error messages are user-safe. The raw exception isn't surfaced;
        known auth failures map to canonical messages."""
        from decant.supabase_session import sign_in, _SESSION_CLIENT_KEY

        with patch("decant.supabase_session.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.auth.sign_in_with_password.side_effect = Exception(
                "Invalid login credentials"
            )
            mock_create.return_value = mock_client

            ok, err = sign_in("user@example.com", "wrong")

        assert ok is False
        assert err == "Invalid email or password."
        assert _SESSION_CLIENT_KEY not in fake_st

    def test_unknown_failure_masked_to_generic_message(self, fake_st):
        """Operational errors (DB failures, infra issues) must not leak."""
        from decant.supabase_session import sign_in

        with patch("decant.supabase_session.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.auth.sign_in_with_password.side_effect = Exception(
                "Database connection error: pgbouncer pool exhausted at host xyz"
            )
            mock_create.return_value = mock_client

            ok, err = sign_in("user@example.com", "any")

        assert ok is False
        assert "pgbouncer" not in err
        assert "Database" not in err
        assert "try again" in err.lower()

    def test_no_session_returned_is_failure(self, fake_st):
        """If Supabase returns a response with no session object,
        treat as auth failure rather than silently caching nothing."""
        from decant.supabase_session import sign_in, _SESSION_CLIENT_KEY

        with patch("decant.supabase_session.create_client") as mock_create:
            mock_client = MagicMock()
            mock_client.auth.sign_in_with_password.return_value = MagicMock(session=None)
            mock_create.return_value = mock_client

            ok, err = sign_in("user@example.com", "password")

        assert ok is False
        assert _SESSION_CLIENT_KEY not in fake_st


class TestSignOut:
    """Sign-out clears session state regardless of upstream success."""

    def test_clears_session_keys(self, fake_st):
        from decant.supabase_session import (
            sign_out,
            _SESSION_CLIENT_KEY,
            _SESSION_USER_KEY,
        )
        client = MagicMock()
        fake_st[_SESSION_CLIENT_KEY] = client
        fake_st[_SESSION_USER_KEY] = "user@example.com"

        sign_out()

        assert _SESSION_CLIENT_KEY not in fake_st
        assert _SESSION_USER_KEY not in fake_st
        client.auth.sign_out.assert_called_once()

    def test_sign_out_when_already_logged_out_does_not_crash(self, fake_st):
        from decant.supabase_session import sign_out
        # No session in state; should not raise
        sign_out()

    def test_remote_sign_out_failure_still_clears_local(self, fake_st):
        """If Supabase's sign-out fails (expired token, network), we
        still want the local session cleared."""
        from decant.supabase_session import (
            sign_out,
            _SESSION_CLIENT_KEY,
            _SESSION_USER_KEY,
            is_authenticated,
        )
        client = MagicMock()
        client.auth.sign_out.side_effect = Exception("token expired")
        fake_st[_SESSION_CLIENT_KEY] = client
        fake_st[_SESSION_USER_KEY] = "user@example.com"

        sign_out()  # should not raise

        assert is_authenticated() is False


class TestNormalizeSecret:
    """Defensive parsing of TOML secrets (strips quotes, whitespace)."""

    def test_strips_whitespace(self):
        from decant.supabase_session import _normalize_secret
        assert _normalize_secret("  hello  ", "X") == "hello"

    def test_strips_double_quotes(self):
        from decant.supabase_session import _normalize_secret
        assert _normalize_secret('"hello"', "X") == "hello"

    def test_strips_single_quotes(self):
        from decant.supabase_session import _normalize_secret
        assert _normalize_secret("'hello'", "X") == "hello"

    def test_strips_smart_quotes(self):
        from decant.supabase_session import _normalize_secret
        assert _normalize_secret("\u201chello\u201d", "X") == "hello"

    def test_raises_on_none(self):
        from decant.supabase_session import _normalize_secret
        with pytest.raises(ValueError, match="missing"):
            _normalize_secret(None, "FOO")

    def test_raises_on_empty(self):
        from decant.supabase_session import _normalize_secret
        with pytest.raises(ValueError, match="empty"):
            _normalize_secret("   ", "FOO")


class TestSessionExpiry:
    """Phase 2 fix: is_authenticated detects expired access tokens.

    Supabase access tokens default to a 1-hour TTL. The old code would
    return True for an expired session, then fail with cryptic RLS
    errors on the next query. The fix surfaces expiry at the gate.
    """

    def test_fresh_session_is_authenticated(self, fake_st):
        from decant.supabase_session import (
            is_authenticated,
            _SESSION_CLIENT_KEY,
            _SESSION_EXPIRES_AT_KEY,
        )
        import time

        fake_st[_SESSION_CLIENT_KEY] = MagicMock()
        fake_st[_SESSION_EXPIRES_AT_KEY] = time.time() + 3600
        assert is_authenticated() is True

    def test_expired_session_is_not_authenticated(self, fake_st):
        from decant.supabase_session import (
            is_authenticated,
            _SESSION_CLIENT_KEY,
            _SESSION_EXPIRES_AT_KEY,
        )
        import time

        fake_st[_SESSION_CLIENT_KEY] = MagicMock()
        fake_st[_SESSION_EXPIRES_AT_KEY] = time.time() - 3600
        assert is_authenticated() is False

    def test_expired_session_state_is_cleared(self, fake_st):
        from decant.supabase_session import (
            is_authenticated,
            _SESSION_CLIENT_KEY,
            _SESSION_USER_KEY,
            _SESSION_EXPIRES_AT_KEY,
        )
        import time

        fake_st[_SESSION_CLIENT_KEY] = MagicMock()
        fake_st[_SESSION_USER_KEY] = "a@b.com"
        fake_st[_SESSION_EXPIRES_AT_KEY] = time.time() - 1

        is_authenticated()

        assert _SESSION_CLIENT_KEY not in fake_st
        assert _SESSION_USER_KEY not in fake_st
        assert _SESSION_EXPIRES_AT_KEY not in fake_st

    def test_session_without_expiry_treated_as_valid(self, fake_st):
        """Backward compat: sessions cached before expiry tracking existed
        should still authenticate."""
        from decant.supabase_session import is_authenticated, _SESSION_CLIENT_KEY
        fake_st[_SESSION_CLIENT_KEY] = MagicMock()
        # No expires_at set.
        assert is_authenticated() is True

    def test_get_client_raises_when_expired(self, fake_st):
        from decant.supabase_session import (
            get_supabase_client,
            _SESSION_CLIENT_KEY,
            _SESSION_EXPIRES_AT_KEY,
        )
        import time

        fake_st[_SESSION_CLIENT_KEY] = MagicMock()
        fake_st[_SESSION_EXPIRES_AT_KEY] = time.time() - 1

        with pytest.raises(RuntimeError, match="Sign in"):
            get_supabase_client()

    def test_sign_out_clears_expires_at(self, fake_st):
        """The sign_out fix: previously only cleared client+user keys."""
        from decant.supabase_session import (
            sign_out,
            _SESSION_CLIENT_KEY,
            _SESSION_USER_KEY,
            _SESSION_EXPIRES_AT_KEY,
        )

        fake_st[_SESSION_CLIENT_KEY] = MagicMock()
        fake_st[_SESSION_USER_KEY] = "a@b.com"
        fake_st[_SESSION_EXPIRES_AT_KEY] = 999999999

        sign_out()

        assert _SESSION_EXPIRES_AT_KEY not in fake_st
