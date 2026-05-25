"""Supabase client wiring for the household-account model.

Phase 2 simplification: there is one Supabase Auth user per household.
Users either sign in interactively from the UI or browse anonymously.
The previous `credential_map` keyed by Streamlit username is gone.

Public API:
    - get_anon_supabase()    Read-only client; safe to call without auth.
    - sign_in(email, pwd)    Authenticate and cache the session client.
    - sign_out()             Clear the session and cached client.
    - get_supabase_client()  Return the authed client if logged in, else
                             raise. Use this where write access is needed.
    - get_user_supabase()    Backward-compatible alias for the above.
    - is_authenticated()     True if a session exists in st.session_state.
    - request_password_reset(email)  Send a magic-link reset email.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st
from supabase import Client, create_client


# Session-state keys. Centralised so the auth form and the rest of the
# app agree on where the session lives.
_SESSION_CLIENT_KEY = "_supabase_session_client"
_SESSION_USER_KEY = "_supabase_session_user"
_SESSION_EXPIRES_AT_KEY = "_supabase_session_expires_at"
_ANON_CLIENT_KEY = "_supabase_anon_client"


def _normalize_secret(raw: object, name: str) -> str:
    """Strip whitespace and accidental quotes from a Streamlit secret."""
    if raw is None:
        raise ValueError(f"{name} is missing from secrets")
    value = str(raw).strip()
    quote_pairs = [('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019")]
    for left, right in quote_pairs:
        if len(value) >= 2 and value.startswith(left) and value.endswith(right):
            value = value[1:-1].strip()
            break
    if not value:
        raise ValueError(f"{name} is empty")
    return value


def _new_client() -> Client:
    """Create a fresh Supabase client from secrets."""
    url = _normalize_secret(st.secrets["SUPABASE_URL"], "SUPABASE_URL")
    key = _normalize_secret(st.secrets["SUPABASE_KEY"], "SUPABASE_KEY")
    return create_client(url, key)


def get_anon_supabase() -> Client:
    """Return a read-only client backed by the anon key.

    Suitable for guest browsing. RLS on the `wines` table must allow
    `SELECT` for the anonymous role to make this useful.
    """
    cached = st.session_state.get(_ANON_CLIENT_KEY)
    if cached is not None:
        return cached
    client = _new_client()
    st.session_state[_ANON_CLIENT_KEY] = client
    return client


def _session_expired() -> bool:
    """True if the cached session's access token has passed its expiry.

    Supabase access tokens default to a 1-hour TTL. The Supabase Python
    client refreshes automatically on its own auth calls but not on
    arbitrary `.table()` operations, so we check explicitly. When the
    token is expired we clear local state and force a re-sign-in rather
    than try to refresh — refresh logic adds complexity for a household
    app that's rarely left open across token boundaries.
    """
    expires_at = st.session_state.get(_SESSION_EXPIRES_AT_KEY)
    if expires_at is None:
        # No expiry recorded — assume the session is fine. This path is
        # only hit for sessions created before the field was tracked.
        return False
    import time
    return time.time() >= float(expires_at)


def _clear_expired_session() -> None:
    """Drop the cached session client and user when the token has expired."""
    st.session_state.pop(_SESSION_CLIENT_KEY, None)
    st.session_state.pop(_SESSION_USER_KEY, None)
    st.session_state.pop(_SESSION_EXPIRES_AT_KEY, None)


def is_authenticated() -> bool:
    """True if a non-expired Supabase session exists in this Streamlit session.

    If a cached session has passed its access-token expiry, the cached
    state is cleared and this returns False — the next interaction will
    prompt the user to sign in again. This trades automatic refresh for
    a simpler model: sessions are valid until they expire, then you
    re-sign-in.
    """
    if st.session_state.get(_SESSION_CLIENT_KEY) is None:
        return False
    if _session_expired():
        _clear_expired_session()
        return False
    return True


def current_user_email() -> Optional[str]:
    """Email of the logged-in user, or None if anonymous."""
    return st.session_state.get(_SESSION_USER_KEY)


def sign_in(email: str, password: str) -> tuple[bool, Optional[str]]:
    """Attempt sign-in. Returns (success, error_message_or_None).

    On success, caches the authenticated client in session state so
    subsequent `get_supabase_client()` calls return it without another
    round-trip to Supabase.

    On failure, the returned error message is user-safe: either a
    canonical "Invalid login credentials" or a generic "Sign-in
    failed." The original exception is logged for debugging but never
    returned to the caller, since the caller is likely to display it.
    """
    import logging
    log = logging.getLogger(__name__)

    try:
        client = _new_client()
        response = client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        session = getattr(response, "session", None)
        if session is None or not session.access_token or not session.refresh_token:
            return False, "Sign-in failed."

        client.auth.set_session(session.access_token, session.refresh_token)
        st.session_state[_SESSION_CLIENT_KEY] = client
        st.session_state[_SESSION_USER_KEY] = email
        st.session_state[_SESSION_EXPIRES_AT_KEY] = getattr(session, "expires_at", None)
        return True, None
    except Exception as exc:
        # Map known-safe Supabase auth messages; mask everything else.
        # Operational details (DB connection, rate limit, internal errors)
        # are logged but not surfaced to the user.
        msg = str(exc).lower()
        if "invalid login credentials" in msg or "invalid email or password" in msg:
            return False, "Invalid email or password."
        if "email not confirmed" in msg:
            return False, "Email not confirmed. Check your inbox."
        log.warning("Sign-in failed (unexpected error): %s", exc)
        return False, "Sign-in failed. Try again in a moment."


def sign_out() -> None:
    """Clear the session-state client and forget the user."""
    client = st.session_state.pop(_SESSION_CLIENT_KEY, None)
    st.session_state.pop(_SESSION_USER_KEY, None)
    st.session_state.pop(_SESSION_EXPIRES_AT_KEY, None)
    if client is not None:
        try:
            client.auth.sign_out()
        except Exception:
            # Best-effort sign-out; if the token is already expired
            # server-side we don't care, we just want the client gone.
            pass


def get_supabase_client() -> Client:
    """Return the authenticated client. Raises if not logged in or expired.

    Use this where a write or RLS-protected read is required. For
    guest-safe reads, call `get_anon_supabase()` directly.
    """
    if not is_authenticated():
        # is_authenticated() already cleared expired state, so the
        # message is the same whether we're freshly anonymous or
        # transitioning from expired-session to anonymous.
        raise RuntimeError(
            "No authenticated Supabase session. Sign in via the sidebar."
        )
    return st.session_state[_SESSION_CLIENT_KEY]


def get_user_supabase() -> Client:
    """Backward-compatible alias for `get_supabase_client`."""
    return get_supabase_client()


def request_password_reset(email: str) -> tuple[bool, Optional[str]]:
    """Trigger a Supabase magic-link password reset email.

    Returns (success, error_message_or_None). Note: Supabase always
    returns success for unknown emails (to prevent enumeration), so a
    True result doesn't guarantee the email exists.
    """
    try:
        client = _new_client()
        client.auth.reset_password_for_email(email)
        return True, None
    except Exception as exc:
        return False, str(exc)
