"""Streamlit UI for the household sign-in flow.

Renders a compact login form in the sidebar (or wherever the caller
places it). On success, the rest of the app sees `is_authenticated()`
return True and can call `get_supabase_client()` freely.

No signup UI. Accounts are pre-created in the Supabase dashboard for
this app's "single household account" model.
"""

from __future__ import annotations

import streamlit as st

from decant.supabase_session import (
    current_user_email,
    is_authenticated,
    request_password_reset,
    sign_in,
    sign_out,
)


def render_auth_block(contact_email: str = "tomasz@example.com") -> None:
    """Render the auth UI (login form when logged out, sign-out when in).

    Args:
        contact_email: Address shown in the "want access?" mailto link
            beneath the login form. Pulled from secrets in the caller.
    """
    if is_authenticated():
        _render_logged_in()
    else:
        _render_login_form(contact_email)


def _render_logged_in() -> None:
    """Compact 'signed in as X' block with sign-out button."""
    email = current_user_email() or "household"
    st.markdown(f"**Signed in as** `{email}`")
    if st.button("Sign out", key="auth_signout"):
        sign_out()
        st.rerun()


def _render_login_form(contact_email: str) -> None:
    """Email + password form with reset link and access mailto."""
    st.markdown("### Sign in")

    # Reset-password mode shows a different form. Toggled via session state.
    if st.session_state.get("_auth_show_reset"):
        _render_reset_form()
        return

    email = st.text_input("Email", key="auth_email", autocomplete="email")
    password = st.text_input(
        "Password", type="password", key="auth_password", autocomplete="current-password"
    )

    col_signin, col_reset = st.columns([2, 1])
    with col_signin:
        if st.button("Sign in", type="primary", key="auth_signin_btn"):
            if not email or not password:
                st.error("Email and password required.")
            else:
                ok, err = sign_in(email.strip(), password)
                if ok:
                    st.rerun()
                else:
                    # `err` is already user-safe (see supabase_session.sign_in).
                    # No "Details:" expansion — we don't want to surface
                    # anything more than the canonical message.
                    st.error(err or "Sign-in failed.")

    with col_reset:
        if st.button("Forgot?", key="auth_reset_btn"):
            st.session_state["_auth_show_reset"] = True
            st.rerun()

    st.caption(
        f"Want access? [Email us](mailto:{contact_email}?subject=Decant%20access)"
    )


def _render_reset_form() -> None:
    """Magic-link password reset request form."""
    st.caption("We'll email you a link to reset your password.")
    email = st.text_input("Email", key="auth_reset_email")
    col_send, col_back = st.columns([2, 1])

    with col_send:
        if st.button("Send reset link", type="primary", key="auth_reset_send"):
            if not email:
                st.error("Email required.")
            else:
                ok, err = request_password_reset(email.strip())
                if ok:
                    st.success(
                        "If that email is registered, a reset link is on its way."
                    )
                    st.session_state["_auth_show_reset"] = False
                    st.rerun()
                else:
                    st.error(f"Couldn't send reset email: {err}")

    with col_back:
        if st.button("Back", key="auth_reset_back"):
            st.session_state["_auth_show_reset"] = False
            st.rerun()
