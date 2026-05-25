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
    """Email + password form, with a help-by-email fallback for stuck users."""
    st.markdown("### Sign in")

    # Help-by-email mode shows a different form. Toggled via session state.
    if st.session_state.get("_auth_show_help"):
        _render_help_form(contact_email)
        return

    email = st.text_input("Email", key="auth_email", autocomplete="email")
    password = st.text_input(
        "Password", type="password", key="auth_password", autocomplete="current-password"
    )

    col_signin, col_help = st.columns([2, 1])
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

    with col_help:
        if st.button("Need help?", key="auth_help_btn"):
            st.session_state["_auth_show_help"] = True
            st.rerun()

    st.caption(
        f"Want access? [Email us](mailto:{contact_email}?subject=Decant%20access)"
    )


def _render_help_form(contact_email: str) -> None:
    """Compose a sign-in help email pre-filled with the user's message.

    The browser handles the click — there's no SMTP, no Supabase call,
    no backend involvement. The user's mail client opens with the
    address, subject, and body already populated; they hit Send in
    their own client. Tomasz (or whoever `contact_email` is) gets the
    email and resets the password manually via the Supabase dashboard
    or admin API.

    This replaces the previous magic-link reset flow, which was
    half-built (the request side worked but the recovery handler that
    would let the user pick a new password never existed).
    """
    st.caption(
        "Stuck signing in? Send us a quick note and we'll get you sorted."
    )

    user_email = st.text_input(
        "Your email (so we know who to reply to)",
        key="auth_help_user_email",
        autocomplete="email",
    )
    message = st.text_area(
        "What's going on?",
        key="auth_help_message",
        placeholder="e.g. forgot my password, can't sign in...",
        height=100,
    )

    # The mailto URL has to be built fresh on every rerun because the
    # form values change. We URL-encode the components since arbitrary
    # text can contain characters (& ? #) that break query parsing.
    from urllib.parse import quote

    body_lines = []
    if user_email.strip():
        body_lines.append(f"From: {user_email.strip()}")
        body_lines.append("")
    if message.strip():
        body_lines.append(message.strip())
    else:
        body_lines.append("(no message)")

    body = "\n".join(body_lines)
    mailto_url = (
        f"mailto:{contact_email}"
        f"?subject={quote('Decant - sign-in help')}"
        f"&body={quote(body)}"
    )

    col_send, col_back = st.columns([2, 1])
    with col_send:
        # Streamlit's link_button renders an anchor tag, so the browser
        # handles the click as a navigation — which for a mailto URL
        # means "open the user's default mail client." st.button would
        # not work here; it'd run Python code instead.
        st.link_button("Open in mail client", mailto_url, type="primary")

    with col_back:
        if st.button("Back", key="auth_help_back"):
            st.session_state["_auth_show_help"] = False
            st.rerun()

    st.caption(
        "Tip: your mail client will open with the email already written. "
        "Just hit send."
    )
