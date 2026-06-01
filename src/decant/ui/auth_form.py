"""Streamlit UI for the household sign-in flow.

Renders a top-right auth control as a popover:
- when signed out: a "Sign in" button; click to open a popover with the
  email/password form
- when signed in: an email button; click to open a popover with a
  "Sign out" action

The popover affordance keeps the auth out of the main reading flow and
out of the sidebar, where it was discoverability-hostile (collapsed by
default on mobile, hard to find the toggle).

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


def render_header_auth(contact_email: str = "tomasz@example.com") -> None:
    """Render the top-right auth popover.

    Use this inside a right-aligned column at the top of the page. The
    popover handles its own open/close state; clicking outside closes
    it. The caller does not need to manage visibility.

    Args:
        contact_email: address shown in the "need help?" form's mailto
            link, pulled from secrets in the caller.
    """
    if is_authenticated():
        _render_logged_in_popover()
    else:
        _render_signin_popover(contact_email)


def _render_logged_in_popover() -> None:
    """Popover showing the current user and a sign-out action."""
    email = current_user_email() or "household"
    # Truncate the visible email so the button stays compact.
    label = email if len(email) < 20 else email[:17] + "..."
    with st.popover(label, use_container_width=False):
        st.markdown(f"**Signed in as**  \n`{email}`")
        if st.button("Sign out", key="auth_signout", type="primary"):
            sign_out()
            st.rerun()


def _render_signin_popover(contact_email: str) -> None:
    """Popover containing the sign-in form, or the help form if toggled."""
    with st.popover("Sign in", type="primary", use_container_width=False):
        if st.session_state.get("_auth_show_help"):
            _render_help_form(contact_email)
        else:
            _render_login_inputs(contact_email)


def _render_login_inputs(contact_email: str) -> None:
    """Email + password inputs with sign-in and 'need help' actions."""
    st.markdown(
        "<div class='form-title'>Sign in</div>",
        unsafe_allow_html=True,
    )

    email = st.text_input("Email", key="auth_email", autocomplete="email")
    password = st.text_input(
        "Password", type="password", key="auth_password",
        autocomplete="current-password",
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
