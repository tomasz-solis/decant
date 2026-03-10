"""
Simple authentication module for Decant.

Provides password protection using streamlit-authenticator.
"""

import streamlit as st
import streamlit_authenticator as stauth
from typing import Optional


def setup_authentication(guest_allowed: bool = True) -> Optional[str]:
    """
    Set up authentication and return username if logged in.

    Args:
        guest_allowed: If True, unauthenticated users can browse in read-only
                       guest mode.  If False, the app stops until login.

    Returns:
        Username if authenticated, None for guest mode.
    """
    # Get credentials from secrets
    try:
        passwords = dict(st.secrets["passwords"])
        cookie_config = dict(st.secrets["cookie"])
    except (FileNotFoundError, KeyError):
        if not guest_allowed:
            st.error("🔒 Authentication not configured")
            st.warning("Add credentials to `.streamlit/secrets.toml` to enable login.")
            st.code("""[passwords]
admin = "your-password-hash-here"

[cookie]
name = "decant_auth"
key = "your-secure-key-here"
expiry_days = 30""", language="toml")
            st.stop()
        return None

    # Create credentials dict
    credentials = {
        "usernames": {
            username: {
                "name": username.title(),
                "password": hashed_password
            }
            for username, hashed_password in passwords.items()
        }
    }

    # Create authenticator
    authenticator = stauth.Authenticate(
        credentials,
        cookie_config["name"],
        cookie_config["key"],
        cookie_config["expiry_days"]
    )

    # Show login form (v0.4.2+ API - no arguments needed)
    try:
        authenticator.login()
    except Exception as e:
        if not guest_allowed:
            st.error(f"Login error: {e}")
            st.stop()
        return None

    # Get authentication status from session state
    authentication_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if authentication_status is False:
        st.error("Username/password is incorrect")
        if not guest_allowed:
            st.stop()
        return None
    elif authentication_status is None:
        # Not logged in — guest mode
        if not guest_allowed:
            st.warning("Please enter your username and password")
            st.stop()
        return None

    # Add logout button in sidebar
    with st.sidebar:
        st.write(f"Logged in as: **{name}**")
        authenticator.logout("Logout", "sidebar")

    return username


def hash_password(password: str) -> str:
    """
    Generate hashed password for use in secrets.toml.

    Usage:
        python -c "from decant.auth import hash_password; print(hash_password('your_password'))"
    """
    return stauth.Hasher([password]).generate()[0]


if __name__ == "__main__":
    # Helper to generate password hashes
    import sys
    if len(sys.argv) > 1:
        password = sys.argv[1]
        print(f"Hashed password: {hash_password(password)}")
    else:
        print("Usage: python auth.py YOUR_PASSWORD")
